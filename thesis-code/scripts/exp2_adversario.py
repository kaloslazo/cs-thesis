"""
exp2_adversario.py — EXP-2 (Semana 4 PFC II): adversario tumoral fortalecido.

Amplía el espacio de acción del tumor (phi_max) y re-entrena MAPPO-CTDE e IPPO por
self-play, comparando el TTP-combinado contra su propio tumor ADAPTATIVO co-entrenado
(un adversario endógeno, no un peor caso garantizado ni un phi fijo débil). Ver
docs/protocolo_experimental_pfc2.md.

Preguntas:
  1. ¿Cómo se degrada el TTP de la terapia aprendida al crecer phi_max?
  2. ¿Gana CTDE (MAPPO) sobre local (IPPO) cuando la amenaza es más realista?
  3. ¿El modo de falla migra hacia "resistencia mayoría" con phi_max alto?

Barrido: phi_max in {0.05, 0.10, 0.20} x variante in {mappo, ippo} x n=15 semillas.
Estadística: bimodal -> mediana [min,max] + tasa de éxito contra Gatenby evaluado
con el mismo tumor; contraste pareado Wilcoxon + corrección Holm.

Reanudable: modelos V2 identificados por presupuesto, variante, phi y semilla; resultados
en outputs/exp2_adversario_v2.json. Re-ejecutar continúa donde quedó.

Uso:
  python scripts/exp2_adversario.py --seeds 15 --steps 120000
  python scripts/exp2_adversario.py --smoke          # 2 semillas, 2k pasos, phi 0.05/0.2
  python scripts/exp2_adversario.py --plot           # regenera figuras/tabla desde json
"""
import os
import sys
import json
import time
import argparse
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats

from gbmarl.tumor_env import TumorEnv
from gbmarl.config import load_calibration, params_fingerprint
from gbmarl.mappo import train_mappo, Agent, obs_therapy, obs_tumor
from gbmarl.evalutils import Gatenby, U_MAX
from gbmarl.outcomes import classify_outcome, is_failure

MODELS = "outputs/models"
STATE = "outputs/exp2_adversario_v2.json"
REPORT = "outputs/exp2_adversario_v2.md"
FIGURE = "outputs/exp2_adversario_v2.png"
PHI_MAXES = [0.05, 0.10, 0.20]
EXPERIMENT_VERSION = 2
CALIBRATION_SIGNATURE = params_fingerprint(load_calibration())


@torch.no_grad()
def ttp_vs_adaptivo(env, th, tu):
    """TTP-combinado con ambos agentes co-entrenados actuando."""
    obs, _ = env.reset(seed=0); state = obs["therapy"]; info = {}
    while True:
        a_th = th.actor_mean(torch.tensor(obs_therapy(state))).clamp(th.a_low, th.a_high)
        a_tu = tu.actor_mean(torch.tensor(obs_tumor(state))).clamp(tu.a_low, tu.a_high)
        obs, rew, terms, truncs, infos = env.step(
            {"therapy": a_th.numpy(), "tumor": a_tu.numpy()})
        state = obs["therapy"]; info = infos["therapy"]
        if terms.get("therapy", False) or truncs.get("therapy", False):
            mode = info.get("failure_mode") or classify_outcome(
                progressed=bool(info.get("progressed")),
                untreatable=bool(info.get("untreatable")),
                extinct=bool(terms.get("therapy") and not info.get("progressed") and
                             not info.get("untreatable")),
                reached_horizon=bool(truncs.get("therapy") and not terms.get("therapy")),
            )
            ttp = info["day"] if is_failure(mode) else env.horizon
            return int(ttp), mode


@torch.no_grad()
def ttp_policy_vs_tumor(env, therapy_fn, tumor):
    """Evalúa una baseline contra exactamente el tumor aprendido de la celda."""
    if hasattr(therapy_fn, "reset"):
        therapy_fn.reset()
    obs, _ = env.reset(seed=0); state = obs["therapy"]
    while True:
        u = float(therapy_fn(state))
        a_tu = tumor.actor_mean(torch.tensor(obs_tumor(state))).clamp(
            tumor.a_low, tumor.a_high)
        obs, _, terms, truncs, infos = env.step(
            {"therapy": np.array([u], np.float32), "tumor": a_tu.numpy()})
        state = obs["therapy"]; info = infos["therapy"]
        if terms.get("therapy", False) or truncs.get("therapy", False):
            mode = info["failure_mode"]
            ttp = info["day"] if is_failure(mode) else env.horizon
            return int(ttp), mode


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"cells": {}, "config": {}}


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)


def train_or_load(env, variant, phi, seed, steps):
    """Entrena (o carga) terapia+tumor para (variante, phi_max, seed). Reanudable."""
    tag = f"exp2_v2_{CALIBRATION_SIGNATURE}_t{steps}_{variant}_phi{phi}_{seed}"
    p_th, p_tu = f"{MODELS}/{tag}_therapy.pt", f"{MODELS}/{tag}_tumor.pt"
    centralized = (variant == "mappo")
    cdim = 3 if centralized else 2
    if os.path.exists(p_th) and os.path.exists(p_tu):
        th = Agent(2, cdim, 1, env.action_space("therapy").low, env.action_space("therapy").high)
        tu = Agent(2, cdim, 1, env.action_space("tumor").low, env.action_space("tumor").high)
        th.load_state_dict(torch.load(p_th, weights_only=True)); tu.load_state_dict(
            torch.load(p_tu, weights_only=True))
        th.eval(); tu.eval(); return th, tu
    th, tu, _ = train_mappo(env, total_timesteps=steps, seed=seed,
                            centralized=centralized, verbose=False)
    os.makedirs(MODELS, exist_ok=True)
    torch.save(th.state_dict(), p_th); torch.save(tu.state_dict(), p_tu)
    th.eval(); tu.eval(); return th, tu


def run(args):
    st = load_state()
    requested = {"version": EXPERIMENT_VERSION, "calibration": CALIBRATION_SIGNATURE,
                 "seeds": args.seeds,
                 "steps": args.steps, "phi_maxes": args.phis}
    if st.get("config") and st["config"] != requested:
        raise ValueError(
            f"{STATE} pertenece a otra configuración. Muévelo o usa sus valores: "
            f"{st['config']}"
        )
    st["config"] = requested
    t0 = time.time()
    for phi in args.phis:
        env = TumorEnv(horizon_days=180, phi_max=phi)
        for variant in ("mappo", "ippo"):
            for seed in range(args.seeds):
                cell = f"{variant}@phi{phi}@s{seed}"
                if cell in st["cells"]:
                    continue
                th, tu = train_or_load(env, variant, phi, seed, args.steps)
                ttp, modo = ttp_vs_adaptivo(env, th, tu)
                gat_ttp, gat_mode = ttp_policy_vs_tumor(env, Gatenby(), tu)
                mtd_ttp, mtd_mode = ttp_policy_vs_tumor(env, lambda _: U_MAX, tu)
                st["cells"][cell] = {"variant": variant, "phi_max": phi,
                                     "seed": seed, "ttp": int(ttp), "modo_falla": modo,
                                     "gatenby_ttp_same_tumor": gat_ttp,
                                     "gatenby_mode_same_tumor": gat_mode,
                                     "mtd_ttp_same_tumor": mtd_ttp,
                                     "mtd_mode_same_tumor": mtd_mode}
                save_state(st)
                print(f"[{variant:5s} phi={phi:.2f} s={seed:2d}] TTP={ttp:3d} ({modo}) "
                      f"[{time.time()-t0:.0f}s]")
    print(f"=== EXP-2 completo ({time.time()-t0:.0f}s) ===")
    make_outputs(st)


def _collect(st, variant, phi):
    return [c["ttp"] for c in st["cells"].values()
            if c["variant"] == variant and c["phi_max"] == phi]


def _modes(st, variant, phi):
    ms = [c["modo_falla"] for c in st["cells"].values()
          if c["variant"] == variant and c["phi_max"] == phi]
    return {m: ms.count(m) for m in set(ms)}


def _collect_key(st, variant, phi, key):
    rows = sorted((c for c in st["cells"].values()
                   if c["variant"] == variant and c["phi_max"] == phi),
                  key=lambda c: c["seed"])
    return [c[key] for c in rows]


def _holm_adjust(pvalues):
    """Holm step-down, conservando el orden original."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [1.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * pvalues[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def make_outputs(st):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "gbmarl-mpl"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phis = st["config"]["phis"] if "phis" in st["config"] else st["config"]["phi_maxes"]
    seeds = st["config"]["seeds"]

    comparisons = []
    for phi in phis:
        m = np.array(_collect_key(st, "mappo", phi, "ttp"), float)
        i = np.array(_collect_key(st, "ippo", phi, "ttp"), float)
        if len(m) and len(i):
            try:
                p = stats.wilcoxon(m, i, alternative="greater").pvalue
            except ValueError:
                p = 1.0
            comparisons.append((phi, m, i, p))
    adjusted = _holm_adjust([row[3] for row in comparisons])
    observed = sorted({c["phi_max"] for c in st["cells"].values()})
    expected_cells = len(phis) * 2 * seeds
    status = "COMPLETO" if len(st["cells"]) == expected_cells else "PARCIAL"

    lines = ["# EXP-2 Adversario fortalecido V2 — resultados\n",
             f"Estado: **{status}** · semillas objetivo n={seeds} · pasos={st['config']['steps']} · "
             f"TTP vs tumor adaptativo co-entrenado\n",
             f"Celdas observadas: {len(st['cells'])}/{expected_cells} · regímenes observados: {observed}\n",
             "Cada baseline se evalúa contra el mismo tumor aprendido de su celda.\n",
             "\n## TTP-combinado por régimen (mediana [min,max]) y contraste pareado\n",
             "| φ_max | MAPPO (CTDE) | IPPO (local) | mediana Δ pareada | Wilcoxon p | Holm p | éxito MAPPO | éxito IPPO |",
             "|---|---|---|---|---|---|---|---|"]
    if not comparisons:
        lines.append("| *(sin celdas MAPPO e IPPO completas en un mismo régimen)* | — | — | — | — | — | — | — |")
    for (phi, m, i, p), p_adj in zip(comparisons, adjusted):
        gm = np.array(_collect_key(st, "mappo", phi, "gatenby_ttp_same_tumor"), float)
        gi = np.array(_collect_key(st, "ippo", phi, "gatenby_ttp_same_tumor"), float)
        success_m = np.mean(m > gm)
        success_i = np.mean(i > gi)
        lines.append(f"| {phi:.2f} | {np.median(m):.0f} [{m.min():.0f},{m.max():.0f}] "
                     f"| {np.median(i):.0f} [{i.min():.0f},{i.max():.0f}] "
                     f"| {np.median(m-i):+.0f} | {p:.3f} | {p_adj:.3f} "
                     f"| {success_m:.0%} | {success_i:.0%} |")
    lines += ["\n## Modo de falla dominante por régimen\n",
              "| φ_max | MAPPO | IPPO |", "|---|---|---|"]
    for phi in phis:
        lines.append(f"| {phi:.2f} | {_modes(st,'mappo',phi)} | {_modes(st,'ippo',phi)} |")
    if observed != list(phis):
        lines += ["\n## Celdas observadas antes de completar el barrido\n",
                  "| Método | φ_max | semilla | TTP | modo | Gatenby mismo tumor |",
                  "|---|---:|---:|---:|---|---:|"]
        for c in sorted(st["cells"].values(), key=lambda row: (row["phi_max"], row["variant"], row["seed"])):
            lines.append(f"| {c['variant'].upper()} | {c['phi_max']:.2f} | {c['seed']} | "
                         f"{c['ttp']} | {c['modo_falla']} | {c['gatenby_ttp_same_tumor']} |")
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Tabla -> {REPORT}")

    # Figura: TTP vs phi_max, MAPPO vs IPPO
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for variant, color, mk in (("mappo", "#2E75B6", "^-"), ("ippo", "#95A5A6", "s-")):
        med, lo, hi, xs = [], [], [], []
        for phi in phis:
            a = np.array(_collect(st, variant, phi), float)
            if len(a) == 0:
                continue
            xs.append(phi); med.append(np.median(a)); lo.append(a.min()); hi.append(a.max())
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
        ax.plot(xs, med, mk, color=color, label=f"{variant.upper()} (mediana)")
    for variant, color in (("mappo", "#D68910"), ("ippo", "#F5B041")):
        gat_x, gat_med = [], []
        for phi in phis:
            values = _collect_key(st, variant, phi, "gatenby_ttp_same_tumor")
            if values:
                gat_x.append(phi); gat_med.append(np.median(values))
        if gat_x:
            ax.plot(gat_x, gat_med, ":", color=color,
                    label=f"Gatenby vs tumores {variant.upper()}")
    ax.set_xlabel("φ_max (fuerza del adversario tumoral)")
    ax.set_ylabel("TTP-combinado vs tumor adaptativo (d)")
    ax.set_title(f"EXP-2 — Degradación y ablación CTDE bajo adversario fuerte (n={seeds})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURE, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Figura -> {FIGURE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--phis", type=float, nargs="+", default=PHI_MAXES)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.steps, args.phis = 2, 2000, [0.05, 0.20]
    # normaliza config key para make_outputs
    if args.plot:
        st = load_state(); st["config"]["phis"] = st["config"].get("phi_maxes", PHI_MAXES)
        make_outputs(st); return
    if args.seeds < 1 or args.steps < 1 or not args.phis:
        ap.error("--seeds, --steps y --phis deben contener valores positivos")
    if any(phi <= 0 for phi in args.phis):
        ap.error("todos los valores de --phis deben ser positivos")
    run(args)


if __name__ == "__main__":
    main()
