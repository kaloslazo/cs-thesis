"""
exp2_adversario.py — EXP-2 (Semana 4 PFC II): adversario tumoral fortalecido.

Amplía el espacio de acción del tumor (phi_max) y re-entrena MAPPO-CTDE e IPPO por
self-play, comparando el TTP-combinado contra su propio tumor ADAPTATIVO co-entrenado
(peor caso, no un phi fijo débil). Ver docs/protocolo_experimental_pfc2.md.

Preguntas:
  1. ¿Cómo se degrada el TTP de la terapia aprendida al crecer phi_max?
  2. ¿Gana CTDE (MAPPO) sobre local (IPPO) cuando la amenaza es más realista?
  3. ¿El modo de falla migra hacia "resistencia mayoría" con phi_max alto?

Barrido: phi_max in {0.05, 0.10, 0.20} x variante in {mappo, ippo} x n=15 semillas.
Estadística: bimodal -> mediana [min,max] + tasa de éxito; contraste MAPPO>IPPO con
Mann-Whitney U (unilateral).

Reanudable: modelos en outputs/models/exp2_{variant}_phi{p}_{seed}_*.pt y resultados
en outputs/exp2_adversario.json. Re-ejecutar continúa donde quedó.

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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats

from gbmarl.tumor_env import TumorEnv
from gbmarl.mappo import train_mappo, Agent, obs_therapy, obs_tumor

MODELS = "outputs/models"
STATE = "outputs/exp2_adversario.json"
PHI_MAXES = [0.05, 0.10, 0.20]
GATENBY_NOMINAL = 27


@torch.no_grad()
def ttp_vs_adaptivo(env, th, tu):
    """TTP-combinado con AMBOS agentes aprendidos actuando (peor caso realista)."""
    obs, _ = env.reset(seed=0); state = obs["therapy"]; info = {}
    while True:
        a_th = th.actor_mean(torch.tensor(obs_therapy(state))).clamp(th.a_low, th.a_high)
        a_tu = tu.actor_mean(torch.tensor(obs_tumor(state))).clamp(tu.a_low, tu.a_high)
        obs, rew, terms, truncs, infos = env.step(
            {"therapy": a_th.numpy(), "tumor": a_tu.numpy()})
        state = obs["therapy"]; info = infos["therapy"]
        if terms.get("therapy", False) or truncs.get("therapy", False):
            return info["day"], ("resistencia" if info.get("untreatable")
                                 else "carga" if info.get("progressed") else "otro")


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
    tag = f"exp2_{variant}_phi{phi}_{seed}"
    p_th, p_tu = f"{MODELS}/{tag}_therapy.pt", f"{MODELS}/{tag}_tumor.pt"
    centralized = (variant == "mappo")
    cdim = 3 if centralized else 2
    if os.path.exists(p_th) and os.path.exists(p_tu):
        th = Agent(2, cdim, 1, env.action_space("therapy").low, env.action_space("therapy").high)
        tu = Agent(2, cdim, 1, env.action_space("tumor").low, env.action_space("tumor").high)
        th.load_state_dict(torch.load(p_th)); tu.load_state_dict(torch.load(p_tu))
        th.eval(); tu.eval(); return th, tu
    th, tu, _ = train_mappo(env, total_timesteps=steps, seed=seed,
                            centralized=centralized, verbose=False)
    os.makedirs(MODELS, exist_ok=True)
    torch.save(th.state_dict(), p_th); torch.save(tu.state_dict(), p_tu)
    th.eval(); tu.eval(); return th, tu


def run(args):
    st = load_state()
    st["config"] = {"seeds": args.seeds, "steps": args.steps, "phi_maxes": args.phis}
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
                st["cells"][cell] = {"variant": variant, "phi_max": phi,
                                     "seed": seed, "ttp": int(ttp), "modo_falla": modo}
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


def make_outputs(st):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phis = st["config"]["phis"] if "phis" in st["config"] else st["config"]["phi_maxes"]
    seeds = st["config"]["seeds"]

    lines = ["# EXP-2 Adversario fortalecido — resultados\n",
             f"Semillas n={seeds} · pasos={st['config']['steps']} · TTP vs tumor adaptativo co-entrenado\n",
             "\n## TTP-combinado por régimen (mediana [min,max]) y contraste CTDE vs IPPO\n",
             "| φ_max | MAPPO (CTDE) | IPPO (local) | Δ mediana | Mann-Whitney p | éxito MAPPO | éxito IPPO |",
             "|---|---|---|---|---|---|---|"]
    for phi in phis:
        m = np.array(_collect(st, "mappo", phi), float)
        i = np.array(_collect(st, "ippo", phi), float)
        if len(m) == 0 or len(i) == 0:
            continue
        exito = lambda a: np.mean(a > GATENBY_NOMINAL)
        try:
            _, p = stats.mannwhitneyu(m, i, alternative="greater")
            pstr = f"{p:.3f}"
        except Exception as e:
            pstr = f"n/a ({type(e).__name__})"
        lines.append(f"| {phi:.2f} | {np.median(m):.0f} [{m.min():.0f},{m.max():.0f}] "
                     f"| {np.median(i):.0f} [{i.min():.0f},{i.max():.0f}] "
                     f"| {np.median(m)-np.median(i):+.0f} | {pstr} "
                     f"| {exito(m):.0%} | {exito(i):.0%} |")
    lines += ["\n## Modo de falla dominante por régimen\n",
              "| φ_max | MAPPO | IPPO |", "|---|---|---|"]
    for phi in phis:
        lines.append(f"| {phi:.2f} | {_modes(st,'mappo',phi)} | {_modes(st,'ippo',phi)} |")
    with open("outputs/exp2_adversario.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Tabla -> outputs/exp2_adversario.md")

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
    ax.axhline(GATENBY_NOMINAL, ls="--", color="#E67E22", label="Gatenby nominal (27)")
    ax.set_xlabel("φ_max (fuerza del adversario tumoral)")
    ax.set_ylabel("TTP-combinado vs tumor adaptativo (d)")
    ax.set_title(f"EXP-2 — Degradación y ablación CTDE bajo adversario fuerte (n={seeds})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("outputs/exp2_adversario.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Figura -> outputs/exp2_adversario.png")


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
    run(args)


if __name__ == "__main__":
    main()
