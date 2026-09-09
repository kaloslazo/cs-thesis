"""
exp1_robustez.py — EXP-1 (Semana 3 PFC II): robustez de la política MAPPO-CTDE.

Fuentes de incertidumbre (ver docs/protocolo_experimental_pfc2.md):
  · EXP-1a  Perturbación paramétrica ±20% OAT y Monte Carlo conjunto sobre las
            constantes calibradas/literatura. La política se ENTRENA en el entorno
            nominal y se EVALÚA en el perturbado (robustez de transferencia).
  · EXP-1b  Ruido de observación gaussiano relativo sobre lo que ve la TERAPIA
            (S+R, c), SOLO en evaluación (wrapper, no se re-entrena ni se toca
            tumor_env.py). Mide robustez de la política ya aprendida.

  · Auditoría de endpoint y controles de K para separar dinámica y definición.

Métrica: TTP-combinado (evalutils). Estadística: bimodal -> mediana, bootstrap
por semilla y tasa de éxito contra Gatenby en el mismo entorno. n=15 semillas.

Reanudable: guarda modelos V2 identificados por presupuesto y semilla, y resultados
parciales en outputs/exp1_robustez_v2.json. Re-ejecutar continúa donde quedó.

Uso:
  python scripts/exp1_robustez.py --seeds 15 --steps 120000
  python scripts/exp1_robustez.py --smoke              # prueba rápida (2 semillas, 2k pasos)
  python scripts/exp1_robustez.py --plot               # solo re-genera figuras/tabla desde el json
"""
import os
import sys
import json
import time
import argparse
import dataclasses
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gbmarl.tumor_env import TumorEnv
from gbmarl.config import Params, load_calibration, params_fingerprint
from gbmarl.mappo import train_mappo, Agent, obs_therapy
from gbmarl.evalutils import ttp_combinado, Gatenby, U_MAX, FIXED_PHI
from gbmarl.outcomes import classify_outcome, scalar_ttp

MODELS = "outputs/models"
STATE = "outputs/exp1_robustez_v2.json"
REPORT = "outputs/exp1_robustez_v2.md"
FIG_OAT = "outputs/exp1a_perturbacion_v2.png"
FIG_NOISE = "outputs/exp1b_ruido_v2.png"
EXPERIMENT_VERSION = 2

# Parámetros a perturbar ±20% y sus factores
PARAM_KEYS = ["ic50_S", "ic50_R", "delta_max_S", "delta_max_R",
              "alpha_S", "alpha_R", "lambda_c", "K"]
FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
SIGMAS = [0.0, 0.05, 0.10, 0.20]
NOISE_REPS = 30
JOINT_SAMPLES = 50
PROGRESSION_THRESHOLDS = [0.70, 0.80, 0.90]
RESISTANCE_THRESHOLDS = [0.40, 0.50, 0.60]

try:
    BASE = load_calibration()
except Exception as e:
    print(f"[exp1] sin calibration.json ({e}); uso placeholders"); BASE = Params()
CALIBRATION_SIGNATURE = params_fingerprint(BASE)


# ───────────────────────── utilidades ──────────────────────────
def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"policies": {}, "exp1a": {}, "exp1a_joint": {},
            "endpoint_sensitivity": {}, "k_controls": {},
            "exp1b": {}, "config": {}}


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)


def make_agent(env):
    return Agent(2, 3, 1, env.action_space("therapy").low,
                 env.action_space("therapy").high)


def policy_from(agent):
    def pol(state):
        with torch.no_grad():
            a = agent.actor_mean(torch.tensor(obs_therapy(state))).clamp(
                agent.a_low, agent.a_high)
        return float(a[0])
    return pol


def train_or_load_policy(seed, steps):
    """Entrena (o carga) la política MAPPO nominal para una semilla. Reanudable."""
    # La auditoría nominal vigente exporta sus actores con este prefijo. Si
    # existe, se reutiliza para que EXP-1 mida robustez de la misma población
    # calibrada que la validación principal, no de una corrida histórica.
    canonical = (f"{MODELS}/mappo_real_20260908_therapy.pt" if seed == 0 else
                 f"{MODELS}/mappo_real_20260908_s{seed}_therapy.pt")
    path = f"{MODELS}/exp1_v2_{CALIBRATION_SIGNATURE}_t{steps}_mappo_{seed}.pt"
    env = TumorEnv(horizon_days=180)               # entorno NOMINAL
    if os.path.exists(canonical):
        ag = make_agent(env)
        ag.load_state_dict(torch.load(canonical, weights_only=True)); ag.eval()
        return ag
    if os.path.exists(path):
        ag = make_agent(env)
        ag.load_state_dict(torch.load(path, weights_only=True)); ag.eval()
        return ag
    th, _, _ = train_mappo(env, total_timesteps=steps, seed=seed,
                           centralized=True, verbose=False)
    os.makedirs(MODELS, exist_ok=True)
    torch.save(th.state_dict(), path)
    th.eval()
    return th


# ───────────────────────── EXP-1b: ruido de observación ──────────────────────────
@torch.no_grad()
def ttp_noisy(env, agent, sigma, rng):
    """TTP-combinado con ruido gaussiano relativo en la observación de la terapia.

    El estado VERDADERO avanza la dinámica; la terapia decide sobre una copia
    ruidosa (S+R, c) -> obs_ruidosa = obs * (1+eps), eps~N(0,sigma^2), trunc >=0.
    """
    obs, _ = env.reset(seed=0)
    state = obs["therapy"]
    info = {}
    while True:
        noisy = state.copy()
        if sigma > 0:
            eps = rng.normal(0.0, sigma, size=3).astype(np.float32)
            noisy = np.clip(state * (1.0 + eps), 0.0, None)
        u = float(agent.actor_mean(torch.tensor(obs_therapy(noisy))).clamp(
            agent.a_low, agent.a_high)[0])
        obs, rew, terms, truncs, infos = env.step(
            {"therapy": np.array([u], np.float32),
             "tumor": np.array([FIXED_PHI], np.float32)})
        state = obs["therapy"]; info = infos["therapy"]
        if terms.get("therapy", False) or truncs.get("therapy", False):
            mode = info.get("failure_mode") or classify_outcome(
                progressed=bool(info.get("progressed")),
                untreatable=bool(info.get("untreatable")),
                extinct=bool(terms.get("therapy") and not info.get("progressed") and
                             not info.get("untreatable")),
                reached_horizon=bool(truncs.get("therapy") and not terms.get("therapy")),
            )
            return scalar_ttp(mode, info["day"], env.horizon), mode


def evaluate_policies(env, agents):
    """Evalúa políticas y devuelve TTP; los modos quedan en el propio entorno."""
    ttps, modes = [], []
    for seed in sorted(agents):
        ttp = ttp_combinado(env, policy_from(agents[seed]))
        ttps.append(int(ttp))
        modes.append(env.last_outcome)
    return ttps, modes


# ───────────────────────── driver ──────────────────────────
def run(args):
    st = load_state()
    requested = {"version": EXPERIMENT_VERSION, "calibration": CALIBRATION_SIGNATURE,
                 "seeds": args.seeds,
                 "steps": args.steps, "factors": FACTORS, "sigmas": SIGMAS,
                 "noise_reps": NOISE_REPS, "joint_samples": args.joint_samples,
                 "progression_thresholds": PROGRESSION_THRESHOLDS,
                 "resistance_thresholds": RESISTANCE_THRESHOLDS}
    if st.get("config") and st["config"] != requested:
        raise ValueError(
            f"{STATE} pertenece a otra configuración. Muévelo o usa sus valores: "
            f"{st['config']}"
        )
    st["config"] = requested
    t0 = time.time()

    # --- Fase A: entrenar/cargar n políticas nominales ---
    agents = {}
    for seed in range(args.seeds):
        ag = train_or_load_policy(seed, args.steps)
        agents[seed] = ag
        st["policies"][str(seed)] = (
            f"{MODELS}/mappo_real_20260908_therapy.pt" if seed == 0 else
            f"{MODELS}/mappo_real_20260908_s{seed}_therapy.pt")
        save_state(st)
        print(f"[A] politica seed={seed} lista ({time.time()-t0:.0f}s)")

    # --- Fase B: EXP-1a perturbacion parametrica +-20% (OAT) ---
    # baseline nominal de las politicas
    env_nom = TumorEnv(horizon_days=180)
    nom, nom_modes = evaluate_policies(env_nom, agents)
    st["exp1a"]["_nominal"] = {"mappo": nom,
                               "modes": nom_modes,
                               "MTD": ttp_combinado(env_nom, lambda s: U_MAX),
                               "Gatenby": ttp_combinado(env_nom, Gatenby())}
    save_state(st)

    for key in PARAM_KEYS:
        base_val = getattr(BASE, key)
        for f in FACTORS:
            cell = f"{key}@{f}"
            if cell in st["exp1a"]:
                continue
            val = base_val * f
            params = dataclasses.replace(BASE, **{key: val})
            env = TumorEnv(horizon_days=180, params=params)
            mappo, modes = evaluate_policies(env, agents)
            mtd = ttp_combinado(env, lambda s: U_MAX)
            gat = ttp_combinado(env, Gatenby())
            st["exp1a"][cell] = {"param": key, "factor": f, "value": round(val, 4),
                                 "mappo": mappo, "modes": modes,
                                 "MTD": mtd, "Gatenby": gat,
                                 "success_same_gatenby": float(np.mean(
                                     np.asarray(mappo) > gat))}
            save_state(st)
            med = float(np.median(mappo))
            print(f"[B] {cell:22s} MAPPO_med={med:4.0f} MTD={mtd:3d} Gat={gat:3d}")

    # --- Fase C: controles para separar el efecto biológico de K del endpoint ---
    for f in FACTORS:
        val = BASE.K * f
        params = dataclasses.replace(BASE, K=val)
        controls = {
            # Mantiene S0/K y R0/K constantes.
            "scaled_initial": TumorEnv(
                horizon_days=180, params=params,
                S0=0.40 * val / BASE.K, R0=0.01 * val / BASE.K),
            # Mantiene el umbral absoluto de carga en 0.8*K_nominal.
            "fixed_load_endpoint": TumorEnv(
                horizon_days=180, params=params,
                progression_threshold=0.80 * BASE.K / val),
        }
        for label, env in controls.items():
            cell = f"K@{f}@{label}"
            if cell in st["k_controls"]:
                continue
            ttps, modes = evaluate_policies(env, agents)
            gat = ttp_combinado(env, Gatenby())
            st["k_controls"][cell] = {
                "factor": f, "control": label, "mappo": ttps, "modes": modes,
                "Gatenby": gat, "MTD": ttp_combinado(env, lambda _: U_MAX),
                "success_same_gatenby": float(np.mean(np.asarray(ttps) > gat)),
            }
            save_state(st)

    # --- Fase D: perturbación conjunta prometida por el protocolo ---
    rng_joint = np.random.default_rng(20260908)
    joint_factors = rng_joint.uniform(0.8, 1.2,
                                      size=(args.joint_samples, len(PARAM_KEYS)))
    for sample, factors in enumerate(joint_factors):
        cell = f"sample@{sample}"
        if cell in st["exp1a_joint"]:
            continue
        replacements = {key: getattr(BASE, key) * float(factor)
                        for key, factor in zip(PARAM_KEYS, factors)}
        params = dataclasses.replace(BASE, **replacements)
        env = TumorEnv(horizon_days=180, params=params)
        ttps, modes = evaluate_policies(env, agents)
        gat = ttp_combinado(env, Gatenby())
        st["exp1a_joint"][cell] = {
            "sample": sample,
            "factors": {key: round(float(factor), 6)
                        for key, factor in zip(PARAM_KEYS, factors)},
            "mappo": ttps, "modes": modes,
            "Gatenby": gat, "MTD": ttp_combinado(env, lambda _: U_MAX),
            "success_same_gatenby": float(np.mean(np.asarray(ttps) > gat)),
        }
        save_state(st)

    # --- Fase E: sensibilidad de los umbrales que definen el endpoint ---
    for prog in PROGRESSION_THRESHOLDS:
        for resistance in RESISTANCE_THRESHOLDS:
            cell = f"load@{prog}@resistance@{resistance}"
            if cell in st["endpoint_sensitivity"]:
                continue
            env = TumorEnv(horizon_days=180, progression_threshold=prog,
                           r_majority=resistance)
            ttps, modes = evaluate_policies(env, agents)
            gat = ttp_combinado(env, Gatenby())
            st["endpoint_sensitivity"][cell] = {
                "progression_threshold": prog,
                "resistance_threshold": resistance,
                "mappo": ttps, "modes": modes,
                "Gatenby": gat, "MTD": ttp_combinado(env, lambda _: U_MAX),
                "success_same_gatenby": float(np.mean(np.asarray(ttps) > gat)),
            }
            save_state(st)

    # --- Fase F: EXP-1b ruido de observacion ---
    for sigma in SIGMAS:
        key = f"sigma@{sigma}"
        if key in st["exp1b"]:
            continue
        per_seed_med = []
        allttp = []
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 + s)
            reps = 1 if sigma == 0 else NOISE_REPS
            evaluations = [ttp_noisy(env_nom, agents[s], sigma, rng)
                           for _ in range(reps)]
            ttps = [row[0] for row in evaluations]
            modes = [row[1] for row in evaluations]
            per_seed_med.append(float(np.median(ttps)))
            allttp.extend(ttps)
            st["exp1b"].setdefault("_modes", {}).setdefault(key, {})[str(s)] = modes
        st["exp1b"][key] = {"sigma": sigma, "per_seed_median": per_seed_med,
                            "policy_level_median": float(np.median(per_seed_med)),
                            "all_episode_median": float(np.median(allttp)),
                            "all_min": int(np.min(allttp)), "all_max": int(np.max(allttp))}
        save_state(st)
        print(f"[F] sigma={sigma:4.2f}  TTP mediana={np.median(per_seed_med):4.0f} "
              f"[{np.min(allttp):.0f},{np.max(allttp):.0f}]")

    print(f"=== EXP-1 completo ({time.time()-t0:.0f}s) ===")
    make_outputs(st)


# ───────────────────────── tabla + figuras ──────────────────────────
def make_outputs(st):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "gbmarl-mpl"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = st["config"]["seeds"]
    nom = st["exp1a"]["_nominal"]["mappo"]
    nom_med = float(np.median(nom))
    nominal_gat = st["exp1a"]["_nominal"]["Gatenby"]
    exito = lambda arr, baseline=nominal_gat: float(np.mean(
        np.asarray(arr, float) > baseline))

    def bootstrap_median_ci(values, reps=5000):
        values = np.asarray(values, float)
        rng = np.random.default_rng(20260908)
        samples = rng.choice(values, size=(reps, len(values)), replace=True)
        return tuple(np.percentile(np.median(samples, axis=1), [2.5, 97.5]))

    # ---- Tabla resumen (markdown) ----
    lines = ["# EXP-1 Robustez — resultados\n",
             f"Semillas n={seeds} · pasos={st['config']['steps']} · métrica TTP-combinado\n",
             f"Nominal MAPPO: mediana={nom_med:.0f} d, tasa éxito vs "
             f"Gatenby({nominal_gat})={exito(nom):.0%}\n",
             "\n## EXP-1a — Perturbación paramétrica ±20% (mediana TTP MAPPO)\n",
             "| Parámetro | ×0.8 | ×0.9 | ×1.0 | ×1.1 | ×1.2 |",
             "|---|---|---|---|---|---|"]
    for key in PARAM_KEYS:
        row = [f"| `{key}` "]
        for f in FACTORS:
            cell = st["exp1a"].get(f"{key}@{f}")
            row.append(f"| {np.median(cell['mappo']):.0f} " if cell else "| - ")
        lines.append("".join(row) + "|")

    violations = []
    for cell, values in st["exp1a"].items():
        if cell == "_nominal":
            continue
        med = float(np.median(values["mappo"]))
        if not (med >= values["Gatenby"] >= values["MTD"]):
            violations.append(cell)
    lines += ["\n**Criterio pre-registrado:** " +
              ("NO se cumple globalmente. Celdas que invierten el orden: " +
               ", ".join(f"`{x}`" for x in violations)
               if violations else "se cumple en todas las celdas OAT."),
              "\n## EXP-1a conjunto — Monte Carlo ±20%\n",
              "| Entornos | mediana de medianas MAPPO | peor mediana | "
              "proporción de entornos con mediana MAPPO > Gatenby del mismo entorno |",
              "|---|---|---|---|"]
    joint = list(st.get("exp1a_joint", {}).values())
    if joint:
        joint_medians = np.array([np.median(c["mappo"]) for c in joint], float)
        joint_success = np.mean([np.median(c["mappo"]) > c["Gatenby"] for c in joint])
        lines.append(f"| {len(joint)} | {np.median(joint_medians):.0f} | "
                     f"{np.min(joint_medians):.0f} | {joint_success:.0%} |")

    lines += ["\n## Sensibilidad del endpoint\n",
              "| umbral carga | umbral resistencia | mediana MAPPO | Gatenby | éxito |",
              "|---|---|---|---|---|"]
    for cell in sorted(st.get("endpoint_sensitivity", {}).values(),
                       key=lambda c: (c["progression_threshold"],
                                      c["resistance_threshold"])):
        lines.append(f"| {cell['progression_threshold']:.2f}K | "
                     f"{cell['resistance_threshold']:.2f} | "
                     f"{np.median(cell['mappo']):.0f} | {cell['Gatenby']} | "
                     f"{cell['success_same_gatenby']:.0%} |")

    lines += ["\n## EXP-1b — Ruido de observación (gaussiano relativo)\n",
              "La unidad independiente es la política/semilla; las repeticiones de ruido están anidadas.\n",
              "| σ | mediana entre políticas | IC bootstrap 95% | [min, max] episodios |",
              "|---|---|---|---|"]
    for sigma in SIGMAS:
        c = st["exp1b"].get(f"sigma@{sigma}")
        if c:
            lo, hi = bootstrap_median_ci(c["per_seed_median"])
            lines.append(f"| {sigma:.2f} | {c['policy_level_median']:.0f} | "
                         f"[{lo:.1f}, {hi:.1f}] | [{c['all_min']}, {c['all_max']}] |")
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Tabla -> {REPORT}")

    # ---- Figura 1: degradación OAT ±20% ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, key in zip(axes.ravel(), PARAM_KEYS):
        xs, med, lo, hi, mtd, gat = [], [], [], [], [], []
        for f in FACTORS:
            c = st["exp1a"].get(f"{key}@{f}")
            if not c:
                continue
            arr = np.array(c["mappo"], float)
            xs.append(f); med.append(np.median(arr))
            lo.append(arr.min()); hi.append(arr.max())
            mtd.append(c["MTD"]); gat.append(c["Gatenby"])
        ax.fill_between(xs, lo, hi, color="#2E75B6", alpha=0.15)
        ax.plot(xs, med, "^-", color="#2E75B6", label="MAPPO (med)")
        ax.plot(xs, gat, "s-", color="#E67E22", label="Gatenby")
        ax.plot(xs, mtd, "o-", color="#C0392B", label="MTD")
        ax.axvline(1.0, ls="--", color="gray", alpha=0.6)
        ax.set_title(key); ax.set_xlabel("factor (±20%)")
        ax.set_ylabel("TTP-combinado (d)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("EXP-1a — Robustez a perturbación paramétrica ±20% (banda = min–max sobre semillas)",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(FIG_OAT, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Figura -> {FIG_OAT}")

    # ---- Figura 2: ruido de observación ----
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [s for s in SIGMAS if f"sigma@{s}" in st["exp1b"]]
    med = [st["exp1b"][f"sigma@{s}"]["policy_level_median"] for s in xs]
    lo = [st["exp1b"][f"sigma@{s}"]["all_min"] for s in xs]
    hi = [st["exp1b"][f"sigma@{s}"]["all_max"] for s in xs]
    ax.fill_between(xs, lo, hi, color="#2E75B6", alpha=0.15)
    ax.plot(xs, med, "^-", color="#2E75B6", label="MAPPO (mediana)")
    ax.axhline(nominal_gat, ls="--", color="#E67E22",
               label=f"Gatenby nominal ({nominal_gat})")
    ax.set_xlabel("σ ruido de observación (relativo)")
    ax.set_ylabel("TTP-combinado (d)")
    ax.set_title(f"EXP-1b — Robustez al ruido de observación (n={seeds})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_NOISE, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Figura -> {FIG_NOISE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--joint-samples", type=int, default=JOINT_SAMPLES)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.steps, args.joint_samples = 2, 2000, 5
    if args.plot:
        make_outputs(load_state()); return
    if args.seeds < 1 or args.steps < 1 or args.joint_samples < 1:
        ap.error("--seeds, --steps y --joint-samples deben ser positivos")
    run(args)


if __name__ == "__main__":
    main()
