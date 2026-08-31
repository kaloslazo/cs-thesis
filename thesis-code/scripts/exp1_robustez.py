"""
exp1_robustez.py — EXP-1 (Semana 3 PFC II): robustez de la política MAPPO-CTDE.

Dos fuentes de incertidumbre (ver docs/protocolo_experimental_pfc2.md):
  · EXP-1a  Perturbación paramétrica ±20% (OAT, un parámetro a la vez) sobre las
            constantes calibradas/literatura. La política se ENTRENA en el entorno
            nominal y se EVALÚA en el perturbado (robustez de transferencia).
  · EXP-1b  Ruido de observación gaussiano relativo sobre lo que ve la TERAPIA
            (S+R, c), SOLO en evaluación (wrapper, no se re-entrena ni se toca
            tumor_env.py). Mide robustez de la política ya aprendida.

Métrica: TTP-combinado (evalutils). Estadística: bimodal -> mediana [min,max] y
tasa de éxito vs Gatenby (TTP>27). n=15 semillas.

Reanudable: guarda modelos en outputs/models/exp1_mappo_{seed}.pt y resultados
parciales en outputs/exp1_robustez.json. Re-ejecutar continúa donde quedó.

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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gbmarl.tumor_env import TumorEnv
from gbmarl.config import Params, load_calibration
from gbmarl.mappo import train_mappo, Agent, obs_therapy
from gbmarl.evalutils import ttp_combinado, Gatenby, U_MAX, FIXED_PHI

MODELS = "outputs/models"
STATE = "outputs/exp1_robustez.json"
GATENBY_NOMINAL = 27  # umbral de éxito (línea base)

# Parámetros a perturbar ±20% y sus factores
PARAM_KEYS = ["ic50_S", "ic50_R", "delta_max_S", "delta_max_R",
              "alpha_S", "alpha_R", "lambda_c", "K"]
FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
SIGMAS = [0.0, 0.05, 0.10, 0.20]
NOISE_REPS = 30

try:
    BASE = load_calibration()
except Exception as e:
    print(f"[exp1] sin calibration.json ({e}); uso placeholders"); BASE = Params()


# ───────────────────────── utilidades ──────────────────────────
def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"policies": {}, "exp1a": {}, "exp1b": {}, "config": {}}


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
    path = f"{MODELS}/exp1_mappo_{seed}.pt"
    env = TumorEnv(horizon_days=180)               # entorno NOMINAL
    if os.path.exists(path):
        ag = make_agent(env)
        ag.load_state_dict(torch.load(path)); ag.eval()
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
            return info["day"]


# ───────────────────────── driver ──────────────────────────
def run(args):
    st = load_state()
    st["config"] = {"seeds": args.seeds, "steps": args.steps,
                    "factors": FACTORS, "sigmas": SIGMAS, "noise_reps": NOISE_REPS}
    t0 = time.time()

    # --- Fase A: entrenar/cargar n políticas nominales ---
    agents = {}
    for seed in range(args.seeds):
        ag = train_or_load_policy(seed, args.steps)
        agents[seed] = ag
        st["policies"][str(seed)] = f"{MODELS}/exp1_mappo_{seed}.pt"
        save_state(st)
        print(f"[A] politica seed={seed} lista ({time.time()-t0:.0f}s)")

    # --- Fase B: EXP-1a perturbacion parametrica +-20% (OAT) ---
    # baseline nominal de las politicas
    env_nom = TumorEnv(horizon_days=180)
    nom = [ttp_combinado(env_nom, policy_from(agents[s])) for s in range(args.seeds)]
    st["exp1a"]["_nominal"] = {"mappo": nom,
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
            mappo = [ttp_combinado(env, policy_from(agents[s])) for s in range(args.seeds)]
            mtd = ttp_combinado(env, lambda s: U_MAX)
            gat = ttp_combinado(env, Gatenby())
            st["exp1a"][cell] = {"param": key, "factor": f, "value": round(val, 4),
                                 "mappo": mappo, "MTD": mtd, "Gatenby": gat}
            save_state(st)
            med = float(np.median(mappo))
            print(f"[B] {cell:22s} MAPPO_med={med:4.0f} MTD={mtd:3d} Gat={gat:3d}")

    # --- Fase C: EXP-1b ruido de observacion ---
    for sigma in SIGMAS:
        key = f"sigma@{sigma}"
        if key in st["exp1b"]:
            continue
        per_seed_med = []
        allttp = []
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 + s)
            reps = 1 if sigma == 0 else NOISE_REPS
            ttps = [ttp_noisy(env_nom, agents[s], sigma, rng) for _ in range(reps)]
            per_seed_med.append(float(np.median(ttps)))
            allttp.extend(ttps)
        st["exp1b"][key] = {"sigma": sigma, "per_seed_median": per_seed_med,
                            "all_median": float(np.median(allttp)),
                            "all_min": int(np.min(allttp)), "all_max": int(np.max(allttp))}
        save_state(st)
        print(f"[C] sigma={sigma:4.2f}  TTP mediana={np.median(allttp):4.0f} "
              f"[{np.min(allttp):.0f},{np.max(allttp):.0f}]")

    print(f"=== EXP-1 completo ({time.time()-t0:.0f}s) ===")
    make_outputs(st)


# ───────────────────────── tabla + figuras ──────────────────────────
def make_outputs(st):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = st["config"]["seeds"]
    nom = st["exp1a"]["_nominal"]["mappo"]
    nom_med = float(np.median(nom))
    exito = lambda arr: float(np.mean([x > GATENBY_NOMINAL for x in arr]))

    # ---- Tabla resumen (markdown) ----
    lines = ["# EXP-1 Robustez — resultados\n",
             f"Semillas n={seeds} · pasos={st['config']['steps']} · métrica TTP-combinado\n",
             f"Nominal MAPPO: mediana={nom_med:.0f} d, tasa éxito vs Gatenby(27)={exito(nom):.0%}\n",
             "\n## EXP-1a — Perturbación paramétrica ±20% (mediana TTP MAPPO)\n",
             "| Parámetro | ×0.8 | ×0.9 | ×1.0 | ×1.1 | ×1.2 |",
             "|---|---|---|---|---|---|"]
    for key in PARAM_KEYS:
        row = [f"| `{key}` "]
        for f in FACTORS:
            cell = st["exp1a"].get(f"{key}@{f}")
            row.append(f"| {np.median(cell['mappo']):.0f} " if cell else "| - ")
        lines.append("".join(row) + "|")
    lines += ["\n## EXP-1b — Ruido de observación (gaussiano relativo)\n",
              "| σ | TTP mediana | [min, max] |",
              "|---|---|---|"]
    for sigma in SIGMAS:
        c = st["exp1b"].get(f"sigma@{sigma}")
        if c:
            lines.append(f"| {sigma:.2f} | {c['all_median']:.0f} | [{c['all_min']}, {c['all_max']}] |")
    with open("outputs/exp1_robustez.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Tabla -> outputs/exp1_robustez.md")

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
    fig.tight_layout(); fig.savefig("outputs/exp1a_perturbacion.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Figura -> outputs/exp1a_perturbacion.png")

    # ---- Figura 2: ruido de observación ----
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [s for s in SIGMAS if f"sigma@{s}" in st["exp1b"]]
    med = [st["exp1b"][f"sigma@{s}"]["all_median"] for s in xs]
    lo = [st["exp1b"][f"sigma@{s}"]["all_min"] for s in xs]
    hi = [st["exp1b"][f"sigma@{s}"]["all_max"] for s in xs]
    ax.fill_between(xs, lo, hi, color="#2E75B6", alpha=0.15)
    ax.plot(xs, med, "^-", color="#2E75B6", label="MAPPO (mediana)")
    ax.axhline(GATENBY_NOMINAL, ls="--", color="#E67E22", label="Gatenby nominal (27)")
    ax.set_xlabel("σ ruido de observación (relativo)")
    ax.set_ylabel("TTP-combinado (d)")
    ax.set_title(f"EXP-1b — Robustez al ruido de observación (n={seeds})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("outputs/exp1b_ruido.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Figura -> outputs/exp1b_ruido.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.steps = 2, 2000
    if args.plot:
        make_outputs(load_state()); return
    run(args)


if __name__ == "__main__":
    main()
