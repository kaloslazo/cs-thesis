"""
inspect_seed.py — Entrena UNA semilla y grafica su política aprendida.
Ejecutar:  python scripts/inspect_seed.py --seed 0     (una cuenca "41")
           python scripts/inspect_seed.py --seed 2     (una cuenca "180")
Compara las dos figuras para entender qué hace distinto la política ganadora.
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch

from gbmarl.tumor_env import TumorEnv
from gbmarl.mappo import train_mappo, obs_therapy, obs_tumor


@torch.no_grad()
def rollout(env, th, tu):
    obs, _ = env.reset(seed=0); state = obs["therapy"]
    S, R, C, U, PHI = [], [], [], [], []; done = False
    while not done:
        s, r, c = state
        a_th = th.actor_mean(torch.tensor(obs_therapy(state))).clamp(th.a_low, th.a_high)
        a_tu = tu.actor_mean(torch.tensor(obs_tumor(state))).clamp(tu.a_low, tu.a_high)
        S.append(s); R.append(r); C.append(c); U.append(float(a_th[0])); PHI.append(float(a_tu[0]))
        obs, rew, term, trunc, info = env.step({"therapy": a_th.numpy(), "tumor": a_tu.numpy()})
        done = (term.get("therapy", True) if term else True) or \
               (trunc.get("therapy", True) if trunc else True)
        state = obs["therapy"]
    return map(np.array, (S, R, C, U, PHI))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=120000); args = ap.parse_args()

    env = TumorEnv(horizon_days=180)
    print(f"Entrenando semilla {args.seed}...")
    th, tu, _ = train_mappo(env, total_timesteps=args.steps, seed=args.seed, verbose=False)
    S, R, C, U, PHI = rollout(env, th, tu)
    fracR = R / (S + R + 1e-9)
    ttp = int(np.argmax(fracR > env.r_majority)) if (fracR > env.r_majority).any() else len(S)

    print("=" * 56)
    print(f"SEMILLA {args.seed}  —  TTP-resistencia = {ttp} días")
    print(f"  Dosis u:  media={U.mean():.3f}  max={U.max():.3f}")
    print(f"  Transición φ: media={PHI.mean():.4f}")
    print(f"  Fracción R final: {fracR[-1]:.3f}")
    print("=" * 56)

    t = np.arange(len(S))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].plot(t, S, label="S", color="#2E75B6", lw=2); ax[0].plot(t, R, label="R", color="#C0392B", lw=2)
    ax[0].set_title(f"Poblaciones (semilla {args.seed})"); ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_xlabel("Días")
    ax[1].plot(t, fracR, color="#8E44AD", lw=2); ax[1].axhline(env.r_majority, ls="--", color="k")
    ax[1].set_title("Fracción resistente"); ax[1].set_ylim(0, 1.05); ax[1].grid(alpha=0.3); ax[1].set_xlabel("Días")
    ax[2].plot(t, U, label="Dosis u", color="#27AE60", lw=2); ax[2].plot(t, PHI, label="φ tumor", color="#C0392B", lw=2)
    ax[2].set_title("Acciones"); ax[2].legend(); ax[2].grid(alpha=0.3); ax[2].set_xlabel("Días")
    fig.suptitle(f"Política aprendida — semilla {args.seed} (TTP={ttp}d)")
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/policy_seed{args.seed}.png"
    fig.savefig(out, dpi=150, facecolor="white"); print(f"Figura: {out}")


if __name__ == "__main__":
    main()