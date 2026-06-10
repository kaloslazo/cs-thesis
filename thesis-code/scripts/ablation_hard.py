"""
ablation_hard.py — ¿CTDE ayuda cuando el adversario es MÁS FUERTE?
Ejecutar:  python scripts/ablation_hard.py --phi_max 0.2 --seeds 6

Repite la ablación MAPPO vs IPPO pero en un régimen difícil:
  · phi_max alto  -> el tumor puede convertir S->R más rápido (más adversarial).
  · evalúa la terapia contra su tumor ADAPTATIVO co-entrenado (peor caso),
    no contra un φ fijo débil.
Hipótesis: con observabilidad parcial más severa, el crítico centralizado
(MAPPO) produce mejores terapias que el local (IPPO).
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import stats

from gbmarl.tumor_env import TumorEnv
from gbmarl.mappo import train_mappo, obs_therapy, obs_tumor


@torch.no_grad()
def ttp_vs_adaptivo(env, th, tu):
    """TTP-combinado con AMBOS agentes aprendidos actuando (métrica correcta)."""
    obs, _ = env.reset(seed=0); state = obs["therapy"]; info = {}
    while True:
        a_th = th.actor_mean(torch.tensor(obs_therapy(state))).clamp(th.a_low, th.a_high)
        a_tu = tu.actor_mean(torch.tensor(obs_tumor(state))).clamp(tu.a_low, tu.a_high)
        obs, rew, terms, truncs, infos = env.step({"therapy": a_th.numpy(), "tumor": a_tu.numpy()})
        state = obs["therapy"]; info = infos["therapy"]
        if terms.get("therapy", False) or truncs.get("therapy", False):
            return info["day"]


def correr(env, centralized, seeds, steps):
    out = []
    for k in range(seeds):
        th, tu, _ = train_mappo(env, total_timesteps=steps, seed=k,
                                centralized=centralized, verbose=False)
        out.append(ttp_vs_adaptivo(env, th, tu))
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_max", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=120000)
    args = ap.parse_args()

    env = TumorEnv(horizon_days=180, phi_max=args.phi_max)
    print("=" * 72)
    print(f"ABLACIÓN EN RÉGIMEN DIFÍCIL — phi_max={args.phi_max} (vs tumor adaptativo)")
    print("=" * 72)
    mappo = correr(env, True, args.seeds, args.steps)
    ippo = correr(env, False, args.seeds, args.steps)
    print(f"  MAPPO (CTDE):  media={mappo.mean():6.1f} ± {mappo.std(ddof=1):5.1f}  {mappo.astype(int).tolist()}")
    print(f"  IPPO  (local): media={ippo.mean():6.1f} ± {ippo.std(ddof=1):5.1f}  {ippo.astype(int).tolist()}")

    if mappo.std() > 0 or ippo.std() > 0:
        t, p = stats.ttest_ind(mappo, ippo, alternative="greater", equal_var=False)
        print("-" * 72)
        print(f"  MAPPO > IPPO:  p = {p:.4f}  ->  ", end="")
        print("✓ CTDE ayuda en régimen difícil (justificación encontrada)" if p < 0.05
              else "~ aún sin ventaja clara de CTDE")
    print("=" * 72)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["MAPPO\n(CTDE)", "IPPO\n(local)"], [mappo.mean(), ippo.mean()],
           yerr=[mappo.std(ddof=1), ippo.std(ddof=1)], capsize=8,
           color=["#2E75B6", "#95A5A6"], alpha=0.85)
    ax.set_ylabel("TTP-resistencia (días)")
    ax.set_title(f"CTDE vs local — adversario fuerte (phi_max={args.phi_max}, n={args.seeds})")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(f"outputs/ablation_hard_phi{args.phi_max}.png", dpi=150, facecolor="white")
    print(f"Figura guardada en outputs/ablation_hard_phi{args.phi_max}.png")


if __name__ == "__main__":
    main()