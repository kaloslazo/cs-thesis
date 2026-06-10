"""
ablation_ctde.py — ¿El crítico centralizado (CTDE) importa? MAPPO vs IPPO.
Ejecutar:  python scripts/ablation_ctde.py --seeds 8

Entrena ambos con las MISMAS semillas y compara el TTP-resistencia:
  · MAPPO  (centralized=True):  crítico ve estado conjunto [S,R,c]
  · IPPO   (centralized=False): crítico ve solo la obs local de cada agente
Si MAPPO supera a IPPO, justifica empíricamente el "CTDE" de tu título.
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from gbmarl.tumor_env import TumorEnv
from gbmarl.mappo import train_mappo, obs_therapy
from gbmarl.evalutils import ttp_combinado, FIXED_PHI

def policy_from(th):
    import torch
    def pol(state):
        with torch.no_grad():
            a = th.actor_mean(torch.tensor(obs_therapy(state))).clamp(th.a_low, th.a_high)
        return float(a[0])
    return pol


def correr(env, centralized, seeds, steps):
    ttps = []
    for k in range(seeds):
        th, _, _ = train_mappo(env, total_timesteps=steps, seed=k,
                               centralized=centralized, verbose=False)
        ttps.append(ttp_combinado(env, policy_from(th)))
    return np.array(ttps, dtype=float)


def resumen(nombre, arr):
    exito = int((arr >= 180).sum())
    print(f"  {nombre:<6} media={arr.mean():6.1f} ± {arr.std(ddof=1):5.1f} | "
          f"mediana={np.median(arr):5.0f} | éxito(180d)={exito}/{len(arr)} | {arr.astype(int).tolist()}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=120000)
    args = ap.parse_args()

    env = TumorEnv(horizon_days=180)
    print("=" * 72)
    print("ABLACIÓN CTDE — MAPPO (crítico global) vs IPPO (crítico local)")
    print("=" * 72)
    print(f"  Entrenando {args.seeds} semillas cada uno...")
    mappo = resumen("MAPPO", correr(env, True, args.seeds, args.steps))
    ippo = resumen("IPPO", correr(env, False, args.seeds, args.steps))

    t, p = stats.ttest_ind(mappo, ippo, alternative="greater", equal_var=False)
    print("-" * 72)
    print(f"  MAPPO > IPPO :  p = {p:.4f}  ->  ", end="")
    print("✓ el crítico centralizado AYUDA (justifica CTDE)" if p < 0.05
          else "~ sin diferencia significativa (reportar honesto)")
    print("=" * 72)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    medias = [mappo.mean(), ippo.mean()]
    errs = [mappo.std(ddof=1), ippo.std(ddof=1)]
    ax.bar(["MAPPO\n(CTDE)", "IPPO\n(local)"], medias, yerr=errs, capsize=8,
           color=["#2E75B6", "#95A5A6"], alpha=0.85)
    for i, v in enumerate(medias):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", fontweight="bold")
    ax.set_ylabel("TTP-combinado (días)")
    ax.set_title(f"Ablación CTDE (n={args.seeds} semillas)")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/ablation_ctde.png", dpi=150, facecolor="white")
    print("Figura guardada en outputs/ablation_ctde.png")


if __name__ == "__main__":
    main()