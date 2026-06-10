"""
validate_seeds.py — Tier 1: significancia estadística del resultado central.
Ejecutar desde la raíz:  python scripts/validate_seeds.py --seeds 10

Entrena MAPPO con N semillas, mide el TTP-resistencia de cada política aprendida,
y compara la media ± desviación contra las baselines deterministas (MTD, Gatenby)
con un test t. Convierte tu "un solo run" en un resultado defendible.
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
from gbmarl.evalutils import ttp_combinado, Gatenby, FIXED_PHI, U_MAX

def policy_from(th):
    import torch
    def pol(state):
        with torch.no_grad():
            a = th.actor_mean(torch.tensor(obs_therapy(state))).clamp(th.a_low, th.a_high)
        return float(a[0])
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=120000)
    args = ap.parse_args()

    env = TumorEnv(horizon_days=180)

    # Baselines deterministas (no dependen de semilla)
    ttp_mtd = ttp_combinado(env, lambda s: U_MAX)
    ttp_gat = ttp_combinado(env, Gatenby())
    print("=" * 60)
    print("VALIDACIÓN MULTI-SEMILLA — TTP-combinado (días, métrica correcta)")
    print("=" * 60)
    print(f"  Baselines:  MTD={ttp_mtd}   Gatenby={ttp_gat}")
    print(f"  Entrenando MAPPO con {args.seeds} semillas ({args.steps} pasos c/u)...")

    mappo_ttps = []
    for k in range(args.seeds):
        th, _, _ = train_mappo(env, total_timesteps=args.steps, seed=k, verbose=False)
        ttp = ttp_combinado(env, policy_from(th))
        mappo_ttps.append(ttp)
        print(f"    semilla {k+1:2d}/{args.seeds}:  TTP = {ttp}")

    arr = np.array(mappo_ttps, dtype=float)
    media, sd = arr.mean(), arr.std(ddof=1)
    # Test t de una muestra: ¿MAPPO supera significativamente a Gatenby?
    t_g, p_g = stats.ttest_1samp(arr, ttp_gat, alternative="greater")
    t_m, p_m = stats.ttest_1samp(arr, ttp_mtd, alternative="greater")

    print("\n" + "=" * 60)
    print(f"  MAPPO:  media = {media:.1f} ± {sd:.1f}  (n={args.seeds})")
    print(f"  vs Gatenby ({ttp_gat}):  p = {p_g:.4f}  ->  ", end="")
    print("✓ significativo" if p_g < 0.05 else "✗ no significativo")
    print(f"  vs MTD ({ttp_mtd}):      p = {p_m:.4f}  ->  ", end="")
    print("✓ significativo" if p_m < 0.05 else "✗ no significativo")
    print("=" * 60)

    # Gráfica de barras con barra de error
    fig, ax = plt.subplots(figsize=(7, 5))
    nombres = ["MTD", "Gatenby", "MAPPO"]
    valores = [ttp_mtd, ttp_gat, media]
    errores = [0, 0, sd]
    colores = ["#C0392B", "#E67E22", "#2E75B6"]
    ax.bar(nombres, valores, yerr=errores, capsize=8, color=colores, alpha=0.85)
    ax.set_ylabel("TTP-combinado (días)")
    ax.set_title(f"Tiempo a dominancia resistente (n={args.seeds} semillas)")
    for i, v in enumerate(valores):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/validation_seeds.png", dpi=150, facecolor="white")
    print("Figura guardada en outputs/validation_seeds.png")


if __name__ == "__main__":
    main()