"""
sensitivity.py — ¿El ranking MAPPO > Gatenby > MTD sobrevive al variar las
constantes elegidas a mano? Análisis de sensibilidad local (un parámetro a la vez).

Ejecutar:
  python scripts/sensitivity.py --fast              # solo baselines (segundos)
  python scripts/sensitivity.py --seeds 3           # con MAPPO (~30 min)

Varía 6 constantes alrededor de su valor por defecto y, en cada valor, recalcula
el TTP-combinado de MTD, Gatenby y MAPPO. Si la curva de MAPPO se mantiene por
encima en todo el rango, el resultado es robusto.
"""
import os
import sys
import argparse
import dataclasses
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from gbmarl.tumor_env import TumorEnv
from gbmarl.config import Params, load_calibration
from gbmarl.evalutils import ttp_combinado, Gatenby, U_MAX

# Params base (calibrados si existen; si no, defaults de literatura)
try:
    BASE = load_calibration()
except Exception as e:
    print(f"[sensitivity] sin calibration.json ({e}); uso defaults"); BASE = Params()


def make_env(name, kind, value):
    """Construye el env con UN parámetro modificado (los demás por defecto)."""
    if kind == "env":
        return TumorEnv(**{name: value})
    return TumorEnv(params=dataclasses.replace(BASE, **{name: value}))


def policy_from(th):
    import torch
    from gbmarl.mappo import obs_therapy
    def pol(state):
        with torch.no_grad():
            a = th.actor_mean(torch.tensor(obs_therapy(state))).clamp(th.a_low, th.a_high)
        return float(a[0])
    return pol


def eval_setting(env, seeds, steps, fast):
    mtd = ttp_combinado(env, lambda s: U_MAX)
    gat = ttp_combinado(env, Gatenby())
    if fast:
        return mtd, gat, np.nan, 0.0
    from gbmarl.mappo import train_mappo
    ttps = []
    for k in range(seeds):
        th, _, _ = train_mappo(env, total_timesteps=steps, seed=k, verbose=False)
        ttps.append(ttp_combinado(env, policy_from(th)))
    arr = np.array(ttps, dtype=float)
    return mtd, gat, float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--fast", action="store_true", help="solo baselines, sin entrenar MAPPO")
    args = ap.parse_args()

    # (nombre, tipo, valores, default)  — tipo: "env" (kwarg) o "param" (calibración)
    specs = [
        ("phi_max",               "env",   [0.02, 0.05, 0.10, 0.20], 0.05),
        ("progression_threshold", "env",   [0.70, 0.80, 0.90],       0.80),
        ("r_majority",            "env",   [0.40, 0.50, 0.60],       0.50),
        ("tox_weight",            "env",   [0.00, 0.05, 0.10],       0.05),
        ("delta_max_S",           "param", [round(BASE.delta_max_S * f, 3) for f in (0.7, 1.0, 1.3)], BASE.delta_max_S),
        ("ic50_R",                "param", [round(BASE.ic50_R * f, 3) for f in (0.7, 1.0, 1.3)],       BASE.ic50_R),
    ]

    print("=" * 78)
    print(f"ANÁLISIS DE SENSIBILIDAD — TTP-combinado  (seeds={args.seeds}, fast={args.fast})")
    print("=" * 78)

    resultados = {}
    for name, kind, valores, defecto in specs:
        print(f"\n-- {name} (default={defecto}) [{kind}] --")
        filas = []
        for v in valores:
            try:
                env = make_env(name, kind, v)
            except Exception as e:
                print(f"   valor {v}: ERROR construyendo env ({e}); se omite"); continue
            mtd, gat, mp, sd = eval_setting(env, args.seeds, args.steps, args.fast)
            filas.append((v, mtd, gat, mp, sd))
            mp_txt = "  (omitido)" if np.isnan(mp) else f"{mp:.1f}±{sd:.1f}"
            print(f"   {name}={v:<7}  MTD={mtd:<4} Gatenby={gat:<4} MAPPO={mp_txt}")
        resultados[name] = (filas, defecto)

    # Figura: un subplot por parámetro
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (name, kind, valores, defecto) in zip(axes.ravel(), specs):
        filas, defv = resultados[name]
        if not filas:
            ax.set_visible(False); continue
        xs = [f[0] for f in filas]
        ax.plot(xs, [f[1] for f in filas], "o-", color="#C0392B", label="MTD")
        ax.plot(xs, [f[2] for f in filas], "s-", color="#E67E22", label="Gatenby")
        if not args.fast:
            mp = np.array([f[3] for f in filas]); sd = np.array([f[4] for f in filas])
            ax.plot(xs, mp, "^-", color="#2E75B6", label="MAPPO")
            ax.fill_between(xs, mp - sd, mp + sd, color="#2E75B6", alpha=0.18)
        ax.axvline(defv, ls="--", color="gray", alpha=0.7)
        ax.set_title(name); ax.set_xlabel("valor"); ax.set_ylabel("TTP-combinado (días)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Sensibilidad del ranking a las constantes elegidas a mano "
                 "(línea gris = valor por defecto)", fontsize=13)
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out = "outputs/sensitivity.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"\nFigura guardada en {out}")


if __name__ == "__main__":
    main()