"""
exp_salud.py — FASE 1 (prueba de concepto): variable de salud del paciente.

Objetivo de esta corrida: "ver si va y cómo va" — que el entorno 4-D con salud
entrena y que emerge el mecanismo que predice H1: MTD (dosis máxima) sacrifica la
salud del paciente (muere por toxicidad), mientras una terapia pulsada la preserva.

NO afecta el path 3-D: usa health_env + mappo_health, todo aditivo.

Uso:
  python scripts/exp_salud.py --smoke            # 2k pasos, rápido
  python scripts/exp_salud.py --steps 120000     # corrida seria
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gbmarl.health_env import TumorEnvHealth
from gbmarl.mappo_health import (train_mappo_health, ttp_health_detalle,
                                 obs_therapy_h, obs_tumor_h)


def eval_heuristica(env, dose_fn, phi=0.01):
    """Evalúa una terapia heurística (función de dosis) en el entorno con salud.
    Devuelve (día, modo_falla, H_final, H_medio, dosis_media)."""
    obs, _ = env.reset(seed=0)
    state = obs["therapy"]
    Hs, doses = [], []
    info = {}
    term = trunc = False
    while True:
        u = float(dose_fn(state))
        obs, _, terms, truncs, infos = env.step(
            {"therapy": np.array([u], np.float32),
             "tumor": np.array([phi], np.float32)})
        state = obs["therapy"]; info = infos["therapy"]
        Hs.append(info["H"]); doses.append(u)
        term = terms.get("therapy", False); trunc = truncs.get("therapy", False)
        if term or trunc:
            break
    if trunc and not term:
        modo = "sobrevivio"
    elif info.get("unhealthy"):
        modo = "salud"
    elif info.get("untreatable"):
        modo = "resistencia"
    elif info.get("progressed"):
        modo = "carga"
    else:
        modo = "extinto"
    return info["day"], modo, float(info["H"]), float(np.mean(Hs)), float(np.mean(doses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 2000

    env = TumorEnvHealth(horizon_days=180, phi_max=0.05)

    print("=== Baselines heurísticas en el entorno CON SALUD ===")
    # MTD: dosis máxima constante. Adaptativo: pulso bajo según carga.
    mtd = eval_heuristica(env, lambda s: 1.0)
    pulso = eval_heuristica(env, lambda s: 0.5 if (s[0] + s[1]) > 0.35 else 0.0)
    sin = eval_heuristica(env, lambda s: 0.0)
    for nombre, r in (("Sin tratar", sin), ("MTD (u=1.0)", mtd), ("Pulsado", pulso)):
        print(f"  {nombre:14s}: TTP={r[0]:3d}d  falla={r[1]:12s}  "
              f"H_final={r[2]:.2f}  H_medio={r[3]:.2f}  dosis_media={r[4]:.2f}")

    print(f"\n=== Entrenando MAPPO-CTDE con salud ({args.steps} pasos, seed={args.seed}) ===")
    th, tu, _ = train_mappo_health(env, total_timesteps=args.steps, seed=args.seed,
                                   centralized=True, verbose=False)
    dia, modo, Hf, Hm = ttp_health_detalle(env, th, tu)
    print(f"  MAPPO-salud    : TTP={dia:3d}d  falla={modo:12s}  H_final={Hf:.2f}  H_medio={Hm:.2f}")

    print("\n=== Lectura (H1: ¿la terapia aprendida protege la salud vs MTD?) ===")
    print(f"  MTD H_medio={mtd[3]:.2f} (falla '{mtd[1]}')  vs  "
          f"MAPPO H_medio={Hm:.2f} (falla '{modo}')")
    if mtd[1] == "salud":
        print("  -> MTD muere por TOXICIDAD: el mecanismo de salud discrimina como se esperaba.")
    else:
        print(f"  -> MTD no murió por salud ({mtd[1]}); revisar calibración de kappa_u/H_min.")


if __name__ == "__main__":
    main()
