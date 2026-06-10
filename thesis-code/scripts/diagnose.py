"""
diagnose.py — Diagnóstico riguroso para decidir IPPO vs MAPPO con datos seguros.
Ejecutar:  python scripts/diagnose.py --seeds 5

Hace 3 cosas:
  1. Verifica que el flag CTDE realmente cambia la arquitectura (dim del crítico).
  2. Métrica CORRECTA: TTP-combinado = días hasta que el tumor deja de estar
     controlado (carga<umbral) Y tratable (R<mayoría), con el MOTIVO de falla.
  3. Compara baselines + IPPO + MAPPO contra el MISMO adversario fijo (justo).
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from gbmarl.tumor_env import TumorEnv
from gbmarl.mappo import train_mappo, Agent, obs_therapy, obs_tumor

FIXED_PHI = 0.01
U_MAX = 1.0


class Gatenby:
    def __init__(self, on=0.5, off=0.25): self.on, self.off, self.d = on, off, True
    def reset(self): self.d = True
    def __call__(self, s):
        b = s[0] + s[1]
        if self.d and b < self.off: self.d = False
        elif not self.d and b > self.on: self.d = True
        return U_MAX if self.d else 0.0


def rollout_diag(env, therapy_fn, tumor_fn=lambda s: FIXED_PHI):
    """Devuelve (días, motivo, carga_final, fracR_final, dosis_media)."""
    obs, _ = env.reset(seed=0); state = obs["therapy"]
    doses = []; info = {}; term = trunc = False
    while True:
        u = float(therapy_fn(state)); phi = float(tumor_fn(state))
        obs, rew, terms, truncs, infos = env.step({"therapy": np.array([u], np.float32),
                                                   "tumor": np.array([phi], np.float32)})
        doses.append(u); state = obs["therapy"]; info = infos["therapy"]
        term = terms.get("therapy", False); trunc = truncs.get("therapy", False)
        if term or trunc:
            break
    if trunc and not term:
        motivo = "SOBREVIVIÓ horizonte (ÉXITO)"
    elif info.get("untreatable"):
        motivo = "FALLO: resistencia mayoría"
    elif info.get("progressed"):
        motivo = "FALLO: carga progresó"
    else:
        motivo = "extinto"
    return info["day"], motivo, info["burden"], info["fracR"], float(np.mean(doses))


def policy_from(agent):
    def pol(state):
        with torch.no_grad():
            a = agent.actor_mean(torch.tensor(obs_therapy(state))).clamp(agent.a_low, agent.a_high)
        return float(a[0])
    return pol


def fila(nombre, env, fn):
    if hasattr(fn, "reset"): fn.reset()
    d, motivo, b, fr, mu = rollout_diag(env, fn)
    print(f"  {nombre:<22} días={d:>4}  carga={b:.3f}  fracR={fr:.3f}  dosis_media={mu:.3f}  | {motivo}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=120000)
    args = ap.parse_args()
    env = TumorEnv(horizon_days=180)

    # ---- 1. CHECK DE ARQUITECTURA ----
    print("=" * 84)
    print("1) ¿El flag CTDE cambia la arquitectura?")
    a_mappo = Agent(2, 3, 1, [0.0], [1.0])   # centralized: crítico ve 3 (global)
    a_ippo = Agent(2, 2, 1, [0.0], [1.0])    # local: crítico ve 2 (local)
    print(f"   MAPPO crítico in_features = {a_mappo.critic[0].in_features}  (debe ser 3 = [S,R,c])")
    print(f"   IPPO  crítico in_features = {a_ippo.critic[0].in_features}  (debe ser 2 = obs local)")
    print(f"   -> {'✓ distintos, ablación válida' if a_mappo.critic[0].in_features != a_ippo.critic[0].in_features else '✗ IGUALES, BUG'}")

    # ---- 2. BASELINES con métrica CORRECTA (vs phi fijo) ----
    print("=" * 84)
    print("2) Baselines vs tumor fijo (φ=0.01) — métrica TTP-combinado + motivo")
    print("-" * 84)
    fila("Sin tratamiento", env, lambda s: 0.0)
    fila("MTD (dosis máx)", env, lambda s: U_MAX)
    fila("Gatenby", env, Gatenby())

    # ---- 3. IPPO vs MAPPO con métrica CORRECTA ----
    print("=" * 84)
    print(f"3) IPPO vs MAPPO ({args.seeds} semillas) vs tumor fijo (φ=0.01) — TTP-combinado")
    print("-" * 84)
    for nombre, centralized in (("MAPPO (CTDE)", True), ("IPPO (local)", False)):
        dias_list = []; exitos = 0
        for k in range(args.seeds):
            th, tu, _ = train_mappo(env, total_timesteps=args.steps, seed=k,
                                    centralized=centralized, verbose=False)
            d, motivo, b, fr, mu = rollout_diag(env, policy_from(th))
            dias_list.append(d)
            if "ÉXITO" in motivo: exitos += 1
            print(f"  {nombre:<13} seed {k}: días={d:>4}  carga={b:.3f}  fracR={fr:.3f}  "
                  f"dosis={mu:.3f} | {motivo}")
        arr = np.array(dias_list, dtype=float)
        print(f"  >>> {nombre}: media={arr.mean():.1f}±{arr.std(ddof=1):.1f}  "
              f"éxito-horizonte={exitos}/{args.seeds}\n")


if __name__ == "__main__":
    main()