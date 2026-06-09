"""
inspect_policies.py — Diagnóstico: ¿qué aprendieron los agentes MAPPO?
Ejecuta el episodio con las políticas entrenadas y grafica dosis, φ y S/R/c.
Ejecutar desde la raíz:  python scripts/inspect_policies.py

Sirve para entender POR QUÉ el adversario no endureció el problema antes de
tocar hiperparámetros. Si ves "dosis suave + φ bajo + S domina", confirma que
la resistencia no le paga al tumor con los parámetros actuales.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch

from gbmarl.tumor_env import TumorEnv
from gbmarl.mappo import Agent, obs_therapy, obs_tumor

HORIZON = 180


def load_agents(env):
    th = Agent(2, 3, 1, env.action_space("therapy").low, env.action_space("therapy").high)
    tu = Agent(2, 3, 1, env.action_space("tumor").low, env.action_space("tumor").high)
    th.load_state_dict(torch.load("outputs/models/mappo_therapy.pt"))
    tu.load_state_dict(torch.load("outputs/models/mappo_tumor.pt"))
    th.eval(); tu.eval()
    return th, tu


@torch.no_grad()
def rollout(env, th, tu, seed=0):
    obs, _ = env.reset(seed=seed); state = obs["therapy"]
    S, R, C, U, PHI = [], [], [], [], []
    done = False
    while not done:
        s, r, c = state
        a_th = th.actor_mean(torch.tensor(obs_therapy(state))).clamp(th.a_low, th.a_high)
        a_tu = tu.actor_mean(torch.tensor(obs_tumor(state))).clamp(tu.a_low, tu.a_high)
        u, phi = float(a_th[0]), float(a_tu[0])
        S.append(s); R.append(r); C.append(c); U.append(u); PHI.append(phi)
        obs, rew, term, trunc, _ = env.step({"therapy": a_th.numpy(), "tumor": a_tu.numpy()})
        done = (term.get("therapy", True) if term else True) or \
               (trunc.get("therapy", True) if trunc else True)
        state = obs["therapy"]
    return map(np.array, (S, R, C, U, PHI))


def main():
    env = TumorEnv(horizon_days=HORIZON)
    th, tu = load_agents(env)
    S, R, C, U, PHI = rollout(env, th, tu)

    print("=" * 60)
    print("DIAGNÓSTICO DE POLÍTICAS APRENDIDAS")
    print("=" * 60)
    print(f"  Dosis terapia (u):  media={U.mean():.3f}  max={U.max():.3f}  (límite 1.0)")
    print(f"  Transición tumor (φ): media={PHI.mean():.4f}  max={PHI.max():.4f}  (límite 0.05)")
    print(f"  Carga final S={S[-1]:.3f}  R={R[-1]:.3f}  -> dominante: ", end="")
    print("RESISTENTES" if R[-1] > S[-1] else "SENSIBLES")
    print("=" * 60)

    t = np.arange(len(S))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(t, S, label="Sensibles (S)", color="#2E75B6", lw=2)
    ax[0].plot(t, R, label="Resistentes (R)", color="#C0392B", lw=2)
    ax[0].plot(t, C, label="Droga (c)", color="#7F8C8D", lw=1, ls=":")
    ax[0].set_title("Trayectoria bajo políticas aprendidas"); ax[0].set_xlabel("Días")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(t, U, label="Dosis terapia (u)", color="#27AE60", lw=2)
    ax[1].plot(t, PHI, label="Transición tumor (φ)", color="#C0392B", lw=2)
    ax[1].set_title("Acciones de los agentes"); ax[1].set_xlabel("Días")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/policies_diagnostic.png", dpi=150, facecolor="white")
    print("Figura guardada en outputs/policies_diagnostic.png")


if __name__ == "__main__":
    main()