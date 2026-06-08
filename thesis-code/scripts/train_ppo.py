"""
train_ppo.py — M4a: entrena la TERAPIA (single-agent) vs tumor fijo.
Ejecutar desde la raíz:  python scripts/train_ppo.py

Criterio de éxito (de-risk): el retorno del agente entrenado debe SUPERAR
claramente al de una política de dosis aleatoria. Si lo hace, tu entorno es
aprendible y tu pipeline RL funciona -> recién ahí pasamos a MAPPO.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch

from gbmarl.single_env import TherapyEnv
from gbmarl.ppo import train, evaluate, random_baseline

TOTAL_STEPS = 60000
FIXED_PHI = 0.01          # el tumor mantiene una transición fija (de-risk)
HORIZON = 180


def main():
    env = TherapyEnv(fixed_phi=FIXED_PHI, horizon_days=HORIZON)

    print("=" * 60)
    print("M4a — PPO single-agent (terapia vs tumor fijo)")
    print("=" * 60)

    base = random_baseline(env, n_eps=20)
    print(f"  Baseline (dosis aleatoria): {base:.3f}\n  Entrenando PPO...")

    ac, history = train(env, total_timesteps=TOTAL_STEPS, verbose=True)

    trained = evaluate(env, ac, n_eps=20)
    print("\n" + "=" * 60)
    print(f"  Retorno aleatorio : {base:8.3f}")
    print(f"  Retorno PPO       : {trained:8.3f}")
    mejora = trained - base
    print(f"  Mejora            : {mejora:+8.3f}  ->  ", end="")
    print("✓ APRENDIÓ (entorno aprendible)" if mejora > 0 else "✗ no superó al azar (revisar)")
    print("=" * 60)

    # Curva de aprendizaje
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(history, color="#2E75B6", lw=2, label="PPO (retorno medio 10 ep)")
    plt.axhline(base, color="#C0392B", ls="--", label="Baseline aleatorio")
    plt.xlabel("Actualización"); plt.ylabel("Retorno")
    plt.title("M4a: la terapia aprende a controlar el tumor")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("outputs/ppo_learning_curve.png", dpi=150, facecolor="white")
    print("Curva guardada en outputs/ppo_learning_curve.png")

    # Guardar el modelo
    os.makedirs("outputs/models", exist_ok=True)
    torch.save(ac.state_dict(), "outputs/models/ppo_therapy.pt")
    print("Modelo guardado en outputs/models/ppo_therapy.pt")


if __name__ == "__main__":
    main()