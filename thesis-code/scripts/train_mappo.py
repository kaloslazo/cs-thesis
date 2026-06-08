"""
train_mappo.py — M4b: MAPPO adversarial (CTDE) por self-play.
Ejecutar desde la raíz:  python scripts/train_mappo.py

Resultado clave (la tesis): la terapia entrenada rinde PEOR contra un tumor que
APRENDE (adaptativo) que contra un tumor FIJO. Eso prueba que el adversario
evolutivo endurece el problema -> justifica el framework adversarial.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch

from gbmarl.tumor_env import TumorEnv
from gbmarl.mappo import train_mappo, obs_therapy, obs_tumor

TOTAL_STEPS = 120000
HORIZON = 180
FIXED_PHI = 0.01


def _clamp(agent, x):
    return agent.actor_mean(x).clamp(agent.a_low, agent.a_high)


@torch.no_grad()
def eval_therapy(env, th, tu=None, fixed_phi=None, n=20, seed=300):
    """Retorno medio de la terapia. tu=agente tumor adaptativo; o fixed_phi fijo."""
    rets = []
    for k in range(n):
        obs, _ = env.reset(seed=seed + k); state = obs["therapy"]; ret = 0.0
        done = False
        while not done:
            a_th = _clamp(th, torch.tensor(obs_therapy(state)))
            if tu is not None:
                a_tu = _clamp(tu, torch.tensor(obs_tumor(state))).numpy()
            else:
                a_tu = np.array([fixed_phi], np.float32)
            obs, rew, term, trunc, _ = env.step({"therapy": a_th.numpy(), "tumor": a_tu})
            done = (term.get("therapy", True) if term else True) or \
                   (trunc.get("therapy", True) if trunc else True)
            state = obs["therapy"]; ret += rew["therapy"]
        rets.append(ret)
    return float(np.mean(rets))


def main():
    env = TumorEnv(horizon_days=HORIZON)
    print("=" * 60)
    print("M4b — MAPPO adversarial (CTDE, self-play)")
    print("=" * 60)
    print("  Entrenando 2 agentes simultáneamente...")
    th, tu, hist = train_mappo(env, total_timesteps=TOTAL_STEPS, verbose=True)

    r_adapt = eval_therapy(env, th, tu=tu)            # vs tumor que aprende
    r_fixed = eval_therapy(env, th, fixed_phi=FIXED_PHI)  # vs tumor fijo
    print("\n" + "=" * 60)
    print(f"  Retorno terapia vs tumor FIJO      : {r_fixed:8.2f}")
    print(f"  Retorno terapia vs tumor ADAPTATIVO: {r_adapt:8.2f}")
    delta = r_fixed - r_adapt
    print(f"  El adversario adaptativo cuesta    : {delta:+8.2f}  ->  ", end="")
    print("✓ el adversario endurece el problema" if delta > 0
          else "~ similar (entrenar más o subir presión del tumor)")
    print("=" * 60)

    # Curvas de aprendizaje de ambos agentes
    h = np.array(hist)
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h[:, 0], color="#2E75B6"); ax[0].set_title("Retorno Terapia")
    ax[0].set_xlabel("Actualización"); ax[0].grid(alpha=0.3)
    ax[1].plot(h[:, 1], color="#C0392B"); ax[1].set_title("Retorno Tumor (adversario)")
    ax[1].set_xlabel("Actualización"); ax[1].grid(alpha=0.3)
    fig.suptitle("M4b: co-evolución terapia vs tumor (MAPPO self-play)")
    fig.tight_layout(); fig.savefig("outputs/mappo_learning_curves.png", dpi=150,
                                    facecolor="white")
    print("Curvas guardadas en outputs/mappo_learning_curves.png")

    os.makedirs("outputs/models", exist_ok=True)
    torch.save(th.state_dict(), "outputs/models/mappo_therapy.pt")
    torch.save(tu.state_dict(), "outputs/models/mappo_tumor.pt")
    print("Modelos guardados en outputs/models/")


if __name__ == "__main__":
    main()