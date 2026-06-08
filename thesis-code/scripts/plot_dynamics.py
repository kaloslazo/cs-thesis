"""
Visualiza la dinámica tumoral bajo 3 regímenes de dosis.
Ejecutar desde la raíz:  python scripts/plot_dynamics.py
Es tu PRIMERA gráfica: muestra por qué el tratamiento estático falla.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from gbmarl.config import Params
from gbmarl.dynamics import simulate

P = Params()
STATE0 = [0.40, 0.01, 0.0]
N_DIAS = 200

regimenes = {
    "Sin tratamiento":        lambda t, s: 0.0,
    "Dosis máxima (MTD)":     lambda t, s: 0.6,
    "Dosis pulsada (1 sí/1 no)": lambda t, s: 0.6 if int(t) // 20 % 2 == 0 else 0.0,
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, (nombre, dose_fn) in zip(axes, regimenes.items()):
    out = simulate(STATE0, dose_fn, lambda t, s: P.phi_base, n_days=N_DIAS, p=P)
    ax.plot(out["t"], out["S"], label="Sensibles (S)", color="#2E75B6", lw=2)
    ax.plot(out["t"], out["R"], label="Resistentes (R)", color="#C0392B", lw=2)
    ax.plot(out["t"], out["S"] + out["R"], label="Tumor total", color="#555", lw=1.2, ls="--")
    ax.set_title(nombre, fontsize=11)
    ax.set_xlabel("Días")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Densidad celular (norm.)")
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Dinámica tumoral: por qué la dosis estática selecciona resistencia",
             fontsize=12, y=1.02)
fig.tight_layout()

os.makedirs("outputs", exist_ok=True)
fig.savefig("outputs/dynamics_baseline.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Figura guardada en outputs/dynamics_baseline.png")