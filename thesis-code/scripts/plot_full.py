"""Genera las figuras del run completo (120k): curvas de co-evolución y TTP."""
import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CK = "outputs/ckpt"; OUT = "outputs"
N_STEPS = 2048

# ---------- Fig 1: curvas de co-evolución MAPPO titular ----------
with open(f"{CK}/mappo_main.pkl", "rb") as f:
    hist = np.array(pickle.load(f)["hist"])
x = np.arange(len(hist)) * N_STEPS
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(x, hist[:, 0], color="#2E75B6", lw=1.8)
ax[0].set_title("Agente Terapia — retorno"); ax[0].set_xlabel("pasos de entorno")
ax[0].set_ylabel("retorno medio (10 ep)"); ax[0].grid(alpha=0.3)
ax[1].plot(x, hist[:, 1], color="#C0392B", lw=1.8)
ax[1].set_title("Agente Tumor — retorno"); ax[1].set_xlabel("pasos de entorno")
ax[1].grid(alpha=0.3)
fig.suptitle("MAPPO self-play (CTDE): co-evolución terapia vs tumor — corrida completa 120k pasos")
fig.tight_layout(); fig.savefig(f"{OUT}/mappo_learning_curves.png", dpi=150, facecolor="white")
plt.close(fig)

# ---------- Fig 2: TTP baselines vs RL + dispersión por semilla ----------
ev = json.load(open(f"{OUT}/eval_full.json"))
base = ev["baselines"]
mappo = sorted([r["ttp"] for r in ev["mappo"]])
ippo = sorted([r["ttp"] for r in ev["ippo"]])
g = base["Gatenby"][0]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
# (a) barras comparativas (mediana RL)
labels = ["Sin\ntratar", "MTD", "Gatenby", "IPPO\n(mediana)", "MAPPO\n(mediana)"]
vals = [base["Sin tratamiento"][0], base["MTD"][0], g, int(np.median(ippo)), int(np.median(mappo))]
colors = ["#7F8C8D", "#E67E22", "#27AE60", "#8E44AD", "#2E75B6"]
bars = ax[0].bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax[0].text(b.get_x() + b.get_width()/2, v + 0.4, f"{v}d", ha="center", fontweight="bold")
ax[0].set_ylabel("TTP-combinado (días)")
ax[0].set_title("(a) Tiempo hasta progresión: heurísticas vs RL")
ax[0].axhline(g, ls="--", color="#27AE60", alpha=0.5)
ax[0].grid(axis="y", alpha=0.3)

# (b) dispersión por semilla (bimodalidad → tasa de éxito)
ax[1].axhline(g, ls="--", color="#27AE60", label=f"Gatenby ({g}d)")
ax[1].scatter([0]*len(ippo), ippo, s=90, color="#8E44AD", alpha=0.8, label="IPPO", zorder=3)
ax[1].scatter([1]*len(mappo), mappo, s=90, color="#2E75B6", alpha=0.8, label="MAPPO-CTDE", zorder=3)
ax[1].plot([-0.25, 0.25], [np.median(ippo)]*2, color="#8E44AD", lw=2)
ax[1].plot([0.75, 1.25], [np.median(mappo)]*2, color="#2E75B6", lw=2)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["IPPO", "MAPPO-CTDE"])
ax[1].set_ylabel("TTP-combinado (días)")
ax[1].set_title("(b) Ablación CTDE: dispersión por semilla (n=5)")
ax[1].legend(loc="upper left"); ax[1].grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/resultados_ttp.png", dpi=150, facecolor="white")
plt.close(fig)
print("[ok] figuras: mappo_learning_curves.png, resultados_ttp.png")
