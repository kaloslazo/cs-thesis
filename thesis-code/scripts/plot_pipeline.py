"""Genera el diagrama del pipeline experimental reproducible (inside-out)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "outputs"
phases = [
    ("1. Datos\nGDSC2 + DepMap", "34 líneas GBM / 24 genómica\nbrecha S:R ~9x", "#34495E"),
    ("2. Calibración", "IC50 y Hill del fármaco\n(datos) + literatura", "#2980B9"),
    ("3. Tests del\nsimulador", "EDO/RK4 validada\n10/10 pruebas", "#16A085"),
    ("4. PPO\nsingle-agent", "de-risk: ¿es aprendible\nel entorno? (tumor fijo)", "#27AE60"),
    ("5. MAPPO\nself-play (CTDE)", "modelo TITULAR\ncrítico centralizado", "#C0392B"),
    ("6. Ablación\nCTDE vs IPPO", "¿aporta el crítico\ncentralizado?", "#8E44AD"),
    ("7. Evaluación\nTTP-combinado", "vs Sin tratar / MTD /\nGatenby (métrica auditada)", "#D68910"),
]

fig, ax = plt.subplots(figsize=(15, 4.2))
n = len(phases)
bw, bh = 1.72, 1.55
gap = 0.36
x = 0.2
centers = []
for title, sub, col in phases:
    box = FancyBboxPatch((x, 1.0), bw, bh, boxstyle="round,pad=0.04,rounding_size=0.12",
                         linewidth=1.5, edgecolor=col, facecolor=col + "20")
    ax.add_patch(box)
    cx = x + bw / 2
    centers.append(cx)
    ax.text(cx, 1.0 + bh - 0.34, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=col)
    ax.text(cx, 1.0 + 0.5, sub, ha="center", va="center", fontsize=7.6, color="#222222")
    x += bw + gap

for i in range(n - 1):
    a = FancyArrowPatch((centers[i] + bw / 2, 1.0 + bh / 2),
                        (centers[i + 1] - bw / 2, 1.0 + bh / 2),
                        arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#555555")
    ax.add_patch(a)

ax.text((centers[0] + centers[-1]) / 2, 3.05,
        "Construcción inside-out: cada fase de-riskea la siguiente",
        ha="center", fontsize=11, style="italic", color="#444444")
ax.text((centers[0] + centers[-1]) / 2, 0.55,
        "Objetivo: RETRASAR la intratabilidad (maximizar TTP-combinado), no curar/reducir el tumor",
        ha="center", fontsize=9.5, color="#7F1D1D")

ax.set_xlim(0, x + 0.1); ax.set_ylim(0.2, 3.3); ax.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT}/pipeline_diagram.png", dpi=160, facecolor="white", bbox_inches="tight")
print("[ok] outputs/pipeline_diagram.png")
