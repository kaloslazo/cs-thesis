"""
evaluate.py — M5: compara estrategias en DOS métricas, contra el mismo tumor.
Ejecutar desde la raíz:  python scripts/evaluate.py

  · TTP-carga      : días hasta que la carga total (S+R) cruza el umbral.
  · TTP-resistencia: diagnóstico del primer día en que las resistentes alcanzan
                     la mayoría, según el estado registrado por TumorEnv.
                     El TTP principal es el combinado, que también considera carga.
  · Frac. R final  : qué tan resistente quedó el tumor.

Estrategias: sin tratamiento, MTD, adaptativa (Gatenby), y MAPPO (si hay modelo).
"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
# Algunos entornos de ejecución no permiten escribir en la caché global de
# Matplotlib. Usar una caché temporal evita ruido y mantiene el script portable.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "gbmarl-mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib.pyplot as plt
from gbmarl.tumor_env import TumorEnv

FIXED_PHI = 0.01
U_MAX = 1.0
R_MAJORITY = 0.5          # umbral de "tumor intratable" (resistentes mayoría)


class AdaptiveTherapy:
    def __init__(self, on=0.5, off=0.25, u=U_MAX):
        self.on, self.off, self.u, self.dosing = on, off, u, True
    def reset(self): self.dosing = True
    def __call__(self, state):
        b = state[0] + state[1]
        if self.dosing and b < self.off: self.dosing = False
        elif not self.dosing and b > self.on: self.dosing = True
        return self.u if self.dosing else 0.0


def rollout(env, therapy_fn, tumor_fn=lambda s: FIXED_PHI):
    obs, _ = env.reset(seed=0); state = obs["therapy"]
    ttp_load = env.horizon; ttp_res = env.horizon
    traj = [state.copy()]; done = False
    while not done:
        u = float(therapy_fn(state)); phi = float(tumor_fn(state))
        obs, rew, term, trunc, info = env.step({"therapy": np.array([u], np.float32),
                                                "tumor": np.array([phi], np.float32)})
        state = obs["therapy"]; traj.append(state.copy())
        event_info = info["therapy"]
        # Reutilizar los eventos calculados por TumorEnv evita que este script
        # tenga una convención distinta en el borde fracR=0.50 o en eventos
        # simultáneos. El TTP combinado vive en gbmarl.evalutils.
        if event_info.get("t_load") is not None:
            ttp_load = int(event_info["t_load"])
        if event_info.get("t_resistance") is not None:
            ttp_res = int(event_info["t_resistance"])
        done = (term.get("therapy", True) if term else True) or \
               (trunc.get("therapy", True) if trunc else True)
    traj = np.array(traj)
    final_burden = traj[-1, 0] + traj[-1, 1]
    fracR_final = traj[-1, 1] / final_burden if final_burden > 0 else 0.0
    return ttp_load, ttp_res, fracR_final, traj


def load_mappo(env, model_path="outputs/models/mappo_therapy.pt"):
    try:
        import torch
        from gbmarl.mappo import Agent, obs_therapy
        th = Agent(2, 3, 1, env.action_space("therapy").low, env.action_space("therapy").high)
        th.load_state_dict(torch.load(model_path, weights_only=True)); th.eval()
        def policy(state):
            with torch.no_grad():
                a = th.actor_mean(torch.tensor(obs_therapy(state))).clamp(th.a_low, th.a_high)
            return float(a[0])
        return policy
    except Exception as e:
        print(f"  (MAPPO no disponible: {e})"); return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/models/mappo_therapy.pt",
                    help="checkpoint de la política MAPPO; se omite para evaluar solo baselines")
    ap.add_argument("--horizon", type=int, default=180,
                    help="horizonte del episodio en días simulados (por defecto: 180)")
    ap.add_argument("--output", default="outputs/evaluation_ttp.png",
                    help="ruta de la figura de salida")
    args = ap.parse_args()
    env = TumorEnv(horizon_days=args.horizon)
    estrategias = {"Sin tratamiento": lambda s: 0.0,
                   "MTD (dosis máx)": lambda s: U_MAX,
                   "Adaptativa (Gatenby)": AdaptiveTherapy()}
    mappo = load_mappo(env, args.model)
    if mappo is not None:
        estrategias["MAPPO (aprendida)"] = mappo

    colores = {"Sin tratamiento": "#7F8C8D", "MTD (dosis máx)": "#C0392B",
               "Adaptativa (Gatenby)": "#E67E22", "MAPPO (aprendida)": "#2E75B6"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))

    print("=" * 78)
    print(f"{'Estrategia':<24}{'TTP-carga':>12}{'TTP-resist.':>14}{'Frac. R final':>18}")
    print("=" * 78)
    for nombre, fn in estrategias.items():
        if hasattr(fn, "reset"): fn.reset()
        ttp_l, ttp_r, fR, traj = rollout(env, fn)
        carga = traj[:, 0] + traj[:, 1]
        burden = traj[:, 0] + traj[:, 1]
        fracR = np.divide(traj[:, 1], burden, out=np.zeros_like(burden),
                          where=burden > 0)
        c = colores.get(nombre, "#333")
        axL.plot(carga, label=f"{nombre}", lw=2, color=c)
        axR.plot(fracR, label=f"{nombre} (R fin={fR:.2f})", lw=2, color=c)
        print(f"{nombre:<24}{ttp_l:>12}{ttp_r:>14}{fR:>18.3f}")
    print("=" * 78)

    axL.axhline(env.prog_thr, color="k", ls="--", lw=1, label="Umbral carga")
    axL.set_title("Carga tumoral (S+R)"); axL.set_xlabel("Días"); axL.set_ylabel("Carga")
    axL.legend(fontsize=8); axL.grid(alpha=0.3)
    axR.axhline(env.r_majority, color="k", ls="--", lw=1, label="Mayoría resistente")
    axR.set_title("Fracción resistente R/(S+R) — la métrica clínica")
    axR.set_xlabel("Días"); axR.set_ylabel("Fracción R"); axR.set_ylim(0, 1.05)
    axR.legend(fontsize=8); axR.grid(alpha=0.3)
    fig.suptitle("M5: MTD controla carga pero crea resistencia; MAPPO la contiene")
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor="white")
    print(f"Figura guardada en {args.output}")


if __name__ == "__main__":
    main()
