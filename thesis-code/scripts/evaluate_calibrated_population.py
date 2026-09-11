"""Evalúa la población nominal calibrada de políticas ya entrenadas.

No reentrena agentes: carga los checkpoints ``mappo_real_20260908_s*`` e
``ippo_real_20260908_s*`` y aplica la métrica TTP-combinado del evaluador único.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats

from gbmarl.config import params_fingerprint, load_calibration
from gbmarl.evalutils import Gatenby, U_MAX, evaluate_episode
from gbmarl.mappo import Agent, obs_therapy
from gbmarl.outcomes import SPANISH_LABELS
from gbmarl.tumor_env import TumorEnv


def load_policy(env: TumorEnv, path: str, variant: str):
    """Carga un actor determinista con la dimensión de crítico correcta."""
    cdim = 3 if variant == "mappo" else 2
    agent = Agent(2, cdim, 1, env.action_space("therapy").low,
                  env.action_space("therapy").high)
    agent.load_state_dict(torch.load(path, weights_only=True))
    agent.eval()

    def policy(state):
        with torch.no_grad():
            action = agent.actor_mean(torch.tensor(obs_therapy(state)))
            action = action.clamp(agent.a_low, agent.a_high)
        return float(action[0])

    return policy


def evaluate_variant(env: TumorEnv, variant: str, seeds: int, model_dir: str):
    rows = []
    for seed in range(seeds):
        if variant == "mappo" and seed == 0:
            # La primera corrida histórica exportó MAPPO sin el sufijo _s0.
            filename = "mappo_real_20260908_therapy.pt"
        else:
            filename = f"{variant}_real_20260908_s{seed}_therapy.pt"
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        result = evaluate_episode(env, load_policy(env, path, variant))
        rows.append({
            "seed": seed,
            "ttp": result.ttp,
            "terminal_day": result.terminal_day,
            "mode": result.mode,
            "mode_es": SPANISH_LABELS[result.mode],
            "censored": result.censored,
            "successful": result.successful,
            "burden": result.burden,
            "frac_resistant": result.frac_resistant,
            "mean_dose": result.mean_dose,
            "t_load": result.t_load,
            "t_resistance": result.t_resistance,
        })
    return rows


def summary(rows, gatenby_ttp):
    ttps = np.asarray([row["ttp"] for row in rows], dtype=float)
    modes = {}
    for row in rows:
        modes[row["mode"]] = modes.get(row["mode"], 0) + 1
    return {
        "n": len(rows),
        "median": float(np.median(ttps)),
        "min": int(ttps.min()),
        "max": int(ttps.max()),
        "mean": float(ttps.mean()),
        "modes": modes,
        "success_vs_gatenby": int(np.sum(ttps > gatenby_ttp)),
        "dose_mean": float(np.mean([row["mean_dose"] for row in rows])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument("--model-dir", default="outputs/models")
    parser.add_argument("--calibration", default="data/processed/calibration.json")
    parser.add_argument("--output", default="outputs/validacion_calibrada_15semillas_2026-09-08")
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds debe ser >= 2")

    params = load_calibration(args.calibration, required=True)
    env = TumorEnv(params=params, horizon_days=180)
    gatenby = evaluate_episode(env, Gatenby())
    mappo = evaluate_variant(env, "mappo", args.seeds, args.model_dir)
    ippo = evaluate_variant(env, "ippo", args.seeds, args.model_dir)
    m = np.asarray([row["ttp"] for row in mappo], dtype=float)
    i = np.asarray([row["ttp"] for row in ippo], dtype=float)
    w = stats.wilcoxon(m, i, alternative="greater")

    payload = {
        "calibration": params_fingerprint(params),
        "horizon_days": env.horizon,
        "phi_fixed": 0.01,
        "steps_effective": 120832,
        "gatenby_ttp": gatenby.ttp,
        "gatenby_mode": gatenby.mode,
        "mappo": mappo,
        "ippo": ippo,
        "summary_mappo": summary(mappo, gatenby.ttp),
        "summary_ippo": summary(ippo, gatenby.ttp),
        "wilcoxon_mappo_gt_ippo": {"statistic": float(w.statistic), "pvalue": float(w.pvalue)},
    }
    output_base = args.output
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    with open(output_base + ".json", "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    def fmt(v):
        return f"{v['median']:.0f} [{v['min']},{v['max']}]"

    lines = [
        f"# Validación calibrada nominal — {args.seeds} semillas\n",
        f"Calibración `{payload['calibration']}` · horizonte {env.horizon} d · "
        f"φ fijo={payload['phi_fixed']} · pasos efectivos={payload['steps_effective']}\n",
        "TTP-combinado: primer fallo de carga o de tratabilidad; el modo de falla se conserva por separado.\n",
        "\n## Resumen\n",
        "| Método | TTP mediana [min,max] | Modos de falla | Éxito vs Gatenby |",
        "|---|---:|---|---:|",
        f"| MAPPO (CTDE) | {fmt(payload['summary_mappo'])} | {payload['summary_mappo']['modes']} | "
        f"{payload['summary_mappo']['success_vs_gatenby']}/{args.seeds} |",
        f"| IPPO (local) | {fmt(payload['summary_ippo'])} | {payload['summary_ippo']['modes']} | "
        f"{payload['summary_ippo']['success_vs_gatenby']}/{args.seeds} |",
        "",
        f"Gatenby determinista: {gatenby.ttp} d ({SPANISH_LABELS[gatenby.mode]}).",
        f"Wilcoxon pareado unilateral MAPPO > IPPO: W={w.statistic:.1f}, p={w.pvalue:.6g}.",
        "La prueba se interpreta junto con mediana, rango y modos de falla; no implica eficacia clínica.",
        "\n## Detalle por semilla\n",
        "| Semilla | MAPPO TTP | Modo MAPPO | IPPO TTP | Modo IPPO |",
        "|---:|---:|---|---:|---|",
    ]
    for mr, ir in zip(mappo, ippo):
        lines.append(f"| {mr['seed']} | {mr['ttp']} | {SPANISH_LABELS[mr['mode']]} | "
                     f"{ir['ttp']} | {SPANISH_LABELS[ir['mode']]} |")
    with open(output_base + ".md", "w") as handle:
        handle.write("\n".join(lines) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "gbmarl-mpl"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.boxplot([m, i], tick_labels=["MAPPO (CTDE)", "IPPO (local)"], showmeans=True)
    ax.axhline(gatenby.ttp, color="#E67E22", linestyle="--", label=f"Gatenby ({gatenby.ttp} d)")
    ax.set_ylabel("TTP-combinado (días)")
    ax.set_title(f"Validación nominal calibrada (n={args.seeds})")
    ax.grid(alpha=0.25, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(output_base + ".png", dpi=150, facecolor="white")
    plt.close(fig)

    print(f"MAPPO: {fmt(payload['summary_mappo'])} · modos={payload['summary_mappo']['modes']}")
    print(f"IPPO:  {fmt(payload['summary_ippo'])} · modos={payload['summary_ippo']['modes']}")
    print(f"Gatenby: {gatenby.ttp} d · Wilcoxon MAPPO>IPPO p={w.pvalue:.6g}")
    print(f"Guardado: {output_base}.{{json,md,png}}")


if __name__ == "__main__":
    main()
