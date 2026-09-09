"""EXP-3: extensión de toxicidad/estado funcional sin tercer agente.

Requiere un JSON externo con fuente y parámetros pre-especificados. El script no
incluye coeficientes clínicos por defecto para evitar ajustar la toxicidad hasta
obtener un ranking favorable. Incluye una ablación contra el entrenamiento 3-D
anterior y evalúa ambas condiciones en el mismo entorno de paciente 4-D.

Ejemplo de estructura (los números deben justificarse externamente):
  {
    "source": "DOI, protocolo o nota de calibración",
    "parameters": {
      "accumulation_rate": ...,
      "recovery_rate": ...,
      "half_saturation": ...,
      "failure_threshold": ...,
      "initial_toxicity": 0.0,
      "toxicity_weight_in_health": ...,
      "burden_weight_in_health": ...
    }
  }
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gbmarl.config import ToxicityParams, load_calibration, params_fingerprint
from gbmarl.evalutils import FIXED_PHI, Gatenby, U_MAX
from gbmarl.mappo import Agent, global_state, obs_therapy, obs_tumor, train_mappo
from gbmarl.outcomes import is_failure, scalar_ttp
from gbmarl.patient_env import PatientTumorEnv
from gbmarl.tumor_env import TumorEnv


EXPERIMENT_VERSION = 1
CONDITIONS = ("health_aware", "health_blind")


def load_toxicity_config(path):
    with open(path) as handle:
        raw = json.load(handle)
    source = str(raw.get("source", "")).strip()
    if not source:
        raise ValueError("el JSON debe documentar `source`")
    params = ToxicityParams(**raw["parameters"])
    fingerprint = hashlib.sha256(
        json.dumps(raw, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return raw, params, fingerprint


def make_agent_pair(env, centralized):
    obs, _ = env.reset(seed=0)
    state = obs["therapy"]
    global_dim = len(global_state(state))
    th_local, tu_local = len(obs_therapy(state)), len(obs_tumor(state))
    th_critic = global_dim if centralized else th_local
    tu_critic = global_dim if centralized else tu_local
    th = Agent(th_local, th_critic, 1, env.action_space("therapy").low,
               env.action_space("therapy").high)
    tu = Agent(tu_local, tu_critic, 1, env.action_space("tumor").low,
               env.action_space("tumor").high)
    return th, tu


def train_or_load(env, condition, variant, seed, steps, model_dir):
    centralized = variant == "mappo"
    prefix = os.path.join(model_dir, f"{condition}_{variant}_{seed}")
    p_th, p_tu = f"{prefix}_therapy.pt", f"{prefix}_tumor.pt"
    if os.path.exists(p_th) and os.path.exists(p_tu):
        th, tu = make_agent_pair(env, centralized)
        th.load_state_dict(torch.load(p_th)); tu.load_state_dict(torch.load(p_tu))
        th.eval(); tu.eval()
        return th, tu
    th, tu, _ = train_mappo(env, total_timesteps=steps, seed=seed,
                            centralized=centralized, verbose=False)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(th.state_dict(), p_th); torch.save(tu.state_dict(), p_tu)
    th.eval(); tu.eval()
    return th, tu


@torch.no_grad()
def evaluate(env, therapy_fn, tumor_fn):
    if hasattr(therapy_fn, "reset"):
        therapy_fn.reset()
    obs, _ = env.reset(seed=0); state = obs["therapy"]
    doses = []
    while True:
        u = float(therapy_fn(state))
        phi = float(tumor_fn(state))
        obs, _, terms, truncs, infos = env.step({
            "therapy": np.array([u], np.float32),
            "tumor": np.array([phi], np.float32),
        })
        doses.append(u); state = obs["therapy"]
        if terms.get("therapy", False) or truncs.get("therapy", False):
            info = infos["therapy"]
            mode = info["failure_mode"]
            return {
                "ttp": scalar_ttp(mode, info["day"], env.horizon),
                "terminal_day": info["day"], "mode": mode,
                "failure_events": info["failure_events"],
                "toxicity_final": info["toxicity"], "health_final": info["health"],
                "quality_adjusted_days": info["quality_adjusted_days"],
                "mean_health": info["quality_adjusted_days"] / info["day"],
                "mean_dose": float(np.mean(doses)),
                "is_failure": is_failure(mode),
            }


def actor_policy(agent, extractor):
    def policy(state):
        action = agent.actor_mean(torch.tensor(extractor(state))).clamp(
            agent.a_low, agent.a_high)
        return float(action[0])
    return policy


def obs_therapy_without_health(state):
    """Observación histórica [carga,c], aun al evaluar en el entorno 4-D."""
    return obs_therapy(np.asarray(state)[:3])


def bootstrap_paired_median_ci(values, reps=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260908)
    samples = rng.choice(values, size=(reps, len(values)), replace=True)
    lo, hi = np.percentile(np.median(samples, axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def run(args):
    raw, toxicity, fingerprint = load_toxicity_config(args.toxicity_config)
    calibration = load_calibration()
    calibration_signature = params_fingerprint(calibration)
    model_dir = (f"outputs/models/exp3_health_{calibration_signature}_"
                 f"{fingerprint}_t{args.steps}")
    state_path = f"outputs/exp3_patient_health_{calibration_signature}_{fingerprint}.json"
    report_path = f"outputs/exp3_patient_health_{fingerprint}.md"
    requested = {"version": EXPERIMENT_VERSION, "calibration": calibration_signature,
                 "fingerprint": fingerprint,
                 "toxicity": raw, "seeds": args.seeds, "steps": args.steps,
                 "conditions": list(CONDITIONS)}
    state = {
        "config": requested,
        "baselines_fixed_tumor": {}, "cells": {},
    }
    if os.path.exists(state_path):
        with open(state_path) as handle:
            state = json.load(handle)
        if state.get("config") != requested:
            raise ValueError(
                f"{state_path} pertenece a otra configuración: "
                f"{state.get('config')}"
            )

    env = PatientTumorEnv(toxicity, params=calibration, horizon_days=180)
    # Control histórico: estado 3-D y castigo lineal de dosis. Ambos controles se
    # evalúan después en ``env`` para observar la misma toxicidad acumulada real.
    training_envs = {
        "health_aware": env,
        "health_blind": TumorEnv(params=env.p, horizon_days=180),
    }
    baselines = {
        "untreated": lambda _: 0.0,
        "mtd": lambda _: U_MAX,
        "gatenby": Gatenby(),
    }
    for name, therapy_fn in baselines.items():
        state["baselines_fixed_tumor"][name] = evaluate(
            env, therapy_fn, lambda _: FIXED_PHI)

    for condition, training_env in training_envs.items():
        for variant in ("mappo", "ippo"):
            for seed in range(args.seeds):
                key = f"{condition}@{variant}@s{seed}"
                if key in state["cells"]:
                    continue
                th, tu = train_or_load(
                    training_env, condition, variant, seed, args.steps, model_dir)
                extractor = (obs_therapy if condition == "health_aware"
                             else obs_therapy_without_health)
                therapy = actor_policy(th, extractor)
                tumor = actor_policy(tu, obs_tumor)
                state["cells"][key] = {
                    "condition": condition, "variant": variant, "seed": seed,
                    "fixed_tumor": evaluate(env, therapy, lambda _: FIXED_PHI),
                    "selfplay_tumor": evaluate(env, therapy, tumor),
                }
                os.makedirs("outputs", exist_ok=True)
                with open(state_path, "w") as handle:
                    json.dump(state, handle, indent=2, ensure_ascii=False)

    lines = [
        "# EXP-3 — Toxicidad y estado funcional (resultados)", "",
        f"Config `{fingerprint}` · fuente: {raw['source']} · "
        f"n={args.seeds} · pasos={args.steps}", "",
        "Los parámetros son externos al script; este reporte no implica validación clínica.", "",
        "Ablación: `health_aware` usa estado 4-D y falla tóxica durante entrenamiento; "
        "`health_blind` reproduce el diseño 3-D anterior. Ambos se evalúan en el "
        "mismo entorno 4-D.", "",
        "| condición/estrategia | adversario | TTP | salud media | días ajustados | "
        "toxicidad final | dosis media | modos |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, result in state["baselines_fixed_tumor"].items():
        lines.append(f"| {name} | phi fijo | {result['ttp']} | "
                     f"{result['mean_health']:.3f} | "
                     f"{result['quality_adjusted_days']:.1f} | "
                     f"{result['toxicity_final']:.3f} | "
                     f"{result['mean_dose']:.3f} | {result['mode']} |")
    for condition in CONDITIONS:
        for variant in ("mappo", "ippo"):
            rows = [cell for cell in state["cells"].values()
                    if cell["condition"] == condition and
                    cell["variant"] == variant]
            for evaluation_name, label in (("fixed_tumor", "phi fijo"),
                                           ("selfplay_tumor", "self-play")):
                results = [row[evaluation_name] for row in rows]
                modes = dict(Counter(x["mode"] for x in results))
                lines.append(f"| {condition}/{variant.upper()} | {label} | "
                             f"{np.median([x['ttp'] for x in results]):.0f} | "
                             f"{np.median([x['mean_health'] for x in results]):.3f} | "
                             f"{np.median([x['quality_adjusted_days'] for x in results]):.1f} | "
                             f"{np.median([x['toxicity_final'] for x in results]):.3f} | "
                             f"{np.median([x['mean_dose'] for x in results]):.3f} | "
                             f"{modes} |")

    lines += ["", "## Ablación pareada bajo phi fijo", "",
              "Diferencias `health_aware − health_blind` por la misma semilla. "
              "Los intervalos bootstrap usan la semilla/política como unidad independiente.",
              "", "| variante | ΔTTP mediana [IC95%] | Δdías ajustados [IC95%] | "
              "Δtoxicidad final [IC95%] |",
              "|---|---:|---:|---:|"]
    for variant in ("mappo", "ippo"):
        aware = {cell["seed"]: cell["fixed_tumor"]
                 for cell in state["cells"].values()
                 if cell["condition"] == "health_aware" and
                 cell["variant"] == variant}
        blind = {cell["seed"]: cell["fixed_tumor"]
                 for cell in state["cells"].values()
                 if cell["condition"] == "health_blind" and
                 cell["variant"] == variant}
        paired_seeds = sorted(set(aware) & set(blind))
        deltas = {
            "ttp": np.array([aware[s]["ttp"] - blind[s]["ttp"]
                             for s in paired_seeds], float),
            "qad": np.array([aware[s]["quality_adjusted_days"] -
                             blind[s]["quality_adjusted_days"]
                             for s in paired_seeds], float),
            "tox": np.array([aware[s]["toxicity_final"] -
                             blind[s]["toxicity_final"]
                             for s in paired_seeds], float),
        }
        rendered = []
        for values in deltas.values():
            lo, hi = bootstrap_paired_median_ci(values)
            rendered.append(f"{np.median(values):+.2f} [{lo:+.2f},{hi:+.2f}]")
        lines.append(f"| {variant.upper()} | {rendered[0]} | {rendered[1]} | "
                     f"{rendered[2]} |")
    with open(report_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Estado -> {state_path}")
    print(f"Reporte -> {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toxicity-config", required=True)
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds, args.steps = 2, 3000
    if args.seeds < 1 or args.steps < 1:
        parser.error("--seeds y --steps deben ser positivos")
    run(args)


if __name__ == "__main__":
    main()
