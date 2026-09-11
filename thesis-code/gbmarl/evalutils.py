"""
evalutils.py — Métrica de evaluación CORRECTA y baselines, en un solo lugar.

TTP-combinado: días hasta que el tumor deja de estar controlado (carga<umbral)
Y tratable (fracR<mayoría) — lo que falle primero. Es un endpoint operacional
del simulador; su equivalencia con un desenlace clínico no está validada.
NO confundir con la métrica vieja (solo-resistencia), que reportaba el horizonte
cuando la falla era por carga, inflando falsamente los resultados.
"""
import numpy as np
from dataclasses import dataclass

from .outcomes import (SPANISH_LABELS, classify_outcome, is_censored,
                       is_success, scalar_ttp)

FIXED_PHI = 0.01
U_MAX = 1.0


@dataclass(frozen=True)
class EvaluationResult:
    """Resultado estructurado de un episodio de evaluación."""

    ttp: int
    terminal_day: int
    mode: str
    censored: bool
    successful: bool
    burden: float
    frac_resistant: float
    mean_dose: float
    t_load: int | None = None
    t_resistance: int | None = None


class Gatenby:
    """Terapia adaptativa heurística: dosifica si la carga sube, descansa si baja."""
    def __init__(self, on=0.5, off=0.25):
        self.on, self.off, self.d = on, off, True

    def reset(self):
        self.d = True

    def __call__(self, s):
        b = s[0] + s[1]
        if self.d and b < self.off:
            self.d = False
        elif not self.d and b > self.on:
            self.d = True
        return U_MAX if self.d else 0.0


def evaluate_episode(env, therapy_fn, tumor_fn=lambda s: FIXED_PHI) -> EvaluationResult:
    """Evalúa un episodio y conserva evento, censura y tiempos componentes."""
    if hasattr(therapy_fn, "reset"):
        therapy_fn.reset()
    obs, _ = env.reset(seed=0)
    state = obs["therapy"]
    info = {}
    doses = []
    term = trunc = False
    while True:
        u = float(therapy_fn(state))
        phi = float(tumor_fn(state))
        obs, rew, terms, truncs, infos = env.step(
            {"therapy": np.array([u], np.float32),
             "tumor": np.array([phi], np.float32)})
        doses.append(u)
        state = obs["therapy"]
        info = infos["therapy"]
        term = terms.get("therapy", False)
        trunc = truncs.get("therapy", False)
        if term or trunc:
            break

    mode = info.get("failure_mode")
    if mode is None:  # Compatibilidad con entornos/artefactos anteriores.
        mode = classify_outcome(
            progressed=bool(info.get("progressed")),
            untreatable=bool(info.get("untreatable")),
            extinct=bool(term and not info.get("progressed") and
                         not info.get("untreatable")),
            reached_horizon=bool(trunc and not term),
        )
    terminal_day = int(info["day"])
    return EvaluationResult(
        ttp=scalar_ttp(mode, terminal_day, env.horizon),
        terminal_day=terminal_day,
        mode=mode,
        censored=is_censored(mode),
        successful=is_success(mode),
        burden=float(info["burden"]),
        frac_resistant=float(info["fracR"]),
        mean_dose=float(np.mean(doses)),
        t_load=info.get("t_load"),
        t_resistance=info.get("t_resistance"),
    )


def ttp_combinado(env, therapy_fn, tumor_fn=lambda s: FIXED_PHI):
    """TTP escalar compatible: primer fracaso; éxito/extinción = horizonte."""
    return evaluate_episode(env, therapy_fn, tumor_fn).ttp


def ttp_combinado_detalle(env, therapy_fn, tumor_fn=lambda s: FIXED_PHI):
    """Como ttp_combinado pero devuelve (días, motivo, carga, fracR, dosis_media)."""
    result = evaluate_episode(env, therapy_fn, tumor_fn)
    return (result.ttp, SPANISH_LABELS[result.mode], result.burden,
            result.frac_resistant, result.mean_dose)
