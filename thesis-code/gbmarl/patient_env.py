"""Extensión opt-in del entorno con toxicidad acumulada y utilidad de salud.

La salud es estado pasivo, no un tercer agente. Los parámetros de toxicidad son
obligatorios y deben documentarse externamente antes de entrenar políticas.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .config import Params, ToxicityParams
from .dynamics import rk4_step_with_toxicity
from .outcomes import (classify_outcome, is_censored, is_failure, is_success,
                       resistant_fraction)
from .tumor_env import TumorEnv


class PatientTumorEnv(TumorEnv):
    """TumorEnv con estado global [S,R,c,A] y falla explícita por toxicidad."""

    metadata = {"name": "tumor_patient_v1"}

    def __init__(self, toxicity: ToxicityParams, params: Params | None = None,
                 health_control_bonus: float = 1.0, **kwargs):
        # A ya modela toxicidad: no se duplica con el castigo lineal -tox_weight*u.
        if float(kwargs.get("tox_weight", 0.0)) != 0.0:
            raise ValueError("PatientTumorEnv requiere tox_weight=0 para no duplicar toxicidad")
        kwargs["tox_weight"] = 0.0
        super().__init__(params=params, **kwargs)
        self.toxicity = toxicity
        self.health_control_bonus = float(health_control_bonus)
        if self.health_control_bonus <= 0:
            raise ValueError("health_control_bonus debe ser positivo")
        self._obs_space = spaces.Box(
            low=0.0,
            high=np.array([2.0, 2.0, 10.0, 1.0], np.float32),
            shape=(4,), dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        _, infos = super().reset(seed=seed, options=options)
        self.state = np.append(self.state, self.toxicity.initial_toxicity)
        self.t_toxicity = None
        self.quality_adjusted_days = 0.0
        return self._obs(), infos

    def _health_utility(self, burden: float, toxicity: float) -> float:
        tp = self.toxicity
        value = (1.0 - tp.toxicity_weight_in_health * toxicity
                 - tp.burden_weight_in_health * burden / self.p.K)
        return float(np.clip(value, 0.0, 1.0))

    def step(self, actions):
        u = float(np.asarray(actions["therapy"]).reshape(-1)[0])
        phi = float(np.asarray(actions["tumor"]).reshape(-1)[0])

        for _ in range(self.steps_per_day):
            self.state = rk4_step_with_toxicity(
                self.state, u, phi, self.dt, self.p, self.toxicity)
        self.day += 1

        S, R, c, A = self.state
        burden = S + R
        fracR = resistant_fraction(S, R)
        progressed = burden >= self.prog_thr
        untreatable = fracR >= self.r_majority
        extinct = burden < self.eps
        toxic = A >= self.toxicity.failure_threshold

        if progressed and self.t_load is None:
            self.t_load = self.day
        if untreatable and self.t_resistance is None:
            self.t_resistance = self.day
        if toxic and self.t_toxicity is None:
            self.t_toxicity = self.day

        outcome = classify_outcome(
            progressed=progressed,
            untreatable=untreatable,
            extinct=extinct,
            toxicity=toxic,
            reached_horizon=self.day >= self.horizon,
        )
        failure = is_failure(outcome)
        self.last_outcome = outcome
        manageable = not failure and not extinct
        health = self._health_utility(burden, A)
        self.quality_adjusted_days += health

        # Objetivo primario: días manejables. La seguridad entra como estado y
        # restricción terminal, no mediante un peso lineal de dosis oculto.
        r_therapy = self.health_control_bonus if manageable else 0.0
        tumor_failure = progressed or untreatable
        r_tumor = fracR + (self.progression_bonus if tumor_failure else 0.0)
        if extinct:
            r_therapy += self.win_bonus
            r_tumor -= self.win_bonus

        rewards = {"therapy": float(r_therapy), "tumor": float(r_tumor)}
        terminated = bool(extinct or failure)
        truncated = bool(is_censored(outcome))
        terms = {agent: terminated for agent in self.agents}
        truncs = {agent: truncated for agent in self.agents}
        obs = self._obs()
        failure_events = [name for name, active in (
            ("load", progressed), ("resistance", untreatable),
            ("toxicity", toxic)) if active]
        infos = {agent: {
            "burden": float(burden), "R": float(R), "fracR": float(fracR),
            "concentration": float(c), "toxicity": float(A), "health": health,
            "quality_adjusted_days": float(self.quality_adjusted_days),
            "day": self.day, "progressed": bool(progressed),
            "untreatable": bool(untreatable), "toxic": bool(toxic),
            "failure_mode": outcome, "failure_events": failure_events,
            "is_failure": failure, "successful": is_success(outcome),
            "censored": is_censored(outcome), "t_load": self.t_load,
            "t_resistance": self.t_resistance, "t_toxicity": self.t_toxicity,
        } for agent in self.agents}

        if terminated or truncated:
            self.agents = []
        return obs, rewards, terms, truncs, infos
