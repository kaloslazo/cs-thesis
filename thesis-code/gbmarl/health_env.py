"""
health_env.py — Entorno PettingZoo 2-agentes con SALUD del paciente (Fase 1).

Igual que tumor_env.py pero con estado 4-D [S,R,c,H] y un TERCER modo de falla:
el paciente muere si H < H_min (por toxicidad acumulada o por carga tumoral).

Cambios frente a TumorEnv (todos aditivos, en clase nueva):
  · Estado y crítico ven H (obs de terapia incluye H: el médico mide al paciente).
  · `alive` exige además H >= H_min.  Modos de falla: carga | resistencia | SALUD.
  · La toxicidad plana (-tox_weight*u) se conserva como señal de gradiente débil,
    pero el costo real de sobredosificar ahora es MECANÍSTICO: baja H -> muerte.

No toca tumor_env.py: la línea base 3-D sigue disponible para la ablación con/sin H.
"""
from __future__ import annotations
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .config import Params, load_calibration
from .dynamics_health import rk4_step_h


class TumorEnvHealth(ParallelEnv):
    metadata = {"name": "tumor_health_v1"}

    def __init__(self, params: Params | None = None, horizon_days: int = 180,
                 dt: float = 0.1, u_max: float = 1.0,
                 phi_min: float = 0.0, phi_max: float = 0.05,
                 tox_weight: float = 0.05, r_majority: float = 0.50,
                 control_bonus: float = 1.0, progression_bonus: float = 10.0,
                 win_bonus: float = 50.0, progression_threshold: float = 0.80,
                 eps: float = 1e-3, S0: float = 0.40, R0: float = 0.01):
        self.p = params or load_calibration()
        self.horizon = horizon_days
        self.dt = dt
        self.steps_per_day = int(round(1.0 / dt))
        self.tox_weight = tox_weight
        self.r_majority = r_majority
        self.control_bonus = control_bonus
        self.progression_bonus = progression_bonus
        self.win_bonus = win_bonus
        self.prog_thr = progression_threshold * self.p.K
        self.eps = eps
        self.S0, self.R0 = S0, R0

        self.possible_agents = ["therapy", "tumor"]
        # obs global 4-D: [S, R, c, H]
        self._obs_space = spaces.Box(
            low=0.0, high=np.array([2.0, 2.0, 10.0, 1.0], np.float32),
            shape=(4,), dtype=np.float32)
        self._act_space = {
            "therapy": spaces.Box(low=0.0, high=u_max, shape=(1,), dtype=np.float32),
            "tumor":   spaces.Box(low=phi_min, high=phi_max, shape=(1,), dtype=np.float32),
        }

    def observation_space(self, agent): return self._obs_space
    def action_space(self, agent): return self._act_space[agent]

    def _obs(self):
        return {a: self.state.astype(np.float32) for a in self.agents}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = list(self.possible_agents)
        self.state = np.array([self.S0, self.R0, 0.0, self.p.H0], dtype=float)
        self.day = 0
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions):
        u = float(np.asarray(actions["therapy"]).reshape(-1)[0])
        phi = float(np.asarray(actions["tumor"]).reshape(-1)[0])

        for _ in range(self.steps_per_day):
            self.state = rk4_step_h(self.state, u, phi, self.dt, self.p)
        self.day += 1

        S, R, c, H = self.state
        burden = S + R
        fracR = R / (burden + 1e-9)
        controlled = burden < self.prog_thr
        treatable = fracR < self.r_majority
        healthy = H >= self.p.H_min                     # NUEVO: paciente vivo
        alive = controlled and treatable and healthy    # las tres condiciones
        extinct = burden < self.eps
        failure = not alive

        r_therapy = (self.control_bonus if alive else 0.0) - self.tox_weight * u
        r_tumor = fracR + (self.progression_bonus if failure else 0.0)

        if extinct:
            r_therapy += self.win_bonus
            r_tumor -= self.win_bonus

        rewards = {"therapy": float(r_therapy), "tumor": float(r_tumor)}
        terminated = bool(extinct or failure)
        truncated = bool(self.day >= self.horizon)
        terms = {a: terminated for a in self.agents}
        truncs = {a: truncated for a in self.agents}
        obs = self._obs()
        infos = {a: {"burden": float(burden), "R": float(R), "fracR": float(fracR),
                     "H": float(H), "day": self.day,
                     "progressed": bool(not controlled),
                     "untreatable": bool(not treatable),
                     "unhealthy": bool(not healthy)} for a in self.agents}

        if terminated or truncated:
            self.agents = []
        return obs, rewards, terms, truncs, infos
