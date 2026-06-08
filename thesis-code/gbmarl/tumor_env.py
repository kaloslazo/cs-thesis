"""
tumor_env.py — Entorno PettingZoo de 2 agentes sobre la dinámica calibrada.

Dos agentes con objetivos OPUESTOS (juego adversarial):
  · "therapy": elige la dosis diaria u  -> quiere MINIMIZAR el tumor.
  · "tumor":   elige la transición φ    -> quiere MAXIMIZAR su supervivencia.

Cada step = 1 día de tratamiento (integra la ODE con RK4 internamente).
La dinámica vive en dynamics.py; aquí solo la envolvemos para RL.
"""
from __future__ import annotations
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .config import Params, load_calibration
from .dynamics import rk4_step


class TumorEnv(ParallelEnv):
    metadata = {"name": "tumor_v0"}

    def __init__(self, params: Params | None = None, horizon_days: int = 180,
                 dt: float = 0.1, u_max: float = 1.0,
                 phi_min: float = 0.0, phi_max: float = 0.05,
                 tox_weight: float = 0.10, S0: float = 0.40, R0: float = 0.01):
        self.p = params or load_calibration()
        self.horizon = horizon_days
        self.dt = dt
        self.steps_per_day = int(round(1.0 / dt))
        self.tox_weight = tox_weight
        self.S0, self.R0 = S0, R0

        self.possible_agents = ["therapy", "tumor"]
        # Observación común: [S, R, c]  (acotada para estabilidad numérica)
        self._obs_space = spaces.Box(low=0.0, high=np.array([2.0, 2.0, 10.0], np.float32),
                                     shape=(3,), dtype=np.float32)
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
        self.state = np.array([self.S0, self.R0, 0.0], dtype=float)
        self.day = 0
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions):
        u = float(np.asarray(actions["therapy"]).reshape(-1)[0])
        phi = float(np.asarray(actions["tumor"]).reshape(-1)[0])

        # Integrar 1 día con u y φ constantes
        for _ in range(self.steps_per_day):
            self.state = rk4_step(self.state, u, phi, self.dt, self.p)
        self.day += 1

        S, R, c = self.state
        burden = S + R

        # Recompensas opuestas
        r_therapy = -burden - self.tox_weight * u      # menos tumor, menos toxicidad
        r_tumor = burden                                # sobrevivir
        rewards = {"therapy": float(r_therapy), "tumor": float(r_tumor)}

        extinto = burden < 1e-3                         # tumor erradicado
        progresion = burden > 0.99 * self.p.K           # tumor llenó la capacidad
        terminated = bool(extinto or progresion)
        truncated = bool(self.day >= self.horizon)

        terms = {a: terminated for a in self.agents}
        truncs = {a: truncated for a in self.agents}
        obs = self._obs()
        infos = {a: {"burden": float(burden), "day": self.day} for a in self.agents}

        if terminated or truncated:
            self.agents = []                            # episodio terminado
        return obs, rewards, terms, truncs, infos