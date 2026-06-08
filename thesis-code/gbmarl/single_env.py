"""
single_env.py — Adaptador single-agent: SOLO entrena la terapia.

Envuelve el TumorEnv (2 agentes) en un entorno Gymnasium estándar donde el
tumor usa una política FIJA (φ constante). Así podemos entrenar la terapia con
PPO clásico antes de meter el segundo agente que aprende (MAPPO).
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .tumor_env import TumorEnv


class TherapyEnv(gym.Env):
    """Terapia vs tumor con transición fija (de-risk antes de MAPPO)."""

    def __init__(self, fixed_phi: float = 0.01, **env_kwargs):
        self.env = TumorEnv(**env_kwargs)
        self.fixed_phi = fixed_phi
        self.observation_space = self.env.observation_space("therapy")
        self.action_space = self.env.action_space("therapy")

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed)
        return obs["therapy"], infos["therapy"]

    def step(self, action):
        actions = {"therapy": np.asarray(action, np.float32).reshape(-1),
                   "tumor": np.array([self.fixed_phi], np.float32)}
        obs, rew, term, trunc, info = self.env.step(actions)
        done_term = term.get("therapy", False) if term else True
        done_trunc = trunc.get("therapy", False) if trunc else True
        return (obs["therapy"], rew["therapy"], done_term, done_trunc,
                info.get("therapy", {}))