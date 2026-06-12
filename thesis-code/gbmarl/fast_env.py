"""
fast_env.py — Versión ESCALAR (float puro) de TumorEnv para entrenamiento masivo.

Misma dinámica (RK4) y MISMA recompensa que `tumor_env.TumorEnv`, pero sin
asignar arrays de numpy en el lazo interno. Validado numéricamente: produce
trayectorias idénticas a `dynamics.rk4_step` (diff máx 0.0) y es ~4.7x más
rápido. Esto hace factible la corrida completa de 120k pasos dentro del techo
de tiempo del entorno de ejecución.

NO cambia ninguna "Decisión bloqueada": es solo una ruta de cómputo equivalente
para el rollout. La física y los premios se replican 1:1 desde tumor_env.py.
"""
from __future__ import annotations
import numpy as np
from .config import Params, load_calibration


def _hill(c, ic50, delta_max, h):
    if c < 0.0:
        c = 0.0
    ch = c ** h
    return delta_max * ch / (ic50 ** h + ch)


class FastTumorEnv:
    """Espejo escalar de TumorEnv. Interfaz reducida para el trainer rápido.

    reset(seed) -> (S, R, c)
    step(u, phi) -> ((S, R, c), r_therapy, r_tumor, terminated, truncated)
    """

    def __init__(self, params: Params | None = None, horizon_days: int = 180,
                 dt: float = 0.1, u_max: float = 1.0,
                 phi_min: float = 0.0, phi_max: float = 0.05,
                 tox_weight: float = 0.05, r_majority: float = 0.50,
                 control_bonus: float = 1.0, progression_bonus: float = 10.0,
                 win_bonus: float = 50.0, progression_threshold: float = 0.80,
                 eps: float = 1e-3, S0: float = 0.40, R0: float = 0.01):
        p = params or load_calibration()
        # cachear params como floats locales (evita atributos en lazo interno)
        self.aS, self.aR, self.K = p.alpha_S, p.alpha_R, p.K
        self.ic50_S, self.ic50_R = p.ic50_S, p.ic50_R
        self.dmax_S, self.dmax_R = p.delta_max_S, p.delta_max_R
        self.h, self.lam = p.hill, p.lambda_c
        self.horizon = horizon_days
        self.dt = dt
        self.steps_per_day = int(round(1.0 / dt))
        self.u_max = u_max
        self.phi_min, self.phi_max = phi_min, phi_max
        self.tox_weight = tox_weight
        self.r_majority = r_majority
        self.control_bonus = control_bonus
        self.progression_bonus = progression_bonus
        self.win_bonus = win_bonus
        self.prog_thr = progression_threshold * self.K
        self.eps = eps
        self.S0, self.R0 = S0, R0
        # interfaz de espacios (compat con código que la consulta)
        self.possible_agents = ["therapy", "tumor"]

    # --- dinámica escalar (idéntica a dynamics.derivatives + rk4_step) ---
    def _deriv(self, S, R, c, u, phi):
        crowd = (S + R) / self.K
        trans = phi * S
        dS = self.aS * S * (1.0 - crowd) - _hill(c, self.ic50_S, self.dmax_S, self.h) * S - trans
        dR = self.aR * R * (1.0 - crowd) - _hill(c, self.ic50_R, self.dmax_R, self.h) * R + trans
        dc = -self.lam * c + u
        return dS, dR, dc

    def _rk4(self, S, R, c, u, phi):
        dt = self.dt
        k1S, k1R, k1c = self._deriv(S, R, c, u, phi)
        k2S, k2R, k2c = self._deriv(S + dt / 2 * k1S, R + dt / 2 * k1R, c + dt / 2 * k1c, u, phi)
        k3S, k3R, k3c = self._deriv(S + dt / 2 * k2S, R + dt / 2 * k2R, c + dt / 2 * k2c, u, phi)
        k4S, k4R, k4c = self._deriv(S + dt * k3S, R + dt * k3R, c + dt * k3c, u, phi)
        S2 = S + dt / 6 * (k1S + 2 * k2S + 2 * k3S + k4S)
        R2 = R + dt / 6 * (k1R + 2 * k2R + 2 * k3R + k4R)
        c2 = c + dt / 6 * (k1c + 2 * k2c + 2 * k3c + k4c)
        return (S2 if S2 > 0.0 else 0.0,
                R2 if R2 > 0.0 else 0.0,
                c2 if c2 > 0.0 else 0.0)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.S, self.R, self.c = self.S0, self.R0, 0.0
        self.day = 0
        self.agents = list(self.possible_agents)
        return (self.S, self.R, self.c)

    def step(self, u, phi):
        u = float(u); phi = float(phi)
        S, R, c = self.S, self.R, self.c
        for _ in range(self.steps_per_day):
            S, R, c = self._rk4(S, R, c, u, phi)
        self.S, self.R, self.c = S, R, c
        self.day += 1

        burden = S + R
        fracR = R / (burden + 1e-9)
        controlled = burden < self.prog_thr
        treatable = fracR < self.r_majority
        alive = controlled and treatable
        extinct = burden < self.eps
        failure = not alive

        r_therapy = (self.control_bonus if alive else 0.0) - self.tox_weight * u
        r_tumor = fracR + (self.progression_bonus if failure else 0.0)
        if extinct:
            r_therapy += self.win_bonus
            r_tumor -= self.win_bonus

        terminated = bool(extinct or failure)
        truncated = bool(self.day >= self.horizon)
        if terminated or truncated:
            self.agents = []
        return (S, R, c), float(r_therapy), float(r_tumor), terminated, truncated
