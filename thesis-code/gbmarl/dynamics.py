"""
Dinámica del tumor: sistema de ODEs (Lotka-Volterra + fármaco) e integrador RK4.

    dS/dt = alpha_S*S*(1 - (S+R)/K) - delta_S(c)*S
    dR/dt = alpha_R*R*(1 - (S+R)/K) - delta_R(c)*R + phi*S
    dc/dt = -lambda_c*c + u

  S = densidad de células sensibles
  R = densidad de células resistentes
  c = concentración de fármaco
  u = dosis administrada (acción del agente terapia)
  phi = tasa de transición sensible->resistente (acción del agente tumor)

Este módulo es MATEMÁTICA PURA: no sabe de RL ni de agentes. u y phi se le
pasan desde afuera, por eso se testea solo.
"""
from __future__ import annotations
import numpy as np
from .config import Params


def hill(c: float, ic50: float, delta_max: float, h: float) -> float:
    """Muerte por fármaco saturante (curva dosis-respuesta de Hill)."""
    c = max(c, 0.0)
    return delta_max * (c ** h) / (ic50 ** h + c ** h)


def derivatives(state, u: float, phi: float, p: Params):
    """Devuelve [dS/dt, dR/dt, dc/dt] dado el estado y las acciones u, phi."""
    S, R, c = state
    crowd = (S + R) / p.K                       # término logístico de competencia
    transicion = phi * S                        # sensibles que se vuelven resistentes
    dS = (p.alpha_S * S * (1 - crowd)
          - hill(c, p.ic50_S, p.delta_max_S, p.hill) * S
          - transicion)                         # <- conserva masa (sale de S)
    dR = (p.alpha_R * R * (1 - crowd)
          - hill(c, p.ic50_R, p.delta_max_R, p.hill) * R
          + transicion)                         # <- entra a R
    dc = -p.lambda_c * c + u
    return np.array([dS, dR, dc], dtype=float)


def rk4_step(state, u: float, phi: float, dt: float, p: Params):
    """Un paso de Runge-Kutta de 4to orden. Clipa a >= 0 (validez biológica)."""
    s = np.asarray(state, dtype=float)
    k1 = derivatives(s, u, phi, p)
    k2 = derivatives(s + dt / 2 * k1, u, phi, p)
    k3 = derivatives(s + dt / 2 * k2, u, phi, p)
    k4 = derivatives(s + dt * k3, u, phi, p)
    s_next = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.maximum(s_next, 0.0)


def simulate(state0, dose_fn, phi_fn, n_days: int, dt: float = 0.1, p: Params | None = None):
    """
    Integra la dinámica n_days días.
      dose_fn(t, state) -> u    (régimen de dosis)
      phi_fn(t, state)  -> phi  (transición fenotípica)
    Devuelve un dict con arrays t, S, R, c.
    """
    p = p or Params()
    n_steps = int(n_days / dt)
    s = np.asarray(state0, dtype=float)
    T = np.zeros(n_steps + 1)
    traj = np.zeros((n_steps + 1, 3))
    traj[0] = s
    for i in range(n_steps):
        t = i * dt
        u = dose_fn(t, s)
        phi = phi_fn(t, s)
        s = rk4_step(s, u, phi, dt, p)
        traj[i + 1] = s
        T[i + 1] = (i + 1) * dt
    return {"t": T, "S": traj[:, 0], "R": traj[:, 1], "c": traj[:, 2]}