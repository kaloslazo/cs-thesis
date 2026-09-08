"""
dynamics_health.py — Dinámica 4-D con SALUD del paciente (Fase 1, Opción A).

Extiende el sistema 3-D de dynamics.py con una 4ª variable de estado H (salud):

    dS/dt = alpha_S*S*(1 - (S+R)/K) - delta_S(c)*S - phi*S
    dR/dt = alpha_R*R*(1 - (S+R)/K) - delta_R(c)*R + phi*S
    dc/dt = -lambda_c*c + u
    dH/dt = rho_H*(1 - H) - kappa_u*u - kappa_b*(S+R)      <-- NUEVO

  H in [0,1] (1 = sano). Se recupera solo hacia 1 (rho_H); la dosis lo daña
  (kappa_u*u, toxicidad) y la carga tumoral lo daña (kappa_b*(S+R)).

NO toca dynamics.py: el path 3-D sigue intacto. Reusa `hill` de dynamics.py para
la muerte por fármaco (una sola definición de la curva dosis-respuesta).

Math pura: no sabe de RL ni agentes. Se testea solo.
"""
from __future__ import annotations
import numpy as np
from .config import Params
from .dynamics import hill


def derivatives_h(state, u: float, phi: float, p: Params):
    """Devuelve [dS/dt, dR/dt, dc/dt, dH/dt] para el estado 4-D [S,R,c,H]."""
    S, R, c, H = state
    crowd = (S + R) / p.K
    transicion = phi * S
    dS = (p.alpha_S * S * (1 - crowd)
          - hill(c, p.ic50_S, p.delta_max_S, p.hill) * S
          - transicion)
    dR = (p.alpha_R * R * (1 - crowd)
          - hill(c, p.ic50_R, p.delta_max_R, p.hill) * R
          + transicion)
    dc = -p.lambda_c * c + u
    dH = p.rho_H * (1 - H) - p.kappa_u * u - p.kappa_b * (S + R)
    return np.array([dS, dR, dc, dH], dtype=float)


def rk4_step_h(state, u: float, phi: float, dt: float, p: Params):
    """Un paso RK4 sobre el estado 4-D. S,R,c se clipan a >=0; H a [0,1]."""
    s = np.asarray(state, dtype=float)
    k1 = derivatives_h(s, u, phi, p)
    k2 = derivatives_h(s + dt / 2 * k1, u, phi, p)
    k3 = derivatives_h(s + dt / 2 * k2, u, phi, p)
    k4 = derivatives_h(s + dt * k3, u, phi, p)
    s_next = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    s_next[:3] = np.maximum(s_next[:3], 0.0)     # densidades y droga no negativas
    s_next[3] = np.clip(s_next[3], 0.0, 1.0)     # salud acotada a [0,1]
    return s_next
