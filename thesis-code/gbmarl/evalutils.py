"""
evalutils.py — Métrica de evaluación CORRECTA y baselines, en un solo lugar.

TTP-combinado: días hasta que el tumor deja de estar controlado (carga<umbral)
Y tratable (fracR<mayoría) — lo que falle primero. Es el desenlace clínico real.
NO confundir con la métrica vieja (solo-resistencia), que reportaba el horizonte
cuando la falla era por carga, inflando falsamente los resultados.
"""
import numpy as np

FIXED_PHI = 0.01
U_MAX = 1.0


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


def ttp_combinado(env, therapy_fn, tumor_fn=lambda s: FIXED_PHI):
    """Días hasta que el episodio termina (carga progresó O resistencia mayoría
    O sobrevivió el horizonte). Es el tiempo de manejo clínico exitoso."""
    if hasattr(therapy_fn, "reset"):
        therapy_fn.reset()
    obs, _ = env.reset(seed=0)
    state = obs["therapy"]
    info = {}
    while True:
        u = float(therapy_fn(state))
        phi = float(tumor_fn(state))
        obs, rew, terms, truncs, infos = env.step(
            {"therapy": np.array([u], np.float32),
             "tumor": np.array([phi], np.float32)})
        state = obs["therapy"]
        info = infos["therapy"]
        if terms.get("therapy", False) or truncs.get("therapy", False):
            return info["day"]


def ttp_combinado_detalle(env, therapy_fn, tumor_fn=lambda s: FIXED_PHI):
    """Como ttp_combinado pero devuelve (días, motivo, carga, fracR, dosis_media)."""
    if hasattr(therapy_fn, "reset"):
        therapy_fn.reset()
    obs, _ = env.reset(seed=0)
    state = obs["therapy"]
    doses = []
    info = {}
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
    if trunc and not term:
        motivo = "SOBREVIVIÓ horizonte (ÉXITO)"
    elif info.get("untreatable"):
        motivo = "FALLO: resistencia mayoría"
    elif info.get("progressed"):
        motivo = "FALLO: carga progresó"
    else:
        motivo = "extinto"
    return info["day"], motivo, info["burden"], info["fracR"], float(np.mean(doses))