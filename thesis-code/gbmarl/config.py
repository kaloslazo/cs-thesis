"""
Parámetros del entorno de simulación.

PROCEDENCIA DE CADA PARÁMETRO (clave para la defensa):
  · DATOS (GDSC2):   efecto del fármaco → IC50 y forma de Hill (delta).
  · LITERATURA:      crecimiento (alpha), capacidad (K), decaimiento (lambda_c).
  · APRENDIDO/ACOTADO: phi(c), la acción del agente tumor (no se calibra aquí).

Los valores de abajo son PLACEHOLDERS de literatura. calibrate.py los
reemplazará con los derivados del dataset y guardará calibration.json.
"""
from dataclasses import dataclass


@dataclass
class Params:
    # --- Crecimiento (LITERATURA: cinética tumoral GBM) ---
    alpha_S: float = 0.15     # tasa de crecimiento células sensibles (1/día)
    alpha_R: float = 0.10     # resistentes crecen más lento = costo de resistencia
    K: float = 1.0            # capacidad de carga (densidad normalizada)

    # --- Muerte por fármaco, función de Hill (DATOS: GDSC2) ---
    delta_max_S: float = 0.50  # muerte máxima de sensibles
    delta_max_R: float = 0.15  # resistentes mueren mucho menos
    ic50_S: float = 0.30       # sensibles mueren a baja concentración
    ic50_R: float = 1.50       # resistentes necesitan mucha más droga
    hill: float = 2.0          # pendiente de la curva dosis-respuesta

    # --- Farmacocinética del fármaco (LITERATURA: PK de TMZ) ---
    lambda_c: float = 0.40     # decaimiento del fármaco (1/día)

    # --- Transición fenotípica (placeholder; será acción del agente tumor) ---
    phi_base: float = 0.005    # tasa basal sensible -> resistente


def load_calibration(path: str = "data/processed/calibration.json") -> Params:
    """
    Devuelve Params con los 4 campos de fármaco REEMPLAZADOS por los valores
    derivados del dataset (calibration.json). El resto (alpha, K, lambda_c) se
    mantiene como literatura. Si no existe el json, devuelve los placeholders.
    """
    import json
    import os
    p = Params()
    if not os.path.exists(path):
        print(f"[config] AVISO: {path} no existe; uso placeholders de literatura.")
        return p
    with open(path) as f:
        d = json.load(f)["derived_params"]
    p.ic50_S = d["ic50_S"]
    p.ic50_R = d["ic50_R"]
    p.delta_max_S = d["delta_max_S"]
    p.delta_max_R = d["delta_max_R"]
    return p