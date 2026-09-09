"""
Parámetros del entorno de simulación.

PROCEDENCIA DE CADA PARÁMETRO (clave para la defensa):
  · DATOS (GDSC2):   efecto del fármaco → IC50 y forma de Hill (delta).
  · LITERATURA:      crecimiento (alpha), capacidad (K), decaimiento (lambda_c).
  · APRENDIDO/ACOTADO: phi(c), la acción del agente tumor (no se calibra aquí).

Los valores de abajo son PLACEHOLDERS de literatura. calibrate.py los
reemplazará con los derivados del dataset y guardará calibration.json.
"""
import hashlib
import json
from dataclasses import asdict, dataclass


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


def params_fingerprint(params: Params) -> str:
    """Huella estable para no mezclar checkpoints de calibraciones distintas."""
    payload = json.dumps(asdict(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ToxicityParams:
    """Parámetros explícitos para la extensión de toxicidad acumulada.

    No tienen valores por defecto a propósito: deben provenir de literatura, datos
    clínicos o un protocolo pre-registrado. Así el entorno no puede producir una
    conclusión sobre salud usando silenciosamente coeficientes ilustrativos.
    """

    accumulation_rate: float
    recovery_rate: float
    half_saturation: float
    failure_threshold: float
    initial_toxicity: float
    toxicity_weight_in_health: float
    burden_weight_in_health: float

    def __post_init__(self):
        if self.accumulation_rate <= 0:
            raise ValueError("accumulation_rate debe ser > 0")
        if self.recovery_rate < 0:
            raise ValueError("recovery_rate debe ser >= 0")
        if self.half_saturation <= 0:
            raise ValueError("half_saturation debe ser > 0")
        if not 0 < self.failure_threshold <= 1:
            raise ValueError("failure_threshold debe estar en (0, 1]")
        if not 0 <= self.initial_toxicity < self.failure_threshold:
            raise ValueError("initial_toxicity debe estar en [0, failure_threshold)")
        weights = (self.toxicity_weight_in_health, self.burden_weight_in_health)
        if any(not 0 <= weight <= 1 for weight in weights):
            raise ValueError("los pesos de salud deben estar en [0, 1]")
        if sum(weights) == 0:
            raise ValueError("al menos un peso de salud debe ser positivo")


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
