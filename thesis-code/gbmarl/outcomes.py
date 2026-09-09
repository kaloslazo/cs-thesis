"""Definiciones compartidas para los desenlaces terminales de GBMARL.

El entorno, los evaluadores y los experimentos deben usar una única clasificación.
Esto evita que cada script infiera el motivo de terminación con prioridades distintas.
"""
from __future__ import annotations


ONGOING = "ongoing"
LOAD = "load"
RESISTANCE = "resistance"
BOTH = "both"
TOXICITY = "toxicity"
MULTIPLE = "multiple"
EXTINCT = "extinct"
HORIZON = "horizon"

FAILURE_MODES = frozenset({LOAD, RESISTANCE, BOTH, TOXICITY, MULTIPLE})
SUCCESS_MODES = frozenset({EXTINCT, HORIZON})

SPANISH_LABELS = {
    LOAD: "FALLO: carga progresó",
    RESISTANCE: "FALLO: resistencia mayoría",
    BOTH: "FALLO: carga y resistencia simultáneas",
    TOXICITY: "FALLO: toxicidad",
    MULTIPLE: "FALLO: múltiples eventos simultáneos",
    EXTINCT: "ÉXITO: tumor extinto",
    HORIZON: "SOBREVIVIÓ horizonte (ÉXITO)",
    ONGOING: "episodio en curso",
}


def resistant_fraction(sensitive: float, resistant: float) -> float:
    """Fracción resistente sin desplazar artificialmente el umbral de 50%."""
    burden = float(sensitive) + float(resistant)
    return float(resistant) / burden if burden > 0.0 else 0.0


def classify_outcome(*, progressed: bool, untreatable: bool,
                     extinct: bool, reached_horizon: bool,
                     toxicity: bool = False) -> str:
    """Clasifica un step con prioridades clínicas explícitas.

    Los eventos de fracaso tienen prioridad sobre la censura administrativa del
    horizonte si ambos ocurren en el mismo día. La extinción tiene prioridad sobre
    todos los eventos tumorales porque una carga extinta no puede ser progresiva.
    """
    tumor_events = int(bool(progressed)) + int(bool(untreatable))
    if toxicity and tumor_events:
        return MULTIPLE
    if toxicity:
        return TOXICITY
    if extinct:
        return EXTINCT
    if progressed and untreatable:
        return BOTH
    if progressed:
        return LOAD
    if untreatable:
        return RESISTANCE
    if reached_horizon:
        return HORIZON
    return ONGOING


def is_failure(mode: str) -> bool:
    return mode in FAILURE_MODES


def is_censored(mode: str) -> bool:
    """Solo el horizonte es censura administrativa; extinción es éxito observado."""
    return mode == HORIZON


def is_success(mode: str) -> bool:
    return mode in SUCCESS_MODES


def scalar_ttp(mode: str, terminal_day: int, horizon: int) -> int:
    """TTP compatible con scripts históricos.

    Un evento de fracaso usa su día real. Horizonte y extinción no son progresión;
    se representan con el horizonte en la vista escalar y se distinguen mediante
    ``mode``/``censored`` en la vista estructurada.
    """
    return int(terminal_day if is_failure(mode) else horizon)
