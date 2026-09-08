"""Tests de sanidad de la dinámica 4-D con salud (Fase 1). Ejecutar: pytest -v"""
import numpy as np
from gbmarl.config import Params
from gbmarl.dynamics_health import rk4_step_h, derivatives_h

P = Params()
STATE0 = [0.40, 0.01, 0.0, 1.0]   # S, R, c, H (paciente sano)


def test_salud_acotada_en_cero_uno():
    """H nunca sale de [0,1] tras integrar, aun con dosis máxima sostenida."""
    s = np.array(STATE0, dtype=float)
    for _ in range(2000):
        s = rk4_step_h(s, u=1.0, phi=0.0, dt=0.1, p=P)
        assert -1e-9 <= s[3] <= 1.0 + 1e-9


def test_dosis_alta_baja_la_salud():
    """Con dosis alta sostenida, la salud debe caer por toxicidad."""
    s = np.array(STATE0, dtype=float)
    for _ in range(300):
        s = rk4_step_h(s, u=1.0, phi=0.0, dt=0.1, p=P)
    assert s[3] < STATE0[3]                      # H bajó respecto al inicio


def test_sin_dosis_y_tumor_bajo_la_salud_se_mantiene_alta():
    """Sin fármaco y con carga pequeña, la recuperación domina: H se mantiene alta."""
    s = np.array([0.05, 0.0, 0.0, 0.8], dtype=float)  # salud algo mermada, tumor bajo
    for _ in range(300):
        s = rk4_step_h(s, u=0.0, phi=0.0, dt=0.1, p=P)
    assert s[3] > 0.8                            # recuperó salud hacia 1


def test_salud_no_altera_la_dinamica_tumoral():
    """dS,dR,dc del path 4-D deben coincidir con la fórmula 3-D (la salud no acopla al tumor)."""
    from gbmarl.dynamics import derivatives as deriv3
    s4 = np.array([0.40, 0.10, 0.5, 0.7], dtype=float)
    d4 = derivatives_h(s4, u=0.3, phi=0.01, p=P)
    d3 = deriv3(s4[:3], u=0.3, phi=0.01, p=P)
    assert np.allclose(d4[:3], d3)              # S,R,c idénticos; H es aditivo
