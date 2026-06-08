"""Tests de sanidad de la dinámica ODE. Ejecutar: pytest -v"""
import numpy as np
from gbmarl.config import Params
from gbmarl.dynamics import simulate, hill

P = Params()
STATE0 = [0.40, 0.01, 0.0]   # mayoría sensibles, subpoblación resistente diminuta


def test_sin_farmaco_el_tumor_crece_hasta_K():
    """Sin dosis, el tumor total debe crecer y acercarse a la capacidad K."""
    out = simulate(STATE0, lambda t, s: 0.0, lambda t, s: P.phi_base, n_days=200, p=P)
    total_final = out["S"][-1] + out["R"][-1]
    assert total_final > (STATE0[0] + STATE0[1])     # creció
    assert total_final <= P.K + 1e-6                  # no se pasa de K


def test_dosis_alta_selecciona_resistentes():
    """Con dosis alta constante: las sensibles colapsan y suben las resistentes."""
    out = simulate(STATE0, lambda t, s: 0.6, lambda t, s: P.phi_base, n_days=200, p=P)
    assert out["S"][-1] < STATE0[0]                   # sensibles colapsan
    assert out["R"][-1] > STATE0[1]                   # resistentes crecen
    frac0 = STATE0[1] / (STATE0[0] + STATE0[1])
    frac_f = out["R"][-1] / (out["S"][-1] + out["R"][-1] + 1e-9)
    assert frac_f > frac0                             # la composición se desplaza


def test_nunca_negativos():
    """Densidades y concentración siempre >= 0."""
    out = simulate(STATE0, lambda t, s: 0.6, lambda t, s: P.phi_base, n_days=200, p=P)
    assert (out["S"] >= 0).all() and (out["R"] >= 0).all() and (out["c"] >= 0).all()


def test_el_farmaco_decae_sin_dosis():
    """Con c inicial > 0 y sin dosis, el fármaco se elimina."""
    out = simulate([0.4, 0.01, 1.0], lambda t, s: 0.0, lambda t, s: 0.0, n_days=50, p=P)
    assert out["c"][-1] < out["c"][0]


def test_hill_monotona_y_acotada():
    """La curva de Hill crece con la dosis y no supera delta_max."""
    assert hill(0.0, P.ic50_S, P.delta_max_S, P.hill) == 0.0
    assert hill(10.0, P.ic50_S, P.delta_max_S, P.hill) < P.delta_max_S
    assert hill(1.0, P.ic50_S, P.delta_max_S, P.hill) > hill(0.2, P.ic50_S, P.delta_max_S, P.hill)