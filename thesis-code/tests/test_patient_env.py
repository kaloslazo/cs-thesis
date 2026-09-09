"""Tests de la extensión opt-in de toxicidad/estado funcional."""
import numpy as np
import pytest

from gbmarl.config import Params, ToxicityParams
from gbmarl.dynamics import toxicity_derivative, toxicity_exposure
from gbmarl.outcomes import TOXICITY
from gbmarl.patient_env import PatientTumorEnv


TP = ToxicityParams(accumulation_rate=1.0, recovery_rate=0.1,
                    half_saturation=0.2, failure_threshold=0.5,
                    initial_toxicity=0.0, toxicity_weight_in_health=0.7,
                    burden_weight_in_health=0.3)


def test_exposicion_es_monotona_y_acotada():
    assert toxicity_exposure(0.0, 0.2) == 0.0
    assert 0 < toxicity_exposure(0.5, 0.2) < toxicity_exposure(2.0, 0.2) < 1


def test_toxicidad_se_recupera_sin_exposicion():
    assert toxicity_derivative(0.5, 0.0, TP) < 0
    assert toxicity_derivative(0.0, 0.0, TP) == 0


def test_entorno_expone_estado_cuatro_dimensiones():
    env = PatientTumorEnv(TP)
    obs, _ = env.reset(seed=0)
    assert obs["therapy"].shape == (4,)
    assert obs["therapy"][3] == 0.0


def test_entorno_rechaza_doble_castigo_de_toxicidad():
    with pytest.raises(ValueError, match="duplicar toxicidad"):
        PatientTumorEnv(TP, tox_weight=0.05)


def test_dosis_alta_puede_terminar_por_toxicidad():
    static = Params(alpha_S=0.0, alpha_R=0.0, delta_max_S=0.0,
                    delta_max_R=0.0, lambda_c=0.0)
    env = PatientTumorEnv(TP, params=static, horizon_days=20,
                          progression_threshold=1.5, r_majority=0.99)
    env.reset(seed=0)
    while env.agents:
        _, _, terms, _, infos = env.step({
            "therapy": np.array([1.0], np.float32),
            "tumor": np.array([0.0], np.float32),
        })
        if terms.get("therapy"):
            assert infos["therapy"]["failure_mode"] == TOXICITY
            assert infos["therapy"]["t_toxicity"] == infos["therapy"]["day"]
            break
    else:
        raise AssertionError("la configuración de prueba no alcanzó toxicidad")
