"""Tests de sanidad del entorno PettingZoo. Ejecutar: python -m pytest -v"""
import numpy as np
from gbmarl.tumor_env import TumorEnv


def _random_actions(env):
    return {a: env.action_space(a).sample() for a in env.agents}


def test_reset_devuelve_dos_agentes():
    env = TumorEnv()
    obs, infos = env.reset(seed=0)
    assert set(obs.keys()) == {"therapy", "tumor"}
    assert obs["therapy"].shape == (3,)


def test_espacios_de_accion_correctos():
    env = TumorEnv(u_max=1.0, phi_max=0.05)
    assert env.action_space("therapy").high[0] == 1.0
    assert env.action_space("tumor").high[0] == 0.05      # transición acotada


def test_rollout_aleatorio_no_crashea_y_respeta_horizonte():
    env = TumorEnv(horizon_days=180)
    env.reset(seed=0)
    pasos = 0
    while env.agents and pasos < 500:
        env.step(_random_actions(env))
        pasos += 1
    assert pasos <= 180


def test_estado_nunca_negativo():
    env = TumorEnv(horizon_days=100)
    env.reset(seed=2)
    while env.agents:
        obs, *_ = env.step(_random_actions(env))
        assert (obs["therapy"] >= 0).all()


def test_recompensa_premia_control_y_castiga_resistencia():
    """Con tumor controlado y poca resistencia, la terapia recibe el bono de control;
    el tumor nunca recibe recompensa negativa por su resistencia (>=0)."""
    env = TumorEnv()
    env.reset(seed=3)
    obs, rew, term, trunc, info = env.step({"therapy": np.array([0.3], np.float32),
                                            "tumor": np.array([0.01], np.float32)})
    assert rew["therapy"] > 0          # día controlado -> bono positivo
    assert rew["tumor"] >= 0           # resistencia siempre suma para el tumor
    assert rew["therapy"] <= env.control_bonus   # no puede exceder el bono máximo