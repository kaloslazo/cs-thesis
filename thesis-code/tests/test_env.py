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
    assert env.action_space("therapy").high[0] == 1.0      # dosis acotada
    assert env.action_space("tumor").high[0] == 0.05       # transición acotada (decisión 7)


def test_rollout_aleatorio_no_crashea_y_respeta_horizonte():
    env = TumorEnv(horizon_days=180)
    env.reset(seed=0)
    pasos = 0
    while env.agents and pasos < 500:
        env.step(_random_actions(env))
        pasos += 1
    assert pasos <= 180                                    # nunca pasa el horizonte


def test_estado_nunca_negativo():
    env = TumorEnv(horizon_days=100)
    env.reset(seed=2)
    while env.agents:
        obs, *_ = env.step(_random_actions(env))
        assert (obs["therapy"] >= 0).all()


def test_recompensas_opuestas():
    """Tumor premia sobrevivir (>=0); terapia penaliza tumor+toxicidad (<=0)."""
    env = TumorEnv()
    env.reset(seed=3)
    obs, rew, *_ = env.step({"therapy": np.array([0.5], np.float32),
                             "tumor": np.array([0.02], np.float32)})
    assert rew["tumor"] >= 0
    assert rew["therapy"] <= 0