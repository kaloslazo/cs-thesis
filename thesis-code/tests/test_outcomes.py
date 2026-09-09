"""Regresiones para la métrica combinada y los motivos terminales."""
import numpy as np

from gbmarl.config import Params
from gbmarl.evalutils import evaluate_episode, ttp_combinado
from gbmarl.exploit import ttp_vs_tumor
from gbmarl.fast_env import FastTumorEnv
from gbmarl.outcomes import (BOTH, EXTINCT, HORIZON, LOAD, RESISTANCE,
                             TOXICITY, classify_outcome)
from gbmarl.tumor_env import TumorEnv


STATIC = Params(alpha_S=0.0, alpha_R=0.0, delta_max_S=0.0,
                delta_max_R=0.0, lambda_c=0.0, phi_base=0.0)
ACTIONS = {"therapy": np.array([0.0], np.float32),
           "tumor": np.array([0.0], np.float32)}


def _step_from(state, *, horizon=180):
    env = TumorEnv(params=STATIC, horizon_days=horizon)
    env.reset(seed=0)
    env.state = np.asarray(state, dtype=float)
    _, _, terms, truncs, infos = env.step(ACTIONS)
    return env, terms, truncs, infos["therapy"]


def test_modo_carga_sin_resistencia():
    _, terms, truncs, info = _step_from([0.79, 0.02, 0.0])
    assert terms["therapy"] and not truncs["therapy"]
    assert info["failure_mode"] == LOAD
    assert info["t_load"] == 1 and info["t_resistance"] is None


def test_modo_resistencia_sin_carga():
    _, _, _, info = _step_from([0.20, 0.30, 0.0])
    assert info["failure_mode"] == RESISTANCE
    assert info["t_load"] is None and info["t_resistance"] == 1


def test_modo_simultaneo_no_pierde_informacion():
    _, _, _, info = _step_from([0.40, 0.40, 0.0])
    assert info["failure_mode"] == BOTH
    assert info["t_load"] == info["t_resistance"] == 1


def test_horizonte_es_censura_y_no_terminacion_clinica():
    _, terms, truncs, info = _step_from([0.40, 0.01, 0.0], horizon=1)
    assert not terms["therapy"] and truncs["therapy"]
    assert info["failure_mode"] == HORIZON
    assert info["censored"]


def test_extincion_no_se_reporta_como_progresion_temprana():
    env = TumorEnv(params=STATIC, horizon_days=20, S0=0.0, R0=0.0)
    result = evaluate_episode(env, lambda _: 0.0, lambda _: 0.0)
    assert result.mode == EXTINCT
    assert result.terminal_day == 1
    assert result.ttp == env.horizon
    assert result.successful and not result.censored
    assert ttp_combinado(env, lambda _: 0.0, lambda _: 0.0) == env.horizon


def test_fast_env_conserva_modo_simultaneo_y_tiempos():
    env = FastTumorEnv(params=STATIC)
    env.reset(seed=0)
    env.S, env.R, env.c = 0.40, 0.40, 0.0
    _, _, _, terminated, truncated = env.step(0.0, 0.0)
    assert terminated and not truncated
    assert env.last_outcome == BOTH
    assert env.last_info["t_load"] == env.last_info["t_resistance"] == 1


def test_fast_env_horizonte_es_censura():
    env = FastTumorEnv(params=STATIC, horizon_days=1)
    env.reset(seed=0)
    _, _, _, terminated, truncated = env.step(0.0, 0.0)
    assert not terminated and truncated
    assert env.last_outcome == HORIZON
    assert env.last_info["censored"]


def test_toxicidad_no_se_oculta_si_coincide_con_extincion():
    assert classify_outcome(progressed=False, untreatable=False, extinct=True,
                            reached_horizon=False, toxicity=True) == TOXICITY


def test_benchmark_acepta_baselines_deterministas_callable():
    env = TumorEnv(params=STATIC, horizon_days=1)
    ttp, mode = ttp_vs_tumor(env, lambda _: 0.0, lambda _: 0.0)
    assert ttp == 1 and mode == HORIZON
