"""
mappo_health.py — MAPPO/IPPO self-play para el entorno con SALUD (estado 4-D).

Reusa el núcleo de PPO de mappo.py (Agent, GAE, update): una sola implementación
de PPO en el repo. Solo cambia la EXTRACCIÓN de observaciones (el estado es 4-D
[S,R,c,H]) y las dimensiones de entrada por agente.

Observación parcial (mismo criterio de realismo clínico que el path 3-D):
  · terapia: [carga total S+R, droga c, salud H]  -> el médico mide tamaño y estado del paciente
  · tumor:   [S, R]                                -> conoce su composición, no la salud
  · crítico: [S, R, c, H]                          -> estado conjunto, solo en entrenamiento (CTDE)
"""
from __future__ import annotations
import numpy as np
import torch

from .mappo import Agent, _gae, _update


def obs_therapy_h(state):                     # [carga total, droga, salud]
    return np.array([state[0] + state[1], state[2], state[3]], dtype=np.float32)

def obs_tumor_h(state):                       # [S, R]
    return np.array([state[0], state[1]], dtype=np.float32)

def global_state_h(state):                    # [S, R, c, H]
    return np.asarray(state, dtype=np.float32)


def train_mappo_health(env, total_timesteps=120000, n_steps=2048, lr=3e-4, gamma=0.99,
                       gae_lambda=0.95, clip=0.2, epochs=10, minibatch=64,
                       ent_coef=0.01, vf_coef=0.5, seed=0, centralized=True, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        if verbose:
            print(f"[mappo_health] determinismo no disponible: {e}")

    # Dimensiones: terapia obs=3, tumor obs=2. Crítico: 4 (CTDE) o local (IPPO).
    cdim_th = 4 if centralized else 3
    cdim_tu = 4 if centralized else 2
    th = Agent(3, cdim_th, 1, env.action_space("therapy").low, env.action_space("therapy").high)
    tu = Agent(2, cdim_tu, 1, env.action_space("tumor").low, env.action_space("tumor").high)
    opt_th = torch.optim.Adam(th.parameters(), lr=lr)
    opt_tu = torch.optim.Adam(tu.parameters(), lr=lr)

    obs, _ = env.reset(seed=seed)
    state = obs["therapy"]
    ret_th = ret_tu = 0.0
    rets_th, rets_tu = [], []
    hist = []
    steps = 0

    while steps < total_timesteps:
        buf = {k: {"O": [], "G": [], "A": [], "LP": [], "R": [], "V": [], "D": []}
               for k in ("th", "tu")}
        for _ in range(n_steps):
            o_th = torch.tensor(obs_therapy_h(state))
            o_tu = torch.tensor(obs_tumor_h(state))
            g = torch.tensor(global_state_h(state))
            cin_th = g if centralized else o_th
            cin_tu = g if centralized else o_tu
            with torch.no_grad():
                raw_th, act_th, lp_th = th.act(o_th); v_th = th.value(cin_th)
                raw_tu, act_tu, lp_tu = tu.act(o_tu); v_tu = tu.value(cin_tu)
            actions = {"therapy": act_th.numpy(), "tumor": act_tu.numpy()}
            nobs, rew, term, trunc, _ = env.step(actions)
            done = (term.get("therapy", True) if term else True) or \
                   (trunc.get("therapy", True) if trunc else True)

            for k, o, cin, raw, lp, v, r in (("th", o_th, cin_th, raw_th, lp_th, v_th, rew["therapy"]),
                                             ("tu", o_tu, cin_tu, raw_tu, lp_tu, v_tu, rew["tumor"])):
                buf[k]["O"].append(o); buf[k]["G"].append(cin); buf[k]["A"].append(raw)
                buf[k]["LP"].append(lp); buf[k]["V"].append(v); buf[k]["R"].append(r)
                buf[k]["D"].append(done)
            ret_th += rew["therapy"]; ret_tu += rew["tumor"]; steps += 1

            if done:
                rets_th.append(ret_th); rets_tu.append(ret_tu); ret_th = ret_tu = 0.0
                nobs, _ = env.reset()
            state = nobs["therapy"]

        for agent, opt, k in ((th, opt_th, "th"), (tu, opt_tu, "tu")):
            o_last = obs_therapy_h(state) if k == "th" else obs_tumor_h(state)
            cin_last = global_state_h(state) if centralized else o_last
            with torch.no_grad():
                last_v = agent.value(torch.tensor(cin_last))
            O = torch.stack(buf[k]["O"]); G = torch.stack(buf[k]["G"])
            A = torch.stack(buf[k]["A"]); LP = torch.stack(buf[k]["LP"])
            V = torch.stack(buf[k]["V"])
            R = torch.tensor(buf[k]["R"], dtype=torch.float32)
            D = torch.tensor(buf[k]["D"], dtype=torch.float32)
            ADV, RET = _gae(R, V, D, last_v, gamma, gae_lambda)
            _update(agent, opt, O, G, A, LP, ADV, RET, clip, epochs, minibatch,
                    ent_coef, vf_coef)

        m_th = np.mean(rets_th[-10:]) if rets_th else float("nan")
        m_tu = np.mean(rets_tu[-10:]) if rets_tu else float("nan")
        hist.append((m_th, m_tu))
        if verbose:
            print(f"  steps {steps:6d} | retorno terapia {m_th:8.2f} | retorno tumor {m_tu:8.2f}")
    return th, tu, hist


@torch.no_grad()
def ttp_health_detalle(env, th, tu):
    """TTP-combinado en el entorno con salud; reporta el modo de falla, incluida SALUD.
    Ambos agentes actúan determinista (media). Devuelve (día, modo, H_final, H_medio)."""
    obs, _ = env.reset(seed=0)
    state = obs["therapy"]
    Hs = []
    info = {}
    term = trunc = False
    while True:
        a_th = th.actor_mean(torch.tensor(obs_therapy_h(state))).clamp(th.a_low, th.a_high)
        a_tu = tu.actor_mean(torch.tensor(obs_tumor_h(state))).clamp(tu.a_low, tu.a_high)
        obs, _, terms, truncs, infos = env.step(
            {"therapy": a_th.numpy(), "tumor": a_tu.numpy()})
        state = obs["therapy"]; info = infos["therapy"]; Hs.append(info["H"])
        term = terms.get("therapy", False); trunc = truncs.get("therapy", False)
        if term or trunc:
            break
    if trunc and not term:
        modo = "sobrevivio"
    elif info.get("unhealthy"):
        modo = "salud"
    elif info.get("untreatable"):
        modo = "resistencia"
    elif info.get("progressed"):
        modo = "carga"
    else:
        modo = "extinto"
    return info["day"], modo, float(info["H"]), float(np.mean(Hs))
