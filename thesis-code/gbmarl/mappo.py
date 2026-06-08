"""
mappo.py — MAPPO adversarial (CTDE) por self-play para 2 agentes.

Diferencia clave vs IPPO:
  · Cada agente decide con OBSERVACIÓN PARCIAL (ejecución descentralizada).
  · El CRÍTICO de cada agente ve el ESTADO CONJUNTO completo (entrenamiento
    centralizado). Eso maneja la no-estacionariedad del aprendizaje simultáneo.

Observación parcial (decisión de diseño = realismo clínico):
  · terapia: [carga total S+R, droga c]   (el médico mide tamaño, no S/R)
  · tumor:   [S, R]                         (conoce su composición)
  · crítico: [S, R, c]                      (estado conjunto, solo en training)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(in_dim, out_dim, hidden=64):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out_dim))


# --- Extractores de observación a partir del estado global [S, R, c] ---
def obs_therapy(state):                       # [carga total, droga]
    return np.array([state[0] + state[1], state[2]], dtype=np.float32)

def obs_tumor(state):                         # [S, R]
    return np.array([state[0], state[1]], dtype=np.float32)

def global_state(state):                      # [S, R, c]
    return np.asarray(state, dtype=np.float32)


class Agent(nn.Module):
    """Actor (obs local) + crítico centralizado (estado conjunto)."""
    def __init__(self, local_dim, global_dim, act_dim, a_low, a_high):
        super().__init__()
        self.actor_mean = mlp(local_dim, act_dim)
        self.log_std = nn.Parameter(np.log(0.3) * torch.ones(act_dim))
        self.critic = mlp(global_dim, 1)        # CENTRALIZADO: ve estado conjunto
        self.register_buffer("a_low", torch.tensor(a_low, dtype=torch.float32))
        self.register_buffer("a_high", torch.tensor(a_high, dtype=torch.float32))

    def dist(self, local_obs):
        return Normal(self.actor_mean(local_obs), self.log_std.exp())

    def value(self, glob):
        return self.critic(glob).squeeze(-1)

    def act(self, local_obs):
        d = self.dist(local_obs)
        raw = d.sample()
        logp = d.log_prob(raw).sum(-1)
        action = torch.clamp(raw, self.a_low, self.a_high)
        return raw, action, logp


def _gae(rewards, values, dones, last_v, gamma, lam):
    n = len(rewards)
    adv = torch.zeros(n); gae = 0.0
    for t in reversed(range(n)):
        next_v = last_v if t == n - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    return adv, adv + values


def _update(agent, opt, O, G, A, LP, ADV, RET, clip, epochs, mb, ent_coef, vf_coef):
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    idx = np.arange(len(O))
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, len(O), mb):
            b = idx[s:s + mb]
            d = agent.dist(O[b])
            logp = d.log_prob(A[b]).sum(-1)
            ratio = (logp - LP[b]).exp()
            s1 = ratio * ADV[b]
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[b]
            pi_loss = -torch.min(s1, s2).mean()
            v_loss = ((agent.value(G[b]) - RET[b]) ** 2).mean()
            ent = d.entropy().sum(-1).mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5); opt.step()


def train_mappo(env, total_timesteps=120000, n_steps=2048, lr=3e-4, gamma=0.99,
                gae_lambda=0.95, clip=0.2, epochs=10, minibatch=64,
                ent_coef=0.0, vf_coef=0.5, seed=0, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)

    th = Agent(2, 3, 1, env.action_space("therapy").low, env.action_space("therapy").high)
    tu = Agent(2, 3, 1, env.action_space("tumor").low, env.action_space("tumor").high)
    opt_th = torch.optim.Adam(th.parameters(), lr=lr)
    opt_tu = torch.optim.Adam(tu.parameters(), lr=lr)

    obs, _ = env.reset(seed=seed)
    state = obs["therapy"]                      # estado global [S,R,c]
    ret_th = ret_tu = 0.0
    rets_th, rets_tu = [], []
    hist = []
    steps = 0

    while steps < total_timesteps:
        buf = {k: {"O": [], "G": [], "A": [], "LP": [], "R": [], "V": [], "D": []}
               for k in ("th", "tu")}
        for _ in range(n_steps):
            o_th = torch.tensor(obs_therapy(state))
            o_tu = torch.tensor(obs_tumor(state))
            g = torch.tensor(global_state(state))
            with torch.no_grad():
                raw_th, act_th, lp_th = th.act(o_th); v_th = th.value(g)
                raw_tu, act_tu, lp_tu = tu.act(o_tu); v_tu = tu.value(g)
            actions = {"therapy": act_th.numpy(), "tumor": act_tu.numpy()}
            nobs, rew, term, trunc, _ = env.step(actions)
            done = (term.get("therapy", True) if term else True) or \
                   (trunc.get("therapy", True) if trunc else True)

            for k, o, raw, lp, v, r in (("th", o_th, raw_th, lp_th, v_th, rew["therapy"]),
                                        ("tu", o_tu, raw_tu, lp_tu, v_tu, rew["tumor"])):
                buf[k]["O"].append(o); buf[k]["G"].append(g); buf[k]["A"].append(raw)
                buf[k]["LP"].append(lp); buf[k]["V"].append(v); buf[k]["R"].append(r)
                buf[k]["D"].append(done)
            ret_th += rew["therapy"]; ret_tu += rew["tumor"]; steps += 1

            if done:
                rets_th.append(ret_th); rets_tu.append(ret_tu); ret_th = ret_tu = 0.0
                nobs, _ = env.reset()
            state = nobs["therapy"]

        # GAE + update por agente, con su crítico centralizado
        for agent, opt, k in ((th, opt_th, "th"), (tu, opt_tu, "tu")):
            o_last = obs_therapy(state) if k == "th" else obs_tumor(state)
            with torch.no_grad():
                last_v = agent.value(torch.tensor(global_state(state)))
            O = torch.stack(buf[k]["O"]); G = torch.stack(buf[k]["G"])
            A = torch.stack(buf[k]["A"]); LP = torch.stack(buf[k]["LP"])
            V = torch.stack(buf[k]["V"])
            R = torch.tensor(buf[k]["R"], dtype=torch.float32)
            D = torch.tensor(buf[k]["D"], dtype=torch.float32)
            ADV, RET = _gae(R, V, D, last_v, gamma, gae_lambda)
            _update(agent, opt, O, G, A, LP, ADV, RET, clip, epochs, minibatch,
                    ent_coef, vf_coef)

        m_th = np.mean(rets_th[-10:]) if rets_th else float('nan')
        m_tu = np.mean(rets_tu[-10:]) if rets_tu else float('nan')
        hist.append((m_th, m_tu))
        if verbose:
            print(f"  steps {steps:6d} | retorno terapia {m_th:8.2f} | retorno tumor {m_tu:8.2f}")
    return th, tu, hist