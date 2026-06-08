"""
ppo.py — PPO compacto (continuo) estilo CleanRL, comentado para entenderlo.

Actor-Crítico:
  · Actor: estado -> media de una gaussiana (la dosis se muestrea de ahí).
  · Crítico: estado -> valor (qué tan buena es la situación).
PPO: ventaja con GAE + objetivo "clipeado" para updates estables.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(in_dim, out_dim, hidden=64):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor_mean = mlp(obs_dim, act_dim)
        self.log_std = nn.Parameter(np.log(0.3) * torch.ones(act_dim))
        self.critic = mlp(obs_dim, 1)

    def dist(self, obs):
        return Normal(self.actor_mean(obs), self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def train(env, total_timesteps=60000, n_steps=2048, lr=3e-4, gamma=0.99,
          gae_lambda=0.95, clip=0.2, epochs=10, minibatch=64,
          ent_coef=0.0, vf_coef=0.5, seed=0, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    a_low = torch.tensor(env.action_space.low)
    a_high = torch.tensor(env.action_space.high)
    ac = ActorCritic(obs_dim, act_dim)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)

    obs, _ = env.reset(seed=seed)
    obs = torch.tensor(obs, dtype=torch.float32)
    ep_ret, ep_rets = 0.0, []
    history = []
    steps_done = 0

    while steps_done < total_timesteps:
        # --- 1. Recolectar un rollout ---
        O, A, LP, R, V, D = [], [], [], [], [], []
        for _ in range(n_steps):
            with torch.no_grad():
                dist = ac.dist(obs); v = ac.value(obs)
                raw = dist.sample()
                logp = dist.log_prob(raw).sum(-1)
            act = torch.clamp(raw, a_low, a_high)        # dosis válida
            nobs, r, term, trunc, _ = env.step(act.numpy())
            done = term or trunc
            O.append(obs); A.append(raw); LP.append(logp); R.append(r); V.append(v); D.append(done)
            ep_ret += r; steps_done += 1
            if done:
                ep_rets.append(ep_ret); ep_ret = 0.0
                nobs, _ = env.reset()
            obs = torch.tensor(nobs, dtype=torch.float32)

        # --- 2. GAE (ventajas) ---
        with torch.no_grad():
            last_v = ac.value(obs)
        O = torch.stack(O); A = torch.stack(A); LP = torch.stack(LP)
        V = torch.stack(V); R = torch.tensor(R, dtype=torch.float32)
        D = torch.tensor(D, dtype=torch.float32)
        adv = torch.zeros(n_steps); gae = 0.0
        for t in reversed(range(n_steps)):
            next_v = last_v if t == n_steps - 1 else V[t + 1]
            mask = 1.0 - D[t]
            delta = R[t] + gamma * next_v * mask - V[t]
            gae = delta + gamma * gae_lambda * mask * gae
            adv[t] = gae
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # --- 3. Update PPO (clipeado), varias epochs ---
        idx = np.arange(n_steps)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, n_steps, minibatch):
                b = idx[s:s + minibatch]
                dist = ac.dist(O[b])
                logp = dist.log_prob(A[b]).sum(-1)
                ratio = (logp - LP[b]).exp()
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[b]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = ((ac.value(O[b]) - ret[b]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + vf_coef * v_loss - ent_coef * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()

        recent = np.mean(ep_rets[-10:]) if ep_rets else float('nan')
        history.append(recent)
        if verbose:
            print(f"  steps {steps_done:6d} | retorno medio (10 ep): {recent:8.3f}")
    return ac, history


@torch.no_grad()
def evaluate(env, ac, n_eps=20, seed=100):
    rets = []
    for k in range(n_eps):
        obs, _ = env.reset(seed=seed + k); obs = torch.tensor(obs, dtype=torch.float32)
        done = False; ret = 0.0
        while not done:
            a = ac.actor_mean(obs).clamp(torch.tensor(env.action_space.low),
                                         torch.tensor(env.action_space.high))
            obs, r, term, trunc, _ = env.step(a.numpy()); done = term or trunc
            obs = torch.tensor(obs, dtype=torch.float32); ret += r
        rets.append(ret)
    return float(np.mean(rets))


def random_baseline(env, n_eps=20, seed=200):
    rets = []
    for k in range(n_eps):
        env.reset(seed=seed + k); done = False; ret = 0.0
        while not done:
            _, r, term, trunc, _ = env.step(env.action_space.sample()); done = term or trunc
            ret += r
        rets.append(ret)
    return float(np.mean(rets))