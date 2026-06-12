"""
train_ckpt.py — Entrenador MAPPO/IPPO REANUDABLE con checkpoint INTRA-corrida.

Resuelve el techo de tiempo del sandbox (no caben 120k pasos en una ventana):
entrena por un PRESUPUESTO de segundos, guarda checkpoint completo
(actores+críticos+optimizadores+RNG+paso+estado del env+historia) y reanuda en
la siguiente invocación hasta alcanzar --steps. Cuando termina imprime 'JOB DONE'.

Reusa gbmarl.mappo (Agent, _gae, _update) y gbmarl.fast_env (rollout escalar,
validado idéntico a TumorEnv). No modifica decisiones bloqueadas: es la misma
física y recompensa, solo encadenada por ventanas.

Uso:
  python scripts/train_ckpt.py --job mappo_main --variant mappo --seed 0 \
         --steps 120000 --budget 35
  (re-ejecutar hasta ver 'JOB DONE')
"""
from __future__ import annotations
import os, sys, time, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from gbmarl.fast_env import FastTumorEnv
from gbmarl.mappo import Agent, _gae, _update

CKPT_DIR = "outputs/ckpt"


def obs_th(S, R, c): return torch.tensor([S + R, c], dtype=torch.float32)
def obs_tu(S, R, c): return torch.tensor([S, R], dtype=torch.float32)
def glob(S, R, c):   return torch.tensor([S, R, c], dtype=torch.float32)


def build(env, variant):
    centralized = (variant == "mappo")
    cdim = 3 if centralized else 2
    lo_th, hi_th = [0.0], [env.u_max]
    lo_tu, hi_tu = [env.phi_min], [env.phi_max]
    th = Agent(2, cdim, 1, np.array(lo_th, np.float32), np.array(hi_th, np.float32))
    tu = Agent(2, cdim, 1, np.array(lo_tu, np.float32), np.array(hi_tu, np.float32))
    return th, tu, centralized


def train_job(job, variant, seed, target, budget, n_steps=2048, lr=3e-4,
              gamma=0.99, lam=0.95, clip=0.2, epochs=10, mb=64,
              ent_coef=0.01, vf_coef=0.5):
    torch.set_num_threads(1)
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = f"{CKPT_DIR}/{job}.pkl"
    env = FastTumorEnv(horizon_days=180)
    th, tu, centralized = build(env, variant)
    opt_th = torch.optim.Adam(th.parameters(), lr=lr)
    opt_tu = torch.optim.Adam(tu.parameters(), lr=lr)

    if os.path.exists(path):
        with open(path, "rb") as f:
            ck = pickle.load(f)
        th.load_state_dict(ck["th"]); tu.load_state_dict(ck["tu"])
        opt_th.load_state_dict(ck["opt_th"]); opt_tu.load_state_dict(ck["opt_tu"])
        np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        steps = ck["steps"]; hist = ck["hist"]
        S, R, c = ck["env_state"]; env.S, env.R, env.c = S, R, c
        env.day = ck["env_day"]; env.agents = ck["env_agents"]
        ret_th, ret_tu = ck["ret_th"], ck["ret_tu"]
        rets_th, rets_tu = ck["rets_th"], ck["rets_tu"]
        if steps >= target:
            print(f"JOB DONE {job} (ya completo: {steps}/{target})")
            return
    else:
        torch.manual_seed(seed); np.random.seed(seed)
        S, R, c = env.reset(seed=seed)
        steps = 0; hist = []
        ret_th = ret_tu = 0.0; rets_th, rets_tu = [], []

    t0 = time.time()
    while steps < target and (time.time() - t0) < budget:
        buf = {k: {"O": [], "G": [], "A": [], "LP": [], "R": [], "V": [], "D": []}
               for k in ("th", "tu")}
        S, R, c = env.S, env.R, env.c
        for _ in range(n_steps):
            o_th, o_tu, g = obs_th(S, R, c), obs_tu(S, R, c), glob(S, R, c)
            cin_th = g if centralized else o_th
            cin_tu = g if centralized else o_tu
            with torch.no_grad():
                raw_th, act_th, lp_th = th.act(o_th); v_th = th.value(cin_th)
                raw_tu, act_tu, lp_tu = tu.act(o_tu); v_tu = tu.value(cin_tu)
            (S, R, c), r_th, r_tu, term, trunc = env.step(float(act_th[0]), float(act_tu[0]))
            done = term or trunc
            for k, o, cin, raw, lp, v, r in (("th", o_th, cin_th, raw_th, lp_th, v_th, r_th),
                                             ("tu", o_tu, cin_tu, raw_tu, lp_tu, v_tu, r_tu)):
                buf[k]["O"].append(o); buf[k]["G"].append(cin); buf[k]["A"].append(raw)
                buf[k]["LP"].append(lp); buf[k]["V"].append(v); buf[k]["R"].append(r)
                buf[k]["D"].append(done)
            ret_th += r_th; ret_tu += r_tu; steps += 1
            if done:
                rets_th.append(ret_th); rets_tu.append(ret_tu); ret_th = ret_tu = 0.0
                S, R, c = env.reset()
        env.S, env.R, env.c = S, R, c

        for agent, opt, k in ((th, opt_th, "th"), (tu, opt_tu, "tu")):
            o_last = obs_th(S, R, c) if k == "th" else obs_tu(S, R, c)
            cin_last = glob(S, R, c) if centralized else o_last
            with torch.no_grad():
                last_v = agent.value(cin_last)
            O = torch.stack(buf[k]["O"]); G = torch.stack(buf[k]["G"])
            A = torch.stack(buf[k]["A"]); LP = torch.stack(buf[k]["LP"])
            V = torch.stack(buf[k]["V"])
            Rr = torch.tensor(buf[k]["R"], dtype=torch.float32)
            D = torch.tensor(buf[k]["D"], dtype=torch.float32)
            ADV, RET = _gae(Rr, V, D, last_v, gamma, lam)
            _update(agent, opt, O, G, A, LP, ADV, RET, clip, epochs, mb, ent_coef, vf_coef)

        m_th = float(np.mean(rets_th[-10:])) if rets_th else float("nan")
        m_tu = float(np.mean(rets_tu[-10:])) if rets_tu else float("nan")
        hist.append((m_th, m_tu))

    ck = {"th": th.state_dict(), "tu": tu.state_dict(),
          "opt_th": opt_th.state_dict(), "opt_tu": opt_tu.state_dict(),
          "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
          "steps": steps, "hist": hist,
          "env_state": (env.S, env.R, env.c), "env_day": env.day,
          "env_agents": env.agents, "ret_th": ret_th, "ret_tu": ret_tu,
          "rets_th": rets_th, "rets_tu": rets_tu,
          "variant": variant, "seed": seed, "target": target, "centralized": centralized}
    with open(path, "wb") as f:
        pickle.dump(ck, f)

    el = time.time() - t0
    if steps >= target:
        # exportar modelos finales en formato .pt para evaluate
        os.makedirs("outputs/models", exist_ok=True)
        torch.save(th.state_dict(), f"outputs/models/{job}_therapy.pt")
        torch.save(tu.state_dict(), f"outputs/models/{job}_tumor.pt")
        print(f"JOB DONE {job} ({steps}/{target}) m_th={hist[-1][0]:.2f} m_tu={hist[-1][1]:.2f} [{el:.1f}s]")
    else:
        print(f"... {job} {steps}/{target} ({100*steps/target:.0f}%) "
              f"m_th={hist[-1][0] if hist else float('nan'):.2f} [{el:.1f}s ventana]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--variant", choices=["mappo", "ippo"], default="mappo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--budget", type=float, default=35.0)
    a = ap.parse_args()
    train_job(a.job, a.variant, a.seed, a.steps, a.budget)


if __name__ == "__main__":
    main()
