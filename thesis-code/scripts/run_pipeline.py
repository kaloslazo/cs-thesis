"""
run_pipeline.py — Orquestador REANUDABLE del pipeline RL (sandbox con techo de 45s).

Ejecuta UN job por invocación, guarda el resultado en un manifiesto JSON y reanuda
desde donde quedó. Pensado para entornos donde cada llamada tiene un límite de
tiempo y los procesos en segundo plano no sobreviven entre llamadas.

Jobs encadenados (escala configurable, por defecto 30k steps × 5 semillas):
  1. ppo            — PPO single-agent (de-risk): terapia vs tumor fijo.
  2. mappo_main     — MAPPO self-play CTDE (modelo titular). Guarda modelos+curvas.
  3. abl_mappo_k    — ablación: MAPPO (crítico global) por semilla k.
  4. abl_ippo_k     — ablación: IPPO (crítico local) por semilla k.
  5. evaluate       — TTP-combinado de baselines vs MAPPO (métrica correcta).

Uso:
  python scripts/run_pipeline.py --init --ppo-steps 30000 --mappo-steps 30000 \
                                 --abl-steps 30000 --seeds 5
  python scripts/run_pipeline.py            # corre el siguiente job pendiente
  python scripts/run_pipeline.py --status   # muestra progreso
  python scripts/run_pipeline.py --all      # corre todos los pendientes que quepan
                                            # en --budget segundos (default 38)
"""
import os
import sys
import json
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

STATE = "data/processed/pipeline_state.json"
MODELS = "outputs/models"
OUTDIR = "outputs"


# ───────────────────────── Manifiesto ──────────────────────────
def build_jobs(seeds):
    jobs = [{"id": "ppo", "type": "ppo", "status": "pending", "result": None},
            {"id": "mappo_main", "type": "mappo_main", "status": "pending", "result": None}]
    for k in range(seeds):
        jobs.append({"id": f"abl_mappo_{k}", "type": "abl", "variant": "mappo",
                     "seed": k, "status": "pending", "result": None})
    for k in range(seeds):
        jobs.append({"id": f"abl_ippo_{k}", "type": "abl", "variant": "ippo",
                     "seed": k, "status": "pending", "result": None})
    jobs.append({"id": "evaluate", "type": "evaluate", "status": "pending", "result": None})
    return jobs


def load_state():
    with open(STATE) as f:
        return json.load(f)


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)


def init_state(args):
    st = {"config": {"ppo_steps": args.ppo_steps, "mappo_steps": args.mappo_steps,
                     "abl_steps": args.abl_steps, "seeds": args.seeds},
          "jobs": build_jobs(args.seeds)}
    save_state(st)
    print(f"[init] manifiesto creado: {len(st['jobs'])} jobs "
          f"(ppo={args.ppo_steps}, mappo={args.mappo_steps}, "
          f"abl={args.abl_steps}×{args.seeds} semillas ×2)")
    return st


# ───────────────────────── Ejecutores de job ──────────────────────────
def run_ppo(cfg):
    from gbmarl.single_env import TherapyEnv
    from gbmarl.ppo import train, evaluate, random_baseline
    import torch
    env = TherapyEnv(fixed_phi=0.01, horizon_days=180)
    base = random_baseline(env, n_eps=20)
    ac, hist = train(env, total_timesteps=cfg["ppo_steps"], verbose=False)
    trained = evaluate(env, ac, n_eps=20)
    os.makedirs(MODELS, exist_ok=True)
    torch.save(ac.state_dict(), f"{MODELS}/ppo_therapy.pt")
    return {"baseline_aleatorio": round(base, 3), "ppo": round(trained, 3),
            "mejora": round(trained - base, 3), "aprendio": bool(trained > base)}


def run_mappo_main(cfg):
    from gbmarl.tumor_env import TumorEnv
    from gbmarl.mappo import train_mappo, obs_therapy, obs_tumor
    import torch
    env = TumorEnv(horizon_days=180)
    th, tu, hist = train_mappo(env, total_timesteps=cfg["mappo_steps"],
                               seed=0, centralized=True, verbose=False)

    @torch.no_grad()
    def eval_ret(tu_agent=None, fixed_phi=None, n=10, seed=300):
        rets = []
        for k in range(n):
            obs, _ = env.reset(seed=seed + k); state = obs["therapy"]; ret = 0.0
            done = False
            while not done:
                a_th = th.actor_mean(torch.tensor(obs_therapy(state))).clamp(th.a_low, th.a_high)
                if tu_agent is not None:
                    a_tu = tu_agent.actor_mean(torch.tensor(obs_tumor(state))).clamp(
                        tu_agent.a_low, tu_agent.a_high).numpy()
                else:
                    a_tu = np.array([fixed_phi], np.float32)
                obs, rew, term, trunc, _ = env.step({"therapy": a_th.numpy(), "tumor": a_tu})
                done = (term.get("therapy", True) if term else True) or \
                       (trunc.get("therapy", True) if trunc else True)
                state = obs["therapy"]; ret += rew["therapy"]
            rets.append(ret)
        return float(np.mean(rets))

    r_fixed = eval_ret(fixed_phi=0.01)
    r_adapt = eval_ret(tu_agent=tu)
    os.makedirs(MODELS, exist_ok=True)
    torch.save(th.state_dict(), f"{MODELS}/mappo_therapy.pt")
    torch.save(tu.state_dict(), f"{MODELS}/mappo_tumor.pt")
    # curvas
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = np.array(hist)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(h[:, 0], color="#2E75B6"); ax[0].set_title("Retorno Terapia")
        ax[0].set_xlabel("Actualización"); ax[0].grid(alpha=0.3)
        ax[1].plot(h[:, 1], color="#C0392B"); ax[1].set_title("Retorno Tumor")
        ax[1].set_xlabel("Actualización"); ax[1].grid(alpha=0.3)
        fig.suptitle("MAPPO self-play (CTDE): co-evolución terapia vs tumor")
        fig.tight_layout(); os.makedirs(OUTDIR, exist_ok=True)
        fig.savefig(f"{OUTDIR}/mappo_learning_curves.png", dpi=150, facecolor="white")
        plt.close(fig)
    except Exception as e:
        print(f"[mappo_main] fig omitida: {type(e).__name__}: {e}")
    return {"retorno_vs_fijo": round(r_fixed, 2), "retorno_vs_adaptativo": round(r_adapt, 2),
            "costo_adversario": round(r_fixed - r_adapt, 2)}


def run_abl(cfg, variant, seed):
    from gbmarl.tumor_env import TumorEnv
    from gbmarl.mappo import train_mappo, obs_therapy
    from gbmarl.evalutils import ttp_combinado
    import torch
    env = TumorEnv(horizon_days=180)
    centralized = (variant == "mappo")
    th, _, _ = train_mappo(env, total_timesteps=cfg["abl_steps"], seed=seed,
                           centralized=centralized, verbose=False)

    def pol(state):
        with torch.no_grad():
            a = th.actor_mean(torch.tensor(obs_therapy(state))).clamp(th.a_low, th.a_high)
        return float(a[0])

    ttp = ttp_combinado(env, pol)
    return {"variant": variant, "seed": seed, "ttp_combinado": float(ttp)}


def run_evaluate(cfg):
    from gbmarl.tumor_env import TumorEnv
    from gbmarl.evalutils import ttp_combinado_detalle, Gatenby
    env = TumorEnv(horizon_days=180)
    U = 1.0
    estrategias = {"Sin tratamiento": lambda s: 0.0,
                   "MTD (dosis max)": lambda s: U,
                   "Adaptativa (Gatenby)": Gatenby()}
    # MAPPO aprendido
    try:
        import torch
        from gbmarl.mappo import Agent, obs_therapy
        th = Agent(2, 3, 1, env.action_space("therapy").low, env.action_space("therapy").high)
        th.load_state_dict(torch.load(f"{MODELS}/mappo_therapy.pt")); th.eval()
        def mappo_pol(state):
            with torch.no_grad():
                a = th.actor_mean(torch.tensor(obs_therapy(state))).clamp(th.a_low, th.a_high)
            return float(a[0])
        estrategias["MAPPO (aprendida)"] = mappo_pol
    except Exception as e:
        print(f"[evaluate] MAPPO no disponible: {e}")

    filas = []
    for nombre, fn in estrategias.items():
        if hasattr(fn, "reset"):
            fn.reset()
        dia, motivo, carga, fracR, dosis = ttp_combinado_detalle(env, fn)
        filas.append({"estrategia": nombre, "ttp_combinado": int(dia), "motivo": motivo,
                      "carga_final": round(float(carga), 3), "fracR_final": round(float(fracR), 3),
                      "dosis_media": round(float(dosis), 3)})
    return {"tabla": filas}


# ───────────────────────── Driver ──────────────────────────
def next_pending(st):
    for j in st["jobs"]:
        if j["status"] == "pending":
            return j
    return None


def run_job(st, job):
    cfg = st["config"]
    t = time.time()
    if job["type"] == "ppo":
        res = run_ppo(cfg)
    elif job["type"] == "mappo_main":
        res = run_mappo_main(cfg)
    elif job["type"] == "abl":
        res = run_abl(cfg, job["variant"], job["seed"])
    elif job["type"] == "evaluate":
        res = run_evaluate(cfg)
    else:
        raise ValueError(f"tipo desconocido: {job['type']}")
    job["result"] = res
    job["status"] = "done"
    job["secs"] = round(time.time() - t, 1)
    save_state(st)
    print(f"[ok] {job['id']} ({job['secs']}s) -> {json.dumps(res, ensure_ascii=False)[:160]}")


def status(st):
    done = [j for j in st["jobs"] if j["status"] == "done"]
    print(f"Progreso: {len(done)}/{len(st['jobs'])} jobs")
    for j in st["jobs"]:
        mark = "✓" if j["status"] == "done" else "·"
        extra = f" {j.get('secs','')}s" if j["status"] == "done" else ""
        print(f"  {mark} {j['id']:<16}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--all", action="store_true", help="corre pendientes hasta agotar budget")
    ap.add_argument("--budget", type=float, default=38.0)
    ap.add_argument("--ppo-steps", type=int, default=30000)
    ap.add_argument("--mappo-steps", type=int, default=30000)
    ap.add_argument("--abl-steps", type=int, default=30000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    if args.init or not os.path.exists(STATE):
        st = init_state(args)
        if args.init:
            return
    st = load_state()

    if args.status:
        status(st); return

    t0 = time.time()
    ran = 0
    while True:
        job = next_pending(st)
        if job is None:
            print("=== PIPELINE COMPLETO ==="); status(st); break
        run_job(st, job); ran += 1
        if not args.all:
            break
        if time.time() - t0 > args.budget:
            print(f"[budget] {args.budget}s agotado tras {ran} jobs; reanuda re-ejecutando.")
            break
    rem = sum(1 for j in st["jobs"] if j["status"] == "pending")
    print(f"Jobs corridos esta llamada: {ran} | pendientes: {rem}")


if __name__ == "__main__":
    main()
