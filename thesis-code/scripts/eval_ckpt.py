"""
eval_ckpt.py — Evalúa con TTP-combinado (métrica CORRECTA) todos los checkpoints
del run completo y los baselines. Reporta tasa de éxito (bimodalidad) sin media±std.
"""
import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from gbmarl.tumor_env import TumorEnv
from gbmarl.evalutils import ttp_combinado_detalle, Gatenby
from gbmarl.mappo import Agent, obs_therapy

CKPT = "outputs/ckpt"
env = TumorEnv(horizon_days=180)


def load_pol(job, variant):
    cdim = 3 if variant == "mappo" else 2
    th = Agent(2, cdim, 1, env.action_space("therapy").low, env.action_space("therapy").high)
    with open(f"{CKPT}/{job}.pkl", "rb") as f:
        th.load_state_dict(pickle.load(f)["th"])
    th.eval()
    def pol(s):
        with torch.no_grad():
            a = th.actor_mean(torch.tensor(obs_therapy(s))).clamp(th.a_low, th.a_high)
        return float(a[0])
    return pol


def ev(fn):
    if hasattr(fn, "reset"): fn.reset()
    dia, mot, carga, fracR, dosis = ttp_combinado_detalle(env, fn)
    return int(dia), mot, round(float(dosis), 3)


U = 1.0
out = {"baselines": {}, "mappo": [], "ippo": []}
for n, fn in [("Sin tratamiento", lambda s: 0.0), ("MTD", lambda s: U),
              ("Gatenby", Gatenby())]:
    out["baselines"][n] = ev(fn)

print("== BASELINES ==")
for n, v in out["baselines"].items():
    print(f"  {n:16s} TTP={v[0]:3d}d  {v[1]}")

GATENBY = out["baselines"]["Gatenby"][0]
for variant in ("mappo", "ippo"):
    print(f"\n== {variant.upper()} (120k, 5 semillas) ==")
    for k in range(5):
        job = f"abl_{variant}_{k}"
        dia, mot, dosis = ev(load_pol(job, variant))
        out[variant].append({"seed": k, "ttp": dia, "motivo": mot, "dosis": dosis})
        flag = "WIN" if dia > GATENBY else "   "
        print(f"  s{k}: TTP={dia:3d}d  dosis={dosis:.3f}  {flag}  {mot}")

# Estadística honesta: tasa de éxito (TTP > Gatenby), no media±std
for variant in ("mappo", "ippo"):
    ttps = [r["ttp"] for r in out[variant]]
    wins = sum(1 for t in ttps if t > GATENBY)
    print(f"\n{variant.upper()}: TTPs={sorted(ttps)} | mediana={int(np.median(ttps))}d "
          f"| max={max(ttps)}d | éxito(>Gatenby {GATENBY}d)={wins}/5")

with open("outputs/eval_full.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\n[ok] outputs/eval_full.json")
