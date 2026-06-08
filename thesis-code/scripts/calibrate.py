"""
calibrate.py — Deriva parámetros del fármaco (TMZ) combinando DATOS + LITERATURA.
Ejecutar desde la raíz:  python scripts/calibrate.py

PRINCIPIO (clave para la defensa): cada fuente responde solo lo que sabe bien.
  · DATOS (GDSC2)  -> POTENCIA: ic50_S, ic50_R (cuánta dosis hace falta) y la
                       razón resistente/sensible. El ensayo es bueno para esto.
  · LITERATURA     -> TECHO de muerte (delta_max): cuánto puede matar el fármaco
                       en la escala de meses del tratamiento. El ensayo de 72h
                       NO puede decir esto (TMZ actúa por metilación, lento).
  · ACOTADO        -> reducción de eficacia de R, informada por datos pero con
                       piso, para que nunca quede biológicamente imposible.

Coherencia garantizada: delta_max se ancla al crecimiento (alpha), así el
fármaco SIEMPRE es una palanca controlable, sin importar el valor de alpha.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from gbmarl.config import Params

DATASET = "data/processed/dataset_marl_gbm_completo.csv"
OUT = "data/processed/calibration.json"
DRUG = "Temozolomide"

# --- Constantes de LITERATURA (documentadas, ajustables) ---
# A dosis saturante sostenida, un alquilante puede ~duplicar la tasa de pérdida
# frente al crecimiento intrínseco. Garantiza que el fármaco supere el crecimiento.
KILL_TO_GROWTH_RATIO = 2.0
# Piso de eficacia de las resistentes (no pueden volverse 100% inmortales).
MIN_EFFICACY_R = 0.5


def main():
    if not os.path.exists(DATASET):
        sys.exit(f"ERROR: no existe {DATASET} (corre build_dataset.py primero).")

    df = pd.read_csv(DATASET, low_memory=False,
                     usecols=lambda c: c in ("DRUG_NAME", "ModelID", "LN_IC50", "AUC"))
    tmz = df[df["DRUG_NAME"].str.contains(DRUG, case=False, na=False)].copy()
    if len(tmz) == 0:
        sys.exit(f"ERROR: no hay registros de {DRUG}.")
    tmz = tmz.groupby("ModelID", as_index=False).agg(
        LN_IC50=("LN_IC50", "mean"), AUC=("AUC", "mean"))

    p = Params()  # para leer alpha (literatura)

    # ── 1. POTENCIA (de DATOS): IC50 de S y R ──────────────────────────────────
    ln = tmz["LN_IC50"]
    med = float(ln.median())
    p10, p90 = float(ln.quantile(0.10)), float(ln.quantile(0.90))
    ic50_S = float(np.exp(p10 - med))   # subpoblación sensible (IC50 bajo)  -> <1
    ic50_R = float(np.exp(p90 - med))   # subpoblación resistente (IC50 alto) -> >1

    # ── 2. TECHO DE MUERTE (de LITERATURA): anclado al crecimiento ─────────────
    delta_max_S = round(KILL_TO_GROWTH_RATIO * p.alpha_S, 3)

    # ── 3. EFICACIA DE R (de DATOS, acotada): reducción del techo para R ───────
    eff_S = 1 - float(tmz["AUC"].quantile(0.10))    # eficacia en líneas sensibles
    eff_R = 1 - float(tmz["AUC"].quantile(0.90))    # eficacia en líneas resistentes
    eff_ratio = eff_R / eff_S if eff_S > 0 else 1.0
    eff_ratio = float(np.clip(eff_ratio, MIN_EFFICACY_R, 1.0))   # piso de seguridad
    delta_max_R = round(delta_max_S * eff_ratio, 3)

    # ── Verificación de coherencia ──────────────────────────────────────────────
    neto_S = p.alpha_S - delta_max_S      # debe ser < 0 (sensibles controlables)
    neto_R = p.alpha_R - delta_max_R      # idealmente < 0 a dosis alta
    palanca_ok = neto_S < 0

    calib = {
        "drug": DRUG,
        "n_lines": int(len(tmz)),
        "raw_stats": {"ln_ic50_median": round(med, 4),
                      "ln_ic50_p10": round(p10, 4), "ln_ic50_p90": round(p90, 4),
                      "eff_sensible": round(eff_S, 4), "eff_resistente": round(eff_R, 4)},
        "derived_params": {"ic50_S": round(ic50_S, 4), "ic50_R": round(ic50_R, 4),
                           "delta_max_S": delta_max_S, "delta_max_R": delta_max_R},
        "coherencia": {"neto_S_dosis_max": round(neto_S, 3),
                       "neto_R_dosis_max": round(neto_R, 3),
                       "hay_palanca": bool(palanca_ok)},
        "provenance": {
            "ic50_S/ic50_R": "DATOS: GDSC2 TMZ LN_IC50, percentiles p10/p90 vs mediana",
            "delta_max_S": f"LITERATURA: {KILL_TO_GROWTH_RATIO} x alpha_S",
            "delta_max_R": f"delta_max_S x eficacia_R (datos, piso {MIN_EFFICACY_R})",
            "alpha/K/lambda_c": "LITERATURA (config.py)"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(calib, open(OUT, "w"), indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"CALIBRACIÓN — {DRUG}   ({calib['n_lines']} líneas)")
    print("=" * 60)
    print(f"  POTENCIA (datos):   ic50_S={ic50_S:.3f}   ic50_R={ic50_R:.3f}   (gap {ic50_R/ic50_S:.1f}x)")
    print(f"  TECHO   (literat.): delta_max_S={delta_max_S}   delta_max_R={delta_max_R}")
    print(f"  COHERENCIA: neto_S a dosis máx = {neto_S:+.3f}  ->  ", end="")
    print("✓ HAY PALANCA" if palanca_ok else "✗ SIN PALANCA (sube KILL_TO_GROWTH_RATIO)")
    print(f"  Guardado en {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()