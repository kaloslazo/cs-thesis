"""
calibrate.py — Deriva los parámetros de EFECTO DEL FÁRMACO desde el dataset real
(Temozolomida) y los guarda en data/processed/calibration.json.

Ejecutar desde la raíz:  python scripts/calibrate.py

QUÉ SE CALIBRA AQUÍ (de DATOS / GDSC2):
  · ic50_S, ic50_R         : sensibilidad de las subpoblaciones S y R.
  · delta_max_S, delta_max_R : muerte máxima por fármaco (de la eficacia / AUC).

QUÉ NO (queda como LITERATURA en config.py):
  · alpha_S, alpha_R, K, lambda_c  (el dataset no tiene info temporal).

POR QUÉ SOLO TEMOZOLOMIDA, SIN PERDER DATA:
  El agente administra TMZ, así que el entorno se calibra a TMZ. Pero usamos
  TODOS los registros de TMZ en las 24 líneas: la DISPERSIÓN entre líneas define
  la brecha S vs R. Sensible = subpoblación de IC50 bajo; Resistente = IC50 alto.
"""
import json
import os
import sys
import numpy as np
import pandas as pd

DATASET = "data/processed/dataset_marl_gbm_completo.csv"   # ajusta si lo moviste a processed/
OUT = "data/processed/calibration.json"
DRUG = "Temozolomide"

# Constante de literatura: tasa máxima de muerte celular inducida por fármaco (1/día).
# Mapea la eficacia (AUC) a una tasa biológica. Documentada, reproducible.
MAX_KILL_RATE = 0.5


def main():
    if not os.path.exists(DATASET):
        sys.exit(f"ERROR: no existe {DATASET} (corre build_dataset.py primero).")

    df = pd.read_csv(DATASET, usecols=lambda c: c in (
        "DRUG_NAME", "CELL_LINE_NAME", "ModelID", "LN_IC50", "AUC", "Z_SCORE"))

    tmz = df[df["DRUG_NAME"].str.contains(DRUG, case=False, na=False)].copy()
    if len(tmz) == 0:
        sys.exit(f"ERROR: no hay registros de {DRUG} en el dataset.")

    # Un valor por línea celular (evita contar réplicas del mismo modelo varias veces)
    tmz = tmz.groupby("ModelID", as_index=False).agg(
        LN_IC50=("LN_IC50", "mean"), AUC=("AUC", "mean"))

    ln = tmz["LN_IC50"]
    med = float(ln.median())
    p10, p90 = float(ln.quantile(0.10)), float(ln.quantile(0.90))

    # --- IC50 de S y R ---
    # Normalizamos: la mediana de TMZ = concentración de referencia (1.0).
    # exp(p10 - mediana) < 1  -> subpoblación sensible (muere a baja dosis)
    # exp(p90 - mediana) > 1  -> subpoblación resistente (necesita más dosis)
    ic50_S = float(np.exp(p10 - med))
    ic50_R = float(np.exp(p90 - med))

    # --- delta_max de S y R (de la eficacia / AUC) ---
    # AUC alto = la célula sobrevive = poca muerte. Eficacia ~ (1 - AUC).
    auc_p10, auc_p90 = float(tmz["AUC"].quantile(0.10)), float(tmz["AUC"].quantile(0.90))
    delta_max_S = round(MAX_KILL_RATE * (1 - auc_p10), 3)   # sensible responde fuerte
    delta_max_R = round(MAX_KILL_RATE * (1 - auc_p90), 3)   # resistente apenas responde

    calib = {
        "drug": DRUG,
        "n_lines": int(len(tmz)),
        "raw_stats": {
            "ln_ic50_median": round(med, 4),
            "ln_ic50_p10": round(p10, 4),
            "ln_ic50_p90": round(p90, 4),
            "auc_p10": round(auc_p10, 4),
            "auc_p90": round(auc_p90, 4),
        },
        "derived_params": {
            "ic50_S": round(ic50_S, 4),
            "ic50_R": round(ic50_R, 4),
            "delta_max_S": delta_max_S,
            "delta_max_R": delta_max_R,
        },
        "provenance": {
            "ic50_S/ic50_R": "GDSC2 TMZ LN_IC50, percentiles p10/p90 normalizados a la mediana",
            "delta_max_*": f"MAX_KILL_RATE({MAX_KILL_RATE}) * (1 - AUC_percentil)",
            "alpha/K/lambda_c": "NO se calibran aquí; literatura (config.py)",
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"CALIBRACIÓN — {DRUG}")
    print("=" * 60)
    print(f"  Líneas GBM con datos de TMZ: {calib['n_lines']}")
    print(f"  LN_IC50  mediana={med:.3f}  p10={p10:.3f}  p90={p90:.3f}")
    print(f"  -> ic50_S = {ic50_S:.3f}   ic50_R = {ic50_R:.3f}")
    print(f"  -> delta_max_S = {delta_max_S}   delta_max_R = {delta_max_R}")
    print(f"  Guardado en {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()