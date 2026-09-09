# EXP-1 Robustez — resultados

> **Histórico:** esta salida no corresponde a la auditoría calibrada del
> 2026-09-08. Usa una corrida anterior y no debe mezclarse con
> `exp1_robustez_calibrada_5semillas_2026-09-08.md`.

Semillas n=15 · pasos=120000 · métrica TTP-combinado

Nominal MAPPO: mediana=35 d, tasa éxito vs Gatenby(27)=93%


## EXP-1a — Perturbación paramétrica ±20% (mediana TTP MAPPO)

| Parámetro | ×0.8 | ×0.9 | ×1.0 | ×1.1 | ×1.2 |
|---|---|---|---|---|---|
| `ic50_S` | 39 | 40 | 35 | 27 | 23 |
| `ic50_R` | 35 | 35 | 35 | 35 | 35 |
| `delta_max_S` | 25 | 29 | 35 | 40 | 40 |
| `delta_max_R` | 35 | 35 | 35 | 35 | 35 |
| `alpha_S` | 40 | 39 | 35 | 27 | 23 |
| `alpha_R` | 42 | 36 | 35 | 32 | 31 |
| `lambda_c` | 37 | 35 | 35 | 34 | 33 |
| `K` | 12 | 18 | 35 | 38 | 36 |

## EXP-1b — Ruido de observación (gaussiano relativo)

| σ | TTP mediana | [min, max] |
|---|---|---|
| 0.00 | 35 | [26, 41] |
| 0.05 | 36 | [26, 41] |
| 0.10 | 38 | [23, 41] |
| 0.20 | 37 | [16, 41] |
