# EXP-1 Robustez — resultados

> **Advertencia:** esta corrida se ejecutó el 2026-09-08 sin
> `data/processed/calibration.json`; usa los placeholders de `gbmarl/config.py`.
> Es un resultado exploratorio de software y no debe presentarse como evidencia
> calibrada de la tesis.

Semillas n=15 · pasos=120000 · métrica TTP-combinado

Nominal MAPPO: mediana=26 d, tasa éxito vs Gatenby(28)=33%


## EXP-1a — Perturbación paramétrica ±20% (mediana TTP MAPPO)

| Parámetro | ×0.8 | ×0.9 | ×1.0 | ×1.1 | ×1.2 |
|---|---|---|---|---|---|
| `ic50_S` | 39 | 32 | 26 | 23 | 20 |
| `ic50_R` | 26 | 26 | 26 | 26 | 26 |
| `delta_max_S` | 21 | 23 | 26 | 29 | 31 |
| `delta_max_R` | 26 | 26 | 26 | 26 | 26 |
| `alpha_S` | 36 | 31 | 26 | 22 | 19 |
| `alpha_R` | 28 | 27 | 26 | 25 | 25 |
| `lambda_c` | 28 | 26 | 26 | 25 | 25 |
| `K` | 11 | 16 | 26 | 37 | 36 |

**Criterio pre-registrado:** NO se cumple globalmente. Celdas que invierten el orden: `ic50_S@1.0`, `ic50_S@1.1`, `ic50_S@1.2`, `ic50_R@0.8`, `ic50_R@0.9`, `ic50_R@1.0`, `ic50_R@1.1`, `ic50_R@1.2`, `delta_max_S@0.8`, `delta_max_S@0.9`, `delta_max_S@1.0`, `delta_max_R@0.8`, `delta_max_R@0.9`, `delta_max_R@1.0`, `delta_max_R@1.1`, `delta_max_R@1.2`, `alpha_S@1.0`, `alpha_S@1.1`, `alpha_S@1.2`, `alpha_R@0.8`, `alpha_R@0.9`, `alpha_R@1.0`, `alpha_R@1.1`, `alpha_R@1.2`, `lambda_c@0.8`, `lambda_c@0.9`, `lambda_c@1.0`, `lambda_c@1.1`, `K@0.8`, `K@0.9`, `K@1.0`

## EXP-1a conjunto — Monte Carlo ±20%

| Entornos | mediana de medianas MAPPO | peor mediana | proporción de entornos con mediana MAPPO > Gatenby del mismo entorno |
|---|---|---|---|
| 50 | 24 | 10 | 38% |

## Sensibilidad del endpoint

| umbral carga | umbral resistencia | mediana MAPPO | Gatenby | éxito |
|---|---|---|---|---|
| 0.70K | 0.40 | 10 | 27 | 0% |
| 0.70K | 0.50 | 10 | 28 | 0% |
| 0.70K | 0.60 | 10 | 29 | 0% |
| 0.80K | 0.40 | 26 | 27 | 33% |
| 0.80K | 0.50 | 26 | 28 | 33% |
| 0.80K | 0.60 | 26 | 29 | 33% |
| 0.90K | 0.40 | 35 | 27 | 100% |
| 0.90K | 0.50 | 41 | 28 | 100% |
| 0.90K | 0.60 | 49 | 29 | 93% |

## EXP-1b — Ruido de observación (gaussiano relativo)

La unidad independiente es la política/semilla; las repeticiones de ruido están anidadas.

| σ | mediana entre políticas | IC bootstrap 95% | [min, max] episodios |
|---|---|---|---|
| 0.00 | 26 | [25.0, 34.0] | [18, 41] |
| 0.05 | 27 | [25.0, 38.0] | [16, 41] |
| 0.10 | 29 | [27.0, 39.0] | [18, 41] |
| 0.20 | 34 | [32.5, 37.0] | [15, 41] |
