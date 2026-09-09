# EXP-1 Robustez — resultados

Semillas n=15 · pasos=120000 · métrica TTP-combinado

Nominal MAPPO: mediana=34 d, tasa éxito vs Gatenby(27)=87%


## EXP-1a — Perturbación paramétrica ±20% (mediana TTP MAPPO)

| Parámetro | ×0.8 | ×0.9 | ×1.0 | ×1.1 | ×1.2 |
|---|---|---|---|---|---|
| `ic50_S` | 38 | 38 | 34 | 29 | 24 |
| `ic50_R` | 34 | 35 | 34 | 34 | 34 |
| `delta_max_S` | 26 | 31 | 34 | 39 | 39 |
| `delta_max_R` | 33 | 34 | 34 | 35 | 35 |
| `alpha_S` | 40 | 38 | 34 | 30 | 26 |
| `alpha_R` | 36 | 37 | 34 | 33 | 33 |
| `lambda_c` | 37 | 35 | 34 | 36 | 36 |
| `K` | 25 | 25 | 34 | 33 | 30 |

**Criterio pre-registrado:** NO se cumple globalmente. Celdas que invierten el orden: `ic50_S@1.2`, `delta_max_S@0.8`, `alpha_S@1.2`, `K@0.8`, `K@0.9`

## EXP-1a conjunto — Monte Carlo ±20%

| Entornos | mediana de medianas MAPPO | peor mediana | proporción de entornos con mediana MAPPO > Gatenby del mismo entorno |
|---|---|---|---|
| 50 | 30 | 15 | 58% |

## Sensibilidad del endpoint

| umbral carga | umbral resistencia | mediana MAPPO | Gatenby | éxito |
|---|---|---|---|---|
| 0.70K | 0.40 | 17 | 25 | 7% |
| 0.70K | 0.50 | 17 | 27 | 0% |
| 0.70K | 0.60 | 17 | 29 | 0% |
| 0.80K | 0.40 | 33 | 25 | 100% |
| 0.80K | 0.50 | 34 | 27 | 87% |
| 0.80K | 0.60 | 34 | 29 | 67% |
| 0.90K | 0.40 | 36 | 25 | 100% |
| 0.90K | 0.50 | 41 | 27 | 100% |
| 0.90K | 0.60 | 48 | 29 | 100% |

## EXP-1b — Ruido de observación (gaussiano relativo)

La unidad independiente es la política/semilla; las repeticiones de ruido están anidadas.

| σ | mediana entre políticas | IC bootstrap 95% | [min, max] episodios |
|---|---|---|---|
| 0.00 | 34 | [29.0, 41.0] | [27, 41] |
| 0.05 | 36 | [29.0, 40.0] | [26, 41] |
| 0.10 | 37 | [31.0, 39.0] | [24, 41] |
| 0.20 | 36 | [35.0, 37.0] | [17, 41] |
