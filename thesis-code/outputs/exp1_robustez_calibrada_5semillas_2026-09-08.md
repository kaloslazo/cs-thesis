# EXP-1 Robustez calibrada — cinco semillas

> **Nota de vigencia:** este archivo conserva un subconjunto temprano de cinco
> semillas. El resultado vigente de EXP-1 usa quince políticas y está en
> `outputs/exp1_robustez_v2.md`; no mezclar sus cifras con este subconjunto.

Fecha: 2026-09-08. Calibración: `1fd8464abb97`.

Se evaluaron las cinco políticas MAPPO recién entrenadas, sin reentrenarlas bajo cada perturbación. El adversario fijo fue `φ=0.01`. La unidad independiente es la semilla; para ruido se usaron 30 episodios por política y σ, excepto σ=0.

## EXP-1a: perturbación OAT ±20%

Valores reportados: mediana TTP-combinado sobre las cinco semillas, en factores ×0.8, ×0.9, ×1.0, ×1.1 y ×1.2.

| Parámetro | ×0.8 | ×0.9 | ×1.0 | ×1.1 | ×1.2 |
|---|---:|---:|---:|---:|---:|
| `ic50_S` | 38 | 39 | 40 | 34 | 25 |
| `ic50_R` | 41 | 41 | 40 | 40 | 40 |
| `delta_max_S` | 29 | 37 | 40 | 40 | 39 |
| `delta_max_R` | 40 | 40 | 40 | 41 | 41 |
| `alpha_S` | 40 | 40 | 40 | 39 | 34 |
| `alpha_R` | 44 | 42 | 40 | 39 | 38 |
| `lambda_c` | 40 | 40 | 40 | 41 | 40 |
| `K` | 11 | 19 | 40 | 38 | 37 |

La fragilidad principal aparece en `K`: al reducirlo 20%, la mediana cae de 40 a 11 días. También hay degradación clara cuando `ic50_S` aumenta 20% y cuando `alpha_S` aumenta 20%. Esto es una sensibilidad del modelo y del endpoint, no evidencia clínica.

## Perturbación conjunta

Se evaluaron 50 entornos, con factores independientes uniformes en [0.8, 1.2] para los ocho parámetros.

- Mediana de las medianas MAPPO: `36 d`.
- Peor mediana entre entornos: `10 d`.
- Entornos donde la mediana MAPPO superó a Gatenby: `66%`.
- Entornos donde ni siquiera la mayoría de las cinco políticas superó a Gatenby: algunos casos (`tasa mínima 0%`).

## Sensibilidad del endpoint

| Umbral de carga | Umbral de resistencia | Mediana MAPPO | Gatenby | MAPPO > Gatenby |
|---:|---:|---:|---:|---:|
| 0.70K | 0.40 | 11 | 25 | 0/5 |
| 0.70K | 0.50 | 11 | 27 | 0/5 |
| 0.70K | 0.60 | 11 | 29 | 0/5 |
| 0.80K | 0.40 | 33 | 25 | 5/5 |
| 0.80K | 0.50 | 40 | 27 | 5/5 |
| 0.80K | 0.60 | 42 | 29 | 5/5 |
| 0.90K | 0.40 | 34 | 25 | 5/5 |
| 0.90K | 0.50 | 41 | 27 | 5/5 |
| 0.90K | 0.60 | 48 | 29 | 5/5 |

Esto demuestra que la conclusión depende materialmente del umbral de carga: con 0.70K el episodio termina por carga antes de que la política pueda mostrar su ventaja. Por eso el umbral debe justificarse y mantenerse fijo antes de comparar métodos.

## EXP-1b: ruido relativo de observación

| σ | Mediana entre políticas | Rango de todos los episodios |
|---:|---:|---:|
| 0.00 | 40 | 32–41 |
| 0.05 | 40 | 30–41 |
| 0.10 | 38 | 24–41 |
| 0.20 | 37 | 24–40 |

## Interpretación y límite

La política es robusta frente a cambios pequeños de potencia, eficacia resistente, crecimiento y eliminación del fármaco, pero no frente a una reducción fuerte de `K` ni a una combinación adversa de parámetros. La muestra actual es `n=5`; sirve para validar el pipeline y detectar fragilidades, pero no reemplaza la corrida pre-registrada de 15 semillas.

Los archivos `outputs/exp1_robustez_v2.*` existentes son resultados exploratorios con placeholders y no deben combinarse con este reporte.
