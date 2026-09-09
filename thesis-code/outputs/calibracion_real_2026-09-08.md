# Calibración real — corrida 2026-09-08

## Estado

Se descargaron y validaron los archivos oficiales de DepMap Public 26Q1 y GDSC2. Después se ejecutaron:

```bash
python scripts/build_dataset.py
python scripts/calibrate.py
python -m pytest -q
python scripts/evaluate.py
python scripts/evaluate_calibrated_population.py --seeds 15
```

## Integración de datos

| Elemento | Resultado |
|---|---:|
| Líneas GBM en Model.csv | 67 |
| Líneas GBM con ensayos GDSC2 | 34 |
| Ensayos GBM mapeados | 8,072 |
| Registros finales | 5,879 |
| Líneas únicas con expresión completa | 24 |
| Fármacos | 286 |
| Columnas | 19,238 |

## Temozolomide

| Parámetro | Valor | Origen |
|---|---:|---|
| `ic50_S` | 0.360 | GDSC2, p10 de `LN_IC50` centrado en la mediana |
| `ic50_R` | 3.274 | GDSC2, p90 de `LN_IC50` centrado en la mediana |
| Brecha `ic50_R / ic50_S` | 9.1× | derivada |
| `delta_max_S` | 0.300 | ancla de literatura: 2 × `alpha_S` |
| `delta_max_R` | 0.150 | eficacia relativa GDSC2, con piso 0.5 |
| `neto_S_dosis_max` | -0.150 | coherencia del modelo |

La calibración confirma que el fármaco tiene una palanca de control sobre la población sensible. No convierte todos los parámetros del simulador en mediciones clínicas: `alpha`, `K` y `lambda_c`, además del techo de muerte usado como ancla, siguen siendo supuestos de literatura/modelado.

## Evaluación rápida con la calibración nueva

| Estrategia | TTP-carga | TTP-resistencia | Fracción R final |
|---|---:|---:|---:|
| Sin tratamiento | 12 d | 180 d | 0.122 |
| MTD | 180 d | 13 d | 0.515 |
| Gatenby | 180 d | 27 d | 0.526 |
| MAPPO nueva | 180 d | 41 d | 0.504 |

MAPPO se entrenó desde cero con 120,832 pasos mediante el entrenador reanudable, usando la calibración cargada por `load_calibration()`. El checkpoint de terapia quedó en `outputs/models/mappo_real_20260908_therapy.pt`; esta tabla es una evaluación determinista de una semilla. La comparación estadística final entre múltiples semillas se reporta en la sección siguiente.

Los motivos terminales fueron: sin tratamiento, `carga` en el día 12; MTD, `resistencia` en el día 13; Gatenby, `resistencia` en el día 27; MAPPO, `resistencia` en el día 41. Ninguno de estos cuatro episodios terminó por censura del horizonte.

## Validación multi-semilla nominal — 15 semillas

Se entrenaron quince políticas MAPPO y quince IPPO desde cero, todas con 120,832 pasos y la misma calibración. El entrenamiento usó `FastTumorEnv`; la evaluación se hizo con `TumorEnv` y `φ=0.01`, que es el adversario fijo del protocolo.

| Semilla | MAPPO TTP | Modo MAPPO | IPPO TTP | Modo IPPO |
|---:|---:|---|---:|---|
| 0 | 41 | resistencia | 32 | carga |
| 1 | 32 | carga | 27 | carga |
| 2 | 40 | resistencia | 31 | carga |
| 3 | 41 | resistencia | 28 | carga |
| 4 | 40 | resistencia | 23 | carga |
| 5 | 40 | carga | 28 | carga |
| 6 | 27 | carga | 31 | carga |
| 7 | 28 | carga | 34 | carga |
| 8 | 30 | carga | 30 | carga |
| 9 | 29 | carga | 30 | carga |
| 10 | 29 | carga | 32 | carga |
| 11 | 41 | resistencia | 32 | carga |
| 12 | 41 | resistencia | 33 | carga |
| 13 | 34 | carga | 32 | carga |
| 14 | 27 | carga | 27 | carga |

Resumen: MAPPO mediana `34 d` [27, 41], IPPO mediana `31 d` [23, 34]. El contraste Wilcoxon pareado unilateral MAPPO > IPPO da `p=0.013759`; se interpreta junto con la mediana, el rango y los modos de falla, no como evidencia clínica. MAPPO supera a Gatenby (`27 d`) en 13/15 semillas; IPPO, en 12/15.

El detalle reproducible y la prueba estadística están en `outputs/validacion_calibrada_15semillas_2026-09-08.md`; los checkpoints locales son `outputs/models/mappo_real_20260908[_s1.._s14]_{therapy,tumor}.pt` e `outputs/models/ippo_real_20260908_s0..s14_{therapy,tumor}.pt`.

## EXP-2: adversario fortalecido, corrida completa

El protocolo completo exige `φ_max ∈ {0.05, 0.10, 0.20}` × `{MAPPO, IPPO}` × 15 semillas:
90 celdas de 120k pasos, todas ejecutadas con la huella `1fd8464abb97`. Cada política se
evalúa contra el tumor adaptativo co-entrenado de su propia celda y Gatenby se evalúa
contra ese mismo tumor. Por ello, el resultado mide robustez frente a un adversario
endógeno comparable, no una cota matemática de peor caso.

| `φ_max` | MAPPO TTP mediana [min,max] | IPPO TTP mediana [min,max] | Wilcoxon MAPPO>IPPO | Holm | Éxito MAPPO | Éxito IPPO |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 44 [24,58] | 29 [26,37] | 0.005 | 0.014 | 15/15 | 13/15 |
| 0.10 | 28 [22,47] | 27 [23,35] | 0.648 | 0.648 | 15/15 | 14/15 |
| 0.20 | 24 [21,50] | 26 [20,34] | 0.265 | 0.530 | 15/15 | 15/15 |

El modo de falla MAPPO fue carga en 11/15, 15/15 y 13/15 celdas, respectivamente, y
resistencia en 4/15, 0/15 y 2/15. IPPO terminó por carga en las 15 semillas de los tres
regímenes. La lectura principal es que la ventaja de CTDE es dependiente del régimen:
aparece con el adversario nominal acotado (`φ_max=0.05`), pero no se conserva cuando el
rango de acción tumoral aumenta. Esto cuestiona una interpretación universal de MAPPO
y fortalece la validez del análisis de sensibilidad adversarial.

El estado y el detalle por semilla están en `outputs/exp2_adversario_v2.json`; el informe
legible y la figura son `outputs/exp2_adversario_v2.md` y `.png`. Los valores anteriores
de 4/90 celdas se conservaron como `outputs/exp2_adversario_v2_partial_legacy.json` y
no deben mezclarse con esta corrida completa.

## Benchmark V2 de explotabilidad y cross-play

Se completó el benchmark titular con 15 semillas, 120k pasos, cinco reinicios de
best-response por terapia y cross-play poblacional. Gatenby recibió el mismo tratamiento
adversarial con cinco reinicios; su atacante más dañino produjo TTP=7 días. En las
terapias, el mejor atacante produjo MAPPO TTP mediana `14 d` [11,14] e IPPO `11 d`
[10,12]. La explotabilidad, definida como la caída desde self-play hasta best-response,
fue `30 d` [11,44] para MAPPO y `18 d` [14,27] para IPPO; la prueba unilateral
MAPPO<IPPO dio `p=0.993`, sin evidencia de menor explotabilidad de MAPPO.

El cross-play frente a adversarios no vistos dio MAPPO→tumor IPPO `35 d` [24,41] e
IPPO→tumor MAPPO `27 d` [12,29]. Las 15 semillas de ambos métodos superaron los 7
días de Gatenby bajo best-response. El benchmark separa así el nivel absoluto de TTP,
la degradación ante un atacante dedicado y la transferencia fuera del compañero de
self-play; ninguno de estos resultados constituye una garantía de peor caso óptimo.

Los artefactos son `outputs/benchmark_exploit_v2.json`, `.md` y `.png`. El smoke test
anterior queda separado en `outputs/benchmark_exploit_v2_smoke.*`.

## Control de horizonte extendido — 360 días

Para distinguir una contención real de un simple tope administrativo, se repitió la
evaluación representativa con `horizon_days=360`, manteniendo la calibración, el
adversario fijo y la métrica auditada. Los TTP combinados fueron: sin tratamiento 12 d,
MTD 13 d, Gatenby 27 d y MAPPO 41 d. Todos los episodios terminaron por un evento de
fracaso antes del día 360; ninguno fue censurado por alcanzar el horizonte. La figura
queda en `outputs/evaluation_ttp_horizon360.png`.

## Verificación

`26 passed in 1.31s` después de la auditoría, incluyendo las pruebas del lector de DepMap y del manejo de desenlaces.

Además, `FastTumorEnv` y `TumorEnv` se compararon con la misma calibración durante 9 pasos hasta la terminación del episodio: diferencia máxima de estado `2.64e-8`, diferencia máxima de recompensa `1.11e-9` y banderas de terminación idénticas.

La procedencia, los hashes y los tamaños de los archivos fuente están en `data/processed/calibration_provenance.md`.
