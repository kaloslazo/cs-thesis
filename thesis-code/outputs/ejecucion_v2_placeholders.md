# Registro de ejecución V2 — exploratorio

**Fecha:** 2026-09-08
**Configuración:** sin `data/processed/calibration.json`; se usaron los placeholders
de `gbmarl/config.py`. Estos números sirven para inspección del comportamiento del
código y no son resultados finales de la tesis.

## EXP-1 — completo

Comando: `scripts/exp1_robustez.py --seeds 15 --steps 120000`

- Políticas nominales entrenadas: 15/15.
- TTP nominal MAPPO: mediana **26 días**.
- Gatenby nominal en esta configuración: **28 días**.
- Monte Carlo conjunto ±20%: 50 entornos; mediana de medianas MAPPO **24 días**;
  peor mediana **10 días**; MAPPO supera a Gatenby en **38%** de los entornos.
- Ruido de observación: medianas para σ=0.00, 0.05, 0.10 y 0.20:
  **26, 27, 29 y 34 días**, respectivamente.
- Sensibilidad del endpoint: con umbral de carga 0.70K la mediana fue **10 días**;
  con 0.80K fue **26 días**; con 0.90K fue **35–49 días** según el umbral de
  resistencia. Esto confirma que la definición del endpoint cambia fuertemente la
  lectura del desempeño.

Detalle completo: `outputs/exp1_robustez_v2.md` y
`outputs/exp1_robustez_v2.json`.

## EXP-2 — parcial

Comando iniciado: `scripts/exp2_adversario.py --seeds 15 --steps 120000`.

Se completaron **3/90 celdas**, todas MAPPO con `phi_max=0.05`:

| semilla | TTP MAPPO | modo de falla | Gatenby mismo tumor | MTD mismo tumor |
|---:|---:|---|---:|---:|
| 0 | 49 | carga | 6 | 6 |
| 1 | 61 | resistencia | 6 | 6 |
| 2 | 24 | carga | 7 | 6 |

La corrida fue detenida para no consumir horas adicionales con parámetros no
calibrados. El estado parcial está en `outputs/exp2_adversario_v2.json` y puede
reanudar la primera celda pendiente cuando exista la calibración real.

## Benchmark y EXP-3

Se validaron previamente con smoke tests, pero no se ejecutó una corrida completa
válida de tesis. EXP-3 requiere además un JSON externo con parámetros de toxicidad y
fuente documentada; no existe todavía en el workspace.

## Conclusión de esta ejecución

El software y el protocolo V2 funcionan, y EXP-1 muestra que la sensibilidad al
endpoint y a `K` es material. No se debe concluir todavía que MAPPO supera a IPPO ni
que protege al paciente: faltan la calibración real, EXP-2 completo, el benchmark
completo y la configuración de toxicidad justificada.
