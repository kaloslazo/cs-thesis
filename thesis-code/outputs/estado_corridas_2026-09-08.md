# Estado de corridas de tesis — 2026-09-08

## Corridas verificadas con la calibración real

- Integración DepMap 26Q1 + GDSC2 y calibración TMZ: [calibracion_real_2026-09-08.md](calibracion_real_2026-09-08.md).
- Comparación nominal MAPPO/IPPO con 15 semillas: mediana MAPPO 34 días [27,41], mediana IPPO 31 días [23,34], `p=0.013759` en Wilcoxon pareado unilateral: [validacion_calibrada_15semillas_2026-09-08.md](validacion_calibrada_15semillas_2026-09-08.md).
- EXP-1a/EXP-1b recalculado con 15 políticas calibradas: [exp1_robustez_v2.md](exp1_robustez_v2.md).
- EXP-2 está completo: 90/90 celdas, 15 semillas por método en `φ_max={0.05,0.10,0.20}`. MAPPO supera a IPPO con `φ_max=0.05` (44 vs. 29 días; Holm `p=0.014`), pero no en `0.10` ni `0.20`; [reporte](exp2_adversario_v2.md).
- Benchmark V2 de explotabilidad completo: 15/15 semillas, 5 reinicios de best-response por terapia y cross-play. Gatenby bajo su propia best-response obtuvo 7 días; MAPPO obtuvo best-response 14 [11,14] e IPPO 11 [10,12], con cross-play MAPPO→tumor IPPO 35 [24,41] e IPPO→tumor MAPPO 27 [12,29]: [reporte](benchmark_exploit_v2.md).
- Suite de regresión: `26 passed`.
- Integración/calibración reproducida nuevamente desde los archivos crudos: 67 líneas GBM, 34 con ensayos, 24 con expresión completa, 5,879 registros, `ic50_S=0.360`, `ic50_R=3.274`, brecha `9.1x` y `neto_S=-0.150`.
- Evaluación nominal reproducida: sin tratamiento 12 días, MTD 13, Gatenby 27 y MAPPO 41; los motivos de falla se mantienen auditados.
- Control de horizonte extendido ejecutado a 360 días: los cuatro episodios representativos fallan antes del horizonte (12, 13, 27 y 41 días), sin censura administrativa.
- Tesis LaTeX compilada en `thesis-latex/main.pdf` (75 páginas), sin referencias/citas indefinidas ni placeholders. La inspección visual de las páginas críticas pasó; quedan únicamente advertencias menores heredadas de maquetación del template y figuras previas.

## Artefactos que no deben mezclarse

- `exp1_robustez_v2.md/json`: corrida exploratoria previa con placeholders; el propio reporte la marca como no calibrada.
- `exp2_adversario_v2.json`: corrida completa actual, con huella de calibración y 90/90 celdas.
- `benchmark_exploit_v2.json`: corrida titular completa con huella `1fd8464abb97`, 15 filas y 5 reinicios por terapia.
- `exp2_adversario_v2_partial_legacy.json`: estado parcial antiguo, separado para conservar trazabilidad; no es citable.
- `exp2_adversario.md`, `benchmark_exploit.md` y `RESULTADOS_pipeline.md`: resultados históricos de otras corridas/configuraciones. No se deben citar como replicación de la corrida real del 2026-09-08 sin volver a ejecutarlos con la huella `1fd8464abb97`.

Los modelos nuevos están en `outputs/models/` y no deben reutilizarse si cambia `calibration.json`, `phi_max`, el horizonte, el endpoint o la arquitectura.
