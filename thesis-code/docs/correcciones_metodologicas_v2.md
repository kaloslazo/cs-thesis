# Correcciones metodológicas V2 — estado de implementación

Este documento registra cambios posteriores a los resultados publicados en
`outputs/*.json`. Los números históricos **no se recalculan automáticamente** y no
deben presentarse como resultados V2.

## 1. Endpoint

- Clasificación centralizada en `gbmarl/outcomes.py`.
- Modos: carga, resistencia, ambos, toxicidad, múltiples, extinción y horizonte.
- `horizon` es censura administrativa; `extinct` es éxito observado.
- La vista escalar asigna el horizonte a ambos éxitos para mantener compatibilidad.
- Se registran tiempos componentes `t_load` y `t_resistance`.
- Tests específicos cubren carga, resistencia, simultaneidad, horizonte y extinción.

## 2. EXP-1 V2

- Estado y salidas nuevos `exp1_*_v2`; se preservan los artefactos anteriores.
- Monte Carlo conjunto ±20% implementado.
- Sensibilidad 3×3 de umbral de carga y resistencia.
- Controles de `K` con condiciones iniciales escaladas y endpoint absoluto fijo.
- Éxito contra Gatenby evaluado en el mismo entorno perturbado.
- Ruido resumido a nivel de política, con bootstrap entre semillas.
- El reporte enumera explícitamente las condiciones que violan el criterio pre-registrado.

## 3. EXP-2 V2

- Gatenby y MTD se evalúan contra el mismo tumor aprendido de cada celda.
- Comparación MAPPO–IPPO pareada por semilla mediante Wilcoxon.
- Corrección Holm para los tres regímenes de `phi_max`.
- Salidas V2 separadas de los resultados históricos.

## 4. Benchmark de explotabilidad V2

- El atacante recibe una recompensa directamente alineada con minimizar TTP.
- Cinco reinicios por defecto para cada terapia.
- El partner de self-play se incluye como candidato; explotabilidad >= 0.
- Gatenby se evalúa contra su propia mejor respuesta aproximada con igual presupuesto;
  no se reutiliza el valor nominal de 27 días como umbral bajo otro adversario.
- Cross-play completo todos-contra-todos entre poblaciones de tumores.
- Contrastes pareados y McNemar exacto para tasa de éxito.
- El resultado se denomina “best-response aproximada”; no se afirma optimalidad.

## 5. Salud/toxicidad

- Implementación opt-in separada en `PatientTumorEnv`.
- Toxicidad acumulada depende de concentración y recuperación.
- Todos los parámetros clínicos/funcionales son obligatorios; no hay defaults ni tercer agente.
- El experimento exige fuente y configuración externa con hash.
- La ablación `health_aware` frente a `health_blind` se evalúa en el mismo entorno
  4-D; el reporte conserva TTP, salud, toxicidad, dosis y modos de falla por separado.

## Estado de evidencia

Los cambios de código deben pasar pruebas y corridas smoke. Después se deben ejecutar
las corridas completas V2 antes de reemplazar tablas, figuras o afirmaciones del
Capítulo 4. Hasta entonces, los resultados publicados siguen siendo V1.
