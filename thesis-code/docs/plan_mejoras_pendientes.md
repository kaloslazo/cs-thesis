# Plan de continuación económico - 2026-09-10

## Estado y reglas para el siguiente modelo

Trabajar en cs-thesis. Leer CLAUDE.md y revisar git status. No sobrescribir cambios
ajenos. No repetir entrenamientos completos por defecto. Los JSON existentes son
históricos inmutables: guardar nuevas corridas con otro nombre/versión. Nunca elegir
parámetros por hacer ganar a MAPPO. No presentar semillas como pacientes.

Ya corregido: conclusiones y recomendaciones que decían EXP-2 4/90 (es 90/90),
benchmark pendiente (está completo), etiqueta equivocada del EXP-1 vigente,
afirmación de personalización mediante genes en el marco metodológico, y atribución
sin fuente específica de constantes a literatura en la tabla de parámetros.
Se añadió configs/calibration_reference_20260908.json y modo required del cargador;
evaluate_calibrated_population.py exige el archivo y acepta --calibration.
Los valores numéricos, políticas, resultados y dinámica no cambiaron.

## 1. Procedencia de checkpoints y entradas (prioridad alta, costo bajo/medio)

Archivos: gbmarl/config.py, scripts/exp1_robustez.py,
scripts/evaluate_calibrated_population.py, scripts/train_ckpt.py y exportadores.

- EXP-1 prioriza checkpoints mappo_real_20260908* solo por existencia, sin validar
  calibración/presupuesto. El evaluador nominal tampoco verifica su procedencia.
- Definir metadatos compartidos: hash del archivo, parámetros completos, horizonte,
  umbrales, límites de acciones, semilla, arquitectura, pasos solicitados/efectivos,
  versión del código y dependencias.
- Rechazar discrepancias antes de cargar/reutilizar un modelo. Para modelos antiguos,
  aceptar solo procedencia reconstruida a partir de evidencia disponible; no inventar
  metadatos de entrenamiento basándose en el nombre o en la configuración actual.
- Extender exigencia de calibración a corridas oficiales EXP-1, EXP-2 y benchmark.
  Conservar modo exploratorio explícito y --plot desde JSON sin datos locales.

Aceptación: tests para archivo ausente, metadatos ausentes/incompatibles, modelo
válido y --plot sin calibración. Cambiar horizonte o calibración debe impedir reutilizar
el checkpoint. No iniciar entrenamiento como fallback durante estos tests.

## 2. Auditoría documental y fuentes (costo medio, requiere investigación)

- Buscar fuentes primarias con valor, unidad y contexto para alpha, lambda_c, Hill,
  KILL_TO_GROWTH_RATIO=2 y piso resistente=0.5. Sin evidencia, mantener SUPUESTO.
- Sincronizar comentarios antiguos de calibrate.py/config.py, capítulo 4 y reportes
  con las limitaciones ahora descritas en capítulo 3. Los reportes históricos se
  preservan y se anotan; no reescribir cifras.
- Auditar cualquier afirmación de genómica personalizada, inevitabilidad de
  resistencia, validación clínica o prerregistro. Llamar prerregistro solo si existe
  un protocolo fechado antes de las corridas correspondientes.
- Distinguir dosis/concentración normalizada y tiempo del simulador de magnitudes
  clínicas. El chequeo alpha-delta_max evalúa saturación, no dosis finita.

Aceptación: tabla parámetro -> fuente puntual/unidad o supuesto; ninguna cita
inventada. Corregir todos los textos vigentes afectados y compilar/revisar el PDF.

## 3. Incertidumbre sin reentrenar (costo bajo)

Consumir JSON nominal y EXP-2. Unir por semilla explícita y rechazar pares faltantes
o duplicados. Bootstrap de pares completos, semilla RNG fija, 10000 remuestreos.
Reportar IC95% de una diferencia definida previamente (p. ej. media de diferencias
pareadas), separada de la diferencia de medianas. Conservar los contrastes originales
y Holm para EXP-2. No mezclar familias ni convertir p no significativo en equivalencia.
Si hay censura, no tratar el horizonte como tiempo exacto de fracaso; detener el
análisis simple y documentar la necesidad de un estimador adecuado.

Aceptación: nuevo JSON/Markdown con método, n, semillas, estimando e intervalos;
test de emparejamiento y ejemplo sintético conocido. Sin modificar resultados fuente.

## 4. Comparaciones adicionales (costo medio/alto, posponer)

Primero acordar un presupuesto concreto. Añadir dosis constante ajustada y Gatenby
con umbrales ajustados en escenarios de desarrollo. Reservar escenarios/adversarios
para evaluación común de todos los métodos. Mantener la misma información disponible.
Medir TTP, modo, dosis y dispersión por semilla. Conservar evaluación contra el
compañero de self-play como análisis separado. No ajustar usando los casos de prueba.

Aceptación: protocolo escrito antes del ajuste, separación desarrollo/evaluación,
configuraciones y comandos guardados, resultados incluso si favorecen baselines.

## 5. Salud/toxicidad (costo alto, depende de datos)

Mantener extensión experimental. Antes de EXP-3 clínico definir variable observable,
unidad, dinámica de recuperación, datos longitudinales, ajuste y validación externa.
No rellenar constantes por conveniencia. No es requisito para cerrar 1-3.

## Verificación final

Desde thesis-code: ../.venv/bin/python -m pytest -q; git diff --check.
Si cambia LaTeX, compilar dos veces y revisar las páginas cambiadas. Documentar
qué fue probado y qué sigue pendiente. No afirmar «100% listo» por pasar tests.
