# Informe de avance — PFC II (Semanas 1–4)

> **Estado documental (2026-09-08):** este informe conserva las cifras de la
> corrida histórica de las semanas 1--4. Sus valores de $n=15$, EXP-2 y tasas de
> éxito no corresponden automáticamente a la calibración vigente DepMap 26Q1 +
> GDSC2 release 8.5. Para las cifras actualmente verificadas, consultar
> `outputs/calibracion_real_2026-09-08.md`,
> `outputs/validacion_calibrada_15semillas_2026-09-08.md` y
> `outputs/exp1_robustez_v2.md` y
> `outputs/estado_corridas_2026-09-08.md`.

**Tesis:** Mitigación de la resistencia evolutiva en glioblastoma multiforme mediante
aprendizaje por refuerzo multiagente adversarial (marco **GBMARL**, MAPPO-CTDE).
**Autores:** Kalos B. Lazo Mera · Gianpier A. Segovia Ureta
**Asesor:** Victor E. Martínez Abaunza · **Ciclo:** 2026-2 (CS5011)
**Alcance de este informe:** cronograma PFC II, Semanas 1 a 4 (línea base + 2 experimentos nuevos).

---

## 1. Resumen ejecutivo (para la reunión)

En PFC I dejamos el marco GBMARL funcionando y validado: modelamos la terapia con
temozolomida y la resistencia del glioblastoma como un **juego adversarial de dos agentes**
resuelto por RL multiagente (MAPPO-CTDE) sobre un simulador de EDO calibrado con datos reales
(GDSC2 + DepMap). El hallazgo central: la política aprendida **retrasa la intratabilidad**
(~33 días de TTP) frente a MTD (13 d) y a la heurística de Gatenby (27 d).

En PFC II, estas primeras 4 semanas se centraron en **darle solidez experimental** al hallazgo:

1. **Congelamos y verificamos la línea base** (reproducibilidad + tag `v1.0`).
2. **Escribimos el protocolo experimental** con criterios de éxito pre-registrados.
3. **EXP-1 (Robustez):** ¿la política aguanta incertidumbre en los parámetros y ruido en las
   mediciones clínicas? → **Sí al ruido de observación; parcialmente a la perturbación de
   parámetros** (identificamos una fragilidad honesta ante la capacidad de carga `K`).
4. **EXP-2 (Adversario fuerte):** ¿cuándo aporta el crítico centralizado (CTDE) frente al
   aprendizaje independiente (IPPO)? → **CTDE gana claramente cuando el adversario es más
   fuerte**, medido por tasa de éxito (53% vs 7% con el adversario más agresivo).

Los dos experimentos usan **n=15 semillas** y la métrica honesta TTP-combinado. Todo es
reproducible y reanudable.

---

## 2. Contexto: qué es GBMARL (recordatorio de 1 minuto)

- **Simulador (entorno):** competencia Lotka–Volterra de dos poblaciones tumorales —sensibles
  `S` y resistentes `R`— más la farmacocinética del fármaco `c`, integrado con Runge–Kutta 4:

  ```
  dS/dt = α_S·S·(1 − (S+R)/K) − δ_S(c)·S − φ·S
  dR/dt = α_R·R·(1 − (S+R)/K) − δ_R(c)·R + φ·S
  dc/dt = −λ_c·c + u(t)
  ```

- **Dos agentes en self-play:**
  - *Terapia:* elige la dosis `u(t)`; objetivo = maximizar los días con tumor manejable.
  - *Tumor:* elige la transición fenotípica `φ` (S→R, acotada por `phi_max`); objetivo = forzar
    la dominancia resistente.
- **Calibración:** los datos GDSC2/DepMap fijan la potencia del fármaco (IC50_S≈0.36, IC50_R≈3.27,
  brecha ≈9×; 34 líneas GBM recuperadas). La literatura ancla el techo de muerte.
- **Métrica honesta (TTP-combinado):** días hasta que el tratamiento falla por **carga**
  (el tumor crece más allá del umbral) **o** por **resistencia** (los resistentes son mayoría),
  reportando *cuál* de las dos causó la falla. Evita la métrica engañosa que solo mira
  resistencia e infla resultados.

---

## 3. Lo que se hizo, semana por semana

### Semana 1 — Congelar y verificar la línea base

**Objetivo (cronograma):** congelar código+datos de PFC I (tag `v1.0`), verificar reproducibilidad
end-to-end, definir el protocolo.

**Qué encontramos e hicimos:**
- El proyecto no tenía un entorno de ejecución reproducible en la máquina. **Reconstruimos el
  entorno** (`.venv` con Python 3.12) y **creamos `requirements.txt`** con las versiones exactas
  de todas las dependencias (torch, pettingzoo, gymnasium, numpy, scipy, matplotlib, pandas).
- **Suite de tests: 10/10 en verde.**
- **Reproducibilidad verificada:** misma semilla → resultado idéntico (el entrenamiento fija
  hilo único + algoritmos deterministas + sembrado completo).
- **Replicación del hallazgo:** los baselines reproducen exactamente el póster
  (Sin tratar 12 d / MTD 13 d / Gatenby 27 d) y MAPPO multi-semilla da **mediana 32 / mejor 41 d,
  4/4 vs Gatenby**.
- **Tag `v1.0`** creado como punto de partida inmutable de PFC II.

**Observación importante (honestidad):** el modelo "titular" guardado (semilla 0) había caído en
una **cuenca de convergencia mala** (subtrata → falla al día 13). Esto **confirma la bimodalidad**
del self-play ya documentada y justifica por qué reportamos por **tasa de éxito y mediana**, no por
promedio.

**Entregable:** repo taggeado `v1.0` + corrida de replicación exitosa.

### Semana 2 — Protocolo experimental escrito

**Objetivo:** diseño experimental de robustez y del adversario fortalecido, aprobado por el asesor.

Redactamos `docs/protocolo_experimental_pfc2.md`, que **pre-registra** (antes de correr, para
evitar sesgo post-hoc):
- Qué se perturba y en qué rango (±20%), cómo se inyecta el ruido de observación, y la definición
  del adversario fuerte (barrido `phi_max` 0.05→0.10→0.20).
- La métrica (TTP-combinado), el número de semillas (n=15) y la **estadística adecuada a datos
  bimodales** (mediana + tasa de éxito + Mann-Whitney/Fisher, no media±std ni t-test ingenuo).
- **Criterios de éxito explícitos** para cada experimento.

**Entregable:** protocolo experimental escrito.

### Semana 3 — EXP-1: Robustez

**Pregunta:** ¿la política aprendida mantiene su ventaja cuando (a) los parámetros calibrados
están mal estimados y (b) las mediciones clínicas tienen ruido?

**Diseño:**
- Se **entrenan 15 políticas** MAPPO en el entorno nominal.
- **EXP-1a (perturbación ±20%):** se evalúan esas políticas en entornos con cada constante
  alterada ×{0.8, 0.9, 1.0, 1.1, 1.2}, una a la vez (8 parámetros).
- **EXP-1b (ruido de observación):** solo en evaluación, se añade ruido gaussiano relativo a lo
  que la terapia mide (`S+R`, `c`) con σ ∈ {0, 5%, 10%, 20%}, 30 repeticiones por semilla.

**Resultados (mediana de TTP-combinado, n=15):**

*Ruido de observación — la política es robusta:*

| σ (ruido) | 0% | 5% | 10% | 20% |
|---|---|---|---|---|
| TTP mediana | 35 | 36 | 38 | 37 |

*Perturbación paramétrica ±20% — robusta en la mayoría, frágil ante `K`:*

| Parámetro | ×0.8 | ×1.0 | ×1.2 | Lectura |
|---|---|---|---|---|
| `ic50_R`, `delta_max_R`, `lambda_c` | ~35 | 35 | ~35 | prácticamente sin efecto |
| `ic50_S` | 39 | 35 | 23 | menor potencia sobre sensibles → peor |
| `alpha_S` | 40 | 35 | 23 | sensibles crecen más rápido → peor |
| `delta_max_S` | 25 | 35 | 40 | más muerte de sensibles → mejor |
| **`K` (capacidad de carga)** | **12** | 35 | 36 | **fragilidad: K−20% colapsa el TTP** |

**Conclusión honesta:** la política es **muy robusta al ruido de medición** (clave para validez
traslacional) y estable ante la mayoría de parámetros, pero **sensible a una subestimación de la
capacidad de carga `K`** (−20% la lleva a 12 d). Esto se reporta como límite de operación, no se
esconde.

**Entregables:** `outputs/exp1_robustez.md` (tabla), `outputs/exp1a_perturbacion.png` y
`outputs/exp1b_ruido.png` (figuras).

### Semana 4 — EXP-2: Adversario tumoral fortalecido

**Pregunta:** al darle más poder al tumor (mayor `phi_max`), (a) ¿cuánto se degrada la terapia y
(b) **cuándo empieza a importar el crítico centralizado (CTDE/MAPPO) frente al local (IPPO)**?
Esta es la pregunta abierta que el póster de PFC I dejó planteada.

**Diseño:** barrido `phi_max` ∈ {0.05, 0.10, 0.20} × variante {MAPPO, IPPO} × 15 semillas =
**90 re-entrenamientos** (180 modelos). Se evalúa contra el **tumor adaptativo co-entrenado**
(peor caso realista, no un φ fijo débil). Contraste MAPPO>IPPO con Mann-Whitney U.

**Resultados:**

| `phi_max` | MAPPO (CTDE) | IPPO (local) | Δ mediana | p (MWU) | Éxito MAPPO | Éxito IPPO |
|---|---|---|---|---|---|---|
| 0.05 (nominal) | 36 [25,61] | 29 [23,32] | +7 | **0.009** | 93% | 80% |
| 0.10 | 27 [22,61] | 27 [23,31] | +0 | 0.265 | 47% | 33% |
| 0.20 (fuerte) | 28 [22,46] | 24 [20,29] | +4 | **0.001** | **53%** | **7%** |

(*Éxito = proporción de semillas que superan a Gatenby, TTP > 27 d.*)

**Lecturas clave:**
1. **CTDE aporta, y aporta más cuando el adversario es fuerte** — en la métrica de **tasa de
   éxito** la brecha se dispara con el adversario más agresivo: **53% (MAPPO) vs 7% (IPPO)**. Esto
   responde afirmativamente la pregunta del póster sobre el valor de CTDE.
2. **El efecto no es monótono:** en `phi_max=0.10` MAPPO e IPPO empatan (p=0.265). Lo reportamos
   con honestidad; es un régimen intermedio donde la ventaja de CTDE no se manifiesta.
3. **La amenaza fuerte cambia el modo de falla:** con `phi_max` alto, todas las corridas fallan
   por **carga** (el tumor crece demasiado), no por resistencia.

**Entregables:** `outputs/exp2_adversario.md` (tabla) y `outputs/exp2_adversario.png` (figura).

---

## 4. Cómo se implementó (nota técnica)

- **Lenguaje/stack:** Python 3.12; PyTorch (RL propio, estilo CleanRL); PettingZoo `ParallelEnv`
  para el entorno de 2 agentes; NumPy/SciPy/Matplotlib. Un solo framework, implementación propia
  (no SB3).
- **Algoritmo:** MAPPO por self-play. Cada agente decide con **observación parcial** (la terapia
  ve `S+R` y `c`, no la fracción resistente → realismo clínico), pero su **crítico ve el estado
  conjunto** `(S,R,c)` durante el entrenamiento (CTDE). La ablación **IPPO** usa crítico local
  (mismo código, flag `centralized=False`).
- **Reproducibilidad:** semillas fijas + hilo único + algoritmos deterministas. Verificado:
  misma semilla ⇒ resultado idéntico.
- **Arnés reanudable:** cada experimento guarda resultados parciales en un JSON y los modelos en
  disco; si se corta, **reanuda donde quedó** sin repetir cómputo. Los dos experimentos corrieron
  en segundo plano (~20 min EXP-1, ~1.8 h EXP-2).
- **Estadística honesta:** por la bimodalidad del self-play, se reporta **mediana + [min,max] +
  tasa de éxito** y contrastes no paramétricos (Mann-Whitney), en vez de media±std / t-test.
- **Código nuevo de esta etapa:**
  - `scripts/exp1_robustez.py` — EXP-1 (robustez), reanudable, genera tabla + 2 figuras.
  - `scripts/exp2_adversario.py` — EXP-2 (adversario fuerte), reanudable, genera tabla + figura.
  - `docs/protocolo_experimental_pfc2.md` — protocolo pre-registrado.
  - `requirements.txt` — entorno reproducible.

---

## 5. Estado actual

| Semana | Actividad | Estado |
|---|---|---|
| S1 | Línea base congelada (tag `v1.0`) + replicación | ✅ Completado |
| S2 | Protocolo experimental escrito | ✅ Completado |
| S3 | EXP-1 Robustez (±20% + ruido de obs) | ✅ Completado |
| S4 | EXP-2 Adversario fuerte (barrido `phi_max`) | ✅ Completado |

**Artefactos disponibles:** 195 modelos entrenados (15 de EXP-1 + 180 de EXP-2), 2 tablas `.md`,
3 figuras `.png`, protocolo y este informe. Todo reproducible desde `v1.0`.

---

## 6. Cómo pensamos seguir (S5 en adelante)

- **S5 — EXP-3 (Bimodalidad):** caracterizar las cuencas de convergencia del self-play (por qué
  unas semillas dan TTP alto y otras bajo) y proponer una estrategia de mitigación
  (selección/checkpoint por cuenca). **Ya tenemos 195 modelos entrenados** para analizar sin
  re-entrenar.
- **S6–S7 — Redacción del Capítulo 5 (Discusión):** integrar EXP-1/2/3, interpretar el mecanismo
  pulsado en diálogo con la literatura (Gatenby, terapia adaptativa, RARL) y declarar límites
  (fragilidad ante `K`, adversario aún mejorable).
- **S8 — EC1:** presentación de avance ante el asesor con resultados nuevos.
- **Líneas abiertas (trabajo futuro del póster):** horizonte a 360 días (¿contención real o tope?),
  calibración con datos clínicos, tratamientos combinados/espaciales.

**Preguntas para el asesor:**
1. ¿Priorizamos EXP-3 (bimodalidad) o profundizamos EXP-2 con una palanca adicional para el tumor?
2. ¿La fragilidad ante `K` amerita re-calibrar `K` con una fuente adicional, o basta reportarla
   como límite de validez?
3. ¿El resultado de tasa de éxito (53% vs 7%) es suficiente como evidencia de CTDE para el jurado,
   o conviene ampliar semillas / regímenes intermedios?
