# ROADMAP — Construcción de cero hasta la tesis

Plan por hitos. Cada hito tiene: tareas, criterio de aceptación, sección de LaTeX que alimenta, y qué aprender ANTES (ver `LEARNING_GUIDE.md`). Marca con `[x]` al cerrar.

Principio: **construir inside-out** (ODE → entorno → PPO single-agent → MAPPO → evaluación). No empieces por el algoritmo.

---

## M0 — Esqueleto del repo (30–45 min) · primera sesión

- [ ] Crear estructura de carpetas (`data/ env/ algo/ tests/ latex/`).
- [ ] `python -m venv .venv` + `requirements.txt` con: `numpy pandas scipy matplotlib seaborn torch pettingzoo gymnasium tqdm pytest`.
- [ ] Copiar `CLAUDE.md` y la plantilla LaTeX a `latex/thesis/`.
- [ ] `git add . && git commit -m "scaffold"`.

**Criterio:** `pytest` corre (aunque sin tests) y los imports no fallan.
**Aprender antes:** nada. Solo arranca.

---

## M1 — Dinámica ODE (½ día) · el corazón del proyecto

- [ ] `env/ode_dynamics.py`: las 3 ecuaciones + integrador RK4 + función de Hill `δ(c)`.
- [ ] `config.py`: parámetros iniciales (α, δ, λ, K) con valores de literatura como placeholder.
- [ ] `tests/test_ode.py`: (a) sin fármaco → tumor crece hasta K; (b) dosis alta constante → S↓, R↑; (c) S,R,c ≥ 0 siempre.
- [ ] Script rápido que plotee S(t), R(t), c(t) bajo 3 regímenes (sin dosis / dosis máxima / dosis pulsada).

**Criterio:** las 3 pruebas pasan y la gráfica muestra la selección (R domina bajo dosis máxima).
**Alimenta LaTeX:** sección 3.4 "Dinámicas del Entorno".
**Aprender antes:** L-BIO (Lotka-Volterra, Hill, terapia adaptativa).

---

## M2 — Calibración (½ día) · cierra el FIX A y C

- [ ] `data/build_dataset.py` con normalización de llaves (FIX A). **Reportar nº de líneas GBM recuperadas (>6 esperado).**
- [ ] `data/calibrate_env.py`: fórmula LN_IC50 → (IC50_S, IC50_R, δ) documentada. Usa SciPy si hace falta ajustar.
- [ ] Reusar `data/eda.py` previo; corregir etiqueta "Concentración" en la figura AUC.

**Criterio:** dataset regenerado, nº de líneas reportado, calibración reproducible y derivada (no constantes mágicas).
**Alimenta LaTeX:** secciones 1 (dataset), 2 (EDA), y la fórmula de calibración.
**Aprender antes:** L-CALIB (ajuste de parámetros con SciPy `curve_fit`/`minimize`).

---

## M3 — Entorno PettingZoo (1 día)

- [ ] `env/tumor_env.py`: `ParallelEnv` con 2 agentes. Estado = [S, R, c] + 30 genes estáticos. Acciones: terapia=dosis continua acotada; tumor=`φ` acotada con costo de fitness. Rewards según decisión 6/7 de CLAUDE.md.
- [ ] `tests/test_env.py`: `reset()` + 100 steps random sin crash; spaces correctos; reward del tumor sube si sobrevive.

**Criterio:** loop random estable 100 steps; tipos y shapes correctos.
**Alimenta LaTeX:** sección de "Modelado del Entorno" + espacios S/A/reward.
**Aprender antes:** L-RL (MDP, estado/acción/reward) + API PettingZoo ParallelEnv.

---

## M4a — PPO single-agent (1 día) · DE-RISKING, no saltar

- [ ] `algo/ppo.py`: PPO continuo para el Agente Terapia **contra un tumor con `φ` fijo** (sin agente tumor todavía). Basarse en CleanRL `ppo_continuous_action.py`, leído línea por línea.
- [ ] `train.py --mode ppo`.

**Criterio:** el reward del terapeuta supera claramente a un baseline de dosis random en el mismo entorno.
**Por qué:** si MAPPO luego no converge, ya tienes un resultado single-agent presentable. Y validas que el entorno es "aprendible".
**Aprender antes:** L-PPO (policy gradient, actor-critic, clip, GAE).

---

## M4b — MAPPO 2 agentes (1.5 días)

- [ ] `algo/networks.py`, `algo/buffer.py`, `algo/mappo.py`: dos actores (obs local) + **crítico centralizado por agente que recibe el estado conjunto** durante entrenamiento.
- [ ] `train.py --mode mappo` con schedule de self-play.

**Criterio:** ambos agentes entrenan; el crítico usa estado conjunto; el reward del terapeuta no colapsa a cero (no lo aplasta el tumor).
**Alimenta LaTeX:** sección de "Arquitectura del Modelo" (esta es la parte que el jurado más ataca → escribir con cuidado).
**Aprender antes:** L-MARL (Dec-POMDP, no-estacionariedad, CTDE) + L-IPPOMAPPO (la diferencia exacta).

---

## M5 — Evaluación no circular (½–1 día)

- [ ] `evaluate.py`: comparar política MARL vs (a) MTD fijo, (b) tumor no-aprendiz con params independientes, (c) terapia adaptativa de Gatenby/Strobl sobre la misma ODE.
- [ ] Gráficos de dinámica S/R/c por escenario + tabla de métricas (tiempo a progresión, dosis acumulada).

**Criterio:** una figura y una tabla que muestren ventaja (o no) de MARL fuera del setting de entrenamiento.
**Alimenta LaTeX:** sección de "Resultados" y "Evaluación".
**Aprender antes:** L-EVAL (qué es validación circular, terapia adaptativa).

---

## Escritura LaTeX (en paralelo, no al final)
Cada hito cierra → escribe esa sección ese mismo día, en caliente. El one-pager ya tiene título, problema, objetivo, related work y datasets correctos. La plantilla larga se llena sección por sección a medida que avanzan M1–M5.

## Entregable de "mañana" (si hay deadline corto)
M0 + M1 + M2 + scaffold de M3 + sección de metodología corregida en LaTeX. Es un Avance defendible: pipeline arreglado, ODE funcionando, EDA, y la corrección IPPO→MAPPO documentada. NO prometas resultados de entrenamiento.

## Reglas de avance
- Un hito a la vez. No abrir M4 con M2 a medias.
- Cada hito termina con: tests verdes + commit + sección LaTeX escrita.
- Si un hito se atasca >1 día, reducir alcance (ej. menos genes, menos steps) antes que abandonar el orden.