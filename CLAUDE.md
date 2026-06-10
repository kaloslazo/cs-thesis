# CLAUDE.md

Contexto del repositorio para cualquier sesión de Claude (Code o chat). **Lee esto antes de escribir código.** No re-debatas las "Decisiones bloqueadas" salvo que detectes un error (entonces dilo explícito antes de cambiar).

## Qué es este proyecto
PFC de Ciencia de la Computación (UTEC, 2026). Autores: Kalos Lazo, Gianpier Segovia. Asesor: Victor Martinez Abaunza.
Nombre del marco: **GBMARL**.

Modelamos la resistencia a la quimioterapia (temozolomida, TMZ) en Glioblastoma como un **juego adversarial de 2 agentes** resuelto por RL multiagente:
- **Agente Terapia:** elige la dosis `u(t)` por día. Objetivo clínico: mantener el tumor controlado y tratable el mayor tiempo posible.
- **Agente Tumor:** controla la transición fenotípica `φ` (sensibles→resistentes). Objetivo: forzar la dominancia resistente.

El aprendizaje ocurre en un **simulador ODE** (Lotka-Volterra + fármaco, RK4). Los datos (GDSC2 + DepMap) **solo calibran el simulador**, no entrenan una red supervisada.

ODEs del entorno (¡el término `−φ·S` en dS es obligatorio, conserva masa!):
```
dS/dt = α_S·S·(1 − (S+R)/K) − δ_S(c)·S − φ·S
dR/dt = α_R·R·(1 − (S+R)/K) − δ_R(c)·R + φ·S
dc/dt = −λ_c·c + u(t)
```

## Hallazgo central (con la métrica CORRECTA, n=15)
```
MAPPO-CTDE: 33.5 ± 5.8 días   (p<0.001 vs baselines)
IPPO:       30.2 días
Gatenby:    27 días
MTD:        13 días
Sin tratar: 12 días
```
- **Framing honesto: retrasar, no curar.** Con la calibración real del GBM, la resistencia es inevitable bajo cualquier tratamiento (nadie llega al horizonte). El objetivo y la métrica miden *demora*, no reducción.
- **Mecanismo aprendido:** dosificación pulsada/intermitente que preserva sensibles competidoras (terapia adaptativa redescubierta).
- **CTDE vs IPPO:** con la métrica correcta MAPPO supera a IPPO, pero el margen es estrecho (borderline a n=5–15). Reportar honesto; citar de Witt 2020.
- **Bimodalidad:** el self-play cae en cuencas distintas según semilla. Reportar **tasa de éxito**, no media±std (el t-test rompe con varianza cero → usar Fisher/descriptivo).

## Decisiones bloqueadas (NO re-debatir salvo error)
1. **Método titular = MAPPO (CTDE):** crítico centralizado condicionado al estado conjunto en entrenamiento, actores con obs local en ejecución, vía self-play. **IPPO está implementado** (flag `centralized=False` en `train_mappo`) **como la ablación CTDE** — es parte del experimento, no está prohibido.
2. Describir como "CTDE con crítico centralizado por agente condicionado al estado conjunto, entrenado por self-play adversarial". NO "MAPPO cooperativo".
3. **Un solo framework.** Implementación propia en PyTorch (referencia: CleanRL). PROHIBIDO el SB3 self-play alternado del código viejo (no es MARL).
4. Entorno = PettingZoo `ParallelEnv`, 2 agentes (`"therapy"`, `"tumor"`).
5. Estado dinámico real = **3-D** (S, R, c). Los genes GBM son contexto estático (condicionan params iniciales vía calibración), no evolucionan en el episodio.
6. **Calibración:** la columna de datos `LN_IC50` deriva la *potencia* del fármaco (IC50 de cada población, brecha S/R). El techo de muerte (`delta_max`) se ancla a literatura vía `KILL_TO_GROWTH_RATIO=2.0`. NO confundir esto con la recompensa del entorno (ver punto 9).
7. La acción del Agente Tumor **debe estar acotada** (`phi_max`) con costo de fitness. Adversario sin restricción = hombre de paja. (Nota: el adversario resultó *débil* aun acotado — fortalecerlo es trabajo futuro abierto.)
8. **Objetivo = retrasar la intratabilidad, NO curar/reducir.** Minimizar carga es la trampa (es lo que hace MTD y por eso pierde). No premiar reducción de tumor.
9. **Recompensa del entorno:** `+control_bonus` por día CONTROLADO (carga<prog_thr) **Y** TRATABLE (fracR<r_majority), menos `tox_weight·u`. El episodio termina si falla cualquiera. Params: `tox_weight=0.05`, `r_majority=0.50`, `control_bonus=1.0`, `progression_bonus=10`, `win_bonus=50`, `prog_thr=0.80·K`, `S0=0.40`, `R0=0.01`, `horizon=180`, `dt=0.1`, `phi_max=0.05`, `ent_coef=0.01`.
10. **Métrica de evaluación = TTP-combinado** (`gbmarl/evalutils.py::ttp_combinado`): días hasta que falla carga O resistencia, reportando el modo de falla. Ver gotcha de métrica abajo.

## Gotchas críticos (errores ya cometidos — NO repetir)
- **BUG DE MÉTRICA (el más grave):** NUNCA puntuar con una métrica "solo-resistencia" que devuelve el horizonte (180) cuando la resistencia no fue la causa de falla. Eso INFLA falsamente los resultados (un episodio que murió por carga al día 30 se anotaba como 180). Usar SIEMPRE `ttp_combinado` y reportar el motivo de falla con `ttp_combinado_detalle`. Auditar con `scripts/diagnose.py`.
- **Reproducibilidad:** `train_mappo` ya fija `torch.set_num_threads(1)` + `use_deterministic_algorithms(True)` + sembrado completo. NO lo quites (sacrifica velocidad por determinismo, intencional). La misma semilla es reproducible *dentro del mismo script*; entre scripts puede diferir por el contexto del RNG → reportar reproducibilidad intra-experimento.
- **Llaves GDSC2↔DepMap:** normalizar AMBOS lados antes de mapear (`.str.upper().str.replace(r'[^A-Z0-9]','',regex=True)`). El `.map()` crudo perdía líneas en silencio (causa del falso n=6; el real fue **34 líneas GBM** con ensayos, **24** con genómica completa). Siempre reportar cuántas se recuperan y por qué se descarta cada una.
- **`except` desnudos:** prohibido `except Exception:` sin loggear. Siempre `except Exception as e: log(...)`.
- **Calibración documentada:** toda fórmula LN_IC50 → parámetro ODE va explícita en código y en LaTeX. Nada de constantes mágicas sin derivación. (Real: `ic50_S≈0.36`, `ic50_R≈3.27`, brecha ~9×.)
- **Validación no circular:** comparar contra baselines (MTD, Gatenby) con la misma métrica y mismo adversario. Ver `evaluate.py` / `scripts/diagnose.py`.
- **Estadística honesta:** datos bimodales → tasa de éxito, no media±std. t-test inválido con varianza cero.

## Arquitectura real del repo
```
thesis-code/
├── data/raw/{depmap,cellmodelpassports}/   datasets crudos
│   processed/calibration.json               params del fármaco calibrados
│   dataset_marl_gbm_completo.csv
├── gbmarl/                 paquete (correr desde la raíz)
│   ├── config.py           Params dataclass + load_calibration()
│   ├── dynamics.py         ODEs + RK4 (math pura)
│   ├── tumor_env.py        PettingZoo ParallelEnv, 2 agentes, recompensa
│   ├── single_env.py       wrapper single-agent (PPO, tumor fijo)
│   ├── ppo.py              PPO single-agent
│   ├── mappo.py            MAPPO/IPPO (flag centralized), train_mappo
│   ├── evalutils.py        ttp_combinado + Gatenby (MÉTRICA CORRECTA)
│   └── tests/              test_dynamics.py · test_env.py
├── scripts/   build_dataset.py · calibrate.py · plot_dynamics.py
│   train_ppo.py · train_mappo.py · evaluate.py
│   validate_seeds.py · inspect_seed.py · inspect_policies.py
│   ablation_ctde.py · ablation_hard.py · diagnose.py
├── pytest.ini · requirements.txt
└── thesis-latex/   main.tex · secciones/{capitulo1,capitulo2,...}.tex
    referencias_relatedwork.bib · referencias_intro.bib
```

## Orden de construcción (inside-out, ya ejecutado)
dynamics → config → tumor_env → **ppo single-agent (de-riskea entorno)** → mappo (2 agentes) → evaluate → validate_seeds → ablación CTDE → diagnose.

## Comandos (desde la raíz)
```bash
python -m pytest -v                          # 10/10 tests
python scripts/build_dataset.py              # dataset + nº líneas GBM
python scripts/calibrate.py                  # calibration.json
python scripts/train_ppo.py                  # fase single-agent
python scripts/train_mappo.py                # MAPPO self-play
python scripts/evaluate.py                   # comparación TTP + figuras
python scripts/validate_seeds.py --seeds 15  # significancia multi-semilla
python scripts/ablation_ctde.py --seeds 15   # MAPPO vs IPPO
python scripts/diagnose.py --seeds 5         # auditoría: arquitectura + métrica + modos de falla
cd thesis-latex && tectonic main.tex
```

## Estado y pendientes
- [✅] simulador + calibración · entorno · PPO · MAPPO/IPPO · evaluación · validación multi-semilla · ablación CTDE · métrica auditada · reproducibilidad.
- [✅] LaTeX: Cap.1 (Introducción) y Cap.2 (Revisión de la Literatura) redactados.
- [ ] Análisis de sensibilidad (¿el ranking sobrevive al variar KILL_TO_GROWTH_RATIO, umbrales, phi_max?).
- [ ] Horizonte 360d (¿contención real o tope?).
- [ ] Adversario más fuerte (palanca extra para el tumor — abre la pregunta de cuándo importa CTDE).
- [ ] LaTeX: Cap.3 (Metodología) y Cap.4 (Resultados).

## Convenciones
- Python 3.9+ (Mac), type hints, docstrings cortos en español.
- Sin notebooks como fuente de verdad: todo en `.py` reproducible.
- Semillas fijas; determinismo activado en `train_mappo`.
- Commits pequeños y descriptivos; un módulo por commit.
- Un archivo nuevo del paquete va en `gbmarl/`; un experimento en `scripts/`.

## Estilo de respuesta esperado del modelo
Español, directo, acción primero. Sin preámbulos ni rellenos. Una decisión/cambio a la vez. Verificar resultados con datos antes de concluir (la métrica engañosa enseñó esto). Si algo de las "Decisiones bloqueadas" parece mal, decirlo explícito antes de cambiarlo, no asumir.