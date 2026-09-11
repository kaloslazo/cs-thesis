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

## Hallazgo vigente (auditoría calibrada del 2026-09-08, métrica CORRECTA, n=15)
```
MAPPO-CTDE: 34 días [27,41]    (13/15 > Gatenby)
IPPO:       31 días [23,34]    (12/15 > Gatenby)
Gatenby:    27 días
MTD:        13 días
Sin tratar: 12 días
```
- La comparación MAPPO--IPPO da `p=0.013759` en Wilcoxon pareado unilateral, pero
  sigue siendo evidencia del simulador, no eficacia clínica. La corrida histórica de
  `n=15` queda documentada como histórica y no debe mezclarse con esta auditoría.
- **Alcance:** en las trayectorias nominales auditadas se observó fracaso por carga o resistencia antes del horizonte. Esto no prueba inevitabilidad bajo cualquier tratamiento. El objetivo mide demora al primer fracaso definido por el simulador.
- **Mecanismo candidato:** la política nominal usa dosis baja/intermitente compatible con preservar sensibles competidoras; esto es una interpretación del simulador, no una demostración clínica.
- **CTDE vs IPPO:** con la métrica correcta MAPPO supera a IPPO en esta muestra de 15 pares; la diferencia sigue siendo evidencia del simulador, no eficacia clínica.
- **Variabilidad:** el self-play puede caer en cuencas distintas según semilla. Reportar mediana, rango y tasa de éxito; no ocultar la dispersión con una sola mejor corrida.
- **EXP-1:** la mediana de MAPPO fue 34 días en el nominal, cayó a 25 días al reducir `K` a `0.8K`, y en la perturbación conjunta llegó a una mediana de medianas de 30 días; hubo entornos con 0% de éxito frente a Gatenby.
- **EXP-2:** con la calibración vigente y 15 semillas por celda, MAPPO obtiene medianas 44, 28 y 24 días para `phi_max={0.05,0.10,0.20}`, mientras IPPO obtiene 29, 27 y 26 días. La ventaja de CTDE es clara en `0.05` y desaparece al fortalecer el adversario; no debe presentarse como universal.
- **Benchmark V2:** frente a cinco reinicios de best-response, MAPPO alcanza TTP 14 [11,14] e IPPO 11 [10,12]; la explotabilidad puntual es mayor en MAPPO (30 vs. 18 días), aunque la prueba unilateral MAPPO<IPPO no es significativa (`p=0.993`). En cross-play, MAPPO obtiene 35 [24,41] frente a tumores IPPO e IPPO 27 [12,29] frente a tumores MAPPO.

## Decisiones bloqueadas (NO re-debatir salvo error)
1. **Método titular = MAPPO (CTDE):** crítico centralizado condicionado al estado conjunto en entrenamiento, actores con obs local en ejecución, vía self-play. **IPPO está implementado** (flag `centralized=False` en `train_mappo`) **como la ablación CTDE** — es parte del experimento, no está prohibido.
2. Describir como "CTDE con crítico centralizado por agente condicionado al estado conjunto, entrenado por self-play adversarial". NO "MAPPO cooperativo".
3. **Un solo framework.** Implementación propia en PyTorch (referencia: CleanRL). PROHIBIDO el SB3 self-play alternado del código viejo (no es MARL).
4. Entorno = PettingZoo `ParallelEnv`, 2 agentes (`"therapy"`, `"tumor"`).
5. Estado dinámico real = **3-D** (S, R, c). DepMap identifica/filtra líneas y aporta expresión al dataset integrado, pero `calibrate.py` solo usa DRUG_NAME, ModelID, LN_IC50 y AUC. La versión actual no condiciona políticas ni parámetros por expresión génica individual.
6. **Calibración:** la columna de datos `LN_IC50` deriva la *potencia* relativa del fármaco (IC50 de cada población, brecha S/R). El techo de muerte (`delta_max`) se ancla al crecimiento vía `KILL_TO_GROWTH_RATIO=2.0`, mientras que `delta_max_R` incorpora una eficacia resistente derivada de los datos con piso explícito. NO confundir esto con la recompensa del entorno (ver punto 9).
7. La acción del Agente Tumor **debe estar acotada** (`phi_max`) con costo de fitness. EXP-2 ya evaluó `phi_max={0.05,0.10,0.20}`; ampliar presupuestos y adversarios comunes sigue pendiente.
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
│   ├── outcomes.py         diagnóstico único de desenlaces y censura
│   └── patient_env.py      extensión experimental de salud/toxicidad
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
python -m pytest -v                          # 26/26 tests
python scripts/build_dataset.py              # dataset + nº líneas GBM
python scripts/calibrate.py                  # calibration.json
python scripts/train_ppo.py                  # fase single-agent
python scripts/train_mappo.py                # MAPPO self-play
python scripts/evaluate.py                   # comparación TTP + figuras
python scripts/validate_seeds.py --seeds 15  # significancia multi-semilla
python scripts/ablation_ctde.py --seeds 15   # MAPPO vs IPPO
python scripts/diagnose.py --seeds 5         # auditoría: arquitectura + métrica + modos de falla
cd thesis-latex && pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Estado y pendientes
- Revisión económica 2026-09-10: conclusiones/recomendaciones sincronizadas; referencia numérica en `thesis-code/configs/calibration_reference_20260908.json`; evaluación nominal exige calibración existente. Plan restante: `thesis-code/docs/plan_mejoras_pendientes.md`.
- [✅] datos oficiales DepMap Public 26Q1 + GDSC2 release 8.5, cruce trazable y calibración reproducible.
- [✅] simulador + entorno + PPO + MAPPO/IPPO + evaluación + métrica auditada + equivalencia FastTumorEnv/TumorEnv.
- [✅] comparación nominal calibrada con 15 semillas por variante: MAPPO 34 d [27,41], IPPO 31 d [23,34].
- [✅] EXP-1 calibrado: perturbación OAT/conjunta, umbrales del endpoint y ruido de observación.
- [✅] LaTeX: Capítulos I--IV, resumen, abstract, conclusiones y recomendaciones sin placeholders; PDF compilado sin referencias indefinidas.
- [✅] Validación nominal pre-registrada con 15 semillas bajo la huella actual; mantener la lectura como evidencia in silico y ampliar solo si se requieren intervalos más estrechos.
- [✅] Completar EXP-2 para `phi_max ∈ {0.05, 0.10, 0.20}` con 90/90 celdas, 15 semillas por método y contrastes pareados con Holm.
- [✅] Ejecutar benchmark V2 de explotabilidad con 5 reinicios, `n=15` y cross-play bajo la huella actual; el resultado no demuestra robustez universal.
- [ ] Calibrar la variable de salud con datos longitudinales de toxicidad antes de interpretarla clínicamente.
- [✅] Evaluar horizonte 360d: sin tratamiento/MTD/Gatenby/MAPPO fallan antes del horizonte (12/13/27/41 d); no hay censura administrativa en las trayectorias auditadas.

## Convenciones
- Python 3.9+ (Mac), type hints, docstrings cortos en español.
- Sin notebooks como fuente de verdad: todo en `.py` reproducible.
- Semillas fijas; determinismo activado en `train_mappo`.
- Commits pequeños y descriptivos; un módulo por commit.
- Un archivo nuevo del paquete va en `gbmarl/`; un experimento en `scripts/`.

## Estilo de respuesta esperado del modelo
Español, directo, acción primero. Sin preámbulos ni rellenos. Una decisión/cambio a la vez. Verificar resultados con datos antes de concluir (la métrica engañosa enseñó esto). Si algo de las "Decisiones bloqueadas" parece mal, decirlo explícito antes de cambiarlo, no asumir.
