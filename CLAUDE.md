Contexto del repositorio para cualquier sesión de Claude (Code o chat). **Lee esto antes de escribir código.** No re-debatas las "Decisiones bloqueadas".

## Qué es este proyecto
PFC de Ciencias de la Computación (UTEC, 2026). Autores: Gianpier Segovia, Kalos Lazo. Asesor: Victor Martinez Abaunza.
Modelamos la resistencia a la quimioterapia en Glioblastoma como un **juego adversarial de 2 agentes**:
- **Agente Terapia:** elige la dosis `u(t)` por día. Minimiza el tumor.
- **Agente Tumor:** controla la transición fenotípica `φ(c)` (sensibles→resistentes). Maximiza supervivencia.
El aprendizaje ocurre en un **simulador ODE** (Lotka-Volterra + fármaco, integrado con RK4). Los datos (GDSC2 + DepMap) **solo calibran el simulador**, no entrenan una red supervisada.

ODEs del entorno:
```
dS/dt = α_S·S·(1 − (S+R)/K) − δ_S(c)·S
dR/dt = α_R·R·(1 − (S+R)/K) − δ_R(c)·R + φ(c)·S
dc/dt = −λ_c·c + u(t)
```

## Decisiones bloqueadas (NO re-debatir)
1. **Algoritmo = MAPPO (CTDE)**, crítico centralizado en entrenamiento, actores locales en ejecución. NO IPPO.
2. Como el problema es **adversarial**, describirlo como "CTDE con crítico centralizado por agente condicionado al estado conjunto, entrenado por self-play". NO "MAPPO cooperativo".
3. **Un solo framework.** Implementación propia en PyTorch (referencia: CleanRL). PROHIBIDO el SB3 self-play alternado del código viejo (no es MARL).
4. Entorno = PettingZoo `ParallelEnv`.
5. Estado dinámico real = **3-D** (S, R, c). Los 30 genes GBM son contexto estático (condicionan params iniciales), no evolucionan en el episodio.
6. Recompensa primaria = **LN_IC50**. AUC se descarta (r≈0.80, redundante). Z_SCORE fuera.
7. El espacio de acción del Agente Tumor **debe estar acotado** con costo de fitness. Adversario sin restricción = hombre de paja.

## Gotchas críticos (errores del código viejo — NO repetir)
- **Llaves GDSC2↔DepMap:** normalizar AMBOS lados antes de mapear (`.str.upper().str.replace(r'[^A-Z0-9]','',regex=True)`). El `.map()` crudo perdía líneas en silencio (probable causa del falso n=6). Siempre reportar cuántas líneas GBM se recuperan y por qué se descarta cada una.
- **`except` desnudos:** prohibido `except Exception:` sin loggear. Siempre `except Exception as e: log(...)`.
- **Calibración documentada:** toda fórmula LN_IC50 → parámetro ODE va explícita en código y en LaTeX. Nada de constantes mágicas sin derivación.
- **Validación no circular:** nunca comparar MARL vs MTD solo en el simulador de entrenamiento. Ver `evaluate.py`.

## Arquitectura
```
data/    build_dataset.py · calibrate_env.py · eda.py
env/     ode_dynamics.py · tumor_env.py
algo/    ppo.py (single-agent) · mappo.py · networks.py · buffer.py
tests/   test_ode.py · test_env.py
train.py · evaluate.py · config.py · requirements.txt
latex/   PFC1.tex   (one-pager) · thesis/ (plantilla larga)
```

## Orden de construcción (inside-out)
ode_dynamics → config → tumor_env → **ppo single-agent (tumor fijo)** → mappo (2 agentes) → evaluate.
Razón: la ODE es testeable sola; el PPO single-agent de-riskea el entorno antes de meter MARL.

## Definition of done por módulo
- `ode_dynamics.py`: tests pasan — sin fármaco el tumor crece a K; con dosis alta constante S↓ y R↑; nunca negativos.
- `tumor_env.py`: `reset()` + 100 steps con acciones random sin crash; obs/action spaces correctos.
- `ppo.py`: el reward del terapeuta supera a un baseline random vs tumor fijo.
- `mappo.py`: entrena 2 agentes; crítico recibe estado conjunto; reward del terapeuta no colapsa.
- `evaluate.py`: figura comparativa de trayectoria tumoral MARL vs MTD vs terapia adaptativa.

## Comandos
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/build_dataset.py        # genera dataset + reporta nº líneas GBM
python data/eda.py                  # figuras EDA
pytest tests/                       # sanity del entorno
python train.py --mode ppo          # fase 3a single-agent
python train.py --mode mappo        # fase 3b MARL
python evaluate.py
cd latex && latexmk -pdf PFC1.tex
```

## Convenciones
- Python 3.11+, type hints, docstrings cortos en español.
- Sin notebooks como fuente de verdad: todo en `.py` reproducible.
- Semillas fijas (`np.random.seed`, `torch.manual_seed`) para reproducibilidad.
- Commits pequeños y descriptivos; un módulo por commit.

## Estilo de respuesta esperado del modelo
Español, directo, acción primero. Sin preámbulos ni rellenos. Una decisión a la vez. Si algo de las "Decisiones bloqueadas" parece mal, decirlo explícito antes de cambiarlo, no asumir.