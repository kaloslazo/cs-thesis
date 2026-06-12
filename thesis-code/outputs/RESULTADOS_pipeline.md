# GBMARL — Resultados del pipeline (CORRIDA COMPLETA 120k)

Fecha: 2026-06-12 · Corrida reproducible end-to-end con arnés reanudable + checkpointing intra-corrida (`scripts/train_ckpt.py`).

## Resumen en una línea

El pipeline corre **end-to-end y reproducible** (datos → calibración → tests → PPO → MAPPO → ablación → evaluación) a la **escala plena de tesis (120 000 pasos, 5 semillas por variante)**. Reproduce el **hallazgo central**: MAPPO-CTDE supera a todos los baselines (mediana 33 d, mejor 41 d, éxito 5/5 vs Gatenby), superando a IPPO (mediana 30 d, 4/5).

## Cómo se logró la escala (técnica clave)

El entorno de ejecución impone ~45 s por proceso y no conserva procesos en segundo plano. Una corrida de 120k no cabe en una ventana. Solución implementada:

1. **`gbmarl/fast_env.py`** — versión escalar (float puro) de `TumorEnv`. Misma EDO/RK4 y misma recompensa, validada **numéricamente idéntica** (diff de estado ~5e-8 por downcast float32 del env original; ~4.7× más rápida).
2. **`scripts/train_ckpt.py`** — arnés reanudable con **checkpoint intra-corrida**: cada ventana entrena un presupuesto de ~35 s y serializa actores+críticos+optimizadores+RNG+paso+estado del entorno. La siguiente ventana reanuda hasta `--steps`. ~4 ventanas por job de 120k.

Reproducibilidad: intra-experimento (hilo único + algoritmos deterministas). El checkpointing entre reinicios es continuo vía RNG serializado.

## 1. Datos y calibración — reproducen CLAUDE.md exacto

- Cruce GDSC2 ↔ DepMap por `SANGER_MODEL_ID`: **34 líneas GBM** con ensayos; **24** con genómica completa.
- `ic50_S = 0.360`, `ic50_R = 3.274` → **brecha 9.1×**. `delta_max_S = 0.30`, `delta_max_R = 0.15`.

## 2. Tests — `10/10` pasan.

## 3. PPO single-agent (de-risk)

| Métrica | Retorno |
|---|---|
| Baseline aleatorio | 11.93 |
| PPO entrenado | **33.89** |

El entorno es aprendible → el pipeline RL funciona antes de añadir el adversario.

## 4. Evaluación — TTP-combinado (métrica auditada) vs baselines

| Estrategia | TTP-combinado | Modo de falla | Dosis media |
|---|---|---|---|
| Sin tratamiento | 12 d | carga progresó | 0.00 |
| MTD (dosis máx) | 13 d | resistencia mayoría | 1.00 |
| Adaptativa (Gatenby) | 27 d | resistencia mayoría | 0.30 |
| **MAPPO-CTDE (120k, mejor)** | **41 d** | resistencia mayoría | 0.05 |

Baselines reproducen CLAUDE.md exacto (deterministas). MAPPO con ~0.05 de dosis media = **dosificación pulsada / terapia adaptativa redescubierta**.

## 5. Ablación CTDE — corrida completa (120k, 5 semillas)

| Variante | s0 | s1 | s2 | s3 | s4 | Mediana | Éxito > Gatenby |
|---|---|---|---|---|---|---|---|
| **MAPPO (CTDE)** | 41 | 32 | 31 | 33 | 41 | **33 d** | **5/5** |
| IPPO (local) | 31 | 29 | 25 | 30 | 40 | 30 d | 4/5 |

Estadística honesta (distribución **bimodal**): se reporta **mediana y tasa de éxito**, no media±std. MAPPO > IPPO, margen estrecho (coincide con CLAUDE.md: MAPPO 33.5, IPPO 30.2). Bimodalidad visible: mayoría en 30–33 d, semillas aisladas en 40–41 d.

## Artefactos generados

- `outputs/ckpt/{mappo_main,abl_mappo_0..4,abl_ippo_0..4}.pkl` — checkpoints completos.
- `outputs/models/*.pt` — modelos finales exportados.
- `outputs/eval_full.json` — TTP por semilla + tasa de éxito.
- `outputs/mappo_learning_curves.png` — co-evolución (120k).
- `outputs/resultados_ttp.png` / `pipeline_diagram.png` — figuras de la tesis.

## Documentación LaTeX (añadida)

- `secciones/capitulo3.tex` → nueva §"Pipeline experimental reproducible": diagrama, tabla fase/propósito/aporte, por qué cada comparación, arnés de checkpointing.
- `secciones/capitulo4.tex` → nueva §"Confirmación con la corrida completa reproducible": tabla por semilla + figura, confirma el hallazgo n=15 con corrida totalmente reproducible.
- Figuras nuevas: `images/pipeline_diagram.png`, `images/resultados_full_120k.png`.
