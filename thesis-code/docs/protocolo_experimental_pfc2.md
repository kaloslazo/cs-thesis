# Protocolo experimental — PFC II (GBMARL)

**Tesis:** Mitigación de la resistencia evolutiva en glioblastoma multiforme mediante
aprendizaje por refuerzo multiagente adversarial (MAPPO-CTDE).
**Autores:** Kalos B. Lazo Mera · Gianpier A. Segovia Ureta · **Asesor:** V. E. Martínez Abaunza.
**Línea base congelada:** tag `v1.0` (commit `a9e2ac5`). Todos los experimentos de PFC II
parten de este estado y se comparan contra él.

> Este documento es el entregable de la **Semana 2** del cronograma PFC II ("diseño
> experimental escrito"). Define el protocolo de robustez (EXP-1) y el adversario tumoral
> fortalecido (EXP-2), con métrica, semillas, estadística y criterios de éxito **antes** de
> correr los experimentos, para evitar sesgo post-hoc.

---

## 0. Principios metodológicos (heredados de PFC I)

1. **Métrica única y honesta:** TTP-combinado (`gbmarl/evalutils.py::ttp_combinado`) — días
   hasta que el episodio falla por **carga** (carga ≥ `prog_thr`) **o** por **resistencia**
   (fracR ≥ `r_majority`), reportando *el modo de falla*. Prohibido usar la métrica
   solo-resistencia que infla resultados (ver "gotcha de métrica" en `CLAUDE.md`).
2. **Estadística según la distribución:** el self-play es **bimodal** (cuencas distintas por
   semilla). Se reporta **tasa de éxito vs. Gatenby** (proporción de semillas con TTP > 27) y
   **mediana [min, max]**, no media±std cuando hay bimodalidad. Para contrastes se usa
   Mann-Whitney U (no paramétrico) o Fisher sobre la tasa de éxito; el t-test solo si la
   varianza intra-grupo lo permite.
3. **Reproducibilidad:** `train_mappo` fija hilo único + `use_deterministic_algorithms(True)` +
   sembrado completo. La reproducibilidad es **intra-script** (misma semilla → mismo resultado,
   verificado en S1). Toda corrida registra: semilla, nº de pasos, `phi_max`, y hash de config.
4. **Comparación no circular:** baselines (Sin tratar, MTD, Gatenby) se evalúan con la **misma
   métrica** y **mismo entorno/adversario** que la política aprendida.

### Parámetros nominales del entorno (línea base v1.0)

| Param | Valor | Origen |
|---|---|---|
| `alpha_S`, `alpha_R` | 0.15, 0.10 | literatura (cinética GBM) |
| `K` | 1.0 | literatura |
| `lambda_c` | 0.40 | PK de TMZ |
| `ic50_S`, `ic50_R` | 0.36, 3.27 (brecha ≈9×) | **calibrado** GDSC2/DepMap (34 líneas) |
| `delta_max_S`, `delta_max_R` | calibrado (`KILL_TO_GROWTH_RATIO=2.0`) | datos + ancla literatura |
| `phi_max` | 0.05 | acotado (acción del tumor) |
| `tox_weight`, `r_majority`, `control_bonus` | 0.05, 0.50, 1.0 | diseño de recompensa |
| `prog_thr`, `S0`, `R0`, `horizon`, `dt` | 0.80·K, 0.40, 0.01, 180, 0.1 | diseño |

---

## EXP-1 — Robustez (Semana 3)

**Pregunta.** ¿La política aprendida (MAPPO-CTDE) mantiene su ventaja de TTP-combinado cuando
(a) los parámetros calibrados se perturban y (b) las observaciones clínicas tienen ruido?
Un modelo calibrado in vitro **debe** demostrar que no es frágil a estas dos fuentes de
incertidumbre para tener validez traslacional.

### EXP-1a — Perturbación paramétrica (±20%)

- **Qué se perturba:** las constantes con mayor incertidumbre de calibración/literatura, una a
  la vez (sensibilidad local, OAT) y luego conjuntamente (peor caso):
  `ic50_S`, `ic50_R`, `delta_max_S`, `delta_max_R`, `alpha_S`, `alpha_R`, `lambda_c`, `K`.
- **Rango:** factor multiplicativo en `{0.8, 0.9, 1.0, 1.1, 1.2}` (±20%, con 1.0 = nominal).
  El protocolo del cronograma exige ±20% (más estricto que el ±30% OAT de PFC I en
  `sensitivity.py`, que se conserva como anexo).
- **Política evaluada:** se **congela** la política MAPPO entrenada en el entorno nominal y se
  **evalúa** en el entorno perturbado (robustez de transferencia, sin re-entrenar). Se comparan
  MTD y Gatenby en el mismo entorno perturbado como referencia.
- **Modo conjunto (peor caso):** muestreo Monte-Carlo de `M=50` entornos con cada parámetro
  perturbado ±20% uniformemente e independiente; se reporta la distribución de TTP.

### EXP-1b — Ruido de observación

- **Dónde:** solo en **tiempo de evaluación**, sobre la observación de la **terapia** `(S+R, c)`
  — que es la magnitud que un clínico mide con error (imagen/volumetría, niveles de fármaco).
  Se implementa como *wrapper* de evaluación (no se toca `tumor_env.py`, no se re-entrena),
  para medir la robustez de la política **ya aprendida** (lo que pide S3).
- **Modelo de ruido:** gaussiano multiplicativo relativo,
  `obs_ruidosa = obs · (1 + ε)`, `ε ~ N(0, σ²)`, truncado a obs ≥ 0.
- **Niveles:** `σ ∈ {0.0, 0.05, 0.10, 0.20}` (0%, 5%, 10%, 20% de ruido relativo).
- **Repeticiones:** por cada σ, `R=30` episodios con distinta realización de ruido; se reporta
  mediana y rango de TTP-combinado.

### Diseño estadístico EXP-1
- **Semillas de política:** `n=15` (consistente con la cifra citada en la tesis).
- **Métrica:** TTP-combinado + modo de falla.
- **Salidas:** (1) tabla de degradación TTP por parámetro y por σ; (2) figura de curvas de
  degradación (TTP vs. magnitud de perturbación) con banda min–max; (3) tasa de éxito vs.
  Gatenby bajo cada condición.
- **Criterio de robustez (pre-registrado):** la política se considera **robusta** si, en todo el
  rango ±20% y hasta σ=0.10, (i) la **mediana** de TTP se mantiene ≥ Gatenby nominal (27 d) y
  (ii) el **orden** MAPPO ≥ Gatenby ≥ MTD nunca se invierte. Degradaciones más allá de σ=0.10
  se reportan con honestidad como límite de operación.

---

## EXP-2 — Adversario tumoral fortalecido (Semana 4)

**Pregunta.** Al ampliar el espacio de acción del agente tumoral (mayor `phi_max`), ¿(a) cómo
se degrada el TTP de la terapia aprendida frente al adversario original, y (b) el crítico
centralizado (CTDE/MAPPO) gana ventaja sobre el local (IPPO) cuando la amenaza es más realista?
El póster de PFC I ya señala al adversario débil como limitación; este experimento la ataca.

### Diseño
- **Palanca del adversario:** `phi_max ∈ {0.05, 0.10, 0.20}` (barrido). 0.05 = nominal v1.0;
  0.10 y 0.20 = adversario fortalecido. El tumor conserva su **costo de fitness** (no es hombre
  de paja).
- **Re-entrenamiento:** por cada `phi_max` se **re-entrena** MAPPO-CTDE **y** IPPO por self-play
  a `n=15` semillas (co-evolución completa; la terapia enfrenta a su propio tumor adaptativo,
  no a un φ fijo débil — peor caso realista).
- **Evaluación:** TTP-combinado con **ambos agentes aprendidos actuando**
  (`ablation_hard.py::ttp_vs_adaptivo` extendido a la métrica combinada), no contra φ fijo.
- **Comparaciones:**
  1. TTP de la terapia aprendida vs. `phi_max` (curva de degradación frente a adversario original).
  2. MAPPO (CTDE) vs. IPPO (local) en cada `phi_max` → **¿cuándo empieza a importar CTDE?**
  3. Modo de falla dominante por régimen (se espera que a `phi_max` alto la falla migre hacia
     "resistencia mayoría").

### Diseño estadístico EXP-2
- **Semillas:** `n=15` por (variante × phi_max).
- **Contraste CTDE vs. IPPO:** Mann-Whitney U (unilateral, MAPPO > IPPO) + tasa de éxito; se
  reporta *p* honesto y tamaño de efecto (diferencia de medianas). Se documenta la
  **bimodalidad** por cuenca.
- **Hipótesis pre-registrada:** con observabilidad parcial más severa (adversario fuerte), la
  brecha MAPPO−IPPO **crece** respecto al régimen nominal (donde era estrecha/borderline). Si
  no crece, se reporta como resultado negativo honesto (CTDE aporta poco incluso con adversario
  fuerte) — ambos desenlaces son publicables.

---

## Presupuesto de cómputo y reproducibilidad

- **Escala de entrenamiento:** 120k pasos/semilla (misma que la corrida reproducible de PFC I).
- **Total de corridas nuevas:**
  - EXP-2: 3 `phi_max` × 2 variantes × 15 semillas = **90 entrenamientos** de 120k.
  - EXP-1: reusa las 15 políticas MAPPO nominales (sin re-entrenar); solo evaluación.
- **Arnés:** se extiende el patrón reanudable con checkpointing de `run_pipeline.py` /
  `train_ckpt.py` para tolerar cortes y garantizar reanudación (no se pierde cómputo).
- **Artefactos versionados por corrida:** modelos `.pt`, `outputs/*.json` con TTP por semilla y
  modo de falla, figuras `.png`, y manifiesto de estado JSON.

## Responsables
- **Experimentos (correr, checkpoints):** Kalos.
- **Análisis (tablas, figuras, estadística, redacción de resultados):** Gianpier.
- **Revisión técnica cruzada y sign-off de asesor:** ambos (reunión semanal viernes).
