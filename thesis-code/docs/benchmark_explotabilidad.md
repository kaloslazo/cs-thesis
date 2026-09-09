# Benchmark de explotabilidad y cross-play — GBMARL

> **Estado V2:** el benchmark titular ya fue ejecutado con 15 semillas, cinco
> reinicios por best-response y cross-play poblacional. “Best-response” significa
> **mejor respuesta aproximada**, no un adversario óptimo garantizado. Gatenby también
> recibió cinco atacantes dedicados con el mismo presupuesto; su mejor TTP fue 7 días.
> Los resultados vigentes están en `outputs/benchmark_exploit_v2.md` y `.json`. Véase
> `docs/correcciones_metodologicas_v2.md`.

**Tesis:** Mitigación de la resistencia evolutiva en glioblastoma multiforme mediante
aprendizaje por refuerzo multiagente adversarial (MAPPO-CTDE).
**Autores:** Kalos B. Lazo Mera · Gianpier A. Segovia Ureta · **Asesor:** V. E. Martínez Abaunza.
**Código:** `gbmarl/exploit.py` (módulo) · `scripts/benchmark_exploit.py` (experimento).
**Línea base:** tag `v1.0`. Régimen titular `phi_max=0.05`, métrica `ttp_combinado`.

> Entregable metodológico posterior a EXP-1/EXP-2. Introduce la **explotabilidad**
> (best-response) como métrica de robustez y el **cross-play** como prueba de
> generalización, para responder de forma no circular a: *¿por qué CTDE > IPPO, y en
> qué sentido preciso?*

---

## 1. Qué problema resuelve (y por qué EXP-1/EXP-2 no bastan)

Una política aprendida por **self-play** solo demostró ser buena contra *el adversario
con el que co-evolucionó*. Eso deja tres huecos que ni la robustez (EXP-1) ni el
barrido de `phi_max` (EXP-2) cubren:

1. **Circularidad del self-play:** medir la terapia contra su propio tumor co-entrenado
   es evaluarse contra un examinador que aprendió a perder de forma "cómoda". No prueba
   resistencia a un atacante que la estudia a propósito.
2. **`phi_max` es magnitud, no inteligencia:** EXP-2 agranda la *caja de acción* del
   tumor, pero el adversario sigue siendo el mismo PPO con la misma observación. Un
   adversario más grande ≠ un adversario mejor.
3. **Diferenciación frágil:** el margen CTDE vs IPPO en media±std es *borderline* y la
   distribución es bimodal. Necesitamos un contraste que no dependa de una media inestable.

El **benchmark de explotabilidad** ataca los tres a la vez con una idea de teoría de
juegos: *la calidad de una estrategia no se mide por lo que gana, sino por cuánto puede
perder frente al mejor rival posible.*

---

## 2. Qué se está haciendo (el mecanismo, reflejado en nuestro sistema)

Dos pruebas, ambas sobre la política **ya entrenada y congelada**:

### (1) Explotabilidad — best-response
`gbmarl/exploit.py::train_best_response`

1. Se **congela** la terapia aprendida (defensor desplegado; actúa con su media
   determinista, sin gradiente).
2. Se entrenan **varios tumores frescos desde cero** con recompensa alineada con minimizar
   el TTP y se conserva el atacante más dañino encontrado. Se les da un **crítico
   centralizado** (estado conjunto `S,R,c`) para construir una mejor respuesta aproximada
   exigente, sin afirmar que PPO encontró el óptimo global.
3. Se define:

   ```
   explotabilidad = TTP_selfplay − TTP_best_response
   ```

   Cuánto cae el desempeño de la terapia cuando la ataca un adversario dedicado.

### (2) Cross-play — generalización fuera del partner
`scripts/benchmark_exploit.py` (sección cross-play)

Se enfrenta la terapia de un método contra el **tumor del otro método** (un adversario
que nunca vio en entrenamiento): MAPPO-terapia vs tumor-IPPO, y viceversa. Mide si la
política transfiere o solo memorizó a su compañero de self-play.

**Reflejo en el sistema:** ninguna de las dos toca el entorno (`tumor_env.py`) ni las
EDOs (`dynamics.py`) ni la métrica (`evalutils.py`). Reusan el mismo `Agent`, los mismos
extractores de observación y el mismo PPO de `mappo.py`. Es una capa de *evaluación
adversarial* sobre el marco existente — congruente con la decisión bloqueada de "un solo
framework".

---

## 3. Cómo se interpreta

| Observación | Interpretación |
|---|---|
| `explotabilidad ≈ 0` | Ninguno de los atacantes entrenados degradó al partner original; es evidencia de robustez dentro de la clase evaluada, no prueba de equilibrio óptimo. |
| `explotabilidad` grande y positiva | Un atacante dedicado la degrada mucho → la política estaba **sobreajustada a su partner**; su TTP de self-play era optimista. |
| `expl(MAPPO) < expl(IPPO)` (Wilcoxon pareado, unilateral) | Evidencia de que CTDE produce políticas *menos explotables* bajo semillas comunes. |
| `TTP_BR(MAPPO) > TTP_BR(Gatenby)` | Ambas estrategias se comparan bajo atacantes dedicados de igual presupuesto; evita trasladar el valor nominal de Gatenby a otro régimen adversarial. |
| cross-play alto y simétrico al self-play | La política **generaliza** a adversarios no vistos (no memorizó a su partner). |
| cross-play que colapsa | Overfitting al compañero de self-play; el TTP reportado no es transferible. |

**Falla esperada del argumento:** si `TTP_best_response` colapsara al nivel de MTD (~13 d)
para *ambos* métodos, el hallazgo central quedaría en duda (la terapia solo funcionaba
contra tumores tontos). Que **no** colapse es, en sí, evidencia a favor del marco.

---

## 4. Qué valor le da a una tesis de Ciencia de la Computación

Este benchmark es lo que convierte el trabajo de "apliqué MARL a un modelo de cáncer" en
una **contribución de CS defendible**:

1. **Rigor de la evaluación, no solo del método.** En RL adversarial, medir contra el
   propio partner de self-play es un error metodológico conocido. Adoptar
   **explotabilidad / best-response** — el estándar en teoría de juegos computacional
   (equilibrios aproximados, NashConv, exploitability en póker/AlphaStar) — muestra dominio
   del estado del arte en *cómo se evalúa* un sistema multiagente, no solo en cómo se entrena.

2. **Aísla la contribución de CTDE con una causa mecanística.** El crítico centralizado ve
   el estado conjunto → mejor asignación de crédito bajo no-estacionariedad → política menos
   explotable. La explotabilidad **operacionaliza y mide** esa hipótesis (anclada a
   de Witt et al., 2020), en vez de afirmarla. Convierte "CTDE es mejor" en "CTDE es un
   *X%* menos explotable, con p=…".

3. **Rompe la circularidad de la validación** (una de las gotchas del proyecto). El atacante
   dedicado es un evaluador *independiente* del entrenamiento: un tercero adversarial, no un
   juez entrenado para perder. Es la versión adversarial del principio "validación no circular".

4. **Transferible más allá del dominio clínico.** El resultado interesante para CS es sobre
   **robustez de políticas MARL bajo un adversario acotado**; el glioblastoma es el banco de
   pruebas. Explotabilidad + cross-play son métricas reusables en cualquier juego general-sum,
   lo que da a la tesis un aporte metodológico independiente del caso de estudio.

5. **Prepara el terreno para trabajo futuro medible.** La misma métrica evaluará el impacto de
   un adversario estructuralmente más fuerte (league/population-based) y de la futura variable
   de salud del paciente: cualquier extensión se juzga por si **baja la explotabilidad**, no
   por una media nueva.

---

## 5. Protocolo (pre-registrado)

- **Semillas:** n=15 (bimodalidad → mediana [min,max] + tasa de éxito, no media±std).
- **Pasos:** 120 000 por entrenamiento (self-play, IPPO y cada reinicio best-response), igual que la línea base.
- **Régimen:** `phi_max=0.05` (titular). Extensible al barrido de EXP-2.
- **Atacante:** cinco reinicios por política; se conserva el menor TTP encontrado.
  Gatenby recibe el mismo número de reinicios y presupuesto.
- **Estadística:** Wilcoxon unilateral pareado por semilla sobre `explotabilidad`
  (H1: MAPPO < IPPO) y McNemar exacto para superar a Gatenby bajo best-response.
- **Reanudable:** modelos V2 en `outputs/models/`, estado en
  `outputs/benchmark_exploit_v2.json`. Re-ejecutar continúa donde quedó.
- **Salidas:** `outputs/benchmark_exploit_v2.md` (tablas) + `outputs/benchmark_exploit_v2.png`
  (caída self-play → best-response por método).

**Comando:**
```bash
python scripts/benchmark_exploit.py --seeds 15 --steps 120000 --br-restarts 5
python scripts/benchmark_exploit.py --smoke                      # validación rápida (2 semillas, 3k)
python scripts/benchmark_exploit.py --plot                       # regenera tablas/figura desde json
```

---

## 6. Criterios de éxito (falsables, fijados antes de correr)

- **H1 (principal):** `expl(MAPPO) < expl(IPPO)` con p<0.05 (Wilcoxon pareado). → CTDE produce
  políticas menos explotables.
- **H2 (validación del hallazgo):** `mediana(TTP_BR(MAPPO)) > TTP_BR(Gatenby)`. → el margen
  sobre Gatenby sobrevive al comparar ambas estrategias con atacantes dedicados.
- **H3 (generalización):** cross-play de MAPPO no cae por debajo de su TTP self-play en más de
  un ~20%. → la política no está sobreajustada a su partner.

Si H1 falla pero H2 se cumple, el mensaje honesto es: "CTDE e IPPO son igualmente robustos,
pero ambos superan a las heurísticas incluso bajo ataque dedicado" — sigue siendo un
resultado publicable y no se maquilla.

---

## 7. Resultados V2 vigentes (n=15, φ_max=0.05, 120k pasos)

La corrida titular completa usó cinco reinicios de best-response por terapia y cinco
para Gatenby. La baseline Gatenby bajo su propio atacante alcanzó 7 días. MAPPO obtuvo
TTP self-play 44 [24,58], best-response 14 [11,14] y explotabilidad 30 [11,44]; IPPO
obtuvo 29 [26,37], 11 [10,12] y 18 [14,27], respectivamente. El contraste unilateral
pre-registrado de menor explotabilidad MAPPO<IPPO no fue significativo (`p=0.993`).

En cross-play, MAPPO frente a la población de tumores IPPO obtuvo 35 [24,41] días,
mientras IPPO frente a tumores MAPPO obtuvo 27 [12,29]. Las 15 semillas de ambos
métodos superaron los 7 días de Gatenby bajo best-response.

La conclusión es deliberadamente matizada: MAPPO conserva un nivel absoluto de TTP
mayor frente al mejor atacante y transfiere mejor en cross-play, pero su brecha desde
self-play es mayor en esta corrida. Por tanto, no se afirma que MAPPO sea menos
explotable; la ventaja de CTDE debe describirse como dependiente del régimen y del
criterio de evaluación.

### Veredicto V2 por hipótesis

- **H1 (menor explotabilidad de MAPPO): no se confirma.** La estimación puntual favorece
  a IPPO en la brecha (`30` frente a `18` días), y el contraste pre-registrado MAPPO<IPPO
  da `p=0.993`.
- **H2 (MAPPO supera a Gatenby bajo ataque dedicado): se confirma en este régimen.** La
  mediana MAPPO best-response es `14 d`, mayor que los `7 d` de Gatenby; las 15 semillas
  superan ese umbral. Esto compara ambos tratamientos con atacantes dedicados, aunque
  no constituye una garantía contra un adversario óptimo.
- **H3 (cross-play MAPPO no cae más de ~20%): queda borderline.** MAPPO pasa de 44 a
  35 días (caída aproximada de 20.5%), apenas por encima del umbral heurístico; IPPO
  pasa de 29 a 27 días. La transferencia de MAPPO es mejor en nivel absoluto, pero el
  criterio no debe declararse cumplido estrictamente.

Artefactos: `outputs/benchmark_exploit_v2.json`, `outputs/benchmark_exploit_v2.md` y
`outputs/benchmark_exploit_v2.png`.

## 8. Resultados históricos V1 (n=10, φ_max=0.05, 120k pasos)

Los valores siguientes documentan la corrida original y **no validan el protocolo V2**.
No deben mezclarse con los artefactos `*_v2` ni citarse como resultados corregidos.

Corrida: `outputs/benchmark_exploit.{md,png,json}`. Régimen titular. Reanudable, 60/60 modelos.

### Explotabilidad (best-response)

| Método | TTP self-play | TTP best-response | Explotabilidad (Δ) | éxito best-resp vs Gatenby |
|---|---|---|---|---|
| **MAPPO (CTDE)** | 48 [29,61] | **45 [27,53]** | 0 [-7,21] | **90%** |
| IPPO (local) | 29 [23,31] | 27 [23,29] | 2 [0,3] | 20% |

### Cross-play (adversario no visto)

| Terapia | Adversario | TTP cross-play |
|---|---|---|
| MAPPO | tumor IPPO | 37 [24,65] |
| IPPO | tumor MAPPO | 27 [23,28] |

### Veredicto por hipótesis

- **H1 (expl. MAPPO < IPPO, MWU) — NO se cumple** (p=0.323). La *brecha* de explotabilidad no
  separa los métodos: **cada uno ya está cerca de su propio peor caso** (el self-play convergió
  a un equilibrio difícil de explotar más). MAPPO Δ mediana 0, IPPO Δ mediana 2 — ambos bajos.
- **H2 (MAPPO best-response > 27) — se cumple con holgura.** Mediana **45 ≫ 27**, y **90% de las
  semillas superan a Gatenby incluso bajo un atacante dedicado.** El margen sobre la heurística
  clínica **no era un artefacto** del adversario débil de self-play.
- **H3 (caída cross-play ≤ ~20%) — borderline.** Caída del **23%** (48→37), justo sobre el
  umbral, pero 37 ≫ 27 (Gatenby): MAPPO **generaliza** a un adversario no visto sin colapsar.

### El diferenciador real que emergió

CTDE no se distingue de IPPO por la *brecha* de explotabilidad (H1), sino por el **nivel
absoluto** en que queda cada política bajo ataque dedicado:

- **Piso best-response:** MAPPO **45 d** vs IPPO **27 d** — el peor caso de MAPPO iguala el
  *mejor* caso típico de IPPO.
- **Supervivencia bajo ataque:** **90% (9/10) vs 20% (2/10)** de semillas superan a Gatenby.
  **Fisher exact unilateral: p=0.0027, OR=36.** Este es el contraste honesto y significativo,
  no la media frágil ni el t-test inválido.

**Lectura:** el aporte de CTDE es de **robustez absoluta**, no de menor explotabilidad relativa.
Ambos métodos son difíciles de explotar *más allá de su self-play*, pero MAPPO se estabiliza en
un régimen mucho mejor (piso ~45 d, casi siempre sobre la heurística clínica) mientras IPPO se
estabiliza al nivel de Gatenby. La métrica correcta para reportarlo es la **tasa de éxito bajo
best-response (Fisher)**, no la brecha Δ. El pre-registro de H1 falló, pero H2 + el contraste de
Fisher dan un resultado más fuerte y clínicamente legible que el previsto.

> Implicación para la tesis: el benchmark **validó el hallazgo central bajo el estándar más
> exigente** (un adversario que ataca a propósito) y **descartó** que la ventaja de MAPPO fuera
> un espejismo del partner de self-play. Reencuadra el mensaje del Cap.4/5: CTDE aporta un *piso
> de desempeño robusto*, medido por explotabilidad absoluta, no por una media.
