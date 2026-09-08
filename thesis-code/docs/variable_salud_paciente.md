# Variable de salud del paciente — plan de diseño (GBMARL)

**Tesis:** Mitigación de la resistencia evolutiva en glioblastoma multiforme mediante
aprendizaje por refuerzo multiagente adversarial (MAPPO-CTDE).
**Autores:** Kalos B. Lazo Mera · Gianpier A. Segovia Ureta · **Asesor:** V. E. Martínez Abaunza.
**Estado:** DISEÑO — **no implementado**. Este documento fija opciones, la realidad de
cada una y un plan por fases *antes* de tocar código.
**Depende de:** el benchmark de explotabilidad (`docs/benchmark_explotabilidad.md`) como
**instrumento de evaluación** — no como dependencia de código.

> Extiende el marco de 2 agentes (terapia vs tumor) con una tercera dimensión clínica:
> el estado del paciente. Responde a: *¿cómo modelamos que el tratamiento también daña
> al paciente, y quién decide ese trade-off?* — sin caer en antropomorfizar la fisiología.

---

## 1. Qué problema resuelve (y por qué importa)

Hoy la toxicidad es un **castigo lineal plano** (`− tox_weight·u`) en la recompensa de la
terapia. Eso tiene dos límites:

1. **No hay estado de paciente.** El único modo de "perder" es que el tumor progrese o se
   vuelva resistente. Un tratamiento no puede *matar al paciente por toxicidad* — que es
   justamente el riesgo real de MTD (dosis máxima tolerada).
2. **El trade-off supervivencia vs. calidad de vida no está modelado.** La historia central
   de la tesis ("retrasar, no curar") gana una segunda dimensión honesta: retrasar la
   progresión *manteniendo al paciente vivo y funcional*, no a cualquier costo.

Agregar una variable de salud `H` cierra ese hueco y, según la opción, abre una
**contribución MARL nueva** (un tercer decisor estratégico legítimo).

---

## 2. Opciones de diseño (con la realidad de cada una)

### Opción A — Salud como 4ª variable de estado `H` (EDO acoplada) · sin nuevo agente
Se añade una ecuación al sistema de `dynamics.py`:

```
dH/dt = ρ_H·(1 − H) − κ_u·u − κ_b·(S + R)
```

- `H ∈ [0,1]` (1 = sano). Se recupera solo hacia 1 a tasa `ρ_H`; la dosis lo daña (`κ_u·u`)
  y la carga tumoral lo daña (`κ_b·(S+R)`).
- **Estado 3-D → 4-D:** `(S, R, c, H)`. La terapia observa `H` (el médico mide el estado del
  paciente); el crítico centralizado ve `(S,R,c,H)`.
- **Nuevo modo de falla:** `H < H_min` → muerte del paciente (por toxicidad o por tumor).
  Modos de falla pasan de 2 (carga, resistencia) a **3** (+ salud).
- **Recompensa:** se sustituye el castigo plano por un término mecanístico (p. ej. penalización
  terminal si muere por toxicidad, o recompensa por día vivo con `H` sobre umbral).

**Realidad:** limpia, defendible, bajo riesgo. Sigue siendo un juego de **2 agentes**
(no aporta MARL nuevo, sí realismo clínico). Requiere valores para `ρ_H, κ_u, κ_b`: se
anclan a literatura o se tratan como constantes de diseño con **análisis de sensibilidad**
(igual que `KILL_TO_GROWTH_RATIO`), documentando que son ilustrativas si no se calibran.

### Opción B — Tercer AGENTE = "clínico/guardián" (cooperativo con la terapia)
Sobre la Opción A, se agrega un tercer agente PettingZoo, el **clínico**, que actúa como un
factor de atenuación `g ∈ [0,1]` sobre la dosis solicitada: `u_efectiva = g · u_terapia`.
Su recompensa optimiza **calidad de vida** (mantener `H` alto) — posiblemente combinada con
supervivencia. El juego pasa a **general-sum de 2-vs-1**: equipo {terapia, clínico} vs tumor.

**Realidad:** es la **contribución MARL genuina** (cooperativo-competitivo mixto, no zero-sum
puro; CTDE con crítico condicionado al estado conjunto de 3 agentes). Cuesta más ingeniería:
`tumor_env.py` y `mappo.py` están cableados a 2 agentes → hay que generalizar a N. Riesgo de
que el clínico colapse a "dosis 0" si su recompensa no equilibra bien supervivencia vs. QoL.

### Opción C — Salud solo en la recompensa (multi-objetivo), sin nueva dimensión
Es esencialmente lo que ya hace `tox_weight`. **Descartada** — no aporta nada nuevo.

### La realidad honesta (el punto que un revisor de CS atacará)
Un **cuerpo biológico no es un optimizador racional**. Modelar "la salud del paciente" como
un agente RL que maximiza una recompensa es antropomorfizar la fisiología. Por eso:
- Si la salud es **pasiva → Opción A** (EDO). Correcto y sin polémica.
- Si queremos que sea **agente → NO lo llamamos "el paciente", lo llamamos "el clínico"**
  (Opción B). Un clínico *sí* optimiza un trade-off real (supervivencia vs. calidad de vida):
  es un actor estratégico legítimo. Esta distinción es central para la defensa.

---

## 3. Plan por fases (recomendado)

**Fase 1 — Opción A (base obligatoria, bajo riesgo).**
`H` como 4ª EDO, toxicidad mecanística, tercer modo de falla ("salud"). El juego sigue en 2
agentes. Se evalúa con el benchmark de explotabilidad: ¿la terapia adaptativa que preserva
salud sigue siendo poco explotable? Entregable: la métrica gana una dimensión y MTD revela su
costo tóxico oculto.

**Fase 2 — Opción B (novedad MARL, solo si la Fase 1 es sólida).**
Agente clínico sobre A → juego general-sum de 3 agentes con CTDE. Aquí crece la contribución
de CS (generalizar `mappo.py` a N agentes; crítico centralizado sobre estado conjunto de 3).

*Regla:* no se pasa a Fase 2 hasta que la Fase 1 esté validada con el benchmark. Una decisión
a la vez.

---

## 4. Cómo se refleja en el sistema

| Componente | Cambio (Fase 1 / Fase 2) |
|---|---|
| `dynamics.py` | +1 ecuación `dH/dt`; `rk4_step` integra 4 variables (F1). |
| `tumor_env.py` | estado 4-D; obs incluye `H`; nuevo `terminated` por `H<H_min`; recompensa con término de salud (F1). Tercer agente `clinico` con acción `g∈[0,1]` (F2). |
| `mappo.py` | crítico centralizado ve `(S,R,c,H)` (F1). Generalizar de 2 a N agentes (F2). |
| `evalutils.py` | `ttp_combinado` reporta el modo "salud"; nueva métrica de QoL (área bajo `H(t)`). |
| `config.py` | params `ρ_H, κ_u, κ_b, H_min, H0` con su derivación documentada. |

Nada de esto rompe las decisiones bloqueadas: sigue siendo PettingZoo `ParallelEnv`, un solo
framework PyTorch, y el término `−φ·S` de conservación de masa intacto.

---

## 5. Cómo se interpreta

| Observación | Interpretación |
|---|---|
| MTD gana muchos episodios por falla de **salud** (no de tumor) | Confirma el costo tóxico oculto de dosis máxima → refuerza la historia "adaptativo > MTD" con evidencia nueva. |
| La terapia adaptativa (MAPPO) mantiene `H` alto **sin** entrenarla para ello | La dosificación pulsada crea ventanas de recuperación → el mecanismo aprendido **también** protege la salud (beneficio emergente). |
| Agregar `H` no cambia el ranking ni la brecha de explotabilidad CTDE>IPPO | La salud se integra **sin romper** el hallazgo central (robustez del resultado). |
| Fase 2: el clínico aprende a atenuar dosis en ventanas de `H` bajo y el TTP se mantiene | Beneficio cooperativo de CTDE con 3 agentes (coordinación terapia↔clínico). |
| El clínico colapsa a `g≈0` (no tratar) | Recompensa mal balanceada → hay que penalizar la progresión tumoral en su objetivo. |

---

## 6. Qué valor le da a una tesis de Ciencia de la Computación

1. **Eleva el juego de 2 a 3 agentes general-sum (Fase 2).** Pasar de un juego casi zero-sum
   (terapia vs tumor) a uno **cooperativo-competitivo mixto** con CTDE es una contribución MARL
   de mayor nivel: coordinación de equipo bajo un adversario, no solo competencia 1-a-1.

2. **RL multi-objetivo / con restricciones.** La salud como objetivo o restricción conecta con
   *constrained RL* y *multi-objective RL* — literatura de CS reconocida, no solo modelado clínico.

3. **Recompensa honesta sin la trampa de "minimizar carga".** Añade un costo real (la salud del
   paciente) manteniendo el objetivo de *retrasar la intratabilidad*, no de reducir tumor. Evita
   la trampa que hace perder a MTD y a la vez la *explica* mecánicamente.

4. **Credibilidad clínica del relato.** "Retrasar, no curar" gana una segunda dimensión medible:
   retrasar la progresión **manteniendo al paciente vivo y funcional**. La métrica de QoL (área
   bajo `H(t)`) es interpretable para un asesor médico.

5. **Se evalúa con la misma vara adversarial.** El impacto de la salud se juzga por si **baja o
   no rompe la explotabilidad** (Sección 8), no por una media nueva — coherente con el estándar
   de evaluación ya adoptado.

---

## 7. Protocolo (pre-registrado, para cuando se implemente)

- **Semillas:** n≥10 (bimodalidad → mediana [min,max] + tasa de éxito).
- **Métrica primaria:** TTP-combinado con **tercer modo de falla "salud"**.
- **Métrica secundaria (QoL):** área bajo `H(t)` normalizada por horizonte.
- **Baselines:** Sin tratar, MTD, Gatenby — reevaluados con `H` (se espera que MTD sufra por salud).
- **Ablación:** con `H` vs. sin `H` (línea base v1.0), para aislar el efecto de la variable.
- **Sensibilidad:** `ρ_H, κ_u, κ_b, H_min` ±30% (protocolo de EXP-1); el ranking debe sobrevivir.
- **Robustez:** benchmark de explotabilidad sobre las políticas con salud (Sección 8).

---

## 8. Dependencia con el benchmark de explotabilidad

- **De código:** ninguna. La variable de salud se construye sobre `dynamics.py`/`tumor_env.py`;
  no importa `gbmarl/exploit.py`.
- **De evaluación:** total. Toda política entrenada con salud se somete al benchmark de
  explotabilidad: se congela y se le entrena un atacante dedicado. El criterio no es "¿subió el
  TTP?" sino "¿la política con salud es **igual o menos explotable** que sin salud?". Así la
  extensión se valida con el mismo estándar adversarial, sin maquillar con una media favorable.

---

## 9. Criterios de éxito (falsables, fijados antes de implementar)

- **H1 (mecanismo emergente):** la terapia adaptativa MAPPO mantiene la mediana de `H` sobre
  `H_min` más tiempo que MTD (la dosificación pulsada protege la salud sin entrenarla para ello).
- **H2 (robustez del hallazgo):** agregar `H` **no** colapsa la brecha de explotabilidad
  CTDE>IPPO ni el margen sobre Gatenby → la salud se integra sin romper el resultado central.
- **H3 (novedad, Fase 2):** el equipo de 3 agentes {terapia, clínico} logra mayor
  TTP·QoL conjunto que la configuración de 2 agentes con castigo de toxicidad plano.

Si H1 falla (la terapia adaptativa **no** protege la salud espontáneamente), el mensaje honesto
es que la salud debe entrar como objetivo explícito → justifica directamente la Fase 2 (agente
clínico). Ningún resultado se descarta; cada uno redirige el plan.
