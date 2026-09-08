# Informe de avance — Semana 5 (PFC II)

**Tesis:** Mitigación de la resistencia evolutiva en glioblastoma multiforme mediante
aprendizaje por refuerzo multiagente adversarial (MAPPO-CTDE).
**Integrantes:** Kalos B. Lazo Mera · Gianpier A. Segovia Ureta · **Asesor:** V. E. Martínez Abaunza.
**Ciclo:** 2026-2. **Semana 5** (7–11 sep). **Fecha del informe:** 7 sep 2026.

> Documento para la reunión de seguimiento. Resume dos avances de esta semana, sus
> resultados y su encaje con el cronograma, y plantea una **decisión de alcance** sobre
> una extensión nueva (la variable de salud del paciente).

---

## 1. Contexto en el cronograma

Las semanas S1–S4 quedaron cerradas: línea base congelada (tag `v1.0`), protocolo
experimental, EXP-1 (robustez) y EXP-2 (adversario fortalecido). Esta semana se avanzó en
**dos frentes**:

- **(A) Benchmark de explotabilidad** — profundiza la validación de robustez y la
  diferenciación CTDE vs IPPO (alimenta la Discusión, Cap. 5, y el hito EC1 de S8).
- **(B) Variable de salud del paciente, Fase 1** — una **extensión de alcance nueva**, no
  prevista en el cronograma original, presentada aquí como prueba de concepto para decidir
  con el asesor si se incorpora formalmente a PFC II.

---

## 2. Avance A — Benchmark de explotabilidad (completado)

### Qué se implementó
Un experimento que mide la **robustez adversarial** de las terapias aprendidas siguiendo el
estándar de teoría de juegos computacional: se **congela** cada política de terapia y se
entrena, desde cero, un **adversario tumoral dedicado exclusivamente a romperla**
(*best-response*). La explotabilidad es cuánto cae el desempeño frente a ese atacante.
Código: `gbmarl/exploit.py`, `scripts/benchmark_exploit.py`. **Ya integrado a `main` y
documentado en el Cap. 4** (`sec:res-explotabilidad`).

### Por qué importa
Cierra la crítica más peligrosa a la tesis: *"tu terapia solo ganaba porque el adversario de
self-play era débil"*. El best-response es un evaluador independiente que ataca a propósito.

### Resultados (n=10, φ_max=0.05)

| Método | TTP self-play | **TTP bajo ataque dedicado** | **Éxito vs Gatenby bajo ataque** |
|---|---|---|---|
| **MAPPO (CTDE)** | 48 d | **45 d** [27, 53] | **90 % (9/10)** |
| IPPO (local) | 29 d | 27 d [23, 29] | 20 % (2/10) |

Contraste (prueba exacta de Fisher, unilateral): **p = 0.0027, OR = 36**.

### Evaluación: ¿buen resultado? Sí, y honesto.
- **A favor:** incluso bajo un atacante dedicado, MAPPO mantiene 45 días y supera a la
  heurística clínica de Gatenby en 9 de cada 10 semillas. El margen **no era un artefacto**.
- **Matiz honesto (se reporta, no se esconde):** la *brecha* de explotabilidad no distingue
  los métodos (ambos ya están cerca de su peor caso; Mann-Whitney p=0.323). La ventaja de
  CTDE es de **nivel absoluto** (su peor caso ≈ el mejor caso típico de IPPO), no de menor
  degradación. El mensaje del Cap. 5 se reencuadra a "CTDE aporta un piso de desempeño robusto".

---

## 3. Avance B — Variable de salud del paciente, Fase 1 (prueba de concepto)

### Qué es y por qué
Hasta ahora el modelo seguía 3 variables (sensibles S, resistentes R, fármaco c) y el paciente
**no podía morir por el tratamiento** — la toxicidad era solo un castigo numérico menor. Se
añade una **cuarta variable H = salud del paciente** [0,1], con dinámica realista:

```
dH/dt = ρ_H·(1 − H)  −  κ_u·u  −  κ_b·(S+R)
        (se recupera)   (fármaco)  (tumor)
```

El paciente se recupera con el tiempo, la dosis lo intoxica y el tumor lo desgasta. Si su salud
cae bajo un umbral, muere: un **tercer modo de falla** (salud), además de carga y resistencia.

### Cómo se aplica y se integra
Tres módulos nuevos: la ecuación (`dynamics_health.py`), el entorno donde el paciente puede
morir por toxicidad (`health_env.py`) y el entrenamiento (`mappo_health.py`, reusa el mismo
MAPPO). **Todo es aditivo:** NO se modificó el código 3-D existente. Se verificó que la línea
base sigue intacta (**los 10 tests originales pasan; 14/14 con los nuevos**). Vive en una
**rama aislada** (`worktree-variable-salud`), separada de `main`; se fusionará solo si la
validación completa confirma su valor.

### Resultados (prueba de concepto, 120k pasos, 1 semilla)

| Terapia | Días controlados | Muere por | **Salud media** |
|---|---|---|---|
| Sin tratar | 12 | carga | 0.95 |
| MTD (dosis máxima) | 9 | resistencia | **0.61** |
| Pulsado (heurística) | 19 | resistencia | 0.91 |
| **MAPPO (aprendido)** | **26** | carga | **0.91** |

### Evaluación: ¿buen resultado?
- **A favor (prueba de concepto exitosa):** el entorno de 4 variables entrena y la política
  aprendida supera a todas las heurísticas (26 d). **Mecanismo emergente:** MAPPO mantiene la
  salud en 0.91 —casi como no tratar (0.95)— **sin recompensa explícita de salud**; la
  dosificación pulsada protege al paciente como efecto colateral. Además **cuantifica el costo
  oculto de MTD** (salud 0.61 vs 0.91), que la métrica anterior no capturaba.
- **Aún no es resultado de tesis (honesto):** es 1 sola semilla (no estadística); las
  constantes (ρ_H, κ_u, κ_b, H_min) son ilustrativas, no calibradas; y el modo de falla
  "salud" todavía no domina (MTD muere de resistencia antes que de toxicidad). Forzarlo
  ajustando constantes sería manipulación metodológica y **no se hizo**.

---

## 4. Plan respecto a la variable de salud

**Fase 1 (actual):** salud como variable pasiva (EDO). Prueba de concepto lista. ✅

**Para cerrar la Fase 1 como resultado de tesis** (próximas semanas):
1. Corrida **multi-semilla** (n≥10): mediana + tasa de éxito (no media, por bimodalidad).
2. **Ablación con-H vs sin-H** sobre las mismas semillas: aislar el efecto de la variable.
3. **Análisis de sensibilidad** de ρ_H, κ_u, κ_b, H_min (±30 %): que el ranking sobreviva.
4. **Explotabilidad** de las políticas con salud (mismo estándar que el Avance A).

**Fase 2 (futura, a decidir):** convertir la salud en un **tercer agente "clínico"** que
optimiza calidad de vida vs. supervivencia → juego general-sum de 3 agentes con CTDE. Es la
contribución MARL más ambiciosa, pero mayor esfuerzo. Detalle en
`docs/variable_salud_paciente.md`.

**Nota de honestidad metodológica:** un cuerpo no "optimiza", así que si la salud pasa a ser
agente, se modela como **clínico** (que sí decide un trade-off real), no como "el paciente".

---

## 5. Preguntas para el asesor

1. **Alcance:** ¿se incorpora la variable de salud formalmente al alcance de PFC II, o se deja
   como trabajo futuro documentado? (La Fase 1 ya es demostrable; la Fase 2 es un compromiso mayor.)
2. **Prioridad:** ¿conviene cerrar primero la validación completa de la salud (Fase 1), o
   retomar el hilo original del cronograma (EXP-3 bimodalidad, S5) antes?
3. **Calibración de la salud:** ¿hay literatura de toxicidad de TMZ que podamos usar para anclar
   κ_u/H_min, en lugar de constantes ilustrativas?
