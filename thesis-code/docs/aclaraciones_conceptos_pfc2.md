# Aclaraciones de conceptos — EXP-1 y EXP-2 (PFC II)

> **Nota de vigencia (2026-09-08):** los ejemplos numéricos y las tablas de
> EXP-2 de este documento pertenecen a corridas históricas. No deben citarse
> como resultados de la calibración vigente. Los resultados recalculados se
> encuentran en `outputs/calibracion_real_2026-09-08.md`,
> `outputs/validacion_calibrada_15semillas_2026-09-08.md` y
> `outputs/exp1_robustez_v2.md`; EXP-2 sigue parcial (4/90 celdas).

Documento de apoyo para entender los términos del informe de avance. Escrito con
ejemplos concretos. Cubre: (1) qué es `K`, (2) ruido de observación vs. perturbación de
parámetros, (3) qué significan `ic50`, `×0.9`, `×1.2`, etc., y (4) la **interpretación**
de los resultados de la Semana 4 (EXP-2).

---

## 1. ¿Qué es `K` y cómo influye?

`K` es la **capacidad de carga** (*carrying capacity*): el **tamaño tumoral máximo** que el
entorno puede sostener. Aparece en el término logístico de las EDOs:

```
dS/dt = α_S · S · (1 − (S+R)/K) − ...
dR/dt = α_R · R · (1 − (S+R)/K) + ...
```

- Cuando la carga total `S+R` se acerca a `K`, el factor `(1 − (S+R)/K)` tiende a 0 y el tumor
  **deja de crecer** (no hay recursos/espacio). Está normalizada: en la línea base `K = 1.0`.

**Por qué es el parámetro más influyente (dos razones):**

1. **Es el motor de la competencia entre poblaciones.** La estrategia que aprende la terapia
   (terapia adaptativa: dejar vivas células sensibles para que *compitan* con las resistentes por
   ese `K` compartido) **depende** de que haya "espacio" que disputar. Si `K` baja, ese
   colchón competitivo se encoge y la estrategia deja de funcionar como fue entrenada.

2. **Mueve el poste de la meta.** El umbral de falla por carga es `prog_thr = 0.80 · K`. Si `K`
   baja 20%, el umbral también baja 20%: el tumor "progresa" (falla) con menos células.
   → Es más fácil perder.

**Ejemplo del resultado:** con `K × 0.8` (capacidad 20% menor), la mediana de TTP se desploma de
**35 → 12 días**. Interpretación honesta: *si el glioblastoma real tolera menos densidad de la que
calibramos, la política pierde casi toda su ventaja.* Es un límite de validez que reportamos, no
lo escondemos.

---

## 2. Ruido de observación vs. perturbación de parámetros

Son **dos tipos de incertidumbre distintos**. La analogía rápida:

| | Perturbación de parámetros | Ruido de observación |
|---|---|---|
| Qué cambia | **La biología real** (las constantes de las EDOs) | **La medición** de lo que ve el médico |
| Naturaleza | Estructural, permanente | Aleatorio, cambia cada día |
| Analogía | "El paciente es distinto al del modelo" | "El escáner/analítica mide con error" |
| ¿Se re-entrena? | No: se evalúa la política ya aprendida | No: solo se ensucia la observación en evaluación |

### 2a. Perturbación de parámetros (EXP-1a)

Las constantes del simulador (crecimiento, IC50, `K`, decaimiento del fármaco) vienen de
**calibración con datos + literatura**, y tienen incertidumbre. Preguntamos: *si el valor real
fuera 20% distinto al que calibramos, ¿la política sigue funcionando?*

- **Ejemplo:** `alpha_S` (velocidad de crecimiento de las sensibles) calibrado = 0.15. Lo cambiamos
  a 0.18 (`×1.2`, +20%) y **volvemos a simular** un tumor que crece más rápido. La política —que
  fue entrenada con 0.15— se enfrenta a ese tumor distinto **sin re-entrenar**. Resultado:
  TTP 35 → 23. Es decir, si las sensibles crecen más rápido de lo calibrado, la terapia rinde
  menos.

### 2b. Ruido de observación (EXP-1b)

En la vida real, el médico **no conoce el estado exacto** del tumor: lo estima por imagen
(volumetría) y análisis, con error. En el modelo, la terapia observa `(S+R, c)` = (carga total,
nivel de fármaco). El ruido de observación **ensucia esa lectura** justo antes de que el agente
decida la dosis, **pero la dinámica real avanza con el valor verdadero**.

- **Modelo de ruido:** `valor_medido = valor_real × (1 + ε)`, con `ε` gaussiano de desviación `σ`.
- **Ejemplo (σ = 10%):** la carga real es `0.50`, pero el agente "lee" `0.54` (error de +8%) o
  `0.47` (−6%), distinto cada día. Preguntamos: *¿la política sigue tomando buenas decisiones con
  mediciones imperfectas?*
- **Resultado:** muy robusta — la mediana de TTP casi no se mueve (35 → 37/38) incluso con σ = 20%.
  Esto es **buena noticia para la validez traslacional**: una política que solo funciona con
  mediciones perfectas no sirve en la clínica; la nuestra tolera error de medición.

---

## 3. ¿Qué son `ic50`, `×0.9`, `×1.2`, etc.?

### Los parámetros perturbados (todas son constantes de las EDOs)

| Parámetro | Qué representa | Valor base | Regla mnemotécnica |
|---|---|---|---|
| `ic50_S` | Concentración de fármaco para media-muerte de **sensibles** | ≈ 0.36 | **más bajo = fármaco más potente** (mata con poco) |
| `ic50_R` | Concentración para media-muerte de **resistentes** | ≈ 3.27 | resistentes necesitan ~9× más fármaco |
| `delta_max_S` | Muerte máxima alcanzable en sensibles | calibrado | techo de eficacia sobre sensibles |
| `delta_max_R` | Muerte máxima alcanzable en resistentes | calibrado | techo de eficacia sobre resistentes |
| `alpha_S` | Velocidad de crecimiento de sensibles | 0.15 | más alto = tumor crece más rápido |
| `alpha_R` | Velocidad de crecimiento de resistentes | 0.10 | resistentes crecen más lento (costo de resistencia) |
| `lambda_c` | Velocidad de eliminación del fármaco del cuerpo | 0.40 | más alto = el fármaco dura menos |
| `K` | Capacidad de carga (ver sección 1) | 1.0 | tope de tamaño tumoral |

> **IC50** = *Half-maximal Inhibitory Concentration*: la concentración de fármaco a la que se logra
> la **mitad** del efecto de muerte máximo. Es la medida estándar de **potencia** de un fármaco en
> farmacología (viene directo de los datos GDSC2). La brecha IC50_R/IC50_S ≈ 9× es lo que hace que
> los resistentes sobrevivan a dosis que matan a los sensibles.

### El factor `×0.8 … ×1.2`

Es un **multiplicador** aplicado al valor calibrado para probar ±20% de incertidumbre:

| Factor | Significado |
|---|---|
| `×0.8` | valor **20% menor** que el calibrado |
| `×0.9` | 10% menor |
| `×1.0` | **valor nominal** (el calibrado; sin cambio) |
| `×1.1` | 10% mayor |
| `×1.2` | 20% mayor |

- **Ejemplo con `ic50_S` (base 0.36):**
  - `×0.8` → 0.29 → el fármaco es **más potente** sobre sensibles (las mata con menos) → TTP sube (39).
  - `×1.2` → 0.43 → el fármaco es **menos potente** (las sensibles aguantan más) → TTP baja (23).

Así, cada fila de la tabla de EXP-1a dice: *"si este parámetro estuviera mal calibrado en ±20%,
¿cuánto cambiaría el desempeño?"*. Si la fila casi no se mueve (`ic50_R`, `delta_max_R`,
`lambda_c`), la conclusión es robusta a ese parámetro.

---

## 4. Interpretación de la Semana 4 (EXP-2, adversario fuerte)

### Primero: ¿qué es `phi_max`?

`phi` (φ) es la **acción del agente tumoral**: la tasa a la que convierte células sensibles en
resistentes (S→R) cada día. `phi_max` es el **tope** de esa tasa = **qué tan fuerte es el arma del
tumor**.

- `phi_max = 0.05` (nominal): adversario "débil" de la línea base.
- `phi_max = 0.20`: adversario **4× más agresivo** (puede generar resistencia mucho más rápido).

El objetivo de EXP-2 es doble: (a) ver cuánto aguanta la terapia cuando el tumor es más peligroso,
y (b) responder la pregunta abierta del póster: **¿cuándo aporta el crítico centralizado (CTDE /
MAPPO) frente al aprendizaje independiente (IPPO)?**

### La tabla, en palabras

| φ_max | MAPPO (CTDE) | IPPO (local) | p | Éxito MAPPO | Éxito IPPO |
|---|---|---|---|---|---|
| 0.05 | 36 [25,61] | 29 [23,32] | 0.009 | 93% | 80% |
| 0.10 | 27 [22,61] | 27 [23,31] | 0.265 | 47% | 33% |
| 0.20 | 28 [22,46] | 24 [20,29] | 0.001 | **53%** | **7%** |

- **"Éxito"** = porcentaje de semillas cuya política **supera a Gatenby** (TTP > 27 días). Es la
  métrica honesta para datos **bimodales**: como el self-play cae en cuencas buenas o malas según
  la semilla, no basta con la mediana; importa **con qué frecuencia** la política es realmente
  buena.
- **`[min, max]`** muestra la dispersión entre semillas (la bimodalidad: p. ej. `[25, 61]` = hubo
  semillas de 25 días y otras de 61).
- **`p` (Mann-Whitney U)** = probabilidad de que la diferencia MAPPO>IPPO sea por azar. `p < 0.05`
  = diferencia estadísticamente significativa.

### Qué significa cada fila

1. **φ_max = 0.05 (adversario nominal):** MAPPO mediana 36 vs IPPO 29, **p = 0.009**. El crítico
   centralizado ya ayuda: la terapia dura más y **93% de las veces** supera a Gatenby (vs 80% de
   IPPO).

2. **φ_max = 0.10 (intermedio):** **empatan** (ambos 27, p = 0.265). Régimen donde la ventaja de
   CTDE **no se manifiesta**. Lo reportamos con honestidad: el efecto **no es monótono**.

3. **φ_max = 0.20 (adversario fuerte):** aquí está el hallazgo clave. Las medianas se parecen
   (28 vs 24), pero la **tasa de éxito se separa dramáticamente: MAPPO 53% vs IPPO 7%**
   (p = 0.001). Traducción: *contra un tumor muy agresivo, la terapia con crítico centralizado
   todavía logra superar al estándar clínico en la mitad de los casos, mientras que la terapia con
   crítico local casi nunca lo logra (7%).*

### La idea central (para explicar al asesor en una frase)

> **El valor del crítico centralizado (CTDE) no se ve tanto en la mediana como en la
> *confiabilidad*: cuanto más fuerte es el adversario, más frecuentemente MAPPO consigue una
> política buena y menos lo consigue IPPO. Con el adversario más agresivo, la brecha de éxito es
> 53% vs 7%.** Esto responde afirmativamente la pregunta abierta del póster: *CTDE importa, y
> especialmente cuando la amenaza es realista y severa.*

### Un detalle adicional: el modo de falla cambia

| φ_max | Cómo falla MAPPO | Cómo falla IPPO |
|---|---|---|
| 0.05 | carga 11 / resistencia 4 | carga 15 |
| 0.10 | carga 14 / resistencia 1 | carga 15 |
| 0.20 | **carga 15** | carga 15 |

- Con `phi_max` bajo, a veces la terapia pierde porque los resistentes se vuelven mayoría
  (resistencia).
- Con `phi_max` alto, **todo** falla por **carga**: el tumor convierte S→R tan rápido que la masa
  total crece y cruza el umbral antes de que la resistencia sea el problema. El adversario fuerte
  cambia *la forma* de perder, no solo *cuándo*.

### Por qué las medianas casi no bajan (36 → 27 → 28) pero sí cambia el éxito

Por la **bimodalidad**: incluso con adversario fuerte, algunas semillas encuentran cuencas buenas
(máximos de 46–61 días), lo que sostiene la mediana; pero **muchas** semillas caen en cuencas
malas. Por eso la métrica que de verdad distingue a CTDE de IPPO es la **tasa de éxito**, no el
promedio. Este es exactamente el tipo de análisis honesto que exige un dato bimodal (y que evita
el error de reportar "media ± desviación" cuando la distribución tiene dos modas).
