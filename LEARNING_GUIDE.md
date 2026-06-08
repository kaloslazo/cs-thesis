# LEARNING_GUIDE — Qué aprender, en qué orden, hasta qué profundidad

Principio: **just-in-time**. No estudies todo RL antes de codear. Aprende cada módulo justo antes del hito que lo usa. Para cada uno: el mínimo que debes poder *explicar en la defensa*, un recurso, y la pregunta que el jurado probablemente hará.

Profundidad objetivo: **explicar con tus palabras + leer código sin perderte**. No necesitas derivar las pruebas de convergencia.

---

## L-BIO — Biología + ODEs · antes de M1 (3–4 h)
**Debes poder explicar:**
- Por qué la resistencia es evolutiva, no estática (selección natural: matas sensibles, liberas espacio a resistentes).
- Modelo logístico/Lotka-Volterra de competencia (qué es K, qué pasa cuando S+R→K).
- Función de Hill `δ(c)`: por qué la muerte celular satura con la dosis.
- Terapia adaptativa (Gatenby 2009): mantener sensibles vivos para que compitan con resistentes.
**Recurso:** Gatenby et al. 2009 (intro y discusión, no la mate completa). Cualquier explicación de "logistic growth" + "Hill equation".
**Pregunta de defensa:** *"¿Por qué dosis máxima acelera la resistencia?"* → porque elimina el competidor sensible y selecciona el clon resistente.

## L-CALIB — Ajuste de parámetros · antes de M2 (2–3 h)
**Debes poder explicar:**
- Diferencia entre "los datos entrenan la red" (NO es tu caso) y "los datos calibran el simulador" (SÍ es tu caso).
- Cómo LN_IC50 → parámetro de muerte δ del modelo.
- Ajuste con `scipy.optimize.curve_fit` o `minimize`.
**Pregunta de defensa:** *"¿De dónde salen IC50_S=0.30 e IC50_R=0.85?"* → tu fórmula de calibración (tenla escrita y derivada).

## L-RL — Fundamentos RL · antes de M3 (1 día)
**Debes poder explicar:**
- MDP: estado, acción, recompensa, transición, política, factor de descuento γ.
- Política (π) vs función de valor (V/Q).
- Diferencia RL vs aprendizaje supervisado: no hay etiquetas, hay recompensa diferida.
**Recurso:** OpenAI Spinning Up — "Part 1: Key Concepts in RL" (spinningup.openai.com). Sutton & Barto, caps. 3 (gratis online).
**Pregunta de defensa:** *"¿Cuál es tu estado, acción y recompensa exactamente?"* → estado [S,R,c]+genes, acción=dosis, reward= −tumor −toxicidad.

## L-PPO — Policy gradient y PPO · antes de M4a (1 día)
**Debes poder explicar:**
- Actor (política) + crítico (valor): el crítico estima qué tan buena es la situación para reducir la varianza del gradiente.
- Por qué PPO "clipea": evita updates que cambien demasiado la política y la rompan.
- GAE (advantage): qué tan mejor que el promedio fue una acción.
**Recurso:** Spinning Up "Vanilla Policy Gradient" → "PPO". Código: CleanRL `ppo_continuous_action.py` (un solo archivo, leerlo entero).
**Pregunta de defensa:** *"¿Por qué PPO y no DQN?"* → acción continua (dosis), on-policy estable, estándar en control biomédico.

## L-MARL — Multi-agente y CTDE · antes de M4b (1 día)
**Debes poder explicar:**
- Por qué multi-agente NO es trivial: **no-estacionariedad** (el entorno "cambia" porque el otro agente aprende → la experiencia vieja envejece).
- Dec-POMDP: cada agente observa parcialmente.
- **CTDE** (Centralized Training, Decentralized Execution): entrenas con info global, ejecutas con info local.
**Recurso:** Yu et al. 2021 (MAPPO, arXiv:2103.01955) — intro y sección de método.
**Pregunta de defensa:** *"¿Qué es lo 'centralizado' en CTDE?"* → el CRÍTICO ve el estado conjunto en entrenamiento; los actores siempre usan solo su observación local.

## L-IPPOMAPPO — La diferencia exacta · con M4b (½ día)
**Debes poder explicar (memorízalo, te lo van a preguntar):**
- **IPPO:** cada agente, su propio crítico que ve solo SU observación. Sufre no-estacionariedad.
- **MAPPO:** crítico centralizado que ve el estado conjunto en entrenamiento → menos varianza, maneja la no-estacionariedad. Actores locales en ejecución.
- Lo único que cambia es **qué ve el crítico**. La política es local en ambos.
- Matiz: MAPPO canónico es cooperativo; el tuyo es adversarial → "crítico centralizado por agente, self-play". Dilo así.
**Pregunta de defensa:** *"Tu título dice MAPPO-CTDE, ¿tu crítico es centralizado?"* → SÍ, recibe [obs_terapia, obs_tumor, estado_ODE]; en deploy cada actor corre con su obs local.

## L-EVAL — Validación honesta · antes de M5 (½ día)
**Debes poder explicar:**
- Qué es **validación circular**: si entrenas y evalúas en el mismo simulador moldeado por tu adversario, ganas por construcción.
- Por qué comparas también contra tumor fijo y contra terapia adaptativa clásica.
**Pregunta de defensa:** *"¿Cómo sabes que tu agente no solo le gana a un strawman?"* → evaluación fuera del setting de entrenamiento (tumor no-aprendiz + adaptativa + cinética publicada).

---

## Ruta crítica (resumen)
```
L-BIO → M1 → L-CALIB → M2 → L-RL → M3 → L-PPO → M4a → L-MARL + L-IPPOMAPPO → M4b → L-EVAL → M5
```

## Cómo estudiar sin perder tiempo
- 1 módulo = leer + escribir en 5 líneas con tus palabras qué entendiste (si no puedes, no lo entendiste).
- Antes de cada hito, repasa solo su pregunta de defensa.
- No abras el siguiente módulo hasta cerrar el hito actual. La teoría sin código se olvida.

## Mini-glosario para la defensa
- **Política (π):** función estado → acción.
- **Crítico/valor (V):** estima retorno futuro desde un estado.
- **CTDE:** entrenas con info global, ejecutas con info local.
- **No-estacionariedad:** el entorno parece cambiar porque otros agentes aprenden.
- **Self-play:** los agentes mejoran enfrentándose entre sí.
- **Hill `δ(c)`:** muerte celular saturante en función de la dosis.
- **Terapia adaptativa:** dosificar para *controlar*, no erradicar, manteniendo competencia entre clones.