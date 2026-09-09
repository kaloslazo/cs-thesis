# Toxicidad y estado funcional del paciente — diseño e implementación V1

**Estado:** implementación experimental opt-in; **sin resultados clínicos ni corrida completa**.

**Código:** `gbmarl/patient_env.py`, `gbmarl/dynamics.py`, `gbmarl/config.py`.

**Experimento:** `scripts/exp3_patient_health.py`.

**Tests:** `tests/test_patient_env.py`.

## Decisión de diseño

La salud se modela como una consecuencia pasiva de la enfermedad y el tratamiento,
no como un agente racional. El juego se mantiene en dos agentes:

- terapia: elige dosis;
- tumor: elige transición fenotípica sensible→resistente.

No se incorpora un tercer “agente paciente”. Tampoco se incorpora por ahora un
agente clínico, porque duplicaría el rol del agente de terapia sin una observación,
escala temporal y responsabilidad claramente distintas.

## Problema de la versión anterior

La línea base representaba toxicidad con `-tox_weight*u`, un castigo instantáneo y
lineal. Ese término no acumulaba exposición, no permitía recuperación y no podía
producir un evento terminal de toxicidad.

El primer plan proponía una EDO única de salud dañada directamente por `u` y por la
carga tumoral. Se descartó esa forma porque:

1. la toxicidad debe depender de concentración/exposición `c`, no solo de la orden `u`;
2. mezclar tumor y toxicidad en una sola variable impide identificar el motivo de falla;
3. los coeficientes sin calibración podían elegirse accidentalmente para favorecer a MAPPO.

## Estado dinámico

La extensión añade toxicidad acumulada `A∈[0,1]`:

```text
dA/dt = k_c · g(c) · (1-A) - rho_A · A
g(c)  = c / (EC50_tox + c)
```

- `k_c`: velocidad de acumulación por exposición;
- `rho_A`: recuperación;
- `EC50_tox`: concentración de semisaturación tóxica;
- `(1-A)`: mantiene la acumulación acotada al aproximarse a 1.

El estado global pasa de `(S,R,c)` a `(S,R,c,A)`. La terapia observa
`(S+R,c,A)`, el tumor conserva `(S,R)` y el crítico CTDE observa el estado 4-D.

## Desenlaces

Los eventos quedan separados:

```text
load        : S+R >= progression_threshold·K
resistance  : R/(S+R) >= r_majority
toxicity    : A >= failure_threshold
multiple    : toxicidad y evento tumoral en el mismo día
extinct     : extinción tumoral (éxito observado)
horizon     : censura administrativa
```

El entorno registra `t_load`, `t_resistance`, `t_toxicity`, `failure_events` y
`failure_mode`. Esto evita deducir la causa mediante prioridades implícitas.

## Estado funcional / calidad de vida simulada

Se calcula una utilidad secundaria:

```text
H(t) = clip(1 - w_A·A(t) - w_B·(S+R)/K, 0, 1)
```

`H` no se presenta como una escala clínica validada. Es una utilidad simulada que
permite calcular:

```text
quality_adjusted_days = suma_t H(t)
```

Antes de usar términos como QoL o QALY, los pesos deben mapearse a un instrumento
validado o declararse explícitamente como constantes de diseño.

## Parámetros obligatorios y prevención de sesgo

`ToxicityParams` no tiene valores clínicos por defecto. El experimento exige un JSON
con la siguiente forma:

```json
{
  "source": "DOI, protocolo o nota de calibración",
  "parameters": {
    "accumulation_rate": "valor requerido",
    "recovery_rate": "valor requerido",
    "half_saturation": "valor requerido",
    "failure_threshold": "valor requerido",
    "initial_toxicity": 0.0,
    "toxicity_weight_in_health": "valor requerido",
    "burden_weight_in_health": "valor requerido"
  }
}
```

La configuración se valida y recibe un hash; modelos y resultados quedan vinculados
a ese hash para no mezclar corridas con parámetros distintos.

## Recompensa y restricción

En `PatientTumorEnv` se elimina por defecto `-tox_weight*u` para no contar dos veces
la toxicidad. La terapia recibe recompensa por cada día manejable y el episodio
termina si se viola el umbral de toxicidad.

Esta es una primera aproximación a una restricción de seguridad. Un siguiente paso
puede usar PPO-Lagrangiano/CPO para formular explícitamente:

```text
maximizar TTP sujeto a P(A >= A_max) <= epsilon
```

La evaluación debe conservar por separado TTP, toxicidad, salud y exposición; no se
debe reducir toda la evidencia a un único peso escalar.

## Ablación implementada

`exp3_patient_health.py` entrena dos condiciones con semillas y presupuesto comunes:

- `health_aware`: terapia observa `A` y el entrenamiento termina por toxicidad;
- `health_blind`: reproduce el estado 3-D y el castigo lineal de dosis anterior.

Luego ambas se evalúan dentro de `PatientTumorEnv`, con idéntica dinámica tóxica. La
comparación bajo `phi` fijo aísla el efecto del diseño de salud; el resultado bajo el
tumor de self-play se conserva como análisis adversarial, pero no debe interpretarse
solo como efecto de salud porque cada condición co-entrena un adversario distinto.
El reporte calcula diferencias pareadas por semilla e intervalos bootstrap para TTP,
días ajustados y toxicidad final. No declara superioridad con una suma ponderada ni
con un margen de no inferioridad elegido después de ver los datos.

## Protocolo experimental

1. Fijar parámetros y fuente antes de observar resultados de MAPPO/IPPO.
2. Validar dinámica: recuperación sin exposición, monotonicidad, límites y RK4.
3. Reentrenar MAPPO e IPPO desde cero con estado 4-D.
4. Reevaluar sin tratamiento, MTD y Gatenby en el mismo entorno.
5. Reportar TTP, modo de falla, `quality_adjusted_days`, salud media, toxicidad final y dosis.
6. Ejecutar sensibilidad individual y conjunta de los parámetros tóxicos.
7. Añadir ruido, sesgo y retraso a la observación de `A`.
8. Ejecutar el benchmark de best-response V2 con múltiples reinicios.
9. Usar la ablación implementada contra la línea base sin salud y presentar una
   frontera TTP–salud.

## Límites

- La extensión es in silico y no constituye recomendación terapéutica.
- Un único compartimento no reproduce toda la mielosupresión, hepatotoxicidad ni
  toxicidades tardías de temozolomida.
- No deben publicarse conclusiones sobre “protección del paciente” hasta calibrar o
  analizar exhaustivamente la sensibilidad de sus parámetros.
- Un tercer agente solo se justificaría si representa una decisión diferente —por
  ejemplo, un supervisor de seguridad en otra escala temporal— y se compara contra
  un supervisor determinista equivalente.
