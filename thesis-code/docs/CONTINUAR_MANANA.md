# ▶️ CONTINUAR MAÑANA — variable de salud (Fase 1)

**Última sesión:** 7 sep 2026 (S5). **Estado:** Fase 1 (salud pasiva) = prueba de concepto lista.

---

## Dónde está todo (¡importante!)

- **Estás en un git WORKTREE aislado**, NO en `main`:
  - Carpeta: `/Users/gianpier/TESIS/cs-thesis/.claude/worktrees/variable-salud/thesis-code`
  - Rama: `worktree-variable-salud`
  - `main` (pusheado, commit `f9f66c2`) tiene el benchmark de explotabilidad y NO tiene la salud.
- **La salud NO está commiteada ni fusionada.** Vive como cambios sin commit en esta rama.
  Si la sesión preguntó "conservar o descartar el worktree" → **CONSERVAR**.

## Cómo retomar (comandos)

El venv está en el repo principal (el worktree no tiene uno). Usar SIEMPRE su ruta completa:

```bash
# situarse en el worktree
cd /Users/gianpier/TESIS/cs-thesis/.claude/worktrees/variable-salud/thesis-code
PY=/Users/gianpier/TESIS/cs-thesis/thesis-code/.venv/bin/python

# verificar que la base 3-D sigue intacta (deben pasar 14/14)
$PY -m pytest tests/ -q

# re-correr la prueba de concepto de salud
$PY scripts/exp_salud.py --smoke            # rápido (2k pasos)
$PY scripts/exp_salud.py --steps 120000     # corrida seria (1 semilla, ~2 min)
```

## Qué se implementó (Fase 1, Opción A — salud pasiva)

Todo ADITIVO — no toca `dynamics.py`/`tumor_env.py`/`mappo.py`:
- `gbmarl/dynamics_health.py` — EDO 4-D: `dH/dt = ρ_H(1−H) − κ_u·u − κ_b(S+R)`
- `gbmarl/health_env.py` — entorno con salud + tercer modo de falla "salud"
- `gbmarl/mappo_health.py` — entrenamiento (terapia observa H; crítico ve S,R,c,H)
- `scripts/exp_salud.py` — experimento smoke-testable
- `tests/test_dynamics_health.py` — 4 tests nuevos (pasan)
- `gbmarl/config.py` — params de salud añadidos (aditivo)
- Docs: `docs/variable_salud_paciente.md` (plan + §10 resultados), `docs/informe_avance_S5_asesor.md`

## Resultado obtenido (120k, 1 semilla)

MAPPO-salud: **TTP=26 d, salud media 0.91** (vs MTD 0.61). Mecanismo emergente confirmado:
la dosificación pulsada protege la salud sin recompensa explícita. Detalle en
`docs/variable_salud_paciente.md` §10.

## Pendientes para cerrar Fase 1 como resultado de tesis (orden sugerido)

1. **[SIGUIENTE] Multi-semilla n≥10** en el entorno con salud → mediana + tasa de éxito.
   *Hacer un `scripts/exp_salud_multiseed.py` reanudable (patrón de `benchmark_exploit.py`:
   JSON de estado + checkpoints en `outputs/models/`). Reusar `train_mappo_health` +
   `ttp_health_detalle`.*
2. **Ablación con-H vs sin-H** sobre las mismas semillas (aislar el efecto de la variable).
3. **Sensibilidad** de ρ_H, κ_u, κ_b, H_min (±30 %) — patrón de EXP-1.
4. **Explotabilidad** de las políticas con salud (adaptar `gbmarl/exploit.py` al entorno 4-D).

## Decisiones abiertas (esperando al asesor)

- ¿Se incorpora la salud al alcance de PFC II o queda como trabajo futuro? (ver
  `informe_avance_S5_asesor.md` §5).
- ¿Calibrar κ_u/H_min con literatura de toxicidad de TMZ, en vez de constantes ilustrativas?
- Fase 2 (agente clínico, juego de 3 agentes) — solo si el asesor aprueba el alcance.

## Caveats a no olvidar

- Las constantes de salud son **ilustrativas**, no calibradas → decirlo siempre; respaldar con
  sensibilidad. NO ajustarlas para fabricar el modo de falla "salud" (trampa metodológica).
- Reproducibilidad es intra-script (misma semilla → mismo resultado dentro del mismo script).
- Bimodalidad → tasa de éxito, no media±std.
