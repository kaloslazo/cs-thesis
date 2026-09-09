# Punto de continuación — cerrado el 2026-09-09

Este archivo registró una pausa de ejecución. El benchmark se reanudó y ya fue completado.

## Ya terminado

- Datos oficiales DepMap Public 26Q1 + GDSC2 release 8.5 y calibración: huella `1fd8464abb97`.
- Validación nominal: 15 semillas MAPPO/IPPO.
- EXP-1 calibrado: perturbación OAT, conjunta, endpoints y ruido.
- EXP-2 completo: 90/90 celdas, `phi_max={0.05,0.10,0.20}`, 15 semillas por método y régimen.
- Documentación EXP-2 y benchmark actualizadas; el capítulo 4 quedó incorporado al PDF final y pasó la verificación visual.

## Benchmark V2 — estado final

Comando titular:

```bash
cd /Users/kalosroots/Documents/ChatGPT/Tesis/cs-thesis/thesis-code
../.venv/bin/python scripts/benchmark_exploit.py --seeds 15 --steps 120000 --br-restarts 5
```

La corrida titular terminó con 15/15 filas, 5 reinicios de best-response por terapia,
Gatenby bajo 5 reinicios y cross-play completo. Los artefactos son
`outputs/benchmark_exploit_v2.json`, `.md` y `.png`.

El benchmark, el control de horizonte 360, el PDF y los checks finales ya están completos.

La variable de salud sigue explícitamente sin calibración clínica: requiere datos
longitudinales de toxicidad y no debe rellenarse con valores inventados.
