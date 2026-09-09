# EXP-2 Adversario fortalecido V2 — resultados

Estado: **COMPLETO** · semillas objetivo n=15 · pasos=120000 · TTP vs tumor adaptativo co-entrenado

Celdas observadas: 90/90 · regímenes observados: [0.05, 0.1, 0.2]

Cada baseline se evalúa contra el mismo tumor aprendido de su celda.


## TTP-combinado por régimen (mediana [min,max]) y contraste pareado

| φ_max | MAPPO (CTDE) | IPPO (local) | mediana Δ pareada | Wilcoxon p | Holm p | éxito MAPPO | éxito IPPO |
|---|---|---|---|---|---|---|---|
| 0.05 | 44 [24,58] | 29 [26,37] | +15 | 0.005 | 0.014 | 100% | 87% |
| 0.10 | 28 [22,47] | 27 [23,35] | -2 | 0.648 | 0.648 | 100% | 93% |
| 0.20 | 24 [21,50] | 26 [20,34] | -1 | 0.265 | 0.530 | 100% | 100% |

## Modo de falla dominante por régimen

| φ_max | MAPPO | IPPO |
|---|---|---|
| 0.05 | {'load': 11, 'resistance': 4} | {'load': 15} |
| 0.10 | {'load': 15} | {'load': 15} |
| 0.20 | {'load': 13, 'resistance': 2} | {'load': 15} |
