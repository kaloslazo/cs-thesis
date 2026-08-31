# EXP-2 Adversario fortalecido — resultados

Semillas n=15 · pasos=120000 · TTP vs tumor adaptativo co-entrenado


## TTP-combinado por régimen (mediana [min,max]) y contraste CTDE vs IPPO

| φ_max | MAPPO (CTDE) | IPPO (local) | Δ mediana | Mann-Whitney p | éxito MAPPO | éxito IPPO |
|---|---|---|---|---|---|---|
| 0.05 | 36 [25,61] | 29 [23,32] | +7 | 0.009 | 93% | 80% |
| 0.10 | 27 [22,61] | 27 [23,31] | +0 | 0.265 | 47% | 33% |
| 0.20 | 28 [22,46] | 24 [20,29] | +4 | 0.001 | 53% | 7% |

## Modo de falla dominante por régimen

| φ_max | MAPPO | IPPO |
|---|---|---|
| 0.05 | {'carga': 11, 'resistencia': 4} | {'carga': 15} |
| 0.10 | {'resistencia': 1, 'carga': 14} | {'carga': 15} |
| 0.20 | {'carga': 15} | {'carga': 15} |
