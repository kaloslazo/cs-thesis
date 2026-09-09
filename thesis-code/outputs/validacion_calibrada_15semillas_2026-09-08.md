# Validación calibrada nominal — 15 semillas

Calibración `1fd8464abb97` · horizonte 180 d · φ fijo=0.01 · pasos efectivos=120832

TTP-combinado: primer fallo de carga o de tratabilidad; el modo de falla se conserva por separado.


## Resumen

| Método | TTP mediana [min,max] | Modos de falla | Éxito vs Gatenby |
|---|---:|---|---:|
| MAPPO (CTDE) | 34 [27,41] | {'resistance': 6, 'load': 9} | 13/15 |
| IPPO (local) | 31 [23,34] | {'load': 15} | 12/15 |

Gatenby determinista: 27 d (FALLO: resistencia mayoría).
Wilcoxon pareado unilateral MAPPO > IPPO: W=77.0, p=0.013759.
La prueba se interpreta junto con mediana, rango y modos de falla; no implica eficacia clínica.

## Detalle por semilla

| Semilla | MAPPO TTP | Modo MAPPO | IPPO TTP | Modo IPPO |
|---:|---:|---|---:|---|
| 0 | 41 | FALLO: resistencia mayoría | 32 | FALLO: carga progresó |
| 1 | 32 | FALLO: carga progresó | 27 | FALLO: carga progresó |
| 2 | 40 | FALLO: resistencia mayoría | 31 | FALLO: carga progresó |
| 3 | 41 | FALLO: resistencia mayoría | 28 | FALLO: carga progresó |
| 4 | 40 | FALLO: resistencia mayoría | 23 | FALLO: carga progresó |
| 5 | 40 | FALLO: carga progresó | 28 | FALLO: carga progresó |
| 6 | 27 | FALLO: carga progresó | 31 | FALLO: carga progresó |
| 7 | 28 | FALLO: carga progresó | 34 | FALLO: carga progresó |
| 8 | 30 | FALLO: carga progresó | 30 | FALLO: carga progresó |
| 9 | 29 | FALLO: carga progresó | 30 | FALLO: carga progresó |
| 10 | 29 | FALLO: carga progresó | 32 | FALLO: carga progresó |
| 11 | 41 | FALLO: resistencia mayoría | 32 | FALLO: carga progresó |
| 12 | 41 | FALLO: resistencia mayoría | 33 | FALLO: carga progresó |
| 13 | 34 | FALLO: carga progresó | 32 | FALLO: carga progresó |
| 14 | 27 | FALLO: carga progresó | 27 | FALLO: carga progresó |
