"""
================================================================================
build_dataset.py  —  Pipeline de integración (Glioblastoma)
Ejecutar SIEMPRE desde la raíz del repo:  python scripts/build_dataset.py
================================================================================

Estructura esperada:
  thesis-code/
  ├── data/raw/depmap/             Model.csv + OmicsExpression...ProteinCodingGenes.csv
  ├── data/raw/cellmodelpassports/ GDSC2_fitted_dose_response*.xlsx|csv + screened_compounds*
  └── scripts/build_dataset.py     (este archivo)

Salida:  data/dataset_marl_gbm_completo.csv  +  reporte de diagnóstico.
================================================================================
"""

import argparse
import glob
import os
import sys
import pandas as pd

DEPMAP_DIR = 'data/raw/depmap'
GDSC_DIR = 'data/raw/cellmodelpassports'
OUT_PATH = 'data/processed/dataset_marl_gbm_completo.csv'

def buscar(carpeta, patron, excluir=()):
    """Encuentra el primer archivo que matchea un patron, excluyendo terminos."""
    hits = [f for f in glob.glob(os.path.join(carpeta, patron))
            if not any(x.lower() in os.path.basename(f).lower() for x in excluir)]
    if not hits:
        sys.exit(f"ERROR: no encontre '{patron}' en {carpeta}/")
    return sorted(hits)[0]


def leer_tabla(path):
    """Lee CSV o Excel segun la extension."""
    if path.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    return pd.read_csv(path)


def cargar_expresion(path):
    """Expresion DepMap 26Q1: columnas ProfileID, is_default_entry, ModelID, genes."""
    expr = pd.read_csv(path)
    if 'is_default_entry' in expr.columns:          # 1 perfil por linea celular
        expr = expr[expr['is_default_entry'] == True]
    expr = expr.drop(columns=[c for c in ('ProfileID', 'is_default_entry')
                              if c in expr.columns])
    if 'ModelID' not in expr.columns:
        primera = expr.columns[0]                    # version vieja: 1ra col sin nombre
        if expr[primera].astype(str).str.startswith('ACH-').mean() > 0.5:
            expr = expr.rename(columns={primera: 'ModelID'})
        else:
            sys.exit(f"ERROR: no hay ModelID en expresion. Cols: {expr.columns.tolist()[:5]}")
    return expr


def norm_key(serie):
    """Llave canonica por nombre: MAYUSCULAS, solo letras y numeros."""
    return serie.astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depmap', default=DEPMAP_DIR)
    ap.add_argument('--gdsc', default=GDSC_DIR)
    ap.add_argument('--out', default=OUT_PATH)
    args = ap.parse_args()

    p_model = os.path.join(args.depmap, 'Model.csv')
    p_expr = buscar(args.depmap, 'OmicsExpression*TPMLogp1*ProteinCodingGenes*.csv',
                    excluir=('Stranded', 'Transcript', 'EffectiveLength',
                             'ExpectedCount', 'RawReadCount', 'AllGenes'))
    p_gdsc2 = buscar(args.gdsc, 'GDSC2_fitted_dose_response*')

    print('=' * 70)
    print('PIPELINE DE INTEGRACION - Glioblastoma')
    print('=' * 70)
    print(f'  Model:      {p_model}')
    print(f'  Expresion:  {p_expr}')
    print(f'  GDSC2:      {p_gdsc2}')

    # 1. Carga
    print('\n[1/5] Cargando datasets crudos...')
    metadata = pd.read_csv(p_model)
    gdsc2 = leer_tabla(p_gdsc2)
    expr = cargar_expresion(p_expr)
    print(f'      Model.csv:   {len(metadata):,} lineas (catalogo)')
    print(f'      GDSC2:       {len(gdsc2):,} ensayos')
    print(f'      Expresion:   {len(expr):,} lineas x {expr.shape[1]-1:,} genes')

    col_ic50 = next((c for c in ('LN_IC50', 'ln_IC50', 'IC50') if c in gdsc2.columns), None)
    print(f'      Columna IC50 detectada: {col_ic50}')
    if col_ic50 is None:
        print(f'      AVISO revisa columnas GDSC2: {gdsc2.columns.tolist()}')

    # 2. Filtrar Glioblastoma
    print('\n[2/5] Filtrando lineas de Glioblastoma...')
    mask = pd.Series(False, index=metadata.index)
    for c in ('OncotreeSubtype', 'OncotreePrimaryDisease'):
        if c in metadata.columns:
            mask |= metadata[c].str.contains('Glioblastoma', case=False, na=False)
    gbm = metadata[mask].copy()
    print(f'      Lineas GBM en catalogo DepMap: {len(gbm)}')
    if len(gbm) == 0:
        sys.exit("ERROR: sin lineas GBM. Revisa columnas de Model.csv.")

    # 3. Cruce: SANGER_MODEL_ID primero, nombre normalizado de respaldo
    print('\n[3/5] Cruzando GDSC2 <-> DepMap...')
    gdsc2['ModelID'] = pd.NA
    n_sanger = 0
    usa_sanger = ('SANGER_MODEL_ID' in gdsc2.columns) and ('SangerModelID' in gbm.columns)
    if usa_sanger:
        sid2model = dict(zip(gbm['SangerModelID'], gbm['ModelID']))
        gdsc2['ModelID'] = gdsc2['SANGER_MODEL_ID'].map(sid2model)
        n_sanger = int(gdsc2['ModelID'].notna().sum())
        print(f'      OK cruce por SANGER_MODEL_ID: {n_sanger:,} ensayos mapeados')
    else:
        print('      AVISO no hay SANGER_MODEL_ID; uso solo nombre normalizado')

    if 'CELL_LINE_NAME' in gdsc2.columns:
        name2model = dict(zip(norm_key(gbm['StrippedCellLineName']), gbm['ModelID']))
        falta = gdsc2['ModelID'].isna()
        gdsc2.loc[falta, 'ModelID'] = norm_key(gdsc2.loc[falta, 'CELL_LINE_NAME']).map(name2model)
        extra = int(gdsc2['ModelID'].notna().sum()) - n_sanger
        print(f'      OK respaldo por nombre normalizado: {extra:,} adicionales')

    gdsc2_gbm = gdsc2.dropna(subset=['ModelID']).copy()
    lineas_gbm = gdsc2_gbm['ModelID'].nunique()
    print(f'      -> Lineas GBM con ensayos en GDSC2: {lineas_gbm}')
    print(f'      -> Ensayos GBM recuperados:         {len(gdsc2_gbm):,}')
    if lineas_gbm <= 6:
        print('      AVISO <=6 lineas: es el numero real del dataset (no era bug de llaves).')
    else:
        print(f'      OK recuperaste {lineas_gbm} lineas (el codigo viejo daba 6).')

    # 4. Unir expresion genica
    print('\n[4/5] Uniendo perfiles genomicos por ModelID...')
    master = pd.merge(gdsc2_gbm, expr, on='ModelID', how='inner')
    print(f'      Filas finales:           {len(master):,}')
    print(f'      Lineas (estados unicos): {master["ModelID"].nunique()}')

    # 5. Exportar
    print('\n[5/5] Exportando (SMILES omitidos; paso aparte cuando se necesiten)...')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    master.to_csv(args.out, index=False)

    print('\n' + '=' * 70)
    print('LISTO: DATASET GENERADO')
    print(f'   {args.out}')
    print(f'   Registros: {master.shape[0]:,}  |  Columnas: {master.shape[1]:,}')
    farmacos = master["DRUG_NAME"].nunique() if "DRUG_NAME" in master.columns else "?"
    print(f'   Lineas GBM: {master["ModelID"].nunique()}  |  Farmacos: {farmacos}')
    print('=' * 70)


if __name__ == '__main__':
    main()