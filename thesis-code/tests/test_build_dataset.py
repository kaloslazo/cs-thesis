import pandas as pd

from scripts.build_dataset import cargar_expresion


def test_cargar_expresion_filtra_bandera_depmap_26q1(tmp_path):
    path = tmp_path / "expression.csv"
    pd.DataFrame({
        "": [0, 1, 2],
        "SequencingID": ["s1", "s2", "s3"],
        "ModelConditionID": ["mc1", "mc1", "mc2"],
        "ModelID": ["ACH-1", "ACH-1", "ACH-2"],
        "IsDefaultEntryForMC": ["Yes", "No", "Yes"],
        "IsDefaultEntryForModel": ["Yes", "No", "Yes"],
        "GENE (1)": [1.0, 2.0, 3.0],
    }).to_csv(path, index=False)

    result = cargar_expresion(path)

    assert result["ModelID"].tolist() == ["ACH-1", "ACH-2"]
    assert result["GENE (1)"].tolist() == [1.0, 3.0]
    assert "IsDefaultEntryForModel" not in result.columns
    assert "IsDefaultEntryForMC" not in result.columns


def test_cargar_expresion_acepta_bandera_booleana_antigua(tmp_path):
    path = tmp_path / "expression_old.csv"
    pd.DataFrame({
        "ModelID": ["ACH-1", "ACH-1"],
        "is_default_entry": [True, False],
        "GENE (1)": [1.0, 2.0],
    }).to_csv(path, index=False)

    result = cargar_expresion(path)

    assert result["ModelID"].tolist() == ["ACH-1"]
    assert result["GENE (1)"].tolist() == [1.0]
