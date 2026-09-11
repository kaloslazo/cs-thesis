from pathlib import Path

import pytest

from gbmarl.config import Params, load_calibration, params_fingerprint


def test_required_calibration_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Calibración requerida"):
        load_calibration(tmp_path / "missing.json", required=True)


def test_exploratory_fallback_remains_explicit(tmp_path, capsys):
    assert load_calibration(tmp_path / "missing.json") == Params()
    assert "placeholders" in capsys.readouterr().out


def test_reference_reproduces_published_parameter_fingerprint():
    path = Path(__file__).resolve().parents[1] / "configs/calibration_reference_20260908.json"
    assert params_fingerprint(load_calibration(path, required=True)) == "1fd8464abb97"
