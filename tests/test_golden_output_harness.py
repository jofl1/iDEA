import importlib.util
from pathlib import Path

import numpy as np


def _load_harness():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "golden_output_harness.py"
    spec = importlib.util.spec_from_file_location("golden_output_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_golden_harness_generates_and_compares_exact_outputs(tmp_path):
    harness = _load_harness()
    case_name = "non_interacting_harmonic_u"

    generated = harness.generate([case_name], golden_dir=tmp_path)
    assert generated[0][0] == case_name
    assert generated[0][1].exists()

    compared = harness.compare([case_name], golden_dir=tmp_path)
    assert compared[0][0] == case_name
    assert compared[0][3] == []


def test_golden_harness_reports_array_mismatch(tmp_path):
    harness = _load_harness()
    case_name = "non_interacting_harmonic_u"

    harness.generate([case_name], golden_dir=tmp_path)
    golden_path = tmp_path / f"{case_name}.npz"
    with np.load(golden_path, allow_pickle=False) as npz:
        payload = {key: npz[key] for key in npz.files}

    payload["density"] = payload["density"].copy()
    payload["density"][0] = payload["density"][0] + 1.0
    np.savez_compressed(golden_path, **payload)

    compared = harness.compare([case_name], golden_dir=tmp_path)
    assert any("density:" in failure for failure in compared[0][3])
