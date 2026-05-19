import importlib.util
from pathlib import Path

import numpy as np


def _load_audit_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "audit_release_equivalence.py"
    spec = importlib.util.spec_from_file_location("audit_release_equivalence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_paths_keep_release_and_current_separate():
    audit = _load_audit_module()

    release_paths = audit.import_paths_for_mode("release")
    current_paths = audit.import_paths_for_mode("current_fast")

    assert release_paths[0] == str(audit.RELEASE_SITE)
    assert current_paths[0] == str(audit.REPO_ROOT)
    assert str(audit.BUILDER_PATH) in release_paths
    assert str(audit.BUILDER_PATH) in current_paths


def test_compare_arrays_classifies_exact_physics_and_contract_change():
    audit = _load_audit_module()

    exact = audit.compare_arrays(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    physics = audit.compare_arrays(np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-13]))
    contract = audit.compare_arrays(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))

    assert exact["status"] == "exact"
    assert physics["status"] == "physics"
    assert contract["status"] == "contract_change"


def test_ch5_case_list_contains_master_and_large_box_cases():
    audit = _load_audit_module()
    cases = audit.ch5_cases()

    assert len(cases) == 24
    assert "AL1.0_AR2.0_d10.0" in cases
    assert "AL1.0_AR2.0_d12.0_L50.0" in cases
    assert audit.parse_ch5_case("AL1.0_AR2.0_d12.0_L50.0") == {
        "AL": 1.0,
        "AR": 2.0,
        "d": 12.0,
        "L": 50.0,
    }


def test_current_worker_import_info_points_at_repo():
    audit = _load_audit_module()

    payload = audit.worker_solver_payload("current_fast", "import_info")

    assert str(audit.REPO_ROOT) in str(payload["iDEA_file"])
