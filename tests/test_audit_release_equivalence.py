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


def test_solver_cases_cover_audit_gaps():
    """Pin the audit coverage so the gap-closing cases aren't dropped
    by future refactors. Each case here exercises something the
    second friend's review identified as a gap.
    """
    audit = _load_audit_module()
    cases = audit.solver_cases()

    # A3: non-canonical spin orderings.
    assert "interacting_harmonic_du" in cases
    assert "interacting_harmonic_dud" in cases
    assert "interacting_harmonic_ddu" in cases

    # A7: at least two non-default FD stencils.
    assert "interacting_harmonic_uu_stencil5" in cases
    assert "interacting_harmonic_uu_stencil11" in cases

    # A8: large-offset asymmetric potential.
    assert "interacting_offset_asymmetric_uu" in cases


def test_non_canonical_spin_worker_payload_agrees_across_modes():
    """A3: dispatched fast path and forced legacy path agree on energy
    and density for non-canonical spin orderings. Catches any
    spin-axis-order regression in build_labelled_state.
    """
    audit = _load_audit_module()
    for case in (
        "interacting_harmonic_du",
        "interacting_harmonic_dud",
        "interacting_harmonic_ddu",
    ):
        fast = audit.worker_solver_payload("current_fast", case)
        compat = audit.worker_solver_payload("current_compat", case)
        assert np.isclose(float(fast["energy"]), float(compat["energy"]), atol=1e-12)
        assert np.allclose(fast["density"], compat["density"], atol=1e-12, rtol=0)


def test_stencil_variant_worker_payload_agrees_across_modes():
    """A7: stencil 5 and 11 give consistent results between fast and
    legacy solvers. Catches stencil-specific bugs in the det basis or
    parity machinery.
    """
    audit = _load_audit_module()
    for case in (
        "interacting_harmonic_uu_stencil5",
        "interacting_harmonic_uu_stencil11",
    ):
        fast = audit.worker_solver_payload("current_fast", case)
        compat = audit.worker_solver_payload("current_compat", case)
        assert np.isclose(float(fast["energy"]), float(compat["energy"]), atol=1e-12)
        assert np.allclose(fast["density"], compat["density"], atol=1e-12, rtol=0)


def test_asymmetric_offset_worker_payload_takes_non_parity_path():
    """A8: large-offset asymmetric potential. Without the A1 rtol fix,
    parity could be mis-detected and one block solved on a Hamiltonian
    that does not conserve parity. Force both modes through and assert
    agreement.
    """
    audit = _load_audit_module()
    fast = audit.worker_solver_payload("current_fast", "interacting_offset_asymmetric_uu")
    compat = audit.worker_solver_payload(
        "current_compat", "interacting_offset_asymmetric_uu"
    )
    # Energy is ~1e6 (offset-dominated); use relative comparison.
    assert np.isclose(
        float(fast["energy"]), float(compat["energy"]), rtol=1e-12, atol=0
    )
    assert np.allclose(fast["density"], compat["density"], atol=1e-8, rtol=0)
