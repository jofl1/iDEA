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


def test_sign_invariant_compare_accepts_global_sign_flip():
    """A4: wavefunction sign flips are physically irrelevant for
    non-degenerate eigenstates. The audit comparator must classify a
    sign-flipped array as exact (with sign_flip=True), not diff.
    """
    audit = _load_audit_module()
    ref = np.array([1.0, 2.0, -0.5])
    cur = -ref

    diff = audit.compare_arrays(ref, cur)
    assert diff["status"] == "diff"
    assert diff["sign_flip"] is False

    sign_inv = audit.compare_arrays(ref, cur, sign_invariant=True)
    assert sign_inv["status"] == "exact"
    assert sign_inv["sign_flip"] is True


def test_sign_invariant_compare_picks_smaller_diff_on_near_flip():
    """If the reference is closer to -cur than +cur, the sign-invariant
    metric must report the smaller diff and mark sign_flip True.
    """
    audit = _load_audit_module()
    ref = np.array([1.0, 2.0, -0.5])
    cur = -ref + 1e-13  # very close to -ref

    row = audit.compare_arrays(ref, cur, sign_invariant=True)
    assert row["status"] == "physics"
    assert row["sign_flip"] is True
    assert row["max_abs"] < 1e-12


def test_audit_env_sets_backend_env_for_each_mode():
    """A5/A6: each of the new mode variants must set the env vars that
    drive the iDEA_DET_* dispatch hooks. Pins the audit hook surface so
    a refactor can't silently break which mode forces which backend.
    """
    audit = _load_audit_module()

    scipy_env = audit.audit_env("current_fast_scipy")
    assert scipy_env["IDEA_DET_EIGSH_BACKEND"] == "scipy"
    assert "IDEA_DET_DISABLE_PRIMME" not in scipy_env

    primme_env = audit.audit_env("current_fast_primme")
    assert primme_env["IDEA_DET_EIGSH_BACKEND"] == "primme"
    assert primme_env["IDEA_DET_PRIMME_MIN_DIM"] == "0"

    no_primme_env = audit.audit_env("current_no_primme")
    assert no_primme_env["IDEA_DET_DISABLE_PRIMME"] == "1"
    assert "IDEA_DET_EIGSH_BACKEND" not in no_primme_env

    # current_fast (default) must NOT set any of the dispatch overrides.
    fast_env = audit.audit_env("current_fast")
    assert "IDEA_DET_EIGSH_BACKEND" not in fast_env
    assert "IDEA_DET_PRIMME_MIN_DIM" not in fast_env
    assert "IDEA_DET_DISABLE_PRIMME" not in fast_env


def test_import_paths_for_each_current_fast_variant():
    audit = _load_audit_module()
    for mode in ("current_fast", "current_fast_scipy", "current_fast_primme", "current_no_primme"):
        paths = audit.import_paths_for_mode(mode)
        assert paths[0] == str(audit.REPO_ROOT), (
            f"mode {mode}: expected REPO_ROOT first, got {paths[0]!r}"
        )


def test_modes_for_interacting_cases_include_backend_variants():
    audit = _load_audit_module()
    for case in ("interacting_harmonic_uu", "interacting_harmonic_du"):
        modes = audit.modes_for_solver_case(case)
        assert "current_fast_scipy" in modes
        assert "current_fast_primme" in modes
        assert "current_no_primme" in modes


def test_is_sign_invariant_array_key_recognises_wavefunctions():
    audit = _load_audit_module()
    assert audit.is_sign_invariant_array_key("space")
    assert audit.is_sign_invariant_array_key("full")
    assert audit.is_sign_invariant_array_key("det_amplitudes")
    assert audit.is_sign_invariant_array_key("td_space")  # time evolution
    # observables and metadata are not sign-invariant.
    assert not audit.is_sign_invariant_array_key("density")
    assert not audit.is_sign_invariant_array_key("energy")
    assert not audit.is_sign_invariant_array_key("v_xc")
    assert not audit.is_sign_invariant_array_key("up_orbitals")


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


def test_propagation_case_is_registered():
    audit = _load_audit_module()
    assert "interacting_propagate_harmonic_uu_kick" in audit.solver_cases()


def test_propagation_worker_payload_agrees_across_modes():
    """A9: propagation from a fast-path initial state must produce
    matching time-dependent density against the legacy labelled solve
    + propagate. Catches reconstruction-at-the-time-evolution-boundary
    bugs even though propagate itself is labelled.
    """
    audit = _load_audit_module()
    fast = audit.worker_solver_payload(
        "current_fast", "interacting_propagate_harmonic_uu_kick"
    )
    compat = audit.worker_solver_payload(
        "current_compat", "interacting_propagate_harmonic_uu_kick"
    )
    assert fast["td_density"].shape == compat["td_density"].shape
    assert np.allclose(
        fast["td_density"], compat["td_density"], atol=1e-10, rtol=0.0
    )
    assert np.isclose(
        float(fast["initial_energy"]),
        float(compat["initial_energy"]),
        atol=1e-12,
    )


def test_asymmetric_offset_worker_payload_takes_non_parity_path():
    """A8: large-offset asymmetric potential. Without the A1 rtol fix,
    parity could be mis-detected and one block solved on a Hamiltonian
    that does not conserve parity. The fast (dispatched det path) and
    compat (legacy labelled) solves must succeed and agree on the
    physics. See worker_solver_payload for the 1e5 offset rationale.
    """
    audit = _load_audit_module()
    fast = audit.worker_solver_payload("current_fast", "interacting_offset_asymmetric_uu")
    compat = audit.worker_solver_payload(
        "current_compat", "interacting_offset_asymmetric_uu"
    )
    assert np.isclose(
        float(fast["energy"]), float(compat["energy"]), rtol=1e-8, atol=0
    )
    assert np.allclose(fast["density"], compat["density"], atol=1e-8, rtol=0)


# ---------------------------------------------------------------------------
# Phase S3.3: SolveContext audit-mode coverage
# ---------------------------------------------------------------------------


def test_solve_context_mode_is_registered():
    audit = _load_audit_module()
    assert "current_fast_ctx" in audit.MODES
    assert "current_fast_ctx" in audit._CURRENT_FAST_MODES


def test_solve_context_env_is_default():
    """current_fast_ctx must NOT set any IDEA_DET_* overrides.

    SolveContext is exercised under the default dispatch (PRIMME if
    available, scipy otherwise). The explicit-backend variants live in
    the current_fast_scipy / current_fast_primme / current_no_primme
    modes, not on ctx.
    """
    audit = _load_audit_module()
    env = audit.audit_env("current_fast_ctx")
    assert "IDEA_DET_EIGSH_BACKEND" not in env
    assert "IDEA_DET_PRIMME_MIN_DIM" not in env
    assert "IDEA_DET_DISABLE_PRIMME" not in env


def test_solve_context_imports_repo_root():
    audit = _load_audit_module()
    paths = audit.import_paths_for_mode("current_fast_ctx")
    assert paths[0] == str(audit.REPO_ROOT), (
        f"current_fast_ctx must import the repo, not the release site; "
        f"got {paths[0]!r}"
    )


def test_disorder_sweep_case_is_registered():
    audit = _load_audit_module()
    assert "interacting_disorder_sweep_uu" in audit.solver_cases()


def test_disorder_sweep_modes_include_ctx():
    """The disorder sweep case must run under all seven modes — six
    baseline modes for the per-call solves plus current_fast_ctx for
    the warm-started sweep — so the audit cross-checks ctx against
    every baseline.
    """
    audit = _load_audit_module()
    modes = audit.modes_for_solver_case("interacting_disorder_sweep_uu")
    assert "current_fast_ctx" in modes
    for baseline in (
        "release",
        "current_compat",
        "current_fast",
        "current_fast_scipy",
        "current_fast_primme",
        "current_no_primme",
    ):
        assert baseline in modes, f"missing baseline mode {baseline!r}"


def test_other_interacting_cases_omit_ctx_mode():
    """current_fast_ctx must NOT leak onto other interacting cases.

    SolveContext is only meaningful for a real sweep; the
    single-solve interacting cases stay on the original six modes so
    the audit report doesn't fill up with degenerate single-call ctx
    runs.
    """
    audit = _load_audit_module()
    for case in (
        "interacting_harmonic_uu",
        "interacting_harmonic_du",
        "interacting_harmonic_uu_stencil5",
        "interacting_offset_asymmetric_uu",
        "interacting_propagate_harmonic_uu_kick",
        "interacting_det_harmonic_uud",
    ):
        modes = audit.modes_for_solver_case(case)
        assert "current_fast_ctx" not in modes, (
            f"case {case!r} unexpectedly includes current_fast_ctx; "
            f"only the disorder sweep should drive ctx mode"
        )


def test_disorder_sweep_payload_agrees_with_current_fast():
    """SolveContext sweep must produce the same energies + densities
    per element as per-call fast-path solves.
    """
    audit = _load_audit_module()
    ctx = audit.worker_solver_payload(
        "current_fast_ctx", "interacting_disorder_sweep_uu"
    )
    fast = audit.worker_solver_payload(
        "current_fast", "interacting_disorder_sweep_uu"
    )
    assert ctx["energies"].shape == fast["energies"].shape
    assert ctx["densities"].shape == fast["densities"].shape
    assert np.allclose(ctx["energies"], fast["energies"], rtol=1e-10, atol=0)
    assert np.allclose(ctx["densities"], fast["densities"], atol=1e-10, rtol=0)


def test_disorder_sweep_payload_agrees_with_current_compat():
    """SolveContext sweep must agree with the legacy labelled solver
    (current_compat forces bypass_det=True), not just the fast path.
    Catches any fast-path-only bug that doesn't show up in ctx-vs-fast.
    """
    audit = _load_audit_module()
    ctx = audit.worker_solver_payload(
        "current_fast_ctx", "interacting_disorder_sweep_uu"
    )
    compat = audit.worker_solver_payload(
        "current_compat", "interacting_disorder_sweep_uu"
    )
    assert np.allclose(
        ctx["energies"], compat["energies"], rtol=1e-9, atol=1e-10
    )
    assert np.allclose(
        ctx["densities"], compat["densities"], atol=1e-9, rtol=0
    )
