"""Cross-machine baseline check for the smallest interacting case.

Loads benchmarks/interacting_baseline/interacting_uu_80.npz (committed) and
verifies the current interacting solver still reproduces energy and density
to within rtol=1e-10. Larger cases (uud_120, uudd_60) are validated via
``python benchmarks/benchmark_interacting.py compare`` to keep pytest fast.
"""

import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

import iDEA.methods.interacting
import iDEA.observables


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "benchmarks" / "interacting_baseline"


def _load_benchmark_module():
    path = REPO_ROOT / "benchmarks" / "benchmark_interacting.py"
    spec = importlib.util.spec_from_file_location("benchmark_interacting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case_name", ["interacting_uu_80"])
def test_interacting_baseline_small_matches(case_name):
    baseline_path = BASELINE_DIR / f"{case_name}.npz"
    if not baseline_path.exists():
        pytest.skip(
            f"baseline missing: {baseline_path}. "
            f"Generate with: python benchmarks/benchmark_interacting.py generate"
        )

    benchmark = _load_benchmark_module()
    s = benchmark.CASES[case_name]()

    with contextlib.redirect_stdout(io.StringIO()):
        state = iDEA.methods.interacting.solve(s, k=0)
    density = iDEA.observables.density(s, state=state)

    with np.load(baseline_path, allow_pickle=False) as baseline:
        ref_energy = float(baseline["energy"])
        ref_density = baseline["density"]

    assert np.isclose(float(state.energy), ref_energy, rtol=1e-10, atol=1e-12), (
        f"energy: current={float(state.energy):.16e} "
        f"baseline={ref_energy:.16e} "
        f"diff={abs(float(state.energy) - ref_energy):.6e}"
    )
    assert np.allclose(density, ref_density, rtol=1e-10, atol=1e-12), (
        f"density: max_abs_diff={float(np.max(np.abs(density - ref_density))):.6e}"
    )
