"""Benchmark and baseline harness for iDEA.methods.interacting.solve.

Phase A measurement infrastructure ahead of the determinant-basis matrix-free
LinearOperator work outlined in notes/friend_optimization_advice.md.

Subcommands:

    time      Measure wall time, peak memory, energy across --runs iterations.
    generate  Save energy + density per case as a baseline .npz.
    compare   Run solver, load stored baseline, validate to rtol/atol.

Run from the repository root:

    python benchmarks/benchmark_interacting.py time --runs 3
    python benchmarks/benchmark_interacting.py generate
    python benchmarks/benchmark_interacting.py compare
"""

import argparse
import contextlib
import io
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Allow `from counting_operator import ...` when the script is run from the
# repo root via ``python benchmarks/benchmark_interacting.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import iDEA
import iDEA.methods.interacting
import iDEA.methods.interacting_det
import iDEA.observables
from counting_operator import CountingLinearOperator


DEFAULT_BASELINE_DIR = Path("benchmarks/interacting_baseline")
SOLVERS = ("labelled", "det")


def interacting_uu_80():
    x = np.linspace(-8, 8, 80)
    v_ext = 0.5 * 0.25**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons="uu")


def interacting_uud_60():
    # Smaller than the friend's prompt's uud_120 because the current labelled
    # tensor basis makes N=120 a ~15-minute generation. Once the determinant
    # basis lands in Phase B, larger cases become tractable.
    x = np.linspace(-8, 8, 60)
    v_ext = 0.5 * 0.25**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons="uud")


def interacting_uudd_30():
    # 4-electron stress at small grid. Labelled-basis dimension (2N)^4 = 12.96M
    # is reduced to C(30,2)^2 = 189,225 in the determinant basis (~70x cut).
    # The labelled iDEA.methods.interacting.solve stochastically fails on this
    # case (antisymmetrize() empties the eigenvector pool); only the determinant
    # solver iDEA.methods.interacting_det.solve produces it reliably.
    x = np.linspace(-8, 8, 30)
    v_ext = 0.5 * 0.25**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons="uudd")


CASES = {
    "interacting_uu_80": interacting_uu_80,
    "interacting_uud_60": interacting_uud_60,
    "interacting_uudd_30": interacting_uudd_30,
}


def peak_memory_mb():
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024**2)
    return rss / 1024


def solve_and_measure(s, solver="labelled"):
    """Run the chosen solver and return (elapsed, energy, density).

    The labelled solver prints to stdout unconditionally; redirect to keep
    benchmark output clean. The determinant solver returns a state whose
    density must be computed via the determinant-aware helper.
    """
    if solver == "labelled":
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            state = iDEA.methods.interacting.solve(s, k=0, bypass_det=True)
        elapsed = time.perf_counter() - start
        density = iDEA.observables.density(s, state=state)
        return elapsed, float(state.energy), density
    if solver == "det":
        start = time.perf_counter()
        state = iDEA.methods.interacting_det.solve(s, k=0)
        elapsed = time.perf_counter() - start
        density = iDEA.methods.interacting_det.density(s, state)
        return elapsed, float(state.energy), density
    raise ValueError(f"Unknown solver: {solver!r}; choose from {SOLVERS}")


def build_baseline_payload(s, energy, density):
    return {
        "x": s.x,
        "v_ext": s.v_ext,
        "dx": np.array(s.dx),
        "electron_count": np.array(s.count),
        "energy": np.array(energy),
        "density": density,
    }


@contextlib.contextmanager
def _count_det_matvecs(counter):
    """Monkey-patch interacting_det._solve_with_preconditioner to wrap the
    operator in a CountingLinearOperator. Restores on exit.

    Only meaningful for ``--solver det`` (labelled solver doesn't go
    through the helper). Counter is a single-element list so the
    contextmanager can mutate it from the closure.
    """
    import iDEA.methods.interacting_det as _det

    original = _det._solve_with_preconditioner

    def counting_helper(op, k=1, tol=0.0, diag_for_prec=None, v0=None, backend=None):
        counting_op = CountingLinearOperator(
            op.matvec, shape=op.shape, dtype=op.dtype
        )
        en, ev = original(
            counting_op, k=k, tol=tol,
            diag_for_prec=diag_for_prec, v0=v0, backend=backend,
        )
        counter[0] += counting_op.count
        return en, ev

    _det._solve_with_preconditioner = counting_helper
    try:
        yield
    finally:
        _det._solve_with_preconditioner = original


def time_cases(case_names, runs, solver="labelled", count_matvecs=False):
    for name in case_names:
        print(f"\n{name} (solver={solver})")
        s = CASES[name]()
        times = []
        matvec_counts = []
        last_energy = None
        last_density = None
        for _ in range(runs):
            counter = [0]
            if count_matvecs and solver == "det":
                with _count_det_matvecs(counter):
                    elapsed, energy, density = solve_and_measure(s, solver=solver)
            else:
                elapsed, energy, density = solve_and_measure(s, solver=solver)
            times.append(elapsed)
            matvec_counts.append(counter[0])
            last_energy = energy
            last_density = density
        median = statistics.median(times)
        print(
            f"  wall time median: {median:.3f}s "
            f"({runs} runs, min={min(times):.3f}s max={max(times):.3f}s)"
        )
        print(f"  peak memory:      {peak_memory_mb():.1f} MB (process cumulative)")
        if count_matvecs:
            if solver == "det":
                print(
                    f"  matvec count:     median={statistics.median(matvec_counts)} "
                    f"(min={min(matvec_counts)} max={max(matvec_counts)})"
                )
            else:
                print("  matvec count:     n/a (labelled solver not instrumented)")
        print(f"  energy:           {last_energy:.10f}")
        print(
            f"  density integral: {float(np.sum(last_density) * s.dx):.6f} "
            f"(should equal electron count = {s.count})"
        )


def generate_baselines(case_names, baseline_dir, solver="labelled"):
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for name in case_names:
        print(f"\ngenerating {name} (solver={solver})")
        s = CASES[name]()
        elapsed, energy, density = solve_and_measure(s, solver=solver)
        payload = build_baseline_payload(s, energy, density)
        path = baseline_dir / f"{name}.npz"
        np.savez_compressed(path, **payload)
        size_kb = path.stat().st_size / 1024
        print(f"  saved {path} ({size_kb:.1f} KB) in {elapsed:.3f}s")
        print(f"  energy:           {energy:.10f}")
        print(
            f"  density integral: {float(np.sum(density) * s.dx):.6f} "
            f"(should equal electron count = {s.count})"
        )


def compare_baselines(case_names, baseline_dir, rtol=1e-10, atol=1e-12, solver="labelled"):
    baseline_dir = Path(baseline_dir)
    any_fail = False
    for name in case_names:
        path = baseline_dir / f"{name}.npz"
        if not path.exists():
            print(f"FAIL {name}: baseline missing ({path})")
            any_fail = True
            continue
        s = CASES[name]()
        elapsed, energy, density = solve_and_measure(s, solver=solver)
        with np.load(path, allow_pickle=False) as baseline:
            ref_energy = float(baseline["energy"])
            ref_density = baseline["density"]
        energy_ok = np.isclose(energy, ref_energy, rtol=rtol, atol=atol)
        density_ok = np.allclose(density, ref_density, rtol=rtol, atol=atol)
        if energy_ok and density_ok:
            print(f"PASS {name} (solver={solver}, {elapsed:.3f}s)")
        else:
            any_fail = True
            print(f"FAIL {name} (solver={solver})")
            if not energy_ok:
                print(
                    f"  energy: current={energy:.16e} baseline={ref_energy:.16e} "
                    f"diff={abs(energy - ref_energy):.6e}"
                )
            if not density_ok:
                diff = float(np.max(np.abs(density - ref_density)))
                print(f"  density: max_abs_diff={diff:.6e}")
    return any_fail


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    time_parser = subparsers.add_parser("time")
    time_parser.add_argument("--runs", type=int, default=3)
    time_parser.add_argument("--case", choices=sorted(CASES), action="append")
    time_parser.add_argument("--solver", choices=SOLVERS, default="labelled")
    time_parser.add_argument(
        "--count-matvecs",
        action="store_true",
        help="Report the median matvec count per case (det solver only).",
    )

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    gen_parser.add_argument("--case", choices=sorted(CASES), action="append")
    gen_parser.add_argument("--solver", choices=SOLVERS, default="labelled")

    cmp_parser = subparsers.add_parser("compare")
    cmp_parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    cmp_parser.add_argument("--case", choices=sorted(CASES), action="append")
    cmp_parser.add_argument("--rtol", type=float, default=1e-10)
    cmp_parser.add_argument("--atol", type=float, default=1e-12)
    cmp_parser.add_argument("--solver", choices=SOLVERS, default="labelled")

    args = parser.parse_args(argv)

    if args.command is None:
        time_cases(sorted(CASES), runs=3)
        return 0

    case_names = args.case or sorted(CASES)

    if args.command == "time":
        time_cases(
            case_names,
            runs=args.runs,
            solver=args.solver,
            count_matvecs=args.count_matvecs,
        )
        return 0
    if args.command == "generate":
        generate_baselines(case_names, args.baseline_dir, solver=args.solver)
        return 0
    if args.command == "compare":
        any_fail = compare_baselines(
            case_names,
            args.baseline_dir,
            rtol=args.rtol,
            atol=args.atol,
            solver=args.solver,
        )
        return 1 if any_fail else 0

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
