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

import iDEA
import iDEA.methods.interacting
import iDEA.observables


DEFAULT_BASELINE_DIR = Path("benchmarks/interacting_baseline")


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


CASES = {
    "interacting_uu_80": interacting_uu_80,
    "interacting_uud_60": interacting_uud_60,
}

# 4-electron case (uudd_30) deliberately deferred to Phase B. The current
# interacting solver works in the labelled tensor basis and calls
# antisymmetrize() after the eigsh solve; for uudd at the default
# _estimate_level=4 the Lanczos-returned eigenvectors stochastically all
# antisymmetrize to zero, producing an IndexError. This is the exact failure
# mode the friend identified (notes/friend_optimization_advice.md): solving
# in the wrong basis. Once the determinant-basis matrix-free LinearOperator
# replaces antisymmetrize-after-solve, the 4-electron case will be added
# back at uudd_30 (and likely uudd_60).


def peak_memory_mb():
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024**2)
    return rss / 1024


def solve_and_measure(s):
    # interacting.solve prints "solving eigenproblem on CPU..." unconditionally;
    # suppress it so benchmark output stays clean.
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        state = iDEA.methods.interacting.solve(s, k=0)
    elapsed = time.perf_counter() - start
    density = iDEA.observables.density(s, state=state)
    return elapsed, float(state.energy), density


def build_baseline_payload(s, energy, density):
    return {
        "x": s.x,
        "v_ext": s.v_ext,
        "dx": np.array(s.dx),
        "electron_count": np.array(s.count),
        "energy": np.array(energy),
        "density": density,
    }


def time_cases(case_names, runs):
    for name in case_names:
        print(f"\n{name}")
        s = CASES[name]()
        times = []
        last_energy = None
        last_density = None
        for _ in range(runs):
            elapsed, energy, density = solve_and_measure(s)
            times.append(elapsed)
            last_energy = energy
            last_density = density
        median = statistics.median(times)
        print(
            f"  wall time median: {median:.3f}s "
            f"({runs} runs, min={min(times):.3f}s max={max(times):.3f}s)"
        )
        print(f"  peak memory:      {peak_memory_mb():.1f} MB (process cumulative)")
        print(f"  energy:           {last_energy:.10f}")
        print(
            f"  density integral: {float(np.sum(last_density) * s.dx):.6f} "
            f"(should equal electron count = {s.count})"
        )


def generate_baselines(case_names, baseline_dir):
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for name in case_names:
        print(f"\ngenerating {name}")
        s = CASES[name]()
        elapsed, energy, density = solve_and_measure(s)
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


def compare_baselines(case_names, baseline_dir, rtol=1e-10, atol=1e-12):
    baseline_dir = Path(baseline_dir)
    any_fail = False
    for name in case_names:
        path = baseline_dir / f"{name}.npz"
        if not path.exists():
            print(f"FAIL {name}: baseline missing ({path})")
            any_fail = True
            continue
        s = CASES[name]()
        elapsed, energy, density = solve_and_measure(s)
        with np.load(path, allow_pickle=False) as baseline:
            ref_energy = float(baseline["energy"])
            ref_density = baseline["density"]
        energy_ok = np.isclose(energy, ref_energy, rtol=rtol, atol=atol)
        density_ok = np.allclose(density, ref_density, rtol=rtol, atol=atol)
        if energy_ok and density_ok:
            print(f"PASS {name} ({elapsed:.3f}s)")
        else:
            any_fail = True
            print(f"FAIL {name}")
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

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    gen_parser.add_argument("--case", choices=sorted(CASES), action="append")

    cmp_parser = subparsers.add_parser("compare")
    cmp_parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    cmp_parser.add_argument("--case", choices=sorted(CASES), action="append")
    cmp_parser.add_argument("--rtol", type=float, default=1e-10)
    cmp_parser.add_argument("--atol", type=float, default=1e-12)

    args = parser.parse_args(argv)

    if args.command is None:
        time_cases(sorted(CASES), runs=3)
        return 0

    case_names = args.case or sorted(CASES)

    if args.command == "time":
        time_cases(case_names, runs=args.runs)
        return 0
    if args.command == "generate":
        generate_baselines(case_names, args.baseline_dir)
        return 0
    if args.command == "compare":
        any_fail = compare_baselines(
            case_names, args.baseline_dir, rtol=args.rtol, atol=args.atol
        )
        return 1 if any_fail else 0

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
