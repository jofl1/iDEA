"""Benchmark reverse-engineering reference and optimized KS inversion paths.

Run from the repository root:

    python benchmarks/benchmark_reverse_engineering.py
"""

import argparse
import statistics
import time

import numpy as np

import iDEA
import iDEA.methods.interacting
import iDEA.methods.non_interacting
import iDEA.observables
import iDEA.reverse_engineering


def harmonic_uu_80():
    x = np.linspace(-8, 8, 80)
    v_ext = 0.5 * 0.25**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons="uu")


def double_well_uu():
    x = np.linspace(-15, 15, 121)
    d = 10.0
    v_ext = -2.0 / (abs(x - 0.5 * d) + 1.0)
    v_ext += -2.0 / (abs(x + 0.5 * d) + 1.0)
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons="uu")


CASES = {
    "harmonic_uu_80": harmonic_uu_80,
    "double_well_uu": double_well_uu,
}


def target_density(s):
    state = iDEA.methods.interacting.solve(s, k=0)
    return iDEA.observables.density(s, state=state)


def reverse_error(s, target_n, s_ks):
    state = iDEA.methods.non_interacting.solve(s_ks, k=0, silent=True)
    n = iDEA.observables.density(s_ks, state=state)
    return float(np.sum(abs(n - target_n)) * s.dx)


def run_once(reverse_func, s, target_n, tol):
    start = time.perf_counter()
    s_ks = reverse_func(
        s,
        target_n,
        iDEA.methods.non_interacting,
        mu=0.8,
        pe=0.1,
        tol=tol,
        silent=True,
    )
    return time.perf_counter() - start, s_ks


def benchmark_case(name, runs, tol):
    print(f"\n{name}")
    s = CASES[name]()
    target_n = target_density(s)

    ref_times = []
    opt_times = []
    ref_error = None
    opt_error = None
    for _ in range(runs):
        ref_elapsed, ref = run_once(
            iDEA.reverse_engineering._reverse_reference, s, target_n, tol
        )
        opt_elapsed, opt = run_once(iDEA.reverse_engineering.reverse, s, target_n, tol)
        if not np.array_equal(ref.v_ext, opt.v_ext):
            raise AssertionError(f"{name}: optimized v_ext differs from reference")
        ref_error = reverse_error(s, target_n, ref)
        opt_error = reverse_error(s, target_n, opt)
        if ref_error != opt_error:
            raise AssertionError(f"{name}: optimized density error differs from reference")
        ref_times.append(ref_elapsed)
        opt_times.append(opt_elapsed)

    ref_median = statistics.median(ref_times)
    opt_median = statistics.median(opt_times)
    speedup = ref_median / opt_median

    print(f"  reference median: {ref_median:.3f}s")
    print(f"  optimized median: {opt_median:.3f}s")
    print(f"  speedup:          {speedup:.2f}x")
    print(f"  density error:    {opt_error:.6e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    args = parser.parse_args()

    case_names = args.case or sorted(CASES)
    for name in case_names:
        benchmark_case(name, args.runs, args.tol)


if __name__ == "__main__":
    main()
