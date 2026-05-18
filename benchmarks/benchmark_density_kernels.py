"""Benchmark static SingleBodyState density-kernel reference and optimized paths.

Run from the repository root:

    python benchmarks/benchmark_density_kernels.py --runs 5
"""

import argparse
import contextlib
import statistics
import time

import numpy as np

import iDEA
import iDEA.methods.hartree_fock
import iDEA.observables


def harmonic_system(n_points, electrons):
    x = np.linspace(-8, 8, n_points)
    v_ext = 0.5 * 0.25**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons=electrons)


def assert_arrays_equal(left, right, label):
    if isinstance(left, tuple):
        if not isinstance(right, tuple) or len(left) != len(right):
            raise AssertionError(f"{label}: tuple structure differs")
        for i, (left_array, right_array) in enumerate(zip(left, right)):
            assert_arrays_equal(left_array, right_array, f"{label}[{i}]")
        return

    if not np.array_equal(left, right):
        raise AssertionError(f"{label}: arrays differ")


def assert_states_equal(ref_state, opt_state):
    assert_arrays_equal(ref_state.up.energies, opt_state.up.energies, "up.energies")
    assert_arrays_equal(ref_state.up.orbitals, opt_state.up.orbitals, "up.orbitals")
    assert_arrays_equal(
        ref_state.up.occupations, opt_state.up.occupations, "up.occupations"
    )
    assert_arrays_equal(ref_state.up.occupied, opt_state.up.occupied, "up.occupied")
    assert_arrays_equal(
        ref_state.down.energies, opt_state.down.energies, "down.energies"
    )
    assert_arrays_equal(
        ref_state.down.orbitals, opt_state.down.orbitals, "down.orbitals"
    )
    assert_arrays_equal(
        ref_state.down.occupations, opt_state.down.occupations, "down.occupations"
    )
    assert_arrays_equal(
        ref_state.down.occupied, opt_state.down.occupied, "down.occupied"
    )


@contextlib.contextmanager
def reference_density_kernels():
    public_density = iDEA.observables.density
    public_density_matrix = iDEA.observables.density_matrix

    def density_reference(
        s, state=None, evolution=None, time_indices=None, return_spins=False
    ):
        if state is not None and type(state) == iDEA.state.SingleBodyState:
            return iDEA.observables._density_single_body_reference(
                s, state, return_spins=return_spins
            )
        return public_density(
            s,
            state=state,
            evolution=evolution,
            time_indices=time_indices,
            return_spins=return_spins,
        )

    def density_matrix_reference(
        s, state=None, evolution=None, time_indices=None, return_spins=False
    ):
        if state is not None and type(state) == iDEA.state.SingleBodyState:
            return iDEA.observables._density_matrix_single_body_reference(
                s, state, return_spins=return_spins
            )
        return public_density_matrix(
            s,
            state=state,
            evolution=evolution,
            time_indices=time_indices,
            return_spins=return_spins,
        )

    iDEA.observables.density = density_reference
    iDEA.observables.density_matrix = density_matrix_reference
    try:
        yield
    finally:
        iDEA.observables.density = public_density
        iDEA.observables.density_matrix = public_density_matrix


def time_call(func, runs):
    times = []
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def benchmark_density_matrix(n_points, runs, tol):
    s = harmonic_system(n_points, "uu")
    state = iDEA.methods.hartree_fock.solve(s, k=0, tol=tol, silent=True)

    ref = iDEA.observables._density_matrix_single_body_reference(
        s, state, return_spins=True
    )
    opt = iDEA.observables.density_matrix(s, state=state, return_spins=True)
    assert_arrays_equal(ref, opt, f"density_matrix_{n_points}")

    ref_median, _ = time_call(
        lambda: iDEA.observables._density_matrix_single_body_reference(
            s, state, return_spins=True
        ),
        runs,
    )
    opt_median, _ = time_call(
        lambda: iDEA.observables.density_matrix(s, state=state, return_spins=True),
        runs,
    )

    print(f"\ndensity_matrix_hf_uu_{n_points}")
    print(f"  reference median: {ref_median:.6f}s")
    print(f"  optimized median: {opt_median:.6f}s")
    print(f"  speedup:          {ref_median / opt_median:.2f}x")


def solve_hf_with_outputs(n_points, electrons, tol, use_reference):
    context = reference_density_kernels() if use_reference else contextlib.nullcontext()
    s = harmonic_system(n_points, electrons)
    with context:
        start = time.perf_counter()
        state = iDEA.methods.hartree_fock.solve(s, k=0, tol=tol, silent=True)
        elapsed = time.perf_counter() - start
        energy = iDEA.methods.hartree_fock.total_energy(s, state)
        density = iDEA.observables.density(s, state=state)
        density_matrix = iDEA.observables.density_matrix(s, state=state)
    return elapsed, state, energy, density, density_matrix


def benchmark_hartree_fock_solve(n_points, electrons, runs, tol):
    ref_times = []
    opt_times = []

    for _ in range(runs):
        ref_elapsed, ref_state, ref_energy, ref_density, ref_density_matrix = (
            solve_hf_with_outputs(n_points, electrons, tol, use_reference=True)
        )
        opt_elapsed, opt_state, opt_energy, opt_density, opt_density_matrix = (
            solve_hf_with_outputs(n_points, electrons, tol, use_reference=False)
        )

        assert_states_equal(ref_state, opt_state)
        assert_arrays_equal(np.array(ref_energy), np.array(opt_energy), "total_energy")
        assert_arrays_equal(ref_density, opt_density, "density")
        assert_arrays_equal(ref_density_matrix, opt_density_matrix, "density_matrix")

        ref_times.append(ref_elapsed)
        opt_times.append(opt_elapsed)

    ref_median = statistics.median(ref_times)
    opt_median = statistics.median(opt_times)

    print(f"\nhartree_fock_{electrons}_{n_points}")
    print(f"  reference median: {ref_median:.6f}s")
    print(f"  optimized median: {opt_median:.6f}s")
    print(f"  speedup:          {ref_median / opt_median:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()

    cases = {
        "density_matrix_160": lambda: benchmark_density_matrix(160, args.runs, args.tol),
        "density_matrix_240": lambda: benchmark_density_matrix(240, args.runs, args.tol),
        "hartree_fock_uu_160": lambda: benchmark_hartree_fock_solve(
            160, "uu", args.runs, args.tol
        ),
        "hartree_fock_ud_160": lambda: benchmark_hartree_fock_solve(
            160, "ud", args.runs, args.tol
        ),
    }

    selected_cases = args.case or sorted(cases)
    unknown_cases = sorted(set(selected_cases) - set(cases))
    if unknown_cases:
        raise ValueError(f"Unknown benchmark cases: {unknown_cases}")

    for case in selected_cases:
        cases[case]()


if __name__ == "__main__":
    main()
