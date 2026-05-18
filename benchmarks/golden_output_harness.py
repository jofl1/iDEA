"""Generate and compare local golden outputs for iDEA performance work.

Golden files are intentionally local to a machine/environment. For exact
comparisons, regenerate them whenever Python, NumPy, SciPy, BLAS, or iDEA's
trusted baseline implementation changes.

Examples:

    python benchmarks/golden_output_harness.py list
    python benchmarks/golden_output_harness.py generate
    python benchmarks/golden_output_harness.py compare --timing
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

import iDEA
import iDEA.methods.hartree_fock
import iDEA.methods.interacting
import iDEA.methods.non_interacting
import iDEA.observables
import iDEA.reverse_engineering


DEFAULT_GOLDEN_DIR = Path("benchmarks/golden_outputs")
METADATA_KEY = "__metadata__"


def _soft_harmonic_system(n_points=40, x_max=6.0, electrons="u"):
    x = np.linspace(-x_max, x_max, n_points)
    v_ext = 0.5 * 0.25**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons=electrons)


def _payload_with_system(s, outputs):
    payload = {
        "x": s.x,
        "v_ext": s.v_ext,
        "v_int": s.v_int,
        "dx": np.array(s.dx),
        "electron_count": np.array(s.count),
    }
    payload.update(outputs)
    return payload


def case_non_interacting_harmonic_u():
    s = _soft_harmonic_system(electrons="u")
    state = iDEA.methods.non_interacting.solve(s, k=0, silent=True)
    density = iDEA.observables.density(s, state=state)
    return _payload_with_system(
        s,
        {
            "up_energies": state.up.energies,
            "up_orbitals": state.up.orbitals,
            "up_occupations": state.up.occupations,
            "up_occupied": state.up.occupied,
            "down_energies": state.down.energies,
            "down_orbitals": state.down.orbitals,
            "down_occupations": state.down.occupations,
            "down_occupied": state.down.occupied,
            "density": density,
        },
    )


def case_hartree_fock_harmonic_uu():
    s = _soft_harmonic_system(electrons="uu")
    state = iDEA.methods.hartree_fock.solve(s, k=0, silent=True)
    density = iDEA.observables.density(s, state=state)
    density_matrix = iDEA.observables.density_matrix(s, state=state)
    return _payload_with_system(
        s,
        {
            "up_energies": state.up.energies,
            "up_orbitals": state.up.orbitals,
            "up_occupations": state.up.occupations,
            "up_occupied": state.up.occupied,
            "down_energies": state.down.energies,
            "down_orbitals": state.down.orbitals,
            "down_occupations": state.down.occupations,
            "down_occupied": state.down.occupied,
            "density": density,
            "density_matrix": density_matrix,
        },
    )


def case_interacting_harmonic_uu():
    s = _soft_harmonic_system(electrons="uu")
    state = iDEA.methods.interacting.solve(s, k=0)
    density = iDEA.observables.density(s, state=state)
    return _payload_with_system(
        s,
        {
            "energy": np.array(state.energy),
            "space": state.space,
            "spin": state.spin,
            "full": state.full,
            "density": density,
        },
    )


def case_reverse_ks_harmonic_uu():
    s = _soft_harmonic_system(electrons="uu")
    state = iDEA.methods.interacting.solve(s, k=0)
    target_n = iDEA.observables.density(s, state=state)
    s_ks = iDEA.reverse_engineering.reverse(
        s,
        target_n,
        iDEA.methods.non_interacting,
        mu=0.8,
        pe=0.1,
        tol=1e-4,
        silent=True,
    )
    state_ks = iDEA.methods.non_interacting.solve(s_ks, k=0, silent=True)
    density_ks = iDEA.observables.density(s_ks, state=state_ks)
    density_error = np.sum(abs(density_ks - target_n)) * s.dx
    return _payload_with_system(
        s,
        {
            "target_density": target_n,
            "v_ks": s_ks.v_ext,
            "ks_density": density_ks,
            "ks_density_error": np.array(density_error),
            "ks_up_energies": state_ks.up.energies,
            "ks_up_orbitals": state_ks.up.orbitals,
            "ks_down_energies": state_ks.down.energies,
            "ks_down_orbitals": state_ks.down.orbitals,
        },
    )


def case_interacting_propagation_uu():
    s = _soft_harmonic_system(n_points=30, x_max=5.0, electrons="uu")
    state = iDEA.methods.interacting.solve(s, k=0)
    t = np.linspace(0.0, 1.0, 8)
    v_ptrb = np.zeros(shape=t.shape + s.x.shape)
    for j, ti in enumerate(t):
        v_ptrb[j, :] = -0.05 * s.x * np.sin(ti)
    evolution = iDEA.methods.interacting.propagate(s, state, v_ptrb, t)
    density = iDEA.observables.density(s, evolution=evolution)
    return _payload_with_system(
        s,
        {
            "t": t,
            "v_ptrb": v_ptrb,
            "td_space": evolution.td_space,
            "td_density": density,
        },
    )


CASES = {
    "non_interacting_harmonic_u": case_non_interacting_harmonic_u,
    "hartree_fock_harmonic_uu": case_hartree_fock_harmonic_uu,
    "interacting_harmonic_uu": case_interacting_harmonic_uu,
    "reverse_ks_harmonic_uu": case_reverse_ks_harmonic_uu,
    "interacting_propagation_uu": case_interacting_propagation_uu,
}


def _metadata(case_name):
    return {
        "case": case_name,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "iDEA_file": str(Path(iDEA.__file__).resolve()),
    }


def _case_path(golden_dir, case_name):
    return Path(golden_dir) / f"{case_name}.npz"


def run_case(case_name):
    start = time.perf_counter()
    payload = CASES[case_name]()
    elapsed = time.perf_counter() - start
    payload[METADATA_KEY] = np.array(json.dumps(_metadata(case_name), sort_keys=True))
    return payload, elapsed


def generate(case_names, golden_dir=DEFAULT_GOLDEN_DIR):
    golden_dir = Path(golden_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case_name in case_names:
        payload, elapsed = run_case(case_name)
        path = _case_path(golden_dir, case_name)
        np.savez_compressed(path, **payload)
        results.append((case_name, path, elapsed))
    return results


def _array_difference_summary(current, golden):
    if current.shape != golden.shape:
        return f"shape current={current.shape} golden={golden.shape}"
    if current.dtype != golden.dtype:
        return f"dtype current={current.dtype} golden={golden.dtype}"
    if current.size == 0:
        return "empty arrays differ"
    if np.issubdtype(current.dtype, np.number):
        diff = np.abs(current - golden)
        return f"max_abs={np.max(diff):.17e}"
    return "non-numeric arrays differ"


def compare_payloads(current, golden, mode="exact", atol=0.0, rtol=0.0):
    current_keys = set(current)
    golden_keys = set(golden)
    failures = []

    if current_keys != golden_keys:
        failures.append(
            "keys differ: "
            f"current_only={sorted(current_keys - golden_keys)} "
            f"golden_only={sorted(golden_keys - current_keys)}"
        )
        return failures

    for key in sorted(current_keys):
        current_array = np.asarray(current[key])
        golden_array = np.asarray(golden[key])
        if mode == "exact":
            ok = np.array_equal(current_array, golden_array)
        elif mode == "close":
            if np.issubdtype(current_array.dtype, np.number):
                ok = np.allclose(current_array, golden_array, atol=atol, rtol=rtol)
            else:
                ok = np.array_equal(current_array, golden_array)
        else:
            raise ValueError(f"Unsupported compare mode: {mode}")
        if not ok:
            failures.append(f"{key}: {_array_difference_summary(current_array, golden_array)}")

    return failures


def compare(
    case_names,
    golden_dir=DEFAULT_GOLDEN_DIR,
    mode="exact",
    atol=0.0,
    rtol=0.0,
):
    results = []
    for case_name in case_names:
        path = _case_path(golden_dir, case_name)
        if not path.exists():
            results.append((case_name, path, None, [f"missing golden file: {path}"]))
            continue
        current, elapsed = run_case(case_name)
        with np.load(path, allow_pickle=False) as golden_npz:
            golden = {key: golden_npz[key] for key in golden_npz.files}
        failures = compare_payloads(current, golden, mode=mode, atol=atol, rtol=rtol)
        results.append((case_name, path, elapsed, failures))
    return results


def _selected_cases(selected):
    return selected or list(CASES)


def _print_generate_results(results):
    for case_name, path, elapsed in results:
        print(f"GENERATED {case_name}: {path} ({elapsed:.3f}s)")


def _print_compare_results(results, timing=False):
    failed = False
    for case_name, path, elapsed, failures in results:
        if failures:
            failed = True
            print(f"FAIL {case_name}: {path}")
            for failure in failures:
                print(f"  {failure}")
        else:
            suffix = f" ({elapsed:.3f}s)" if timing else ""
            print(f"PASS {case_name}{suffix}")
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--golden-dir", default=str(DEFAULT_GOLDEN_DIR))
    generate_parser.add_argument("--case", action="append", choices=sorted(CASES))

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--golden-dir", default=str(DEFAULT_GOLDEN_DIR))
    compare_parser.add_argument("--case", action="append", choices=sorted(CASES))
    compare_parser.add_argument("--mode", choices=("exact", "close"), default="exact")
    compare_parser.add_argument("--atol", type=float, default=0.0)
    compare_parser.add_argument("--rtol", type=float, default=0.0)
    compare_parser.add_argument("--timing", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "list":
        for case_name in sorted(CASES):
            print(case_name)
        return 0

    if args.command == "generate":
        results = generate(_selected_cases(args.case), golden_dir=args.golden_dir)
        _print_generate_results(results)
        return 0

    if args.command == "compare":
        results = compare(
            _selected_cases(args.case),
            golden_dir=args.golden_dir,
            mode=args.mode,
            atol=args.atol,
            rtol=args.rtol,
        )
        failed = _print_compare_results(results, timing=args.timing)
        return 1 if failed else 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
