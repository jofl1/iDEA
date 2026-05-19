"""Release-equivalence audit for iDEA speedup work.

This script compares the optimized checkout against a fresh PyPI/release
``iDEA-latest==1.0.3`` install. It intentionally runs iDEA imports in
subprocesses with controlled ``PYTHONPATH`` so release and current code paths
cannot accidentally share modules.

Typical use from the repository root:

    python benchmarks/audit_release_equivalence.py setup
    python benchmarks/audit_release_equivalence.py solver-audit
    python benchmarks/audit_release_equivalence.py ch5-audit
    python benchmarks/audit_release_equivalence.py report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from statistics import median
from pathlib import Path

import numpy as np


AUDIT_ROOT = Path("/private/tmp/idea_release_audit")
RELEASE_SITE = AUDIT_ROOT / "release_site"
WORKER_CWD = AUDIT_ROOT / "worker_cwd"
LOG_DIR = AUDIT_ROOT / "logs"
SOLVER_DIR = AUDIT_ROOT / "solver_outputs"
CH5_DIR = AUDIT_ROOT / "ch5_outputs"
REPORT_JSON = AUDIT_ROOT / "report.json"
REPORT_MD = AUDIT_ROOT / "report.md"
TIMING_CSV = AUDIT_ROOT / "timings.csv"

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
BUILDER_PATH = Path("/Users/joshfleming/src/jofl1_diss/nearsightedness")
RELEASE_PACKAGE = "iDEA-latest==1.0.3"
RELEASE_DEPENDENCY_MODE = "no-deps; release iDEA package uses the current interpreter NumPy/SciPy"

MODES = (
    "release",
    "current_compat",
    "current_fast",
    # Phase F (second friend) audit gap A5/A6 — variants of current_fast
    # exercising the eigensolver-backend dispatch under explicit control.
    "current_fast_scipy",   # IDEA_DET_EIGSH_BACKEND=scipy (force fallback)
    "current_fast_primme",  # IDEA_DET_EIGSH_BACKEND=primme + threshold=0
    "current_no_primme",    # IDEA_DET_DISABLE_PRIMME=1 (treat as absent)
)

_CURRENT_FAST_MODES = (
    "current_fast",
    "current_fast_scipy",
    "current_fast_primme",
    "current_no_primme",
)
CH5_ARRAY_KEYS = (
    "x",
    "n",
    "v_ext",
    "v_H",
    "v_ks",
    "v_xc",
    "v_xc_raw",
    "v_xc_lda",
    "v_xc_lda_raw",
    "energy",
    "ks_density_error",
)


def ensure_dirs() -> None:
    for path in (AUDIT_ROOT, WORKER_CWD, LOG_DIR, SOLVER_DIR, CH5_DIR):
        path.mkdir(parents=True, exist_ok=True)


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def json_load(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def append_timing(row: dict) -> None:
    TIMING_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "command",
        "mode",
        "kind",
        "case",
        "elapsed_s",
        "output",
    ]
    exists = TIMING_CSV.exists()
    with TIMING_CSV.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        out = {key: row.get(key, "") for key in fieldnames}
        writer.writerow(out)


def audit_env(mode: str, builder_path: Path = BUILDER_PATH) -> dict:
    env = os.environ.copy()
    env.pop("IDEA_DET_EIGSH_BACKEND", None)
    env.pop("IDEA_DET_PRIMME_MIN_DIM", None)
    env.pop("IDEA_DET_DISABLE_PRIMME", None)
    if mode == "current_fast_scipy":
        env["IDEA_DET_EIGSH_BACKEND"] = "scipy"
    elif mode == "current_fast_primme":
        env["IDEA_DET_EIGSH_BACKEND"] = "primme"
        # Threshold 0 so PRIMME is exercised on every dispatched solve,
        # not just the cases above the default 2000-dimension cutoff.
        env["IDEA_DET_PRIMME_MIN_DIM"] = "0"
    elif mode == "current_no_primme":
        env["IDEA_DET_DISABLE_PRIMME"] = "1"
    env["MPLCONFIGDIR"] = str(AUDIT_ROOT / "matplotlib-cache" / mode)
    env["XDG_CACHE_HOME"] = str(AUDIT_ROOT / "xdg-cache" / mode)
    env["PYTHONPATH"] = os.pathsep.join(import_paths_for_mode(mode, builder_path))
    return env


def import_paths_for_mode(mode: str, builder_path: Path = BUILDER_PATH) -> list[str]:
    if mode == "release":
        return [str(RELEASE_SITE), str(builder_path)]
    if mode == "current_compat" or mode in _CURRENT_FAST_MODES:
        return [str(REPO_ROOT), str(builder_path)]
    raise ValueError(f"Unknown audit mode: {mode}")


def require_release_site() -> None:
    if not RELEASE_SITE.exists():
        raise SystemExit(
            f"Missing release install at {RELEASE_SITE}. "
            "Run: python benchmarks/audit_release_equivalence.py setup"
        )
    shadowed = release_site_shadowed_dependencies()
    if shadowed:
        joined = ", ".join(path.name for path in shadowed[:8])
        raise SystemExit(
            f"Release site contains dependency packages that would shadow the current runtime: {joined}. "
            "Run: python benchmarks/audit_release_equivalence.py setup --force"
        )


def release_site_shadowed_dependencies() -> list[Path]:
    dependency_names = (
        "numpy",
        "scipy",
        "matplotlib",
        "sklearn",
        "pandas",
        "numba",
        "llvmlite",
        "h5py",
    )
    if not RELEASE_SITE.exists():
        return []
    return [RELEASE_SITE / name for name in dependency_names if (RELEASE_SITE / name).exists()]


def run_subprocess(args: list[str], env: dict | None = None, log_name: str | None = None):
    ensure_dirs()
    start = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=str(WORKER_CWD),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if log_name is not None:
        (LOG_DIR / log_name).write_text(proc.stdout)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(f"Command failed with exit code {proc.returncode}: {args}")
    return proc, elapsed


def setup_release(force: bool = False) -> None:
    ensure_dirs()
    if RELEASE_SITE.exists() and any(RELEASE_SITE.iterdir()) and not force:
        print(f"Release site already exists: {RELEASE_SITE}")
        shadowed = release_site_shadowed_dependencies()
        if shadowed:
            joined = ", ".join(path.name for path in shadowed[:8])
            print(f"Dependency packages are present and would shadow the current runtime: {joined}")
        print("Use --force to recreate the package-only release target.")
        return
    if RELEASE_SITE.exists() and force:
        shutil.rmtree(RELEASE_SITE)
    RELEASE_SITE.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-deps",
        "--target",
        str(RELEASE_SITE),
        RELEASE_PACKAGE,
    ]
    print("Installing release baseline:")
    print("  " + " ".join(args))
    proc, elapsed = run_subprocess(args, log_name="setup_pip_install.log")
    print(proc.stdout)
    append_timing(
        {
            "timestamp": time.time(),
            "command": "setup",
            "mode": "release",
            "kind": "pip",
            "case": RELEASE_PACKAGE,
            "elapsed_s": f"{elapsed:.6f}",
            "output": str(RELEASE_SITE),
        }
    )


def worker_output_path(kind: str, mode: str, case_name: str) -> Path:
    root = SOLVER_DIR if kind == "solver" else CH5_DIR
    return root / mode / f"{case_name}.npz"


def worker_meta_path(output_path: Path) -> Path:
    return output_path.with_suffix(".json")


def run_worker(kind: str, mode: str, case_name: str, force: bool = False) -> Path:
    ensure_dirs()
    if mode == "release":
        require_release_site()
    output = worker_output_path(kind, mode, case_name)
    if output.exists() and worker_meta_path(output).exists() and not force:
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable,
        str(SCRIPT_PATH),
        "worker",
        "--kind",
        kind,
        "--mode",
        mode,
        "--case",
        case_name,
        "--output",
        str(output),
    ]
    log_name = f"{kind}_{mode}_{case_name}.log"
    proc, elapsed = run_subprocess(args, env=audit_env(mode), log_name=log_name)
    if proc.stdout.strip():
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    append_timing(
        {
            "timestamp": time.time(),
            "command": f"{kind}-audit",
            "mode": mode,
            "kind": kind,
            "case": case_name,
            "elapsed_s": f"{elapsed:.6f}",
            "output": str(output),
        }
    )
    return output


def solver_cases() -> list[str]:
    return [
        "import_info",
        "non_interacting_harmonic_uu",
        "hartree_potential_large_toeplitz",
        "reverse_ks_harmonic_uu",
        "interacting_harmonic_uu",
        "interacting_harmonic_ud",
        # A3: non-canonical spin orderings (reconstruction depends on
        # electron-axis order, not just counts).
        "interacting_harmonic_du",
        "interacting_harmonic_dud",
        "interacting_harmonic_ddu",
        # A7: representative non-default finite-difference stencils.
        "interacting_harmonic_uu_stencil5",
        "interacting_harmonic_uu_stencil11",
        # A8: large-offset asymmetric potential — exercises the parity
        # rtol fix end-to-end; must take the non-parity path.
        "interacting_offset_asymmetric_uu",
        "interacting_det_harmonic_uud",
    ]


def modes_for_solver_case(case_name: str) -> tuple[str, ...]:
    if case_name == "interacting_det_harmonic_uud":
        # No release counterpart (the legacy solver can't reliably
        # produce uud at any speed). Run the dispatched path under
        # every backend permutation so the variants cross-check each
        # other implicitly via the audit report.
        return _CURRENT_FAST_MODES
    if case_name.startswith("interacting_"):
        return MODES
    return ("release", "current_fast")


def ch5_cases(include_l50: bool = True) -> list[str]:
    cases = []
    for al, d in [
        (1.0, 4.0),
        (1.0, 6.0),
        (1.0, 8.0),
        (1.0, 10.0),
        (1.0, 12.0),
        (1.2, 4.0),
        (1.2, 6.0),
        (1.2, 8.0),
        (1.2, 10.0),
        (1.5, 4.0),
        (1.5, 6.0),
        (1.5, 8.0),
        (1.5, 10.0),
        (1.5, 12.0),
        (1.8, 4.0),
        (1.8, 6.0),
        (1.8, 8.0),
        (1.8, 10.0),
        (2.0, 10.0),
    ]:
        cases.append(ch5_case_name(al, d, 30.0))
    if include_l50:
        for d in (4.0, 6.0, 8.0, 10.0, 12.0):
            cases.append(ch5_case_name(1.0, d, 50.0))
    return cases


def ch5_case_name(al: float, d: float, length: float) -> str:
    suffix = f"_L{length}" if length != 30.0 else ""
    return f"AL{al}_AR2.0_d{d}{suffix}"


def parse_ch5_case(case_name: str) -> dict:
    parts = case_name.split("_")
    al = float(parts[0].removeprefix("AL"))
    ar = float(parts[1].removeprefix("AR"))
    d = float(parts[2].removeprefix("d"))
    length = 30.0
    for part in parts[3:]:
        if part.startswith("L"):
            length = float(part.removeprefix("L"))
    return {"AL": al, "AR": ar, "d": d, "L": length}


def solver_audit(force: bool = False, cases: list[str] | None = None) -> None:
    selected = cases or solver_cases()
    for case_name in selected:
        for mode in modes_for_solver_case(case_name):
            print(f"Running solver case {case_name} [{mode}]")
            run_worker("solver", mode, case_name, force=force)
    report()


def ch5_audit(
    force: bool = False,
    cases: list[str] | None = None,
    modes: list[str] | None = None,
    limit: int | None = None,
    include_l50: bool = True,
) -> None:
    selected_cases = cases or ch5_cases(include_l50=include_l50)
    if limit is not None:
        selected_cases = selected_cases[:limit]
    selected_modes = modes or list(MODES)
    for case_name in selected_cases:
        for mode in selected_modes:
            print(f"Running Ch5 case {case_name} [{mode}]")
            run_worker("ch5", mode, case_name, force=force)
    report()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def max_rel_diff(current: np.ndarray, reference: np.ndarray) -> float:
    denom = np.maximum(np.abs(reference), 1e-300)
    return float(np.max(np.abs(current - reference) / denom))


_SIGN_INVARIANT_KEY_TOKENS = ("space", "full", "det_amplitudes")


def is_sign_invariant_array_key(key: str) -> bool:
    """True if a single global sign flip is physically irrelevant for the
    array stored under ``key``.

    Applied to single-wavefunction arrays (``state.full``, ``state.space``,
    ``det_amplitudes``). Not applied to orbital matrices, where each
    column carries its own independent sign and a global flip is too
    coarse a check.
    """
    lowered = key.lower()
    return any(token in lowered for token in _SIGN_INVARIANT_KEY_TOKENS)


def compare_arrays(
    reference,
    current,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    sign_invariant: bool = False,
) -> dict:
    ref = np.asarray(reference)
    cur = np.asarray(current)
    result = {
        "reference_shape": list(ref.shape),
        "current_shape": list(cur.shape),
        "reference_dtype": str(ref.dtype),
        "current_dtype": str(cur.dtype),
        "exact": False,
        "physics": False,
        "sign_flip": False,
        "max_abs": None,
        "max_rel": None,
        "status": "contract_change",
    }
    if ref.shape != cur.shape or ref.dtype != cur.dtype:
        return result

    result["exact"] = bool(np.array_equal(ref, cur))
    if not result["exact"] and sign_invariant:
        # Try the sign-flipped match. For non-degenerate eigenstates a
        # global sign flip is a physically irrelevant convention; raw
        # array equality would otherwise classify it as a diff.
        if np.array_equal(-cur, ref):
            result["exact"] = True
            result["sign_flip"] = True
    if ref.size == 0:
        result["physics"] = result["exact"]
        result["status"] = "exact" if result["exact"] else "diff"
        return result

    if np.issubdtype(ref.dtype, np.number):
        diff_pos = np.abs(cur - ref)
        max_abs = float(np.max(diff_pos))
        max_rel = max_rel_diff(cur, ref)
        physics = bool(np.allclose(cur, ref, rtol=rtol, atol=atol))
        if sign_invariant and not physics:
            diff_neg = np.abs(cur + ref)
            max_abs_neg = float(np.max(diff_neg))
            if max_abs_neg < max_abs:
                max_abs = max_abs_neg
                max_rel = max_rel_diff(-cur, ref)
                result["sign_flip"] = True
            physics = bool(np.allclose(-cur, ref, rtol=rtol, atol=atol))
        result["max_abs"] = max_abs
        result["max_rel"] = max_rel
        result["physics"] = physics
    else:
        result["physics"] = result["exact"]
    if result["exact"]:
        result["status"] = "exact"
    elif result["physics"]:
        result["status"] = "physics"
    else:
        result["status"] = "diff"
    return result


def compare_npz_pair(
    reference_path: Path,
    current_path: Path,
    *,
    case_name: str,
    mode: str,
    kind: str,
) -> list[dict]:
    reference = load_npz(reference_path)
    current = load_npz(current_path)
    rows = []
    for key in sorted(set(reference) | set(current)):
        if key not in reference or key not in current:
            rows.append(
                {
                    "kind": kind,
                    "case": case_name,
                    "mode": mode,
                    "array": key,
                    "status": "contract_change",
                    "missing_in": "reference" if key not in reference else "current",
                }
            )
            continue
        row = compare_arrays(
            reference[key],
            current[key],
            sign_invariant=is_sign_invariant_array_key(key),
        )
        row.update({"kind": kind, "case": case_name, "mode": mode, "array": key})
        note = comparison_note(row)
        if note:
            row["intent"] = note
        rows.append(row)
    return rows


def comparison_note(row: dict) -> str:
    if row.get("status") == "exact":
        return ""
    kind = row.get("kind")
    case = row.get("case")
    mode = row.get("mode")
    array = row.get("array")

    if kind == "solver" and case == "import_info":
        return "import isolation smoke check; path/version differences are expected"
    if kind == "solver" and case.startswith("interacting_") and mode == "current_fast":
        return "known intentional determinant eigensolver/reconstruction fast-path difference"
    if kind == "solver" and case == "hartree_potential_large_toeplitz":
        return "large-grid Hartree fast path is physics-matched but not bitwise identical"
    if kind == "solver" and case in ("non_interacting_harmonic_uu", "reverse_ks_harmonic_uu"):
        if "energ" in str(array) or "orbital" in str(array) or "occupation" in str(array):
            return "known risk: subset-eigenpair sc_step changes full-spectrum public array shapes"
        return "derived density/potential physics-matched; not bitwise release-identical"
    if kind == "ch5" and mode == "current_compat":
        return "labelled interacting path matches release; current reverse/non-interacting path prevents bitwise Ch5 equality"
    if kind == "ch5" and mode == "current_fast":
        return "determinant interacting plus current reverse/non-interacting fast paths; requires physics/tolerance review"
    if row.get("status") == "physics":
        return "physics-matched numerical difference; review whether bitwise identity is required"
    if row.get("status") == "contract_change":
        return "requires API/compatibility decision"
    return "requires investigation"


def available_outputs(kind: str) -> list[tuple[str, str, Path]]:
    root = SOLVER_DIR if kind == "solver" else CH5_DIR
    rows = []
    if not root.exists():
        return rows
    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir():
            continue
        for path in sorted(mode_dir.glob("*.npz")):
            rows.append((mode_dir.name, path.stem, path))
    return rows


def build_comparisons() -> list[dict]:
    comparisons = []
    for kind in ("solver", "ch5"):
        outputs = {(mode, case): path for mode, case, path in available_outputs(kind)}
        cases = sorted({case for _mode, case in outputs})
        for case_name in cases:
            ref = outputs.get(("release", case_name))
            if ref is None:
                continue
            for mode in ("current_compat", "current_fast"):
                cur = outputs.get((mode, case_name))
                if cur is None:
                    continue
                comparisons.extend(
                    compare_npz_pair(ref, cur, case_name=case_name, mode=mode, kind=kind)
                )
    comparisons.extend(compare_ch5_metrics())
    return comparisons


def standalone_output_rows() -> list[dict]:
    rows = []
    for kind in ("solver", "ch5"):
        outputs = {(mode, case): path for mode, case, path in available_outputs(kind)}
        cases = sorted({case for _mode, case in outputs})
        for case_name in cases:
            if ("release", case_name) in outputs:
                continue
            for mode in MODES:
                path = outputs.get((mode, case_name))
                if path is None:
                    continue
                data = load_npz(path)
                rows.append(
                    {
                        "kind": kind,
                        "case": case_name,
                        "mode": mode,
                        "path": str(path),
                        "arrays": sorted(data),
                    }
                )
    return rows


def compute_ch5_metrics_for_mode(mode: str) -> dict | None:
    root = CH5_DIR / mode
    if not root.exists():
        return None

    required_cases = set(ch5_cases(include_l50=False))
    present_cases = {path.stem for path in root.glob("AL*_AR2.0_d*.npz") if "_L" not in path.stem}
    missing_cases = sorted(required_cases - present_cases)
    if missing_cases:
        return {
            "error": (
                "master-plot metrics require the full 19-case non-L50 set; "
                f"{len(missing_cases)} cases missing"
            )
        }

    try:
        from scipy.stats import spearmanr
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import cross_val_predict
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "scikit-learn/scipy metric dependencies unavailable"}

    def splitting_metric(n_in, v_in, k=5):
        x = n_in.reshape(-1, 1)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(x)
        _dist, indices = nn.kneighbors(x)
        return float(np.mean([np.std(v_in[indices[i]]) for i in range(len(x))]))

    def krr_loo_mae(x, y):
        alpha_grid = np.logspace(-8, -1, 8)
        gamma_grid = np.logspace(-3, 2, 12)
        x_scaled = StandardScaler().fit_transform(x)
        best = np.inf
        for alpha in alpha_grid:
            for gamma in gamma_grid:
                krr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
                pred = cross_val_predict(krr, x_scaled, y, cv=len(y))
                best = min(best, float(np.mean(np.abs(y - pred))))
        return float(best)

    rows = []
    for path in sorted(root.glob("AL*_AR2.0_d*.npz")):
        # Master plot excludes L=50 systems.
        if "_L" in path.stem:
            continue
        data = load_npz(path)
        x = data["x"]
        n = data["n"]
        v_xc = data["v_xc"] - data["v_xc"][-1]
        margin = max(3, len(x) // 20)
        sl = slice(margin, -margin)
        x_in = x[sl]
        n_in = n[sl]
        v_in = v_xc[sl]
        rows.append(
            {
                "case": path.stem,
                "splitting": splitting_metric(n_in, v_in),
                "mae_d1": krr_loo_mae(n_in.reshape(-1, 1), v_in),
                "mae_oracle": krr_loo_mae(np.column_stack([n_in, x_in]), v_in),
            }
        )

    if not rows:
        return None
    splitting = np.array([row["splitting"] for row in rows])
    mae_d1 = np.array([row["mae_d1"] for row in rows])
    mae_oracle = np.array([row["mae_oracle"] for row in rows])
    rho_d1, p_d1 = spearmanr(splitting, mae_d1)
    rho_oracle, p_oracle = spearmanr(splitting, mae_oracle)
    return {
        "rows": rows,
        "rho_d1": float(rho_d1),
        "p_d1": float(p_d1),
        "rho_oracle": float(rho_oracle),
        "p_oracle": float(p_oracle),
    }


def compare_ch5_metrics() -> list[dict]:
    reference = compute_ch5_metrics_for_mode("release")
    if reference is None or "error" in reference:
        return []
    rows = []
    ref_rows = {row["case"]: row for row in reference["rows"]}
    for mode in ("current_compat", "current_fast"):
        current = compute_ch5_metrics_for_mode(mode)
        if current is None or "error" in current:
            continue
        cur_rows = {row["case"]: row for row in current["rows"]}
        for case in sorted(set(ref_rows) | set(cur_rows)):
            if case not in ref_rows or case not in cur_rows:
                rows.append(
                    {
                        "kind": "ch5_metric",
                        "case": case,
                        "mode": mode,
                        "array": "metric_row",
                        "status": "contract_change",
                    }
                )
                continue
            for key in ("splitting", "mae_d1", "mae_oracle"):
                row = compare_arrays(np.array(ref_rows[case][key]), np.array(cur_rows[case][key]))
                row.update({"kind": "ch5_metric", "case": case, "mode": mode, "array": key})
                rows.append(row)
        for key in ("rho_d1", "p_d1", "rho_oracle", "p_oracle"):
            row = compare_arrays(np.array(reference[key]), np.array(current[key]))
            row.update({"kind": "ch5_metric", "case": "master_plot", "mode": mode, "array": key})
            rows.append(row)
    return rows


def report() -> None:
    ensure_dirs()
    comparisons = build_comparisons()
    timings = []
    if TIMING_CSV.exists():
        with TIMING_CSV.open(newline="") as handle:
            timings = list(csv.DictReader(handle))
    payload = {
        "audit_root": str(AUDIT_ROOT),
        "repo_root": str(REPO_ROOT),
        "release_site": str(RELEASE_SITE),
        "release_package": RELEASE_PACKAGE,
        "release_dependency_mode": RELEASE_DEPENDENCY_MODE,
        "builder_path": str(BUILDER_PATH),
        "generated_at": time.time(),
        "platform": platform.platform(),
        "comparisons": comparisons,
        "standalone_outputs": standalone_output_rows(),
        "timings": timings,
    }
    json_dump(REPORT_JSON, payload)
    REPORT_MD.write_text(render_markdown_report(payload))
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")


def aggregate_comparison_rows(comparisons: list[dict]) -> list[dict]:
    rows: dict[tuple[str, str, str], dict] = {}
    statuses = ("exact", "physics", "diff", "contract_change")
    for row in comparisons:
        key = (row.get("kind", ""), row.get("case", ""), row.get("mode", ""))
        target = rows.setdefault(
            key,
            {
                "kind": key[0],
                "case": key[1],
                "mode": key[2],
                "arrays": 0,
                "exact": 0,
                "physics": 0,
                "diff": 0,
                "contract_change": 0,
            },
        )
        target["arrays"] += 1
        status = row.get("status", "diff")
        if status in statuses:
            target[status] += 1
        else:
            target["diff"] += 1
    output = []
    for row in rows.values():
        row["exact_pass"] = row["arrays"] > 0 and row["exact"] == row["arrays"]
        row["tolerance_pass"] = row["arrays"] > 0 and row["exact"] + row["physics"] == row["arrays"]
        output.append(row)
    return sorted(output, key=lambda item: (item["kind"], item["case"], item["mode"]))


def timing_speedup_rows(timings: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in timings:
        try:
            elapsed = float(row.get("elapsed_s", ""))
        except (TypeError, ValueError):
            continue
        key = (
            row.get("command", ""),
            row.get("kind", ""),
            row.get("case", ""),
            row.get("mode", ""),
        )
        grouped.setdefault(key, []).append(elapsed)

    output = []
    logical_cases = sorted({(command, kind, case) for command, kind, case, _mode in grouped})
    for command, kind, case in logical_cases:
        release_values = grouped.get((command, kind, case, "release"))
        if not release_values:
            continue
        release_median = median(release_values)
        for mode in ("current_compat", "current_fast"):
            current_values = grouped.get((command, kind, case, mode))
            if not current_values:
                continue
            current_median = median(current_values)
            output.append(
                {
                    "command": command,
                    "kind": kind,
                    "case": case,
                    "mode": mode,
                    "release_median_s": release_median,
                    "current_median_s": current_median,
                    "speedup": release_median / current_median if current_median else float("inf"),
                    "release_runs": len(release_values),
                    "current_runs": len(current_values),
                }
            )
    return output


def render_markdown_report(payload: dict) -> str:
    comparisons = payload["comparisons"]
    aggregate_rows = aggregate_comparison_rows(comparisons)
    speedup_rows = timing_speedup_rows(payload.get("timings", []))
    standalone_rows = payload.get("standalone_outputs", [])
    counts = {}
    for row in comparisons:
        counts[row.get("status", "unknown")] = counts.get(row.get("status", "unknown"), 0) + 1

    lines = [
        "# iDEA Release-Equivalence Audit",
        "",
        f"- Audit root: `{payload['audit_root']}`",
        f"- Repo root: `{payload['repo_root']}`",
        f"- Release site: `{payload['release_site']}`",
        f"- Release package: `{payload['release_package']}`",
        f"- Release dependency mode: `{payload['release_dependency_mode']}`",
        f"- Builder path: `{payload['builder_path']}`",
        "",
        "## Summary",
        "",
    ]
    if counts:
        for status in sorted(counts):
            lines.append(f"- `{status}`: {counts[status]}")
    else:
        lines.append("- No comparisons generated yet.")

    lines.extend(
        [
            "",
            "## Exact-Match Table",
            "",
            "| kind | case | mode | arrays | exact | physics | diff | contract | exact_pass |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            f"| {row['kind']} | {row['case']} | {row['mode']} | {row['arrays']} | "
            f"{row['exact']} | {row['physics']} | {row['diff']} | {row['contract_change']} | "
            f"{row['exact_pass']} |"
        )

    lines.extend(
        [
            "",
            "## Tolerance-Match Table",
            "",
            "| kind | case | mode | arrays | exact_or_physics | failures | tolerance_pass |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in aggregate_rows:
        exact_or_physics = row["exact"] + row["physics"]
        failures = row["arrays"] - exact_or_physics
        lines.append(
            f"| {row['kind']} | {row['case']} | {row['mode']} | {row['arrays']} | "
            f"{exact_or_physics} | {failures} | {row['tolerance_pass']} |"
        )

    lines.extend(
        [
            "",
            "## Runtime Speedups",
            "",
            "| command | kind | case | mode | release_median_s | current_median_s | speedup | runs |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    if speedup_rows:
        for row in speedup_rows:
            lines.append(
                f"| {row['command']} | {row['kind']} | {row['case']} | {row['mode']} | "
                f"{row['release_median_s']:.6f} | {row['current_median_s']:.6f} | "
                f"{row['speedup']:.3f} | {row['release_runs']}/{row['current_runs']} |"
            )
    else:
        lines.append("|  |  |  |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Standalone Outputs Without Release Pair",
            "",
            "| kind | case | mode | arrays | path |",
            "|---|---|---|---|---|",
        ]
    )
    if standalone_rows:
        for row in standalone_rows:
            lines.append(
                f"| {row['kind']} | {row['case']} | {row['mode']} | "
                f"{', '.join(row['arrays'])} | `{row['path']}` |"
            )
    else:
        lines.append("|  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Non-Exact Or Contract-Changing Outputs",
            "",
            "| kind | case | mode | array | status | ref shape | cur shape | ref dtype | cur dtype | max_abs | max_rel | note |",
            "|---|---|---|---|---|---|---|---|---|---:|---:|---|",
        ]
    )
    for row in comparisons:
        if row.get("status") == "exact":
            continue
        lines.append(
            "| {kind} | {case} | {mode} | {array} | {status} | {ref_shape} | {cur_shape} | "
            "{ref_dtype} | {cur_dtype} | {max_abs} | {max_rel} | {note} |".format(
                kind=row.get("kind", ""),
                case=row.get("case", ""),
                mode=row.get("mode", ""),
                array=row.get("array", ""),
                status=row.get("status", ""),
                ref_shape=row.get("reference_shape", ""),
                cur_shape=row.get("current_shape", ""),
                ref_dtype=row.get("reference_dtype", ""),
                cur_dtype=row.get("current_dtype", ""),
                max_abs=format_optional_float(row.get("max_abs")),
                max_rel=format_optional_float(row.get("max_rel")),
                note=row.get("intent", row.get("missing_in", "")),
            )
        )

    lines.extend(["", "## Timing Rows", "", "| command | mode | kind | case | elapsed_s |", "|---|---|---|---|---:|"])
    for row in payload.get("timings", []):
        lines.append(
            f"| {row.get('command','')} | {row.get('mode','')} | {row.get('kind','')} | "
            f"{row.get('case','')} | {row.get('elapsed_s','')} |"
        )

    lines.extend(
        [
            "",
            "## Known Audit Interpretation Rules",
            "",
            "- `exact` means `np.array_equal` against PyPI/release output.",
            "- `physics` means shape/dtype match and `np.allclose(..., rtol=1e-10, atol=1e-12)` passes.",
            "- `contract_change` means shape, dtype, or key presence differs.",
            "- determinant/PRIMME `current_fast` differences are expected to be non-bitwise and must be judged by physics rows.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_optional_float(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.6e}"


def worker(kind: str, mode: str, case_name: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if kind == "solver":
        payload = worker_solver_payload(mode, case_name)
    elif kind == "ch5":
        payload = worker_ch5_payload(mode, case_name)
    else:
        raise ValueError(f"Unknown worker kind: {kind}")
    elapsed = time.perf_counter() - start
    np.savez_compressed(output, **payload)
    meta = worker_metadata(mode, kind, case_name, elapsed)
    json_dump(worker_meta_path(output), meta)
    print(f"Wrote {output} ({elapsed:.3f}s)")


def worker_metadata(mode: str, kind: str, case_name: str, elapsed: float) -> dict:
    import iDEA
    import scipy

    return {
        "mode": mode,
        "kind": kind,
        "case": case_name,
        "elapsed_s": elapsed,
        "iDEA_file": str(Path(iDEA.__file__).resolve()),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "sys_path_prefix": sys.path[:5],
    }


def harmonic_system(n_points=40, electrons="uu"):
    import iDEA

    x = np.linspace(-6, 6, n_points)
    v_ext = 0.5 * 0.25**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons=electrons)


def worker_solver_payload(mode: str, case_name: str) -> dict:
    import iDEA
    import iDEA.methods.hartree_fock
    import iDEA.methods.interacting
    import iDEA.methods.non_interacting
    import iDEA.observables
    import iDEA.reverse_engineering

    if case_name == "import_info":
        import scipy

        return {
            "iDEA_file": np.array(str(Path(iDEA.__file__).resolve())),
            "version": np.array(getattr(iDEA, "__version__", "unknown")),
            "numpy_version": np.array(np.__version__),
            "scipy_version": np.array(scipy.__version__),
            "mode": np.array(mode),
        }

    if case_name == "non_interacting_harmonic_uu":
        s = harmonic_system(n_points=48, electrons="uu")
        state = iDEA.methods.non_interacting.solve(s, k=0, silent=True)
        density = iDEA.observables.density(s, state=state)
        density_matrix = iDEA.observables.density_matrix(s, state=state)
        return {
            "x": s.x,
            "up_energies": state.up.energies,
            "up_orbitals": state.up.orbitals,
            "up_occupations": state.up.occupations,
            "down_energies": state.down.energies,
            "down_orbitals": state.down.orbitals,
            "down_occupations": state.down.occupations,
            "density": density,
            "density_matrix": density_matrix,
        }

    if case_name == "hartree_potential_large_toeplitz":
        s = harmonic_system(n_points=512, electrons="uu")
        density = 2.0 * np.exp(-(s.x / 2.5) ** 2)
        density = density / (np.sum(density) * s.dx) * s.count
        v_h = iDEA.observables.hartree_potential(s, density)
        return {"x": s.x, "density": density, "v_H": v_h}

    if case_name == "reverse_ks_harmonic_uu":
        s = harmonic_system(n_points=48, electrons="uu")
        target_n = 2.0 * np.exp(-(s.x / 2.0) ** 2)
        target_n = target_n / (np.sum(target_n) * s.dx) * s.count
        s_ks = iDEA.reverse_engineering.reverse(
            s,
            target_n,
            iDEA.methods.non_interacting,
            mu=0.8,
            pe=0.1,
            tol=1e-5,
            silent=True,
        )
        state_ks = iDEA.methods.non_interacting.solve(s_ks, k=0, silent=True)
        n_ks = iDEA.observables.density(s_ks, state=state_ks)
        return {
            "target_n": target_n,
            "v_ext": s_ks.v_ext,
            "ks_density": n_ks,
            "ks_density_error": np.array(np.sum(np.abs(n_ks - target_n)) * s.dx),
            "ks_up_energies": state_ks.up.energies,
            "ks_up_orbitals": state_ks.up.orbitals,
            "ks_down_energies": state_ks.down.energies,
            "ks_down_orbitals": state_ks.down.orbitals,
        }

    if case_name in (
        "interacting_harmonic_uu",
        "interacting_harmonic_ud",
        "interacting_harmonic_du",
        "interacting_harmonic_dud",
        "interacting_harmonic_ddu",
    ):
        electrons = case_name.rsplit("_", 1)[-1]
        s = harmonic_system(n_points=32, electrons=electrons)
        kwargs = {}
        if mode == "current_compat":
            kwargs["bypass_det"] = True
        state = iDEA.methods.interacting.solve(s, k=0, **kwargs)
        density = iDEA.observables.density(s, state=state)
        return {
            "x": s.x,
            "energy": np.array(state.energy),
            "space": state.space,
            "spin": state.spin,
            "full": state.full,
            "density": density,
        }

    if case_name.startswith("interacting_harmonic_uu_stencil"):
        # A7: same harmonic problem, different finite-difference stencil.
        stencil = int(case_name.rsplit("stencil", 1)[-1])
        s = harmonic_system(n_points=32, electrons="uu")
        s.stencil = stencil
        kwargs = {}
        if mode == "current_compat":
            kwargs["bypass_det"] = True
        state = iDEA.methods.interacting.solve(s, k=0, **kwargs)
        density = iDEA.observables.density(s, state=state)
        return {
            "x": s.x,
            "stencil": np.array(stencil),
            "energy": np.array(state.energy),
            "density": density,
        }

    if case_name == "interacting_offset_asymmetric_uu":
        # A8: v_ext = harmonic + 1e6 offset + asymmetric kink. Must take
        # the non-parity path (covers the rtol fix end-to-end).
        n_points = 28
        x = np.linspace(-6, 6, n_points)
        v_ext = 0.5 * 0.25**2 * x**2 + 1e6
        v_ext[n_points // 3] += 0.1  # asymmetric kink
        v_int = iDEA.interactions.softened_interaction(x)
        s = iDEA.system.System(x, v_ext, v_int, electrons="uu")
        kwargs = {}
        if mode == "current_compat":
            kwargs["bypass_det"] = True
        state = iDEA.methods.interacting.solve(s, k=0, **kwargs)
        density = iDEA.observables.density(s, state=state)
        return {
            "x": s.x,
            "v_ext": s.v_ext,
            "energy": np.array(state.energy),
            "density": density,
        }

    if case_name == "interacting_det_harmonic_uud":
        import iDEA.methods.interacting_det as det

        s = harmonic_system(n_points=24, electrons="uud")
        state = det.solve(s, k=0)
        density = det.density(s, state)
        comps = det._build_solver_components(s)
        parity_eligible = det._v_ext_is_parity_symmetric(s) and det._v_int_is_parity_symmetric(s)
        expected_solve_dim = comps.D // 2 if parity_eligible else comps.D
        backend = det._resolve_backend(expected_solve_dim)
        psi = state.det_amplitudes.ravel()
        residual = np.linalg.norm(comps.op @ psi - state.energy * psi) / np.linalg.norm(psi)
        return {
            "x": s.x,
            "energy": np.array(state.energy),
            "density": density,
            "det_amplitudes": state.det_amplitudes,
            "parity": np.array(state.parity),
            "parity_eligible": np.array(parity_eligible),
            "det_basis_dim": np.array(comps.D),
            "expected_solve_dim": np.array(expected_solve_dim),
            "backend": np.array(backend),
            "residual": np.array(residual),
        }

    raise ValueError(f"Unknown solver case: {case_name}")


def worker_ch5_payload(mode: str, case_name: str) -> dict:
    import iDEA
    import iDEA.methods.interacting
    import iDEA.methods.lda
    import iDEA.methods.non_interacting
    import iDEA.observables
    import iDEA.reverse_engineering
    from system_builders import build_system

    params = parse_ch5_case(case_name)
    system_params = {
        "L": params["L"],
        "d": params["d"],
        "A_left": params["AL"],
        "A_right": params["AR"],
        "s": 1.0,
        "dx": 0.25,
        "electrons": "uu",
        "interaction_strength": 1.0,
        "interaction_softening": 1.0,
    }
    system = build_system("E", system_params)
    solve_kwargs = {}
    if mode == "current_compat":
        solve_kwargs["bypass_det"] = True
    state = iDEA.methods.interacting.solve(system, k=0, **solve_kwargs)
    n = iDEA.observables.density(system, state=state)
    s_ks = iDEA.reverse_engineering.reverse(
        system,
        n,
        iDEA.methods.non_interacting,
        mu=0.8,
        pe=0.1,
        tol=1e-5,
        silent=True,
    )
    v_ks = s_ks.v_ext
    v_h = iDEA.observables.hartree_potential(system, n)
    v_xc_raw = v_ks - system.v_ext - v_h
    v_xc = v_xc_raw - v_xc_raw[-1]
    v_xc_lda_raw = iDEA.methods.lda.exchange_correlation_potential(system, n)
    v_xc_lda = v_xc_lda_raw - v_xc_lda_raw[-1]
    state_ks = iDEA.methods.non_interacting.solve(s_ks, k=0, silent=True)
    n_ks = iDEA.observables.density(s_ks, state=state_ks)
    ks_err = np.sum(np.abs(n_ks - n)) * system.dx
    return {
        "x": system.x,
        "n": n,
        "v_ext": system.v_ext,
        "v_H": v_h,
        "v_ks": v_ks,
        "v_xc": v_xc,
        "v_xc_raw": v_xc_raw,
        "v_xc_lda": v_xc_lda,
        "v_xc_lda_raw": v_xc_lda_raw,
        "energy": np.array(state.energy),
        "ks_density_error": np.array(ks_err),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup")
    setup_parser.add_argument("--force", action="store_true")

    solver_parser = subparsers.add_parser("solver-audit")
    solver_parser.add_argument("--force", action="store_true")
    solver_parser.add_argument("--case", action="append", choices=solver_cases())

    ch5_parser = subparsers.add_parser("ch5-audit")
    ch5_parser.add_argument("--force", action="store_true")
    ch5_parser.add_argument("--case", action="append")
    ch5_parser.add_argument("--mode", action="append", choices=MODES)
    ch5_parser.add_argument("--limit", type=int)
    ch5_parser.add_argument("--no-l50", action="store_true")

    subparsers.add_parser("report")

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--kind", choices=("solver", "ch5"), required=True)
    worker_parser.add_argument("--mode", choices=MODES, required=True)
    worker_parser.add_argument("--case", required=True)
    worker_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "setup":
        setup_release(force=args.force)
    elif args.command == "solver-audit":
        solver_audit(force=args.force, cases=args.case)
    elif args.command == "ch5-audit":
        ch5_audit(
            force=args.force,
            cases=args.case,
            modes=args.mode,
            limit=args.limit,
            include_l50=not args.no_l50,
        )
    elif args.command == "report":
        report()
    elif args.command == "worker":
        worker(args.kind, args.mode, args.case, args.output)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
