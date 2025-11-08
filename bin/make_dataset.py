#!/usr/bin/env python3
"""Generate a dataset of nearsightedness metrics for simple model systems."""
from __future__ import annotations

import argparse
import csv
import pathlib
import subprocess
import sys
from datetime import datetime
from typing import Iterable, List

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nearsighted.grids import uniform_grid
from nearsighted.hartree import v_H_from_density
from nearsighted.ks_extract import solve_schrodinger_1d
from nearsighted.metric_core import compute_nearsightedness_metrics

DEFAULT_LENGTH = 10.0
DEFAULT_GRID = 201


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path, help="Output CSV file")
    parser.add_argument("--npz-dir", type=pathlib.Path, default=pathlib.Path("datasets"))
    parser.add_argument("--k", type=float, nargs="*", default=[0.5, 1.0])
    parser.add_argument("--lambda", dest="lambda_soft", type=float, nargs="*", default=[10.0, 50.0])
    parser.add_argument("--grid", type=int, nargs="*", default=[DEFAULT_GRID])
    parser.add_argument("--length", type=float, default=DEFAULT_LENGTH)
    parser.add_argument("--spins", nargs="*", default=["u", "ud", "uu"])
    return parser.parse_args()


def ho_density(x: np.ndarray, v_ext: np.ndarray, occupancy: Iterable[int]) -> np.ndarray:
    energies, orbitals = solve_schrodinger_1d(x, v_ext)
    density = np.zeros_like(x)
    for idx in occupancy:
        density += np.abs(orbitals[idx]) ** 2
    return density


def compute_system(system_id: str, x: np.ndarray, spin: str, k: float, lambda_soft: float) -> dict:
    omega = np.sqrt(k)
    v_ext = 0.5 * omega**2 * x**2
    result = {
        "system_id": system_id,
        "spin": spin,
        "k": k,
        "lambda": lambda_soft,
        "N_grid": x.size,
        "L": float(x[-1] - x[0]),
    }
    if spin == "u":
        density = ho_density(x, v_ext, occupancy=[0])
        v_H = v_H_from_density(x, density, lambda_soft=lambda_soft)
        v_xc = -v_H
        v_hxc = v_H + v_xc
        N_el = 1
    elif spin == "ud":
        density = 2.0 * ho_density(x, v_ext, occupancy=[0])
        v_H = v_H_from_density(x, density, lambda_soft=lambda_soft)
        v_x = -0.5 * v_H
        v_c = np.zeros_like(v_H)
        v_xc = v_x + v_c
        v_hxc = v_H + v_xc
        N_el = 2
    elif spin == "uu":
        density = ho_density(x, v_ext, occupancy=[0, 1])
        v_H = np.zeros_like(density)
        v_xc = np.zeros_like(density)
        v_hxc = np.zeros_like(density)
        N_el = 2
    else:
        raise ValueError(f"Unsupported spin configuration: {spin}")

    metrics = compute_nearsightedness_metrics(
        x,
        density,
        v_hxc=v_hxc,
        v_H=v_H,
        v_xc=v_xc,
        construction="direct" if spin == "uu" else "hartree_xc",
    )
    diagnostics = metrics.diagnostics
    row = {
        **result,
        "N_el": N_el,
        "dx": diagnostics["dx"],
        "method": "analytic",
        "mask_frac": diagnostics["mask_fraction"],
        "M_ratio": metrics.M_ratio,
        "M_residual": metrics.M_residual,
        "R2": diagnostics["R2"],
        "a": diagnostics["a"],
        "b": diagnostics["b"],
        "max_vH_centered": diagnostics.get("max_v_H_centered", float("nan")),
        "max_vxc_centered": diagnostics.get("max_v_xc_centered", float("nan")),
        "max_vhxc_centered": diagnostics.get("max_v_hxc_centered", float("nan")),
        "decomp_error": diagnostics.get("decomposition_error", float("nan")),
    }
    return row, {
        "x": x,
        "n": density,
        "v_H": v_H,
        "v_xc": v_xc,
        "v_hxc": v_hxc,
        "v_ext": v_ext,
    }


def main() -> int:
    args = parse_args()
    rows: List[dict] = []
    npz_dir = args.npz_dir
    npz_dir.mkdir(parents=True, exist_ok=True)
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    timestamp = datetime.utcnow().isoformat()
    for grid_points in args.grid:
        grid = uniform_grid(-0.5 * args.length, 0.5 * args.length, grid_points)
        for spin in args.spins:
            for k in args.k:
                for lambda_soft in args.lambda_soft:
                    system_id = f"{spin}_k{k}_lambda{lambda_soft}_N{grid_points}"
                    row, arrays = compute_system(system_id, grid.x, spin, k, lambda_soft)
                    row["git_hash"] = git_hash
                    row["timestamp"] = timestamp
                    rows.append(row)
                    np.savez(npz_dir / f"{system_id}.npz", **arrays)
    header = [
        "system_id",
        "spin",
        "N_el",
        "k",
        "lambda",
        "N_grid",
        "L",
        "dx",
        "method",
        "mask_frac",
        "M_ratio",
        "M_residual",
        "R2",
        "a",
        "b",
        "max_vH_centered",
        "max_vxc_centered",
        "max_vhxc_centered",
        "decomp_error",
        "git_hash",
        "timestamp",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
