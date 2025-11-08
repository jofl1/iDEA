#!/usr/bin/env python3
"""CLI to compute nearsightedness metrics from stored arrays."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from datetime import datetime
from typing import Dict, Tuple

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nearsighted.metric_core import compute_nearsightedness_metrics


def _load_arrays(path: pathlib.Path) -> Dict[str, np.ndarray]:
    if path.suffix == ".npz":
        data = np.load(path)
        return {key: data[key] for key in data}
    if path.suffix == ".csv":
        arr = np.genfromtxt(path, delimiter=",", names=True)
        return {name: arr[name] for name in arr.dtype.names}
    raise ValueError(f"Unsupported file format: {path.suffix}")


def parse_args(argv: Tuple[str, ...]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path, help="Input NPZ/CSV file")
    parser.add_argument("output", type=pathlib.Path, help="Output CSV file")
    parser.add_argument("--construction", choices=["direct", "hartree_xc", "sanity_vs_minus_vext"], default="direct")
    parser.add_argument("--metadata", help="JSON string with metadata fields", default="{}")
    return parser.parse_args(argv)


def main(argv: Tuple[str, ...] | None = None) -> int:
    args = parse_args(tuple(sys.argv[1:]) if argv is None else argv)
    arrays = _load_arrays(args.input)
    x = arrays["x"]
    n = arrays["n"]
    metrics = compute_nearsightedness_metrics(
        x,
        n,
        v_hxc=arrays.get("v_hxc"),
        v_H=arrays.get("v_H"),
        v_xc=arrays.get("v_xc"),
        v_ext=arrays.get("v_ext"),
        v_s=arrays.get("v_s"),
        construction=args.construction,
    )
    metadata = json.loads(args.metadata)
    diagnostics = metrics.diagnostics
    row = {
        "system_id": metadata.get("system_id", "unknown"),
        "spin": metadata.get("spin", "-"),
        "N_el": metadata.get("N_el", float(np.trapz(n, x))),
        "k": metadata.get("k", np.nan),
        "lambda": metadata.get("lambda", np.nan),
        "N_grid": x.size,
        "L": x[-1] - x[0],
        "dx": diagnostics["dx"],
        "method": metadata.get("method", "manual"),
        "mask_frac": diagnostics["mask_fraction"],
        "M_ratio": metrics.M_ratio,
        "M_residual": metrics.M_residual,
        "R2": diagnostics["R2"],
        "a": diagnostics["a"],
        "b": diagnostics["b"],
        "max_vH_centered": diagnostics.get("max_v_H_centered", np.nan),
        "max_vxc_centered": diagnostics.get("max_v_xc_centered", np.nan),
        "max_vhxc_centered": diagnostics.get("max_v_hxc_centered", np.nan),
        "decomp_error": diagnostics.get("decomposition_error", np.nan),
        "git_hash": metadata.get("git_hash", ""),
        "timestamp": datetime.utcnow().isoformat(),
    }
    header = list(row.keys())
    write_header = not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
