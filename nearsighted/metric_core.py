"""Core implementation of the Week-6 nearsightedness metric."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .grids import apply_mask, central_difference, remove_gauge, strip_boundary_mask

EPSILON = 1e-10


@dataclass
class MetricResult:
    """Container for nearsightedness metrics and diagnostics."""

    M_ratio: float
    M_residual: float
    diagnostics: Dict[str, np.ndarray | float | int | str]


def _construct_v_hxc(
    *,
    v_hxc: Optional[np.ndarray],
    v_H: Optional[np.ndarray],
    v_xc: Optional[np.ndarray],
    v_ext: Optional[np.ndarray],
    v_s: Optional[np.ndarray],
    construction: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    sources: Dict[str, np.ndarray] = {}
    if construction == "direct":
        if v_hxc is None:
            raise ValueError("v_hxc must be provided for direct construction")
        sources["v_hxc_raw"] = np.asarray(v_hxc, dtype=float)
        return sources["v_hxc_raw"], sources
    if construction == "hartree_xc":
        if v_H is None or v_xc is None:
            raise ValueError("v_H and v_xc required for hartree_xc construction")
        v_hxc_arr = np.asarray(v_H, dtype=float) + np.asarray(v_xc, dtype=float)
        sources["v_H"] = np.asarray(v_H, dtype=float)
        sources["v_xc"] = np.asarray(v_xc, dtype=float)
        sources["v_hxc_raw"] = v_hxc_arr
        return v_hxc_arr, sources
    if construction == "sanity_vs_minus_vext":
        if v_s is None or v_ext is None:
            raise ValueError("v_s and v_ext required for sanity construction")
        sources["v_s"] = np.asarray(v_s, dtype=float)
        sources["v_ext"] = np.asarray(v_ext, dtype=float)
        v_hxc_arr = sources["v_s"] - sources["v_ext"]
        sources["v_hxc_raw"] = v_hxc_arr
        return v_hxc_arr, sources
    raise ValueError(f"Unknown construction '{construction}'")


def compute_nearsightedness_metrics(
    x: np.ndarray,
    n: np.ndarray,
    *,
    v_hxc: Optional[np.ndarray] = None,
    v_H: Optional[np.ndarray] = None,
    v_xc: Optional[np.ndarray] = None,
    v_ext: Optional[np.ndarray] = None,
    v_s: Optional[np.ndarray] = None,
    construction: str = "direct",
    mask_fraction: float = 0.05,
) -> MetricResult:
    """Compute Week-6 nearsightedness metrics on a uniform grid."""

    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    if x.ndim != 1 or n.ndim != 1 or x.shape != n.shape:
        raise ValueError("x and n must be 1D arrays of equal length")
    if x.size < 3:
        raise ValueError("grid must contain at least three points")
    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx):
        raise ValueError("x must be uniformly spaced")

    v_hxc_arr, sources = _construct_v_hxc(
        v_hxc=v_hxc,
        v_H=v_H,
        v_xc=v_xc,
        v_ext=v_ext,
        v_s=v_s,
        construction=construction,
    )

    mask = strip_boundary_mask(x, fraction=mask_fraction)
    weights = n / np.max(n)

    dv_hxc = central_difference(v_hxc_arr, dx)
    dn = central_difference(n, dx)

    dv_hxc_masked, dn_masked, weights_masked, n_masked = apply_mask(
        dv_hxc, dn, weights, n, mask=mask
    )
    x_masked = x[mask]

    numerator = np.trapz(weights_masked * dv_hxc_masked**2, x_masked)
    denominator = np.trapz(weights_masked * dn_masked**2, x_masked) + EPSILON
    M_ratio = float(numerator / denominator)

    y = dv_hxc_masked
    design = dn_masked
    W = weights_masked
    W_sum = float(np.sum(W))
    if W_sum == 0:
        raise ValueError("Density weights vanish everywhere")
    sqrt_W = np.sqrt(W)
    A = np.column_stack([design, np.ones_like(design)])
    Aw = sqrt_W[:, None] * A
    yw = sqrt_W * y
    beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    a = float(beta[0])
    b = float(beta[1])
    fit = a * design + b
    residuals = y - fit
    sse = float(np.sum(W * residuals**2))
    mean_y = float(np.sum(W * y) / W_sum)
    sst = float(np.sum(W * (y - mean_y) ** 2)) + EPSILON
    R2 = 1.0 - sse / sst
    M_residual = float(np.sqrt(sse / W_sum))

    diagnostics: Dict[str, np.ndarray | float | int | str] = {
        "dx": dx,
        "mask": mask,
        "mask_fraction": mask_fraction,
        "weights": weights,
        "weights_masked": weights_masked,
        "a": a,
        "b": b,
        "R2": R2,
        "dv_hxc": dv_hxc,
        "dn": dn,
        "n": n,
        "M_ratio": M_ratio,
        "M_residual": M_residual,
    }
    diagnostics.update(sources)

    for key, array in ("v_H", v_H), ("v_xc", v_xc), ("v_hxc", v_hxc_arr):
        if array is None:
            continue
        centered = remove_gauge(np.asarray(array), weights=None)
        diagnostics[f"max_{key}_centered"] = float(np.max(np.abs(centered[mask])))
    diagnostics["decomposition_error"] = float("nan")

    return MetricResult(M_ratio=M_ratio, M_residual=M_residual, diagnostics=diagnostics)
