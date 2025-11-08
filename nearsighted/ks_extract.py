"""Utilities for constructing and validating Kohn–Sham potentials."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .grids import remove_gauge


def build_v_hxc_from_components(v_H: np.ndarray, v_xc: np.ndarray) -> np.ndarray:
    """Combine Hartree and exchange–correlation potentials."""

    v_H = np.asarray(v_H, dtype=float)
    v_xc = np.asarray(v_xc, dtype=float)
    if v_H.shape != v_xc.shape:
        raise ValueError("v_H and v_xc must have the same shape")
    return v_H + v_xc


def single_electron_xc_from_hartree(v_H: np.ndarray) -> np.ndarray:
    """Exact single-electron relation v_xc = -v_H (up to a constant)."""

    return -np.asarray(v_H, dtype=float)


def singlet_exchange_from_hartree(v_H: np.ndarray) -> np.ndarray:
    """Exact singlet relation v_x = -0.5 * v_H (up to a constant)."""

    return -0.5 * np.asarray(v_H, dtype=float)


def decomposition_check(
    v_s: np.ndarray, v_ext: np.ndarray, v_H: np.ndarray, v_xc: np.ndarray
) -> float:
    """Return infinity norm of v_s - (v_ext + v_H + v_xc) after gauge removal."""

    v_s = np.asarray(v_s, dtype=float)
    v_ext = np.asarray(v_ext, dtype=float)
    v_H = np.asarray(v_H, dtype=float)
    v_xc = np.asarray(v_xc, dtype=float)
    if not (v_s.shape == v_ext.shape == v_H.shape == v_xc.shape):
        raise ValueError("All potentials must share the same shape")
    residual = v_s - (v_ext + v_H + v_xc)
    residual_centered = remove_gauge(residual)
    return float(np.max(np.abs(residual_centered)))


def _kinetic_matrix(npts: int, dx: float) -> np.ndarray:
    diag = np.full(npts, 1.0 / dx**2)
    off = np.full(npts - 1, -0.5 / dx**2)
    T = np.diag(diag * (-2.0 * 0.5))
    T += np.diag(off, 1)
    T += np.diag(off, -1)
    return T


def solve_schrodinger_1d(x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 1D KS equation for given potential (Dirichlet boundaries)."""

    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    if x.shape != v.shape:
        raise ValueError("x and v must share shape")
    dx = x[1] - x[0]
    if not np.allclose(np.diff(x), dx):
        raise ValueError("Grid must be uniform")
    npts = x.size
    kinetic = -0.5 * (
        np.diag(np.full(npts, -2.0 / dx**2))
        + np.diag(np.full(npts - 1, 1.0 / dx**2), 1)
        + np.diag(np.full(npts - 1, 1.0 / dx**2), -1)
    )
    hamiltonian = kinetic + np.diag(v)
    energies, orbitals = np.linalg.eigh(hamiltonian)
    dx = float(dx)
    for i in range(orbitals.shape[1]):
        norm = np.sqrt(np.sum(np.abs(orbitals[:, i]) ** 2) * dx)
        orbitals[:, i] /= norm
    return energies, orbitals.T


@dataclass
class InverseKSResult:
    potential: np.ndarray
    orbitals: np.ndarray
    energies: np.ndarray
    iterations: int
    converged: bool


def inverse_ks_two_orbital(
    x: np.ndarray,
    density: np.ndarray,
    *,
    v_init: np.ndarray,
    num_orbitals: int = 2,
    max_iter: int = 200,
    alpha: float = 0.1,
    tol: float = 1e-8,
) -> InverseKSResult:
    """Inverse KS solver for systems with two occupied orbitals."""

    x = np.asarray(x, dtype=float)
    density = np.asarray(density, dtype=float)
    v = np.asarray(v_init, dtype=float).copy()
    if x.shape != density.shape or v.shape != density.shape:
        raise ValueError("All arrays must share shape")
    dx = x[1] - x[0]
    if not np.allclose(np.diff(x), dx):
        raise ValueError("Grid must be uniform")
    for iteration in range(1, max_iter + 1):
        energies, orbitals = solve_schrodinger_1d(x, v)
        occ_orbitals = orbitals[:num_orbitals]
        n_current = np.sum(np.abs(occ_orbitals) ** 2, axis=0)
        error = n_current - density
        if np.max(np.abs(error)) < tol:
            return InverseKSResult(v, occ_orbitals, energies[:num_orbitals], iteration, True)
        correction = alpha * error / (density + 1e-12)
        correction = correction - np.mean(correction)
        v -= correction
    return InverseKSResult(v, occ_orbitals, energies[:num_orbitals], max_iter, False)
