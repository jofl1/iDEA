import numpy as np

from nearsighted.grids import uniform_grid
from nearsighted.hartree import v_H_from_density
from nearsighted.ks_extract import decomposition_check, solve_schrodinger_1d
from nearsighted.metric_core import compute_nearsightedness_metrics


def harmonic_potential(x, k):
    omega = np.sqrt(k)
    return 0.5 * omega**2 * x**2


def ho_density(orbitals):
    return np.sum(np.abs(orbitals) ** 2, axis=0)


def test_single_electron_limit_metrics_small():
    grid = uniform_grid(-6.0, 6.0, 401)
    v_ext = harmonic_potential(grid.x, k=1.0)
    energies, orbitals = solve_schrodinger_1d(grid.x, v_ext)
    density = np.abs(orbitals[0]) ** 2
    v_H = v_H_from_density(grid.x, density, lambda_soft=20.0)
    v_xc = -v_H
    metrics = compute_nearsightedness_metrics(
        grid.x,
        density,
        v_H=v_H,
        v_xc=v_xc,
        construction="hartree_xc",
    )
    v_s = v_ext + v_H + v_xc
    assert decomposition_check(v_s, v_ext, v_H, v_xc) < 1e-9
    assert metrics.M_ratio < 1e-8
    assert metrics.M_residual < 1e-8
    v_hxc_centered = metrics.diagnostics["max_v_hxc_centered"]
    assert v_hxc_centered < 1e-9


def _singlet_system(grid_points):
    grid = uniform_grid(-8.0, 8.0, grid_points)
    v_ext = harmonic_potential(grid.x, k=0.5)
    energies, orbitals = solve_schrodinger_1d(grid.x, v_ext)
    density = 2.0 * np.abs(orbitals[0]) ** 2
    lambda_soft = 50.0
    v_H = v_H_from_density(grid.x, density, lambda_soft=lambda_soft)
    v_x = -0.5 * v_H
    v_c = np.zeros_like(v_H)
    v_xc = v_x + v_c
    metrics = compute_nearsightedness_metrics(
        grid.x,
        density,
        v_H=v_H,
        v_xc=v_xc,
        construction="hartree_xc",
    )
    return metrics, v_H, v_x, v_xc, grid, v_ext


def test_two_electron_singlet_relations():
    metrics, v_H, v_x, v_xc, grid, v_ext = _singlet_system(201)
    centered = v_x + 0.5 * v_H
    centered -= np.mean(centered)
    assert np.max(np.abs(centered)) < 1e-9
    assert metrics.M_ratio < 1e-4
    v_s = v_ext + v_H + v_xc
    assert decomposition_check(v_s, v_ext, v_H, v_xc) < 1e-9


def test_two_electron_singlet_grid_refinement():
    metrics_coarse, *_ = _singlet_system(201)
    metrics_fine, *_ = _singlet_system(251)
    delta = abs(metrics_coarse.M_ratio - metrics_fine.M_ratio)
    denom = max(metrics_coarse.M_ratio, 1e-12)
    assert delta / denom < 0.02


def test_spin_polarised_two_electron_reference():
    grid = uniform_grid(-6.0, 6.0, 401)
    v_ext = harmonic_potential(grid.x, k=1.0)
    energies, orbitals = solve_schrodinger_1d(grid.x, v_ext)
    occ = orbitals[:2]
    density = ho_density(occ)
    v_hxc = np.zeros_like(density)
    metrics = compute_nearsightedness_metrics(
        grid.x,
        density,
        v_hxc=v_hxc,
        construction="direct",
    )
    assert metrics.M_ratio < 1e-4
    assert metrics.M_residual < 1e-4
    v_s = v_ext + v_hxc
    assert decomposition_check(v_s, v_ext, np.zeros_like(v_ext), np.zeros_like(v_ext)) < 1e-9
