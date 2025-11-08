import numpy as np

from nearsighted.hartree import v_H_from_density
from nearsighted.grids import central_difference


def solve_poisson_fd(x, n):
    dx = x[1] - x[0]
    npts = x.size
    A = np.zeros((npts, npts))
    rhs = np.zeros(npts)
    # Dirichlet zero boundary
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, npts - 1):
        A[i, i - 1] = 1.0 / dx**2
        A[i, i] = -2.0 / dx**2
        A[i, i + 1] = 1.0 / dx**2
        rhs[i] = -4.0 * np.pi * n[i]
    return np.linalg.solve(A, rhs)


def test_hartree_matches_poisson_solution():
    x = np.linspace(-5.0, 5.0, 101)
    n = np.exp(-x**2)
    v_direct = v_H_from_density(x, n, lambda_soft=1.0)
    v_poisson = solve_poisson_fd(x, n)
    v_direct_centered = v_direct - np.mean(v_direct)
    v_poisson_centered = v_poisson - np.mean(v_poisson)
    np.testing.assert_allclose(v_direct_centered, v_poisson_centered, atol=1e-2)


def test_hartree_symmetry():
    x = np.linspace(-4.0, 4.0, 81)
    n = np.exp(-x**2)
    v_H = v_H_from_density(x, n, lambda_soft=0.5)
    np.testing.assert_allclose(v_H, v_H[::-1], atol=1e-12)


def test_hartree_derivative_behaviour():
    x = np.linspace(-3.0, 3.0, 61)
    n = np.exp(-x**2)
    v_H = v_H_from_density(x, n, lambda_soft=1.0)
    dv = central_difference(v_H, x[1] - x[0])
    assert np.isclose(dv[x.size // 2], 0.0, atol=1e-8)
