import numpy as np

from nearsighted.ks_extract import (
    InverseKSResult,
    build_v_hxc_from_components,
    decomposition_check,
    inverse_ks_two_orbital,
    singlet_exchange_from_hartree,
    single_electron_xc_from_hartree,
    solve_schrodinger_1d,
)


def test_build_v_hxc_sum():
    v_H = np.array([0.0, 0.5, 1.0])
    v_xc = np.array([1.0, -0.5, 0.1])
    combined = build_v_hxc_from_components(v_H, v_xc)
    np.testing.assert_allclose(combined, v_H + v_xc)


def test_single_electron_relation():
    x = np.linspace(-1, 1, 5)
    v_H = x**2
    v_xc = single_electron_xc_from_hartree(v_H)
    np.testing.assert_allclose(v_H + v_xc, 0.0, atol=1e-12)


def test_singlet_exchange_relation():
    x = np.linspace(-1, 1, 5)
    v_H = x**2
    v_x = singlet_exchange_from_hartree(v_H)
    np.testing.assert_allclose(v_H + 2 * v_x, 0.0, atol=1e-12)


def test_inverse_ks_two_orbital_non_interacting():
    x = np.linspace(-4.0, 4.0, 81)
    omega = 1.0
    v_ext = 0.5 * omega**2 * x**2
    energies, orbitals = solve_schrodinger_1d(x, v_ext)
    occ = orbitals[:2]
    density = np.sum(np.abs(occ) ** 2, axis=0)
    result = inverse_ks_two_orbital(x, density, v_init=v_ext, alpha=0.05, tol=1e-10)
    assert isinstance(result, InverseKSResult)
    assert result.converged
    np.testing.assert_allclose(result.potential - np.mean(result.potential), v_ext - np.mean(v_ext), atol=1e-6)


def test_decomposition_check_zero():
    x = np.linspace(-1, 1, 5)
    v_ext = x**2
    v_H = x
    v_xc = -x
    v_s = v_ext + v_H + v_xc
    err = decomposition_check(v_s, v_ext, v_H, v_xc)
    assert np.isclose(err, 0.0)
