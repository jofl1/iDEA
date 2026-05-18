import contextlib

import numpy as np
import pytest

import iDEA
import iDEA.methods.hartree_fock
import iDEA.methods.non_interacting
import iDEA.observables


def _harmonic_system(electrons, n_points=32):
    x = np.linspace(-6, 6, n_points)
    v_ext = 0.5 * 0.25**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons=electrons)


def _synthetic_state(n_points=12):
    state = iDEA.state.SingleBodyState()
    rng = np.random.default_rng(1)
    state.up.orbitals = rng.normal(size=(n_points, n_points))
    state.down.orbitals = rng.normal(size=(n_points, n_points))
    state.up.energies = np.arange(n_points, dtype=float)
    state.down.energies = np.arange(n_points, dtype=float)
    state.up.occupations = np.zeros(n_points)
    state.down.occupations = np.zeros(n_points)
    state.up.occupations[[0, 3, 7]] = [1.0, 0.25, 0.5]
    state.down.occupations[[1, 4]] = [0.75, 0.125]
    state.up.occupied = np.nonzero(state.up.occupations)[0]
    state.down.occupied = np.nonzero(state.down.occupations)[0]
    return state


def _assert_equal_arrays_or_tuples(left, right):
    if isinstance(left, tuple):
        assert isinstance(right, tuple)
        assert len(left) == len(right)
        for left_array, right_array in zip(left, right):
            assert np.array_equal(left_array, right_array)
    else:
        assert np.array_equal(left, right)


def _assert_density_kernels_match_reference(s, state):
    for return_spins in (False, True):
        ref_n = iDEA.observables._density_single_body_reference(
            s, state, return_spins=return_spins
        )
        opt_n = iDEA.observables.density(s, state=state, return_spins=return_spins)
        _assert_equal_arrays_or_tuples(ref_n, opt_n)

        ref_p = iDEA.observables._density_matrix_single_body_reference(
            s, state, return_spins=return_spins
        )
        opt_p = iDEA.observables.density_matrix(
            s, state=state, return_spins=return_spins
        )
        _assert_equal_arrays_or_tuples(ref_p, opt_p)


def _assert_single_body_states_equal(ref_state, opt_state):
    assert np.array_equal(ref_state.up.energies, opt_state.up.energies)
    assert np.array_equal(ref_state.up.orbitals, opt_state.up.orbitals)
    assert np.array_equal(ref_state.up.occupations, opt_state.up.occupations)
    assert np.array_equal(ref_state.up.occupied, opt_state.up.occupied)
    assert np.array_equal(ref_state.down.energies, opt_state.down.energies)
    assert np.array_equal(ref_state.down.orbitals, opt_state.down.orbitals)
    assert np.array_equal(ref_state.down.occupations, opt_state.down.occupations)
    assert np.array_equal(ref_state.down.occupied, opt_state.down.occupied)


@contextlib.contextmanager
def _reference_density_kernels():
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


@pytest.mark.parametrize("electrons", ["u", "uu", "ud"])
def test_non_interacting_static_density_kernels_match_reference(electrons):
    s = _harmonic_system(electrons)
    state = iDEA.methods.non_interacting.solve(s, k=0, silent=True)

    _assert_density_kernels_match_reference(s, state)


@pytest.mark.parametrize("electrons", ["uu", "ud"])
def test_hartree_fock_static_density_kernels_match_reference(electrons):
    s = _harmonic_system(electrons)
    state = iDEA.methods.hartree_fock.solve(s, k=0, tol=1e-7, silent=True)

    _assert_density_kernels_match_reference(s, state)


def test_fractional_occupation_density_kernels_match_reference():
    s = _harmonic_system("u", n_points=12)
    state = _synthetic_state(n_points=12)

    _assert_density_kernels_match_reference(s, state)


@pytest.mark.parametrize("electrons", ["uu", "ud"])
def test_hartree_fock_solve_matches_reference_density_kernels(electrons):
    s = _harmonic_system(electrons)

    with _reference_density_kernels():
        ref_state = iDEA.methods.hartree_fock.solve(s, k=0, tol=1e-7, silent=True)
        ref_energy = iDEA.methods.hartree_fock.total_energy(s, ref_state)
        ref_density = iDEA.observables.density(s, state=ref_state)
        ref_density_matrix = iDEA.observables.density_matrix(s, state=ref_state)

    opt_state = iDEA.methods.hartree_fock.solve(s, k=0, tol=1e-7, silent=True)
    opt_energy = iDEA.methods.hartree_fock.total_energy(s, opt_state)
    opt_density = iDEA.observables.density(s, state=opt_state)
    opt_density_matrix = iDEA.observables.density_matrix(s, state=opt_state)

    _assert_single_body_states_equal(ref_state, opt_state)
    assert np.array_equal(np.array(ref_energy), np.array(opt_energy))
    assert np.array_equal(ref_density, opt_density)
    assert np.array_equal(ref_density_matrix, opt_density_matrix)


@pytest.mark.parametrize("kernel", ["density", "density_matrix"])
def test_complex_static_state_uses_reference_kernel(monkeypatch, kernel):
    s = _harmonic_system("u", n_points=8)
    state = _synthetic_state(n_points=8)
    state.up.orbitals = state.up.orbitals.astype(complex)

    sentinel = object()
    reference_name = f"_{kernel}_single_body_reference"
    optimized_name = f"_{kernel}_single_body_optimized"

    monkeypatch.setattr(
        iDEA.observables,
        reference_name,
        lambda *args, **kwargs: sentinel,
    )
    monkeypatch.setattr(
        iDEA.observables,
        optimized_name,
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("optimized path should not be used")
        ),
    )

    result = getattr(iDEA.observables, kernel)(s, state=state)

    assert result is sentinel


@pytest.mark.parametrize("kernel", ["density", "density_matrix"])
def test_nonfinite_zero_occupation_orbital_uses_reference_kernel(
    monkeypatch, kernel
):
    s = _harmonic_system("u", n_points=8)
    state = _synthetic_state(n_points=8)
    zero_up_index = np.nonzero(state.up.occupations == 0)[0][0]
    state.up.orbitals[0, zero_up_index] = np.nan

    sentinel = object()
    reference_name = f"_{kernel}_single_body_reference"
    optimized_name = f"_{kernel}_single_body_optimized"

    monkeypatch.setattr(
        iDEA.observables,
        reference_name,
        lambda *args, **kwargs: sentinel,
    )
    monkeypatch.setattr(
        iDEA.observables,
        optimized_name,
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("optimized path should not be used")
        ),
    )

    result = getattr(iDEA.observables, kernel)(s, state=state)

    assert result is sentinel
