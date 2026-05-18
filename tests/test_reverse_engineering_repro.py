import types

import numpy as np

import iDEA
import iDEA.methods.interacting
import iDEA.methods.non_interacting
import iDEA.observables
import iDEA.reverse_engineering


def _harmonic_system(electrons):
    x = np.linspace(-6, 6, 40)
    v_ext = 0.5 * 0.25**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    return iDEA.system.System(x, v_ext, v_int, electrons=electrons)


def _target_density(s):
    state = iDEA.methods.interacting.solve(s, k=0)
    return iDEA.observables.density(s, state=state)


def _assert_states_equal(ref_state, opt_state):
    assert np.array_equal(ref_state.up.energies, opt_state.up.energies)
    assert np.array_equal(ref_state.up.orbitals, opt_state.up.orbitals)
    assert np.array_equal(ref_state.up.occupations, opt_state.up.occupations)
    assert np.array_equal(ref_state.up.occupied, opt_state.up.occupied)
    assert np.array_equal(ref_state.down.energies, opt_state.down.energies)
    assert np.array_equal(ref_state.down.orbitals, opt_state.down.orbitals)
    assert np.array_equal(ref_state.down.occupations, opt_state.down.occupations)
    assert np.array_equal(ref_state.down.occupied, opt_state.down.occupied)


def _assert_reverse_matches_reference(s, target_n, **kwargs):
    ref_kwargs = dict(kwargs)
    opt_kwargs = dict(kwargs)
    if "v_guess" in kwargs:
        ref_kwargs["v_guess"] = kwargs["v_guess"].copy()
        opt_kwargs["v_guess"] = kwargs["v_guess"].copy()

    ref = iDEA.reverse_engineering._reverse_reference(
        s, target_n, iDEA.methods.non_interacting, silent=True, **ref_kwargs
    )
    opt = iDEA.reverse_engineering.reverse(
        s, target_n, iDEA.methods.non_interacting, silent=True, **opt_kwargs
    )

    assert np.array_equal(ref.v_ext, opt.v_ext)

    ref_state = iDEA.methods.non_interacting.solve(ref, k=0, silent=True)
    opt_state = iDEA.methods.non_interacting.solve(opt, k=0, silent=True)
    _assert_states_equal(ref_state, opt_state)

    ref_n = iDEA.observables.density(ref, state=ref_state)
    opt_n = iDEA.observables.density(opt, state=opt_state)
    assert np.array_equal(ref_n, opt_n)

    ref_error = np.sum(abs(ref_n - target_n)) * s.dx
    opt_error = np.sum(abs(opt_n - target_n)) * s.dx
    assert np.array_equal(np.array(ref_error), np.array(opt_error))


def test_reverse_non_interacting_harmonic_uu_matches_reference():
    s = _harmonic_system("uu")
    target_n = _target_density(s)
    _assert_reverse_matches_reference(s, target_n, mu=0.8, pe=0.1, tol=1e-4)


def test_reverse_non_interacting_harmonic_ud_matches_reference():
    s = _harmonic_system("ud")
    target_n = _target_density(s)
    _assert_reverse_matches_reference(s, target_n, mu=0.8, pe=0.1, tol=1e-4)


def test_reverse_non_interacting_with_v_guess_matches_reference():
    s = _harmonic_system("uu")
    target_n = _target_density(s)
    v_guess = s.v_ext + 0.05 * np.cos(s.x)
    _assert_reverse_matches_reference(
        s, target_n, v_guess=v_guess, mu=0.8, pe=0.1, tol=1e-4
    )


def test_reverse_unsupported_kwargs_use_reference_path(monkeypatch):
    sentinel = object()

    def fake_reference(*args, **kwargs):
        return sentinel

    def fail_fast_path(*args, **kwargs):
        raise AssertionError("optimized path should not be used")

    monkeypatch.setattr(iDEA.reverse_engineering, "_reverse_reference", fake_reference)
    monkeypatch.setattr(
        iDEA.reverse_engineering, "_reverse_non_interacting", fail_fast_path
    )

    result = iDEA.reverse_engineering.reverse(
        _harmonic_system("uu"),
        np.zeros(40),
        iDEA.methods.non_interacting,
        silent=True,
        mixing=0.5,
    )

    assert result is sentinel


def test_reverse_custom_method_uses_reference_path(monkeypatch):
    sentinel = object()
    custom_method = types.SimpleNamespace()

    def fake_reference(*args, **kwargs):
        return sentinel

    def fail_fast_path(*args, **kwargs):
        raise AssertionError("optimized path should not be used")

    monkeypatch.setattr(iDEA.reverse_engineering, "_reverse_reference", fake_reference)
    monkeypatch.setattr(
        iDEA.reverse_engineering, "_reverse_non_interacting", fail_fast_path
    )

    result = iDEA.reverse_engineering.reverse(
        _harmonic_system("uu"),
        np.zeros(40),
        custom_method,
        silent=True,
    )

    assert result is sentinel
