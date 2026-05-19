import numpy as np

import iDEA
import iDEA.methods.non_interacting


def test_non_interacting_solve_returns_full_single_particle_spectrum():
    x = np.linspace(-6, 6, 24)
    v_ext = 0.5 * 0.25**2 * x**2
    v_int = iDEA.interactions.softened_interaction(x)
    s = iDEA.system.System(x, v_ext, v_int, electrons="uu")

    state = iDEA.methods.non_interacting.solve(s, k=0, silent=True)

    assert state.up.energies.shape == s.x.shape
    assert state.down.energies.shape == s.x.shape
    assert state.up.orbitals.shape == (s.x.shape[0], s.x.shape[0])
    assert state.down.orbitals.shape == (s.x.shape[0], s.x.shape[0])
    assert state.up.occupations.shape == s.x.shape
    assert state.down.occupations.shape == s.x.shape
