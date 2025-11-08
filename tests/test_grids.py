import numpy as np
import pytest

from nearsighted.grids import (
    Grid,
    apply_mask,
    central_difference,
    remove_gauge,
    strip_boundary_mask,
    uniform_grid,
)


def test_uniform_grid_spacing():
    grid = uniform_grid(-1.0, 1.0, 5)
    assert isinstance(grid, Grid)
    assert np.isclose(grid.dx, 0.5)
    np.testing.assert_allclose(grid.x, np.linspace(-1.0, 1.0, 5))


def test_central_difference_polynomial():
    grid = uniform_grid(-1, 1, 11)
    x = grid.x
    f = x**3
    df_exact = 3 * x**2
    df_num = central_difference(f, grid.dx)
    np.testing.assert_allclose(df_num[1:-1], df_exact[1:-1], atol=1e-3)


def test_strip_boundary_mask_fraction():
    grid = uniform_grid(-1, 1, 100)
    mask = strip_boundary_mask(grid.x, fraction=0.05)
    assert mask.sum() == 90
    with pytest.raises(ValueError):
        strip_boundary_mask(grid.x, fraction=0.6)


def test_apply_mask_shape_mismatch():
    grid = uniform_grid(-1, 1, 10)
    mask = strip_boundary_mask(grid.x, fraction=0.2)
    with pytest.raises(ValueError):
        apply_mask(np.ones(9), mask=mask)


def test_remove_gauge_weighted():
    values = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 1.0, 2.0])
    centered = remove_gauge(values, weights)
    assert np.isclose(centered.sum(), 0.0)
    # Weighted mean equals 1.25
    np.testing.assert_allclose(centered, np.array([-1.25, -0.25, 1.75]))
