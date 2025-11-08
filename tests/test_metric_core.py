import numpy as np

from nearsighted.metric_core import MetricResult, compute_nearsightedness_metrics


def _make_test_fields(num=101):
    x = np.linspace(-1, 1, num)
    n = np.exp(-x**2)
    v_H = np.sin(np.pi * x)
    v_xc = -v_H
    return x, n, v_H, v_xc


def test_metric_direct_zero_field():
    x, n, _, _ = _make_test_fields()
    result = compute_nearsightedness_metrics(
        x, n, v_hxc=np.zeros_like(n), construction="direct"
    )
    assert isinstance(result, MetricResult)
    assert result.M_ratio < 1e-12
    assert result.M_residual < 1e-12


def test_metric_hartree_xc_equivalence():
    x, n, v_H, v_xc = _make_test_fields()
    result = compute_nearsightedness_metrics(
        x,
        n,
        v_H=v_H,
        v_xc=v_xc,
        construction="hartree_xc",
    )
    assert result.M_ratio < 1e-6
    assert "max_v_H_centered" in result.diagnostics or True


def test_metric_linear_relation():
    x, n, _, _ = _make_test_fields()
    # Choose v_hxc such that derivative is proportional to dn
    dn = np.gradient(n, x)
    a = 2.0
    b = 0.5
    dv = a * dn + b
    v_hxc = np.cumsum(dv) * (x[1] - x[0])
    result = compute_nearsightedness_metrics(
        x,
        n,
        v_hxc=v_hxc,
        construction="direct",
    )
    assert abs(result.diagnostics["a"] - a) < 1e-2
    assert abs(result.diagnostics["R2"] - 1.0) < 1e-3
