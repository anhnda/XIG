"""Deterministic checks of the SPI small-noise correction (fast; the Monte-Carlo check
of the full claim is experiments/verify_spi_expansion.py)."""

import torch

from pathattr import theory as TH


def test_ig_exact_is_complete():
    F = TH.toy_mlp()
    ig = TH.ig_exact(F, 5)
    D = float(F(torch.ones(5, dtype=TH.DT)) - F(torch.zeros(5, dtype=TH.DT)))
    assert abs(float(ig.sum()) - D) < 1e-10


def test_correction_sums_to_zero():
    for F in (TH.toy_mlp(), TH.toy_saturating()):
        C = TH.spi_correction(F, 5, alpha=10.0)
        assert abs(float(C.sum())) < 1e-10 * (1 + float(C.abs().max()))
        assert float(C.abs().max()) > 1e-6            # non-trivial for non-multilinear F


def test_correction_vanishes_for_multilinear():
    C = TH.spi_correction(TH.toy_multilinear(), 5, alpha=10.0)
    assert float(C.abs().max()) < 1e-12


def test_correction_scales_as_one_over_alpha_plus_one():
    F = TH.toy_mlp()
    c1 = TH.spi_correction(F, 5, alpha=9.0)
    c2 = TH.spi_correction(F, 5, alpha=99.0)
    assert torch.allclose(c1 * 10.0, c2 * 100.0, atol=1e-12)


def test_polygon_integral_straight_equals_ig_and_complete():
    F = TH.toy_mlp()
    d, N = 5, 16
    u, wq = TH.gauss_legendre(8)
    gF = TH.vmap(TH.grad(F))
    t = torch.linspace(0, 1, N + 1, dtype=TH.DT)
    S = t.expand(1, d, N + 1).clone()
    a = TH._polygon_integral(gF, S, u, wq)[0]
    assert torch.allclose(a, TH.ig_exact(F, d), atol=1e-9)
    mc = TH.spi_monte_carlo(F, d, alpha=3.0, n_paths=64, N=N, Q=8,
                            gen=torch.Generator().manual_seed(0))
    assert abs(mc["residual"]) < 1e-9                     # every path is complete


def test_third_derivative_symmetric():
    F = TH.toy_mlp()
    T = TH.third_derivatives(F, 0.3 * torch.ones(5, dtype=TH.DT))
    assert torch.allclose(T, T.permute(1, 0, 2), atol=1e-10)
    assert torch.allclose(T, T.permute(0, 2, 1), atol=1e-10)
