"""Deterministic checks of the SPI small-noise correction (fast; the Monte-Carlo check
of the full claim is experiments/verify_spi_expansion.py)."""

import pytest
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
    # every path is complete up to the per-segment quadrature error (GL is exact only for
    # polynomials; at α = 3 a segment can jump across most of [0, 1]): Q = 8 → ~1e-7
    res = {}
    for Q in (8, 16):
        mc = TH.spi_monte_carlo(F, d, alpha=3.0, n_paths=64, N=N, Q=Q,
                                gen=torch.Generator().manual_seed(0))
        res[Q] = abs(mc["residual"])
    assert res[8] < 1e-5 and res[16] < 1e-8 and res[16] < res[8] / 100


def test_shapley_exact_matches_permutations_and_multilinear_ig():
    import itertools
    F = TH.toy_mlp(4)
    phi = torch.zeros(4, dtype=TH.DT)
    perms = list(itertools.permutations(range(4)))
    for p in perms:
        z = torch.zeros(4, dtype=TH.DT)
        prev = float(F(z))
        for i in p:
            z[i] = 1.0
            cur = float(F(z))
            phi[i] += cur - prev
            prev = cur
    assert torch.allclose(TH.shapley_exact(F, 4), phi / len(perms), atol=1e-12)
    G = TH.toy_multilinear(6)
    assert torch.allclose(TH.shapley_exact(G, 6), TH.ig_exact(G, 6), atol=1e-10)


def test_ridge_correction_needs_unequal_weights():
    # F = g(w·z): T_ij = g''' w_i w_j (w_j − w_i) → C = 0 for equal weights
    assert float(TH.spi_correction(TH.toy_saturating(5, spread=0.0), 5, 10.0).abs().max()) < 1e-12
    assert float(TH.spi_correction(TH.toy_saturating(5, spread=1.0), 5, 10.0).abs().max()) > 1e-6
    for name, mk in TH.TOYS.items():
        C = TH.spi_correction(mk(6, 1), 6, 3.0)
        assert abs(float(C.sum())) < 1e-10, name


def test_batched_equals_single():
    for fam in ("mlp", "satint", "multilinear"):
        fn, P = TH.stack_params(fam, 5, [0, 3, 7])
        ig, C, sh = TH.ig_batch(fn, P, 5), TH.correction_batch(fn, P, 5, 4.0), TH.shapley_batch(fn, P, 5)
        for k, s in enumerate([0, 3, 7]):
            F = TH.TOYS[fam](5, s)
            assert torch.allclose(ig[k], TH.ig_exact(F, 5), atol=1e-10), fam
            assert torch.allclose(C[k], TH.spi_correction(F, 5, 4.0), atol=1e-10), fam
            assert torch.allclose(sh[k], TH.shapley_exact(F, 5), atol=1e-10), fam


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dp_cdf_cuda_moments_and_monotone():
    TH.set_device("cuda")
    try:
        alpha, N = 3.0, 8
        F = TH.dp_cdf(200000, N, alpha, torch.Generator().manual_seed(0)).cpu()
        assert float((F[:, 1:] - F[:, :-1]).min()) >= 0 and float(F[:, -1].min()) == 1.0
        t = torch.arange(N + 1, dtype=torch.float64) / N
        assert float((F.mean(0) - t).abs().max()) < 3e-3
        var = t * (1 - t) / (alpha + 1)
        assert float((F.var(0) - var).abs().max()) < 3e-3
    finally:
        TH.set_device("cpu")


def test_third_derivative_symmetric():
    F = TH.toy_mlp()
    T = TH.third_derivatives(F, 0.3 * torch.ones(5, dtype=TH.DT))
    assert torch.allclose(T, T.permute(1, 0, 2), atol=1e-10)
    assert torch.allclose(T, T.permute(0, 2, 1), atol=1e-10)
