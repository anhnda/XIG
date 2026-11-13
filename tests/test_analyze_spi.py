"""Pure helpers of experiments.analyze_spi."""

import math

import torch

from experiments import analyze_spi as AN


def test_alpha_of():
    assert AN.alpha_of("spi:alpha=10,K=8,est=mid") == 10.0
    assert AN.alpha_of("spi:alpha=100000,K=8,est=mid") == 1e5
    assert math.isinf(AN.alpha_of("ig:N=24,est=mid"))
    assert AN.akey(math.inf) == "inf" and AN.akey(100.0) == "100"


def test_profile_stats_monotone_and_not():
    f = torch.linspace(0, 1, 25) ** 0.2                    # monotone, saturates early
    s = AN.profile_stats(f)
    assert abs(s["tv"] - 1) < 1e-12 and s["neg"] == 0.0
    assert s["t90"] < 0.7 and s["area"] > 0.7
    g = torch.tensor([0.0, 2.0, 1.0, 1.0])                # overshoot then come back
    s = AN.profile_stats(g)
    assert abs(s["tv"] - 3.0) < 1e-12 and abs(s["neg"] - 1 / 3) < 1e-12
    e = AN.profile_stats(torch.linspace(0, 1, 11, dtype=torch.float64))
    assert abs(e["ent"] - 1.0) < 1e-12 and abs(e["area"] - 0.5) < 1e-12


def test_debiased_gap_removes_seed_noise():
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(20000, generator=g, dtype=torch.float64)
    A = ref + 0.5 * torch.randn(4, 20000, generator=g, dtype=torch.float64)   # no true gap
    assert AN.debiased_gap(A, ref) < 0.05
    assert float((A.mean(0) - ref).norm() / ref.norm()) > 0.2                # the raw gap is not
    B = ref * 1.1 + 0.5 * torch.randn(4, 20000, generator=g, dtype=torch.float64)
    assert abs(AN.debiased_gap(B, ref) - 0.1) < 0.03


def test_gini_top_spearman_coarse():
    assert abs(AN.gini(torch.ones(100))) < 1e-12
    v = torch.zeros(100)
    v[0] = 1
    assert AN.gini(v) > 0.98 and abs(AN.top_share(v) - 1) < 1e-12
    rho, n = AN.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 11])
    assert n == 5 and abs(rho - 1) < 1e-12
    assert abs(AN.coarse_share(torch.ones(32, 32)) - 1) < 1e-12
    chk = (torch.arange(32)[:, None] + torch.arange(32)[None]) % 2 * 2.0 - 1
    assert AN.coarse_share(chk) < 1e-12
