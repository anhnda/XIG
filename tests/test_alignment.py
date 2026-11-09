"""Bug B1: Δf must be aligned with d_k and sum to f(x) − f(x')."""

import torch

import utility
from ig import compute_ig
from pathattr import eval_path, legacy_df
from pathattr import methods as M
from pathattr import paths
from tests.toy import setup


def test_eval_path_df_sums_to_gap():
    model, oracle, x, xb, D = setup()
    pe = eval_path(oracle, paths.straight(xb, x, 10))
    assert abs(float(pe.df.sum()) - D) < 1e-10
    assert torch.allclose(pe.d, (pe.g * pe.h).flatten(1).sum(1))
    assert pe.f.shape[0] == 11 and pe.g.shape[0] == 10


def test_legacy_df_is_shifted_by_one():
    model, oracle, x, xb, D = setup()
    pe = eval_path(oracle, paths.straight(xb, x, 10))
    dl = legacy_df(pe.f)
    assert float(dl[0]) == 0.0
    assert torch.allclose(dl[1:], pe.df[:-1])
    assert abs(float(dl.sum()) - D) > 1e-8          # last step dropped


def test_idgi_complete_without_rescale():
    model, oracle, x, xb, D = setup()
    a = M.idgi(oracle, xb, x, N=16).attr
    assert abs(float(a.sum()) - D) < 1e-9


def test_new_ig_matches_legacy_ig():
    model, oracle, x, xb, D = setup()
    a_new = M.ig(oracle, xb, x, N=12).attr
    a_old = compute_ig(model, x, {"baseline": xb, "N": 12}).attributions
    assert torch.allclose(a_new, a_old, atol=1e-12)


def test_legacy_flag_in_straight_line_pass():
    model, oracle, x, xb, D = setup()
    pe = eval_path(oracle, paths.straight(xb, x, 10))
    old = utility.FIX_DF_ALIGNMENT
    try:
        utility.FIX_DF_ALIGNMENT = False
        *_, df_list, _, _ = utility._straight_line_pass(model, x, xb, 10)
        # legacy Δf_0 = f(γ_0) − f(x') from two different forward calls: 0 up to round-off
        assert abs(df_list[0]) < 1e-12
        assert abs(sum(df_list) - D) > 1e-8
        utility.FIX_DF_ALIGNMENT = True
        *_, df_list, _, _ = utility._straight_line_pass(model, x, xb, 10)
        assert abs(sum(df_list) - D) < 1e-10
        assert torch.allclose(torch.tensor(df_list, dtype=torch.float64), pe.df, atol=1e-12)
    finally:
        utility.FIX_DF_ALIGNMENT = old


def test_lig_mu_left_matches_legacy_lig_idig_fixed():
    from lig_idig import compute_lig_idig
    model, oracle, x, xb, D = setup()
    old = utility.FIX_DF_ALIGNMENT
    try:
        utility.FIX_DF_ALIGNMENT = True
        a_old = compute_lig_idig(model, x, {"baseline": xb, "N": 12}).attributions
    finally:
        utility.FIX_DF_ALIGNMENT = old
    a_new = M.lig_mu(oracle, xb, x, N=12, node="left", mu="adam").attr
    assert float((a_new - a_old).norm() / a_old.norm()) < 1e-5
    assert abs(float(a_new.sum()) - D) < 1e-9


def test_lig_mu_mid_and_proj_run():
    model, oracle, x, xb, D = setup()
    for kw in (dict(node="mid"), dict(node="mid", mu="proj"),
               dict(node="mid", path="dnoise", N=8)):
        r = M.lig_mu(oracle, xb, x, torch.Generator().manual_seed(0), **kw)
        assert torch.isfinite(r.attr).all()
        assert abs(float(r.attr.sum()) - D) < 1e-9


def test_oracle_counts():
    model, oracle, x, xb, D = setup()
    oracle.reset()
    M.ig(oracle, xb, x, N=12)
    assert oracle.counts()[1] == 12
