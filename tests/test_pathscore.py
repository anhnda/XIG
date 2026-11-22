"""PATHSCORE — test (CPU, không ResNet, không dữ liệu ảnh)."""

from __future__ import annotations

import copy
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from probe import config as C  # noqa: E402
from probe import pathscore as PS  # noqa: E402
from probe.data import MEAN, STD  # noqa: E402


def _cfg(debug=False):
    return C.load(os.path.join(ROOT, "config.yaml"), debug=debug)


def test_best_of_m_exact():
    x = [0.3, -0.1, 0.7, 0.2, 0.05, 0.4]
    for m in range(1, 7):
        brute = sum(max(c) for c in itertools.combinations(x, m)) / math.comb(6, m)
        assert PS.best_of_m(x, m) == pytest.approx(brute, abs=1e-12)


def test_mask_bases_values():
    x = torch.rand(1, 3, 8, 8)
    ic = {"blur_kernel": 3, "blur_sigma": 1.0}
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    sd = torch.tensor(STD).view(1, 3, 1, 1)
    assert torch.allclose(PS.mask_base(x, "white", ic) * sd + m, torch.ones_like(x), atol=1e-6)
    assert torch.allclose(PS.mask_base(x, "grey", ic) * sd + m, torch.full_like(x, 0.5), atol=1e-6)
    assert torch.allclose(PS.mask_base(x, "black", ic) * sd + m, torch.zeros_like(x), atol=1e-6)
    mb = PS.mask_base(x, "mean", ic)
    assert torch.allclose(mb[0, :, 0, 0], x[0].mean(dim=(1, 2))) and torch.allclose(mb.std(dim=(2, 3)), torch.zeros(1, 3), atol=1e-6)
    assert PS.mask_base(x, "blur", ic).shape == x.shape
    with pytest.raises(ValueError):
        PS.mask_base(x, "pink", ic)


def _synthetic(kind: str, n_img=6, K=16, bases=("blur", "black", "white"), seed=0):
    """identical: mọi metric bằng nhau; independent: điểm các metric độc lập."""
    rng = np.random.default_rng(seed)
    P, E = [], []
    for i in range(n_img):
        base = rng.normal(0.2, 0.02, K)
        rowsK = []
        for k in range(K):
            r = {"image": f"im{i}", "seed": 0, "path": k, "K": K}
            for b in bases:
                r[f"insdel_{b}"] = base[k] if kind == "identical" else rng.normal(0.2, 0.02)
            for px in PS.PROXIES:
                r[px] = rng.normal()
            rowsK.append(r)
        P += rowsK
        e = {"image": f"im{i}", "seed": 0, "K": K}
        for b in bases:
            e[f"insdel_{b}"] = np.mean([r[f"insdel_{b}"] for r in rowsK]) + 0.01
        E.append(e)
    return pd.DataFrame(P), pd.DataFrame(E), list(bases)


def test_cross_identical_metrics():
    paths, spi, bases = _synthetic("identical")
    st = PS.per_group_stats(paths, spi, bases)
    cs = PS.summarize(st["cross"], ["select_on", "eval_on"], ["best_gain", "best_rank"], 200, 0)
    assert np.allclose(cs.best_rank, 1.0)                              # cùng đường là tốt nhất ở mọi metric
    assert np.allclose(cs.best_gain.values, cs.best_gain.values[0], atol=1e-12) and cs.best_gain.values[0] > 0
    cr = PS.summarize(st["corr"], ["a", "b"], ["spearman"], 200, 0)
    assert np.allclose(cr.spearman, 1.0)
    sp = PS.summarize(st["spread"], ["metric"], ["spi_minus_mean"], 200, 0)
    assert np.allclose(sp.spi_minus_mean, 0.01)


def test_cross_independent_metrics():
    paths, spi, bases = _synthetic("independent", n_img=40, seed=3)
    st = PS.per_group_stats(paths, spi, bases)
    cs = PS.summarize(st["cross"], ["select_on", "eval_on"], ["best_gain", "best_rank"], 500, 0)
    off = cs[cs.select_on != cs.eval_on]
    diag = cs[cs.select_on == cs.eval_on]
    assert np.all(diag.best_rank == 1.0) and np.all(diag.best_gain > 0.02)
    assert abs(off.best_gain.mean()) < 0.006 and abs(off.best_rank.mean() - 8.5) < 1.5
    bom = st["bestofm"]
    g = bom.groupby("m").gain.mean()
    assert g.loc[1] == pytest.approx(0.0, abs=1e-12) and g.is_monotonic_increasing


def test_summarize_mean_std_min_max_over_images():
    df = pd.DataFrame({"k": ["a"] * 6, "image": ["i1", "i1", "i2", "i2", "i3", "i3"], "v": [1.0, 3.0, 4.0, 4.0, 7.0, 9.0]})
    s = PS.summarize(df, ["k"], ["v"], 200, 0).iloc[0]          # theo ảnh (trung bình seed trước): 2, 4, 8
    assert s.v == pytest.approx(14 / 3) and s.v_min == 2.0 and s.v_max == 8.0
    assert s.v_std == pytest.approx(np.std([2.0, 4.0, 8.0], ddof=1)) and s.n_images == 3
    lt = PS.long_stats(PS.summarize(df, ["k"], ["v"], 200, 0), ["k"], ["v"], {"v": "giá trị"})
    assert list(lt.columns[:2]) == ["k", "đại lượng"] and lt.iloc[0]["đại lượng"] == "giá trị"
    assert lt.iloc[0]["min"] == 2.0 and lt.iloc[0]["max"] == 8.0


def test_figures_and_tables_on_synthetic(tmp_path):
    paths, spi, bases = _synthetic("independent", n_img=5, K=8, seed=1)
    st = PS.per_group_stats(paths, spi, bases)
    sp = PS.summarize(st["spread"], ["metric"], ["mean_path", "spi_minus_mean"], 100, 0)
    cs = PS.summarize(st["cross"], ["select_on", "eval_on"], ["best_gain", "best_rank", "worst_gain", "worst_rank", "spi_gain"], 100, 0)
    cr = PS.summarize(st["corr"], ["a", "b"], ["spearman"], 100, 0)
    bom = st["bestofm"].groupby(["metric", "m"]).gain.mean().reset_index()
    files = PS.make_figures(str(tmp_path), bases, cs, cr, sp, bom, paths, 8)
    assert len(files) == 4 and all(os.path.exists(f) for f in files)


def test_config_separate_from_p4():
    from probe.sampling_benchmark import digests, settings
    cfg = _cfg()
    S = settings(cfg)
    pc = PS.pcfg(cfg)
    assert list(pc["mask_bases"]) == ["blur", "black", "white", "grey", "mean"]
    d0 = digests(cfg, S)
    assert "pathscore" not in d0
    cfg2 = copy.deepcopy(cfg)
    cfg2["pathscore"]["mask_bases"] = ["blur", "black"]
    assert digests(cfg2, settings(cfg2)) == d0                        # đổi pathscore không đụng cache P4
    assert C.digest(cfg2) == C.digest(cfg)                            # cũng không đụng P0–P3
    p0, p2 = PS.digests(cfg, S, pc), PS.digests(cfg2, settings(cfg2), PS.pcfg(cfg2))
    assert p0["attr"] == p2["attr"] and p0["scores"] != p2["scores"]   # đổi metric: không phải tính lại bản đồ
    e = PS.estimate(S, pc, 0.002, 0.0007)
    assert e["grad_per_image"] == pc["K"] * pc["seeds"] * S["N"] * S["M"]
    bad = copy.deepcopy(cfg)
    bad["pathscore"]["mask_bases"] = ["pink"]
    with pytest.raises(SystemExit):
        PS.pcfg(bad)
    dbg = PS.pcfg(_cfg(debug=True))
    assert dbg["K"] == 4 and dbg["n_images"] == 2
