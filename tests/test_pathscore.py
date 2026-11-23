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


def test_config_and_digests():
    from probe.pathbank import settings
    cfg = _cfg()
    S = settings(cfg)
    pc = PS.pcfg(cfg)
    assert list(pc["mask_bases"]) == ["blur", "black", "white", "grey", "mean"]
    cfg2 = copy.deepcopy(cfg)
    cfg2["pathscore"]["mask_bases"] = ["blur", "black"]
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


# ─────────────────────────────────────────────────────────────────────────────
# metric bổ sung (probe/pathscore_metrics.py)
# ─────────────────────────────────────────────────────────────────────────────

from probe import pathscore_metrics as PM  # noqa: E402


def _auroc_brute(v, y):
    pos, neg = v[y], v[~y]
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).double() + 0.5 * (d == 0).double()).mean())


def _ap_brute(v, y):
    v, y = v.tolist(), y.tolist()
    P = sum(y)
    ap, prev_r = 0.0, 0.0
    for t in sorted(set(v), reverse=True):
        sel = [yy for vv, yy in zip(v, y) if vv >= t]
        tp = sum(sel)
        r = tp / P
        ap += (r - prev_r) * tp / len(sel)
        prev_r = r
    return ap


def test_auroc_ap_against_bruteforce():
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        v = torch.randint(0, 6, (12, 12), generator=g).double()          # nhiều giá trị hoà
        y = torch.rand(12, 12, generator=g) < 0.3
        assert PM.auroc(v, y) == pytest.approx(_auroc_brute(v.flatten(), y.flatten()), abs=1e-12)
        assert PM.average_precision(v, y) == pytest.approx(_ap_brute(v.flatten(), y.flatten()), abs=1e-12)
    assert math.isnan(PM.auroc(torch.rand(4, 4), torch.zeros(4, 4, dtype=torch.bool)))


def test_localization_ideal_and_pointing_tol():
    H = W = 16
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[4:8, 4:8] = True
    A = torch.zeros(3, H, W)
    A[:, 4:8, 4:8] = torch.rand(3, 4, 4) + 0.1                         # mọi khối lượng trong mask
    perm = torch.randperm(H * W, generator=torch.Generator().manual_seed(1))
    q = PM.localization(A, mask, perm, 0.05, 0)
    for k in ("iou", "mass_in_mask", "energy_mask", "pointing", "prec_top", "auroc_mask", "ap_mask"):
        assert q[k] == pytest.approx(1.0), k
    B = torch.zeros(3, H, W)
    B[0, 4, 10] = 1.0                                                  # đỉnh cách mask 3 px theo chiều ngang
    assert PM.localization(B, mask, perm, 0.05, 0)["pointing"] == 0.0
    assert PM.localization(B, mask, perm, 0.05, 3)["pointing"] == 1.0
    assert PM.localization(B, mask, perm, 0.05, 0)["mass_in_mask"] == 0.0


def test_road_impute_reproduces_linear_image():
    H = W = 16
    ii, jj = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    x = torch.stack([0.1 * ii + 0.05 * jj, -0.2 * ii + 0.3, 0.07 * jj - 0.1 * ii + 1.0])[None].float()
    rem = torch.zeros(2, H, W, dtype=torch.bool)
    rem[0, 3:12, 2:9] = True                                           # khối trong, không chạm biên
    rem[1, 5:7, 5:14] = True
    z = torch.randn(2, 3, H, W, generator=torch.Generator().manual_seed(2))
    out = PM.road_impute(x, rem, z, torch.zeros(3), iters=5000, tol=1e-12)
    assert torch.allclose(out, x.expand(2, -1, -1, -1), atol=1e-4)    # trung bình đối xứng giữ nguyên hàm tuyến tính
    noisy = PM.road_impute(x, rem, z, torch.full((3,), 0.5), iters=5000, tol=1e-12)
    assert torch.equal(noisy[0][:, ~rem[0]], x[0][:, ~rem[0]])         # pixel đã biết không đổi, nhiễu chỉ ở pixel được điền
    assert float((noisy[0][:, rem[0]] - x[0][:, rem[0]]).abs().max()) > 0.05
    assert torch.allclose(PM.road_noise_sd(0.01, "cpu"), 0.01 / torch.tensor(STD, dtype=torch.float64))


def test_road_masks_morf_lerf():
    A = torch.rand(3, 10, 10, generator=torch.Generator().manual_seed(3))
    perm = torch.arange(100)
    fr = [0.2, 0.5, 0.7]
    R = PM.road_masks(A, perm, fr)
    assert R.shape == (6, 10, 10)
    for i, p in enumerate(fr):
        assert int(R[i].sum()) == round(p * 100) and int(R[3 + i].sum()) == round(p * 100)
    m = PM.pixmap(A)
    assert float(m[R[0]].min()) >= float(m[~R[0]].max())              # MoRF xoá đúng phần lớn nhất
    assert float(m[R[3]].max()) <= float(m[~R[3]].min())              # LeRF xoá đúng phần nhỏ nhất
    assert not bool((R[0] & R[5]).any()) and bool((R[1] | R[4]).all())   # MoRF(p) và LeRF(1−p) bù nhau


def test_subset_scores_and_masks():
    g = torch.Generator().manual_seed(4)
    S = PM.subset_masks(20, 20, 9, [4, 8, 20], g)
    assert S.shape == (9, 20, 20) and [int(s.sum()) for s in S[:3]] == [16, 64, 400]
    A = torch.randn(3, 20, 20, generator=g)
    pred = (A.double().sum(0)[None] * S.double()).sum((1, 2))
    q = PM.subset_scores(A, S, pred)
    assert q["infid"] == pytest.approx(0.0, abs=1e-12) and q["sens_n"] == pytest.approx(1.0)
    q2 = PM.subset_scores(2 * A, S, pred)                              # gấp đôi: tương quan vẫn 1, infidelity = 1
    assert q2["infid"] == pytest.approx(1.0) and q2["sens_n"] == pytest.approx(1.0)


def test_direction_flip_for_lower_is_better():
    paths, spi, bases = _synthetic("independent", n_img=6, K=8, bases=("blur",), seed=5)
    paths["infid"] = -paths["insdel_blur"]                            # nhỏ = tốt ⇒ sau khi đổi dấu trùng thứ tự với blur
    paths["iou"] = paths["insdel_blur"] * 0.5 + 1.0
    spi["infid"], spi["iou"] = -spi["insdel_blur"], spi["insdel_blur"] * 0.5 + 1.0
    st = PS.per_group_stats(paths, spi, bases, ["iou", "infid"])
    assert np.allclose(st["corr"].spearman, 1.0)
    cs = st["cross"]
    assert np.allclose(cs.best_rank, 1.0) and np.all(cs.best_gain > 0)
    z = cs.groupby("image").best_z.agg(["min", "max"])
    assert np.allclose(z["min"], z["max"])                             # cùng z cho mọi cặp (biến đổi affine của nhau)
    sp = st["spread"]
    assert np.allclose(sp[sp.metric == "infid"].spi.values, -sp[sp.metric == "blur"].spi.values)   # đơn vị gốc giữ dấu gốc


def test_figures_with_extra_metrics(tmp_path):
    paths, spi, bases = _synthetic("independent", n_img=4, K=8, bases=("blur", "black"), seed=6)
    rng = np.random.default_rng(6)
    extra = ["iou", "pointing", "road", "infid"]
    for m in extra:
        paths[m] = (rng.random(len(paths)) > 0.5).astype(float) if m == "pointing" else rng.normal(0.5, 0.1, len(paths))
        spi[m] = rng.normal(0.5, 0.1, len(spi))
    st = PS.per_group_stats(paths, spi, bases, extra)
    sp = PS.summarize(st["spread"], ["metric"], ["mean_path", "spi_minus_mean", "spi_z"], 100, 0)
    cs = PS.summarize(st["cross"], ["select_on", "eval_on"], ["best_gain", "best_rank", "best_z"], 100, 0)
    cr = PS.summarize(st["corr"], ["a", "b"], ["spearman"], 100, 0)
    bom = PS.summarize(st["bestofm"], ["metric", "m"], ["gain", "gain_z"], 100, 0)
    files = PS.make_figures(str(tmp_path), bases, cs, cr, sp, bom, paths, 8, extra)
    assert len(files) == 4 and all(os.path.exists(f) for f in files)


def test_extra_config_and_digest():
    cfg = _cfg()
    from probe.pathbank import settings
    S, pc = settings(cfg), PS.pcfg(cfg)
    xm = PS.xcfg(pc)
    assert xm["loc"] and xm["road"]["enabled"] and xm["subset"]["enabled"]
    cfg2 = copy.deepcopy(cfg)
    cfg2["pathscore"]["extra_metrics"] = {"road": {"enabled": False}}
    p0, p2 = PS.digests(cfg, S, pc), PS.digests(cfg2, settings(cfg2), PS.pcfg(cfg2))
    assert p0["attr"] == p2["attr"] and p0["scores"] != p2["scores"]   # đổi metric bổ sung: chỉ tính lại điểm
    cfg3 = copy.deepcopy(cfg)
    cfg3["pathscore"]["report_extra"] = ["iou"]
    assert PS.digests(cfg3, settings(cfg3), PS.pcfg(cfg3)) == p0       # chọn metric để báo cáo không đụng cache
    bad = copy.deepcopy(cfg)
    bad["pathscore"]["extra_metrics"] = {"road": {"fracs": [0.0, 0.5]}}
    with pytest.raises(SystemExit):
        PS.pcfg(bad)
    dbg = PS.xcfg(PS.pcfg(_cfg(debug=True)))
    assert dbg["road"]["fracs"] == [0.2, 0.5, 0.8] and dbg["subset"]["n"] == 12 and dbg["road"]["noise"] == 0.01
