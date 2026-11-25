"""Kiểm toán chứng minh Phase 2 (symbolic + MC) và phần CPU của chẩn đoán có dấu / không dấu."""

from __future__ import annotations

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
from probe import proof_check as PC  # noqa: E402
from probe import signdiag as SD  # noqa: E402


def test_symbolic_patterns_exact():
    rows = PC.pattern_check(True)
    assert all(r["hiệu (phải = 0)"] in ("0", "—") for r in rows)
    sym = PC.symmetry_check()
    assert sym["∫ t v − D/2"] == "0" and sym["∫ κ₃"] == "0"
    assert PC.deg5_check()["hiệu"] == "0"


def test_symbolic_theorem4_all_overlaps():
    rows = PC.cov_check()
    assert all(r["hiệu"] in ("0", "—") for r in rows)


def test_shapley_limit_and_compression():
    assert PC.compress_check() < 1e-12
    rows = PC.shapley_mc(P=40000)
    assert max(abs(r["z"]) for r in rows) < 4.5


def test_map_stats_signs():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(3, 8, 8, generator=g, dtype=torch.float64)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[2:5, 2:5] = True
    perm = torch.randperm(64, generator=g)
    s = SD.map_stats(A, float(A.sum()), mask, perm)
    assert abs(s["compl_err"]) < 1e-12 and s["cancel"] >= 1 and 0 < s["neg_frac"] < 1 and 0 < s["chan_cancel"] <= 1
    Ap = A.abs()
    s2 = SD.map_stats(Ap, float(Ap.sum()), mask, perm)
    assert s2["cancel"] == pytest.approx(1.0) and s2["neg_frac"] == 0 and s2["spearman_signed_abs"] == pytest.approx(1.0)
    assert s2["iou_abs"] == pytest.approx(s2["iou_pos"])


def test_signdiag_report_offline(tmp_path):
    cfg = C.load(os.path.join(ROOT, "config.yaml"))
    cfg["local"] = {**cfg.get("local", {}), "out": str(tmp_path / "runs")}
    cfg["pathfam"]["protocol"]["bootstrap"] = 200
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        for c, h in (("spi_a1_N32", 0.5), ("ln_a1_N32", 0.5), ("cmix_a1_N32", 0.5), ("smooth_d0.674_N32", 0.09), ("spi_a10_N64", 0.09)):
            for est in ("rev_B256_K8", "mean_all"):
                inf = 10 + 5 * (1 - h) + rng.normal(0, 0.5)
                canc = 1.5 + 3 * h + rng.normal(0, 0.1)
                rows.append({"image": f"seg{i:05d}", "config": c, "family": c.split("_")[0], "N": 32, "h": h, "estimator": est, "infid": inf,
                             "infid_beta": inf * 0.9, "compl_err": rng.normal(0, 0.01), "cancel": canc, "neg_frac": 0.3, "chan_cancel": 0.6,
                             "insdel_abs": 0.2 - 0.02 * canc + rng.normal(0, 0.01), "insdel_signed": 0.2 - 0.01 * inf + rng.normal(0, 0.01),
                             "energy_abs": 0.5, "energy_pos": 0.5 + rng.normal(0, 0.01), "iou_abs": 0.4 + rng.normal(0, 0.01),
                             "iou_pos": 0.4 + rng.normal(0, 0.01), "spearman_signed_abs": 0.3})
    os.makedirs(SD.out_dir(cfg), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(SD.out_dir(cfg), "signdiag.csv"), index=False)
    txt = open(SD.report(cfg)).read()
    assert "H-SU" in txt and "τ(insdel_signed) − τ(insdel_abs)" in txt and "smooth" in txt


def test_insdel_baseline_mismatch():
    """InsDel thật (`sampling.insdel_full`) với mô hình tuyến tính: nền đen + xếp hạng có dấu đúng chiều; nền blur mất ý nghĩa."""
    import torch.nn as nn
    from probe import sampling as SM

    class Lin(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.w = w

        def forward(self, x):
            z = (x * self.w).flatten(1).sum(1)
            return torch.stack([z, torch.zeros_like(z)], 1)
    H = W = 8
    x = torch.ones(1, 1, H, W)
    x[0, 0, ::3, ::3] = 0.5
    w = torch.ones(1, H, W) * 0.05
    w[0, ::3, ::3] = 0.4
    w[0, 1::3, 1::3] = -0.3
    b = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate"), torch.ones(1, 1, 3, 3) / 9)
    A = w * x[0]
    m = Lin(w)
    r = {(nm, sg): SM.insdel_full(m, 0, x, base, A, 16, 64, signed=sg)["diff"] for nm, base in (("black", torch.zeros_like(x)), ("blur", b))
         for sg in (False, True)}
    assert r[("black", True)] > 0.5 and r[("black", False)] < 0
    assert r[("blur", True)] < 0.1 and r[("blur", False)] < 0.1


def test_scale_terms_and_matched():
    from probe import signdiag2 as S2
    from probe.sampling_infid import cname, norm_conf
    st = S2.scale_terms(4.0, 2.0, 1.0)                    # y = 2d
    assert st["beta_opt"] == pytest.approx(0.5) and st["L_1"] == pytest.approx(1.0) and st["L_opt"] == pytest.approx(0.0)
    assert S2.L_at(0.5, 4.0, 2.0, 1.0) == pytest.approx(0.0) and S2.L_at(0.0, 4.0, 2.0, 1.0) == pytest.approx(1.0)
    cfg = C.load(os.path.join(ROOT, "config.yaml"))
    confs = [norm_conf(c) for c in ({"family": "ln", "alpha": 3, "N": 32}, {"family": "smooth", "delta": 1.0, "N": 32},
                                     {"family": "spi", "alpha": 1, "N": 32}, {"family": "spi", "alpha": 10, "N": 64})]
    names = {cname(c) for c in S2.matched_configs(cfg, confs)}
    assert {"spi_a3_N32", "cmix_a3_N32", "spi_a4_N32", "ln_a4_N32", "cmix_a4_N32", "ln_a1_N32", "cmix_a1_N32"} <= names
    assert "ln_a3_N32" not in names and not any(n.endswith("_N64") for n in names)


def test_signdiag2_report_offline(tmp_path):
    from probe import signdiag2 as S2
    cfg = C.load(os.path.join(ROOT, "config.yaml"))
    cfg["local"] = {**cfg.get("local", {}), "out": str(tmp_path / "runs")}
    cfg["pathfam"]["protocol"]["bootstrap"] = 200
    rng = np.random.default_rng(0)
    rows = []
    for i in range(16):
        dF = float(rng.normal(3, 1))
        for c, fam, h in (("spi_a1_N32", "spi", 0.5), ("ln_a1_N32", "ln", 0.5), ("cmix_a1_N32", "cmix", 0.5)):
            for K in (2, 4, 8, 16, 32):
                lab = "mean_all" if K == 32 else f"rev_K{K}"
                yy, yd, dd = 1.2 + 1 / K + rng.normal(0, 0.05), 0.9, 1.0
                r = {"image": f"seg{i:05d}", "config": c, "family": fam, "N": 32, "h": h, "estimator": lab, "K": K, "B": 32 * K, "matched_extra": fam != "spi",
                     "F0": 0.0, "F1": dF, "dF": dF, "sumA": dF + rng.normal(0, 1 / math.sqrt(K)), "l1": 5.0,
                     "yy_sel": yy, "yd_sel": yd, "dd_sel": dd, "yy_eval": yy * 1.01, "yd_eval": yd, "dd_eval": dd,
                     "compl_err": 0.0, "cancel": 2.0, "neg_frac": 0.3, "chan_cancel": 0.6 + rng.normal(0, 0.02), "energy_abs": 0.5, "energy_pos": 0.5,
                     "iou_abs": 0.4 + rng.normal(0, 0.02), "iou_pos": 0.4, "spearman_signed_abs": 0.3}
                if K in (8, 32):
                    r.update({k: float(rng.normal(0.1, 0.02)) for k in ("insdel_abs_blur", "insdel_signed_blur", "insdel_abs_black", "insdel_signed_black")})
                rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(S2.out_dir(cfg), "signdiag2.csv"), index=False)
    txt = open(S2.report(cfg)).read()
    for key in ("## 1. Thang đo", "## 2. So sánh có kiểm soát", "## 3. InsDel", "## 4. Completeness", "τ(chan_cancel, infid)"):
        assert key in txt


def test_final_audit_beta_and_completeness(tmp_path):
    from probe import pathfam_final as FF
    yy, yd, dd = np.array([4.0, 1.0, 2.0]), np.array([2.0, 0.5, -0.4]), np.array([1.0, 1.0, 3.0])
    bp = FF.beta_parts(yy, yd, dd)
    assert np.allclose(bp["beta"], yd / yy) and np.allclose(bp["Lopt"], 1 - yd ** 2 / (yy * dd))
    assert np.allclose(bp["L1"], FF.L_beta(1.0, yy, yd, dd))
    b = FF.pooled_beta(yy, yd, dd)
    grid = np.linspace(-1, 2, 3001)
    assert abs(grid[np.argmin([FF.L_beta(x, yy, yd, dd).sum() for x in grid])] - b) < 2e-3
    lo = FF.loio_beta(yy, yd, dd)
    assert lo[0] == pytest.approx(FF.pooled_beta(yy[1:], yd[1:], dd[1:]))
    r = FF.completeness_sim(reps=600, nb=300, Ks=(2, 8, 32))
    for v in r.values():
        assert -0.62 < v["slope"] < -0.38
    cfg = C.load(os.path.join(ROOT, "config.yaml"))
    cfg["local"] = {**cfg.get("local", {}), "out": str(tmp_path / "runs")}
    rec = FF.lock_diag(cfg)
    assert "insdel_signed_blur" in rec["excluded"] and "insdel_signed_black" in rec["admissible_metrics"]


def test_completeness_blocks_and_slope_parts():
    from probe import pathfam_final as FF
    rng = np.random.default_rng(0)
    n = 3000
    sig, b = np.exp(0.5 * rng.standard_normal(n)), 0.3 * rng.standard_normal(n)
    pairs = b[:, None] + sig[:, None] * rng.standard_normal((n, 16))
    S = np.stack([pairs[:, :k].mean(1) for k in (1, 2, 4, 8, 16)], 1)
    B, nn = FF.blocks_from_nested(S)
    assert np.allclose(B[:, 0], pairs[:, 0]) and np.allclose(B[:, 1], pairs[:, 1]) and np.allclose(B[:, 4], pairs[:, 8:].sum(1))
    m, s2 = FF.within_var(B, nn)
    assert np.allclose(m, pairs.mean(1))
    assert abs(s2.mean() / (sig ** 2).mean() - 1) < 0.06                     # không chệch cho σ², không bị b_i làm phồng
    assert abs((m ** 2 - s2 / 16).mean() - (b ** 2).mean()) < 0.02            # b̂² không chệch
    sp = FF.slope_parts(S)
    assert sp[0] == pytest.approx(sp[1] + sp[2])                              # phân rã chính xác
    # chuẩn, không lệch: độ dốc ≈ −½; có b_i: phẳng hơn
    w = sig[:, None] * rng.standard_normal((n, 16))
    z = np.stack([w[:, :k].mean(1) for k in (1, 2, 4, 8, 16)], 1)
    assert -0.56 < FF.slope_parts(z)[0] < -0.44 and FF.slope_parts(S)[0] > -0.45


def test_slope_shape_sim_directions():
    from probe import pathfam_final as FF
    rows = {(r["hình dạng sai số cặp"], r["độ lệch riêng ảnh tb"]): r for r in
            FF.slope_shape_sim(128, 0.5, shapes=("normal", "spike"), tbs=(0.0, 0.3), R=40, nb=60)}
    e = {k: float(v["E độ dốc TB|err| (SD)"].split()[0]) for k, v in rows.items()}
    assert abs(e[("normal", 0.0)] + 0.5) < 0.05
    assert e[("spike", 0.0)] > -0.42 and e[("normal", 0.3)] > -0.45


def test_completeness_diag_offline():
    import pandas as pd
    from probe import pathfam_final as FF
    rng = np.random.default_rng(1)
    rows = []
    for i in range(40):
        p = np.exp(rng.normal()) * rng.standard_normal(16)
        for c, K in zip((1, 2, 4, 8, 16), (2, 4, 8, 16, 32)):
            rows.append(dict(config="spi_a10_N32", estimator=f"rev_K{K}", K=K, image=i, sumA=1.0 + p[:c].mean(), dF=1.0))
        rows.append(dict(config="spi_a10_N32", estimator="mean_all", K=32, image=i, sumA=1.0 + p.mean(), dF=1.0))
    out = "\n".join(FF.completeness_diag(pd.DataFrame(rows), 50, 0, 0.05, R=10))
    assert "### 2c" in out and "### 2d" in out and "spi_a10_N32" in out


def test_offset_shared_detects_common_bias():
    import pandas as pd
    from probe import pathfam_final as FF
    rng = np.random.default_rng(2)
    rows = []
    n = 400
    b = 0.8 * rng.standard_normal(n)                                         # độ lệch theo ảnh, chung mọi cấu hình
    dF = rng.normal(5, 2, n)
    for cfg, shared in (("a_N32", True), ("b_N32", True), ("c_N64", True), ("d_N32", False)):
        for i in range(n):
            e = (b[i] if shared else 0.0) + rng.standard_normal(16).mean()
            rows.append(dict(config=cfg, estimator="rev_K32", K=32, image=i, sumA=dF[i] + e, dF=dF[i]))
    out = "\n".join(FF.offset_shared(pd.DataFrame(rows), 200, 0, 0.05))
    assert "### 2e" in out and "chỉ N = 32" in out
    t = pd.DataFrame(rows)
    t2 = t[t.config != "d_N32"]
    X = t2.assign(err=t2.sumA - t2.dF).pivot_table(index="image", columns="config", values="err").values
    s = X.sum(1)
    cr = ((s ** 2 - (X ** 2).sum(1)) / (3 * 2)).mean()
    assert abs(cr - (b ** 2).mean()) < 0.12
    print(out)
