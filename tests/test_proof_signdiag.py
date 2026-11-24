"""Kiểm toán chứng minh Phase 2 (symbolic + MC) và phần CPU của chẩn đoán có dấu / không dấu."""

from __future__ import annotations

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
