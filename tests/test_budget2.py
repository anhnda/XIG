"""
budget-audit / BUDGET2 — unit test CPU cho hàm thuần: tuyến tính hoá R ⇔ D, Fieller, BCa, bootstrap-t, cổng dev, kế hoạch n,
quyết định test; danh sách cấu hình dev.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from probe import budget2 as B2  # noqa: E402
from probe import budget_audit as BA  # noqa: E402


def _tbf(n=48, seed=0, gain=0.3):
    rng = np.random.default_rng(seed)
    b = rng.lognormal(3.0, 0.4, n)
    f = 0.03 * b
    t = b - gain * (b - f) + rng.normal(0, 0.5, n)
    return t, b, f


def test_linearization_equivalence():
    t, b, f = _tbf()
    R = BA.ratio_R(t, b, f)
    for r0 in (R - 0.05, R + 0.05):
        lt = BA.linear_test(t, b, f, r0)
        assert (lt["D_mean"] > 0) == (R > r0)
    assert BA.linear_test(t, b, f, R)["D_mean"] == pytest.approx(0.0, abs=1e-9)


def test_fieller_contains_point_and_orders():
    t, b, f = _tbf()
    R = BA.ratio_R(t, b, f)
    lo, hi = BA.fieller(t, b, f, 0.95)
    assert lo < R < hi
    lo1, hi1 = BA.fieller(t, b, f, 0.95, one_sided=True)
    assert lo < lo1 < R and math.isinf(hi1)
    # Fieller: R ∈ CI ⇔ |t(D_R)| ≤ q — kiểm tại hai đầu mút
    assert abs(BA.linear_test(t, b, f, lo)["t"]) == pytest.approx(BA.t_ppf(0.975, len(t) - 1), rel=1e-6)


def test_bca_and_boot_t():
    t, b, f = _tbf(gain=0.3)
    X = np.stack([t, b, f], 1)
    lo, hi = BA.bca(lambda Z: BA.ratio_R(Z[:, 0], Z[:, 1], Z[:, 2]), X, 2000, 1)
    R = BA.ratio_R(t, b, f)
    assert lo < R < hi
    D = (b - t) - 0.2 * (b - f)
    assert BA.boot_t_p(D, 2000, 1) < 0.05            # hiệu ứng thật 0,3 ≫ 0,2
    D0 = (b - t) - 0.35 * (b - f)                    # r₀ > hiệu ứng thật ⇒ không bác H0
    assert BA.boot_t_p(D0, 2000, 1) > 0.05


def test_gate_and_plan():
    rng = np.random.default_rng(1)
    jr = rng.lognormal(2.5, 0.4, 32)
    jc = 0.8 * jr + rng.normal(0, 0.2, 32)
    g = B2.gate(jr, jc, 0.05, 0.8, 1000, 2)
    assert g["pass"] and g["gain_rel"] == pytest.approx(0.2, abs=0.03)
    g2 = B2.gate(jr, jr * 0.99, 0.05, 0.8, 1000, 2)
    assert not g2["pass"]
    pl = B2.plan_n(jr, jc, 0.05, 0.05, 0.8)
    assert pl["g_star"] == pytest.approx((pl["g_dev"] + 0.05) / 2)
    assert pl["n_required"] >= 1
    assert math.isnan(B2.plan_n(jr, jr * 1.01, 0.05, 0.05, 0.8)["n_required"])


def test_decide_test():
    rng = np.random.default_rng(3)
    jr = rng.lognormal(2.5, 0.4, 64)
    jn = 0.75 * jr + rng.normal(0, 0.1, 64)
    d = B2.decide_test(jr, jn, 0.05, 2000, 4, 0.05)
    assert d["p_boot_t"] < 0.05 and d["g_lo"] > 0.05 and d["wins"] > 50
    d = B2.decide_test(jr, jr * 0.99, 0.05, 2000, 4, 0.05)
    assert d["p_boot_t"] > 0.05 and d["g_hi"] < 0.05


def test_dev_configs_dedup():
    cfg = {"budget2": {}}
    cf = B2.dev_configs(cfg)
    names = [B2.cname(c) for c in cf]
    assert len(names) == len(set(names))
    assert "spi_a1_N32" in names and "spi_a0.1_N64" in names
    assert sum(n.startswith("hier_") for n in names) == 6
    assert sum(n.startswith("spi_") for n in names) == 10   # 3 α × 3 N + α = 0,1 N = 64 (incumbent nằm trong lưới)
