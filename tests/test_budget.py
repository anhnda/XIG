"""
BUDGET — unit test CPU cho hàm thuần (không ResNet, không ảnh): K từ ngân sách, khả thi coupling, J, J thực nghiệm khớp công thức,
bảng dài J theo B, chọn θ*(B) + xác suất bootstrap.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from probe import budget_opt as BO  # noqa: E402
from probe import infid_core as IC  # noqa: E402

F64 = torch.float64


def test_k_of_and_coupling():
    assert BO.k_of(256, 64, 1) == 4
    assert BO.k_of(1024, 128, 1) == 8
    assert BO.k_of(100, 64, 1) is None
    assert BO.k_of(32, 64, 1) is None
    assert BO.coupling_ok("indep", 1) and not BO.coupling_ok("rev", 1)
    assert BO.coupling_ok("rev", 2) and not BO.coupling_ok("rev", 3)
    assert not BO.coupling_ok("indep", None)


def test_j_val():
    assert BO.j_val(1.0, 2.0, 4.0, 4, "indep") == pytest.approx(1.0 + 6.0 / 4)
    assert BO.j_val(1.0, 2.0, 4.0, 4, "rev") == pytest.approx(1.0 + 4.0 / 4)


def _pairs(m, n, rho, seed):
    """Cặp (y1, y2) Gauss, tương quan rho giữa thành viên; trung bình mu, phương sai 1 mỗi ô."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.linspace(-1, 1, n, dtype=F64)
    a = torch.randn(m, n, generator=g, dtype=F64)
    b = torch.randn(m, n, generator=g, dtype=F64)
    y1 = mu + a
    y2 = mu + rho * a + (1 - rho * rho) ** 0.5 * b
    return y1, y2, mu


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.6])
def test_empirical_J_matches_formula(rho):
    m, n = 4096, 16
    y1, y2, mu = _pairs(m, n, rho, 3)
    d = mu + 0.3
    st = IC.pair_core(y1, y2, d, [4, 16])
    emp = BO.empirical_J(y1, y2, d, [4, 16])
    for K in (4, 16):
        for cp in ("indep", "rev"):
            f = BO.j_val(st["L_mu"], st["Ve"], st["Vo"], K, cp)
            e = emp[f"Jemp_{cp}_{K}"]
            assert abs(e - f) / f < 0.05, (cp, K, e, f)
    assert emp["Jemp_rev_4_ng"] == m // 2 and emp["Jemp_indep_4_ng"] == 2 * (m // 4)


def _toy_long():
    import pandas as pd
    rows = []
    rng = np.random.default_rng(0)
    for i in range(20):
        for cfgname, a, N, L, Ve, Vo in (("spi_a0.1_N64", 0.1, 64, 1.0, 8.0, 9.0), ("spi_a10_N64", 10.0, 64, 2.0, 1.0, 1.1),
                                         ("spi_a1_N32", 1.0, 32, 1.5, 3.0, 3.5)):
            rows.append({"image": f"im{i}", "config": cfgname, "family": "spi", "alpha": a, "N": N, "G": np.nan, "eta": np.nan,
                         "set": "sel", "group": "all", "dset": "del_sq", "L_mu": L + 0.1 * rng.standard_normal(), "Ve": Ve, "Vo": Vo})
    return pd.DataFrame(rows)


def test_j_long_feasibility():
    jl = BO.j_long(_toy_long(), "sel", "del_sq", [64, 128], 1)
    q = jl[(jl.B == 64) & (jl.N == 64)]
    assert set(q.coupling) == {"indep"} and set(q.K) == {1}
    q = jl[(jl.B == 64) & (jl.N == 32)]
    assert set(q.coupling) == {"indep", "rev"} and set(q.K) == {2}
    assert len(jl[(jl.B == 128) & (jl.N == 64)]) == 2 * 20 * 2


def test_select_budget():
    jl = BO.j_long(_toy_long(), "sel", "del_sq", [256, 2048], 1)
    s = BO.select_budget(jl, 256, 500, 1)
    # K = 4: α = 10 rev J ≈ 2 + 0.5 = 2.5; α = 0.1 rev ≈ 1 + 4 = 5; α = 1 (N 32, K 8) rev ≈ 1.5 + 0.75 = 2.25
    assert s["chosen"] == "spi_a1_N32|rev" and s["K"] == 8
    assert s["table"][0]["p_select"] > 0.9
    s = BO.select_budget(jl, 2048, 500, 1)
    # K = 32: α = 0.1 rev ≈ 1 + 0.5 = 1.5 thắng
    assert s["chosen"] == "spi_a0.1_N64|rev"
    s = BO.select_budget(jl, 256, 200, 1, allowed=["spi_a10_N64|indep", "spi_a10_N64|rev"])
    assert s["chosen"] == "spi_a10_N64|rev"
