"""PATH_FAMILIES — unit test CPU cho lịch trình 4 họ đường (đầu mút, đơn điệu, E Z = t, Var, đảo, hiệu chuẩn σ)."""

from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from probe import path_families as PF  # noqa: E402

PARAMS = {"dp": {"alpha": 1.0}, "ln": {"sigma": 2.0}, "cmix": {"lam": 0.5}, "smooth": {"delta": 0.8}, "ig": {}}


@pytest.mark.parametrize("fam", PF.FAMILIES)
def test_endpoints_monotone_mean(fam):
    g = torch.Generator().manual_seed(1)
    N, d = 16, 40000
    Z = PF.schedule(fam, N, (d,), g, PARAMS[fam])
    assert Z.shape == (N + 1, d)
    assert torch.allclose(Z[0], torch.zeros(d, dtype=torch.float64)) and torch.allclose(Z[-1], torch.ones(d, dtype=torch.float64))
    assert float((Z[1:] - Z[:-1]).min()) >= -1e-15
    t = torch.linspace(0, 1, N + 1, dtype=torch.float64)
    assert float((Z.mean(1) - t).abs().max()) < 0.01


def test_cmix_variance_exact_schedule():
    g = torch.Generator().manual_seed(2)
    N, d, lam = 16, 500000, 1 / 11
    Z = PF.schedule("cmix", N, (d,), g, {"lam": lam})
    t = torch.linspace(0, 1, N + 1, dtype=torch.float64)
    tgt = lam * t * (1 - t)
    # sát hai đầu mút Var do biến cố hiếm (nhảy sớm / muộn) ⇒ SE tương đối ≈ 2–3 % ở d = 5·10⁵
    assert torch.allclose(Z.var(1)[1:-1], tgt[1:-1], rtol=0.06)


def test_dp_cmix_same_variance_schedule():
    """λ = h = 1/(α + 1): cùng E Z và Var Z tại mọi nút với DP lưới."""
    N, d, a = 16, 200000, 3.0
    Zd = PF.schedule("dp", N, (d,), torch.Generator().manual_seed(3), {"alpha": a})
    Zc = PF.schedule("cmix", N, (d,), torch.Generator().manual_seed(4), {"lam": 1 / (a + 1)})
    assert torch.allclose(Zd.var(1)[1:-1], Zc.var(1)[1:-1], rtol=0.05)
    assert abs(PF.d_hat(Zd) - PF.d_dp_grid(a, N)) < 0.002


def test_smooth_monotone_bound():
    with pytest.raises(ValueError):
        PF.schedule("smooth", 8, (4,), torch.Generator().manual_seed(0), {"delta": 1.2})


@pytest.mark.parametrize("fam", ["dp", "ln", "cmix", "smooth"])
def test_reversal_invariance(fam):
    g = torch.Generator().manual_seed(5)
    N, d = 16, 20000
    Z = PF.schedule(fam, N, (d,), g, PARAMS[fam])
    R = 1.0 - Z.flip(0)
    for k in (N // 4, N // 2):
        assert PF.ks2(Z[k].numpy(), R[k].numpy()) < 1.36 * (2 / d) ** 0.5 * 1.5


def test_calibrate_ln_matches_dp():
    N, a = 16, 3.0
    tgt = PF.d_dp_grid(a, N)
    sig = PF.calibrate_ln(tgt, N, 20000, 10)
    Z = PF.schedule("ln", N, (20000,), torch.Generator().manual_seed(7), {"sigma": sig})
    assert abs(PF.d_hat(Z) - tgt) / tgt < 0.05


# ── Mệnh đề Phase 0 §5 (toy đa thức, tích phân chính xác) ───────────────────

@pytest.mark.parametrize("kw", [{}, {"quartic": 1.0}])
def test_proposition_deg3_deg4(kw):
    """Bậc 3 (mọi họ E Z = t, độc lập) và bậc 4 (thêm bất biến khi đảo): E A − A_IG = D·a₁ với cùng a₁."""
    rows = PF.toy_check(n_paths=100000, **kw)
    assert max(r["max |z| theo tọa độ"] for r in rows) < 4.5, rows


def test_proposition_fails_deg5_kurtosis():
    """dp α = 1 và cmix λ = ½ cùng hàm hiệp phương sai nhưng khác moment bậc 4 ⇒ E A₀ khác với F = x₀(x₁ − ½)⁴."""
    rows = PF.kurtosis_check(n_paths=200000)
    by = {r["họ"]: r for r in rows}
    assert abs(by["dp"]["D"] - by["cmix"]["D"]) < 0.002
    assert abs(by["cmix"]["z so với dp"]) > 6


# ── Phase 1 runner: hàm thuần + kiểm toán chi phí ───────────────────────────

def test_conf_names_and_params():
    from probe.sampling_infid import cname, norm_conf
    for c, nm in (({"family": "ln", "alpha": 1, "N": 32}, "ln_a1_N32"), ({"family": "cmix", "alpha": 3, "N": 32}, "cmix_a3_N32"),
                  ({"family": "smooth", "delta": 0.674, "N": 32}, "smooth_d0.674_N32"), ({"family": "spi", "alpha": 1, "N": 32}, "spi_a1_N32")):
        assert cname(c) == nm
        p = PF.family_params(norm_conf(c))
        assert p
    assert PF.family_params(norm_conf({"family": "cmix", "alpha": 3, "N": 32}))["lam"] == pytest.approx(0.25)


def test_dev_grid_20_configs():
    from probe import pathfam as PFM
    names = [PFM.cname(c) for c in PFM.dev_configs({})]
    assert len(names) == 20 and len(set(names)) == 20
    assert "spi_a1_N32" in names and "spi_a0.1_N64" in names and "spi_a10_N64" in names
    assert sum(n.startswith("smooth_") for n in names) == 3


@pytest.mark.parametrize("fam", ["spi", "ln", "cmix", "smooth"])
def test_grad_cost_and_completeness(fam):
    from probe import pathfam as PFM
    from probe.sampling_infid import norm_conf
    c = norm_conf({"family": fam, "alpha": 1, "delta": 0.674, "N": 8} if fam != "smooth" else {"family": fam, "delta": 0.674, "N": 8})
    r = PFM.grads_per_path(c, 1)
    assert r["grads_per_path"] == r["expected"] == 8
    assert r["completeness_err_linear"] < 1e-10


def test_ni_tests_and_power():
    import numpy as np
    from probe import pathfam as PFM
    rng = np.random.default_rng(0)
    ji = rng.lognormal(2.5, 0.4, 96)
    same = ji * (1 + rng.normal(0, 0.02, 96))
    r = PFM.tests_for(ji, same, 0.10, 2000, 1, 0.05)
    assert r["p_ni"] < 0.01 and r["p_tost"] < 0.01 and r["p_sup"] > 0.05
    worse = ji * 1.3
    r2 = PFM.tests_for(ji, worse, 0.10, 2000, 1, 0.05)
    assert r2["p_ni"] > 0.5 and r2["g_hi"] < -0.10
    r2["p_ni_holm"] = 1.0
    assert PFM.decide(r2, 0.10, 0.05) == "KÉM HƠN thực tiễn"
    pl = PFM.plan_ni(ji, same, 0.10, 0.0125, 0.8)
    assert pl["n_req_g0"] >= 1 and pl["n_req_gdev"] >= 1


# ── Phase 2: định lý ở mức phân phối đường ──────────────────────────────────

def test_theorem2_levy_area_and_rev_exact():
    from probe import path_theory as PT
    rows = PT.t2_t5(P=20000)
    for r in rows:
        assert r["max |A − IG − Δ(H∘𝔄)Δ| (từng đường)"] < 1e-12
        assert r["max |A(Z) + A(RZ) − 2 IG| (từng đường)"] < 1e-12
        if r["bất biến khi đảo"]:
            assert r["max |z| của E 𝔄 (≈ 0 nếu bất biến)"] < 4.5
        else:
            assert r["‖E A − IG‖"] > 0.1


def test_theorem4_area_variance():
    from probe import path_theory as PT
    rows = {(r["họ"], r["N"]): r for r in PT.t4a(P=200000)}
    assert abs(rows[("smooth", 32)]["lệch tương đối"]) < 0.02
    for fam in ("dp", "cmix"):
        assert abs(rows[(fam, 128)]["lệch tương đối"]) < abs(rows[(fam, 32)]["lệch tương đối"]) + 0.01   # hội tụ theo N
        assert abs(rows[(fam, 128)]["lệch tương đối"]) < 0.03


def test_theorem4b_same_mean_different_variance():
    from probe import path_theory as PT
    r = PT.t4b(P=100000)[0]
    assert r["max |z| E A (≈ 0: cùng μ)"] < 4.5 and r["max |z| Var A (lớn: khác phương sai)"] > 10
