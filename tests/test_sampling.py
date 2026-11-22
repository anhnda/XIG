"""
P4 — unit / integration test cho sampler, chẩn đoán, runner và thống kê (CPU, không ResNet, không dữ liệu ảnh).

T1 tính hợp lệ của đường; T2 antithetic; T3 thiết kế cân bằng; T4 kiểm chứng phân phối biên (chẩn đoán thống kê, không phải
chứng minh); T5 estimator số học + nhất quán metric + hash/cache; T6 không hồi quy (chạy cùng tests/test_units.py).
"""

from __future__ import annotations

import copy
import math
import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from probe import config as C  # noqa: E402
from probe import sampling as SM  # noqa: E402
from probe.mechanism import aw_stream  # noqa: E402
from probe.metrics import insdel as metrics_insdel  # noqa: E402
from probe.model import ProbFn  # noqa: E402
from probe.paths import cum_nodes, spi_increments  # noqa: E402
from probe.quad import evaluate, rules  # noqa: E402

F64 = torch.float64


def gen(s):
    return C.generator(s, "cpu")


def cgen(s):
    return torch.Generator().manual_seed(s)


def valid(Z, atol=1e-12):
    assert torch.all(Z[0].abs() <= atol) and torch.all((Z[-1] - 1).abs() <= atol)
    assert torch.all(Z[1:] - Z[:-1] >= -atol)
    assert torch.all(Z >= -atol) and torch.all(Z <= 1 + atol)


# ─────────────────────────────────────────────────────────────────────────────
# T1 — tính hợp lệ của đường
# ─────────────────────────────────────────────────────────────────────────────

def test_T1_paths_valid_rgb():
    shape, N, g = (3, 8, 8), 12, 2
    lab = SM.grid_elem_labels(shape, g)
    x0 = torch.zeros(1, *shape)
    x = torch.rand(1, *shape) + 0.5
    dx = x - x0
    Zs = SM.spi_schedule(10.0, N, shape, gen(1))
    phi, _ = SM.phase_design("global", g * g, 4, cgen(2), g=g)
    Zp = SM.expand_patch(SM.patch_schedule(10.0, N, phi[1], gen(3)), lab, shape)
    for Z in (Zs, Zp, SM.reverse_nodes(Zs), SM.reverse_nodes(Zp), SM.mix(Zp, Zs, 0.5)):
        assert Z.shape == (N + 1,) + shape
        valid(Z)
        X = SM.nodes(x0, dx, Z)
        assert torch.allclose(X[0], x0[0]) and torch.allclose(X[-1], x[0], atol=1e-6)
    # lịch trình patch: 3 kênh của một pixel và mọi pixel trong patch dùng chung lịch trình
    assert torch.equal(Zp[:, 0], Zp[:, 1]) and torch.equal(Zp[:, 0], Zp[:, 2])
    assert torch.equal(Zp[:, 0, 0, 0], Zp[:, 0, 3, 3]) and not torch.equal(Zp[:, 0, 0, 0], Zp[:, 0, 7, 7])


def test_T1_grid_labels():
    lab = SM.grid_pixel_labels(224, 224, 7).view(224, 224)
    assert int(lab.max()) == 48 and torch.all(torch.bincount(lab.flatten()) == 32 * 32)
    assert int(lab[0, 31]) == 0 and int(lab[0, 32]) == 1 and int(lab[32, 0]) == 7
    assert torch.all(torch.bincount(SM.grid_pixel_labels(224, 224, 14)) == 16 * 16)
    p, q = SM.pair_index(3, "nbr4")
    assert len(p) == 12 and torch.all(p < q)


# ─────────────────────────────────────────────────────────────────────────────
# T2 — antithetic
# ─────────────────────────────────────────────────────────────────────────────

def test_T2_double_reversal_and_increment_reversal():
    Z = SM.spi_schedule(3.0, 16, (5,), gen(4))
    assert torch.allclose(SM.reverse_nodes(SM.reverse_nodes(Z)), Z, atol=1e-15)
    R = SM.reverse_nodes(Z)
    assert torch.allclose(R[1:] - R[:-1], (Z[1:] - Z[:-1]).flip(0), atol=1e-14)


def _toy_paths(d, N, s):
    Z = SM.spi_schedule(2.0, N, (d, 1, 1), gen(s))
    return Z, SM.reverse_nodes(Z)


def test_T2_antithetic_exact_additive_pairwise():
    """F = a·z + zᵀBz (cộng tính + cặp đa tuyến tính + tự bình phương): trung bình một cặp antithetic = a_i + ½Σ_j(B_ij + B_ji)
    CHÍNH XÁC (cầu phương trung điểm chính xác trên từng đoạn vì tích phân bậc nhất theo s)."""
    torch.manual_seed(0)
    d, N = 6, 10
    a = torch.randn(d, dtype=F64)
    Bm = torch.randn(d, d, dtype=F64)
    fn = lambda X: (X.flatten(1) @ a) + ((X.flatten(1) @ Bm) * X.flatten(1)).sum(1)
    expect = a + 0.5 * (Bm + Bm.T).sum(1)
    for s in range(5):
        Z, R = _toy_paths(d, N, s)
        A = rules(evaluate(fn, Z, 64, quads=("mid",)))[0]["canon_mid"].flatten()
        AR = rules(evaluate(fn, R, 64, quads=("mid",)))[0]["canon_mid"].flatten()
        assert torch.allclose((A + AR) / 2, expect, atol=1e-10)
        assert not torch.allclose(A, expect, atol=1e-3)                  # một đường đơn thì không chính xác


def test_T2_antithetic_interior_nonlinear_identical_pair():
    """F = (z1 − ½)(z2 − ½)² phản đối xứng dưới z → 1 − z: đường đảo cho attribution GIỐNG HỆT (tương quan cặp = +1) —
    tốn hai đường nhưng không thêm thông tin. Không được giả định antithetic luôn có tương quan âm."""
    fn = lambda X: (X[:, 0] - 0.5).flatten() * ((X[:, 1] - 0.5) ** 2).flatten()
    for s in range(3):
        Z, R = _toy_paths(2, 12, s)
        A = rules(evaluate(fn, Z, 64, quads=("mid",)))[0]["canon_mid"].flatten()
        AR = rules(evaluate(fn, R, 64, quads=("mid",)))[0]["canon_mid"].flatten()
        assert torch.allclose(A, AR, atol=1e-12)
    # và với F đối xứng (z1 − ½)(z2 − ½): A^R = −A (tương quan −1)
    fs = lambda X: ((X[:, 0] - 0.5) * (X[:, 1] - 0.5)).flatten()
    Z, R = _toy_paths(2, 12, 7)
    A = rules(evaluate(fs, Z, 64, quads=("mid",)))[0]["canon_mid"].flatten()
    AR = rules(evaluate(fs, R, 64, quads=("mid",)))[0]["canon_mid"].flatten()
    assert torch.allclose(A, -AR, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# T3 — thiết kế gán pha
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["global", "nbr"])
@pytest.mark.parametrize("Kmax", [2, 4, 16, 32])
def test_T3_balanced_prefix_coverage(kind, Kmax):
    g = 5
    P = g * g
    common = SM.common_phase_noise(P, Kmax, cgen(11))
    phi, info = SM.phase_design(kind, P, Kmax, cgen(12), g=g, common=common)
    assert phi.shape == (Kmax, P) and torch.all((phi >= 0) & (phi < 1))
    K = 2
    while K <= Kmax:
        assert torch.all(SM.arc_coverage(phi, common[0], K) == K), K
        K *= 2
    # tái lập theo seed
    phi2, _ = SM.phase_design(kind, P, Kmax, cgen(12), g=g, common=SM.common_phase_noise(P, Kmax, cgen(11)))
    assert torch.equal(phi, phi2)
    phi3, _ = SM.phase_design(kind, P, Kmax, cgen(13), g=g, common=SM.common_phase_noise(P, Kmax, cgen(14)))
    assert not torch.equal(phi, phi3)


def test_T3_nbr_relative_phase_spread():
    g, Kmax = 6, 16
    P = g * g
    _, info = SM.phase_design("nbr", P, Kmax, cgen(3), g=g)
    z = info["z"]
    p, q = SM.pair_index(g, "nbr4")
    assert torch.all(torch.remainder(z[p] - z[q], 4) == 2)
    b = SM.vdc(Kmax)
    for K in (4, 8, 16):
        for i in range(len(p)):
            vals = set(torch.remainder((z[p[i]] - z[q[i]]) * b[:K], Kmax).tolist())
            assert len(vals) == K // 2                              # K/2 độ lệch pha khác nhau, cách đều


def test_T3_fixed_indep_and_vdc():
    assert SM.vdc(8).tolist() == [0, 4, 2, 6, 1, 5, 3, 7]
    phi, _ = SM.phase_design("fixed", 9, 8, cgen(1), g=3)
    assert torch.all(phi == phi[0:1])
    phi, _ = SM.phase_design("indep", 9, 8, cgen(1), g=3)
    assert len(set(phi[:, 0].tolist())) == 8
    with pytest.raises(ValueError):
        SM.vdc(6)


def test_T3_lime_high_last():
    rank = torch.linspace(0, 1, 16, dtype=F64)
    phi, _ = SM.phase_design("lime", 16, 64, cgen(5), lime_rank=rank, tau=0.05)
    m = phi.mean(0)
    assert float(torch.corrcoef(torch.stack([m, rank]))[0, 1]) > 0.95     # hệ số cao → kích hoạt muộn
    assert torch.all((phi >= 0) & (phi < 1))


def test_T3_coverage_stats():
    tau = torch.tensor([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]], dtype=F64).repeat(1, 2)   # K = 3, P = 4 (lưới 2×2)
    c = SM.coverage_stats(tau, 2, 0.125)
    assert c["cover_third"] == pytest.approx(1.0) and c["deficit_third"] == pytest.approx(0.0)
    tau_sync = torch.full((4, 4), 0.5, dtype=F64)
    c2 = SM.coverage_stats(tau_sync, 2, 0.125)
    assert c2["nbr4_overlap"] == pytest.approx(1.0) and c2["nbr4_tie"] == pytest.approx(1.0) and c2["tau_spread"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# T4 — kiểm chứng phân phối biên (chẩn đoán, không phải chứng minh)
# ─────────────────────────────────────────────────────────────────────────────

def test_T4_anchor_rotation_preserves_dp_moments():
    alpha, N, d = 10.0, 16, 40000
    Zd = cum_nodes(spi_increments(alpha, N, (d,), gen(21)))
    delta = spi_increments(alpha, N, (d,), gen(22))
    phase = torch.rand(d, generator=cgen(23), dtype=F64)
    rot = SM.anchor_rotate(delta, phase)
    assert torch.equal(rot.argmax(0), (phase * N).floor().long())           # anchor đúng vị trí
    assert torch.allclose(rot.sum(0), torch.ones(d, dtype=F64))
    Zr = cum_nodes(rot)
    t = torch.linspace(0, 1, N + 1, dtype=F64)
    for Z in (Zd, Zr, SM.reverse_nodes(Zd)):
        assert torch.allclose(Z.mean(1), t, atol=0.01)
        v = t * (1 - t) / (alpha + 1)
        assert torch.allclose(Z.var(1)[1:-1], v[1:-1], rtol=0.1)
    # argmax của DP gốc đều trên {0..N−1}
    cnt = torch.bincount(delta.argmax(0), minlength=N).double() / d
    assert float((cnt - 1 / N).abs().max()) < 0.01
    # đối chứng: anchor CỐ ĐỊNH ở ô 0 đổi phân phối (bước nhảy lớn luôn ở đầu) → không còn là DP
    Zfix = cum_nodes(SM.anchor_rotate(delta, torch.zeros(d, dtype=F64)))
    assert float(Zfix[1].mean()) > 0.15 and float(Zd[1].mean()) < 0.08


@pytest.mark.parametrize("kind", ["global", "nbr", "fixed", "indep"])
def test_T4_per_path_phase_uniform_iid(kind):
    """Qua nhiều seed (π mới), pha của một (đường, patch) ~ U(0,1), và hai patch trong cùng đường không tương quan."""
    g, Kmax, R = 4, 8, 3000
    vals = torch.stack([SM.phase_design(kind, g * g, Kmax, cgen(1000 + r), g=g)[0] for r in range(R)])   # (R, Kmax, P)
    v = vals[:, 5, 3]
    assert abs(float(v.mean()) - 0.5) < 0.02 and abs(float(v.var()) - 1 / 12) < 0.01
    hist = torch.histc(v.float(), bins=10, min=0, max=1) / R
    assert float((hist - 0.1).abs().max()) < 0.03
    c = float(torch.corrcoef(torch.stack([vals[:, 5, 3], vals[:, 5, 4]]))[0, 1])
    assert abs(c) < 0.06


def test_T4_patch_schedule_exploration_matches_dp():
    from probe.e10 import d_dp_grid
    alpha, N, P = 10.0, 32, 20000
    phi = torch.rand(P, generator=cgen(31), dtype=F64)
    Zp = SM.patch_schedule(alpha, N, phi, gen(32))
    assert SM.d_hat_cols(Zp) == pytest.approx(d_dp_grid(alpha, N), rel=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# T5 — estimator, metric, hash/cache
# ─────────────────────────────────────────────────────────────────────────────

def test_T5_completeness_linear_exact_and_cubic_unbiased():
    shape = (3, 4, 4)
    a = torch.randn(1, *shape, dtype=F64, generator=cgen(1))
    fn = lambda X: (a * X).flatten(1).sum(1)
    x0 = torch.zeros(1, *shape, dtype=F64)
    x = torch.rand(1, *shape, dtype=F64, generator=cgen(2)) + 0.1
    phi, _ = SM.phase_design("nbr", 4, 4, cgen(3), g=2)
    lab = SM.grid_elem_labels(shape, 2)
    for Z in (SM.spi_schedule(10.0, 8, shape, gen(4)),
              SM.mix(SM.expand_patch(SM.patch_schedule(10.0, 8, phi[2], gen(5)), lab, shape), SM.spi_schedule(10.0, 8, shape, gen(6)), 0.5)):
        A, _ = aw_stream(fn, SM.nodes(x0, x - x0, Z), 1, gen(7), 64)
        assert torch.allclose(A, (a * (x - x0))[0], atol=1e-10)
    X = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=F64).view(4, 1, 1, 1)
    cub = lambda Z: (Z ** 3).flatten(1).sum(1)
    vals = [float(aw_stream(cub, X, 1, gen(100 + s), 64)[0].sum()) for s in range(2000)]
    assert abs(sum(vals) / len(vals) - 1.0) < 0.08                         # không chệch với tích phân trên đường gấp khúc


def test_T5_insdel_full_matches_metrics_insdel():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5)).double().eval()
    x = torch.rand(1, 3, 8, 8, dtype=F64)
    base = torch.zeros_like(x)
    A = torch.randn(3, 8, 8, dtype=F64)
    ref = metrics_insdel(ProbFn(net, 2), x, base, A, 8, 3)
    r = SM.insdel_full(net, 2, x, base, A, 8, 3, curves=True)
    assert r["diff"] == pytest.approx(ref, abs=1e-12)
    assert r["curves"]["ins_prob"].shape == (9,) and r["curves"]["frac"][-1] == pytest.approx(1.0)
    rs = SM.insdel_full(net, 2, x, base, A, 8, 3, signed=True)
    assert math.isfinite(rs["diff"])


def test_T5_lime_recovers_linear_patch_model():
    g, H = 2, 4
    pl = SM.grid_pixel_labels(H, H, g)
    w = torch.tensor([0.5, -1.0, 2.0, 0.1], dtype=F64)
    x0 = torch.zeros(1, 3, H, H, dtype=F64)
    x = torch.ones(1, 3, H, H, dtype=F64)
    cnt = torch.bincount(pl).double()

    def F(X):
        s = torch.zeros(X.shape[0], 4, dtype=F64).index_add_(1, pl, (X - x0).mean(1).flatten(1))
        return (s / cnt * w).sum(1)
    r = SM.lime_grid(F, x, x0, pl, 4, 64, cgen(9), ridge=1e-9, batch=16)
    assert torch.allclose(r["beta"], w, atol=1e-6)
    assert r["rank"].argsort().tolist() == w.argsort().tolist() and r["n_queries"] == 64


def test_T5_order_effect_detects_planted_pair():
    g, n = 3, 1000
    tau = torch.rand(n, g * g, generator=cgen(4), dtype=F64)
    Cp = torch.zeros(n, g * g, dtype=F64)
    Cp[:, 0] = (tau[:, 1] < tau[:, 0]).double()                             # credit của patch 0 phụ thuộc thứ tự với patch 1
    rows = SM.order_effect_by_distance(Cp, tau, g, 1.0, 5, cgen(5))
    assert [r["dist"] for r in rows] == [1, 2]
    assert rows[0]["abs_I_rel"] > 3 * rows[0]["null_abs_I_rel"]
    # sau khi khử phụ thuộc vào τ_0 (E[C_0 | τ_0] = τ_0), chỉ còn cặp (0, 1): I[0,1] = 2/3, các cặp khác ≈ 0
    R = SM.residualize_own_time(Cp, tau)
    I = SM._order_effect(R, tau, tau)
    assert float(I[0, 1]) == pytest.approx(2 / 3, abs=0.1)
    assert abs(float(I[0, 3])) < 0.1 and abs(float(I[0, 8])) < 0.1


def test_T5_order_effect_ignores_own_time_effect():
    """Credit chỉ phụ thuộc thời điểm của CHÍNH patch (không có tương tác) → không được báo là tác động thứ tự."""
    g, n = 3, 1000
    tau = torch.rand(n, g * g, generator=cgen(6), dtype=F64)
    Cp = 2 * tau - tau ** 3 + 0.1 * torch.randn(n, g * g, generator=cgen(7), dtype=F64)
    rows = SM.order_effect_by_distance(Cp, tau, g, 1.0, 10, cgen(8))
    for r in rows:
        assert r["abs_I_rel"] < 2 * r["null_abs_I_rel"]


def _cfg(debug=False):
    return C.load(os.path.join(ROOT, "config.yaml"), debug=debug)


def test_T5_settings_digests_and_cache_invalidation():
    from probe.sampling_benchmark import blocks, digests, estimate, method_paths, plan, settings
    cfg = _cfg()
    S = settings(cfg)
    assert S["alpha"] == float(cfg["paths"]["spi10"]["alpha"]) and S["N"] == int(cfg["paths"]["spi10"]["steps"])
    assert S["Kmax"] == max(S["K"])
    d0 = digests(cfg, S)
    cfg2 = copy.deepcopy(cfg)
    cfg2["sampling"]["K"] = [2, 4, 8]
    d2 = digests(cfg2, settings(cfg2))
    assert d2["primary"] != d0["primary"]                                    # đổi sampler → cache khối bị vô hiệu
    cfg3 = copy.deepcopy(cfg)
    cfg3["sampling"]["stats"]["bootstrap"] = 123
    cfg3["sampling"]["budget"]["minutes"] = 1
    assert digests(cfg3, settings(cfg3)) == d0                                # thống kê / ngân sách không đụng cache
    cfg4 = copy.deepcopy(cfg)
    cfg4["sampling"]["secondary"] = []
    assert digests(cfg4, settings(cfg4))["primary"] == d0["primary"]
    assert C.digest(cfg2) == C.digest(cfg)                                    # hash pilot P0–P3 không đổi theo mục sampling
    bad = copy.deepcopy(cfg)
    bad["sampling"]["K"] = [3, 6]
    with pytest.raises(SystemExit):
        settings(bad)
    # S1 dùng lại nửa đầu S0, S3 dùng lại nửa đầu S2; mỗi tiền tố K của S1/S3 gồm K/2 cặp
    s0, s1 = method_paths("S0", 7, 0.0, 16), method_paths("S1", 7, 0.0, 16)
    assert set(s1[:8]) - {("spiR", j) for j in range(4)} == set(s0[:4])
    assert method_paths("S3", 7, 0.0, 16)[0] == method_paths("S2", 7, 0.0, 16)[0]
    est = estimate(cfg, S, 0.002, 0.0007)
    assert est["blocks"]["primary"]["unique_paths_per_seed"] == 16 + 8 + 16 + 8 + 16 + 16
    pl = plan(S, est)
    assert set(pl["secondary_run"]) | set(pl["secondary_skipped"]) == {b["name"] for b in blocks(S)[1:]}


def test_T5_debug_settings():
    from probe.sampling_benchmark import settings
    S = settings(_cfg(debug=True))
    assert S["N"] == 8 and S["K"] == [2, 4] and S["seeds"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# A / B / C
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["global", "nbr"])
def test_A_pixel_phase_coverage_and_uniform(kind):
    shape, Kmax = (3, 6, 6), 16
    n = 3 * 36
    common = SM.common_phase_noise(n, Kmax, cgen(40))
    phi, info = SM.pixel_phase_design(kind, shape, Kmax, cgen(41), common=common)
    assert phi.shape == (Kmax, n) and torch.all((phi >= 0) & (phi < 1))
    for K in (2, 4, 8, 16):
        assert torch.all(SM.arc_coverage(phi, common[0], K) == K)
    if kind == "nbr":                                                       # pixel kề ngang/dọc: z_i − z_j ≡ 2 (mod 4)
        z = info["z"].view(shape)
        assert torch.all(torch.remainder(z[:, :, 1:] - z[:, :, :-1], 4) == 2)
        assert torch.all(torch.remainder(z[:, 1:, :] - z[:, :-1, :], 4) == 2)
    vals = torch.stack([SM.pixel_phase_design(kind, (1, 2, 2), 8, cgen(500 + r))[0][3, 1] for r in range(3000)])
    assert abs(float(vals.mean()) - 0.5) < 0.02 and abs(float(vals.var()) - 1 / 12) < 0.01


def test_A_pixel_schedule_same_increments_as_S0():
    """S2p dùng đúng số gia của S0 cùng khoá sinh: mỗi phần tử chỉ là phép xoay vòng của dãy số gia S0, anchor đúng ô thiết kế."""
    shape, N = (3, 4, 4), 16
    phi, _ = SM.pixel_phase_design("global", shape, 4, cgen(3))
    Z0 = SM.spi_schedule(10.0, N, shape, gen(9))
    Zp = SM.pixel_schedule(10.0, N, shape, phi[1], gen(9))
    valid(Zp)
    d0 = (Z0[1:] - Z0[:-1]).reshape(N, -1)
    dp = (Zp[1:] - Zp[:-1]).reshape(N, -1)
    assert torch.allclose(d0.sort(0).values, dp.sort(0).values, atol=1e-12)
    assert torch.equal(dp.argmax(0), (phi[1] * N).floor().long())
    assert not torch.allclose(Z0, Zp)


def test_A_pixel_schedule_preserves_dp_moments():
    alpha, N = 10.0, 16
    shape = (1, 100, 200)
    n = 20000
    phi = torch.rand(n, generator=cgen(50), dtype=F64)
    Z = SM.pixel_schedule(alpha, N, shape, phi, gen(51)).reshape(N + 1, -1)
    t = torch.linspace(0, 1, N + 1, dtype=F64)
    assert torch.allclose(Z.mean(1), t, atol=0.01)
    assert torch.allclose(Z.var(1)[1:-1], (t * (1 - t) / (alpha + 1))[1:-1], rtol=0.1)


def test_B_eta_one_is_exactly_spi():
    shape, N = (3, 4, 4), 8
    Zs = SM.spi_schedule(10.0, N, shape, gen(1))
    phi, _ = SM.phase_design("indep", 4, 4, cgen(2), g=2)
    Zp = SM.expand_patch(SM.patch_schedule(10.0, N, phi[0], gen(3)), SM.grid_elem_labels(shape, 2), shape)
    assert torch.equal(SM.mix(Zp, Zs, 1.0), Zs)
    assert torch.equal(SM.mix(Zp, Zs, 0.0), Zp)


def test_ABC_registry_and_estimates():
    from probe.sampling_benchmark import (blocks, digests, estimate, estimate_quad, label, method_paths, operator,
                                          settings)
    from probe.sampling_report import eta_of, holm_family
    cfg = _cfg()
    S = settings(cfg)
    names = [b["name"] for b in blocks(S)]
    assert {"pixel", "eta025", "eta05", "eta075", "eta1"} <= set(names)
    pix = {"name": "pixel", "grid": S["grid"], "eta": S["eta"]}
    for m in ("S2p", "S3p", "S4p"):
        assert operator(m, 7, 0.0) == "spi" and label(m, pix, S) == m
    assert method_paths("S3p", 7, 0.0, 8)[:2] == [("pix", "global", 0), ("pixR", "global", 0)]
    est = estimate(cfg, S, 0.002, 0.0007)
    assert est["blocks"]["primary"]["unique_paths_per_seed"] == 80
    assert est["blocks"]["pixel"]["unique_paths_per_seed"] == 16 + 8 + 16
    q = estimate_quad(S, 0.002, 0.0007)
    assert q["grad_per_image"] == 48 * S["N"] * sum(S["quad_check"]["M"])
    # thêm khối phụ / quad_check không làm mất cache của lượt chính
    cfg2 = copy.deepcopy(cfg)
    cfg2["sampling"]["secondary"] = cfg2["sampling"]["secondary"][:1]
    cfg2["sampling"]["quad_check"]["M"] = [1, 2]
    d0, d2 = digests(cfg, S), digests(cfg2, settings(cfg2))
    assert d0["primary"] == d2["primary"] and d0["ref"] == d2["ref"] and d0["quad"] != d2["quad"]
    assert eta_of("S2@g7e0.25", 7, 0.0) == 0.25 and eta_of("S2", 7, 0.0) == 0.0 and eta_of("S2@g14e0", 7, 0.0) is None
    assert holm_family("vs_same_operator_S6", "S2@g7e0.5", "S6@g7e0.5", "insdel_blur") == "B_eta"
    assert holm_family("vs_same_operator_S6", "S2@g14e0", "S6@g14e0", "insdel_blur") is None
    assert holm_family("vs_S0", "S4p", "S0", "insdel_black") == "A_pixel"
    assert holm_family("vs_S0", "S2", "S0", "insdel_blur") == "primary"


# ─────────────────────────────────────────────────────────────────────────────
# thống kê của báo cáo
# ─────────────────────────────────────────────────────────────────────────────

def test_stats_holm_boot_and_verdict():
    import numpy as np
    import pandas as pd
    from probe.sampling_report import aggregate, boot_stats, holm, verdict
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    s = boot_stats([0.1] * 10 + [0.2] * 10, 500, 0)
    assert s["lo95"] > 0 and s["frac_improved"] == 1.0 and s["sign_p"] < 1e-4
    rng = np.random.default_rng(0)
    rows = []
    for i in range(16):
        b0 = rng.normal(0.4, 0.1)
        for m, shift in (("S0", 0.0), ("S1", 0.0), ("S2", 0.05), ("S3", 0.0), ("S4", 0.0)):
            for K in (2, 4):
                rows.append({"image": f"im{i}", "method": m, "K": K, "n_seeds": 2, "B_grad": K * 64,
                             "insdel_blur": b0 + shift + rng.normal(0, 0.002), "insdel_black": b0 + shift + rng.normal(0, 0.002)})
    pi = pd.DataFrame(rows)
    st = {"bootstrap": 500, "seed": 1, "alpha": 0.05, "min_effect": 0.01, "min_images": 16, "win_frac": 0.5}
    ag = aggregate(pi, st)
    V = verdict(ag, st, 16, False, [2, 4])
    assert V["category"] == "Positive" and V["positive_methods"] == ["S2"]
    V2 = verdict(ag, st, 16, True, [2, 4])
    assert V2["category"] == "Inconclusive"
