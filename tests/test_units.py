"""
Unit test không dùng ResNet (CPU, vài giây). Test A–D theo task, cộng các test cho bẫy, tín hiệu, coupling DP,
xử lý ‖g‖² ≈ 0 của IDGI, Guided IG và BlurIG trên một CNN ReLU nhỏ.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe.components import Signal, Trap, direction, kappa_for_eta, predicted_attenuation
from probe.config import generator
from probe.model import TargetFn
from probe.paths import (blur_nodes, coarsen, cum_nodes, gaussian_blur, guided_nodes, spi_increments,
                         spi_nodes_from, straight_nodes)
from probe.quad import evaluate, idgi_diagnostics, rules, shift, unshift

F64 = torch.float64


def lin_fn(a):
    return lambda X: (a * X).flatten(1).sum(1)


# ─────────────────────────────────────────────────────────────────────────────
# Test A — mô hình cộng tính tuyến tính: canonical đúng a_i(z_i − z′_i) trên MỌI đường
# ─────────────────────────────────────────────────────────────────────────────

def test_A_linear_any_path():
    torch.manual_seed(0)
    a = torch.randn(1, 3, 4, 4, dtype=F64)
    x0 = torch.zeros(1, 3, 4, 4, dtype=F64)
    x = torch.rand(1, 3, 4, 4, dtype=F64) + 0.1
    fn = lin_fn(a)
    star = (a * (x - x0))[0]
    gen = generator(1, "cpu")
    paths = {
        "straight": straight_nodes(x, x0, 7),
        "spi": spi_nodes_from(spi_increments(3.0, 8, x.shape[1:], gen), x0, x - x0),
    }
    Xg, fgd, gg, gap = guided_nodes(fn, x, x0, 6, 0.25, 0.02, 64)
    for name, X in paths.items():
        ev = evaluate(fn, X, 64, quads=("mid", "rand"), gen=generator(2, "cpu"))
        A, _ = rules(ev)
        for r in ("canon_left", "canon_mid", "canon_rand"):
            assert torch.allclose(A[r].double(), star, atol=1e-6), (name, r)
    ev = evaluate(fn, Xg, 64, quads=("rand",), gen=generator(3, "cpu"), f_nodes=fgd, g_nodes=gg)
    A, _ = rules(ev)
    assert torch.allclose(A["canon_left"].double(), star, atol=1e-6)
    assert gap < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Test B — z³ với một bước nhảy: Riemann trái sai lớn, cầu phương ngẫu nhiên hội tụ về đúng
# ─────────────────────────────────────────────────────────────────────────────

def test_B_cubic_jump():
    X = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=F64).view(4, 1, 1, 1)
    fn = lambda Z: (Z ** 3).flatten(1).sum(1)
    ev = evaluate(fn, X, 64, quads=("mid",))
    A, _ = rules(ev)
    assert abs(float(A["canon_left"].sum()) - 0.0) < 1e-12        # đúng là 1
    assert abs(float(A["canon_mid"].sum()) - 0.75) < 1e-12
    vals = []
    for s in range(2000):
        ev = evaluate(fn, X, 64, quads=("rand",), gen=generator(s, "cpu"))
        vals.append(float(rules(ev)[0]["canon_rand"].sum()))
    m = sum(vals) / len(vals)
    assert abs(m - 1.0) < 0.08, m                                  # SE ≈ 0.02


# ─────────────────────────────────────────────────────────────────────────────
# Test C — IDGI và đặc trưng không đổi
# ─────────────────────────────────────────────────────────────────────────────

def test_C_idgi_unchanged_feature():
    fn = lambda Z: Z.flatten(1).sum(1)
    x0 = torch.zeros(1, 2, 1, 1, dtype=F64)
    x = torch.tensor([1.0, 0.0], dtype=F64).view(1, 2, 1, 1)
    A, _ = rules(evaluate(fn, straight_nodes(x, x0, 10), 64, quads=()))
    assert torch.allclose(A["canon_left"].flatten().double(), torch.tensor([1.0, 0.0], dtype=F64))
    assert torch.allclose(A["idgi"].flatten().double(), torch.tensor([0.5, 0.5], dtype=F64))


# ─────────────────────────────────────────────────────────────────────────────
# Test D — tín hiệu cộng tính phân tán
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m", [1, 10, 100, 1000])
def test_D_diffuse_signal(m):
    G, B = 2.0, 1.0
    a = torch.full((1, m + 1, 1, 1), B / m, dtype=F64)
    a[0, 0] = G
    fn = lin_fn(a)
    x0 = torch.zeros(1, m + 1, 1, 1, dtype=F64)
    x = torch.ones(1, m + 1, 1, 1, dtype=F64)
    A, _ = rules(evaluate(fn, straight_nodes(x, x0, 5), 64, quads=()))
    assert abs(float(A["canon_left"].flatten()[1:].sum()) - B) < 1e-9
    pred = (G + B) * (B * B / m) / (G * G + B * B / m)
    assert abs(float(A["idgi"].flatten()[1:].sum()) - pred) < 1e-9


def test_D_idgi_group_credit_vanishes():
    G, B = 2.0, 1.0
    vals = [(G + B) * (B * B / m) / (G * G + B * B / m) for m in (1, 10, 100, 1000, 10000)]
    assert all(u > v for u, v in zip(vals, vals[1:])) and vals[-1] < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# IDGI: bước có ‖g‖² = 0 bị bỏ minh bạch
# ─────────────────────────────────────────────────────────────────────────────

def test_idgi_zero_gradient_step_is_reported():
    fn = lambda Z: torch.relu(Z - 0.5).flatten(1).sum(1)
    x0 = torch.zeros(1, 1, 1, 1, dtype=F64)
    x = torch.ones(1, 1, 1, 1, dtype=F64)
    A, info = rules(evaluate(fn, straight_nodes(x, x0, 1), 64, quads=()))
    assert info["idgi"]["n_dropped"] == 1
    assert abs(info["idgi"]["dropped_dF"] - 0.5) < 1e-12
    assert float(A["idgi"].sum()) == 0.0            # credit không bị bịa ra; residual = 0.5


def test_idgi_raw_equals_idgi_for_uniform_std():
    torch.manual_seed(1)
    a = torch.randn(1, 3, 2, 2, dtype=F64)
    fn = lambda Z: ((a * Z) ** 2).flatten(1).sum(1)
    x0 = torch.zeros(1, 3, 2, 2, dtype=F64)
    x = torch.rand(1, 3, 2, 2, dtype=F64)
    std = torch.full((3, 1, 1), 0.3, dtype=F64)
    A, _ = rules(evaluate(fn, straight_nodes(x, x0, 9), 64, quads=()), std=std)
    assert torch.allclose(A["idgi"], A["idgi_raw"], atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# coupling DP giữa các lưới
# ─────────────────────────────────────────────────────────────────────────────

def test_dp_coupling_subsamples_nodes():
    d = spi_increments(5.0, 16, (2, 3, 3), generator(4, "cpu"))
    Zf = cum_nodes(d)
    Zc = cum_nodes(coarsen(d, 4))
    assert torch.allclose(Zc, Zf[::4], atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# tín hiệu và bẫy
# ─────────────────────────────────────────────────────────────────────────────

def test_signal_grad_matches_autograd_and_shift_roundtrip():
    torch.manual_seed(2)
    x0 = torch.zeros(1, 3, 4, 4, dtype=F64)
    R = torch.zeros(1, 3, 4, 4); R[..., 1:3, 1:3] = 1
    sig = Signal(R.double(), x0, 0.7)
    P = torch.rand(5, 3, 4, 4, dtype=F64, requires_grad=True)
    (g,) = torch.autograd.grad(sig.value(P).sum(), P)
    assert torch.allclose(g, sig.grad(P).expand_as(g))
    fn = lambda Z: (Z ** 2).flatten(1).sum(1)
    ev = evaluate(fn, straight_nodes(torch.rand(1, 3, 4, 4, dtype=F64), x0, 6), 64, quads=("rand",), gen=generator(5, "cpu"))
    back = unshift(shift(ev, sig), sig)
    assert torch.allclose(back.g, ev.g) and torch.allclose(back.f, ev.f)


@pytest.mark.parametrize("rho", [0.03, float("inf")])
def test_trap_on_line(rho):
    torch.manual_seed(3)
    x0 = torch.zeros(1, 3, 6, 6, dtype=torch.float32)
    x = torch.rand(1, 3, 6, 6) + 0.2
    dx = x - x0
    u = direction("high", dx, generator(6, "cpu"), 2.0)
    tr = Trap(x0, dx, u, 5.0, rho)
    for s in (0.0, 0.25, 0.5, 0.8):
        p = (x0 + s * dx).requires_grad_(True)
        h = tr.value(p)
        assert abs(float(h)) < 1e-5
        (g,) = torch.autograd.grad(h.sum(), p)
        assert torch.allclose(g, 5.0 * math.sin(math.pi * s) ** 2 * u, atol=1e-5)
    off = x0 + 0.5 * dx + 0.3 * float(dx.norm()) * u
    assert abs(float(tr.value(off))) > 0


def test_trap_ig_change_and_eta():
    torch.manual_seed(4)
    x0 = torch.zeros(1, 3, 6, 6, dtype=F64)
    x = torch.rand(1, 3, 6, 6, dtype=F64) + 0.2
    dx = x - x0
    u = direction("high", dx, generator(7, "cpu"), 2.0)
    fn = lambda Z: torch.tanh(Z).flatten(1).sum(1)
    X = straight_nodes(x, x0, 200)
    ev = evaluate(fn, X, 512, quads=())
    t = torch.linspace(0, 1, 201, dtype=F64)[:-1]
    kap = kappa_for_eta(1.0, ev.g, t)
    tr = Trap(x0, dx, u, kap, 0.05)
    gh = tr.grad(X[:-1])
    ratio = gh.flatten(1).norm(dim=1).pow(2).mean().sqrt() / ev.g.flatten(1).norm(dim=1).pow(2).mean().sqrt()
    assert abs(float(ratio) - 1.0) < 1e-6
    A0, _ = rules(ev)
    A1, _ = rules(shift(ev, tr))
    d = (A1["canon_left"] - A0["canon_left"]).double()
    expect = (kap / 2) * (u * dx)[0]
    assert float((d - expect).norm() / expect.norm()) < 1e-6
    assert predicted_attenuation(1 / 11, float("inf")) == 1.0
    assert predicted_attenuation(1 / 11, 0.03) < predicted_attenuation(1 / 11, 0.3)


# ─────────────────────────────────────────────────────────────────────────────
# CNN ReLU nhỏ: Guided IG tới đích, IDGI completeness khi không bỏ bước, BlurIG đầu-cuối
# ─────────────────────────────────────────────────────────────────────────────

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        h = torch.relu(self.c2(torch.relu(self.c1(x))))
        return self.fc(h.mean(dim=(2, 3)))


@pytest.fixture(scope="module")
def toy():
    torch.manual_seed(5)
    net = Toy().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, 3, 16, 16)
    return TargetFn(net, 0), x, torch.full_like(x, -1.5)


def test_guided_and_idgi_completeness(toy):
    fn, x, x0 = toy
    X, f, g, gap = guided_nodes(fn, x, x0, 32, 0.25, 0.02, 64)
    assert gap < 1e-5
    ev = evaluate(fn, X, 64, quads=("rand",), gen=generator(8, "cpu"), f_nodes=f, g_nodes=g)
    A, info = rules(ev)
    D = float(f[-1] - f[0])
    if info["idgi"]["n_dropped"] == 0:
        assert abs(float(A["idgi"].double().sum()) - D) < 1e-4 * max(1.0, abs(D))


def test_blur_nodes_endpoints(toy):
    fn, x, x0 = toy
    X = blur_nodes(x, 12, 50.0)
    assert torch.allclose(X[0:1], gaussian_blur(x, 50.0)) and torch.allclose(X[-1:], x)


def test_diagnostics_shapes(toy):
    fn, x, x0 = toy
    ev = evaluate(fn, straight_nodes(x, x0, 16), 64, quads=())
    mask = torch.zeros(16, 16, dtype=torch.bool); mask[4:12, 4:12] = True
    d = idgi_diagnostics(ev, mask)
    assert d["q"].shape[0] == 16 and 0 <= float(d["q"].min()) and float(d["q"].max()) <= 1 + 1e-9
    assert 0 <= d["P_M"] <= 1 + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# E1–E3: cầu phương phân tầng, A/W, phân rã, toy chính xác
# ─────────────────────────────────────────────────────────────────────────────

from probe.mechanism import aw, decomp, nested_pick, strat_points, summ, toy_exact  # noqa: E402


def test_strat_points_and_aw_linear():
    torch.manual_seed(7)
    a = torch.randn(1, 3, 4, 4, dtype=F64)
    fn = lin_fn(a)
    x0 = torch.zeros(1, 3, 4, 4, dtype=F64)
    x = torch.rand(1, 3, 4, 4, dtype=F64) + 0.1
    X = spi_nodes_from(spi_increments(3.0, 8, x.shape[1:], generator(9, "cpu")), x0, x - x0)
    P, s = strat_points(X, 4, generator(10, "cpu"))
    assert P.shape[0] == 8 * 4 and float(s.min()) >= 0 and float(s.max()) <= 1
    assert torch.all((s * 4).floor() == torch.arange(4, dtype=F64).view(1, 4))     # một điểm mỗi tầng
    from probe.quad import fg
    _, G = fg(fn, P, 64)
    A, W = aw(G, X[1:] - X[:-1], 4)
    assert torch.allclose(A, (a * (x - x0))[0], atol=1e-12)
    assert torch.allclose(W, (a.abs() * (x - x0).abs())[0], atol=1e-12)            # đường đơn điệu: W = Σ|a||Δx|


def test_nested_pick_strata():
    ix = nested_pick(5, 16, 4, generator(11, "cpu"), "cpu")
    assert ix.shape == (5, 4)
    assert torch.all(ix // 4 == torch.arange(4).view(1, 4))


def test_decomposition_nonnegative_and_exact():
    torch.manual_seed(8)
    m3 = torch.zeros(3, 4, 4, dtype=torch.bool); m3[:, :2] = True
    fn = lambda Z: torch.sin(3 * Z).flatten(1).sum(1) * (Z[:, :1] ** 2).flatten(1).sum(1)
    x0 = torch.zeros(1, 3, 4, 4, dtype=F64)
    x = torch.rand(1, 3, 4, 4, dtype=F64) + 0.5
    from probe.quad import fg
    rs, Asum = [], None
    for p in range(6):
        X = spi_nodes_from(spi_increments(2.0, 8, x.shape[1:], generator(20 + p, "cpu")), x0, x - x0)
        P, _ = strat_points(X, 2, generator(40 + p, "cpu"))
        _, G = fg(fn, P, 64)
        A, W = aw(G, X[1:] - X[:-1], 2)
        rs.append(summ(A, W, m3))
        Asum = A if Asum is None else Asum + A
    D = float(fn(x) - fn(x0))
    d = decomp(rs, Asum / 6, D, m3)
    for k in ("within", "between", "across", "within_in", "between_in"):
        assert d[k] >= -1e-12, k
    assert abs(d["EW"] - abs(D) - (d["within"] + d["between"] + d["across"] + d["resid"])) < 1e-9


def test_toy_exact_matches_fine_quadrature():
    g = torch.Generator().manual_seed(3)
    d = 5
    b = torch.randn(d, generator=g, dtype=F64)
    Cm = torch.randn(d, d, generator=g, dtype=F64); Cm = (Cm + Cm.T) / 2; Cm.fill_diagonal_(0)
    Z = cum_nodes(spi_increments(4.0, 6, (d,), g)).unsqueeze(0) * (0.5 + torch.rand(d, generator=g, dtype=F64))
    A, W = toy_exact(Z, b, Cm)
    t = torch.linspace(0, 1, 20001, dtype=F64)[:-1] + 0.5 / 20000                     # midpoint rất mịn
    dZ = Z[0, 1:] - Z[0, :-1]
    gz = b + (Z[0, :-1].unsqueeze(1) + t.view(1, -1, 1) * dZ.unsqueeze(1)) @ Cm.T    # (N, T, d)
    Wn = (gz.abs().mean(1) * dZ.abs()).sum()
    An = (gz.mean(1) * dZ).sum(0)
    assert torch.allclose(A[0], An, atol=1e-8) and abs(float(W[0] - Wn)) < 1e-6
    D = float(b @ Z[0, -1] + 0.5 * Z[0, -1] @ Cm @ Z[0, -1])
    assert abs(float(A[0].sum()) - D) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# E4: tách mode chính xác, exposure trên hàm tuyến tính, hồ sơ trạng thái
# ─────────────────────────────────────────────────────────────────────────────

from probe.cross import MODES, exposure, mode_split, state_rows  # noqa: E402


def test_mode_split_exact():
    G = torch.randn(4, 3, 16, 16, dtype=F64)
    ms = mode_split(G, 2.0)
    assert torch.allclose(sum(ms[m] for m in MODES), G, atol=1e-12)
    assert torch.allclose(ms["cd_low"].sum(1), torch.zeros(4, 16, 16, dtype=F64), atol=1e-12)


def test_exposure_linear_and_state():
    torch.manual_seed(9)
    a = torch.randn(1, 3, 8, 8, dtype=F64)
    fn = lin_fn(a)
    x0 = torch.zeros(1, 3, 8, 8, dtype=F64)
    x = torch.rand(1, 3, 8, 8, dtype=F64) + 0.1
    from probe.mechanism import strat_points
    from probe.quad import fg
    X = spi_nodes_from(spi_increments(3.0, 8, x.shape[1:], generator(12, "cpu")), x0, x - x0)
    P, _ = strat_points(X, 2, generator(13, "cpu"))
    _, G = fg(fn, P, 64)
    e = exposure(G, X, 2, 1.0, chunk=3)
    assert torch.allclose(e["A"], (a * (x - x0))[0], atol=1e-12)
    assert abs(e["len"] - float((x - x0).abs().sum())) < 1e-9 and abs(e["chord_l1"] - e["len"]) < 1e-9
    assert abs(float(e["segW"].sum()) - float(e["W"].sum())) < 1e-9
    for m in MODES:                                     # gradient hằng: W_mode = Σ|mode(a)||Δx|
        ref = (mode_split(a, 1.0)[m].abs() * (x - x0).abs())[0]
        assert torch.allclose(e["Wm"][m], ref, atol=1e-10), m
    rs = state_rows({"image": "t"}, e, 5, [0.0, 0.05, 0.1, 10.0])
    assert abs(sum(r["share"] for r in rs if r["axis"] == "tau") - 1) < 1e-9
    assert abs(sum(r["share"] for r in rs if r["axis"] == "dist") - 1) < 1e-9
    Xl = straight_nodes(x, x0, 6)
    P, _ = strat_points(Xl, 2, generator(14, "cpu"))
    _, G = fg(fn, P, 64)
    assert float(exposure(G, Xl, 2, 1.0)["dist"].abs().max()) < 1e-9          # đường thẳng: khoảng cách 0


# ─────────────────────────────────────────────────────────────────────────────
# E5: lịch trình, hoán vị, hình học
# ─────────────────────────────────────────────────────────────────────────────

from probe.e5 import geom_dist, perm_index, permute, schedule, sim  # noqa: E402


@pytest.mark.parametrize("kind", ["bin", "pixel", "block"])
def test_permutation_valid(kind):
    torch.manual_seed(10)
    x0 = torch.zeros(1, 3, 32, 32, dtype=F64)
    x = torch.rand(1, 3, 32, 32, dtype=F64)
    x[0, 1, :4, :4] = 0                                             # một số phần tử Δx = 0
    dx = x - x0
    Z = cum_nodes(spi_increments(3.0, 10, dx.shape[1:], generator(15, "cpu")))
    X = x0 + Z * dx
    Zs = schedule(X, x0, dx)
    assert torch.allclose(x0 + Zs * dx, X, atol=1e-12)
    ix = perm_index(kind, dx, torch.Generator().manual_seed(1), nbins=5, block=8)
    assert torch.equal(ix.sort().values, torch.arange(ix.numel()))   # là một hoán vị
    Xp = x0 + permute(Zs, ix) * dx
    assert torch.allclose(Xp[0], x0[0]) and torch.allclose(Xp[-1], x[0])
    assert bool((Xp[1:] - Xp[:-1]).mul(dx.sign()).ge(-1e-12).all())  # đơn điệu từng toạ độ
    if kind == "bin":
        n = 32 * 32
        assert torch.equal(ix // n, torch.arange(3 * n) // n)          # không đổi kênh


def test_geom_dist_linear_in_lambda():
    x0 = torch.zeros(1, 3, 8, 8, dtype=F64)
    x = torch.rand(1, 3, 8, 8, dtype=F64) + 0.1
    Zs = cum_nodes(spi_increments(3.0, 16, x.shape[1:], generator(16, "cpu")))
    t = torch.linspace(0, 1, 17, dtype=F64).view(-1, 1, 1, 1)
    d1 = geom_dist(x0 + Zs * (x - x0))
    for lam in (0.0, 0.3, 0.7):
        assert abs(geom_dist(x0 + ((1 - lam) * t + lam * Zs) * (x - x0)) - lam * d1) < 1e-10
    assert geom_dist(straight_nodes(x, x0, 9)) < 1e-12


def test_sim_uniform_scaling_confound():
    torch.manual_seed(11)
    Wref = torch.rand(16, 16, dtype=F64) + 0.1
    a = sim(0.4 * Wref, 0.8 * Wref, Wref, Wref, Wref)
    assert a["spearman_avoided_raw"] > 0.999                        # confound: giảm đều vẫn cho tương quan 1
    assert abs(a["c_m"] - 0.4) < 1e-12 and abs(a["c_n"] - 0.8) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# E6: chia sẻ lịch trình, chuyển lịch trình
# ─────────────────────────────────────────────────────────────────────────────

from probe.e6 import group_index, shared_schedule, transfer  # noqa: E402


@pytest.mark.parametrize("kind,n", [("indep", 3 * 16 * 16), ("rgb", 16 * 16), ("block4", 16), ("block8", 4), ("global", 1)])
def test_group_index(kind, n):
    lab, k = group_index(kind, (3, 16, 16))
    assert k == n and int(lab.max()) == n - 1 and lab.numel() == 3 * 16 * 16


def test_shared_schedule_global_is_straight_line():
    x0 = torch.zeros(1, 3, 16, 16, dtype=F64)
    x = torch.rand(1, 3, 16, 16, dtype=F64) + 0.1
    Z, n = shared_schedule("global", 5.0, 12, (3, 16, 16), generator(17, "cpu"))
    assert n == 1 and torch.allclose(Z, Z[:, :1, :1, :1].expand_as(Z))
    assert geom_dist(x0 + Z * (x - x0)) < 1e-12
    Zr, _ = shared_schedule("rgb", 5.0, 12, (3, 16, 16), generator(18, "cpu"))
    assert torch.equal(Zr[:, 0], Zr[:, 1]) and not torch.equal(Zr[:, :, 0, 0], Zr[:, :, 0, 1])
    assert torch.all(Zr[0] == 0) and torch.all(Zr[-1] == 1)


def test_transfer_rank_matching():
    torch.manual_seed(12)
    Zd = cum_nodes(spi_increments(3.0, 6, (3, 8, 8), generator(19, "cpu")))
    ad, at = torch.rand(3, 8, 8, dtype=F64), torch.rand(3, 8, 8, dtype=F64)
    Zt = transfer(Zd, ad, at)
    for c in range(3):
        k = int(torch.argmax(at[c]).item())
        kd = int(torch.argmax(ad[c]).item())
        assert torch.equal(Zt[:, c].flatten(1)[:, k], Zd[:, c].flatten(1)[:, kd])      # hạng cao nhất ↔ hạng cao nhất
        assert torch.equal(Zt[:, c].flatten(1).sort(1).values, Zd[:, c].flatten(1).sort(1).values)


# ─────────────────────────────────────────────────────────────────────────────
# E7: nhóm cùng cỡ; phản ví dụ giải tích (cùng biên, khác phụ thuộc → khác attribution)
# ─────────────────────────────────────────────────────────────────────────────

from probe.e7 import group_labels  # noqa: E402


@pytest.mark.parametrize("kind,size", [("block4", 16), ("rand16", 16), ("block8", 64), ("rand64", 64)])
def test_group_labels_equal_size(kind, size):
    lab, n = group_labels(kind, (3, 32, 32), torch.Generator().manual_seed(2))
    assert n == 32 * 32 // size
    counts = torch.bincount(lab, minlength=n)
    assert torch.all(counts == 3 * size)                                      # mỗi nhóm: size pixel × 3 kênh
    assert torch.equal(lab[:1024], lab[1024:2048]) and torch.equal(lab[:1024], lab[2048:])   # 3 kênh chung nhóm


def test_rand_groups_are_spatially_scattered():
    lab, _ = group_labels("rand16", (1, 32, 32), torch.Generator().manual_seed(3))
    g = lab.view(32, 32)
    same_right = float((g[:, 1:] == g[:, :-1]).float().mean())
    lab_b, _ = group_labels("block4", (1, 32, 32), torch.Generator().manual_seed(3))
    gb = lab_b.view(32, 32)
    assert same_right < 0.05 and float((gb[:, 1:] == gb[:, :-1]).float().mean()) > 0.7


def test_shared_vs_independent_schedule_counterexample():
    """F = z1² z2, (0,0) → (1,1): chung lịch trình (đường chéo) → (2/3, 1/3); độc lập (bật lần lượt, trung bình hai thứ tự) → (1/2, 1/2)."""
    fn = lambda Z: (Z[:, 0] ** 2 * Z[:, 1]).flatten(1).sum(1)
    from probe.mechanism import aw, strat_points
    from probe.quad import fg

    def A_of(X):
        P, _ = strat_points(X, 1024, generator(20, "cpu"))       # sd của ước lượng phân tầng ≈ 1/(3·M^1.5) ≈ 1e-5
        _, G = fg(fn, P, 4096)
        return aw(G, X[1:] - X[:-1], 1024)[0].flatten()

    diag = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=F64).view(2, 2, 1, 1)
    assert torch.allclose(A_of(diag), torch.tensor([2 / 3, 1 / 3], dtype=F64), atol=5e-4)
    o1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=F64).view(3, 2, 1, 1)
    o2 = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=F64).view(3, 2, 1, 1)
    assert torch.allclose((A_of(o1) + A_of(o2)) / 2, torch.tensor([0.5, 0.5], dtype=F64), atol=5e-4)


# ─────────────────────────────────────────────────────────────────────────────
# E8: Holm, luật phán quyết trên dữ liệu tổng hợp
# ─────────────────────────────────────────────────────────────────────────────

def test_holm_and_verdicts():
    import numpy as np
    import pandas as pd
    from probe.e8_report import analyse, holm
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        base = rng.normal(0.4, 0.05)
        for fam, shift in (("a", -0.05), ("b", 0.0), ("c", 0.001)):
            rows.append({"image": f"i{i}", "family": fam, "iou": base + shift + rng.normal(0, 0.01)})
    m = pd.DataFrame(rows)
    hyps = [{"id": "L", "a": "a", "b": "b", "metrics": ["iou"], "test": "less", "family": "primary"},
            {"id": "G", "a": "a", "b": "b", "metrics": ["iou"], "test": "greater", "family": "primary"},
            {"id": "E", "a": "c", "b": "b", "metrics": ["iou"], "test": "equivalence", "margin": 0.02},
            {"id": "X", "a": "a", "b": "b", "metrics": ["iou"], "test": "equivalence", "margin": 0.02}]
    t = analyse(m, hyps, {"bootstrap": 2000, "alpha": 0.05, "seed": 1}).set_index("ID")
    assert t.loc["L", "verdict"].startswith("XÁC NHẬN")
    assert t.loc["G", "verdict"] == "NGƯỢC DỰ ĐOÁN"
    assert t.loc["E", "verdict"].startswith("TƯƠNG ĐƯƠNG")
    assert t.loc["X", "verdict"].startswith("KHÁC")


# ─────────────────────────────────────────────────────────────────────────────
# E9: tích phân theo s, ghép kênh q, tương tác hữu hạn sai phân
# ─────────────────────────────────────────────────────────────────────────────

from probe.e9 import ig_uniform_points, interactions, q_schedule, s_uniform_points  # noqa: E402


def test_s_uniform_matches_ig_points():
    x0 = torch.zeros(1, 3, 8, 8, dtype=F64)
    dx = torch.rand(1, 3, 8, 8, dtype=F64) + 0.1
    Zs = cum_nodes(spi_increments(2.0, 16, (1,), generator(30, "cpu")))[:, 0]
    Pg, s_hat, s, _ = s_uniform_points(Zs, x0, dx, 50, generator(31, "cpu"))
    Pi, si = ig_uniform_points(x0, dx, 50, generator(31, "cpu"))
    assert float((s_hat - s).abs().max()) < 1e-12 and torch.allclose(Pg, Pi, atol=1e-12)


def test_q_schedule_endpoints_and_marginals():
    Zi = cum_nodes(spi_increments(3.0, 8, (3, 6, 6), generator(32, "cpu")))
    Zp = cum_nodes(spi_increments(3.0, 8, (1, 6, 6), generator(33, "cpu")))
    U = torch.rand(6, 6, generator=generator(34, "cpu"), dtype=F64)
    Z0, s0 = q_schedule(Zi, Zp, U, 0.0)
    Z1, s1 = q_schedule(Zi, Zp, U, 1.0)
    assert torch.equal(Z0, Zi) and torch.equal(Z1, Zp.expand_as(Zi)) and not s0.any() and s1.all()
    Zh, sh = q_schedule(Zi, Zp, U, 0.5)
    assert torch.equal(Zh[:, :, sh], Zp.expand_as(Zi)[:, :, sh]) and torch.equal(Zh[:, :, ~sh], Zi[:, :, ~sh])
    Z25, s25 = q_schedule(Zi, Zp, U, 0.25)
    assert bool((s25 <= sh).all())                                             # ghép đơn điệu theo q


def test_interactions_additive_vs_product():
    x0 = torch.zeros(1, 3, 8, 8, dtype=F64)
    dx = torch.rand(1, 3, 8, 8, dtype=F64) + 0.5
    add = lambda Z: (Z ** 2).flatten(1).sum(1)                                  # cộng tính theo phần tử → I = 0
    r, _ = interactions(add, x0, dx, 2, [0.5], 0.5, 64)
    assert max(max(d["rgb"], d["far"]) for d in r) < 1e-10
    prod = lambda Z: (Z[:, 0] * Z[:, 1]).flatten(1).sum(1)                      # tương tác kênh 0×1 tại cùng pixel
    r, _ = interactions(prod, x0, dx, 2, [0.5], 0.5, 64)
    assert min(d["rgb"] for d in r) > 1e-6 and max(d["far"] for d in r) < 1e-10


def test_aw_stream_matches_strat_points():
    from probe.mechanism import aw, aw_stream, strat_points
    from probe.quad import fg
    torch.manual_seed(21)
    fn = lambda Z: torch.sin(2 * Z).flatten(1).sum(1) * Z[:, :1].flatten(1).sum(1)
    x0 = torch.zeros(1, 3, 4, 4, dtype=F64)
    x = torch.rand(1, 3, 4, 4, dtype=F64) + 0.2
    X = spi_nodes_from(spi_increments(3.0, 9, x.shape[1:], generator(40, "cpu")), x0, x - x0)
    P, _ = strat_points(X, 5, generator(41, "cpu"))
    _, G = fg(fn, P, 64)
    A1, W1 = aw(G, X[1:] - X[:-1], 5, max_points=7)
    A2, W2 = aw_stream(fn, X, 5, generator(41, "cpu"), 64, max_points=12)
    assert torch.allclose(A1, A2, atol=1e-12) and torch.allclose(W1, W2, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# E4d: tương quan thứ hạng giữa heatmap
# ─────────────────────────────────────────────────────────────────────────────

def test_kendall_and_map_sim():
    try:
        from probe.cross import kendall_tau_b, map_sim
    except ImportError:
        pytest.skip("cross.py chưa có E4d (map_sim)")
    a = torch.rand(10, 10, dtype=F64)
    assert abs(kendall_tau_b(a, a) - 1) < 1e-12 and abs(kendall_tau_b(a, -a) + 1) < 1e-12
    A = torch.randn(3, 32, 32, dtype=F64)
    m = {"abs": A.abs().sum(0), "signed": A.sum(0)}
    perm = torch.randperm(32 * 32)
    s = map_sim(m, m, [4, 16], [0.1], perm)
    assert all(abs(v - 1) < 1e-9 for v in s.values()), s
    m2 = {"abs": torch.rand(32, 32, dtype=F64), "signed": torch.randn(32, 32, dtype=F64)}
    s2 = map_sim(m, m2, [4], [0.1], perm)
    assert abs(s2["spearman_224"]) < 0.3                                          # bản đồ độc lập: tương quan nhỏ


# ─────────────────────────────────────────────────────────────────────────────
# OCS: thước đo tương đồng, xáo khối, đối chứng khớp hạng
# ─────────────────────────────────────────────────────────────────────────────

from probe.ocs import Map, block_shuffle, matched_control, rbo_ext, top_mask  # noqa: E402
from probe.ocs import sim as ocs_sim  # noqa: E402  (tránh trùng tên với probe.e5.sim)

_OCS_M = {"pools": [4, 8], "topk": [0.05, 0.1], "rbo_p": 0.99, "radii": [2], "sw_proj": 8, "sw_pool": 4}


def test_ocs_identical_maps():
    A = torch.randn(3, 32, 32, dtype=F64)
    m = Map(A)
    perm = torch.randperm(32 * 32)
    s = ocs_sim(m, m, _OCS_M, perm, generator(50, "cpu"))
    for k, v in s.items():
        target = 0.0 if k in ("l1_norm_dist", "sliced_w1") else 1.0
        assert abs(v - target) < 1e-9, (k, v)
    assert abs(rbo_ext(m.abs, m.abs, 0.99, perm) - 1) < 1e-9


def test_ocs_signed_flip_and_shuffle():
    A = torch.randn(3, 32, 32, dtype=F64)
    perm = torch.randperm(32 * 32)
    s = ocs_sim(Map(A), Map(-A), _OCS_M, perm, generator(51, "cpu"))
    assert abs(s["spearman_abs"] - 1) < 1e-9 and s["cos_elem"] < -0.999 and s["sign_agree_top10"] == 0.0
    sh = block_shuffle(Map(A), 8, torch.Generator().manual_seed(0))
    assert torch.equal(sh.A.flatten().sort().values, A.flatten().sort().values) and not torch.equal(sh.A, A)


def test_matched_control():
    v = torch.rand(40 * 40, dtype=F64)
    perm = torch.randperm(1600)
    T = top_mask(v, 0.2, perm)
    o = perm[torch.argsort(v[perm], descending=True, stable=True)]
    r = torch.empty_like(o); r[o] = torch.arange(1600)
    Cset = T & (torch.rand(1600, generator=torch.Generator().manual_seed(1)) < 0.3)
    ctrl = matched_control(r, T, Cset, torch.Generator().manual_seed(2))
    assert int(ctrl.sum()) == int(Cset.sum()) and not (ctrl & Cset).any() and bool((ctrl <= T).all())


# ─────────────────────────────────────────────────────────────────────────────
# E10: các họ lịch trình, khớp mức exploration
# ─────────────────────────────────────────────────────────────────────────────

from probe.e10 import calibrate, d_dp_grid, d_hat, sched  # noqa: E402


@pytest.mark.parametrize("fam", ["ig", "dp", "smooth", "lognormal"])
def test_e10_schedules_valid(fam):
    delta = math.sqrt(5 / 11)
    Z = sched(fam, 32, (4000,), generator(60, "cpu"), 10.0, delta, 0.8)
    assert torch.all(Z[0] == 0) and torch.allclose(Z[-1], torch.ones_like(Z[-1]))
    assert bool(((Z[1:] - Z[:-1]) >= -1e-12).all())                               # đơn điệu
    t = torch.linspace(0, 1, 33, dtype=F64)
    assert float((Z.mean(1) - t).abs().max()) < 0.03                              # E Z(t) = t


def test_e10_exploration_matching():
    cfg = {"e10": {"alpha": 10, "N": 64, "calib": {"d": 20000, "seed": 3}}}
    cal = calibrate(cfg)
    assert abs(cal["D_target_dp_grid"] - d_dp_grid(10, 64)) < 1e-15
    assert abs(cal["D_lognormal_grid_mc"] / cal["D_target_dp_grid"] - 1) < 1e-3              # σ khớp D trên lưới
    assert abs(cal["D_dp_grid_mc"] / cal["D_target_dp_grid"] - 1) < 0.05                     # Monte Carlo DP khớp giải tích
    assert abs(cal["D_smooth_grid_mc"] / cal["D_smooth_continuous"] - 1) < 0.01              # δ²/30 trên lưới mịn
    assert abs(cal["D_smooth_continuous"] - cal["D_dp_continuous"]) < 1e-12                  # δ = √(5/(α+1)) khớp DP liên tục