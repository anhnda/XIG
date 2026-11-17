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