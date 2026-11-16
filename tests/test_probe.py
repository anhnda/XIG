"""Test trên một CNN ReLU nhỏ (CPU, ảnh 3×16×16). Kiểm công thức bẫy, completeness, các đường, tín hiệu."""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe.config import generator
from probe.integrate import RULES
from probe.metrics import cos_deb
from probe.model import TargetFn
from probe.paths import gaussian_blur, run_method, spi_nodes
from probe.traps import Signal, Trap, direction, predicted_attenuation

torch.manual_seed(0)


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
def setup():
    torch.manual_seed(1)
    net = Toy().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, 3, 16, 16)
    x0 = torch.full_like(x, -1.5)
    return net, x, x0, TargetFn(net, 0)


METHODS = {
    "ig": {"kind": "straight", "steps": 64},
    "spi": {"kind": "spi", "alpha": 10, "steps": 16, "paths": 8, "n_sub": 4},
    "guided": {"kind": "guided", "steps": 32, "fraction": 0.25, "max_dist": 0.02},
    "blur": {"kind": "blur", "steps": 32, "max_sigma": 50.0},
}


@pytest.mark.parametrize("name", list(METHODS))
def test_idgi_completeness_exact(setup, name):
    net, x, x0, fn = setup
    o, meta = run_method(name, METHODS[name], fn, x, x0, batch=64, seed_key=("t",))
    D = o["_fN"] - o["_f0"]
    assert abs(float(o["idgi"].double().sum()) - D) < 1e-4 * max(1.0, abs(D))
    if name != "blur":
        assert torch.allclose(meta["start"], x0)


def test_canonical_completeness_fine_grid(setup):
    net, x, x0, fn = setup
    o, _ = run_method("ig", {"kind": "straight", "steps": 2000}, fn, x, x0, batch=512)
    D = o["_fN"] - o["_f0"]
    assert abs(float(o["canonical"].double().sum()) - D) < 0.01 * abs(D) + 1e-4


def test_guided_reaches_input(setup):
    net, x, x0, fn = setup
    o, _ = run_method("guided", METHODS["guided"], fn, x, x0, batch=64)
    assert o["_end_gap"] < 1e-5


def test_blur_endpoints(setup):
    net, x, x0, fn = setup
    o, meta = run_method("blur", METHODS["blur"], fn, x, x0, batch=64)
    assert torch.allclose(meta["start"], gaussian_blur(x, 50.0))
    assert abs(o["_fN"] - float(fn(x))) < 1e-5


def test_spi_nodes():
    Z = spi_nodes(10.0, 16, (3, 4, 4), generator(0, "cpu"))
    assert torch.all(Z[0] == 0) and torch.all(Z[-1] == 1) and torch.all(Z[1:] >= Z[:-1])


def _trap(x, x0, rho, bend=0.0):
    gen = generator(3, "cpu")
    dx = x - x0
    q = direction("high", x, [dx], gen, 2.0) if bend else None
    u = direction("high", x, [dx] + ([q] if q is not None else []), gen, 2.0)
    return Trap(x0, dx, u, kappa=7.0, rho=rho, bend=bend, q=q), u, q, dx


@pytest.mark.parametrize("rho", [0.01, float("inf")])
@pytest.mark.parametrize("bend", [0.0, 0.2])
def test_trap_zero_on_curve_and_gradient(setup, rho, bend):
    _, x, x0, _ = setup
    trap, u, q, dx = _trap(x, x0, rho, bend)
    L = float(dx.norm())
    for s in (0.0, 0.2, 0.5, 0.9, 1.0):
        c = x0 + s * dx + (bend * L * math.sin(math.pi * s) * q if bend else 0.0)
        c = c.detach().requires_grad_(True)
        h = trap(c)
        assert abs(float(h)) < 1e-4
        (g,) = torch.autograd.grad(h.sum(), c)
        assert torch.allclose(g, 7.0 * math.sin(math.pi * s) ** 2 * u, atol=1e-4)


def test_trap_ig_change_formula(setup):
    """IG(F + H) − IG(F) = (κ/2)·u⊙Δx, tổng bằng 0."""
    net, x, x0, fn = setup
    trap, u, _, dx = _trap(x, x0, 0.01)
    mc = {"kind": "straight", "steps": 400}
    a, _ = run_method("ig", mc, fn, x, x0, batch=512)
    b, _ = run_method("ig", mc, fn.with_extra(trap), x, x0, batch=512)
    d = (b["canonical"] - a["canonical"]).double()
    expect = (7.0 / 2) * (u * dx)[0].double()
    assert float((d - expect).norm() / expect.norm()) < 0.01
    assert abs(float(d.sum())) < 1e-3 * float(expect.abs().sum())


def test_spi_escapes_thin_trap(setup):
    """Bẫy mỏng quanh đường IG: SPI (α=10) gần như không bị ảnh hưởng, IG bị ảnh hưởng hoàn toàn."""
    net, x, x0, fn = setup
    trap, u, _, dx = _trap(x, x0, 0.001)
    a, _ = run_method("spi", METHODS["spi"], fn, x, x0, batch=64, seed_key=("crn",))
    b, _ = run_method("spi", METHODS["spi"], fn.with_extra(trap), x, x0, batch=64, seed_key=("crn",))
    d = (b["canonical"] - a["canonical"]).double()
    ig_change = (7.0 / 2) * (u * dx)[0].double()
    assert float(d.norm() / ig_change.norm()) < 0.05
    assert predicted_attenuation(1 / 11, 0.001) < 1e-3


def test_signal_retention_canonical(setup):
    """Tín hiệu tuyến tính: với tích phân thường, A(F + S) − A(F) = w·(x − x_start) trên R."""
    net, x, x0, fn = setup
    R = torch.zeros(1, 1, 16, 16)
    R[..., 4:9, 4:9] = 1
    sig = Signal(R, 0.3)
    for name in ("ig", "blur"):
        a, meta = run_method(name, METHODS[name], fn, x, x0, batch=64)
        b, _ = run_method(name, METHODS[name], fn.with_extra(sig), x, x0, batch=64)
        d = (b["canonical"] - a["canonical"]).double()
        star = sig.expected(x, meta["start"]).double()
        assert float((d - star).norm() / star.norm()) < 1e-3


def test_cos_deb_noisy_copies():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(3, 32, 32, generator=g)
    s = 0.5
    B1 = A + s * torch.randn(A.shape, generator=g)
    B2 = A + s * torch.randn(A.shape, generator=g)
    V = torch.full_like(A, s * s)
    assert abs(cos_deb(B1, V, B2, V) - 1.0) < 0.05


def test_attenuation_monotone():
    vals = [predicted_attenuation(h, 0.05) for h in (1e-5, 1e-4, 1e-3, 1e-2)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert predicted_attenuation(0.1, float("inf")) == 1.0
