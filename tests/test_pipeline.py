"""Test nhanh cho các phần không phải sampler: spec, seed, gamma, integrate, IoU, thống kê, lưu trữ."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
import torch

from spi_study.attribute import estimate, integrate
from spi_study.config import beta_spec, canon, frac_str, line_spec, mix_spec, num, parse_spec
from spi_study.metrics import iou_topk, tie_perm
from spi_study.samplers import IGShift, SPIDP, gamma_ge1
from spi_study.seeding import generator, seed_of
from spi_study.stats import holm, n_required
from spi_study.storage import Store
from toy import F64, ToyGradF, f_u1


def test_spec_canon():
    assert canon("spi_dp:node=rand,alpha=10") == "spi_dp:alpha=10,node=rand"
    assert canon("ig_shift") == "ig_shift"
    assert num("1/11") == pytest.approx(1 / 11)
    assert num("1e5") == 1e5
    assert frac_str(Fraction(2, 4)) == "1/2" and frac_str(Fraction(3, 1)) == "3"
    assert mix_spec(10) == "coord_mix:lam=1/11,node=rand"
    assert line_spec(10) == "line:s=2/11" and line_spec(1) == "line:s=1"
    assert beta_spec(100) == "spi_dp:alpha=100,node=rand"
    assert parse_spec("line:s=2/11") == ("line", {"s": "2/11"})


def test_seed_stable():
    a = seed_of(1, "seg00001", "spi", 0, 8, 24, 0)
    assert a == seed_of(1, "seg00001", "spi", 0, 8, 24, 0)
    assert a != seed_of(1, "seg00001", "spi", 0, 8, 24, 1)
    assert 0 <= a < 2 ** 63


@pytest.mark.parametrize("k", [1.0, 1.5, 5.0, 4001.0])
def test_gamma_moments(k):
    x = gamma_ge1(k, (200_000,), generator(0))
    assert float(x.mean()) == pytest.approx(k, rel=4 * math.sqrt(1 / (k * 200_000)) + 1e-3)
    assert float(x.var()) == pytest.approx(k, rel=0.03)


def test_dirichlet_small_shape_finite():
    smp = SPIDP(0.5, 400)                                  # a = 1.25e-3
    d = smp.increments(3, (1000,), generator(1))
    assert torch.isfinite(d).all()
    assert torch.allclose(d.sum(0), torch.ones_like(d[0]), atol=1e-12)


def test_integrate_path_mapping():
    """Nhiều đường trong một lần rút phải cho đúng kết quả từng đường (index_add theo r % P)."""
    smp = IGShift(7)
    gF = ToyGradF(f_u1, 2)
    Z, dZ = smp.draw(5, (2,), generator(3), F64)
    A = integrate(gF, Z, dZ, (2,), batch=4)                 # batch lẻ để cắt ngang ranh giới
    for p in range(5):
        Ap = integrate(gF, Z[:, p:p + 1], dZ[:, p:p + 1], (2,), batch=100)[0]
        assert torch.allclose(A[p], Ap, atol=1e-12)


def test_estimate_counts_backward():
    gF = ToyGradF(f_u1, 2)
    estimate(gF, SPIDP(10, 24), 3, generator(0), (2,), batch=10, dtype=F64)
    assert gF.n_bwd == 3 * 24


def test_iou_tiebreak_and_valid():
    H = W = 8
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[:4] = True
    A = torch.ones(3, H, W)                                 # hoà hoàn toàn
    perm = tie_perm("img", H * W, 0)
    v1 = iou_topk(A, mask, perm)
    assert v1 == iou_topk(A, mask, perm)                    # tất định
    A2 = torch.zeros(3, H, W)
    A2[:, :4] = 1.0
    assert iou_topk(A2, mask, perm) == 1.0
    valid = torch.ones(H, W, dtype=torch.bool)
    valid[3] = False                                        # bỏ một hàng của mask khỏi tính toán
    assert iou_topk(A2, mask, perm, valid) == 1.0


def test_power_table():
    # bảng 6.5: SD_d 0.08/0.10/0.12/0.15 → 30/47/68/106 (protocol làm tròn; ceil có thể +1)
    for sd, n in [(0.08, 30), (0.10, 47), (0.12, 68), (0.15, 106)]:
        assert n_required(sd, 0.045) in (n, n + 1)


def test_holm():
    assert holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert holm([0.04, 0.01]) == pytest.approx([0.04, 0.02])


def test_store_roundtrip(tmp_path):
    st = Store(str(tmp_path))
    A = torch.randn(3, 4, 4)
    st.save_attr("im", "coord_mix:lam=1/11,node=rand", 2, 1, A, {"K": 8})
    assert st.has_attr("im", "coord_mix:lam=1/11,node=rand", 2, 1)
    assert torch.equal(st.load_attr("im", "coord_mix:lam=1/11,node=rand", 2, 1), A)
    st.add_extension("x", (128, 384))
    assert st.extensions() == {"x": [[128, 384]]}
