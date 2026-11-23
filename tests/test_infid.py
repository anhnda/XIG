"""
Proposal infidelity — unit test CPU (không ResNet, không dữ liệu ảnh).

C1a/C1b: phép đảo giữ phân phối biên (spi, patch, hier, pix), indexing, đầu mút.
T5/T6: quy tắc tích phân (mid chính xác bậc 2, left lệch (θ − ½)·B∘H, rand khớp bit aw_stream, lưới lồng nhau).
T10: V̂_e, V̂_o, L̂(μ) không chệch; sign-flip giữ mức ý nghĩa; J(K) suy từ cặp khớp mô phỏng trực tiếp.
D: ô vuông khớp pathscore_metrics.subset_masks; phép chiếu integral image khớp mask; sàn có / không completeness.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from probe import config as C  # noqa: E402
from probe import infid_core as IC  # noqa: E402
from probe import pathscore_metrics as PM  # noqa: E402
from probe import sampling as SM  # noqa: E402
from probe import toy_spi as TS  # noqa: E402
from probe.pathquad import NestedLattice, aw_stream, integrate, offsets  # noqa: E402
from probe.paths import cum_nodes, spi_increments  # noqa: E402

F64 = torch.float64


def gen(s):
    return C.generator(s, "cpu")


def ks_stat(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.sort().values, b.sort().values
    allv = torch.cat([a, b]).sort().values
    Fa = torch.searchsorted(a, allv, right=True).double() / a.numel()
    Fb = torch.searchsorted(b, allv, right=True).double() / b.numel()
    return float((Fa - Fb).abs().max())


# ─────────────────────────────────────────────────────────────────────────────
# C1 — phép đảo
# ─────────────────────────────────────────────────────────────────────────────

def test_C1b_reverse_indexing_endpoints():
    Z = SM.spi_schedule(2.0, 10, (4,), gen(1))
    R = SM.reverse_nodes(Z)
    assert torch.all(R[0] == 0) and torch.allclose(R[-1], torch.ones_like(R[-1]))
    for k in range(11):
        assert torch.allclose(R[k], 1 - Z[10 - k], atol=1e-15)
    d = spi_increments(2.0, 10, (4,), gen(1))
    assert torch.allclose(SM.reverse_nodes(cum_nodes(d)), cum_nodes(d.flip(0)), atol=1e-13)
    assert torch.allclose(TS.rev(Z.unsqueeze(0)), R.unsqueeze(0))


@pytest.mark.parametrize("alpha,N", [(1.0, 8), (10.0, 16), (2.0, 4)])
def test_C1b_reverse_preserves_marginal_spi(alpha, N):
    """Z^R_k và Z_k cùng phân phối (Beta(k·a, (N−k)·a)): mô-men + KS hai mẫu độc lập."""
    n = 40000
    Z = SM.spi_schedule(alpha, N, (n,), gen(2))
    R = SM.reverse_nodes(SM.spi_schedule(alpha, N, (n,), gen(3)))
    a = alpha / N
    for k in (1, N // 2, N - 1):
        m = k * a / alpha
        v = m * (1 - m) / (alpha + 1)
        assert abs(float(R[k].mean()) - m) < 5 * math.sqrt(v / n)
        assert abs(float(R[k].var()) - v) < 0.06 * v
        assert ks_stat(Z[k], R[k]) < 1.95 * math.sqrt(2 / n)


def test_C1b_reverse_preserves_marginal_patch_and_hier():
    n, N, g = 6000, 8, 2
    shape = (3, 4, 4)
    lab = SM.grid_elem_labels(shape, g)
    vals, valsR = [], []
    for s in range(n // 4):
        phi = torch.rand(g * g, generator=torch.Generator().manual_seed(10 + s), dtype=F64)
        Zp = SM.expand_patch(SM.patch_schedule(10.0, N, phi, gen(100 + s)), lab, shape)
        Zs = SM.spi_schedule(10.0, N, shape, gen(5000 + s))
        Zh = SM.mix(Zp, Zs, 0.5)
        vals.append(Zh[3].flatten()[:4])
        phi2 = torch.rand(g * g, generator=torch.Generator().manual_seed(90000 + s), dtype=F64)
        Zp2 = SM.expand_patch(SM.patch_schedule(10.0, N, phi2, gen(200000 + s)), lab, shape)
        Zh2 = SM.mix(Zp2, SM.spi_schedule(10.0, N, shape, gen(300000 + s)), 0.5)
        valsR.append(SM.reverse_nodes(Zh2)[3].flatten()[:4])
    a, b = torch.cat(vals), torch.cat(valsR)
    assert abs(float(a.mean() - b.mean())) < 0.02
    assert ks_stat(a, b) < 0.07


def test_C1_pixel_design_preserves_marginal():
    """pix (S2p/S4p): anchor_rotate theo phần tử với pha cân bằng ⇒ từng đường ~ q_SPI."""
    N, Kmax, shape = 8, 4, (1, 8, 8)
    a, b = [], []
    for s in range(300):
        phi, _ = SM.pixel_phase_design("nbr", shape, Kmax, torch.Generator().manual_seed(s))
        Z = SM.pixel_schedule(5.0, N, shape, phi[s % Kmax], gen(1000 + s))
        a.append(Z[2].flatten())
        b.append(SM.spi_schedule(5.0, N, shape, gen(9000 + s))[2].flatten())
    a, b = torch.cat(a), torch.cat(b)
    assert abs(float(a.mean() - b.mean())) < 0.01
    assert ks_stat(a, b) < 0.03


# ─────────────────────────────────────────────────────────────────────────────
# quy tắc tích phân
# ─────────────────────────────────────────────────────────────────────────────

def _quad_fn(d, s=0):
    g = torch.Generator().manual_seed(s)
    a = torch.randn(d, generator=g, dtype=F64)
    Bm = torch.randn(d, d, generator=g, dtype=F64)
    Bm = Bm + Bm.T
    return a, Bm, (lambda X: X.flatten(1) @ a + 0.5 * ((X.flatten(1) @ Bm) * X.flatten(1)).sum(1))


def test_T5_mid_exact_quadratic_and_left_bias_T6():
    d, N = 5, 7
    a, Bm, fn = _quad_fn(d, 1)
    Z = SM.spi_schedule(3.0, N, (d, 1, 1), gen(4))
    ex = integrate(fn, Z, 4, "gl", batch=64).flatten()
    mid = integrate(fn, Z, 1, "mid", batch=64).flatten()
    left = integrate(fn, Z, 1, "left", batch=64).flatten()
    assert torch.allclose(mid, ex, atol=1e-10)
    dl = (Z[1:] - Z[:-1]).reshape(N, d)
    H = dl.T @ dl
    assert torch.allclose(left - ex, -0.5 * (Bm * H).sum(1), atol=1e-10)
    th = integrate(fn, Z, 1, "theta", batch=64, theta=0.8).flatten()
    assert torch.allclose(th - ex, 0.3 * (Bm * H).sum(1), atol=1e-10)


def test_rand_matches_aw_stream_bitwise():
    d, N = 4, 6
    _, _, fn = _quad_fn(d, 2)
    Z = SM.spi_schedule(3.0, N, (d, 1, 1), gen(5))
    A1, _ = aw_stream(fn, Z, 2, gen(77), 64)
    A2 = integrate(fn, Z, 2, "rand", gen=gen(77), batch=64)
    assert torch.equal(A1, A2)


def test_offsets_and_nested_lattice():
    s, w = offsets("mid", 4)
    assert torch.allclose(s, torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=F64)) and abs(float(w.sum()) - 1) < 1e-15
    d, N = 3, 5
    fn = lambda X: torch.sin(X.flatten(1)).sum(1) * X.flatten(1)[:, 0]
    Z = SM.spi_schedule(2.0, N, (d, 1, 1), gen(6))
    nl = NestedLattice(fn, Z, 64)
    for _ in range(4):
        nl.refine()
    for L in range(3):
        assert torch.allclose(nl.left(L), integrate(fn, Z, 1 << L, "left", batch=64), atol=1e-12)
        assert torch.allclose(nl.mid(L), integrate(fn, Z, 1 << L, "mid", batch=64), atol=1e-12)
        assert torch.allclose(nl.left(L + 1), (nl.left(L) + nl.mid(L)) / 2, atol=1e-12)
    assert nl.grads == N * 8


# ─────────────────────────────────────────────────────────────────────────────
# T10 — bộ ước lượng chỉ dùng cặp
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_pairs(m, n, Ve_dir, Vo_dir, mu, g):
    E = mu + torch.randn(m, n, generator=g, dtype=F64) * Ve_dir
    O = torch.randn(m, n, generator=g, dtype=F64) * Vo_dir
    return E + O, E - O


def test_T10_unbiased_and_J():
    n, m, R = 6, 5, 4000
    g = torch.Generator().manual_seed(0)
    d = torch.tensor([1.0, -2.0, 0.5, 3.0, 1.0, -1.0], dtype=F64)
    mu = d + torch.tensor([0.3, 0.1, -0.2, 0.0, 0.5, 0.2], dtype=F64)
    se, so = 0.4, 0.9
    den = float((d * d).sum())
    tv = {"Ve": n * se ** 2 / den, "Vo": n * so ** 2 / den, "L_mu": float(((mu - d) ** 2).sum()) / den}
    acc = {k: 0.0 for k in tv}
    for _ in range(R):
        y1, y2 = _synthetic_pairs(m, n, se, so, mu, g)
        r = IC.pair_core(y1, y2, d, [2, 4])
        for k in acc:
            acc[k] += r[k] / R
    for k in tv:
        assert abs(acc[k] - tv[k]) < 0.03 * max(tv[k], 0.05), (k, acc[k], tv[k])
    r = IC.pair_core(*_synthetic_pairs(m, n, se, so, mu, g), d, [4])
    assert abs(r["J_indep_4"] - r["J_rev_4"] - r["dV_4"]) < 1e-12


def test_signflip_level_and_power():
    g = torch.Generator().manual_seed(1)
    d = torch.ones(5, dtype=F64)
    ps0, ps1 = [], []
    for s in range(300):
        y1, y2 = _synthetic_pairs(8, 5, 0.3, 0.5, d, g)
        ps0.append(IC.pair_core(y1, y2, d, [2], n_flip=199, gen=torch.Generator().manual_seed(s))["signflip_p"])
        ps1.append(IC.pair_core(y1 + 1.0, y2 - 1.0, d, [2], n_flip=199, gen=torch.Generator().manual_seed(s))["signflip_p"])
    rej0 = sum(p <= 0.05 for p in ps0) / len(ps0)
    rej1 = sum(p <= 0.05 for p in ps1) / len(ps1)
    assert rej0 < 0.09 and rej1 > 0.9


def test_indep_identity_and_c2():
    g = torch.Generator().manual_seed(2)
    d = torch.randn(7, generator=g, dtype=F64)
    Y = d + 0.2 + torch.randn(32, 7, generator=g, dtype=F64)
    r = IC.indep_core(Y, d, [1, 16])
    assert abs(r["identity_gap"]) < 1e-12
    assert abs(r["J_1"] - r["mean_path_infid"]) < 1e-10
    assert "KV_16" in r and "KV_1" in r


# ─────────────────────────────────────────────────────────────────────────────
# D — ô, chiếu, sàn
# ─────────────────────────────────────────────────────────────────────────────

def test_square_boxes_match_pathscore_masks_and_projection():
    H = W = 40
    boxes = IC.square_boxes(H, W, 12, [4, 8, 16], torch.Generator().manual_seed(9))
    Sm = PM.subset_masks(H, W, 12, [4, 8, 16], torch.Generator().manual_seed(9))
    assert torch.equal(IC.box_masks(boxes, H, W, "cpu"), Sm)
    part = IC.Part("deletion", "black", torch.full((12,), 0.5, dtype=F64), boxes[:, 2], boxes=boxes)
    A = torch.randn(3, H, W, dtype=F64)
    y = IC.project_part(part, A.sum(0, keepdim=True))[0]
    ref = 0.5 * (Sm.double() * A.sum(0)).sum((1, 2))
    assert torch.allclose(y, ref, atol=1e-10)
    dF = torch.randn(12, dtype=F64)
    assert abs(float(IC.infid((y / 0.5).unsqueeze(0), dF)[0]) - PM.subset_scores(A, Sm, dF)["infid"]) < 1e-10


def test_segment_projection_matches_masks():
    H = W = 10
    maps = torch.stack([torch.arange(H * W).view(H, W) // 7, (torch.arange(H * W).view(H, W) % 5)])
    sm = torch.tensor([0, 1, 0, 1])
    si = torch.tensor([3, 2, 0, 4])
    part = IC.Part("deletion", "black", torch.ones(4, dtype=F64), torch.zeros(4, dtype=torch.long), seg_map=sm, seg_id=si, maps=maps)
    a = torch.randn(1, H, W, dtype=F64)
    y = IC.project_part(part, a)[0]
    mk = part.masks(0, 4, H, W, "cpu").double()
    assert torch.allclose(y, (mk * a[0]).sum((1, 2)), atol=1e-12)


def test_floor_free_and_comp_appendix():
    X = torch.eye(2, dtype=F64).repeat(50, 1)
    dF = torch.ones(100, dtype=F64)
    r = IC.floor_fit(X, dF, X[:2], dF[:2], 1.0, [0.0], 4, torch.Generator().manual_seed(0))
    assert abs(r["L_free_oos"]) < 1e-10 and abs(r["L_comp_oos"] - 0.25) < 1e-10 and abs(r["sumA_comp"] - 1) < 1e-10


def test_toy_oracles_fast_subset():
    """Một phần oracle với ít mẫu (nhanh) — bản đầy đủ: bash scripts/run.sh toy."""
    g = gen(3)
    f = TS.quad_from_terms(2, cross={(0, 1): 1.0})
    vals = [float(TS.attribute(f, TS.dir_paths(2.0, 2, 20000, 2, g), "mid")[:, 0].var()) for _ in range(8)]
    m = sum(vals) / len(vals)
    assert abs(m - TS.var_w_formula(2.0, 2)) < 0.003
    r = TS.pair_mc(f, 2, 2.0, 2, "left", TS._P(2, [(0,)]), torch.ones(1, dtype=F64), 8, 20000, g)
    assert abs(r["rho"][0] - 2 / 7) < 0.02
    h = TS.quad_from_terms(2, cross={(0, 1): 1.0}, sq={0: 10.0})
    r = TS.pair_mc(h, 2, 2.0, 2, "left", TS._P(2, [(0,)]), torch.ones(1, dtype=F64), 8, 20000, g)
    assert abs(r["rho"][0] - 214 / 109) < 0.03
    r = TS.pair_mc(h, 2, 2.0, 2, "mid", TS._P(2, [(0,)]), torch.ones(1, dtype=F64), 2, 5000, g)
    assert r["rho"][0] < 1e-8
    det = TS.oracle_deterministic()
    assert all(x["pass"] for x in det), [x for x in det if not x["pass"]]


# ─────────────────────────────────────────────────────────────────────────────
# C3a / C3b — quy tắc `rand`
# ─────────────────────────────────────────────────────────────────────────────

def test_C3a_rand_unbiased_conditional_on_path():
    """Cố định đường Z; trung bình theo U của `rand` (aw_stream thật) → tích phân chính xác trên polyline, F phi tuyến."""
    d, N = 3, 5
    w = torch.tensor([1.5, -2.0, 1.0], dtype=F64)
    fn = lambda X: torch.sigmoid(X.flatten(1) @ w * 2.0) * (1 + X.flatten(1)[:, 0])
    Z = SM.spi_schedule(3.0, N, (d, 1, 1), gen(11))
    ex = integrate(fn, Z, 12, "gl", batch=256).flatten()
    R = 1500
    draws = torch.stack([integrate(fn, Z, 1, "rand", gen=gen(1000 + s), batch=256).flatten() for s in range(R)])
    se = draws.std(0) / math.sqrt(R)
    assert torch.all((draws.mean(0) - ex).abs() < 4.5 * se + 1e-12), (draws.mean(0), ex, se)
    assert float(draws.std(0).max()) > 1e-4                                   # có nhiễu thật (không suy biến)


class _FakeCtx:
    batch = 64
    seed = 1


class _FakeE:
    """Env tối thiểu cho sampling_infid.pair_block: ghi lại mọi khoá generator."""

    def __init__(self):
        self.ctx, self.image, self.shape, self.H, self.W, self.D = _FakeCtx(), "img", (1, 4, 4), 4, 4, 1.0
        self.x = torch.rand(1, 1, 4, 4, dtype=F64, generator=torch.Generator().manual_seed(0)) + 0.5
        self.x0 = torch.zeros_like(self.x)
        self.dx = self.x - self.x0
        a = torch.randn(16, dtype=F64, generator=torch.Generator().manual_seed(1))
        self.fn = lambda X: torch.sigmoid(X.flatten(1) @ a)
        self.keys = []

    def gen(self, *parts):
        self.keys.append(parts)
        return C.generator(C.seed_of(1, *parts), "cpu")

    def cgen(self, *parts):
        return torch.Generator().manual_seed(C.seed_of(1, *parts))


def test_C3b_pair_quadrature_streams_independent():
    """Mỗi thành viên (Z, RZ) của mỗi cặp rút U từ một khoá RIÊNG; không khoá nào lặp lại; đường đảo không rút số gia mới."""
    from probe import sampling_infid as SI
    E = _FakeE()
    boxes = IC.square_boxes(4, 4, 6, [1, 2], torch.Generator().manual_seed(3))
    part = IC.Part("deletion", "black", torch.ones(6, dtype=F64), boxes[:, 2], boxes=boxes, dF=torch.randn(6, dtype=F64))
    dsets = {"d": {"eval": IC.DSet("d", [part])}}
    pr = {"rule": "rand", "M": 1, "quad_probe": 0, "quad_probe_factor": 4}
    m = 3
    res = SI.pair_block(E, {"family": "spi", "alpha": 10, "N": 4}, m, pr, dsets, "t")
    qk = [k for k in E.keys if k[0] == "INFq"]
    sk = [k for k in E.keys if k[0] == "INFspi"]
    assert len(qk) == 2 * m and len(set(qk)) == 2 * m
    assert {k[-1] for k in qk} == {0, 1}                                      # w = 0 (Z) và w = 1 (RZ) tách khoá
    assert len(sk) == m and len(set(sk)) == m
    y1, y2 = res["Y"][("d", "eval")]
    assert y1.shape == y2.shape == (m, 6) and not torch.allclose(y1, y2)


def test_C3b_rand_noise_cancels_in_VoVe_toy():
    """f bậc 2: chính xác ⇒ V_e = 0, V_o = 1/24; với rand (U độc lập trong cặp) V_o − V_e vẫn = 1/24, V_e = τ²/2 > 0."""
    g = gen(21)
    h = TS.quad_from_terms(2, cross={(0, 1): 1.0}, sq={0: 10.0})
    r = TS.pair_mc(h, 2, 2.0, 2, "rand", TS._P(2, [(0,)]), torch.ones(1, dtype=F64), 8, 20000, g)
    assert abs(r["VoVe"][0] - 1 / 24) < 4 * r["VoVe"][1] + 2e-3
    assert r["Ve"][0] > 10 * r["Ve"][1]
