"""Masks, token paths / Shapley reference, infidelity and approximate ROAD."""

import torch
import torch.nn as nn

from experiments.common import TargetProb
from pathattr import Oracle
from pathattr import estimators as EST
from pathattr import masks as MK
from pathattr import metrics as MET
from pathattr import tokens as TK
from tests.toy import ToyBackbone, setup


def test_shapley_kernel_masks_paired_nonempty():
    m = MK.shapley_kernel_masks(10, 40, torch.Generator().manual_seed(0))
    s = m.sum(dim=1)
    assert int(s.min()) >= 1 and int(s.max()) <= 9
    assert torch.equal(m[20:], ~m[:20])


def test_grid_partition_contiguous_and_build_masks():
    model, oracle, x, xb, D = setup()
    lbl = MK.partition(x, "grid:4:2:2")
    G = int(lbl.max()) + 1
    assert set(lbl.unique().tolist()) == set(range(G))
    ms = MK.build_masks(oracle, x, xb, ["grid:4", "grid:4:2:2"], 10, torch.Generator().manual_seed(0))
    assert ms.M == 10 and ms.pix.shape == (10, 64)
    full = MK.game_values(oracle, x, xb, torch.ones(1, 64, dtype=torch.bool))
    assert abs(float(full[0]) - D) < 1e-10


class ToyText(nn.Module):
    """(B, L, h) → (B,): interactions between positions, smooth."""

    def __init__(self, L=6, h=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(L * h, 5, generator=g, dtype=torch.float64) * 0.5)
        self.v = nn.Parameter(torch.randn(5, generator=g, dtype=torch.float64))

    def forward(self, e):
        return torch.tanh(e.flatten(1) @ self.W) @ self.v


def _token_setup():
    g = torch.Generator().manual_seed(1)
    L, h = 6, 4
    E = torch.randn(1, L, h, generator=g, dtype=torch.float64)
    mask_vec = torch.randn(h, generator=g, dtype=torch.float64)
    pos = torch.arange(1, L - 1)                                     # positions 0 and L−1 are special
    Eb = E.clone()
    Eb[0, pos] = mask_vec
    model = ToyText(L, h).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return TK.TokenInput(E=E, Eb=Eb, pos=pos), Oracle(model)


def test_token_path_endpoints_and_specials_fixed():
    ti, oracle = _token_setup()
    s = TK.spi_sched(ti.F, 8, 1.0, torch.Generator().manual_seed(0))
    pts = TK.build_path(ti, s)
    assert torch.equal(pts[0], ti.Eb[0]) and torch.equal(pts[-1], ti.E[0])
    assert torch.equal(pts[:, 0], ti.E[0, 0].expand(9, -1))
    assert float((s[1:] - s[:-1]).min()) >= -1e-12


def test_token_gl_estimator_complete_and_restrict():
    ti, oracle = _token_setup()
    D = oracle.endpoint_gap(ti.Eb, ti.E)
    pts = TK.build_path(ti, TK.straight_sched(ti.F, 6))
    a = EST.estimate(oracle, pts, "gl", 16)                         # (1, L, h)
    assert abs(float(a.sum()) - D) < 1e-8
    ar = ti.restrict(a)
    assert ar.shape == (1, ti.F * ti.h)
    assert abs(float(ti.token_sum(ar).sum()) - D) < 1e-8


def test_shapley_reference_efficiency_and_accuracy():
    ti, oracle = _token_setup()
    phi, info = TK.shapley_permutation(oracle, ti, max_perm=4000, min_pairs=2000, tol_rel=0.0,
                                       gen=torch.Generator().manual_seed(0))
    assert abs(float(phi.sum()) - info["D"]) < 1e-9
    # exact Shapley by enumerating all coalitions of F = 4 tokens
    import itertools, math
    F = ti.F
    f0 = float(oracle.f(ti.Eb)[0])
    v = {}
    for S in itertools.product([0, 1], repeat=F):
        v[S] = float(oracle.f(TK.coalition_emb(ti, torch.tensor([S], dtype=torch.bool)))[0]) - f0
    exact = torch.zeros(F, dtype=torch.float64)
    for S, val in v.items():
        for i in range(F):
            if S[i] == 0:
                T_ = list(S)
                T_[i] = 1
                s = sum(S)
                w = math.factorial(s) * math.factorial(F - s - 1) / math.factorial(F)
                exact[i] += w * (v[tuple(T_)] - val)
    assert float((phi - exact).abs().max()) < 0.05 * float(exact.abs().max()) + 1e-6


def test_infidelity_and_road_run():
    model, oracle, x, xb, D = setup()
    bb = ToyBackbone().eval()
    pm = TargetProb(bb, 1)
    attr = torch.randn(x.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    r = MET.infidelity(model, x, attr, sigma=0.1, n=32, gen=torch.Generator().manual_seed(0))
    assert 0.0 <= r["infd"] <= 1.0 + 1e-9
    rd = MET.road(pm, x, attr, fracs=(0.25, 0.5), iters=10, gen=torch.Generator().manual_seed(0))
    assert all(0.0 <= rd[k] <= 1.0 for k in ("road_morf", "road_lerf"))
