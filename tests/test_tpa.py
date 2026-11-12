"""TPA core: solver, Gibbs vs aggregation, closed-form TPA-R, KernelSHAP, wrappers."""

import math

import torch

from pathattr import masks as MK
from pathattr import methods as M
from pathattr import tpa as T
from pathattr.complete import complete_stack, support_scale
from tests.toy import setup


def _psd(K, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(K, 7, generator=g, dtype=torch.float64)
    return X @ X.t() / 7


def test_entropic_qp_fixed_point_and_optimality():
    R = _psd(5)
    eps = 0.3
    q, info = T.solve_entropic_qp(R, eps)
    assert info["converged"], info
    assert abs(float(q.sum()) - 1) < 1e-12 and float(q.min()) > 0
    fp = torch.softmax(-2.0 * (R @ q) / eps, dim=0)          # q_k ∝ exp(−2(Rq)_k/ε)
    assert float((fp - q).abs().max()) < 1e-8
    obj = lambda v: float(v @ R @ v) + eps * float((v * torch.log(5 * v)).sum())
    g = torch.Generator().manual_seed(1)
    for _ in range(200):
        v = torch.distributions.Dirichlet(torch.ones(5, dtype=torch.float64)).sample()
        assert obj(q) <= obj(v) + 1e-10


def test_diagonal_R_is_not_gibbs():
    R = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    q, _ = T.solve_entropic_qp(R, 0.0, iters=200000, tol=1e-14)
    assert torch.allclose(q, torch.tensor([0.8, 0.2], dtype=torch.float64), atol=1e-4)
    qg = T.gibbs_weights(torch.diag(R), 0.0)
    assert qg.tolist() == [1.0, 0.0]


def test_same_solution_with_different_residuals():
    r = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)           # equidistant from r̄ = 0
    R = r @ r.t()
    q, _ = T.solve_entropic_qp(R, 0.5)
    qg = T.gibbs_weights((r ** 2).mean(dim=1), 0.5)
    assert torch.allclose(q, qg, atol=1e-8)
    assert T.equidistance(r, qg) < 1e-12


def test_ambiguity_identity():
    g = torch.Generator().manual_seed(0)
    r = torch.randn(4, 9, generator=g, dtype=torch.float64)
    q = torch.softmax(torch.randn(4, generator=g, dtype=torch.float64), 0)
    a = T.ambiguity(r, q)
    assert abs(a["L"] - (a["G"] - a["Dv"])) < 1e-12


def _small_game(seed=0, E=8, F=4, M=6, K=3, zero_elem=None):
    g = torch.Generator().manual_seed(seed)
    feat = torch.arange(E) % F
    B = (torch.rand(M, F, generator=g) < 0.5).double()
    y = torch.randn(M, generator=g, dtype=torch.float64)
    A = torch.randn(K, E, generator=g, dtype=torch.float64)
    if zero_elem is not None:
        A[:, zero_elem] = 0.0
    D = float(torch.randn(1, generator=g, dtype=torch.float64))
    return T.Game(B=B, y=y, feat=feat, F=F, D=D), A


def _dense_delta(game, A, lam, kappa, q):
    """Direct KKT solve of min (1/M)‖r(q) − Bδ‖² + κ δᵀΛ⁺δ, 1ᵀδ = b(q), δ = 0 off supp."""
    lam = lam / lam.sum()
    S = (lam > 0).nonzero().flatten()
    Be = game.B[:, game.feat][:, S]                                   # (M, |S|)
    Mm = game.M
    r = game.y - game.apply(q @ A)
    b = game.D - float((q @ A).sum())
    n = S.numel()
    H = Be.t() @ Be / Mm + kappa * torch.diag(1.0 / lam[S])
    KKT = torch.zeros(n + 1, n + 1, dtype=torch.float64)
    KKT[:n, :n] = 2 * H
    KKT[:n, n] = 1.0
    KKT[n, :n] = 1.0
    rhs = torch.cat([2 * Be.t() @ r / Mm, torch.tensor([b], dtype=torch.float64)])
    sol = torch.linalg.solve(KKT, rhs)
    delta = torch.zeros(A.shape[1], dtype=torch.float64)
    delta[S] = sol[:n]
    obj = float(((r - game.apply(delta)) ** 2).mean()) + kappa * float((delta[S] ** 2 / lam[S]).sum())
    return delta, obj


def test_r_correction_matches_dense_kkt_full_and_singular_support():
    for zero in (None, [1, 5]):
        game, A = _small_game(zero_elem=zero)
        lam = support_scale(A)
        kappa = 0.7
        Delta, rho, Rt, _ = T.r_correction(A, game, lam, kappa)
        g = torch.Generator().manual_seed(3)
        for _ in range(5):
            q = torch.softmax(torch.randn(A.shape[0], generator=g, dtype=torch.float64), 0)
            d_dense, obj_dense = _dense_delta(game, A, lam, kappa, q)
            assert torch.allclose(q @ Delta, d_dense, atol=1e-9)
            assert abs(float(q @ Rt @ q) - obj_dense) < 1e-9
        if zero is not None:
            assert float(Delta[:, zero].abs().max()) == 0.0


def test_fit_r_is_complete_and_keeps_dummies():
    game, A = _small_game(zero_elem=[2, 6])
    a, q, info = T.fit(A, game, "r", eps=0.1, kappa=1.0)
    assert abs(float(a.sum()) - game.D) < 1e-9
    assert float(a[[2, 6]].abs().max()) == 0.0


def test_fit_large_kappa_reduces_to_support_completion():
    game, A = _small_game()
    lam = support_scale(A)
    Delta, _, _, _ = T.r_correction(A, game, lam, 1e12)
    w = lam / lam.sum()
    for k in range(A.shape[0]):
        b = game.D - float(A[k].sum())
        assert torch.allclose(Delta[k], b * w, atol=1e-8)


def test_kernelshap_recovers_exact_shapley():
    v = lambda z: 2 * z[:, 0] + 3 * z[:, 1] + z[:, 2] + 4 * z[:, 0] * z[:, 1] + 2 * z[:, 1] * z[:, 2]
    m = MK.all_coalitions(3)
    y = v(m.double())
    w = MK.shapley_kernel_weight(m)
    phi = T.kernelshap(m.double(), y, 12.0, weights=w, ridge=0.0)
    assert torch.allclose(phi, torch.tensor([4.0, 6.0, 2.0], dtype=torch.float64), atol=1e-10)


def test_tpa_and_kernelshap_methods_run_on_toy():
    model, oracle, x, xb, D = setup("tanh", dummy_channel=2)
    gen = lambda: torch.Generator().manual_seed(0)
    outs = {}
    for mode in ("uniform", "gibbs", "core", "r"):
        r = M.tpa(oracle, xb, x, gen(), alpha=1.0, K=3, N=4, parts="grid:4+grid:4:2:2", M=24,
                  mode=mode, eps=0.1, kappa=1.0)
        assert torch.isfinite(r.attr).all()
        assert abs(float(r.attr.sum()) - D) < 1e-8
        assert float(r.attr[:, 2].abs().max()) == 0.0                 # dummy channel stays 0
        outs[mode] = r
    assert len(outs["core"].info["q"]) == 3
    ks = M.kernelshap(oracle, xb, x, gen(), parts="grid:4", M=24)
    assert abs(float(ks.attr.sum()) - D) < 1e-6


def test_completion_support_exact_and_dummy():
    model, oracle, x, xb, D = setup("tanh", dummy_channel=2)
    A = T.spi_image_candidates(oracle, xb, x, torch.Generator().manual_seed(0), 1.0, 3, 4, "mid")
    Ac, info = complete_stack(A, D, "support")
    for k in range(3):
        assert abs(float(Ac[k].sum()) - D) < 1e-10
    C, H, W = x.shape[1:]
    assert float(Ac.view(3, C, H, W)[:, 2].abs().max()) == 0.0
    assert max(abs(v) for v in info["residual_after"]) < 1e-10