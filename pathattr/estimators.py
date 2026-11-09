"""
Estimators of the path integral  A_i(γ) = Σ_k ∫_0^1 ∇_i f(γ_k + u h_k) h_{k,i} du
over the polygon through pts (h_k = γ_{k+1} − γ_k).

kind      nodes per segment                    cost (backward)  property
--------  -----------------------------------  ---------------  ----------------------------
left      u = 0                                 N               = classic Riemann sum (IG,
                                                                  Guided IG, dnoise)
mid       u = 1/2                               N               deterministic
random    u ~ U(0,1), independent per segment   N·Q             unbiased for the path
                                                                  integral (a.e. differentiable)
gl        Gauss–Legendre, Q nodes               N·Q             near-exact reference

`cv_complete` turns K unbiased draws into an estimate that is still unbiased,
exactly complete and keeps dummy features at zero (control variate on the
completeness residual), replacing the multiplicative rescale.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .oracle import Oracle


def segment_nodes(kind: str, N: int, Q: int = 1,
                  gen: Optional[torch.Generator] = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """nodes, weights: (N, Q) float64 on CPU; weights sum to 1 per segment."""
    Q = max(1, int(Q))
    if kind == "left":
        return torch.zeros(N, 1, dtype=torch.float64), torch.ones(N, 1, dtype=torch.float64)
    if kind == "mid":
        return torch.full((N, 1), 0.5, dtype=torch.float64), torch.ones(N, 1, dtype=torch.float64)
    if kind == "random":
        u = torch.rand(N, Q, generator=gen, dtype=torch.float64)
        return u, torch.full((N, Q), 1.0 / Q, dtype=torch.float64)
    if kind == "gl":
        xs, ws = np.polynomial.legendre.leggauss(Q)
        u = torch.tensor((xs + 1.0) / 2.0, dtype=torch.float64).repeat(N, 1)
        w = torch.tensor(ws / 2.0, dtype=torch.float64).repeat(N, 1)
        return u, w
    raise ValueError(f"unknown estimator kind '{kind}'")


def integrate(oracle: Oracle, pts: torch.Tensor,
              nodes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Σ_k Σ_q w_kq ∇f(γ_k + u_kq h_k) ⊙ h_k  →  (1, C, H, W)."""
    N = pts.shape[0] - 1
    h = pts[1:] - pts[:-1]
    nodes = nodes.to(device=pts.device, dtype=pts.dtype)
    weights = weights.to(device=pts.device, dtype=pts.dtype)
    attr = torch.zeros_like(pts[:1])
    for q in range(nodes.shape[1]):
        P = pts[:-1] + nodes[:, q].view(N, 1, 1, 1) * h
        _, g = oracle.grad(P)
        attr += (weights[:, q].view(N, 1, 1, 1) * g * h).sum(dim=0, keepdim=True)
    return attr


def estimate(oracle: Oracle, pts: torch.Tensor, kind: str = "left", Q: int = 1,
             gen: Optional[torch.Generator] = None) -> torch.Tensor:
    nodes, weights = segment_nodes(kind, pts.shape[0] - 1, Q, gen)
    return integrate(oracle, pts, nodes, weights)


def random_draws(oracle: Oracle, pts: torch.Tensor, K: int,
                 gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """K independent one-node-per-segment random draws → (K, C, H, W)."""
    return torch.cat([estimate(oracle, pts, "random", 1, gen) for _ in range(K)], dim=0)


def cv_complete(A_pilot: torch.Tensor, a_new: torch.Tensor, D: float,
                eps: float = 1e-24) -> torch.Tensor:
    """
    Control-variate completeness correction.

        ã = â + w* (D − Σ_j â_j),   w*_i = Cov(Â_i, S) / Var(S),  S = Σ_j Â_j

    A_pilot: (K, ...) draws independent of a_new (needed for exact unbiasedness).
    a_new:   (1, ...) the draw to correct.
    Σ w* = 1 (renormalised for round-off) → Σ ã = D exactly.
    A feature that is identically 0 in all pilot draws gets w = 0 (dummy kept).
    """
    K = A_pilot.shape[0]
    A = A_pilot.reshape(K, -1).double()
    a = a_new.reshape(-1).double()
    S = A.sum(dim=1)
    Sc = S - S.mean()
    denom = float((Sc * Sc).sum())
    if K >= 2 and denom > eps:
        w = (A - A.mean(dim=0)).t() @ Sc / denom
    else:
        # no residual variance to exploit: allocate proportionally to |mean|
        m = A.mean(dim=0).abs()
        w = m / m.sum().clamp_min(eps)
    ws = float(w.sum())
    if abs(ws) > 1e-12:
        w = w / ws
    out = a + w * (D - float(a.sum()))
    return out.view_as(a_new).to(a_new.dtype)
