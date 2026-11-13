"""
Small-noise theory of SPI (stochastic path integration), checked numerically.

Notation: F(z) = f(x' + z ⊙ Δx), z ∈ [0,1]^d (so an attribution is ∫ ∂_iF dz_i).
SPI schedules s_i(t) = F_{G_i}(t), G_i ~ DP(α, U[0,1]) iid, written s_i = t + ξ_i with
    E ξ_i = 0,  Var ξ_i(t) = v(t) = t(1−t)/(α+1),  ξ_i independent across i.

Claim (expansion to first order in 1/(α+1), exact line integrals):
    SPI_i = IG_i + C_i + O(1/α²),
    C_i   = 1/(2(α+1)) ∫₀¹ t(1−t) Σ_j [∂_i∂_j²F − ∂_j∂_i²F](t·1) dt.
Derivation sketch: expand ∂_iF(t1+ξ) to second order; E[ξ_j ξ_l] = v δ_jl gives
½ Σ_j ∂_i∂_j²F · v dt; the own-increment term E[ξ_i dξ_i] = ½ dv (exact line integral
over each jump) gives ½ ∂_i²F · v'(t) dt; integrating that by parts (v(0)=v(1)=0)
yields −½ ∫ v Σ_j ∂_j∂_i²F dt.
Properties: the bracket T_ij is antisymmetric ⇒ Σ_i C_i = 0 (completeness kept);
T_ij ≡ 0 for multilinear F ⇒ SPI = IG = Shapley there (theorem T2).

Everything is float64 and uses torch.func, so F must be a pure function of one
(d,) tensor returning a scalar.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
from torch.func import grad, jacfwd, jacrev, vmap

from . import paths

DT = torch.float64


def gauss_legendre(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes and weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.tensor((x + 1) / 2, dtype=DT), torch.tensor(w / 2, dtype=DT)


def ig_exact(F: Callable, d: int, n_t: int = 64) -> torch.Tensor:
    """IG_i = ∫₀¹ ∂_iF(t·1) dt by Gauss–Legendre (exact to ~1e-14 for smooth toys)."""
    t, w = gauss_legendre(n_t)
    g = vmap(grad(F))(t[:, None] * torch.ones(1, d, dtype=DT))
    return (w[:, None] * g).sum(dim=0)


def third_derivatives(F: Callable, z: torch.Tensor) -> torch.Tensor:
    """T[a, b, c] = ∂_a ∂_b ∂_c F(z), shape (d, d, d)."""
    return jacfwd(jacrev(grad(F)))(z)


def spi_correction(F: Callable, d: int, alpha: float, n_t: int = 64) -> torch.Tensor:
    """C_i of the first-order expansion (see module docstring)."""
    t, w = gauss_legendre(n_t)
    C = torch.zeros(d, dtype=DT)
    for tk, wk in zip(t, w):
        T = third_derivatives(F, tk * torch.ones(d, dtype=DT))
        t_ijj = torch.einsum("ijj->ij", T)          # ∂_i ∂_j² F
        t_jii = torch.einsum("jii->ij", T)          # ∂_j ∂_i² F   (indexed [i, j])
        C = C + wk * tk * (1 - tk) * (t_ijj - t_jii).sum(dim=1)
    return C / (2.0 * (alpha + 1.0))


def _polygon_integral(gF: Callable, S: torch.Tensor, u: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """Line integral of ∇F along the polygon through S[:, :, k] (b, d, N+1), Q Gauss–Legendre
    nodes per segment (exact for polynomials of degree ≤ 2Q−1 along each segment)."""
    b, d, n1 = S.shape
    Q = u.numel()
    S0 = S[:, :, :-1].transpose(1, 2)                                   # (b, N, d)
    H = (S[:, :, 1:] - S[:, :, :-1]).transpose(1, 2)                    # (b, N, d)
    P = S0[:, :, None, :] + u[None, None, :, None] * H[:, :, None, :]   # (b, N, Q, d)
    G = gF(P.reshape(-1, d)).view(b, n1 - 1, Q, d)
    return ((G * wq[None, None, :, None]).sum(dim=2) * H).sum(dim=1)


def spi_monte_carlo(F: Callable, d: int, alpha: float, n_paths: int = 20000, N: int = 128,
                    Q: int = 8, gen: Optional[torch.Generator] = None, chunk: int = 512,
                    antithetic: bool = True) -> dict:
    """E_{DP(α)} [∫ ∇F · ds] by Monte Carlo over paths, each integrated with Q Gauss–Legendre
    nodes per segment of an N-bin polygon (exact DP bin masses). Antithetic pairs use
    s̃(t) = 1 − s(1 − t), which has the same law for a uniform base measure.
    Returns mean (d,), standard error (d,), and the completeness residual of the mean."""
    u, wq = gauss_legendre(Q)
    gF = vmap(grad(F))
    outs = []
    done = 0
    while done < n_paths:
        b = min(chunk, n_paths - done)
        S = paths.dp_grid_cdf(b * d, N, alpha, gen).view(b, d, N + 1)
        A = _polygon_integral(gF, S, u, wq)
        if antithetic:
            A = 0.5 * (A + _polygon_integral(gF, 1.0 - S.flip(-1), u, wq))
        outs.append(A)
        done += b
    A = torch.cat(outs)
    mean = A.mean(dim=0)
    se = A.std(dim=0) / math.sqrt(A.shape[0])
    D = float(F(torch.ones(d, dtype=DT)) - F(torch.zeros(d, dtype=DT)))
    return {"mean": mean, "se": se, "residual": D - float(mean.sum()), "n": A.shape[0]}


# ─────────────────────────────────────────────────────────────────────────────
# toy functions
# ─────────────────────────────────────────────────────────────────────────────

def toy_mlp(d: int = 5, width: int = 16, seed: int = 0, scale: float = 1.5) -> Callable:
    """Generic smooth interactions: F(z) = v · tanh(W z + c)."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(width, d, generator=g, dtype=DT) * scale
    c = torch.randn(width, generator=g, dtype=DT) * 0.5
    v = torch.randn(width, generator=g, dtype=DT)
    return lambda z: (v * torch.tanh(W @ z + c)).sum()


def toy_saturating(d: int = 5, seed: int = 0, sharp: float = 8.0, offset: float = 1.0) -> Callable:
    """Single-index model that saturates early on the diagonal (like ResNet's logit):
    F(z) = 4 σ(sharp · w·z − offset), w > 0."""
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(d, generator=g, dtype=DT) + 0.2
    w = w / w.sum()
    return lambda z: 4.0 * torch.sigmoid(sharp * (w @ z) - offset)


def toy_multilinear(d: int = 5, seed: int = 0) -> Callable:
    """Σ_S c_S Π_{i∈S} z_i over random subsets of size 1–3: every T_ij is 0."""
    g = torch.Generator().manual_seed(seed)
    subsets = []
    for size in (1, 2, 3):
        for _ in range(d):
            idx = torch.randperm(d, generator=g)[:size].tolist()
            subsets.append((sorted(idx), float(torch.randn(1, generator=g, dtype=DT))))

    def F(z):
        out = z.sum() * 0.0
        for idx, c in subsets:
            out = out + c * torch.prod(z[idx])
        return out
    return F
