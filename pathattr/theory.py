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
    d3 = vmap(jacfwd(jacrev(grad(F))))
    step = max(1, 4_000_000 // d ** 3)              # bound the (nodes, d, d, d) tensor
    C = torch.zeros(d, dtype=DT)
    for k0 in range(0, n_t, step):
        tk, wk = t[k0:k0 + step], w[k0:k0 + step]
        T = d3(tk[:, None] * torch.ones(1, d, dtype=DT))  # (k, d, d, d)
        t_ijj = torch.einsum("kijj->kij", T)        # ∂_i ∂_j² F
        t_jii = torch.einsum("kjii->kij", T)        # ∂_j ∂_i² F   (indexed [i, j])
        C = C + ((wk * tk * (1 - tk))[:, None] * (t_ijj - t_jii).sum(dim=2)).sum(dim=0)
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
                    Q: int = 8, gen: Optional[torch.Generator] = None, chunk: int = 0,
                    antithetic: bool = True) -> dict:
    """E_{DP(α)} [∫ ∇F · ds] by Monte Carlo over paths, each integrated with Q Gauss–Legendre
    nodes per segment of an N-bin polygon (exact DP bin masses). Antithetic pairs use
    s̃(t) = 1 − s(1 − t), which has the same law for a uniform base measure.
    Returns mean (d,), standard error (d,), and the completeness residual of the mean."""
    u, wq = gauss_legendre(Q)
    gF = vmap(grad(F))
    if chunk <= 0:                                   # ≈ 2.6M evaluation coordinates per call
        chunk = max(8, (512 * 128 * 8 * 5) // (N * Q * d))
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

def toy_mlp(d: int = 5, width: int = 16, seed: int = 0, scale: float = 1.5,
            normalized: bool = False) -> Callable:
    """Generic smooth interactions: F(z) = v · tanh(W z + c).
    normalized=True scales W by 1/√d, so the pre-activations (and the path noise they
    receive, Σ_j W_kj² v) stay O(1) as d grows; without it they grow like √d."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(width, d, generator=g, dtype=DT) * scale
    if normalized:
        W = W / math.sqrt(d)
    c = torch.randn(width, generator=g, dtype=DT) * 0.5
    v = torch.randn(width, generator=g, dtype=DT)
    return lambda z: (v * torch.tanh(W @ z + c)).sum()


def _pos_weights(d: int, g: torch.Generator, spread: float) -> torch.Tensor:
    w = torch.exp(spread * torch.randn(d, generator=g, dtype=DT))
    return w / w.sum()


def toy_saturating(d: int = 5, seed: int = 0, sharp: float = 8.0, offset: float = 1.0,
                   spread: float = 1.0) -> Callable:
    """Single-index model that saturates early on the diagonal: F(z) = 4 σ(sharp · w·z − offset),
    w > 0 log-normal with the given spread. For F = g(w·z) the SPI bracket is
    T_ij = g‴ w_i w_j (w_j − w_i): zero for equal weights, so the spread must be > 0."""
    g = torch.Generator().manual_seed(seed)
    w = _pos_weights(d, g, spread)
    return lambda z: 4.0 * torch.sigmoid(sharp * (w @ z) - offset)


def toy_sat_interact(d: int = 5, seed: int = 0, sharp: float = 6.0, offset: float = 1.0,
                     gamma: float = 1.0) -> Callable:
    """Saturation + pairwise interactions: F(z) = 4 σ(sharp · (w·z + γ zᵀBz / d) − offset)."""
    g = torch.Generator().manual_seed(seed)
    w = _pos_weights(d, g, 1.0)
    B = torch.randn(d, d, generator=g, dtype=DT)
    B = (B + B.t()) / 2
    return lambda z: 4.0 * torch.sigmoid(sharp * (w @ z + gamma * (z @ (B @ z)) / d) - offset)


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


TOYS = {
    "mlp": lambda d, s: toy_mlp(d, seed=s),
    "mlpn": lambda d, s: toy_mlp(d, seed=s, normalized=True),
    "saturating": lambda d, s: toy_saturating(d, seed=s),
    "satint": lambda d, s: toy_sat_interact(d, seed=s),
    "multilinear": lambda d, s: toy_multilinear(d, seed=s),
}


# ─────────────────────────────────────────────────────────────────────────────
# exact Shapley of the vertex game v(S) = F(1_S) − F(0)
# ─────────────────────────────────────────────────────────────────────────────

def shapley_exact(F: Callable, d: int, chunk: int = 1 << 16) -> torch.Tensor:
    """φ_i = Σ_{S ∌ i} |S|!(d−|S|−1)!/d! · (v(S ∪ i) − v(S)), all 2^d vertices (d ≤ 20)."""
    if d > 20:
        raise ValueError("shapley_exact enumerates 2^d vertices; d ≤ 20")
    n = 1 << d
    ids = torch.arange(n)
    bits = torch.arange(d)
    vals = []
    fF = vmap(F)
    for i0 in range(0, n, chunk):
        Z = ((ids[i0:i0 + chunk, None] >> bits) & 1).to(DT)
        vals.append(fF(Z))
    v = torch.cat(vals)
    size = ((ids[:, None] >> bits) & 1).sum(dim=1)
    w = torch.tensor([math.factorial(s) * math.factorial(d - s - 1) / math.factorial(d)
                      for s in range(d)], dtype=DT)
    phi = torch.zeros(d, dtype=DT)
    for i in range(d):
        S = ids[((ids >> i) & 1) == 0]
        phi[i] = (w[size[S]] * (v[S | (1 << i)] - v[S])).sum()
    return phi
