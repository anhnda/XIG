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

Device: everything lives on theory.DEVICE (CPU by default, so tests stay on CPU);
scripts call set_device() to use the GPU. Toys are functional, fn(params, z), so the
sweep can vmap over many random toys at once (stack_params / *_batch).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
from torch.func import grad, jacfwd, jacrev, vmap

from . import paths

DT = torch.float64
DEVICE = torch.device("cpu")


def set_device(dev: Optional[str] = None) -> torch.device:
    """None → cuda if available, else cpu."""
    global DEVICE
    DEVICE = torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))
    return DEVICE


def _budget() -> int:
    """Evaluation coordinates per vectorised call."""
    return 20_000_000 if DEVICE.type == "cuda" else 2_600_000


def gauss_legendre(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes and weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return (torch.tensor((x + 1) / 2, dtype=DT, device=DEVICE),
            torch.tensor(w / 2, dtype=DT, device=DEVICE))


def ones(d: int) -> torch.Tensor:
    return torch.ones(d, dtype=DT, device=DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# single function F(z)
# ─────────────────────────────────────────────────────────────────────────────

def ig_exact(F: Callable, d: int, n_t: int = 64) -> torch.Tensor:
    """IG_i = ∫₀¹ ∂_iF(t·1) dt by Gauss–Legendre (exact to ~1e-14 for smooth toys)."""
    t, w = gauss_legendre(n_t)
    g = vmap(grad(F))(t[:, None] * ones(d)[None])
    return (w[:, None] * g).sum(dim=0)


def third_derivatives(F: Callable, z: torch.Tensor) -> torch.Tensor:
    """T[a, b, c] = ∂_a ∂_b ∂_c F(z), shape (d, d, d)."""
    return jacfwd(jacrev(grad(F)))(z)


def _bracket(T: torch.Tensor) -> torch.Tensor:
    """(..., d, d, d) third derivatives → (..., d): Σ_j ∂_i∂_j²F − ∂_j∂_i²F."""
    return torch.einsum("...ijj->...i", T) - torch.einsum("...jii->...i", T)


def spi_correction(F: Callable, d: int, alpha: float, n_t: int = 64) -> torch.Tensor:
    """C_i of the first-order expansion (see module docstring)."""
    t, w = gauss_legendre(n_t)
    d3 = vmap(jacfwd(jacrev(grad(F))))
    step = max(1, _budget() // d ** 3)
    C = torch.zeros(d, dtype=DT, device=DEVICE)
    for k0 in range(0, n_t, step):
        tk, wk = t[k0:k0 + step], w[k0:k0 + step]
        T = d3(tk[:, None] * ones(d)[None])                       # (k, d, d, d)
        C = C + ((wk * tk * (1 - tk))[:, None] * _bracket(T)).sum(dim=0)
    return C / (2.0 * (alpha + 1.0))


def _vertices(d: int, i0: int, i1: int) -> torch.Tensor:
    ids = torch.arange(i0, i1, device=DEVICE)
    return ((ids[:, None] >> torch.arange(d, device=DEVICE)) & 1).to(DT)


def _shapley_from_values(v: torch.Tensor, d: int) -> torch.Tensor:
    """v: (S, 2^d) vertex values (bit i of the index = feature i present) → (S, d)."""
    n = 1 << d
    ids = torch.arange(n, device=DEVICE)
    size = ((ids[:, None] >> torch.arange(d, device=DEVICE)) & 1).sum(dim=1)
    w = torch.tensor([math.factorial(s) * math.factorial(d - s - 1) / math.factorial(d)
                      for s in range(d)], dtype=DT, device=DEVICE)
    phi = torch.zeros(v.shape[0], d, dtype=DT, device=DEVICE)
    for i in range(d):
        S = ids[((ids >> i) & 1) == 0]
        phi[:, i] = (w[size[S]] * (v[:, S | (1 << i)] - v[:, S])).sum(dim=1)
    return phi


def shapley_exact(F: Callable, d: int) -> torch.Tensor:
    """φ_i = Σ_{S ∌ i} |S|!(d−|S|−1)!/d! · (v(S ∪ i) − v(S)), all 2^d vertices (d ≤ 20)."""
    if d > 20:
        raise ValueError("shapley_exact enumerates 2^d vertices; d ≤ 20")
    n, step = 1 << d, max(1, _budget() // d)
    fF = vmap(F)
    v = torch.cat([fF(_vertices(d, i0, min(n, i0 + step))) for i0 in range(0, n, step)])
    return _shapley_from_values(v[None], d)[0]


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


def dp_cdf(P: int, N: int, alpha: float, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Same law as paths.dp_grid_cdf (Dirichlet(α/N, …) bin masses), sampled on DEVICE.
    CPU → paths.dp_grid_cdf itself; CUDA → torch gamma on the GPU, seeded from `gen`."""
    if DEVICE.type != "cuda":
        return paths.dp_grid_cdf(P, N, alpha, gen).to(DEVICE)
    seed = int(torch.randint(0, 2 ** 62, (1,), generator=gen))
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    a = float(alpha) / N
    conc = torch.full((P, N), a + 1.0, dtype=DT, device=DEVICE)
    try:
        G1 = torch._standard_gamma(conc, generator=g)
    except TypeError:                                    # older torch: global CUDA RNG
        torch.cuda.manual_seed(seed)
        G1 = torch._standard_gamma(conc)
    U = torch.rand(P, N, generator=g, dtype=DT, device=DEVICE)
    logg = torch.log(G1) + torch.log(U + 1e-300) / a     # G(a) = G(a+1) · U^{1/a}
    logg = logg - logg.max(dim=1, keepdim=True).values
    w = torch.exp(logg)
    w = w / w.sum(dim=1, keepdim=True)
    F = torch.cat([torch.zeros(P, 1, dtype=DT, device=DEVICE), torch.cumsum(w, dim=1)], dim=1)
    F = F.clamp(max=1.0)
    F[:, -1] = 1.0
    return F


def spi_monte_carlo(F: Callable, d: int, alpha: float, n_paths: int = 20000, N: int = 128,
                    Q: int = 8, gen: Optional[torch.Generator] = None, chunk: int = 0,
                    antithetic: bool = True) -> dict:
    """E_{DP(α)} [∫ ∇F · ds] by Monte Carlo over paths, each integrated with Q Gauss–Legendre
    nodes per segment of an N-bin polygon (exact DP bin masses). Antithetic pairs use
    s̃(t) = 1 − s(1 − t), which has the same law for a uniform base measure.
    Returns mean (d,), standard error (d,), and the completeness residual of the mean."""
    u, wq = gauss_legendre(Q)
    gF = vmap(grad(F))
    if chunk <= 0:
        chunk = max(8, _budget() // (N * Q * d))
    outs = []
    done = 0
    while done < n_paths:
        b = min(chunk, n_paths - done)
        S = dp_cdf(b * d, N, alpha, gen).view(b, d, N + 1)
        A = _polygon_integral(gF, S, u, wq)
        if antithetic:
            A = 0.5 * (A + _polygon_integral(gF, 1.0 - S.flip(-1), u, wq))
        outs.append(A)
        done += b
    A = torch.cat(outs)
    mean = A.mean(dim=0)
    se = A.std(dim=0) / math.sqrt(A.shape[0])
    D = float(F(ones(d)) - F(0 * ones(d)))
    return {"mean": mean, "se": se, "residual": D - float(mean.sum()), "n": A.shape[0]}


# ─────────────────────────────────────────────────────────────────────────────
# toy families: fn(params, z) and params(d, seed); same random draws as before
# ─────────────────────────────────────────────────────────────────────────────

def _dev(p: dict) -> dict:
    return {k: v.to(device=DEVICE, dtype=DT) for k, v in p.items()}


def mlp_fn(p: dict, z: torch.Tensor) -> torch.Tensor:
    return (p["v"] * torch.tanh(p["W"] @ z + p["c"])).sum()


def mlp_params(d: int, seed: int = 0, width: int = 16, scale: float = 1.5,
               normalized: bool = False) -> dict:
    """F(z) = v · tanh(W z + c). normalized=True scales W by 1/√d, so the pre-activations
    (and the path noise they receive, Σ_j W_kj² v) stay O(1) as d grows."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(width, d, generator=g, dtype=DT) * scale
    if normalized:
        W = W / math.sqrt(d)
    c = torch.randn(width, generator=g, dtype=DT) * 0.5
    v = torch.randn(width, generator=g, dtype=DT)
    return _dev({"W": W, "c": c, "v": v})


def _pos_weights(d: int, g: torch.Generator, spread: float) -> torch.Tensor:
    w = torch.exp(spread * torch.randn(d, generator=g, dtype=DT))
    return w / w.sum()


def sat_fn(p: dict, z: torch.Tensor) -> torch.Tensor:
    return 4.0 * torch.sigmoid(p["a"] * (p["w"] @ z) - p["b"])


def sat_params(d: int, seed: int = 0, sharp: float = 8.0, offset: float = 1.0,
               spread: float = 1.0) -> dict:
    """F = 4 σ(sharp · w·z − offset), w > 0 log-normal. For F = g(w·z) the SPI bracket is
    T_ij = g‴ w_i w_j (w_j − w_i): zero for equal weights (spread = 0)."""
    g = torch.Generator().manual_seed(seed)
    return _dev({"w": _pos_weights(d, g, spread), "a": torch.tensor(sharp), "b": torch.tensor(offset)})


def satint_fn(p: dict, z: torch.Tensor) -> torch.Tensor:
    d = z.shape[0]
    return 4.0 * torch.sigmoid(p["a"] * (p["w"] @ z + p["g"] * (z @ (p["B"] @ z)) / d) - p["b"])


def satint_params(d: int, seed: int = 0, sharp: float = 6.0, offset: float = 1.0,
                  gamma: float = 1.0) -> dict:
    """Saturation + pairwise interactions: F = 4 σ(sharp · (w·z + γ zᵀBz / d) − offset)."""
    g = torch.Generator().manual_seed(seed)
    w = _pos_weights(d, g, 1.0)
    B = torch.randn(d, d, generator=g, dtype=DT)
    return _dev({"w": w, "B": (B + B.t()) / 2, "a": torch.tensor(sharp),
                 "b": torch.tensor(offset), "g": torch.tensor(gamma)})


def ml_fn(p: dict, z: torch.Tensor) -> torch.Tensor:
    terms = torch.where(p["M"] > 0, z[None, :], torch.ones_like(p["M"])).prod(dim=1)
    return (p["c"] * terms).sum()


def ml_params(d: int, seed: int = 0) -> dict:
    """Σ_S c_S Π_{i∈S} z_i over random subsets of size 1–3: every T_ij is 0."""
    g = torch.Generator().manual_seed(seed)
    rows, cs = [], []
    for size in (1, 2, 3):
        for _ in range(d):
            idx = torch.randperm(d, generator=g)[:size]
            m = torch.zeros(d, dtype=DT)
            m[idx] = 1.0
            rows.append(m)
            cs.append(torch.randn(1, generator=g, dtype=DT))
    return _dev({"M": torch.stack(rows), "c": torch.cat(cs)})


FAMILIES = {
    "mlp": (mlp_fn, lambda d, s: mlp_params(d, s)),
    "mlpn": (mlp_fn, lambda d, s: mlp_params(d, s, normalized=True)),
    "saturating": (sat_fn, lambda d, s: sat_params(d, s)),
    "satint": (satint_fn, lambda d, s: satint_params(d, s)),
    "multilinear": (ml_fn, lambda d, s: ml_params(d, s)),
}


def bind(fn: Callable, p: dict) -> Callable:
    return lambda z: fn(p, z)


TOYS = {name: (lambda d, s, name=name: bind(FAMILIES[name][0], FAMILIES[name][1](d, s)))
        for name in FAMILIES}


def toy_mlp(d: int = 5, width: int = 16, seed: int = 0, scale: float = 1.5,
            normalized: bool = False) -> Callable:
    return bind(mlp_fn, mlp_params(d, seed, width, scale, normalized))


def toy_saturating(d: int = 5, seed: int = 0, sharp: float = 8.0, offset: float = 1.0,
                   spread: float = 1.0) -> Callable:
    return bind(sat_fn, sat_params(d, seed, sharp, offset, spread))


def toy_sat_interact(d: int = 5, seed: int = 0, sharp: float = 6.0, offset: float = 1.0,
                     gamma: float = 1.0) -> Callable:
    return bind(satint_fn, satint_params(d, seed, sharp, offset, gamma))


def toy_multilinear(d: int = 5, seed: int = 0) -> Callable:
    return bind(ml_fn, ml_params(d, seed))


# ─────────────────────────────────────────────────────────────────────────────
# batched over many toys of one family (vmap over params)
# ─────────────────────────────────────────────────────────────────────────────

def stack_params(family: str, d: int, seeds) -> tuple[Callable, dict]:
    fn, mk = FAMILIES[family]
    ps = [mk(d, int(s)) for s in seeds]
    return fn, {k: torch.stack([p[k] for p in ps]) for k in ps[0]}


def _slice(P: dict, i0: int, i1: int) -> dict:
    return {k: v[i0:i1] for k, v in P.items()}


def _nbatch(P: dict) -> int:
    return next(iter(P.values())).shape[0]


def ig_batch(fn: Callable, P: dict, d: int, n_t: int = 64) -> torch.Tensor:
    """(S, d) exact IG of S toys."""
    t, w = gauss_legendre(n_t)
    Z = t[:, None] * ones(d)[None]
    g = vmap(vmap(grad(fn, argnums=1), in_dims=(None, 0)), in_dims=(0, None))(P, Z)   # (S, n_t, d)
    return (w[None, :, None] * g).sum(dim=1)


def correction_batch(fn: Callable, P: dict, d: int, alpha: float = 0.0, n_t: int = 64) -> torch.Tensor:
    """(S, d) first-order SPI correction C of S toys (alpha = 0 gives (α+1)·C)."""
    t, w = gauss_legendre(n_t)
    Z = t[:, None] * ones(d)[None]
    d3 = jacfwd(jacrev(grad(fn, argnums=1), argnums=1), argnums=1)
    f = vmap(vmap(d3, in_dims=(None, 0)), in_dims=(0, None))
    S = _nbatch(P)
    step = max(1, _budget() // (n_t * d ** 3))
    out = []
    for i0 in range(0, S, step):
        T = f(_slice(P, i0, i0 + step), Z)                               # (s, n_t, d, d, d)
        out.append(((w * t * (1 - t))[None, :, None] * _bracket(T)).sum(dim=1))
    return torch.cat(out) / (2.0 * (alpha + 1.0))


def shapley_batch(fn: Callable, P: dict, d: int) -> torch.Tensor:
    """(S, d) exact Shapley of the vertex games of S toys."""
    n, S = 1 << d, _nbatch(P)
    f = vmap(vmap(fn, in_dims=(None, 0)), in_dims=(0, None))
    vstep = max(1, min(n, _budget() // max(1, S * d)))
    v = torch.cat([f(P, _vertices(d, i0, min(n, i0 + vstep))) for i0 in range(0, n, vstep)], dim=1)
    return _shapley_from_values(v, d)
