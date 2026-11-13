"""
Discrete paths from a baseline x' to an input x.

Every function returns pts of shape (N+1, C, H, W) with pts[0] = x' and
pts[N] = x exactly.

Families
--------
straight        IG line.
pixel_schedule  each pixel i follows its own schedule
                    s_i(t) = clip((t − spread·T_i) / (1 − spread), 0, 1)
                T_i ∈ (0, 1) is a per-pixel threshold; spread = 0 gives the
                straight line, spread = 1 a hard on/off step at T_i.
threshold_map   how T is laid out over the image:
                    bayer       ordered dither (Bayer matrix): every tile×tile
                                block holds every level once → "interleaved
                                evenly over the image" (spatially stratified)
                    stratified  random permutation of the levels inside each tile
                    random      random permutation over the whole image
                    block       one random time per block×block patch
                                ("random 14×14 blocks, LIG style", slide 21)
dither          threshold_map + pixel_schedule = our reconstruction of "dnoise".
spi             Stick-breaking Path Integration paths (Jeon et al., ICCV 2023):
                each feature follows the CDF of its own stick-breaking measure.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# straight line
# ─────────────────────────────────────────────────────────────────────────────

def straight(xb: torch.Tensor, x: torch.Tensor, N: int) -> torch.Tensor:
    t = torch.arange(N + 1, device=x.device, dtype=x.dtype).view(-1, 1, 1, 1) / N
    pts = xb + t * (x - xb)
    pts[0] = xb[0]
    pts[N] = x[0]
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# threshold maps
# ─────────────────────────────────────────────────────────────────────────────

def bayer_matrix(n: int) -> torch.Tensor:
    """n×n Bayer index matrix with entries 0 … n²−1 (n a power of 2)."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Bayer tile must be a power of 2, got {n}")
    M = torch.zeros(1, 1, dtype=torch.long)
    while M.shape[0] < n:
        top = torch.cat([4 * M, 4 * M + 2], dim=1)
        bot = torch.cat([4 * M + 3, 4 * M + 1], dim=1)
        M = torch.cat([top, bot], dim=0)
    return M


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def threshold_map(
    H: int,
    W: int,
    kind: str = "bayer",
    tile: int = 8,
    block: int = 14,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-pixel activation thresholds T ∈ (0, 1), shape (H, W), float64 on CPU."""
    if kind == "bayer":
        M = bayer_matrix(tile)
        n2 = tile * tile
        big = M.repeat(_ceil_div(H, tile) + 1, _ceil_div(W, tile) + 1)
        if gen is not None:   # random phase so different seeds give shifted patterns
            dy = int(torch.randint(tile, (1,), generator=gen))
            dx = int(torch.randint(tile, (1,), generator=gen))
        else:
            dy = dx = 0
        T = big[dy:dy + H, dx:dx + W].double()
        return (T + 0.5) / n2

    if kind == "stratified":
        n2 = tile * tile
        nty, ntx = _ceil_div(H, tile), _ceil_div(W, tile)
        r = torch.rand(nty * ntx, n2, generator=gen, dtype=torch.float64)
        ranks = r.argsort(dim=1).argsort(dim=1)                    # 0 … n²−1 per tile
        ranks = ranks.view(nty, ntx, tile, tile).permute(0, 2, 1, 3)
        ranks = ranks.reshape(nty * tile, ntx * tile)[:H, :W]
        return (ranks.double() + 0.5) / n2

    if kind == "random":
        n = H * W
        ranks = torch.randperm(n, generator=gen).view(H, W)
        return (ranks.double() + 0.5) / n

    if kind == "block":
        nby, nbx = _ceil_div(H, block), _ceil_div(W, block)
        nb = nby * nbx
        ranks = torch.randperm(nb, generator=gen).view(nby, nbx).double()
        T = (ranks + 0.5) / nb
        T = T.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
        return T[:H, :W].contiguous()

    raise ValueError(f"unknown threshold kind '{kind}'")


# ─────────────────────────────────────────────────────────────────────────────
# per-pixel schedule
# ─────────────────────────────────────────────────────────────────────────────

def pixel_schedule(
    xb: torch.Tensor,
    x: torch.Tensor,
    N: int,
    T: torch.Tensor,
    spread: float = 0.5,
    noise_std: float = 0.0,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    T: (H, W) shared by all channels, or (C, H, W) one map per channel.
    noise_std > 0 adds N(0, noise_std²·|x − x'|²) jitter to the interior points
    γ_1 … γ_{N−1} (endpoints stay exact; the path is then no longer monotone).
    """
    dev, dt = x.device, x.dtype
    T = T.to(device=dev, dtype=dt)
    if T.dim() == 2:
        T = T.view(1, 1, *T.shape)
    elif T.dim() == 3:
        T = T.unsqueeze(0)
    else:
        raise ValueError(f"T must be (H, W) or (C, H, W), got {tuple(T.shape)}")

    t = torch.arange(N + 1, device=dev, dtype=dt).view(-1, 1, 1, 1) / N
    if spread >= 1.0:
        s = (t >= T).to(dt)                                          # hard step at T_i
    elif spread <= 0.0:
        s = t.expand(N + 1, *T.shape[1:])                             # straight line
    else:
        s = ((t - spread * T) / (1.0 - spread)).clamp(0.0, 1.0)
    s = s.clone()
    s[0] = 0.0
    s[N] = 1.0

    pts = xb + s * (x - xb)                                           # (N+1, C, H, W)

    if noise_std > 0.0 and N > 1:
        eps = torch.randn((N - 1,) + tuple(x.shape[1:]), generator=gen,
                          dtype=torch.float64).to(device=dev, dtype=dt)
        pts[1:N] = pts[1:N] + noise_std * (x - xb).abs() * eps
    return pts


def dither(
    xb: torch.Tensor,
    x: torch.Tensor,
    N: int = 24,
    kind: str = "bayer",
    tile: int = 8,
    block: int = 14,
    spread: float = 0.5,
    noise_std: float = 0.0,
    per_channel: bool = False,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Reconstruction of "dnoise" (slide 20: 'dither s=0.5').

    Defaults: Bayer 8×8 thresholds shared across RGB, spread 0.5, 24 steps,
    no extra jitter. See REVISION.md for the assumptions.
    """
    _, C, H, W = x.shape
    if per_channel:
        T = torch.stack([threshold_map(H, W, kind, tile, block, gen) for _ in range(C)])
    else:
        T = threshold_map(H, W, kind, tile, block, gen)
    return pixel_schedule(xb, x, N, T, spread=spread, noise_std=noise_std, gen=gen)


# ─────────────────────────────────────────────────────────────────────────────
# SPI (stick-breaking random paths)
# ─────────────────────────────────────────────────────────────────────────────

def _stick_breaking_cdf(P: int, N: int, alpha: float, K: int,
                        gen: Optional[torch.Generator]) -> torch.Tensor:
    """F_i(k/N) for P independent SBP(U(0,1), alpha) measures, shape (P, N+1)."""
    U = torch.rand(P, K, generator=gen, dtype=torch.float64).clamp_min(1e-300)
    log_1m_beta = torch.log(U) / alpha                  # log(1 − β), β ~ Beta(1, α)
    beta = -torch.expm1(log_1m_beta)                    # β = 1 − exp(log(1 − β))
    log_rem = torch.cumsum(log_1m_beta, dim=1)
    log_rem = torch.cat([torch.zeros(P, 1, dtype=torch.float64), log_rem[:, :-1]], dim=1)
    pi = beta * torch.exp(log_rem)                      # stick weights
    tj = torch.rand(P, K, generator=gen, dtype=torch.float64)   # atoms ~ U(0, 1)
    b = torch.ceil(tj * N).long().clamp(1, N)           # t_j ≤ k/N  ⇔  ceil(t_j N) ≤ k
    M = torch.zeros(P, N + 1, dtype=torch.float64).scatter_add_(1, b, pi)
    F = torch.cumsum(M, dim=1)
    return F / F[:, -1:].clamp_min(1e-300)              # renormalise the truncation


def dp_grid_cdf(P: int, N: int, alpha: float, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """EXACT law of F_G(k/N), k = 0 … N, for P independent G ~ DP(α, U(0,1)):
    the masses of the N equal bins are Dirichlet(α/N, …, α/N) — no truncation,
    valid for every α (stick-breaking with a capped number of atoms is not:
    α = 1000 with 256 atoms behaves like α ≈ 250).
    Small shapes a = α/N are sampled in log space: G(a) = G(a+1) · U^{1/a}.
    Randomness comes from `gen` (a numpy Generator seeded from it). (P, N+1) float64."""
    seed = int(torch.randint(0, 2 ** 62, (1,), generator=gen))
    rng = np.random.default_rng(seed)
    a = float(alpha) / N
    logg = np.log(rng.gamma(a + 1.0, size=(P, N))) + np.log(rng.random((P, N)) + 1e-300) / a
    logg -= logg.max(axis=1, keepdims=True)
    w = np.exp(logg)
    w /= w.sum(axis=1, keepdims=True)
    F = np.concatenate([np.zeros((P, 1)), np.cumsum(w, axis=1)], axis=1)
    F[:, -1] = 1.0
    return torch.from_numpy(F)


def spi(
    xb: torch.Tensor,
    x: torch.Tensor,
    N: int = 24,
    alpha: float = 10.0,
    per_channel: bool = True,
    tail: float = 1e-4,
    max_atoms: int = 256,
    chunk: int = 16384,
    gen: Optional[torch.Generator] = None,
    sampler: str = "dirichlet",
) -> torch.Tensor:
    """γ_i(t) = x'_i + F_{G_i}(t)(x_i − x'_i), G_i ~ DP(α, U(0,1)).

    sampler='dirichlet' (default): exact bin masses, any α.
    sampler='stick': legacy truncated stick-breaking (≤ max_atoms atoms; wrong for
    α ≳ max_atoms / log(1/tail), kept only to reproduce old runs).
    Small α → each feature jumps at one random time (hard random order);
    large α → close to the straight line (α → ∞ is IG).
    """
    shape = tuple(x.shape[1:]) if per_channel else (1,) + tuple(x.shape[2:])
    P = int(math.prod(shape))
    if sampler == "dirichlet":
        parts = [dp_grid_cdf(min(chunk, P - i), N, alpha, gen) for i in range(0, P, chunk)]
    elif sampler == "stick":
        K = int(math.ceil(math.log(1.0 / tail) / math.log1p(1.0 / alpha)))
        K = max(8, min(K, max_atoms))
        parts = [_stick_breaking_cdf(min(chunk, P - i), N, alpha, K, gen)
                 for i in range(0, P, chunk)]
    else:
        raise ValueError(f"unknown sampler '{sampler}'")
    F = torch.cat(parts, dim=0)                          # (P, N+1)
    s = F.t().contiguous().view((N + 1,) + shape).to(device=x.device, dtype=x.dtype)
    s[0] = 0.0
    s[N] = 1.0
    return xb + s * (x - xb)
