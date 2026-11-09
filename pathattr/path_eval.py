"""
The single, correct evaluation of a discrete path (fix for bug B1).

pts[0] = x', pts[N] = x. For each segment k = 0..N-1:
    g_k  = ∇f(γ_k)
    h_k  = γ_{k+1} − γ_k
    d_k  = g_k · h_k
    Δf_k = f(γ_{k+1}) − f(γ_k)          ← aligned with d_k
so Σ_k Δf_k = f(x) − f(x') exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .oracle import Oracle


@dataclass
class PathEval:
    f: torch.Tensor   # (N+1,)       f(γ_0) … f(γ_N)
    g: torch.Tensor   # (N, C, H, W) ∇f(γ_k), k < N
    h: torch.Tensor   # (N, C, H, W) γ_{k+1} − γ_k
    d: torch.Tensor   # (N,)
    df: torch.Tensor  # (N,)

    @property
    def N(self) -> int:
        return int(self.h.shape[0])


def eval_path(oracle: Oracle, pts: torch.Tensor) -> PathEval:
    """N backward + 1 extra forward (for f(γ_N))."""
    if pts.dim() != 4 or pts.shape[0] < 2:
        raise ValueError(f"pts must be (N+1, C, H, W) with N >= 1, got {tuple(pts.shape)}")
    N = pts.shape[0] - 1
    f_left, g = oracle.grad(pts[:N])
    f_end = oracle.f(pts[N:])
    f = torch.cat([f_left, f_end])
    h = pts[1:] - pts[:-1]
    d = (g * h).flatten(1).sum(1)
    df = f[1:] - f[:-1]
    return PathEval(f=f, g=g, h=h, d=d, df=df)


def legacy_df(f: torch.Tensor) -> torch.Tensor:
    """Reproduce the shifted Δf of the original code from correct f(γ_0..γ_N).

    legacy Δf_0 = 0, legacy Δf_k = f(γ_k) − f(γ_{k−1}) for k ≥ 1
    (the last step is dropped).
    """
    df = f[1:] - f[:-1]
    return torch.cat([df.new_zeros(1), df[:-1]])
