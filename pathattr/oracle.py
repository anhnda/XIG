"""
Model access with call counting.

Every attribution method in `pathattr` talks to the model only through an
`Oracle`, so the number of forward / backward passes (per sample) is recorded
for the compute-budget comparisons asked by reviewer R3.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CountingModule(nn.Module):
    """Wraps a scalar-output model (B, ...) -> (B,) and counts samples.

    n_fwd: samples passed through forward.
    n_bwd: samples whose input requires grad (an input-gradient follows).
    Legacy code that calls the model directly is counted the same way.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.n_fwd = 0
        self.n_bwd = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = int(x.shape[0])
        self.n_fwd += b
        if x.requires_grad and torch.is_grad_enabled():
            self.n_bwd += b
        return self.inner(x)


class Oracle:
    """f(x) and ∇f(x) in chunks, through a CountingModule."""

    def __init__(self, model: nn.Module, chunk: int = 32):
        self.model = model if isinstance(model, CountingModule) else CountingModule(model)
        self.chunk = max(1, int(chunk))

    # counters ---------------------------------------------------------------
    def counts(self) -> tuple[int, int]:
        return self.model.n_fwd, self.model.n_bwd

    def reset(self) -> None:
        self.model.n_fwd = 0
        self.model.n_bwd = 0

    # calls ------------------------------------------------------------------
    @torch.no_grad()
    def f(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B,)"""
        outs = []
        for i in range(0, x.shape[0], self.chunk):
            outs.append(self.model(x[i:i + self.chunk]).reshape(-1))
        return torch.cat(outs)

    def grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, H, W) -> (f (B,), ∇f (B, C, H, W))"""
        fs, gs = [], []
        for i in range(0, x.shape[0], self.chunk):
            xb = x[i:i + self.chunk].detach().clone().requires_grad_(True)
            with torch.enable_grad():
                out = self.model(xb).reshape(-1)
                (g,) = torch.autograd.grad(out.sum(), xb)
            fs.append(out.detach())
            gs.append(g.detach())
        return torch.cat(fs), torch.cat(gs)

    @torch.no_grad()
    def endpoint_gap(self, xb: torch.Tensor, x: torch.Tensor) -> float:
        """D = f(x) − f(x')  (2 forward passes, counted)."""
        f = self.f(torch.cat([xb, x], dim=0))
        return float(f[1] - f[0])
