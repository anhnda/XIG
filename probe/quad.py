"""Đánh giá mạng theo lô: giá trị và gradient. Counter đếm số lượt forward+backward."""

from __future__ import annotations

import torch


class Counter:
    n = 0          # số lượt forward+backward của mạng


def fg(fn, X: torch.Tensor, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """X: (M, C, H, W) -> f (M,) float64, g (M, C, H, W)."""
    fs, gs = [], []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            out = fn(xb)
            (g,) = torch.autograd.grad(out.sum(), xb)
        fs.append(out.detach().double())
        gs.append(g.detach())
        Counter.n += int(xb.shape[0])
    return torch.cat(fs), torch.cat(gs)
