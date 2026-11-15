"""
Giao diện chung (protocol 6.1): A = Σ_k grad_F(Z_eval_k) ⊙ dZ_k, trung bình qua K đường.

grad_F là callable nhận Zb (B, *e) (e broadcast được tới `shape`) và trả ∂F tại các trạng thái đó,
shape (B, *shape). Có thuộc tính `device` và bộ đếm `n_bwd`.
"""

from __future__ import annotations

import torch

from .samplers import Sampler


def integrate(grad_F, Z: torch.Tensor, dZ: torch.Tensor, shape, batch: int) -> torch.Tensor:
    """Z, dZ: (N, P, *e). Trả attribution theo từng đường, (P, *shape), float64."""
    N, P = int(Z.shape[0]), int(Z.shape[1])
    rows = N * P
    Zr = Z.reshape((rows,) + tuple(Z.shape[2:]))
    dZr = dZ.reshape((rows,) + tuple(dZ.shape[2:]))
    dev = grad_F.device
    A = torch.zeros((P,) + tuple(shape), dtype=torch.float64, device=dev)
    for r0 in range(0, rows, batch):
        r1 = min(r0 + batch, rows)
        g = grad_F(Zr[r0:r1].to(dev))
        c = (g * dZr[r0:r1].to(device=dev, dtype=g.dtype)).to(torch.float64)
        idx = torch.arange(r0, r1, device=dev) % P          # hàng r ↔ (k = r // P, đường p = r % P)
        A.index_add_(0, idx, c)
    return A


def estimate(grad_F, sampler: Sampler, K: int, gen: torch.Generator, shape, batch: int,
             dtype=torch.float32, paths_per_draw: int = 1) -> torch.Tensor:
    """Trung bình K đường, mỗi đường N bước. Trả (*shape), float64. Số backward = K·N."""
    acc = torch.zeros(tuple(shape), dtype=torch.float64, device=grad_F.device)
    done = 0
    K = max(1, int(K))
    while done < K:
        p = min(int(paths_per_draw), K - done)
        Z, dZ = sampler.draw(p, shape, gen, dtype)
        acc += integrate(grad_F, Z, dZ, shape, batch).sum(dim=0)
        done += p
    return acc / K


def residual(D: float, A: torch.Tensor) -> float:
    """(D − ΣA)/D."""
    s = float(A.double().sum())
    return (D - s) / D if D != 0 else float("nan")
