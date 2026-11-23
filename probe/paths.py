"""
Sinh nút đường: IG (thẳng) và SPI (số gia Dirichlet(α/N, …, α/N) độc lập theo phần tử, log-space). Đánh giá mạng nằm ở
quad.py / pathquad.py.
"""

from __future__ import annotations

import math

import torch

F64 = torch.float64


def straight_nodes(x, x0, N: int) -> torch.Tensor:
    t = torch.linspace(0, 1, N + 1, device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
    return x0 + t * (x - x0)


def gamma_ge1(k: float, size, gen) -> torch.Tensor:
    """Gamma(k, 1), k ≥ 1 (Marsaglia–Tsang), float64, mọi nguồn ngẫu nhiên qua `gen`."""
    d = k - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    n_tot = 1
    for s in size:
        n_tot *= int(s)
    out = torch.empty(n_tot, dtype=F64, device=gen.device)
    todo = torch.arange(n_tot, device=gen.device)
    while todo.numel():
        n = todo.numel()
        x = torch.randn(n, generator=gen, device=gen.device, dtype=F64)
        u = torch.rand(n, generator=gen, device=gen.device, dtype=F64)
        v = 1.0 + c * x
        pos = v > 0
        v3 = torch.where(pos, v ** 3, torch.ones_like(v))
        ok = pos & (torch.log(u) < 0.5 * x * x + d - d * v3 + d * torch.log(v3))
        out[todo[ok]] = d * v3[ok]
        todo = todo[~ok]
    return out.view(tuple(size))


def spi_increments(alpha: float, N: int, shape, gen) -> torch.Tensor:
    """Số gia (N, *shape) ~ Dirichlet(α/N, …, α/N) theo từng phần tử, float64."""
    a = alpha / N
    size = (N,) + tuple(shape)
    L = torch.log(gamma_ge1(a + 1.0, size, gen)) + \
        torch.log(torch.rand(size, generator=gen, device=gen.device, dtype=F64).clamp_min(1e-300)) / a
    return torch.softmax(L, dim=0)


def cum_nodes(delta: torch.Tensor) -> torch.Tensor:
    """Z tại nút 0..N; nửa trên tính bằng phần bù để giữ chính xác sát 1; cummax giữ đơn điệu."""
    N = delta.shape[0]
    m = N // 2
    zero = torch.zeros_like(delta[:1])
    fwd = torch.cumsum(delta[:m], dim=0)
    tail = torch.flip(torch.cumsum(torch.flip(delta[m + 1:], [0]), dim=0), [0])
    Z = torch.cat([zero, fwd, 1.0 - tail, zero + 1.0], dim=0).clamp_(0.0, 1.0)
    return torch.cummax(Z, dim=0).values
