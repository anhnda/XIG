"""
Các họ đường đi — chỉ sinh NÚT; đánh giá mạng và quy tắc nằm ở quad.py.

straight : x_k = x′ + (k/N)·Δx                                                       (IG)
spi      : x_k = x′ + Z_k ⊙ Δx; số gia Z theo quá trình Dirichlet độc lập từng phần tử, Dirichlet(α/N,…)
           trên N ô (log-space). Coupling giữa các lưới: rút ở lưới mịn N_max, lưới thô = cộng các khối
           số gia liên tiếp (tính chất gộp của Dirichlet) → cùng một đường DP ở nhiều độ phân giải.
blur     : BlurIG — x_k = GaussBlur(x, σ_k), σ_k = σ_max·(1 − k/N). Điểm đầu là ảnh mờ, KHÔNG phải ảnh đen.
guided   : Guided IG, theo thuật toán tham chiếu thư viện PAIR `saliency` (fraction, max_dist). Đường phụ
           thuộc mô hình; trả về nút, f và gradient tại nút (đã tính khi chạy thuật toán).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .quad import fg

F64 = torch.float64


def gaussian_blur(x: torch.Tensor, sigma: float, truncate: float = 4.0) -> torch.Tensor:
    if sigma <= 0:
        return x
    H, W = x.shape[-2:]
    r = min(int(math.ceil(truncate * sigma)), H - 1, W - 1)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t * t / (2 * sigma * sigma))
    k = k / k.sum()
    C = x.shape[1]
    y = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)


def straight_nodes(x, x0, N: int) -> torch.Tensor:
    t = torch.linspace(0, 1, N + 1, device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
    return x0 + t * (x - x0)


def blur_nodes(x, N: int, smax: float) -> torch.Tensor:
    return torch.cat([gaussian_blur(x, smax * (1 - k / N)) for k in range(N + 1)])


# ─────────────────────────────────────────────────────────────────────────────
# SPI
# ─────────────────────────────────────────────────────────────────────────────

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


def coarsen(delta: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return delta
    N = delta.shape[0]
    if N % factor:
        raise ValueError(f"N={N} không chia hết cho {factor}")
    return delta.view(N // factor, factor, *delta.shape[1:]).sum(1)


def spi_nodes_from(delta: torch.Tensor, x0, dx, factor: int = 1) -> torch.Tensor:
    Z = cum_nodes(coarsen(delta, factor)).to(dx.dtype)
    return x0 + Z * dx


# ─────────────────────────────────────────────────────────────────────────────
# Guided IG
# ─────────────────────────────────────────────────────────────────────────────

def guided_nodes(fn, x, x0, N: int, fraction: float, max_dist: float, batch: int):
    """-> X (N+1, C,H,W), f (N+1,), g (N, C,H,W) tại nút 0..N−1, end_gap."""
    eps = 1e-9
    dx = (x - x0)[0]
    adx = dx.abs()
    zero = adx == 0
    z = torch.zeros_like(dx)
    z[zero] = 1.0
    l1_total = float(adx.sum())
    states, fs, gs = [], [], []
    for step in range(N):
        xc = (x0[0] + z * dx).unsqueeze(0)
        f, g = fg(fn, xc, batch)
        states.append(xc[0]); fs.append(f[0]); gs.append(g[0])
        g = g[0]
        alpha = (step + 1) / N
        a_min, a_max = max(alpha - max_dist, 0.0), min(alpha + max_dist, 1.0)
        l1_target = l1_total * (1 - alpha)
        for _ in range(10_000):
            z = torch.where(~zero & (z < a_min), torch.full_like(z, a_min), z)
            l1_cur = float((adx * (1 - z)).sum())
            if math.isclose(l1_target, l1_cur, rel_tol=eps, abs_tol=eps):
                break
            ga = g.abs().double()
            ga[(z >= a_max) | zero] = float("inf")
            fin = torch.isfinite(ga)
            if not bool(fin.any()):
                break
            thr = torch.quantile(ga.flatten(), fraction, interpolation="lower")    # trên MỌI phần tử, như bản tham chiếu
            s = fin & (ga <= thr)
            l1_s = float((adx * (a_max - z) * s).sum())
            gamma = (l1_cur - l1_target) / l1_s if l1_s > 0 else float("inf")
            if gamma > 1.0:
                z = torch.where(s, torch.full_like(z, a_max), z)
            else:
                z = torch.where(s, z + gamma * (a_max - z), z)
                break
    xN = (x0[0] + z * dx).unsqueeze(0)
    fN, _ = fg(fn, xN, batch)
    X = torch.stack(states + [xN[0]])
    f = torch.cat([torch.stack(fs), fN])
    return X, f, torch.stack(gs), float((xN - x).abs().max())
