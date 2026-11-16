"""
Lõi tính toán: f và ∂f/∂x tại một dãy trạng thái, rồi hai quy tắc chia credit trên CÙNG các trạng thái.

    canonical = Σ_k g_k ⊙ (x_{k+1} − x_k)                     (Riemann trái; tổng ≈ f(x_N) − f(x_0))
    idgi      = Σ_k g_k ⊙ g_k / ‖g_k‖² · (f(x_{k+1}) − f(x_k))  (tổng = f(x_N) − f(x_0) chính xác)

g_k = ∂f/∂x tại x_k trong không gian ảnh (đã chuẩn hoá ImageNet), như IDGI định nghĩa.
"""

from __future__ import annotations

import torch

RULES = ("canonical", "idgi")


class Counter:
    """Đếm số lần forward+backward (để ước tính thời gian)."""
    n = 0


def fg(fn, X: torch.Tensor, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """X: (M, 3, H, W) -> f (M,) float64, g (M, 3, H, W)."""
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


def accumulate(X: torch.Tensor, f: torch.Tensor, g: torch.Tensor) -> dict:
    """X: (N+1, …) trạng thái; f: (N+1,); g: (N+1, …) hoặc (N, …) (chỉ dùng g_0..g_{N−1})."""
    N = X.shape[0] - 1
    G = g[:N].double()
    dX = (X[1:] - X[:-1]).double()
    df = (f[1:] - f[:-1]).double()
    canon = (G * dX).sum(0)
    w = G * G
    norm = w.flatten(1).sum(1).clamp_min(1e-300).view(-1, *([1] * (G.dim() - 1)))
    idgi = (w / norm * df.view(-1, *([1] * (G.dim() - 1)))).sum(0)
    return {"canonical": canon.float(), "idgi": idgi.float()}


def run_states(fn, X: torch.Tensor, batch: int) -> dict:
    """Trạng thái cho trước (đường không phụ thuộc mô hình): một lượt fg cho cả N+1 điểm."""
    f, g = fg(fn, X, batch)
    out = accumulate(X, f, g)
    out["_f0"], out["_fN"] = float(f[0]), float(f[-1])
    return out
