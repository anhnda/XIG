"""
Thành phần cộng vào hàm mục tiêu: tín hiệu đã biết S (thí nghiệm C) và bẫy gradient H (thí nghiệm B).
Mỗi thành phần có value(X) -> (B,) và grad(P) -> tensor broadcast được tới P; __call__ = value (để cộng vào
TargetFn khi phải chạy lại mạng, ví dụ Guided IG).
"""

from __future__ import annotations

import math

import torch

from .paths import gaussian_blur


# ─────────────────────────────────────────────────────────────────────────────
# tín hiệu cộng tính S(x) = a·Σ_{i∈R} (x_i − x′_i)
# ─────────────────────────────────────────────────────────────────────────────

class Signal:
    """R: mặt nạ phần tử (1, C, H, W) bool/float. Đóng góp đúng cho tích phân thường từ x_s tới x:
    A*_i = a·(x_i − x_s,i) trên R, 0 ngoài R (với x_s = x′ thì tổng = B)."""

    def __init__(self, R: torch.Tensor, x0: torch.Tensor, a: float):
        self.R = R.to(x0.dtype)
        self.x0 = x0
        self.a = float(a)

    def value(self, X):
        return self.a * ((X - self.x0) * self.R).flatten(1).sum(1)

    __call__ = value

    def grad(self, P):
        return self.a * self.R

    def expected(self, x, x_start):
        return (self.a * (x - x_start) * self.R)[0]

    @property
    def n_elements(self) -> int:
        return int(self.R.sum())


def square_region(mask: torch.Tensor, side: int, inside: bool, thr: float, gen, tries: int, C: int = 3):
    H, W = mask.shape
    m = mask.float()
    for _ in range(tries):
        i = int(torch.randint(0, H - side + 1, (1,), generator=gen))
        j = int(torch.randint(0, W - side + 1, (1,), generator=gen))
        frac = float(m[i:i + side, j:j + side].mean())
        if (inside and frac >= thr) or (not inside and frac <= thr):
            R = torch.zeros(1, C, H, W)
            R[..., i:i + side, j:j + side] = 1.0
            return R, {"i": i, "j": j, "mask_frac": frac}
    return None, None


def scatter_region(mask: torch.Tensor, n_pixels: int, gen, C: int = 3):
    H, W = mask.shape
    idx = torch.randperm(H * W, generator=gen)[:n_pixels]
    R = torch.zeros(1, C, H * W)
    R[..., idx] = 1.0
    frac = float(mask.flatten()[idx].float().mean())
    return R.view(1, C, H, W), {"mask_frac": frac}


# ─────────────────────────────────────────────────────────────────────────────
# bẫy gradient quanh đường IG
# ─────────────────────────────────────────────────────────────────────────────

def unit_perp(v: torch.Tensor, against: torch.Tensor) -> torch.Tensor:
    v = v.double().flatten()
    a = against.double().flatten()
    v = v - (v @ a) / (a @ a) * a
    return (v / v.norm()).view(against.shape).to(against.dtype)


def direction(kind: str, dx: torch.Tensor, gen, low_sigma: float) -> torch.Tensor:
    v = torch.randn(dx.shape, generator=gen, device=gen.device, dtype=dx.dtype).to(dx.device)
    if kind == "low":
        v = gaussian_blur(v, low_sigma)
    elif kind != "high":
        raise ValueError(kind)
    return unit_perp(v, dx)


class Trap:
    """H(x) = κ sin²(πs) ⟨u, w⟩ exp(−‖w‖²/(2r²)), s = ⟨x − x′, Δx⟩/‖Δx‖², w = x − x′ − sΔx, r = ρ‖Δx‖, u ⟂ Δx.
    Trên đường IG: H = 0, ∇H = κ sin²(πs) u."""

    def __init__(self, x0, dx, u, kappa: float, rho: float):
        self.x0, self.dx, self.u = x0.flatten(1), dx.flatten(1), u.flatten(1)
        self.dn2 = float((self.dx * self.dx).sum())
        self.kappa, self.rho = float(kappa), float(rho)
        self.r2 = (rho ** 2) * self.dn2 if math.isfinite(rho) else float("inf")

    def value(self, X):
        y = X.flatten(1) - self.x0
        s = (y @ self.dx.T)[:, 0] / self.dn2
        w = y - s[:, None] * self.dx
        v = self.kappa * torch.sin(math.pi * s) ** 2 * (w @ self.u.T)[:, 0]
        if math.isfinite(self.r2):
            v = v * torch.exp(-(w * w).sum(1) / (2 * self.r2))
        return v

    __call__ = value

    def grad(self, P, batch: int = 64):
        out = []
        for i in range(0, P.shape[0], batch):
            p = P[i:i + batch].detach().clone().requires_grad_(True)
            with torch.enable_grad():
                (g,) = torch.autograd.grad(self.value(p).sum(), p)
            out.append(g.detach())
        return torch.cat(out)


def kappa_for_eta(eta: float, g_ig: torch.Tensor, t: torch.Tensor) -> float:
    """κ sao cho trên các nút IG: rms_k ‖∇H‖ = η · rms_k ‖∇F‖ (‖∇H(x_k)‖ = κ sin²(π t_k) trên đường IG)."""
    gn = g_ig.double().flatten(1).norm(dim=1)
    s2 = torch.sin(math.pi * t.double()) ** 2
    return float(eta * gn.pow(2).mean().sqrt() / s2.pow(2).mean().sqrt())


def eta_actual(trap: Trap, X: torch.Tensor, g: torch.Tensor) -> float:
    gh = trap.grad(X[:g.shape[0]])
    return float(gh.double().flatten(1).norm(dim=1).pow(2).mean().sqrt()
                 / g.double().flatten(1).norm(dim=1).pow(2).mean().sqrt())


def predicted_attenuation(h: float, rho: float, n: int = 2001) -> float:
    """XẤP XỈ bậc nhất (bỏ đạo hàm của thừa số Gauss): a(ρ) = 2∫ sin²(πt) exp(−h t(1−t)/(2ρ²)) dt."""
    if not math.isfinite(rho):
        return 1.0
    t = torch.linspace(0, 1, n, dtype=torch.float64)
    y = torch.sin(math.pi * t) ** 2 * torch.exp(-h * t * (1 - t) / (2 * rho * rho))
    return float(2 * torch.trapezoid(y, t))