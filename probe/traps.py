"""
Bẫy gradient H (thí nghiệm B) và tín hiệu đã biết S (thí nghiệm C), cài cộng vào hàm mục tiêu.

Bẫy, trong không gian ảnh (đã chuẩn hoá). Đặt y = x − x′, s(x) = ⟨y, Δx⟩/‖Δx‖² (tham số chiếu lên trục IG),
đường tham chiếu c(s) = x′ + sΔx + b·‖Δx‖·sin(πs)·q  (b = 0: đường IG; b > 0: "bent"), w(x) = y − (c(s) − x′):

    H(x) = κ · sin²(π s) · ⟨u, w⟩ · exp(−‖w‖² / (2 r²)),   r = ρ·‖Δx‖  (ρ = ∞: bỏ thừa số Gauss)

với u, q ⟂ Δx, u ⟂ q, ‖u‖ = ‖q‖ = 1. Trên đường c: w = 0 nên H = 0, nhưng ∇H = κ sin²(πs) u ≠ 0.
Với b = 0: H(x′) = H(x) = 0 và IG(F + H) − IG(F) = (κ/2)·u ⊙ Δx (tổng bằng 0, năng lượng tuỳ ý).

Dự đoán cho SPI (thô, bậc nhất): trên đường SPI, ‖w‖² ≈ h·t(1−t)·‖Δx‖² với h = 1/(α+1), nên phần
thay đổi do bẫy "line" bị nhân với a(ρ) = 2∫₀¹ sin²(πt)·exp(−h t(1−t)/(2ρ²)) dt so với IG.
"""

from __future__ import annotations

import math

import torch

from .paths import gaussian_blur


def _unit_perp(v: torch.Tensor, against: list[torch.Tensor]) -> torch.Tensor:
    v = v.double().flatten().clone()
    for a in against:
        a = a.double().flatten()
        v = v - (v @ a) / (a @ a) * a
    return (v / v.norm()).view(against[0].shape).float()


def direction(kind: str, like: torch.Tensor, against: list[torch.Tensor], gen, low_sigma: float) -> torch.Tensor:
    v = torch.randn(like.shape, generator=gen, device=like.device, dtype=like.dtype)
    if kind == "low":
        v = gaussian_blur(v, low_sigma)
    elif kind != "high":
        raise ValueError(kind)
    return _unit_perp(v, against)


class Trap:
    def __init__(self, x0, dx, u, kappa: float, rho: float, bend: float = 0.0, q=None):
        self.x0, self.dx = x0.flatten(1), dx.flatten(1)               # (1, d)
        self.u = u.flatten(1)
        self.q = None if q is None else q.flatten(1)
        self.dn2 = float((self.dx * self.dx).sum())
        self.L = math.sqrt(self.dn2)
        self.kappa, self.rho, self.bend = float(kappa), float(rho), float(bend)
        self.r2 = (rho * self.L) ** 2 if math.isfinite(rho) else float("inf")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x.flatten(1) - self.x0
        s = (y @ self.dx.T)[:, 0] / self.dn2
        c = s[:, None] * self.dx
        if self.bend and self.q is not None:
            c = c + self.bend * self.L * torch.sin(math.pi * s)[:, None] * self.q
        w = y - c
        val = self.kappa * torch.sin(math.pi * s) ** 2 * (w @ self.u.T)[:, 0]
        if math.isfinite(self.r2):
            val = val * torch.exp(-(w * w).sum(1) / (2 * self.r2))
        return val


def kappa_for(tau: float, A_ig_norm: float, u: torch.Tensor, dx: torch.Tensor) -> float:
    """κ sao cho ‖(κ/2)·u⊙Δx‖ = tau·‖A_IG(F)‖."""
    return 2.0 * tau * A_ig_norm / float((u * dx).norm())


def predicted_attenuation(h: float, rho: float, n: int = 2001) -> float:
    if not math.isfinite(rho):
        return 1.0
    t = torch.linspace(0, 1, n, dtype=torch.float64)
    y = torch.sin(math.pi * t) ** 2 * torch.exp(-h * t * (1 - t) / (2 * rho * rho))
    return float(2 * torch.trapezoid(y, t))


# ─────────────────────────────────────────────────────────────────────────────
# tín hiệu đã biết
# ─────────────────────────────────────────────────────────────────────────────

class Signal:
    """S(x) = Σ_{i∈R} w·x_i với R = ô vuông (mọi kênh). Với mọi phương pháp tích phân thường từ x_s tới x,
    đóng góp đúng là w·(x − x_s) trên R và 0 ngoài R."""

    def __init__(self, region: torch.Tensor, w: float):
        self.R = region.float()                                        # (1, 1, H, W)
        self.w = float(w)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * (x * self.R).flatten(1).sum(1)

    def expected(self, x: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        return (self.w * (x - x_start) * self.R)[0]


def pick_patch(mask: torch.Tensor, size: int, inside: bool, thr: float, gen, tries: int):
    H, W = mask.shape
    m = mask.float()
    for _ in range(tries):
        i = int(torch.randint(0, H - size + 1, (1,), generator=gen))
        j = int(torch.randint(0, W - size + 1, (1,), generator=gen))
        frac = float(m[i:i + size, j:j + size].mean())
        if (inside and frac >= thr) or (not inside and frac <= thr):
            R = torch.zeros(1, 1, H, W)
            R[..., i:i + size, j:j + size] = 1.0
            return R, (i, j, frac)
    return None, None
