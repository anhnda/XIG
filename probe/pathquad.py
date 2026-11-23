"""
Quy tắc tích phân trên một đường gấp khúc cho trước (proposal infidelity, P0 / T5 / E-quadrature / C3).

Đường = N+1 nút x_0..x_N, đoạn k nội suy tuyến tính x_k + s·Δx_k, s ∈ [0, 1]. Mọi quy tắc tính
    A = Σ_k Σ_m w_m · ∇F(x_k + s_m·Δx_k) ⊙ Δx_k
và chỉ khác bộ (s_m, w_m) — cùng số gradient N·M mỗi đường:

  rand   s_m = (m + U_{k,m}) / M, U ~ U(0,1) dùng chung mọi phần tử     (= aw_stream, canon_rand khi M = 1;
                                                                       MẶC ĐỊNH CŨ của P4 / PATHSCORE — giữ để khớp cache)
  left   s_m = m / M                                                    (Riemann trái tổng hợp; cờ tương thích lý thuyết T9)
  mid    s_m = (m + ½) / M                                              (điểm giữa tổng hợp; chính xác cho mọi F bậc 2 — T5)
  theta  s_m = (m + θ) / M                                              (họ tỷ lệ T6; θ = 0 ≡ left, θ = ½ ≡ mid)
  gl     Gauss–Legendre M điểm/đoạn                                     (chỉ dùng cho toy / tham chiếu; chính xác bậc 2M−1)

Lưới lồng nhau (E-quadrature): mức L có 2^L điểm trái mỗi đoạn, offset j/2^L. Điểm MỚI của mức L+1 là offset lẻ
(2j+1)/2^{L+1} = điểm giữa của mức L. Gọi S_L = Σ_k Σ_{j<2^L} g(x_k + j/2^L·Δx_k) ⊙ Δx_k:
    left(N·2^L) = S_L / 2^L,   mid(N·2^L) = (S_{L+1} − S_L) / 2^L,   left(N·2^{L+1}) = (left_L + mid_L) / 2.
⇒ ba giá trị KHÔNG độc lập (mid = một bước Richardson của left). Tinh dần cho tới khi sai số chiếu giữa hai mức liên tiếp
dưới tolerance (không mặc định 4N là đủ) — xem infid_quad.py.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .quad import fg

F64 = torch.float64
RULES = ("rand", "left", "mid", "theta", "gl")


def aw_stream(fn, X: torch.Tensor, M: int, gen, batch: int, max_points: int = 512):
    """Quy tắc `rand` (canon_rand khi M = 1) — chép nguyên aw_stream cũ của P4 (cùng thứ tự rút U ⇒ khớp bit cache):
    s = (m + U_{k,m}) / M, U ~ U(0,1) dùng chung mọi phần tử. Trả (A, W) với W = Σ |g|·|Δx| (không dùng trong proposal)."""
    N = X.shape[0] - 1
    U = torch.rand(N, M, generator=gen, device=gen.device, dtype=F64).to(X.device)
    s = (torch.arange(M, device=X.device, dtype=F64).view(1, M) + U) / M
    dX = X[1:] - X[:-1]
    shp = (-1, M) + (1,) * (X.dim() - 1)
    A = torch.zeros(tuple(X.shape[1:]), dtype=F64, device=X.device)
    W = torch.zeros_like(A)
    ck = max(1, max_points // M)
    for k0 in range(0, N, ck):
        k1 = min(N, k0 + ck)
        P = X[k0:k1].unsqueeze(1) + s[k0:k1].view(shp).to(X.dtype) * dX[k0:k1].unsqueeze(1)
        _, G = fg(fn, P.reshape((-1,) + tuple(X.shape[1:])), batch)
        g = G.view((k1 - k0, M) + tuple(X.shape[1:])).double()
        d = dX[k0:k1].double()
        A += (g.sum(1) * d).sum(0)
        W += (g.abs().sum(1) * d.abs()).sum(0)
        del P, G, g, d
    return A / M, W / M


def offsets(rule: str, M: int, theta: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Offset (M,) và trọng số (M,) trên [0, 1] cho quy tắc tất định."""
    M = int(M)
    if M < 1:
        raise ValueError("M ≥ 1")
    m = torch.arange(M, dtype=F64)
    if rule == "left":
        return m / M, torch.full((M,), 1.0 / M, dtype=F64)
    if rule == "mid":
        return (m + 0.5) / M, torch.full((M,), 1.0 / M, dtype=F64)
    if rule == "theta":
        if not 0.0 <= float(theta) <= 1.0:
            raise ValueError("θ ∈ [0, 1]")
        return (m + float(theta)) / M, torch.full((M,), 1.0 / M, dtype=F64)
    if rule == "gl":
        x, w = np.polynomial.legendre.leggauss(M)
        return torch.tensor((x + 1) / 2, dtype=F64), torch.tensor(w / 2, dtype=F64)
    raise ValueError(f"quy tắc tất định không biết: {rule}")


def lattice_sum(fn, X: torch.Tensor, s: torch.Tensor, w: torch.Tensor, batch: int, max_points: int = 512) -> torch.Tensor:
    """Σ_k Σ_m w_m · ∇F(x_k + s_m Δx_k) ⊙ Δx_k, float64, sinh điểm theo khối đoạn (không giữ toàn bộ gradient)."""
    N = X.shape[0] - 1
    M = int(s.numel())
    s = s.to(X.device)
    w = w.to(X.device)
    dX = X[1:] - X[:-1]
    shp = (1, M) + (1,) * (X.dim() - 1)
    A = torch.zeros(tuple(X.shape[1:]), dtype=F64, device=X.device)
    ck = max(1, int(max_points) // M)
    for k0 in range(0, N, ck):
        k1 = min(N, k0 + ck)
        P = X[k0:k1].unsqueeze(1) + s.view(shp).to(X.dtype) * dX[k0:k1].unsqueeze(1)
        _, G = fg(fn, P.reshape((-1,) + tuple(X.shape[1:])), batch)
        g = G.view((k1 - k0, M) + tuple(X.shape[1:])).double()
        A += ((g * w.view(shp)).sum(1) * dX[k0:k1].double()).sum(0)
        del P, G, g
    return A


def integrate(fn, X: torch.Tensor, M: int, rule: str, gen=None, batch: int = 64, theta: float = 0.5,
              max_points: int = 512) -> torch.Tensor:
    """Attribution (C,H,W) float64 của đường X theo `rule` với M điểm/đoạn. rand cần `gen` (giống aw_stream: cùng gen ⇒ cùng U
    ⇒ khớp bit với cache P4 / PATHSCORE)."""
    if rule == "rand":
        if gen is None:
            raise ValueError("rule = rand cần generator")
        A, _ = aw_stream(fn, X, int(M), gen, batch, max_points)
        return A
    s, w = offsets(rule, M, theta)
    return lattice_sum(fn, X, s, w, batch, max_points)


def n_grad(N: int, M: int) -> int:
    return int(N) * int(M)


# ─────────────────────────────────────────────────────────────────────────────
# lưới lồng nhau (E-quadrature)
# ─────────────────────────────────────────────────────────────────────────────

def level_offsets(L: int) -> torch.Tensor:
    """Offset MỚI của mức L: L = 0 → {0}; L ≥ 1 → {(2j+1)/2^L : j < 2^{L−1}}."""
    if L == 0:
        return torch.zeros(1, dtype=F64)
    h = 1 << (L - 1)
    return (2 * torch.arange(h, dtype=F64) + 1) / (2 * h)


class NestedLattice:
    """Tích luỹ S_L theo mức; mỗi lần `refine()` chỉ tính gradient ở điểm mới. Truy cập left(L), mid(L) (mid(L) cần mức L+1)."""

    def __init__(self, fn, X: torch.Tensor, batch: int, max_points: int = 512):
        self.fn, self.X, self.batch, self.max_points = fn, X, int(batch), int(max_points)
        self.N = X.shape[0] - 1
        self.S: list[torch.Tensor] = []          # S_L
        self.new: list[torch.Tensor] = []        # tổng tại điểm mới của mức L (không trọng số)
        self.grads = 0

    @property
    def level(self) -> int:
        return len(self.S) - 1

    def refine(self) -> None:
        L = len(self.S)
        s = level_offsets(L)
        nw = lattice_sum(self.fn, self.X, s, torch.ones_like(s), self.batch, self.max_points)
        self.grads += self.N * int(s.numel())
        self.new.append(nw)
        self.S.append(nw if L == 0 else self.S[-1] + nw)

    def left(self, L: int) -> torch.Tensor:
        return self.S[L] / float(1 << L)

    def mid(self, L: int) -> torch.Tensor:
        if L + 1 >= len(self.S):
            raise IndexError("mid(L) cần mức L+1")
        return self.new[L + 1] / float(1 << L)

    def richardson(self, L: int) -> torch.Tensor:
        """2·left_{L+1} − left_L (= mid_L). Chỉ để báo cáo; với F không trơn (ReLU) ngoại suy không có bảo đảm."""
        return 2.0 * self.left(L + 1) - self.left(L)


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    d = float(ref.double().norm())
    return float((a.double() - ref.double()).norm()) / d if d > 0 else math.nan
