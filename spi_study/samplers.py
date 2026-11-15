"""
Sampler: sinh các cặp (Z_eval, dZ) cho attribution A = Σ_k ∂F(Z_eval_k) ⊙ dZ_k  (protocol 6.1–6.2).

Mọi sampler làm việc trong tọa độ chuẩn hoá z ∈ [0,1]^shape. Giao diện:

    draw(P, shape, gen, dtype)  -> Z_eval, dZ   cùng shape (N, P, *e), e = shape hoặc (1,)*len(shape)
                                                 (e thu gọn khi mọi phần tử dùng chung giá trị, ví dụ IG)
    nodes(P, shape, gen)        -> Z            (N+1, P, *shape): trạng thái tại t = k/N (cho U8, S_ε)

P là số đường độc lập trong một lần rút; ngẫu nhiên độc lập theo từng phần tử của `shape`
(kênh × pixel), trừ những đại lượng protocol yêu cầu dùng chung trong một đường
(U_r của IG_shift, V_k của SPI_BQ, điểm đánh giá trong bước của luật nút 'rand').

Luật nút (node) cho SPI_DP và COORD_MIX
---------------------------------------
'mid'  : Z_eval = Z_{k−1} + δ_k/2 (đúng protocol 6.2). CHỆCH và độ chệch KHÔNG giảm khi N tăng
         ở các bước nhảy có kích thước O(1) (atom của DP, phần tử kiểu Shapley của COORD_MIX):
         một bước nhảy của tọa độ i chỉ được tính bằng ∂ᵢF tại điểm giữa, không phải F(sau) − F(trước).
         Ví dụ U5 (z₁³ − 3z₁z₂²) với COORD_MIX λ=½ cho A₁ ≈ −0.375 thay vì −0.25; U6 (ReLU) sai.
'rand' : Z_eval = Z_{k−1} + V_k·δ_k, V_k ~ U(0,1) dùng chung cho mọi phần tử trong bước k của một
         đường. Không chệch với tích phân đường trên đoạn thẳng nối hai nút (tầng A_{α,N} ở 2.4),
         và A_{α,N} → A_α khi N → ∞. Đây là mặc định của bộ code (lệch có chủ đích khỏi protocol).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .config import num

F64 = torch.float64


def _ones(shape) -> tuple:
    return (1,) * len(tuple(shape))


def _rand(size, gen: torch.Generator, dtype=F64) -> torch.Tensor:
    return torch.rand(tuple(size), generator=gen, device=gen.device, dtype=dtype)


def gamma_ge1(k: float, size, gen: torch.Generator) -> torch.Tensor:
    """Gamma(k, 1) với k ≥ 1, Marsaglia–Tsang (2000), float64, trên thiết bị của `gen`.

    Tự cài để mọi nguồn ngẫu nhiên đi qua `gen` (tái lập được), không phụ thuộc API nội bộ."""
    if k < 1.0:
        raise ValueError(f"gamma_ge1 cần k ≥ 1, nhận {k}")
    d = k - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    total = int(np.prod(tuple(size))) if len(tuple(size)) else 1
    out = torch.empty(total, dtype=F64, device=gen.device)
    todo = torch.arange(total, device=gen.device)
    while todo.numel() > 0:
        n = todo.numel()
        x = torch.randn(n, generator=gen, device=gen.device, dtype=F64)
        u = torch.rand(n, generator=gen, device=gen.device, dtype=F64)
        v = 1.0 + c * x
        pos = v > 0
        v3 = torch.where(pos, v * v * v, torch.ones_like(v))
        ok = pos & (torch.log(u) < 0.5 * x * x + d - d * v3 + d * torch.log(v3))
        out[todo[ok]] = d * v3[ok]
        todo = todo[~ok]
    return out.view(tuple(size))


class Sampler:
    name = "base"
    stochastic = True

    def __init__(self, N: int):
        self.N = int(N)

    @property
    def h(self) -> float:
        """Var Z(t) = h·t(1−t)."""
        raise NotImplementedError

    def draw(self, P: int, shape, gen: torch.Generator, dtype=torch.float32):
        raise NotImplementedError

    def nodes(self, P: int, shape, gen: torch.Generator) -> torch.Tensor:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# IG
# ─────────────────────────────────────────────────────────────────────────────

class IGMid(Sampler):
    """t_k = (k + ½)/N, Z = t·1, dZ = 1/N."""
    name = "ig_mid"
    stochastic = False

    @property
    def h(self) -> float:
        return 0.0

    def draw(self, P, shape, gen, dtype=torch.float32):
        N, dev = self.N, gen.device
        t = (torch.arange(N, device=dev, dtype=F64) + 0.5) / N
        Z = t.view((N, 1) + _ones(shape)).expand((N, P) + _ones(shape)).to(dtype)
        return Z, torch.full(Z.shape, 1.0 / N, device=dev, dtype=dtype)

    def nodes(self, P, shape, gen):
        N = self.N
        t = torch.arange(N + 1, device=gen.device, dtype=F64) / N
        return t.view((N + 1, 1) + _ones(shape)).expand((N + 1, P) + tuple(shape))


class IGShift(IGMid):
    """Mỗi đường rút U_r ~ U(0,1) dùng chung cho mọi phần tử; t_{r,k} = (k + U_r)/N. Không chệch với IG."""
    name = "ig_shift"
    stochastic = True

    def draw(self, P, shape, gen, dtype=torch.float32):
        N, dev = self.N, gen.device
        U = _rand((P,), gen)
        t = (torch.arange(N, device=dev, dtype=F64).view(N, 1) + U.view(1, P)) / N
        Z = t.view((N, P) + _ones(shape)).to(dtype)
        return Z, torch.full(Z.shape, 1.0 / N, device=dev, dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# SPI: quá trình Dirichlet trên lưới (số gia Dirichlet chính xác)
# ─────────────────────────────────────────────────────────────────────────────

def _node_offsets(node: str, N: int, P: int, shape, gen) -> torch.Tensor | float:
    if node == "mid":
        return 0.5
    if node == "rand":
        return _rand((N, P) + _ones(shape), gen)
    raise ValueError(f"node phải là 'mid' hoặc 'rand', nhận '{node}'")


class SPIDP(Sampler):
    """G_i ~ DP(α, U[0,1]) độc lập theo phần tử; số gia trên N ô bằng nhau ~ Dirichlet(α/N, …).

    Rút trong log-space (đúng với mọi a = α/N > 0):
        G ~ Gamma(a+1), V ~ U(0,1), L = log G + log(V)/a, δ = softmax(L) theo trục thời gian.
    """
    name = "spi_dp"

    def __init__(self, alpha: float, N: int, node: str = "rand"):
        super().__init__(N)
        self.alpha = float(alpha)
        self.node = node
        if self.alpha <= 0:
            raise ValueError("alpha phải > 0")

    @property
    def h(self) -> float:
        return 1.0 / (self.alpha + 1.0)

    def increments(self, P, shape, gen) -> torch.Tensor:
        N = self.N
        a = self.alpha / N
        size = (N, P) + tuple(shape)
        logg = torch.log(gamma_ge1(a + 1.0, size, gen))
        logv = torch.log(_rand(size, gen).clamp_min(1e-300))
        return torch.softmax(logg + logv / a, dim=0)            # (N, P, *shape), float64

    def draw(self, P, shape, gen, dtype=torch.float32):
        delta = self.increments(P, shape, gen)
        csum = torch.cumsum(delta, dim=0).clamp_(max=1.0)
        z_prev = torch.cat([torch.zeros_like(delta[:1]), csum[:-1]], dim=0)
        s = _node_offsets(self.node, self.N, P, shape, gen)
        Z = z_prev + s * delta
        return Z.to(dtype), delta.to(dtype)

    def nodes(self, P, shape, gen):
        delta = self.increments(P, shape, gen)
        Z = torch.cat([torch.zeros_like(delta[:1]), torch.cumsum(delta, dim=0)], dim=0).clamp_(0.0, 1.0)
        Z[-1] = 1.0
        return Z


# ─────────────────────────────────────────────────────────────────────────────
# Hỗn hợp IG/Shapley theo tọa độ
# ─────────────────────────────────────────────────────────────────────────────

class CoordMix(Sampler):
    """Theo phần tử: B_i ~ Bernoulli(λ), T_i ~ U(0,1), c_i = floor(T_i·N).
    B_i = 0 (kiểu IG): z đi đều; B_i = 1 (kiểu Shapley): nhảy 0 → 1 trong bước c_i.
    Phân phối biên tại t: (1−λ)·δ_t + λ·[(1−t)·δ_0 + t·δ_1]."""
    name = "coord_mix"

    def __init__(self, lam: float, N: int, node: str = "rand"):
        super().__init__(N)
        self.lam = float(lam)
        self.node = node
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam phải trong [0, 1]")

    @property
    def h(self) -> float:
        return self.lam

    def _coords(self, P, shape, gen):
        e = (P,) + tuple(shape)
        B = _rand(e, gen) < self.lam
        T = _rand(e, gen)
        c = torch.floor(T * self.N).long().clamp_(max=self.N - 1)
        return B, c

    def draw(self, P, shape, gen, dtype=torch.float32):
        N, dev = self.N, gen.device
        B, c = self._coords(P, shape, gen)
        s = _node_offsets(self.node, N, P, shape, gen)
        if not torch.is_tensor(s):
            s = torch.full((N, P) + _ones(shape), float(s), device=dev, dtype=F64)
        k = torch.arange(N, device=dev).view((N, 1) + _ones(shape))
        z_ig = (k.to(F64) + s) / N                                     # (N, P, 1, …)
        jump = k == c.unsqueeze(0)                                     # (N, P, *shape)
        z_sh = torch.where(jump, s.expand(jump.shape), (k > c.unsqueeze(0)).to(F64))
        Bx = B.unsqueeze(0)
        Z = torch.where(Bx, z_sh, z_ig.expand(jump.shape))
        dZ = torch.where(Bx, jump.to(F64), torch.full((), 1.0 / N, device=dev, dtype=F64))
        return Z.to(dtype), dZ.to(dtype)

    def nodes(self, P, shape, gen):
        N, dev = self.N, gen.device
        B, c = self._coords(P, shape, gen)
        k = torch.arange(N + 1, device=dev).view((N + 1, 1) + _ones(shape))
        z_ig = (k.to(F64) / N).expand((N + 1, P) + tuple(shape))
        z_sh = (c.unsqueeze(0) < k).to(F64)
        return torch.where(B.unsqueeze(0), z_sh, z_ig)


# ─────────────────────────────────────────────────────────────────────────────
# Beta-quantile (tuỳ chọn, không chệch với population)
# ─────────────────────────────────────────────────────────────────────────────

class SPIBQ(Sampler):
    """Z_i(τ) = Q_{α,τ}(U_i) (quantile của Beta(ατ, α(1−τ))), U_i ~ U(0,1) theo phần tử.
    Mỗi ô k: τ_k = (k + V_k)/N, V_k dùng chung cho mọi phần tử. Ż bằng sai phân trung tâm.
    Tính bằng scipy.special.betaincinv trên CPU, float64. Chỉ dùng α ≥ 10."""
    name = "spi_bq"

    def __init__(self, alpha: float, N: int, fd_eps: float = 1e-4):
        super().__init__(N)
        self.alpha = float(alpha)
        self.fd_eps = float(fd_eps)

    @property
    def h(self) -> float:
        return 1.0 / (self.alpha + 1.0)

    def _Q(self, t: np.ndarray, U: np.ndarray) -> np.ndarray:
        from scipy.special import betaincinv
        return betaincinv(self.alpha * t, self.alpha * (1.0 - t), U)

    def draw(self, P, shape, gen, dtype=torch.float32):
        N, dev = self.N, gen.device
        U = _rand((P,) + tuple(shape), gen).cpu().numpy()
        V = _rand((N, P), gen).cpu().numpy()
        tau = (np.arange(N, dtype=np.float64)[:, None] + V) / N
        tb = tau.reshape((N, P) + _ones(shape))
        Ub = U[None]
        Z = self._Q(tb, Ub)
        tp = np.minimum(tb + self.fd_eps, 1.0 - 1e-12)
        tm = np.maximum(tb - self.fd_eps, 1e-12)
        zdot = (self._Q(tp, Ub) - self._Q(tm, Ub)) / (tp - tm)
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device=dev, dtype=dtype)
        return to(Z), to(zdot / N)

    def nodes(self, P, shape, gen):
        N = self.N
        U = _rand((P,) + tuple(shape), gen).cpu().numpy()
        t = (np.arange(1, N, dtype=np.float64) / N).reshape((N - 1, 1) + _ones(shape))
        inner = self._Q(t, U[None])
        Z = np.concatenate([np.zeros((1,) + U.shape), inner, np.ones((1,) + U.shape)], axis=0)
        return torch.from_numpy(Z).to(gen.device)


# ─────────────────────────────────────────────────────────────────────────────
# Đường thẳng IG–SPI(1): chỉ dùng cho S_ε (attribution LINE tính từ tham chiếu)
# ─────────────────────────────────────────────────────────────────────────────

class LineMix(Sampler):
    """Mỗi đường: với xác suất s dùng một đường SPI_DP(1), ngược lại đường IG (biến chọn dùng chung)."""
    name = "line"

    def __init__(self, s: float, N: int):
        super().__init__(N)
        self.s = float(s)

    @property
    def h(self) -> float:
        return self.s / 2.0

    def nodes(self, P, shape, gen):
        sel = _rand((P,), gen) < self.s
        z_spi = SPIDP(1.0, self.N).nodes(P, shape, gen)
        z_ig = IGMid(self.N).nodes(P, shape, gen)
        return torch.where(sel.view((1, P) + _ones(shape)), z_spi, z_ig)

    def draw(self, P, shape, gen, dtype=torch.float32):
        raise NotImplementedError("LINE được tính từ các tham chiếu đã hội tụ: A_s = (1−s)·A_IG + s·A_SPI(1)")


# ─────────────────────────────────────────────────────────────────────────────

def make_sampler(name: str, params: dict, N: int) -> Sampler:
    if name == "ig_mid":
        return IGMid(N)
    if name == "ig_shift":
        return IGShift(N)
    if name == "spi_dp":
        return SPIDP(num(params["alpha"]), N, params.get("node", "rand"))
    if name == "spi_bq":
        return SPIBQ(num(params["alpha"]), N, num(params.get("fd_eps", 1e-4)))
    if name == "coord_mix":
        return CoordMix(num(params["lam"]), N, params.get("node", "rand"))
    if name == "line":
        return LineMix(num(params["s"]), N)
    raise KeyError(f"sampler không biết: '{name}'")


def s_eps(sampler: Sampler, eps: float, P: int, shape, gen: torch.Generator, chunk: int = 0) -> float:
    """S_ε = ∫₀¹ P[Z(t) < ε hoặc Z(t) > 1−ε] dt, Monte Carlo từ chính sampler (trapezoid theo nút)."""
    chunk = chunk or P
    acc, done = None, 0
    while done < P:
        p = min(chunk, P - done)
        Z = sampler.nodes(p, shape, gen)
        ind = ((Z < eps) | (Z > 1.0 - eps)).to(F64)
        v = ind.flatten(1).mean(dim=1) * p
        acc = v if acc is None else acc + v
        done += p
    prob = acc / P
    return float(torch.trapezoid(prob, dx=1.0 / sampler.N))
