"""Hàm đồ chơi F(z) tác động trực tiếp lên z (d = 2–3), gradient bằng autograd, float64."""

from __future__ import annotations

import math
import os

import torch

from spi_study.attribute import integrate
from spi_study.seeding import generator

F64 = torch.float64
P_MAX = int(os.environ.get("SPI_TEST_PMAX", "200000"))       # số đường tối đa cho một ước lượng
P_MAX_BQ = int(os.environ.get("SPI_TEST_PMAX_BQ", "20000"))  # SPI_BQ dùng scipy trên CPU, chậm hơn


class ToyGradF:
    def __init__(self, F, d: int):
        self.F, self.d = F, d
        self.device = torch.device("cpu")
        self.n_bwd = 0

    def __call__(self, Zb: torch.Tensor) -> torch.Tensor:
        z = Zb.to(F64).expand(Zb.shape[0], self.d).clone().requires_grad_(True)
        with torch.enable_grad():
            out = self.F(z)
            (g,) = torch.autograd.grad(out.sum(), z)
        self.n_bwd += int(z.shape[0])
        return g


def f_u1(z):            # z₁²z₂
    return z[:, 0] ** 2 * z[:, 1]


def f_u4(z):            # z₁⁴z₂
    return z[:, 0] ** 4 * z[:, 1]


def f_u5(z):            # z₁³ − 3z₁z₂²  (điều hoà)
    return z[:, 0] ** 3 - 3 * z[:, 0] * z[:, 1] ** 2


B6 = torch.tensor([0.3, 0.6, 0.8], dtype=F64)


def f_u6(z):            # Σ ReLU(zᵢ − bᵢ)
    return torch.relu(z - B6.to(z.device)).sum(dim=1)


def f_u7(z):            # z₁z₂z₃
    return z[:, 0] * z[:, 1] * z[:, 2]


def endpoints(F, d: int) -> float:
    z = torch.stack([torch.zeros(d, dtype=F64), torch.ones(d, dtype=F64)])
    v = F(z)
    return float(v[1] - v[0])


def mc(sampler, F, d: int, seed: int, chunk: int = 2000, se_target: float = 1e-3, p_max: int | None = None):
    """-> (mean A (d,), se (d,), mean ΣA, se ΣA). Dừng khi SE < se_target hoặc đạt p_max đường."""
    p_max = p_max or (P_MAX_BQ if sampler.name == "spi_bq" else P_MAX)
    gen = generator(seed, "cpu")
    gF = ToyGradF(F, d)
    N = sampler.N
    parts, total = [], 0
    while True:
        p = 1 if not sampler.stochastic else chunk
        Z, dZ = sampler.draw(p, (d,), gen, F64)
        parts.append(integrate(gF, Z, dZ, (d,), batch=N * p))
        total += p
        A = torch.cat(parts)
        if not sampler.stochastic:
            z0 = torch.zeros(d, dtype=F64)
            return A.mean(0), z0, float(A.sum(1).mean()), 0.0
        se = A.std(0, unbiased=True) / math.sqrt(total)
        s = A.sum(1)
        if float(se.max()) < se_target or total >= p_max:
            return A.mean(0), se, float(s.mean()), float(s.std(unbiased=True) / math.sqrt(total))


def assert_close(est, se, ans, label=""):
    ans = torch.as_tensor(ans, dtype=F64)
    tol = 3 * se + 1e-3
    bad = (est - ans).abs() >= tol
    assert not bool(bad.any()), f"{label}: ước lượng {est.tolist()} ± {se.tolist()} vs đáp án {ans.tolist()}"
