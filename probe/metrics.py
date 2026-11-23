"""Tiện ích metric dùng chung: hoán vị phá hoà, IoU top-k với mask, Spearman, nền (đen / blur)."""

from __future__ import annotations

import torch

from .config import seed_of
from .data import MEAN, STD


def amap(A: torch.Tensor) -> torch.Tensor:
    return A.abs().sum(0)


def tie_perm(image: str, n: int, seed: int, device) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed_of("tie", seed, image))
    return torch.randperm(n, generator=g).to(device)


def _rank_top(m: torch.Tensor, k: int, perm: torch.Tensor) -> torch.Tensor:
    m = m.flatten()
    order = perm[torch.argsort(m[perm], descending=True, stable=True)]
    top = torch.zeros_like(m, dtype=torch.bool)
    top[order[:k]] = True
    return top


def iou(A, mask, perm) -> float:
    mk = mask.flatten().to(A.device)
    k = int(mk.sum())
    if k == 0:
        return float("nan")
    top = _rank_top(amap(A), k, perm)
    return int((top & mk).sum()) / int((top | mk).sum())


def spearman(A, B) -> float:
    def rk(m):
        m = m.flatten().double()
        r = torch.empty_like(m)
        r[torch.argsort(m)] = torch.arange(m.numel(), device=m.device, dtype=m.dtype)
        return r - r.mean()
    a, b = rk(amap(A)), rk(amap(B))
    return float((a * b).sum() / (a.norm() * b.norm()))


def make_base(x, kind, k=31, s=10.0):
    if kind == "black":
        m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
        sd = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
        return ((0 - m) / sd).expand_as(x).clone()
    if kind == "blur":
        import torchvision.transforms.functional as TF
        return TF.gaussian_blur(x, [k, k], [s, s])
    raise ValueError(kind)
