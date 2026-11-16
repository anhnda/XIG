"""
Metric. Bản đồ hiển thị m(p) = Σ_c |A_{p,c}|.

Với phương pháp ngẫu nhiên (SPI), A là trung bình các sub-estimate và V là phương sai MC theo phần tử;
các đại lượng bậc hai được khử nhiễu bằng cách trừ V (E[Â²] = A² + V). Phương pháp tất định: V = 0.

energy_mask    E = Σ_{mask} Σ_c (A² − V) / Σ Σ_c (A² − V)
energy_total   Σ (A² − V)
iou            top-k theo m (k = diện tích mask) so với mask; hoà phá bằng hoán vị cố định
cancel         ‖A‖₁ / |ΣA|  (1 = không triệt tiêu; chỉ mô tả, ‖A‖₁ của SPI còn bị nhiễu MC thổi phồng)
cos_deb        ⟨A,B⟩ / √((‖A‖² − ΣV_A)(‖B‖² − ΣV_B))  (cosine có dấu, khử nhiễu)
spearman       tương quan hạng giữa m_A và m_B
topk           |top_k(m_A) ∩ top_k(m_B)| / k
insdel_<base>  AUC insertion − AUC deletion của xác suất softmax lớp đích (baseline blur hoặc đen)
road           LeRF − MoRF (ROAD xấp xỉ, nội suy tuyến tính có nhiễu)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .config import seed_of
from .data import MEAN, STD
from .paths import gaussian_blur


def amap(A: torch.Tensor) -> torch.Tensor:
    return A.abs().sum(0)


def _V(A, V):
    return torch.zeros_like(A) if V is None else V.to(A.device)


def energy_mask(A, V, mask) -> float:
    e = (A.double() ** 2 - _V(A, V).double()).sum(0)
    tot = float(e.sum())
    return float(e[mask.to(e.device)].sum()) / tot if tot > 0 else float("nan")


def energy_total(A, V) -> float:
    return float((A.double() ** 2 - _V(A, V).double()).sum())


def cancel(A) -> float:
    s = abs(float(A.double().sum()))
    return float(A.double().abs().sum()) / s if s > 0 else float("inf")


def cos_deb(A, VA, B, VB) -> float:
    a2 = energy_total(A, VA)
    b2 = energy_total(B, VB)
    if a2 <= 0 or b2 <= 0:
        return float("nan")
    return float((A.double() * B.double()).sum()) / math.sqrt(a2 * b2)


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


def topk_overlap(A, B, frac, perm) -> float:
    k = max(1, int(frac * A.shape[-1] * A.shape[-2]))
    return int((_rank_top(amap(A), k, perm) & _rank_top(amap(B), k, perm)).sum()) / k


def spearman(A, B) -> float:
    def rk(m):
        m = m.flatten().double()
        r = torch.empty_like(m)
        r[torch.argsort(m)] = torch.arange(m.numel(), device=m.device, dtype=m.dtype)
        return r - r.mean()
    a, b = rk(amap(A)), rk(amap(B))
    return float((a * b).sum() / (a.norm() * b.norm()))


# ─────────────────────────────────────────────────────────────────────────────
# insertion / deletion, ROAD (cùng cài đặt với các đợt trước)
# ─────────────────────────────────────────────────────────────────────────────

def make_base(x, kind, k=31, s=10.0):
    if kind == "black":
        m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
        sd = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
        return ((0 - m) / sd).expand_as(x).clone()
    if kind == "blur":
        import torchvision.transforms.functional as TF
        return TF.gaussian_blur(x, [k, k], [s, s])
    raise ValueError(kind)


@torch.no_grad()
def insdel(prob, x, base, A, n_steps, batch) -> float:
    _, C, H, W = x.shape
    n = H * W
    order = torch.argsort(amap(A).flatten().to(x.device), descending=True)
    ranks = torch.empty(n, dtype=torch.long, device=x.device)
    ranks[order] = torch.arange(n, device=x.device)
    counts = torch.linspace(0, n, n_steps + 1, device=x.device).long()
    ins, dele = [], []
    for i in range(0, counts.numel(), batch):
        c = counts[i:i + batch]
        m = (ranks.view(1, -1) < c.view(-1, 1)).view(-1, 1, H, W).expand(-1, C, H, W)
        ins.append(prob(torch.where(m, x.expand_as(m), base.expand_as(m))).double())
        dele.append(prob(torch.where(m, base.expand_as(m), x.expand_as(m))).double())
    ins, dele = torch.cat(ins), torch.cat(dele)
    auc = lambda v: float((v[:-1] + v[1:]).sum() / (2 * n_steps))
    return auc(ins) - auc(dele)


_K = torch.tensor([[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0.0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]], dtype=torch.float64)


def _impute(x, removed, iters, noise, gen):
    _, C, H, W = x.shape
    k = _K.to(x.device, x.dtype).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    norm = F.conv2d(torch.ones(1, C, H, W, device=x.device, dtype=x.dtype), k, padding=1, groups=C)
    rem = removed.to(x.device).view(1, 1, H, W).expand(1, C, H, W)
    cur = torch.where(rem, x[~rem].mean().expand_as(x), x)
    for _ in range(iters):
        cur = torch.where(rem, F.conv2d(cur, k, padding=1, groups=C) / norm, x)
    eps = torch.randn(x.shape, generator=gen, dtype=torch.float64).to(x.device, x.dtype)
    return torch.where(rem, cur + noise * eps, cur)


@torch.no_grad()
def road(prob, x, A, fracs, iters, noise, seed) -> float:
    _, C, H, W = x.shape
    n = H * W
    order = torch.argsort(amap(A).flatten(), descending=True).cpu()
    gen = torch.Generator().manual_seed(int(seed))
    mo, le = [], []
    for fr in fracs:
        k = int(round(fr * n))
        rm = torch.zeros(n, dtype=torch.bool); rm[order[:k]] = True
        mo.append(float(prob(_impute(x, rm.view(H, W), iters, noise, gen))[0]))
        rl = torch.zeros(n, dtype=torch.bool); rl[order[n - k:]] = True
        le.append(float(prob(_impute(x, rl.view(H, W), iters, noise, gen))[0]))
    return sum(le) / len(le) - sum(mo) / len(mo)


def quality(A, V, mask, perm, prob, x, mcfg, seed, full: bool = True) -> dict:
    row = {"energy_mask": energy_mask(A, V, mask), "energy_total": energy_total(A, V),
           "iou": iou(A, mask, perm), "cancel": cancel(A), "sumA": float(A.double().sum())}
    if full:
        ic = mcfg["insdel"]
        for b in ic["bases"]:
            row[f"insdel_{b}"] = insdel(prob, x, make_base(x, b, ic["blur_kernel"], ic["blur_sigma"]),
                                        A, ic["n_steps"], ic["batch"])
        rc = mcfg["road"]
        if rc.get("enabled", True):
            row["road"] = road(prob, x, A, rc["fracs"], rc["iters"], rc["noise"], seed)
    return row
