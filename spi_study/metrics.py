"""
Metric (protocol 2.5, 6.4).

Bản đồ m(p) = Σ_c |A_{p,c}|.

iou            top-k pixel theo m, k = diện tích mask, hoà phá bằng hoán vị ngẫu nhiên cố định theo ảnh.
               Có `valid` (trimap Pet): pixel biên bị loại khỏi ứng viên, tử số và mẫu số.
insdelp_<b>    copy nguyên `pathattr.metrics.insertion_deletion(normalize=False)` của code đã tái lập:
               xác suất softmax lớp đích, AUC hình thang trên n_steps, thứ hạng theo m (argsort,
               không phá hoà, như code cũ); baseline blur = Gaussian blur kernel 31, σ = 10.
road           copy nguyên `pathattr.metrics.road` (ROAD xấp xỉ, nội suy tuyến tính có nhiễu, Jacobi
               200 vòng, noise 0.01, các tỉ lệ 0.1…0.9, xác suất softmax). road = lerf − morf.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .data import IMAGENET_MEAN, IMAGENET_STD
from .seeding import seed_of


def _stats(x: torch.Tensor):
    m = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return m, s


def make_baseline(x: torch.Tensor, kind: str, blur_kernel: int = 31, blur_sigma: float = 10.0) -> torch.Tensor:
    m, s = _stats(x)
    if kind == "black":
        return ((0.0 - m) / s).expand_as(x).clone()
    if kind == "gray":
        return torch.zeros_like(x)
    if kind == "white":
        return ((1.0 - m) / s).expand_as(x).clone()
    if kind == "blur":
        return TF.gaussian_blur(x, [blur_kernel, blur_kernel], [blur_sigma, blur_sigma])
    raise ValueError(f"baseline không biết: '{kind}'")


def attribution_map(attr: torch.Tensor) -> torch.Tensor:
    """(3, H, W) hoặc (1, 3, H, W) -> (H, W)."""
    a = attr[0] if attr.dim() == 4 else attr
    return a.abs().sum(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────────────────────────────

def tie_perm(image: str, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed_of("iou_tiebreak", seed, image) % (2 ** 63))
    return torch.randperm(n, generator=g)


def iou_topk(attr: torch.Tensor, mask: torch.Tensor, perm: torch.Tensor,
             valid: Optional[torch.Tensor] = None) -> float:
    m = attribution_map(attr).double().flatten().cpu()
    mk = mask.flatten().bool().cpu()
    v = torch.ones_like(mk) if valid is None else valid.flatten().bool().cpu()
    cand = perm[v[perm]]                                     # pixel hợp lệ, theo thứ tự hoán vị
    k = int((mk & v).sum())
    if k == 0:
        return float("nan")
    order = cand[torch.argsort(m[cand], descending=True, stable=True)]
    top = torch.zeros_like(mk)
    top[order[:k]] = True
    inter = int((top & mk & v).sum())
    union = int(((top | mk) & v).sum())
    return inter / union if union else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# tỉ lệ năng lượng trong mask (khử nhiễu Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────────

def energy_frac(attr: torch.Tensor, mask: torch.Tensor, valid: Optional[torch.Tensor] = None,
                var: Optional[torch.Tensor] = None) -> float:
    """E = Σ_{p∈mask} Σ_c (A² − V) / Σ_p Σ_c (A² − V).

    A là trung bình của các sub-estimate, V là phương sai MC theo phần tử của nó (s²/n_sub). Vì
    E[Â²] = A² + V nên tử và mẫu là ước lượng không chệch của năng lượng population. var=None cho
    bản thô (bị nhiễu kéo về tỉ lệ diện tích). Pixel ngoài `valid` (biên trimap) bị loại khỏi cả hai."""
    a = attr[0] if attr.dim() == 4 else attr
    e = a.double().pow(2)
    if var is not None:
        e = e - var.double().to(e.device)
    e = e.sum(dim=0).cpu()                                    # (H, W)
    mk = mask.bool().cpu()
    if valid is not None:
        v = valid.bool().cpu()
        e, mk = e[v], mk[v]
    tot = float(e.sum())
    return float(e[mk].sum()) / tot if tot > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# insertion / deletion (copy từ code đã tái lập)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def insertion_deletion(model: nn.Module, x: torch.Tensor, base: torch.Tensor, attr: torch.Tensor,
                       n_steps: int = 100, batch_size: int = 16, normalize: bool = False) -> dict:
    if attr.dim() == 3:
        attr = attr.unsqueeze(0)
    device = x.device
    _, C, H, W = x.shape
    n_pix = H * W
    order = torch.argsort(attribution_map(attr).flatten().to(device), descending=True)
    ranks = torch.empty(n_pix, dtype=torch.long, device=device)
    ranks[order] = torch.arange(n_pix, device=device)
    counts = torch.linspace(0, n_pix, n_steps + 1, device=device).long()
    S = counts.shape[0]
    ins = torch.empty(S, device=device, dtype=torch.float64)
    dele = torch.empty(S, device=device, dtype=torch.float64)
    for i0 in range(0, S, batch_size):
        i1 = min(i0 + batch_size, S)
        m = (ranks.view(1, -1) < counts[i0:i1].view(-1, 1)).view(i1 - i0, 1, H, W)
        m = m.expand(i1 - i0, C, H, W)
        ins[i0:i1] = model(torch.where(m, x.expand_as(m), base.expand_as(m))).double()
        dele[i0:i1] = model(torch.where(m, base.expand_as(m), x.expand_as(m))).double()
    dx = 1.0 / n_steps
    ins_auc = float((ins[:-1] + ins[1:]).sum() * dx / 2)
    del_auc = float((dele[:-1] + dele[1:]).sum() * dx / 2)
    if not normalize:
        return {"ins": ins_auc, "del": del_auc, "insdel": ins_auc - del_auc}
    f_x = float(model(x).reshape(-1)[0])
    f_b = float(model(base).reshape(-1)[0])
    rng = abs(f_x - f_b)
    if rng > 1e-12:
        ins_n, del_n = (ins_auc - f_b) / rng, (del_auc - f_b) / rng
    else:
        ins_n = del_n = float("nan")
    return {"ins": ins_n, "del": del_n, "insdel": ins_n - del_n}


# ─────────────────────────────────────────────────────────────────────────────
# ROAD (copy từ code đã tái lập)
# ─────────────────────────────────────────────────────────────────────────────

_ROAD_K = torch.tensor([[1 / 12, 1 / 6, 1 / 12],
                        [1 / 6, 0.0, 1 / 6],
                        [1 / 12, 1 / 6, 1 / 12]], dtype=torch.float64)


def impute_linear(x: torch.Tensor, removed: torch.Tensor, iters: int = 200, noise: float = 0.01,
                  gen: Optional[torch.Generator] = None) -> torch.Tensor:
    _, C, H, W = x.shape
    k = _ROAD_K.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    ones = torch.ones(1, C, H, W, device=x.device, dtype=x.dtype)
    norm = F.conv2d(ones, k, padding=1, groups=C)
    rem = removed.to(x.device).view(1, 1, H, W).expand(1, C, H, W)
    keep_mean = x[~rem].mean() if bool((~rem).any()) else x.mean()
    cur = torch.where(rem, keep_mean.expand_as(x), x)
    for _ in range(iters):
        avg = F.conv2d(cur, k, padding=1, groups=C) / norm
        cur = torch.where(rem, avg, x)
    if noise > 0:
        eps = torch.randn(x.shape, generator=gen, dtype=torch.float64).to(device=x.device, dtype=x.dtype)
        cur = torch.where(rem, cur + noise * eps, cur)
    return cur


@torch.no_grad()
def road(prob_model: nn.Module, x: torch.Tensor, attr: torch.Tensor,
         fracs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), iters: int = 200,
         noise: float = 0.01, gen: Optional[torch.Generator] = None) -> dict:
    if attr.dim() == 3:
        attr = attr.unsqueeze(0)
    _, C, H, W = x.shape
    order = torch.argsort(attribution_map(attr).flatten(), descending=True).cpu()
    n = H * W
    morf, lerf = [], []
    for fr in fracs:
        k = int(round(fr * n))
        rm = torch.zeros(n, dtype=torch.bool)
        rm[order[:k]] = True
        morf.append(float(prob_model(impute_linear(x, rm.view(H, W), iters, noise, gen)).reshape(-1)[0]))
        rl = torch.zeros(n, dtype=torch.bool)
        rl[order[n - k:]] = True
        lerf.append(float(prob_model(impute_linear(x, rl.view(H, W), iters, noise, gen)).reshape(-1)[0]))
    m, l_ = sum(morf) / len(morf), sum(lerf) / len(lerf)
    return {"road_morf": m, "road_lerf": l_, "road": l_ - m}


# ─────────────────────────────────────────────────────────────────────────────
# tất cả metric cho một bản đồ
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """Giữ các thứ cố định theo ảnh (mô hình xác suất, baseline đánh giá, mask, hoán vị phá hoà)."""

    def __init__(self, mcfg: dict, backbone: nn.Module, x: torch.Tensor, target: int, image: str,
                 index: int, mask: torch.Tensor, valid, D: float, seed: int, expensive: bool = True):
        from .model import TargetProb
        self.cfg, self.x, self.mask, self.valid, self.D = mcfg, x, mask, valid, D
        self.index, self.expensive = int(index), expensive
        self.prob = TargetProb(backbone, target)
        ic = mcfg["insdel"]
        self.bases = {b: make_baseline(x, b, ic["blur_kernel"], ic["blur_sigma"]) for b in ic["bases"]}
        self.perm = tie_perm(image, mask.numel(), seed)

    def __call__(self, A: torch.Tensor) -> dict:
        A = A.to(self.x.device)
        row = {"iou": iou_topk(A, self.mask, self.perm, self.valid),
               "residual": (self.D - float(A.double().sum())) / self.D if self.D else float("nan")}
        if not self.expensive:
            return row
        ic, rc = self.cfg["insdel"], self.cfg["road"]
        for b, base in self.bases.items():
            s = insertion_deletion(self.prob, self.x, base, A, n_steps=ic["n_steps"],
                                   batch_size=ic["batch"], normalize=False)
            row[f"insp_{b}"], row[f"delp_{b}"], row[f"insdelp_{b}"] = s["ins"], s["del"], s["insdel"]
        if rc.get("enabled", True):
            gen = torch.Generator().manual_seed(7_919 + self.index)     # cùng nhiễu cho mọi phương pháp
            row.update(road(self.prob, self.x, A, tuple(rc["fracs"]), rc["iters"], rc["noise"], gen))
        return row