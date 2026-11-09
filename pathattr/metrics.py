"""
Evaluation metrics.

Baselines (inputs are ImageNet-normalised):
    black  → (0 − mean)/std      true black pixels
    gray   → 0                   dataset-mean colour (what the old batch mode used)
    white  → (1 − mean)/std
    blur   → Gaussian blur of x  (RISE-style insertion start)

Insertion/deletion follow the legacy definition (logit of the target class,
normalised by f(x) − f(baseline), trapezoid AUC over n_steps) so numbers are
comparable with the existing tables; only the baseline is now explicit.

Localisation definitions (assumed — the revision code is lost):
    overlap   Σ_{mask} a / Σ a           a = Σ_c |attr_c|  (attribution energy inside mask)
    iou       IoU(top-|mask| pixels of a, mask)
    pointing  1 if argmax a lies inside the mask
    checker   high-frequency energy of a outside the mask:
              ‖a − avgpool3(a)‖² / ‖a‖²  (outside mask; whole image if no mask)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _stats(x: torch.Tensor):
    m = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return m, s


def make_baseline(x: torch.Tensor, kind: str, blur_kernel: int = 31,
                  blur_sigma: float = 10.0) -> torch.Tensor:
    m, s = _stats(x)
    if kind == "black":
        return ((0.0 - m) / s).expand_as(x).clone()
    if kind == "gray":
        return torch.zeros_like(x)
    if kind == "white":
        return ((1.0 - m) / s).expand_as(x).clone()
    if kind == "blur":
        return TF.gaussian_blur(x, [blur_kernel, blur_kernel], [blur_sigma, blur_sigma])
    raise ValueError(f"unknown baseline '{kind}'")


def attribution_map(attr: torch.Tensor) -> torch.Tensor:
    """(1, C, H, W) -> (H, W): Σ_c |attr|, the ranking used everywhere."""
    return attr[0].abs().sum(dim=0)


@torch.no_grad()
def insertion_deletion(model: nn.Module, x: torch.Tensor, base: torch.Tensor,
                       attr: torch.Tensor, n_steps: int = 100,
                       batch_size: int = 16) -> dict:
    """Same protocol as utility.compute_insertion_deletion, baseline explicit."""
    device = x.device
    _, C, H, W = x.shape
    n_pix = H * W
    order = torch.argsort(attribution_map(attr).flatten(), descending=True)
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

    f_x = float(model(x).reshape(-1)[0])
    f_b = float(model(base).reshape(-1)[0])
    dx = 1.0 / n_steps
    ins_auc = float((ins[:-1] + ins[1:]).sum() * dx / 2)
    del_auc = float((dele[:-1] + dele[1:]).sum() * dx / 2)
    rng = abs(f_x - f_b)
    if rng > 1e-12:
        ins_n, del_n = (ins_auc - f_b) / rng, (del_auc - f_b) / rng
    else:
        ins_n = del_n = float("nan")
    return {"ins": ins_n, "del": del_n, "insdel": ins_n - del_n}


def localisation(attr: torch.Tensor, mask: torch.Tensor) -> dict:
    """mask: (H, W) bool."""
    a = attribution_map(attr).double()
    mask = mask.to(device=a.device, dtype=torch.bool)
    total = float(a.sum())
    overlap = float(a[mask].sum()) / total if total > 0 else float("nan")
    k = int(mask.sum())
    if k == 0:
        return {"overlap": overlap, "iou": float("nan"), "pointing": float("nan")}
    top = torch.zeros(a.numel(), dtype=torch.bool, device=a.device)
    top[torch.topk(a.flatten(), k).indices] = True
    top = top.view_as(mask)
    inter = float((top & mask).sum())
    union = float((top | mask).sum())
    pointing = float(mask.flatten()[int(a.flatten().argmax())])
    return {"overlap": overlap, "iou": inter / union, "pointing": pointing}


def checker_energy(attr: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    a = attribution_map(attr).double()[None, None]
    smooth = F.avg_pool2d(a, 3, stride=1, padding=1, count_include_pad=False)
    hf = (a - smooth)[0, 0]
    a = a[0, 0]
    region = torch.ones_like(a, dtype=torch.bool) if mask is None else ~mask.to(a.device).bool()
    den = float((a[region] ** 2).sum())
    return float((hf[region] ** 2).sum()) / den if den > 0 else float("nan")
