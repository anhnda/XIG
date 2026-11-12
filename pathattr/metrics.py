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
                       batch_size: int = 16, normalize: bool = True) -> dict:
    """Same protocol as utility.compute_insertion_deletion, baseline explicit.

    normalize=True : legacy — (AUC − f(base)) / |f(x) − f(base)| on the logit
                     (unstable when f(base) ≈ f(x), e.g. a blur baseline).
    normalize=False: raw AUC of the model output (use with a softmax-probability
                     model: RISE convention, bounded in [0, 1]).
    """
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
    if not normalize:
        return {"ins": ins_auc, "del": del_auc, "insdel": ins_auc - del_auc}
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


# ─────────────────────────────────────────────────────────────────────────────
# infidelity (Yeh et al., NeurIPS 2019) — Gaussian perturbation family
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infidelity(model: nn.Module, x: torch.Tensor, attr: torch.Tensor, sigma: float = 0.1,
               n: int = 64, gen: Optional[torch.Generator] = None, batch_size: int = 16) -> dict:
    """I ~ N(0, σ² I) in the (normalised) input space,  v = f(x) − f(x − I),  u = Iᵀa.
        infd_raw  = E[(u − v)²]
        infd      = min_β E[(βu − v)²] / E[v²]   ∈ [0, 1]   (scale-free; lower is better)
    A Gaussian-kernel metric: the counterpart of ins/del's binary-mask kernel."""
    fx = float(model(x).reshape(-1)[0])
    a = attr.reshape(-1).double()
    us, vs = [], []
    done = 0
    while done < n:
        b = min(batch_size, n - done)
        I = torch.randn((b,) + tuple(x.shape[1:]), generator=gen, dtype=torch.float64)
        I = (I * sigma).to(device=x.device, dtype=x.dtype)
        fv = model(x - I).double().reshape(-1)
        vs.append(fx - fv.cpu())
        us.append((I.reshape(b, -1).double().cpu() @ a.cpu()))
        done += b
    u, v = torch.cat(us), torch.cat(vs)
    raw = float(((u - v) ** 2).mean())
    uu = float((u * u).sum())
    beta = float((u * v).sum()) / uu if uu > 0 else 0.0
    vv = float((v * v).mean())
    infd = float(((beta * u - v) ** 2).mean()) / vv if vv > 0 else float("nan")
    return {"infd": infd, "infd_raw": raw, "infd_beta": beta}


# ─────────────────────────────────────────────────────────────────────────────
# ROAD (Rong et al., ICML 2022) — approximate noisy linear imputation
# ─────────────────────────────────────────────────────────────────────────────

_ROAD_K = torch.tensor([[1 / 12, 1 / 6, 1 / 12],
                        [1 / 6, 0.0, 1 / 6],
                        [1 / 12, 1 / 6, 1 / 12]], dtype=torch.float64)


def impute_linear(x: torch.Tensor, removed: torch.Tensor, iters: int = 200,
                  noise: float = 0.01, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Removed pixels (H, W bool) ← weighted mean of their 8 neighbours (ROAD weights
    1/6 direct, 1/12 diagonal), solved by Jacobi iteration (approximation of ROAD's
    exact sparse solve), then + N(0, noise²). Kept pixels are unchanged."""
    _, C, H, W = x.shape
    k = _ROAD_K.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    ones = torch.ones(1, C, H, W, device=x.device, dtype=x.dtype)
    norm = F.conv2d(ones, k, padding=1, groups=C)                       # border-aware weight sum
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
    """Target-class probability after removing a fraction of pixels and imputing them.
        road_morf: mean prob, most relevant first  (lower is better)
        road_lerf: mean prob, least relevant first (higher is better)
        road     : lerf − morf                      (higher is better)
    Uses the probability model, not the logit (ROAD reports accuracy/probability)."""
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
