"""
Feature groups and masking games for images.

Partitions (label maps (H, W), labels 0 … G−1):
    "slic:<n>[:<compactness>]"   SLIC superpixels (needs scikit-image)
    "grid:<cell>[:<dy>:<dx>]"    square cells of <cell> px, optional shift
Several partitions can be stacked (multi-scale / overlapping groups); each mask
row lives on one partition but is stored at pixel level, so stacking is free.

Mask families (rows m ∈ {0,1}^G on one partition):
    shapley    coalition size s ∈ {1 … G−1} with p(s) ∝ 1/(s(G−s)), then a
               uniform subset of that size (= sampling from the Shapley kernel);
               paired with its complement (antithetic). Regression weight = 1:
               do NOT multiply by the kernel again.
    bernoulli  every group present independently with prob p (default ½)

Game value (same baseline as the paths!):
    y_m = f(x ⊙ m + x' ⊙ (1 − m)) − f(x')
so the empty coalition gives 0 and the full one gives D = f(x) − f(x').
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from .metrics import IMAGENET_MEAN, IMAGENET_STD
from .oracle import Oracle


# ─────────────────────────────────────────────────────────────────────────────
# partitions
# ─────────────────────────────────────────────────────────────────────────────

def relabel(lbl: torch.Tensor) -> tuple[torch.Tensor, int]:
    uniq, inv = torch.unique(lbl.flatten(), return_inverse=True)
    return inv.view_as(lbl).long(), int(uniq.numel())


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) ImageNet-normalised → (H, W, 3) uint8."""
    m = torch.tensor(IMAGENET_MEAN, dtype=torch.float64).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD, dtype=torch.float64).view(3, 1, 1)
    img = (x[0].detach().double().cpu() * s + m).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def partition(x: torch.Tensor, spec: str) -> torch.Tensor:
    """Label map (H, W) long on x.device, contiguous labels."""
    _, _, H, W = x.shape
    parts = spec.split(":")
    kind = parts[0]
    if kind == "grid":
        cell = int(parts[1])
        dy = int(parts[2]) if len(parts) > 2 else 0
        dx = int(parts[3]) if len(parts) > 3 else 0
        r = (torch.arange(H) + dy) // cell
        c = (torch.arange(W) + dx) // cell
        lbl = r[:, None] * (W // cell + 2) + c[None, :]
    elif kind == "slic":
        try:
            from skimage.segmentation import slic
        except ImportError as e:  # pragma: no cover
            raise ImportError("partition 'slic' needs scikit-image (pip install scikit-image)") from e
        n = int(parts[1])
        comp = float(parts[2]) if len(parts) > 2 else 10.0
        lab = slic(to_uint8(x), n_segments=n, compactness=comp, start_label=0)
        lbl = torch.from_numpy(np.asarray(lab, dtype=np.int64))
    else:
        raise ValueError(f"unknown partition '{spec}'")
    lbl, _ = relabel(lbl)
    return lbl.to(x.device)


# ─────────────────────────────────────────────────────────────────────────────
# coalition sampling
# ─────────────────────────────────────────────────────────────────────────────

def shapley_kernel_masks(G: int, M: int, gen: Optional[torch.Generator] = None,
                         paired: bool = True) -> torch.Tensor:
    """(M, G) bool, sizes ~ p(s) ∝ 1/(s(G−s)), s = 1 … G−1."""
    if G < 2:
        raise ValueError("need at least 2 groups")
    s = torch.arange(1, G, dtype=torch.float64)
    p = 1.0 / (s * (G - s))
    n = (M + 1) // 2 if paired else M
    size = torch.multinomial(p / p.sum(), n, replacement=True, generator=gen) + 1     # (n,)
    ranks = torch.rand(n, G, generator=gen, dtype=torch.float64).argsort(dim=1).argsort(dim=1)
    m = ranks < size[:, None]
    if paired:
        m = torch.cat([m, ~m], dim=0)[:M]
    return m


def bernoulli_masks(G: int, M: int, gen: Optional[torch.Generator] = None,
                    p: float = 0.5) -> torch.Tensor:
    return torch.rand(M, G, generator=gen, dtype=torch.float64) < p


def all_coalitions(G: int) -> torch.Tensor:
    """(2^G − 2, G) bool, every coalition except ∅ and the full set (tests / tiny G)."""
    idx = torch.arange(1, 2 ** G - 1)
    return ((idx[:, None] >> torch.arange(G)[None, :]) & 1).bool()


def shapley_kernel_weight(m: torch.Tensor) -> torch.Tensor:
    """Kernel-SHAP weight (G−1) / (C(G,s) s (G−s)) for each row of a (M, G) bool."""
    G = m.shape[1]
    s = m.sum(dim=1).double()
    logC = torch.lgamma(torch.tensor(G + 1.0, dtype=torch.float64)) - torch.lgamma(s + 1) - torch.lgamma(G - s + 1)
    return (G - 1) / (torch.exp(logC) * s * (G - s))


# ─────────────────────────────────────────────────────────────────────────────
# mask sets
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MaskSet:
    """Feature-level (pixel-level) masks and game values.

    pix:    (M, H·W) bool — pixel present
    y:      (M,) float64 — f(x_m) − f(x')
    groups: list of (spec, label map, G) used
    origin: (M,) long — which partition each row came from
    gmask:  list of (M_r, G_r) bool group masks per partition (for KernelSHAP)
    """
    pix: torch.Tensor
    y: torch.Tensor
    groups: list = field(default_factory=list)
    origin: Optional[torch.Tensor] = None
    gmask: list = field(default_factory=list)

    @property
    def M(self) -> int:
        return int(self.pix.shape[0])

    def B(self, dtype=torch.float64) -> torch.Tensor:
        return self.pix.to(dtype)


@torch.no_grad()
def game_values(oracle: Oracle, x: torch.Tensor, xb: torch.Tensor,
                pix: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """y_m = f(x_m) − f(x') for pixel masks pix (M, H·W) bool. M + 1 forwards."""
    _, C, H, W = x.shape
    f0 = float(oracle.f(xb)[0])
    out = []
    for i in range(0, pix.shape[0], chunk):
        m = pix[i:i + chunk].to(x.device).view(-1, 1, H, W)
        xm = torch.where(m, x.expand(m.shape[0], C, H, W), xb.expand(m.shape[0], C, H, W))
        out.append(oracle.f(xm).double() - f0)
    return torch.cat(out).cpu()


def build_masks(oracle: Oracle, x: torch.Tensor, xb: torch.Tensor,
                specs: Sequence[str], M: int, gen: Optional[torch.Generator] = None,
                family: str = "shapley", p: float = 0.5,
                labels: Optional[list] = None, chunk: int = 32) -> MaskSet:
    """M masks split evenly over the partitions in `specs` (computed from x
    unless `labels` is given, aligned with specs)."""
    if labels is None:
        labels = [partition(x, s) for s in specs]
    R = len(specs)
    counts = [M // R + (1 if r < M % R else 0) for r in range(R)]
    pix, origin, gmask, groups = [], [], [], []
    for r, (spec, lbl) in enumerate(zip(specs, labels)):
        G = int(lbl.max()) + 1
        groups.append((spec, lbl, G))
        if counts[r] == 0:
            gmask.append(torch.zeros(0, G, dtype=torch.bool))
            continue
        if family == "shapley":
            gm = shapley_kernel_masks(G, counts[r], gen)
        elif family == "bernoulli":
            gm = bernoulli_masks(G, counts[r], gen, p)
        else:
            raise ValueError(f"unknown mask family '{family}'")
        gmask.append(gm)
        pix.append(gm[:, lbl.flatten().cpu()])
        origin.append(torch.full((gm.shape[0],), r, dtype=torch.long))
    pix_t = torch.cat(pix, dim=0)
    y = game_values(oracle, x, xb, pix_t, chunk)
    return MaskSet(pix=pix_t, y=y, groups=groups, origin=torch.cat(origin), gmask=gmask)
