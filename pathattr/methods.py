"""
Attribution methods built on `pathattr`.

Each method: fn(oracle, xb, x, gen, **params) -> Attribution
    xb, x: (1, C, H, W); gen: CPU torch.Generator (randomness only via gen).
Counts of forward/backward passes are read from the oracle by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from . import estimators as EST
from . import paths
from .oracle import Oracle
from .path_eval import eval_path


@dataclass
class Attribution:
    attr: torch.Tensor                     # (1, C, H, W)
    info: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# shared: integrate a (possibly random) path with a chosen estimator
# ─────────────────────────────────────────────────────────────────────────────

def integrate_path(oracle: Oracle, pts: torch.Tensor, estimator: str = "left",
                   Q: int = 1, K_cv: int = 8, xb: Optional[torch.Tensor] = None,
                   x: Optional[torch.Tensor] = None,
                   gen: Optional[torch.Generator] = None) -> tuple[torch.Tensor, dict]:
    """
    estimator ∈ {left, mid, random, gl, random_cv}
      random     : Q independent interior nodes per segment (unbiased)
      random_cv  : K_cv pilot draws + 1 fresh draw, corrected by cv_complete
                   (unbiased + exactly complete); cost (K_cv + 1)·N backward
    """
    info = {"estimator": estimator}
    if estimator in ("left", "mid", "random", "gl"):
        return EST.estimate(oracle, pts, estimator, Q, gen), info
    if estimator == "random_cv":
        if xb is None or x is None:
            raise ValueError("random_cv needs xb and x")
        D = oracle.endpoint_gap(xb, x)
        pilot = EST.random_draws(oracle, pts, K_cv, gen)
        fresh = EST.estimate(oracle, pts, "random", 1, gen)
        info["raw_residual"] = D - float(fresh.sum())
        return EST.cv_complete(pilot, fresh, D), info
    raise ValueError(f"unknown estimator '{estimator}'")


def _ensemble(make_pts: Callable[[], torch.Tensor], oracle: Oracle, K: int,
              estimator: str, Q: int, K_cv: int, xb, x, gen) -> Attribution:
    """Average over K independently drawn paths (K = 1: a single path)."""
    acc, info = None, {}
    for _ in range(max(1, int(K))):
        pts = make_pts()
        a, info = integrate_path(oracle, pts, estimator, Q, K_cv, xb, x, gen)
        acc = a if acc is None else acc + a
    info["K_paths"] = int(K)
    return Attribution(acc / max(1, int(K)), info)


# ─────────────────────────────────────────────────────────────────────────────
# IG
# ─────────────────────────────────────────────────────────────────────────────

def ig(oracle: Oracle, xb, x, gen=None, N: int = 50, est: str = "left",
       Q: int = 1) -> Attribution:
    """Integrated Gradients (Sundararajan et al., 2017). est='left' matches ig.py."""
    pts = paths.straight(xb, x, N)
    a, info = integrate_path(oracle, pts, est, Q, xb=xb, x=x, gen=gen)
    return Attribution(a, info)


# ─────────────────────────────────────────────────────────────────────────────
# IDGI (Yang, Wang, Bilen, CVPR 2023) — with aligned Δf, exactly complete
# ─────────────────────────────────────────────────────────────────────────────

def idgi(oracle: Oracle, xb, x, gen=None, N: int = 50, node: str = "left") -> Attribution:
    """A = Σ_k g_k² · Δf_k / ‖g_k‖²  with Δf_k = f(γ_{k+1}) − f(γ_k).

    node: where g_k is taken inside step k — 'left' γ_k (IDGI paper),
          'mid' (γ_k + γ_{k+1})/2, 'right' γ_{k+1}.
    The legacy buggy IDGI ≈ node='right' (minus the last step).
    Σ_i A_i = Σ_k Δf_k = f(x) − f(x') for every node: no rescale needed.
    """
    u = {"left": 0.0, "mid": 0.5, "right": 1.0}[node]
    pts = paths.straight(xb, x, N)
    f = oracle.f(pts)
    df = f[1:] - f[:-1]
    _, g = oracle.grad(pts[:-1] + u * (pts[1:] - pts[:-1]))
    gn2 = g.flatten(1).pow(2).sum(1)
    ok = gn2 > 1e-30
    scale = torch.where(ok, df / torch.where(ok, gn2, torch.ones_like(gn2)),
                        torch.zeros_like(gn2))
    attr = (g.pow(2) * scale.view(-1, 1, 1, 1)).sum(dim=0, keepdim=True)
    lost = float(df[~ok].sum()) if (~ok).any() else 0.0
    return Attribution(attr, {"dropped_df_zero_grad": lost, "node": node})


# ─────────────────────────────────────────────────────────────────────────────
# Guided IG (Kapishnikov et al., CVPR 2021) — faithful port of the PAIR
# reference implementation (saliency/core/guided_ig.py, no anchors)
# ─────────────────────────────────────────────────────────────────────────────

def guided_ig_path(oracle: Oracle, xb, x, N: int = 50, fraction: float = 0.25,
                   max_dist: float = 0.02, max_inner: int = 10_000
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns pts (N+1, C, H, W) and grads (N, C, H, W); one gradient per step."""
    dt, shape = x.dtype, x.shape
    x_in = x.detach().reshape(-1).double()
    x_bl = xb.detach().reshape(-1).double()
    diff = x_in - x_bl
    nz = diff != 0
    safe_diff = torch.where(nz, diff, torch.ones_like(diff))
    l1_total = float(diff.abs().sum())

    cur = x_bl.clone()
    pts = [cur.clone()]
    grads = []
    for step in range(N):
        _, g = oracle.grad(cur.view(shape).to(dt))
        g_act = g.reshape(-1).double()
        grads.append(g)
        alpha = (step + 1.0) / N
        a_min, a_max = max(alpha - max_dist, 0.0), min(alpha + max_dist, 1.0)
        x_min = x_bl + diff * a_min
        x_max = x_bl + diff * a_max
        l1_target = l1_total * (1.0 - (step + 1.0) / N)

        gamma, it = math.inf, 0
        while gamma > 1.0 and it < max_inner:
            it += 1
            x_alpha = torch.where(nz, (cur - x_bl) / safe_diff, torch.full_like(cur, a_max))
            cur = torch.where(x_alpha < a_min, x_min, cur)          # catch up stragglers
            l1_cur = float((cur - x_in).abs().sum())
            if math.isclose(l1_target, l1_cur, rel_tol=1e-9, abs_tol=1e-9):
                break
            grad = g_act.clone()
            grad[cur == x_max] = math.inf                           # exclude finished features
            absg = grad.abs()
            finite = torch.isfinite(absg)
            if not bool(finite.any()):
                break
            thr = torch.quantile(absg, fraction, interpolation="lower")
            sel = (absg <= thr) & finite
            l1_s = float(((cur - x_max).abs() * sel).sum())
            gamma = (l1_cur - l1_target) / l1_s if l1_s > 0 else math.inf
            if gamma > 1.0:
                cur = torch.where(sel, x_max, cur)
            elif gamma > 0.0:
                cur = torch.where(sel, cur + (x_max - cur) * gamma, cur)
            else:
                break
        if step == N - 1:
            cur = x_in.clone()       # final point is x (removes round-off)
        pts.append(cur.clone())

    pts_t = torch.stack([p.view(shape[1:]) for p in pts]).to(dt)
    g_t = torch.cat(grads, dim=0)
    return pts_t, g_t


def guided_ig(oracle: Oracle, xb, x, gen=None, N: int = 50, fraction: float = 0.25,
              max_dist: float = 0.02) -> Attribution:
    pts, g = guided_ig_path(oracle, xb, x, N, fraction, max_dist)
    attr = (g * (pts[1:] - pts[:-1])).sum(dim=0, keepdim=True)
    return Attribution(attr, {})


# ─────────────────────────────────────────────────────────────────────────────
# dither / dnoise and other per-pixel schedules
# ─────────────────────────────────────────────────────────────────────────────

def dither(oracle: Oracle, xb, x, gen=None, N: int = 24, thresh: str = "bayer",
           tile: int = 8, block: int = 14, spread: float = 0.5, noise: float = 0.0,
           per_channel: bool = False, est: str = "left", Q: int = 1, K: int = 1,
           K_cv: int = 8) -> Attribution:
    """Per-pixel schedule path. Defaults = reconstruction of "dnoise".

    K > 1 averages K independently drawn paths (each N steps).
    """
    make = lambda: paths.dither(xb, x, N, thresh, tile, block, spread, noise,
                                per_channel, gen)
    return _ensemble(make, oracle, K, est, Q, K_cv, xb, x, gen)


def smooth_ig(oracle: Oracle, xb, x, gen=None, N: int = 24, sigma: float = 0.3,
              mode: str = "clean", K: int = 1, unit: str = "delta") -> Attribution:
    """Straight line with per-element jitter at interior points.

    unit='delta' : ε_k ~ N(0, σ²|Δx|²)  (legacy: scales with |x − x'|, collapses
                   with a black baseline, proposition 4)
    unit='abs'   : ε_k ~ N(0, σ²)       σ in input units (normalised pixel space)

    γ̃_k = γ_k + ε_k (ε_0 = ε_N = 0), g_k = ∇f(γ̃_k).
      mode='clean'      Σ g_k ⊙ (γ_{k+1} − γ_k)      smoothed-gradient IG (has a limit as N→∞)
      mode='noise_only' Σ g_k ⊙ (ε_{k+1} − ε_k)      the jitter term alone (Stein: ≈ −σ²·curvature)
      mode='jitter'     Σ g_k ⊙ (γ̃_{k+1} − γ̃_k)    = clean + noise_only = dither(spread=0, noise=σ)
    K > 1 averages K independent noise draws (K·N gradients).
    """
    if mode not in ("clean", "noise_only", "jitter"):
        raise ValueError(f"unknown mode '{mode}'")
    pts = paths.straight(xb, x, N)
    h_clean = pts[1:] - pts[:-1]
    acc = torch.zeros_like(x)
    for _ in range(max(1, int(K))):
        eps = torch.randn((N + 1,) + tuple(x.shape[1:]), generator=gen,
                          dtype=torch.float64).to(device=x.device, dtype=x.dtype)
        if unit == "delta":
            eps = eps * sigma * (x - xb).abs()
        elif unit == "abs":
            eps = eps * sigma
        else:
            raise ValueError(f"unknown unit '{unit}'")
        eps[0] = 0.0
        eps[N] = 0.0
        _, g = oracle.grad((pts + eps)[:-1])
        if mode == "clean":
            h = h_clean
        elif mode == "noise_only":
            h = eps[1:] - eps[:-1]
        else:
            h = h_clean + eps[1:] - eps[:-1]
        acc = acc + (g * h).sum(dim=0, keepdim=True)
    return Attribution(acc / max(1, int(K)), {"sigma": sigma, "mode": mode, "unit": unit})


def spi(oracle: Oracle, xb, x, gen=None, N: int = 24, alpha: float = 10.0,
        per_channel: bool = True, est: str = "left", Q: int = 1, K: int = 1,
        K_cv: int = 8, sampler: str = "dirichlet") -> Attribution:
    """Stochastic Path Integration, mean over K paths (SPI-E).
    sampler='dirichlet' exact DP(α) bin masses; 'stick' = legacy truncated."""
    make = lambda: paths.spi(xb, x, N, alpha, per_channel, gen=gen, sampler=sampler)
    return _ensemble(make, oracle, K, est, Q, K_cv, xb, x, gen)


# ─────────────────────────────────────────────────────────────────────────────
# TPA and KernelSHAP (image wrappers around pathattr.tpa / pathattr.masks)
# ─────────────────────────────────────────────────────────────────────────────

def _specs(parts: str) -> list[str]:
    """'slic:64+grid:28:14:14' -> ['slic:64', 'grid:28:14:14'] ('+' joins partitions)."""
    return [p for p in str(parts).split("+") if p]


def tpa(oracle: Oracle, xb, x, gen=None, alpha: float = 1.0, K: int = 8, N: int = 24,
        est: str = "mid", Q: int = 1, per_channel: bool = True, completion: str = "support",
        eta: float = 0.0, parts: str = "slic:64", M: int = 256, family: str = "shapley",
        mode: str = "core", eps: float = 0.1, kappa: float = 1.0, K_cv: int = 8) -> Attribution:
    """Entropic path aggregation over K SPI(α) paths, fitted on M masks.

    mode: uniform (SPI) | gibbs | core | r   — see pathattr.tpa.
    completion: support (default, keeps Dummy) | rescale | none | cv (est → random_cv).
    Same seed + same (alpha, K, N, est, parts, M, family) ⇒ identical candidates
    and masks across modes, so variants are paired.
    """
    from . import masks as MK
    from . import tpa as T
    from .complete import complete_stack, support_scale

    D = oracle.endpoint_gap(xb, x)
    if completion == "cv":
        est = "random_cv"
    A_raw = T.spi_image_candidates(oracle, xb, x, gen, alpha, K, N, est, Q, per_channel, K_cv, D)
    lam = support_scale(A_raw)
    A, cinfo = complete_stack(A_raw, D, "none" if completion == "cv" else completion, eta, lam)
    ms = MK.build_masks(oracle, x, xb, _specs(parts), M, gen, family)
    feat, F = T.image_feat(x)
    game = T.Game(B=ms.B(), y=ms.y, feat=feat, F=F, D=D)
    a, q, info = T.fit(A, game, mode, eps, kappa, lam)
    info.update({"alpha": alpha, "K": K, "N": N, "est": est, "completion": completion,
                 "parts": parts, "n_groups": [g[2] for g in ms.groups],
                 "max_abs_residual_before": max(abs(v) for v in cinfo["residual_before"]),
                 "support_size": cinfo["support_size"]})
    return Attribution(a.view_as(x).to(device=x.device, dtype=x.dtype), info)


def kernelshap(oracle: Oracle, xb, x, gen=None, parts: str = "slic:64", M: int = 256,
               family: str = "shapley", ridge: float = 1e-10) -> Attribution:
    """KernelSHAP on the first partition in `parts` (no path prior).
    Group value φ_g is spread uniformly over its pixels and channels."""
    from . import masks as MK

    D = oracle.endpoint_gap(xb, x)
    spec = _specs(parts)[0]
    ms = MK.build_masks(oracle, x, xb, [spec], M, gen, family)
    attr, G = kernelshap_from_masks(ms, x, D, ridge)
    return Attribution(attr, {"G": G, "parts": spec, "M": M})


def kernelshap_from_masks(ms, x: torch.Tensor, D: float, ridge: float = 1e-10
                          ) -> tuple[torch.Tensor, int]:
    """KernelSHAP on partition 0 of a MaskSet (rows with origin 0), spread to pixels."""
    from . import tpa as T
    _, lbl, G = ms.groups[0]
    rows = ms.origin == 0
    phi = T.kernelshap(ms.gmask[0].double(), ms.y[rows], D, ridge=ridge)    # (G,)
    lbl_c = lbl.flatten().cpu()
    n_g = torch.bincount(lbl_c, minlength=G).double()
    _, C, H, W = x.shape
    pix = (phi / n_g.clamp_min(1.0))[lbl_c].view(1, 1, H, W) / C
    attr = pix.expand(1, C, H, W).contiguous().to(device=x.device, dtype=x.dtype)
    return attr, G


# ─────────────────────────────────────────────────────────────────────────────
# registry
# ─────────────────────────────────────────────────────────────────────────────

METHODS: dict[str, Callable[..., Attribution]] = {
    "ig": ig,
    "idgi": idgi,
    "guided_ig": guided_ig,
    "dither": dither,
    "spi": spi,
    "smooth_ig": smooth_ig,
    "tpa": tpa,
    "kernelshap": kernelshap,
}

# presets: name -> (method, default params). Params given on the command line
# override these.
PRESETS: dict[str, tuple[str, dict]] = {
    "dnoise":        ("dither", dict(N=24, thresh="bayer", tile=8, spread=0.5)),
    "pixel_random":  ("dither", dict(N=24, thresh="random", spread=1.0)),
    "pixel_strat":   ("dither", dict(N=24, thresh="stratified", tile=8, spread=1.0)),
    "block14":       ("dither", dict(N=24, thresh="block", block=14, spread=1.0)),
    "noise_only":    ("smooth_ig", dict(mode="noise_only")),
}

STOCHASTIC = {"dither", "spi", "smooth_ig", "tpa", "kernelshap"}


def resolve(name: str) -> tuple[Callable[..., Attribution], dict, bool]:
    """name -> (fn, default params, is_stochastic)."""
    if name in PRESETS:
        base, params = PRESETS[name]
        return METHODS[base], dict(params), base in STOCHASTIC
    if name in METHODS:
        return METHODS[name], {}, name in STOCHASTIC
    raise KeyError(f"unknown method '{name}'. Known: {sorted(METHODS) + sorted(PRESETS)}")
