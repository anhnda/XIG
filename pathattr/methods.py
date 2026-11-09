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
# LIG Phase 1 (μ only, "LIG-IDIG") with a choice of gradient node and path
# ─────────────────────────────────────────────────────────────────────────────

def project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto {μ ≥ 0, Σμ = 1}."""
    u, _ = torch.sort(v, descending=True)
    css = u.cumsum(0) - 1.0
    k = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    rho = int((u - css / k > 0).nonzero().max())
    return (v - css[rho] / (rho + 1)).clamp(min=0.0)


def lig_mu(oracle: Oracle, xb, x, gen=None, N: int = 50, node: str = "left",
           mu: str = "adam", lam: float = 1.0, tau: float = 0.01, mu_iter: int = 300,
           path: str = "straight", rescale: bool = True) -> Attribution:
    """
    A = Σ_k μ_k g_k ⊙ h_k  (then multiplicative rescale, as in the paper),
    g_k taken at γ_k + u·h_k, u = 0 / 0.5 / 1 for node = left / mid / right.
    μ: 'adam'  → utility.optimize_mu_signal_harvesting (paper's Phase 1)
       'proj'  → Proj_simplex((λ/τ)|d|), the closed-form optimum without Var
       'uniform' → μ_k = 1/N (IG / dnoise with that node)
    path: 'straight' or 'dnoise' (dither defaults, N steps, drawn with gen).
    node='left', mu='adam', path='straight' ≡ legacy LIG-IDIG with aligned Δf.
    """
    u = {"left": 0.0, "mid": 0.5, "right": 1.0}[node]
    if path == "straight":
        pts = paths.straight(xb, x, N)
    elif path == "dnoise":
        pts = paths.dither(xb, x, N, gen=gen)
    else:
        raise ValueError(f"unknown path '{path}'")
    f = oracle.f(pts)
    df = f[1:] - f[:-1]
    h = pts[1:] - pts[:-1]
    _, g = oracle.grad(pts[:-1] + u * h)
    gh = g * h
    d = gh.flatten(1).sum(1)

    if mu == "adam":
        import utility  # legacy optimiser (repo root)
        w = utility.optimize_mu_signal_harvesting(
            d.detach().float().cpu(), df.detach().float().cpu(),
            lam=lam, tau=tau, n_iter=mu_iter).to(device=x.device, dtype=x.dtype)
    elif mu == "proj":
        w = project_simplex((lam / tau) * d.abs().double()).to(x.dtype)
    elif mu == "uniform":
        w = torch.full((N,), 1.0 / N, device=x.device, dtype=x.dtype)
    else:
        raise ValueError(f"unknown mu '{mu}'")

    attr = (w.view(-1, 1, 1, 1) * gh).sum(dim=0, keepdim=True)
    D = float(f[-1] - f[0])
    s = float(attr.sum())
    if rescale and abs(s) > 1e-12:
        attr = attr * (D / s)
    info = {"node": node, "mu_mode": mu, "path": path,
            "mu_argmax": int(w.argmax()), "mu_support": int((w > 1e-3).sum()),
            "transition_step": int(df.abs().argmax()), "raw_sum": s}
    return Attribution(attr, info)


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


def spi(oracle: Oracle, xb, x, gen=None, N: int = 24, alpha: float = 10.0,
        per_channel: bool = True, est: str = "left", Q: int = 1, K: int = 1,
        K_cv: int = 8) -> Attribution:
    """Stick-breaking Path Integration, mean over K paths (SPI-E)."""
    make = lambda: paths.spi(xb, x, N, alpha, per_channel, gen=gen)
    return _ensemble(make, oracle, K, est, Q, K_cv, xb, x, gen)


# ─────────────────────────────────────────────────────────────────────────────
# registry
# ─────────────────────────────────────────────────────────────────────────────

METHODS: dict[str, Callable[..., Attribution]] = {
    "ig": ig,
    "idgi": idgi,
    "lig_mu": lig_mu,
    "guided_ig": guided_ig,
    "dither": dither,
    "spi": spi,
}

# presets: name -> (method, default params). Params given on the command line
# override these.
PRESETS: dict[str, tuple[str, dict]] = {
    "dnoise":        ("dither", dict(N=24, thresh="bayer", tile=8, spread=0.5)),
    "pixel_random":  ("dither", dict(N=24, thresh="random", spread=1.0)),
    "pixel_strat":   ("dither", dict(N=24, thresh="stratified", tile=8, spread=1.0)),
    "block14":       ("dither", dict(N=24, thresh="block", block=14, spread=1.0)),
}

STOCHASTIC = {"dither", "spi"}


def resolve(name: str) -> tuple[Callable[..., Attribution], dict, bool]:
    """name -> (fn, default params, is_stochastic)."""
    if name in PRESETS:
        base, params = PRESETS[name]
        return METHODS[base], dict(params), base in STOCHASTIC
    if name in METHODS:
        return METHODS[name], {}, name in STOCHASTIC
    raise KeyError(f"unknown method '{name}'. Known: {sorted(METHODS) + sorted(PRESETS)}")
