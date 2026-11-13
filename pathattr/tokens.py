"""
Path attribution on token embeddings (text classification), model-agnostic.

Features are the non-special token positions `pos` (F of them). Every feature
has ONE schedule s_i(t) shared by all h embedding dims:
    emb(t)[p] = E_b[p] + s_i(t) (E[p] − E_b[p])      for p = pos[i]
Special tokens ([CLS]/[SEP]/<s>…) are identical in E and E_b and never move.
E_b puts the [MASK] embedding at every feature position, so the path baseline
and the masking game coincide:  v(S) = f(emb with tokens ∉ S → [MASK]) − f(E_b).

The model is any nn.Module mapping embeddings (B, L, h) → (B,) scalar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from . import paths
from .oracle import Oracle


@dataclass
class TokenInput:
    E: torch.Tensor      # (1, L, h) input embeddings
    Eb: torch.Tensor     # (1, L, h) baseline embeddings ([MASK] at feature positions)
    pos: torch.Tensor    # (F,) long, feature positions

    @property
    def F(self) -> int:
        return int(self.pos.numel())

    @property
    def h(self) -> int:
        return int(self.E.shape[-1])

    def feat(self) -> torch.Tensor:
        """element (feature i, dim j) → i, for attributions restricted to `pos`."""
        return torch.arange(self.F).repeat_interleave(self.h)

    def restrict(self, a: torch.Tensor) -> torch.Tensor:
        """(…, L, h) or flattened (…, L·h) → (…, F·h) at the feature positions."""
        L, h = self.E.shape[1], self.h
        a = a.reshape(a.shape[:-1] + (L, h)) if a.shape[-1] == L * h else a
        return a[..., self.pos.to(a.device), :].reshape(a.shape[:-2] + (self.F * h,))

    def token_sum(self, a_elem: torch.Tensor) -> torch.Tensor:
        """(…, F·h) → (…, F) token attributions."""
        return a_elem.reshape(a_elem.shape[:-1] + (self.F, self.h)).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# schedules and paths
# ─────────────────────────────────────────────────────────────────────────────

def straight_sched(F: int, N: int) -> torch.Tensor:
    t = torch.arange(N + 1, dtype=torch.float64) / N
    return t[:, None].expand(N + 1, F).clone()


def spi_sched(F: int, N: int, alpha: float, gen: Optional[torch.Generator] = None,
              tail: float = 1e-4, max_atoms: int = 256) -> torch.Tensor:
    """(N+1, F): CDF at k/N of an independent DP(α, U(0,1)) per token (exact Dirichlet
    bin masses; tail / max_atoms are unused, kept for signature compatibility)."""
    s = paths.dp_grid_cdf(F, N, alpha, gen).t().contiguous()
    s[0] = 0.0
    s[N] = 1.0
    return s


def build_path(ti: TokenInput, s: torch.Tensor) -> torch.Tensor:
    """s: (N+1, F) schedules → pts (N+1, L, h); pts[0] = E_b, pts[N] = E."""
    N1 = s.shape[0]
    L = ti.E.shape[1]
    full = torch.ones(N1, L, dtype=ti.E.dtype, device=ti.E.device)
    full[:, ti.pos.to(ti.E.device)] = s.to(device=ti.E.device, dtype=ti.E.dtype)
    pts = ti.Eb + full[..., None] * (ti.E - ti.Eb)
    pts[0] = ti.Eb[0]
    pts[-1] = ti.E[0]
    return pts


def smooth_candidates(oracle: Oracle, ti: TokenInput, N: int = 24, sigma: float = 0.1,
                      K: int = 1, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """smooth IG on embeddings, clean mode: Σ_k ∇f(γ_k + ε_k)·(γ_{k+1} − γ_k),
    ε ~ N(0, (σ·ρ)²) per dim at feature positions, ρ = RMS of the input embeddings
    over feature positions (σ in units of the embedding scale, not of |x − x'|).
    Returns (K, F·h)."""
    pts = build_path(ti, straight_sched(ti.F, N))
    hstep = pts[1:] - pts[:-1]
    rho = float(ti.E[0, ti.pos.to(ti.E.device)].double().pow(2).mean().sqrt())
    out = []
    for _ in range(int(K)):
        eps = torch.randn((N + 1, ti.F, ti.h), generator=gen, dtype=torch.float64) * sigma * rho
        eps[0] = 0.0
        eps[N] = 0.0
        noise = torch.zeros_like(pts)
        noise[:, ti.pos.to(pts.device)] = eps.to(device=pts.device, dtype=pts.dtype)
        _, g = oracle.grad((pts + noise)[:-1])
        a = (g * hstep).sum(dim=0, keepdim=True)
        out.append(ti.restrict(a.double().cpu())[0])
    return torch.stack(out)


# ─────────────────────────────────────────────────────────────────────────────
# masking game
# ─────────────────────────────────────────────────────────────────────────────

def coalition_emb(ti: TokenInput, m: torch.Tensor) -> torch.Tensor:
    """m: (n, F) bool (True = token present) → (n, L, h)."""
    n = m.shape[0]
    L = ti.E.shape[1]
    keep = torch.zeros(n, L, dtype=torch.bool, device=ti.E.device)
    keep[:, ti.pos.to(ti.E.device)] = m.to(ti.E.device)
    return torch.where(keep[..., None], ti.E.expand(n, -1, -1), ti.Eb.expand(n, -1, -1))


@torch.no_grad()
def game_values(oracle: Oracle, ti: TokenInput, m: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """y = f(coalition) − f(E_b), float64 CPU."""
    f0 = float(oracle.f(ti.Eb)[0])
    out = []
    for i in range(0, m.shape[0], chunk):
        out.append(oracle.f(coalition_emb(ti, m[i:i + chunk])).double().cpu() - f0)
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.float64)


@torch.no_grad()
def shapley_permutation(oracle: Oracle, ti: TokenInput, max_perm: int = 5000, min_pairs: int = 50,
                        tol_rel: float = 0.02, gen: Optional[torch.Generator] = None,
                        chunk: int = 64) -> tuple[torch.Tensor, dict]:
    """Shapley values of the [MASK] game by antithetic permutation sampling.

    Each sample = one permutation π and its reverse (averaged), F − 1 forwards
    each (v(∅) = 0 and v(full) = D are known). Stops when
        max_i SE_i ≤ tol_rel · mean_i |φ̂_i|   (after ≥ min_pairs samples)
    or after max_perm permutations. Returns φ (F,) and {se, n_perm, n_fwd}.
    """
    F = ti.F
    f0 = float(oracle.f(ti.Eb)[0])
    D = float(oracle.f(ti.E)[0]) - f0
    s1 = torch.zeros(F, dtype=torch.float64)
    s2 = torch.zeros(F, dtype=torch.float64)
    n_pairs = 0
    n_fwd = 2

    def contrib(perm: torch.Tensor) -> torch.Tensor:
        nonlocal n_fwd
        ranks = torch.empty(F, dtype=torch.long)
        ranks[perm] = torch.arange(F)
        if F > 1:
            m = ranks[None, :] < torch.arange(1, F)[:, None]          # states S_1 … S_{F−1}
            v_mid = game_values(oracle, ti, m, chunk) if F > 1 else torch.zeros(0)
            n_fwd += F - 1
            v = torch.cat([torch.zeros(1, dtype=torch.float64), v_mid,
                           torch.tensor([D], dtype=torch.float64)])
        else:
            v = torch.tensor([0.0, D], dtype=torch.float64)
        phi = torch.empty(F, dtype=torch.float64)
        phi[perm] = v[1:] - v[:-1]
        return phi

    while 2 * n_pairs < max_perm:
        perm = torch.randperm(F, generator=gen)
        sample = 0.5 * (contrib(perm) + contrib(perm.flip(0)))
        s1 += sample
        s2 += sample * sample
        n_pairs += 1
        if n_pairs >= min_pairs:
            mean = s1 / n_pairs
            var = (s2 / n_pairs - mean * mean).clamp_min(0.0) * n_pairs / max(n_pairs - 1, 1)
            se = (var / n_pairs).sqrt()
            if float(se.max()) <= tol_rel * float(mean.abs().mean()):
                break
    mean = s1 / max(n_pairs, 1)
    var = (s2 / max(n_pairs, 1) - mean * mean).clamp_min(0.0) * n_pairs / max(n_pairs - 1, 1)
    se = (var / max(n_pairs, 1)).sqrt()
    return mean, {"se_max": float(se.max()), "se_mean": float(se.mean()), "n_perm": 2 * n_pairs,
                  "n_fwd": n_fwd, "D": D}


# ─────────────────────────────────────────────────────────────────────────────
# evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _rank(v: torch.Tensor) -> torch.Tensor:
    return v.argsort().argsort().double()


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-30))


def kendall(a: torch.Tensor, b: torch.Tensor) -> float:
    da = torch.sign(a[:, None] - a[None, :])
    db = torch.sign(b[:, None] - b[None, :])
    iu = torch.triu_indices(len(a), len(a), 1)
    x, y = da[iu[0], iu[1]], db[iu[0], iu[1]]
    conc = float(((x * y) > 0).sum())
    disc = float(((x * y) < 0).sum())
    tx = float(((x == 0) & (y != 0)).sum())
    ty = float(((y == 0) & (x != 0)).sum())
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den > 0 else float("nan")


def reference_distances(phi: torch.Tensor, ref: torch.Tensor) -> dict:
    phi, ref = phi.double(), ref.double()
    k = max(1, int(round(0.2 * len(ref))))
    top_p = set(phi.argsort(descending=True)[:k].tolist())
    top_r = set(ref.argsort(descending=True)[:k].tolist())
    return {"ref_spearman": spearman(phi, ref), "ref_kendall": kendall(phi, ref),
            "ref_l1": float((phi - ref).abs().sum() / ref.abs().sum().clamp_min(1e-30)),
            "ref_top20_overlap": len(top_p & top_r) / k}


@torch.no_grad()
def eraser_scores(prob_oracle: Oracle, ti: TokenInput, phi: torch.Tensor,
                  bins=(0.01, 0.05, 0.10, 0.20, 0.50)) -> dict:
    """ERASER comprehensiveness / sufficiency on the predicted-class probability.
    Tokens ranked by signed attribution (most positive first); removed tokens → [MASK].
        comp(k) = p(x) − p(x without top-k)        (higher is better)
        suff(k) = p(x) − p(only top-k)             (lower is better)
    aopc_* = mean over bins."""
    F = ti.F
    p_full = float(prob_oracle.f(ti.E)[0])
    order = phi.argsort(descending=True)
    comps, suffs = [], []
    for b in bins:
        k = max(1, int(math.ceil(b * F)))
        top = torch.zeros(F, dtype=torch.bool)
        top[order[:k]] = True
        both = torch.stack([~top, top])
        p = prob_oracle.f(coalition_emb(ti, both)).double().cpu()
        comps.append(p_full - float(p[0]))
        suffs.append(p_full - float(p[1]))
    return {"comp": sum(comps) / len(comps), "suff": sum(suffs) / len(suffs),
            "comp_bins": comps, "suff_bins": suffs}
