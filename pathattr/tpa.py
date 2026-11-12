"""
Tempered / entropic path aggregation (TPA) and its baselines.

Notation (generic, works for pixels and tokens)
    elements e = 1 … E   coordinates that carry attribution (C·H·W, or L·h)
    features p = 1 … F   what a mask switches on/off (pixels, or tokens)
    feat: (E,) long      element → feature
    B: (M, F)            feature-level masks (1 = present)
    y: (M,)              y_m = f(x_m) − f(x')            (y_∅ = 0, y_full = D)
    A: (K, E)            candidate path attributions a^(k)
    apply(a) = B · Σ_{e∈p} a_e                           (model of y_m)

Loss of an attribution a on a mask set:  L(a) = (1/M) ‖y − apply(a)‖²
(masks are drawn from the target kernel, so all regression weights are 1).
Residuals r_k = y − apply(a^(k)); for a mixture, Σ q_k = 1  ⇒  r(q) = Σ q_k r_k.

Variants (`mode`)
    uniform  q = u                                   (SPI average, ε = ∞)
    gibbs    q_k ∝ exp(−S_k/ε),  S_k = L(a^(k))      (true Gibbs selection)
    core     argmin_q  qᵀRq + ε KL(q‖u),  R = r rᵀ/M  (entropic aggregation)
    r        argmin_{q,δ} (1/M)‖r(q) − Bδ‖² + κ δᵀΛ⁺δ + ε KL(q‖u)
             s.t. 1ᵀ(Aᵀq + δ) = D,  δ = 0 off supp(Λ)
             δ is eliminated in closed form, which leaves the same entropic QP
             with a modified Gram matrix R̃ (see r_correction).
Ambiguity decomposition (Krogh–Vedelsby):  L(q) = G(q) − D(q),
    G = Σ q_k ‖r_k‖²/M,   D = Σ q_k ‖r_k − r̄‖²/M.
ε is given relative to s = Var(y) on the fit masks: ε_abs = eps · s.
κ is dimensionless because Λ is normalised to Σ Λ = 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from . import complete as CMP
from . import estimators as EST
from . import paths
from .oracle import Oracle


# ─────────────────────────────────────────────────────────────────────────────
# the masking game in matrix form
# ─────────────────────────────────────────────────────────────────────────────

def feat_sum(a: torch.Tensor, feat: torch.Tensor, F: int) -> torch.Tensor:
    """(…, E) → (…, F): sum of the elements of each feature."""
    out = a.new_zeros(a.shape[:-1] + (F,))
    return out.index_add_(a.dim() - 1, feat, a)


@dataclass
class Game:
    B: torch.Tensor        # (M, F) float64
    y: torch.Tensor        # (M,)   float64
    feat: torch.Tensor     # (E,)   long
    F: int
    D: float

    @property
    def M(self) -> int:
        return int(self.B.shape[0])

    def apply(self, a: torch.Tensor) -> torch.Tensor:
        """(E,) → (M,);  (K, E) → (K, M)."""
        return feat_sum(a, self.feat, self.F) @ self.B.t()

    def adjoint(self, z: torch.Tensor) -> torch.Tensor:
        """(M,) → (E,);  (n, M) → (n, E)."""
        return (z @ self.B)[..., self.feat]

    def scale(self) -> float:
        return max(float(self.y.var(unbiased=False)), 1e-30)

    def rel_loss(self, a: torch.Tensor) -> float:
        """L(a) / Var(y): 0 = perfect additive fit, 1 = no better than a constant."""
        return float(((self.y - self.apply(a)) ** 2).mean()) / self.scale()


# ─────────────────────────────────────────────────────────────────────────────
# solvers
# ─────────────────────────────────────────────────────────────────────────────

def _xlogx_kl(q: torch.Tensor) -> torch.Tensor:
    K = q.numel()
    return (q * torch.log(K * q.clamp_min(1e-300))).sum()


def qp_stationarity(R: torch.Tensor, q: torch.Tensor, eps: float) -> float:
    """Optimality measure of q for  qᵀRq + eps·KL(q‖u)  on the simplex.
    eps > 0: fixed-point residual  max_k |softmax(−2Rq/eps)_k − q_k|  (KKT ⇔ 0)
    eps = 0: Frank–Wolfe gap  gᵀq − min_k g_k,  g = 2Rq  (≥ f(q) − f*)."""
    g = 2.0 * (R @ q)
    if eps > 0:
        return float((torch.softmax(-g / eps, dim=0) - q).abs().max())
    return float(g @ q - g.min())


def _qp_grad(R: torch.Tensor, q: torch.Tensor, eps: float) -> torch.Tensor:
    g = 2.0 * (R @ q)
    if eps > 0:
        g = g + eps * (torch.log(q.numel() * q.clamp_min(1e-300)) + 1.0)
    return g


def solve_entropic_qp(R: torch.Tensor, eps: float, iters: int = 100000,
                      tol: float = 1e-12, warm: float = 1e-4,
                      newton_iters: int = 100) -> tuple[torch.Tensor, dict]:
    """argmin_{q ∈ Δ_K} qᵀRq + eps·KL(q‖u)  (R symmetric PSD, eps ≥ 0).
    Convex; unique when eps > 0.

    Phase 1 — entropic mirror descent with the FIXED step 1/(2 max|R| + eps).
        That step is always admissible (the quadratic is 2max|R|-smooth and eps·KL
        is eps-smooth relative to the entropy), so every step decreases the
        objective without evaluating it. No function-value line search: near the
        optimum the decrease is O(stat²) and drowns in round-off.
    Phase 2 (eps > 0) — damped Newton on the KKT system
        [2R + eps·diag(1/q)   1] [dq]   [−∇f]
        [1ᵀ                   0] [dν] = [ 0 ],
        step kept inside the simplex, accepted when the projected-gradient
        residual decreases (measurable to ~1e-16). Quadratic convergence.
    Stops on optimality (qp_stationarity):
        eps > 0 → fixed-point residual < tol;   eps = 0 → FW gap < tol·max|R|."""
    R = R.double()
    R = 0.5 * (R + R.t())
    K = R.shape[0]
    q = torch.full((K,), 1.0 / K, dtype=torch.float64)
    step = 1.0 / (2.0 * float(R.abs().max()) + eps + 1e-300)
    tol_eff = tol if eps > 0 else tol * (float(R.abs().max()) + 1e-300)
    stop1 = max(tol_eff, warm) if eps > 0 else tol_eff
    stat = qp_stationarity(R, q, eps)
    it = 0
    while it < iters and stat >= stop1:                      # phase 1
        it += 1
        q = torch.softmax(torch.log(q.clamp_min(1e-300)) - step * _qp_grad(R, q, eps), dim=0)
        stat = qp_stationarity(R, q, eps)

    n_newton = 0
    if eps > 0 and stat >= tol_eff:                          # phase 2
        one = torch.ones(K, 1, dtype=torch.float64)
        for n_newton in range(1, newton_iters + 1):
            g = _qp_grad(R, q, eps)
            res0 = float((g - g.mean()).abs().max())
            H = 2.0 * R + eps * torch.diag(1.0 / q.clamp_min(1e-300))
            KKT = torch.cat([torch.cat([H, one], 1),
                             torch.cat([one.t(), torch.zeros(1, 1, dtype=torch.float64)], 1)], 0)
            rhs = torch.cat([-g, torch.zeros(1, dtype=torch.float64)])
            try:
                dq = torch.linalg.solve(KKT, rhs)[:K]
            except RuntimeError:
                break
            neg = dq < 0
            a = 1.0
            if bool(neg.any()):
                a = min(1.0, 0.99 * float((q[neg] / (-dq[neg])).min()))
            accepted = False
            while a > 1e-12:
                qn = (q + a * dq).clamp_min(1e-300)
                qn = qn / qn.sum()
                gn = _qp_grad(R, qn, eps)
                if float((gn - gn.mean()).abs().max()) < res0:
                    accepted = True
                    break
                a *= 0.5
            if not accepted:
                break
            q = qn
            stat = qp_stationarity(R, q, eps)
            if stat < tol_eff:
                break

    f = float(q @ R @ q) + (eps * float(_xlogx_kl(q)) if eps > 0 else 0.0)
    return q, {"iters": it, "newton": n_newton, "obj": f, "stationarity": stat,
               "converged": stat < tol_eff}


def gibbs_weights(S: torch.Tensor, eps: float) -> torch.Tensor:
    """q_k ∝ exp(−S_k/eps); eps = 0 → argmin, eps = ∞ → uniform."""
    K = S.numel()
    if math.isinf(eps):
        return torch.full((K,), 1.0 / K, dtype=torch.float64)
    if eps <= 0:
        q = torch.zeros(K, dtype=torch.float64)
        q[int(S.argmin())] = 1.0
        return q
    return torch.softmax(-S.double() / eps, dim=0)


def kernelshap(Bg: torch.Tensor, y: torch.Tensor, D: float,
               weights: Optional[torch.Tensor] = None, ridge: float = 1e-10) -> torch.Tensor:
    """Constrained (weighted) least squares on group coalitions:
        min Σ_m w_m (y_m − b_mᵀφ)² + ridge‖φ‖²   s.t. 1ᵀφ = D.
    With masks drawn from the Shapley kernel use weights = None (all 1)."""
    Bg = Bg.double()
    y = y.double()
    G = Bg.shape[1]
    w = torch.ones(Bg.shape[0], dtype=torch.float64) if weights is None else weights.double()
    H = Bg.t() @ (w[:, None] * Bg) + ridge * torch.eye(G, dtype=torch.float64)
    rhs = Bg.t() @ (w * y)
    KKT = torch.zeros(G + 1, G + 1, dtype=torch.float64)
    KKT[:G, :G] = 2.0 * H
    KKT[:G, G] = 1.0
    KKT[G, :G] = 1.0
    b = torch.cat([2.0 * rhs, torch.tensor([D], dtype=torch.float64)])
    return torch.linalg.solve(KKT, b)[:G]


# ─────────────────────────────────────────────────────────────────────────────
# TPA-R: closed-form elimination of δ
# ─────────────────────────────────────────────────────────────────────────────

def r_correction(A: torch.Tensor, game: Game, lam: torch.Tensor, kappa: float
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    For each candidate k (i.e. q = e_k) solve
        min_δ (1/M)‖r_k − Bδ‖² + κ δᵀΛ⁺δ   s.t. 1ᵀδ = b_k = D − 1ᵀa^(k),  δ ∈ range(Λ)
    Closed form (W = I/M, T = κM·I + BΛBᵀ, an M×M matrix):
        w(v) = v − Bᵀ T⁻¹ B Λ v,     δ(v) = Λ w(v) / κ,     κ δᵀΛ⁺δ = w Λ w / κ
        v = Bᵀr_k/M + ν_k·1,         ν_k from the constraint.
    δ*(q) is affine in q, so for Σq = 1:  δ*(q) = Σ q_k δ_k and the profiled
    objective is qᵀ R̃ q with  R̃_kl = ρ_kᵀρ_l/M + w_kᵀΛw_l/κ,  ρ_k = r_k − Bδ_k.
    Returns Δ (K, E), ρ (K, M), R̃ (K, K), info.
    """
    if kappa <= 0:
        raise ValueError("TPA-R needs kappa > 0 (kappa → 0 is plain regression)")
    K, E = A.shape
    M = game.M
    lam = lam.double().clamp_min(0.0)
    tot = float(lam.sum())
    if tot <= 0:
        raise ValueError("empty support: all candidates are zero")
    lam = lam / tot
    lam_f = feat_sum(lam, game.feat, game.F)                        # (F,)
    B = game.B
    T = kappa * M * torch.eye(M, dtype=torch.float64) + (B * lam_f[None, :]) @ B.t()
    L = torch.linalg.cholesky(T)

    def w_of(V: torch.Tensor) -> torch.Tensor:                       # (n, E) → (n, E)
        BLV = game.apply(V * lam[None, :])                           # (n, M)
        Z = torch.cholesky_solve(BLV.t(), L).t()                     # (n, M)
        return V - game.adjoint(Z)

    r = game.y[None, :] - game.apply(A)                              # (K, M)
    Gk = game.adjoint(r) / M                                         # (K, E)
    Wg = w_of(Gk)
    W1 = w_of(torch.ones(1, E, dtype=torch.float64))
    dg = lam[None, :] * Wg / kappa
    d1 = lam[None, :] * W1 / kappa
    c = float(d1.sum())
    b = game.D - A.sum(dim=1)                                        # (K,)
    if abs(c) > 1e-300:
        nu = (b - dg.sum(dim=1)) / c
    else:
        nu = torch.zeros(K, dtype=torch.float64)
    Delta = dg + nu[:, None] * d1
    Wk = Wg + nu[:, None] * W1
    rho = r - game.apply(Delta)
    pen = (Wk * lam[None, :]) @ Wk.t() / kappa
    Rt = rho @ rho.t() / M + pen
    Rt = 0.5 * (Rt + Rt.t())
    return Delta, rho, Rt, {"nu": nu.tolist(), "c": c}


# ─────────────────────────────────────────────────────────────────────────────
# fitting
# ─────────────────────────────────────────────────────────────────────────────

def ambiguity(r: torch.Tensor, q: torch.Tensor) -> dict:
    """G, D, L = G − D of residuals r (K, M) under weights q."""
    rbar = q @ r
    L = float((rbar ** 2).mean())
    G = float((q * (r ** 2).mean(dim=1)).sum())
    Dv = float((q * ((r - rbar[None, :]) ** 2).mean(dim=1)).sum())
    return {"G": G, "Dv": Dv, "L": L}


def equidistance(r: torch.Tensor, q: torch.Tensor) -> float:
    """CV of ‖r_k − r̄(q)‖²: 0 ⇔ q (if Gibbs-optimal) is also aggregation-optimal."""
    dist = ((r - (q @ r)[None, :]) ** 2).mean(dim=1)
    m = float(dist.mean())
    return float(dist.std(unbiased=False)) / m if m > 0 else 0.0


def fit(A: torch.Tensor, game: Game, mode: str = "core", eps: float = 0.1,
        kappa: float = 1.0, lam: Optional[torch.Tensor] = None
        ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """A: (K, E) candidates (completed or not). Returns (a_final (E,), q, info)."""
    A = A.double()
    K = A.shape[0]
    s = game.scale()
    eps_abs = eps * s if not math.isinf(eps) else float("inf")
    r = game.y[None, :] - game.apply(A)
    S = (r ** 2).mean(dim=1)
    info: dict = {"mode": mode, "eps_rel": eps, "scale": s,
                  "S_rel": (S / s).tolist()}
    Delta = None
    if mode == "uniform":
        q = torch.full((K,), 1.0 / K, dtype=torch.float64)
    elif mode == "gibbs":
        q = gibbs_weights(S, eps_abs)
    elif mode == "core":
        if math.isinf(eps_abs):
            q = torch.full((K,), 1.0 / K, dtype=torch.float64)
        else:
            q, sinfo = solve_entropic_qp(r @ r.t() / game.M, eps_abs)
            info["solver"] = sinfo
    elif mode == "r":
        if lam is None:
            lam = CMP.support_scale(A)
        Delta, rho, Rt, rinfo = r_correction(A, game, lam, kappa)
        info["kappa"] = kappa
        if math.isinf(eps_abs):
            q = torch.full((K,), 1.0 / K, dtype=torch.float64)
        else:
            q, sinfo = solve_entropic_qp(Rt, eps_abs)
            info["solver"] = sinfo
        info["amb_corrected"] = {k: v / s for k, v in ambiguity(rho, q).items()}
    else:
        raise ValueError(f"unknown mode '{mode}'")

    a = q @ A if Delta is None else q @ (A + Delta)
    amb = ambiguity(r, q)
    info.update({"q": q.tolist(), "q_entropy": float(-(q * torch.log(q.clamp_min(1e-300))).sum()),
                 "G_rel": amb["G"] / s, "D_rel": amb["Dv"] / s, "Lmix_rel": amb["L"] / s,
                 "fit_rel": game.rel_loss(a), "residual": game.D - float(a.sum())})
    if not math.isinf(eps_abs) and eps_abs > 0:
        info["equidist_gibbs"] = equidistance(r, gibbs_weights(S, eps_abs))
    return a, q, info


# ─────────────────────────────────────────────────────────────────────────────
# candidates
# ─────────────────────────────────────────────────────────────────────────────

def path_candidates(oracle: Oracle, make_path: Callable[[], torch.Tensor], K: int,
                    est: str = "mid", Q: int = 1, K_cv: int = 8, D: Optional[float] = None,
                    gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """(K, E) float64 on CPU: one attribution per independently drawn path."""
    out = []
    for _ in range(int(K)):
        pts = make_path()
        if est == "random_cv":
            if D is None:
                raise ValueError("random_cv needs D")
            pilot = EST.random_draws(oracle, pts, K_cv, gen)
            fresh = EST.estimate(oracle, pts, "random", 1, gen)
            a = EST.cv_complete(pilot, fresh, D)
        else:
            a = EST.estimate(oracle, pts, est, Q, gen)
        out.append(a.reshape(-1).double().cpu())
    return torch.stack(out)


def image_feat(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """element (c, h, w) → pixel h·W + w, for x of shape (1, C, H, W)."""
    _, C, H, W = x.shape
    return torch.arange(C * H * W) % (H * W), H * W


def spi_image_candidates(oracle: Oracle, xb, x, gen=None, alpha: float = 1.0, K: int = 8,
                         N: int = 24, est: str = "mid", Q: int = 1, per_channel: bool = True,
                         K_cv: int = 8, D: Optional[float] = None) -> torch.Tensor:
    make = lambda: paths.spi(xb, x, N, alpha, per_channel, gen=gen)
    return path_candidates(oracle, make, K, est, Q, K_cv, D, gen)