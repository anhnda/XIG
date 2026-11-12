"""
Completeness corrections for path-attribution candidates.

Every candidate a^(k) of a path method satisfies Dummy exactly (a feature with
∂_i f ≡ 0 along the path, or with Δx_i = 0, gets a_i = 0 exactly), but is only
complete up to discretisation error:  b_k = D − Σ_i a_i^(k) ≠ 0.

support   (default)  additive, proportional to the support scale
              Λ_i = mean_k |a_i^(k)|            (mean of |·|, not |mean|)
              δ_i = b · w_i / Σ_j w_j,   w_i = Λ_i + η·1[Λ_i > 0]
          → exactly complete, and δ_i = 0 wherever every candidate is 0
            (true dummies, pixels with x_i = x'_i): Dummy preserved without
            knowing the dummies in advance.
          It is the κ → ∞ limit of TPA-R:  argmin δᵀΛ⁺δ  s.t. 1ᵀδ = b.
rescale   multiplicative  a · D / Σa  (keeps zeros; unstable when Σa ≈ 0)
none      leave as is (report the residual)
cv        use estimators.cv_complete on random-node draws (done by the caller:
          the candidate must be computed with estimator 'random_cv')
"""

from __future__ import annotations

from typing import Optional

import torch


def support_scale(A: torch.Tensor) -> torch.Tensor:
    """A: (K, ...) candidates → Λ: (...) = mean_k |a^(k)|, float64."""
    return A.double().abs().mean(dim=0)


def support_weights(lam: torch.Tensor, eta_rel: float = 0.0) -> torch.Tensor:
    """w = Λ + η·1[Λ>0],  η = eta_rel · mean(Λ on its support). Zero off support."""
    lam = lam.double()
    supp = lam > 0
    if eta_rel > 0 and bool(supp.any()):
        eta = eta_rel * float(lam[supp].mean())
        return torch.where(supp, lam + eta, torch.zeros_like(lam))
    return torch.where(supp, lam, torch.zeros_like(lam))


def complete_support(a: torch.Tensor, D: float, w: torch.Tensor) -> torch.Tensor:
    """a + b·w/Σw with b = D − Σa. a and w have the same number of elements."""
    wsum = float(w.sum())
    b = D - float(a.double().sum())
    if wsum <= 0.0:
        return a.clone()            # empty support: nothing can be moved (then D ≈ 0)
    return (a.double() + (b / wsum) * w.reshape(a.shape)).to(a.dtype)


def complete_rescale(a: torch.Tensor, D: float, tiny: float = 1e-12) -> torch.Tensor:
    s = float(a.double().sum())
    if abs(s) < tiny:
        return a.clone()
    return a * (D / s)


def complete_stack(A: torch.Tensor, D: float, how: str = "support",
                   eta_rel: float = 0.0, lam: Optional[torch.Tensor] = None
                   ) -> tuple[torch.Tensor, dict]:
    """Complete every candidate of A: (K, ...). Returns (A_completed, info).

    info: residual_before (list), residual_after (list), support_size.
    """
    K = A.shape[0]
    before = [D - float(A[k].double().sum()) for k in range(K)]
    if how in ("none", "cv"):
        out = A.clone()
        supp = None
    elif how == "rescale":
        out = torch.stack([complete_rescale(A[k], D) for k in range(K)])
        supp = None
    elif how == "support":
        lam = support_scale(A) if lam is None else lam.double()
        w = support_weights(lam, eta_rel)
        out = torch.stack([complete_support(A[k], D, w) for k in range(K)])
        supp = int((w > 0).sum())
    else:
        raise ValueError(f"unknown completion '{how}'")
    after = [D - float(out[k].double().sum()) for k in range(K)]
    return out, {"residual_before": before, "residual_after": after, "support_size": supp}
