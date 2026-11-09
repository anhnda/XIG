"""
blurig.py - Blur Integrated Gradients (Xu, Venugopalan, Sundararajan, CVPR 2020)
================================================================================
BlurIG: integrates gradients along a Gaussian-blur path from an explicit
baseline x_b to the input x. The path is parametrised as

    L(sigma) = x_b + G_sigma * (x - x_b)

with sigma running from max_sigma (most blurred end) to 0 (input). This
generalises the original BlurIG (which assumed x_b = 0) to use an explicit
baseline, matching the IG / Guided IG interface.

Endpoints:
    L(sigma = 0)         = x_b + (x - x_b)         = x         (input)
    L(sigma = max_sigma) ~ x_b + 0                 ~ x_b       (baseline)

The maximally-blurred end approaches x_b only asymptotically; the residual
G_{max_sigma} * (x - x_b) is reported as a completeness check.

Usage:
    from blurig import compute_blurig
    attribution_result = compute_blurig(
        model=model,
        input=x,
        params={'baseline': baseline, 'N': 100, 'max_sigma': 50.0}
    )
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utility import (
    AttributionResult,
    _forward_scalar,
    _forward_and_gradient,
    _pack_result,
)


# ---------------------------------------------------------------------------
# Gaussian blur (PyTorch, depthwise separable)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(sigma: float, device, dtype, truncate: float = 4.0) -> torch.Tensor:
    """1D Gaussian kernel with radius = ceil(truncate * sigma), normalised to sum 1."""
    if sigma <= 0.0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = max(1, int(math.ceil(truncate * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Depthwise separable 2D Gaussian blur over (H, W); channels untouched.
    Equivalent to scipy.ndimage.gaussian_filter(image, sigma=[sigma, sigma, 0],
    mode='constant').
    """
    if sigma <= 0.0:
        return image
    if image.dim() != 4:
        raise ValueError(f"_gaussian_blur expects (B, C, H, W); got {tuple(image.shape)}")

    B, C, H, W = image.shape
    device, dtype = image.device, image.dtype

    k1d = _gaussian_kernel_1d(sigma, device=device, dtype=dtype)
    radius = (k1d.numel() - 1) // 2

    k_h = k1d.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous()
    k_w = k1d.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous()

    padded = F.pad(image, (0, 0, radius, radius), mode="constant", value=0.0)
    blurred = F.conv2d(padded, k_h, groups=C)
    padded = F.pad(blurred, (radius, radius, 0, 0), mode="constant", value=0.0)
    blurred = F.conv2d(padded, k_w, groups=C)

    return blurred


# ---------------------------------------------------------------------------
# BlurIG with explicit baseline
# ---------------------------------------------------------------------------

def compute_blurig(
    model: nn.Module,
    input: torch.Tensor,
    params: dict,
) -> AttributionResult:
    """
    Compute Blur Integrated Gradients with an explicit baseline.

    Path:  L(sigma) = x_b + G_sigma * (x - x_b),  sigma in [0, max_sigma].

    Args:
        model: PyTorch model that outputs scalar logits (use ClassLogitModel wrapper)
        input: Input tensor (1, C, H, W)
        params: Dictionary with:
            - baseline: Baseline tensor (1, C, H, W)  [REQUIRED, like IG]
            - N: number of integration steps (default: 100)
            - max_sigma: maximum Gaussian std-dev (default: 50.0)
            - grad_step: finite-difference step for d L / d sigma (default: 0.01)
            - sqrt: if True, sigma_i grows as sqrt(i / N) * max_sigma; else linear.

    Returns:
        AttributionResult containing attributions and per-step diagnostics
        consistent with ig.py / guided_ig.py.

    Path indexing (k = 0 ... N):
        gamma_pts[0] = L(sigma = max_sigma) ~ baseline
        gamma_pts[N] = L(sigma = 0)         = input
    """
    # -------- argument parsing / validation ----------------------------------
    if "baseline" not in params or params["baseline"] is None:
        raise ValueError("compute_blurig requires `baseline` in params (same interface as IG).")

    baseline = params["baseline"]
    N = int(params.get("N", 100))
    max_sigma = float(params.get("max_sigma", 50.0))
    grad_step = float(params.get("grad_step", 0.01))
    sqrt_spacing = bool(params.get("sqrt", False))

    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    if max_sigma <= 0:
        raise ValueError(f"max_sigma must be > 0, got {max_sigma}")
    if grad_step <= 0:
        raise ValueError(f"grad_step must be > 0, got {grad_step}")
    if input.dim() != 4 or input.shape[0] != 1:
        raise ValueError(f"input must have shape (1, C, H, W); got {tuple(input.shape)}")
    if baseline.shape != input.shape:
        raise ValueError(f"baseline shape {baseline.shape} != input shape {input.shape}")

    t0 = time.time()
    device = input.device

    # Residual signal that gets blurred along the path.
    delta = input - baseline  # x - x_b

    # -------- sigma schedule -------------------------------------------------
    # sigmas[0] = 0 (input end), sigmas[N] = max_sigma (baseline end).
    # We integrate over i = 0..N-1 with step (sigmas[i+1] - sigmas[i]),
    # then flip sign so attribution describes baseline -> input.
    if sqrt_spacing:
        sigmas = [math.sqrt(float(i) * max_sigma / float(N)) for i in range(N + 1)]
    else:
        sigmas = [float(i) * max_sigma / float(N) for i in range(N + 1)]
    step_diffs = [sigmas[i + 1] - sigmas[i] for i in range(N)]

    # -------- main loop ------------------------------------------------------
    # Integrand: at scale sigma_i,
    #     L_i = baseline + G_{sigma_i} * delta
    #     dL/dsigma ~ (G_{sigma_i + h} * delta - G_{sigma_i} * delta) / h
    # attr = -sum_i  step_diffs[i] * (dL/dsigma at sigma_i) * grad f(L_i)
    attr = torch.zeros_like(input)

    f_vals_inputorder: list[float] = []
    gnorms_inputorder: list[float] = []
    d_inputorder: list[float] = []
    df_inputorder: list[float] = []
    gamma_inputorder: list[torch.Tensor] = []

    # L_0 = baseline + delta = input (since sigma_0 = 0).
    L_curr = input
    f_curr = _forward_scalar(model, L_curr)
    gamma_inputorder.append(L_curr.detach().clone())
    f_vals_inputorder.append(f_curr)

    for i in range(N):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]

        # L_i along the path
        if i == 0:
            L_i = L_curr  # = input
            blur_delta_i = delta  # G_0 * delta = delta
        else:
            blur_delta_i = _gaussian_blur(delta, sigma_i)
            L_i = baseline + blur_delta_i

        # dL/dsigma at sigma_i via forward finite difference of the blurred residual.
        blur_delta_i_plus = _gaussian_blur(delta, sigma_i + grad_step)
        dL_dsigma = (blur_delta_i_plus - blur_delta_i) / grad_step

        # grad f at L_i
        _, grad_at_Li = _forward_and_gradient(model, L_i)

        # Integrand contribution; sign flip applied at end.
        attr = attr + step_diffs[i] * dL_dsigma * grad_at_Li

        # ---- diagnostics (input-space ordering: i goes input -> baseline) ----
        blur_delta_next = _gaussian_blur(delta, sigma_next)
        L_next = baseline + blur_delta_next
        f_next = _forward_scalar(model, L_next)
        delta_gamma = L_next - L_i  # points toward more-blurred (baseline) direction

        gnorms_inputorder.append(float(grad_at_Li.norm()))
        d_inputorder.append(float((grad_at_Li * delta_gamma).sum()))
        df_inputorder.append(f_next - f_curr)
        gamma_inputorder.append(L_next.detach().clone())
        f_vals_inputorder.append(f_next)

        f_curr = f_next

    # Conventional sign: path runs baseline -> input.
    attr = -attr

    # ---- reverse to baseline -> input ordering, like ig.py / guided_ig.py ---
    gamma_pts = list(reversed(gamma_inputorder))
    f_vals = list(reversed(f_vals_inputorder))
    gnorms = list(reversed(gnorms_inputorder))
    d_list = [-d for d in reversed(d_inputorder)]
    df_list = [-df for df in reversed(df_inputorder)]

    # Uniform measure: BlurIG sits at (lambda=0, tau->inf) in the LIG framework.
    mu = torch.full((N,), 1.0 / N, device=device)

    return _pack_result(
        "BlurIG",
        attr,
        d_list=d_list,
        df_list=df_list,
        f_vals=f_vals,
        gnorms=gnorms,
        mu=mu,
        N=N,
        t0=t0,
        gamma_pts=gamma_pts,
    )