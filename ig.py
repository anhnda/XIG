"""
ig.py - Standard Integrated Gradients (Sundararajan et al., 2017)
===================================================================

Standard IG: uniform measure μ_k = 1/N, straight-line path from baseline to input.

Usage:
    from ig import compute_ig

    attribution_result = compute_ig(
        model=model,
        input=x,
        params={'baseline': baseline, 'N': 50}
    )
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn

from utility import (
    AttributionResult,
    _straight_line_pass,
    _pack_result,
)


def compute_ig(
    model: nn.Module,
    input: torch.Tensor,
    params: dict,
) -> AttributionResult:
    """
    Compute standard Integrated Gradients attribution.

    Args:
        model: PyTorch model that outputs scalar logits (use ClassLogitModel wrapper)
        input: Input tensor (1, C, H, W)
        params: Dictionary with:
            - baseline: Baseline tensor (1, C, H, W)
            - N: Number of interpolation steps (default: 50)

    Returns:
        AttributionResult containing attributions and metrics
    """
    baseline = params['baseline']
    N = params.get('N', 50)

    t0 = time.time()

    # Compute gradients along straight line
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, input, baseline, N)

    # Standard IG: uniform average of gradients
    grad_sum = torch.cat(grads, dim=0).sum(dim=0, keepdim=True)  # (1, C, H, W)
    attr = delta_x * grad_sum / N

    # Uniform measure
    mu = torch.full((N,), 1.0 / N, device=input.device)

    return _pack_result("IG", attr, d_list, df_list, f_vals, gnorms, mu, N, t0)
