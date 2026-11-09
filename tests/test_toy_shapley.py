"""
Toy checks of the claims in the review rounds (doc §2.5):
  * multilinear F: E[batched path] = Shapley = IG (any batch size)
  * F = z0² z1 (d = 4): batch m = 2 is biased by q·(1/6, −1/6), q = (m−1)/(d−1)
  * the left-endpoint estimator is not complete even for multilinear F
Segment integrals use 4-point Gauss–Legendre: exact for these polynomials.
"""

import itertools
import math

import numpy as np
import torch

XS, WS = np.polynomial.legendre.leggauss(4)
U = torch.tensor((XS + 1) / 2, dtype=torch.float64)
W = torch.tensor(WS / 2, dtype=torch.float64)


def seg_exact(F, a, b):
    h = b - a
    Z = (a + U[:, None] * h).requires_grad_(True)
    (g,) = torch.autograd.grad(F(Z).sum(), Z)
    return h * (W[:, None] * g).sum(0)


def seg_left(F, a, b):
    z = a[None].clone().requires_grad_(True)
    (g,) = torch.autograd.grad(F(z).sum(), z)
    return (b - a) * g[0]


def batched(F, perm, m, rule):
    d = len(perm)
    z, A = torch.zeros(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64)
    for s in range(0, d, m):
        z2 = z.clone()
        z2[list(perm[s:s + m])] = 1.0
        A += rule(F, z, z2)
        z = z2
    return A


def expected(F, d, m, rule=seg_exact):
    perms = list(itertools.permutations(range(d)))
    return sum(batched(F, p, m, rule) for p in perms) / len(perms)


def shapley(F, d):
    phi = torch.zeros(d, dtype=torch.float64)
    for p in itertools.permutations(range(d)):
        z = torch.zeros(1, d, dtype=torch.float64)
        prev = float(F(z))
        for i in p:
            z[0, i] = 1.0
            cur = float(F(z))
            phi[i] += cur - prev
            prev = cur
    return phi / math.factorial(d)


def F_multi(z):
    return z[:, 0] * z[:, 1] * z[:, 2] + 2 * z[:, 1] * z[:, 3] - z[:, 0] * z[:, 2] * z[:, 3]


def F_sq(z):
    return z[:, 0] ** 2 * z[:, 1]


def test_multilinear_ig_equals_shapley_and_batches():
    sh = shapley(F_multi, 4)
    assert torch.allclose(sh, torch.tensor([0, 4 / 3, 0, 2 / 3], dtype=torch.float64), atol=1e-12)
    for m in (1, 2, 4):                        # m = 4 is the straight line (IG)
        assert torch.allclose(expected(F_multi, 4, m), sh, atol=1e-10)


def test_left_endpoint_not_complete_for_multilinear():
    a = expected(F_multi, 4, 2, seg_left)
    assert torch.allclose(a, torch.tensor([0, 5 / 6, 0, 1 / 2], dtype=torch.float64), atol=1e-10)
    assert abs(float(a.sum()) - 2.0) > 0.5     # 4/3 vs D = 2


def test_non_multilinear_batch_bias():
    sh = shapley(F_sq, 4)
    assert torch.allclose(sh[:2], torch.tensor([0.5, 0.5], dtype=torch.float64), atol=1e-12)
    ig = expected(F_sq, 4, 4)
    assert torch.allclose(ig[:2], torch.tensor([2 / 3, 1 / 3], dtype=torch.float64), atol=1e-10)
    assert torch.allclose(expected(F_sq, 4, 1), sh, atol=1e-10)
    bias = expected(F_sq, 4, 2)[:2] - sh[:2]
    assert torch.allclose(bias, torch.tensor([1 / 18, -1 / 18], dtype=torch.float64), atol=1e-10)
