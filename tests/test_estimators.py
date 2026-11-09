import torch

from pathattr import estimators as EST
from pathattr import methods as M
from pathattr import paths
from tests.toy import setup


def test_gauss_legendre_is_complete_on_smooth_model():
    model, oracle, x, xb, D = setup("tanh")
    pts = paths.dither(xb, x, N=6, spread=0.5, tile=4)
    a16 = EST.estimate(oracle, pts, "gl", Q=16)
    a8 = EST.estimate(oracle, pts, "gl", Q=8)
    assert abs(float(a16.sum()) - D) < 1e-8
    assert float((a16 - a8).norm() / a16.norm()) < 1e-6


def test_left_estimator_is_riemann_sum():
    model, oracle, x, xb, D = setup("tanh")
    pts = paths.straight(xb, x, 5)
    a = EST.estimate(oracle, pts, "left")
    _, g = oracle.grad(pts[:-1])
    assert torch.allclose(a, (g * (pts[1:] - pts[:-1])).sum(0, keepdim=True), atol=1e-12)


def test_random_interior_is_unbiased():
    model, oracle, x, xb, D = setup("tanh")
    pts = paths.dither(xb, x, N=4, spread=0.5, tile=4)
    ref = EST.estimate(oracle, pts, "gl", Q=16)
    mean = EST.estimate(oracle, pts, "random", Q=4000, gen=torch.Generator().manual_seed(0))
    assert float((mean - ref).norm() / ref.norm()) < 0.03


def test_cv_complete_exact_sum_and_dummy():
    model, oracle, x, xb, D = setup("tanh", dummy_channel=2)
    pts = paths.dither(xb, x, N=6, spread=0.5, tile=4)
    gen = torch.Generator().manual_seed(0)
    pilot = EST.random_draws(oracle, pts, 16, gen)
    fresh = EST.estimate(oracle, pts, "random", 1, gen)
    assert float(pilot[:, 2].abs().max()) == 0.0               # dummy channel
    a = EST.cv_complete(pilot, fresh, D)
    assert abs(float(a.sum()) - D) < 1e-9
    assert float(a[:, 2].abs().max()) == 0.0


def test_random_cv_method_runs_and_is_complete():
    model, oracle, x, xb, D = setup("tanh")
    res = M.dither(oracle, xb, x, torch.Generator().manual_seed(0), N=6, tile=4,
                   est="random_cv", K_cv=6)
    assert abs(float(res.attr.sum()) - D) < 1e-9
