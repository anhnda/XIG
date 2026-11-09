import pytest
import torch

from pathattr import methods as M
from pathattr import paths
from tests.toy import setup


def _xy(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 3, 16, 16, generator=g, dtype=torch.float64)
    xb = torch.randn(1, 3, 16, 16, generator=g, dtype=torch.float64)
    return xb, x


CONFIGS = [
    dict(kind="bayer", spread=0.5), dict(kind="bayer", spread=0.0), dict(kind="bayer", spread=1.0),
    dict(kind="stratified", spread=0.5, tile=4), dict(kind="random", spread=1.0),
    dict(kind="block", spread=1.0, block=5), dict(kind="bayer", spread=0.5, noise_std=0.1),
    dict(kind="random", spread=0.75, per_channel=True),
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_dither_endpoints_exact(cfg):
    xb, x = _xy()
    pts = paths.dither(xb, x, N=12, gen=torch.Generator().manual_seed(3), **cfg)
    assert pts.shape == (13, 3, 16, 16)
    assert torch.equal(pts[0], xb[0])
    assert torch.allclose(pts[-1], x[0], atol=1e-12)


@pytest.mark.parametrize("cfg", [c for c in CONFIGS if c.get("noise_std", 0) == 0])
def test_dither_monotone_without_noise(cfg):
    xb, x = _xy()
    pts = paths.dither(xb, x, N=12, gen=torch.Generator().manual_seed(3), **cfg)
    step = (pts[1:] - pts[:-1]) * torch.sign(x - xb)
    assert float(step.min()) >= -1e-12


def test_spread_zero_is_straight_line():
    xb, x = _xy()
    a = paths.dither(xb, x, N=10, spread=0.0)
    b = paths.straight(xb, x, 10)
    assert torch.allclose(a, b, atol=1e-12)


def test_bayer_every_window_has_every_level():
    T = paths.threshold_map(16, 16, "bayer", tile=8, gen=torch.Generator().manual_seed(1))
    levels = (torch.arange(64, dtype=torch.float64) + 0.5) / 64
    for i in (0, 3, 8):
        for j in (0, 5, 8):
            w = T[i:i + 8, j:j + 8].flatten().sort().values
            assert torch.allclose(w, levels)


def test_stratified_tiles_have_every_level():
    T = paths.threshold_map(12, 12, "stratified", tile=4, gen=torch.Generator().manual_seed(1))
    levels = (torch.arange(16, dtype=torch.float64) + 0.5) / 16
    for i in range(0, 12, 4):
        for j in range(0, 12, 4):
            assert torch.allclose(T[i:i + 4, j:j + 4].flatten().sort().values, levels)


def test_hard_schedule_moves_each_pixel_once():
    xb, x = _xy()
    pts = paths.dither(xb, x, N=8, kind="random", spread=1.0, gen=torch.Generator().manual_seed(0))
    moved = ((pts[1:] - pts[:-1]).abs() > 0).sum(dim=0)       # per (c, i, j)
    assert int(moved.max()) == 1 and int(moved.min()) == 1


@pytest.mark.parametrize("alpha", [0.01, 1.0, 10.0])
def test_spi_endpoints_and_monotone(alpha):
    xb, x = _xy()
    pts = paths.spi(xb, x, N=12, alpha=alpha, gen=torch.Generator().manual_seed(0))
    assert torch.equal(pts[0], xb[0])
    assert torch.allclose(pts[-1], x[0], atol=1e-12)
    step = (pts[1:] - pts[:-1]) * torch.sign(x - xb)
    assert float(step.min()) >= -1e-12


def test_spi_small_alpha_is_almost_one_jump():
    xb, x = _xy()
    pts = paths.spi(xb, x, N=12, alpha=0.01, gen=torch.Generator().manual_seed(0))
    frac = (pts[1:] - pts[:-1]) / (x - xb)                    # share of each step
    # P(first stick ≥ 0.98) = 0.02^(−α)… ≈ 0.96 for α = 0.01
    assert float(frac.max(dim=0).values.mean()) > 0.95


def test_guided_ig_path_reaches_input_monotonically():
    model, oracle, x, xb, D = setup()
    oracle.reset()
    pts, g = M.guided_ig_path(oracle, xb, x, N=20)
    assert oracle.counts()[1] == 20
    assert torch.equal(pts[0], xb[0])
    assert torch.allclose(pts[-1], x[0], atol=1e-12)
    step = (pts[1:] - pts[:-1]) * torch.sign(x - xb)
    assert float(step.min()) >= -1e-12
