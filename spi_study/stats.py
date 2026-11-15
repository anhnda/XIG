"""Thống kê: BCa bootstrap, Wilcoxon một phía, Holm, power, bảng dài."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import NormalDist

import numpy as np

_N = NormalDist()


def z(q: float) -> float:
    return _N.inv_cdf(q)


def bca_ci(d, conf: float = 0.975, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Khoảng tin cậy BCa (hai phía, mức `conf`) cho trung bình của d."""
    d = np.asarray(d, dtype=np.float64)
    if len(d) < 3 or np.allclose(d, d[0]):
        m = float(d.mean()) if len(d) else float("nan")
        return m, m
    from scipy.stats import bootstrap
    kw = dict(confidence_level=conf, n_resamples=n_boot, method="BCa", vectorized=True, axis=0)
    try:                                                   # scipy ≥ 1.15
        res = bootstrap((d,), np.mean, rng=np.random.default_rng(seed), **kw)
    except TypeError:                                      # scipy cũ
        res = bootstrap((d,), np.mean, random_state=np.random.default_rng(seed), **kw)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def pct_ci(d, conf: float = 0.95, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    d = np.asarray(d, dtype=np.float64)
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    a = (1 - conf) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def wilcoxon_greater(d) -> float:
    d = np.asarray(d, dtype=np.float64)
    if len(d) < 2 or np.allclose(d, 0):
        return float("nan")
    from scipy.stats import wilcoxon
    return float(wilcoxon(d, alternative="greater").pvalue)


def holm(pvals: list[float]) -> list[float]:
    """p hiệu chỉnh Holm–Bonferroni (NaN giữ nguyên)."""
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    out = list(pvals)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * pvals[i]))
        out[i] = running
    return out


def n_required(sd: float, delta: float, alpha_two_sided: float = 0.025, power: float = 0.8) -> int:
    """n = ((z_{1−α/2} + z_{power}) · SD / δ)²; protocol: α = 0.05/2 (Bonferroni) → z_{0.9875}."""
    k = z(1 - alpha_two_sided / 2) + z(power)
    return int(math.ceil((k * sd / delta) ** 2))


def achieved_power(sd: float, delta: float, n: int, alpha_two_sided: float = 0.025) -> float:
    if sd <= 0 or n <= 0:
        return float("nan")
    return _N.cdf(math.sqrt(n) * delta / sd - z(1 - alpha_two_sided / 2))


# ─────────────────────────────────────────────────────────────────────────────
# bảng dài metrics.jsonl
# ─────────────────────────────────────────────────────────────────────────────

class Table:
    """Tra cứu (spec, level, b) -> {image: row}."""

    def __init__(self, rows: list[dict]):
        self.t = defaultdict(dict)
        for r in rows:
            self.t[(r["spec"], int(r["level"]), r["b"])][r["image"]] = r

    def get(self, spec, level, b="mean") -> dict:
        return self.t.get((spec, int(level), b), {})

    def levels(self, spec, images, b="mean") -> list[int]:
        out = []
        for (s, L, bb), d in self.t.items():
            if s == spec and bb == b and all(im in d for im in images):
                out.append(L)
        return sorted(set(out))

    def values(self, spec, level, metric, images, b="mean") -> np.ndarray:
        d = self.get(spec, level, b)
        return np.array([float(d[im].get(metric, float("nan"))) if im in d else float("nan") for im in images])

    def sub_se(self, spec, level, metric, images, n_sub) -> np.ndarray:
        """Sai số chuẩn theo ảnh của metric trên tham chiếu, từ n_sub sub-estimate: sd/√n_sub."""
        vals = np.stack([self.values(spec, level, metric, images, b) for b in range(n_sub)])
        return vals.std(axis=0, ddof=1) / math.sqrt(n_sub)


def describe(v) -> str:
    v = np.asarray(v, dtype=np.float64)
    v = v[~np.isnan(v)]
    if not len(v):
        return "–"
    return f"{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}"
