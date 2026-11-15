"""
Unit test U1–U9 (protocol 8.1) cho mọi sampler, trên toy có đáp án giải tích.

Quy ước: N = 200, đạt khi |ước lượng − đáp án| < 3·SE + 1e-3. Các sampler SPI_DP/COORD_MIX dùng
luật nút 'rand' (mặc định của bộ code). Luật 'mid' của protocol được kiểm riêng ở cuối file: nó
CHỆCH ở U5 (đó là lý do bộ code lệch khỏi 6.2).

Thời gian: vài phút trên CPU. SPI_TEST_PMAX / SPI_TEST_PMAX_BQ giảm số đường nếu cần chạy nhanh.
"""

from __future__ import annotations

import math

import pytest
import torch

from spi_study.samplers import CoordMix, IGMid, IGShift, LineMix, SPIBQ, SPIDP
from spi_study.seeding import generator
from toy import (F64, assert_close, endpoints, f_u1, f_u4, f_u5, f_u6, f_u7, mc)

N = 200


def h_of(alpha):
    return 1.0 / (alpha + 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# U1–U3: z₁²z₂
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("make", [lambda: IGMid(N), lambda: IGShift(N)], ids=["ig_mid", "ig_shift"])
def test_u1_ig(make):
    est, se, _, _ = mc(make(), f_u1, 2, seed=1)
    assert_close(est, se, [2 / 3, 1 / 3], "U1")


U2_ANS = [2 / 3 - h_of(10) / 6, 1 / 3 + h_of(10) / 6]      # (0.651515, 0.348485)


def test_u2_spi_dp():
    est, se, _, _ = mc(SPIDP(10, N, "rand"), f_u1, 2, seed=2)
    assert_close(est, se, U2_ANS, "U2 SPI_DP α=10")


@pytest.mark.slow
def test_u2_spi_bq():
    est, se, _, _ = mc(SPIBQ(10, N), f_u1, 2, seed=22)
    assert_close(est, se, U2_ANS, "U2 SPI_BQ α=10")


def test_u3_coord_mix():
    est, se, _, _ = mc(CoordMix(1 / 11, N, "rand"), f_u1, 2, seed=3)
    assert_close(est, se, U2_ANS, "U3 COORD_MIX λ=1/11")


# ─────────────────────────────────────────────────────────────────────────────
# U4: z₁⁴z₂ — kiểm qua mô-men của sampler: A₂ = ∫₀¹ E[Z₁(t)⁴] dt
# ─────────────────────────────────────────────────────────────────────────────

A2_SPI1 = 251 / 720
A2_MIX_HALF = 252 / 720


def a2_moment(sampler, seed, d=1000, paths=200, chunk=20):
    """∫ Z(t)⁴ dt theo từng (đường, phần tử), trapezoid trên nút k/N."""
    gen = generator(seed, "cpu")
    vals = []
    done = 0
    while done < paths:
        p = min(chunk, paths - done)
        Z = sampler.nodes(p, (d,), gen)                       # (N+1, p, d)
        vals.append(torch.trapezoid(Z ** 4, dx=1.0 / sampler.N, dim=0).flatten())
        done += p
    v = torch.cat(vals)
    return float(v.mean()), float(v.std(unbiased=True) / math.sqrt(v.numel()))


@pytest.mark.parametrize("make,ans,paths", [
    (lambda: SPIDP(1, N, "rand"), A2_SPI1, 200),
    (lambda: SPIBQ(1, N), A2_SPI1, 40),        # chỉ mô-men tại nút (không dùng Ż), α=1 cho phép ở đây
    (lambda: CoordMix(0.5, N, "rand"), A2_MIX_HALF, 200),
], ids=["spi_dp_a1", "spi_bq_a1", "coord_mix_half"])
def test_u4_moment(make, ans, paths):
    m, se = a2_moment(make(), seed=4, paths=paths)
    assert abs(m - ans) < 3 * se + 1e-3, f"U4: {m:.6f} ± {se:.6f} vs {ans:.6f}"


@pytest.mark.parametrize("make,a2", [
    (lambda: SPIDP(1, N, "rand"), A2_SPI1),
    (lambda: CoordMix(0.5, N, "rand"), A2_MIX_HALF),
], ids=["spi_dp_a1", "coord_mix_half"])
def test_u4_gradient(make, a2):
    """Cùng đáp án qua gradient: A = (1 − A₂, A₂) (công thức dạng tích, 8.2)."""
    est, se, _, _ = mc(make(), f_u4, 2, seed=44)
    assert_close(est, se, [1 - a2, a2], "U4 gradient")


@pytest.mark.slow
def test_u4_distinguish():
    """Hai họ trùng ở bậc ≤ 4 nhưng khác ở bậc 5: 251/720 vs 252/720 phải phân biệt được."""
    m1, s1 = a2_moment(SPIDP(1, 100, "rand"), seed=5, d=2000, paths=2000, chunk=20)
    m2, s2 = a2_moment(CoordMix(0.5, 100, "rand"), seed=6, d=2000, paths=2000, chunk=20)
    diff, sd = m2 - m1, math.hypot(s1, s2)
    assert abs(diff - 1 / 720) < 3 * sd + 2e-5, f"Δ = {diff:.6f} ± {sd:.6f}, kỳ vọng {1 / 720:.6f}"
    assert diff > 3 * sd, f"không phân biệt được hai họ: Δ = {diff:.6f} ± {sd:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# U5: z₁³ − 3z₁z₂²
# ─────────────────────────────────────────────────────────────────────────────

def test_u5_ig():
    est, se, _, _ = mc(IGMid(N), f_u5, 2, seed=7)
    assert_close(est, se, [0.0, -2.0], "U5 IG")


@pytest.mark.parametrize("make", [lambda: SPIDP(1, N, "rand"), lambda: CoordMix(0.5, N, "rand")],
                         ids=["spi_dp_a1", "coord_mix_half"])
def test_u5_spread(make):
    h = 0.5
    est, se, _, _ = mc(make(), f_u5, 2, seed=8)
    assert_close(est, se, [-h / 2, -2 + h / 2], "U5")


# ─────────────────────────────────────────────────────────────────────────────
# U6: Σ ReLU(zᵢ − bᵢ) — đối chứng âm, không phụ thuộc current
# ─────────────────────────────────────────────────────────────────────────────

_SLOW = pytest.mark.slow
ALL = {
    "ig_mid": lambda: IGMid(N),
    "ig_shift": lambda: IGShift(N),
    "spi_dp_a1": lambda: SPIDP(1, N, "rand"),
    "spi_dp_a10": lambda: SPIDP(10, N, "rand"),
    "coord_mix_half": lambda: CoordMix(0.5, N, "rand"),
    "coord_mix_1_11": lambda: CoordMix(1 / 11, N, "rand"),
    "spi_bq_a10": lambda: SPIBQ(10, N),
}
# SPI_BQ: scipy trên CPU và Ż có đuôi nặng → chạy trong test_slow (tuỳ chọn, không chặn Cửa 1)
KEYS = [pytest.param(k, marks=_SLOW) if k.startswith("spi_bq") else k for k in ALL]


@pytest.mark.parametrize("key", KEYS)
def test_u6_additive(key):
    est, se, _, _ = mc(ALL[key](), f_u6, 3, seed=9)
    assert_close(est, se, [0.7, 0.4, 0.2], f"U6 {key}")


# ─────────────────────────────────────────────────────────────────────────────
# U7: z₁z₂z₃ — đối xứng
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", KEYS)
def test_u7_symmetric(key):
    est, se, _, _ = mc(ALL[key](), f_u7, 3, seed=10)
    assert_close(est, se, [1 / 3, 1 / 3, 1 / 3], f"U7 {key}")


# ─────────────────────────────────────────────────────────────────────────────
# U8: E Z(t) = t, Var Z(t) = h·t(1−t); SPI_DP thêm KS so với Beta(αt, α(1−t)) tại nút
# ─────────────────────────────────────────────────────────────────────────────

U8 = {
    "ig_mid": lambda n: IGMid(n),
    "ig_shift": lambda n: IGShift(n),
    "spi_dp_a1": lambda n: SPIDP(1, n, "rand"),
    "spi_dp_a10": lambda n: SPIDP(10, n, "rand"),
    "spi_dp_a100": lambda n: SPIDP(100, n, "rand"),
    "spi_bq_a10": lambda n: SPIBQ(10, n),
    "coord_mix_1_4": lambda n: CoordMix(0.25, n, "rand"),
    "line_s_half": lambda n: LineMix(0.5, n),
}


@pytest.mark.parametrize("key", list(U8))
def test_u8_moments(key):
    smp = U8[key](10)
    gen = generator(11, "cpu")
    # LINE dùng chung biến chọn trong một đường: cần nhiều đường, ít phần tử
    P, d = (4000, 5) if key.startswith("line") else (5, 4000)
    Z = smp.nodes(P, (d,), gen)                              # (11, P, d)
    for t in (0.1, 0.3, 0.5, 0.9):
        z = Z[round(t * smp.N)].flatten().to(F64)
        n = z.numel()
        var_exp = smp.h * t * (1 - t)
        mean_se = math.sqrt(max(var_exp, 1e-30) / n)
        assert abs(float(z.mean()) - t) < 4 * mean_se + 1e-9, f"{key} t={t}: mean {float(z.mean())}"
        v = float(z.var(unbiased=True))
        m4 = float(((z - z.mean()) ** 4).mean())
        v_se = math.sqrt(max(m4 - v * v, 0.0) / n)
        assert abs(v - var_exp) < 4 * v_se + 1e-9, f"{key} t={t}: var {v:.6f} vs {var_exp:.6f} (se {v_se:.2e})"
        if key.startswith("spi_dp"):
            from scipy.stats import beta
            a = smp.alpha
            p = ks_censored(z.numpy(), beta(a * t, a * (1 - t)).cdf)
            assert p > 1e-3, f"{key} t={t}: KS (kiểm duyệt) p = {p:.2e}"


def ks_censored(z, cdf, lo=1e-12, hi=1 - 1e-12):
    """KS một mẫu, kiểm duyệt hai đầu (0, lo] và [hi, 1).

    float64 không biểu diễn được phần bù 1 − Z < 1.1e-16: với α nhỏ (Beta(0.9, 0.1) tại t = 0.9) vài phần
    trăm mẫu trùng đúng 1.0. kstest coi phân phối liên tục nên tính D tại điểm hoà đó bằng cả khối dồn
    (~2.5%) và bác bỏ sai. Ở đây chỉ so ECDF với CDF trên (lo, hi) và so khối hai đầu."""
    import numpy as np
    from scipy.stats import kstwo
    z = np.sort(np.asarray(z, dtype=np.float64))
    n = len(z)
    i = np.arange(1, n + 1)
    F = cdf(z)
    ok = (z > lo) & (z < hi)
    d = 0.0
    if ok.any():
        d = max(float((i / n - F)[ok].max()), float((F - (i - 1) / n)[ok].max()))
    d = max(d, abs(float(np.mean(z <= lo)) - float(cdf(lo))), abs(float(np.mean(z < hi)) - float(cdf(hi))))
    return float(kstwo.sf(d, n))


# ─────────────────────────────────────────────────────────────────────────────
# U9: ΣA = F(1) − F(0) trong sai số cầu phương
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("F,d", [(f_u1, 2), (f_u5, 2), (f_u7, 3)], ids=["u1", "u5", "u7"])
def test_u9_completeness(key, F, d):
    _, _, s, s_se = mc(ALL[key](), F, d, seed=12)
    D = endpoints(F, d)
    assert abs(s - D) < 3 * s_se + 1e-3, f"U9 {key}: ΣA = {s:.5f} ± {s_se:.5f}, D = {D:.5f}"


# ─────────────────────────────────────────────────────────────────────────────
# Luật nút 'mid' (protocol 6.2): chệch ở bước nhảy — tài liệu hoá lý do lệch protocol
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("make,expected_bias", [
    (lambda: CoordMix(0.5, N, "mid"), -0.125),     # phần tử kiểu Shapley: 3·(½)² thay vì ∫3u²du = 1, × λ
    (lambda: SPIDP(1, N, "mid"), -1 / 12),         # −E Σ δ³/4 = −(1/4)·2/((α+1)(α+2))
], ids=["coord_mix_half_mid", "spi_dp_a1_mid"])
def test_mid_node_is_biased(make, expected_bias):
    est, se, _, _ = mc(make(), f_u5, 2, seed=13)
    bias = float(est[0]) - (-0.25)
    assert abs(bias - expected_bias) < 3 * float(se[0]) + 5e-3, \
        f"độ chệch A₁ = {bias:+.4f}, lý thuyết {expected_bias:+.4f}"