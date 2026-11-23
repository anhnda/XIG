"""
Thống kê cho proposal infidelity: đơn vị là ẢNH (mỗi đại lượng tính trong ảnh từ mọi cặp của ảnh, rồi bootstrap theo ảnh;
không lấy mẫu lại cặp ở tầng trong). Quy tắc quyết định đặt trước:
  xác nhận   cận dưới CI ≥ δ
  bác bỏ     cận trên CI < δ           (non-superiority một phía)
  còn lại    chưa kết luận ("CI chứa 0" KHÔNG phải bác bỏ)
G0 là cổng PHÂN BỔ TÀI NGUYÊN một chiều: không bao giờ dùng để dừng nghiên cứu.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from statistics import NormalDist

import numpy as np

CONFIRM, REJECT, OPEN, UNSET = "xác nhận", "bác bỏ", "chưa kết luận", "δ chưa khoá"


def boot_ci(v, B: int = 10000, seed: int = 0, alpha: float = 0.05, stat=np.mean) -> tuple:
    """(ước lượng, lo, hi) — bootstrap percentile theo ảnh."""
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return (math.nan,) * 3
    if len(v) == 1:
        return float(stat(v)), math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (int(B), len(v)))
    bs = np.apply_along_axis(stat, 1, v[idx]) if stat is not np.mean else v[idx].mean(1)
    return float(stat(v)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def boot_ratio(num, den, B: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple:
    """Tỷ số Σnum/Σden với bootstrap theo ảnh (cặp (num, den) đi cùng nhau). Có chệch hữu hạn mẫu."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    if len(num) == 0 or den.sum() == 0:
        return (math.nan,) * 3
    est = float(num.sum() / den.sum())
    if len(num) == 1:
        return est, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (int(B), len(num)))
    bs = num[idx].sum(1) / np.where(den[idx].sum(1) == 0, np.nan, den[idx].sum(1))
    return est, float(np.nanquantile(bs, alpha / 2)), float(np.nanquantile(bs, 1 - alpha / 2))


def decide(lo: float, hi: float, delta) -> str:
    if delta is None:
        return UNSET
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return OPEN
    if lo >= float(delta):
        return CONFIRM
    if hi < float(delta):
        return REJECT
    return OPEN


def decide_g0(lo: float, delta) -> str:
    if delta is None:
        return UNSET
    if np.isfinite(lo) and lo >= float(delta):
        return "mở E1a + E1b với ngân sách đầy đủ"
    return ("chưa thấy dư địa — xem lại độ chặt của sàn (patch là cận dưới), sai số ước lượng và K; KHÔNG dừng nghiên cứu, "
            "chỉ dừng một nhánh khi có cận đủ tin cậy cho thấy hiệu ứng thực tiễn < ngưỡng")


def n_required(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8, one_sided: bool = True) -> float:
    """Số ảnh cho phép so theo cặp (xấp xỉ chuẩn): n = ((z_{1−α} + z_{power}) · sd / δ)²."""
    if not (np.isfinite(sd) and delta and delta > 0):
        return math.nan
    z = NormalDist().inv_cdf
    za = z(1 - alpha) if one_sided else z(1 - alpha / 2)
    return float(math.ceil(((za + z(power)) * sd / float(delta)) ** 2))


def plan_m(per_image_est, per_image_jk_se, m_used: int, ms=(4, 8, 16, 32, 64), ns=(16, 32, 48, 64), z: float = 1.96) -> list:
    """Chọn số cặp m theo độ rộng CI: Var(ước lượng ảnh) = s_b² + s_w²/m, s_w² = m_used·mean(SE²_jk), s_b² = Var_ảnh − mean(SE²_jk).
    Nửa độ rộng CI của trung bình ảnh ≈ z·sqrt((s_b² + s_w²/m)/n)."""
    v = np.asarray(per_image_est, float)
    se = np.asarray(per_image_jk_se, float)
    ok = np.isfinite(v) & np.isfinite(se)
    v, se = v[ok], se[ok]
    if len(v) < 2:
        return []
    w2 = float(np.mean(se ** 2))
    sw2 = m_used * w2
    sb2 = max(float(np.var(v, ddof=1)) - w2, 0.0)
    return [{"m": m, "n_images": n, "half_width": z * math.sqrt((sb2 + sw2 / m) / n), "s_between2": sb2, "s_within2": sw2}
            for m in ms for n in ns]


# ─────────────────────────────────────────────────────────────────────────────
# khoá protocol
# ─────────────────────────────────────────────────────────────────────────────

def write_prereg(path: str, digest: str, payload: dict) -> None:
    rec = {"digest": digest, "time": _dt.datetime.now().isoformat(timespec="seconds"), **payload}
    if os.path.exists(path):
        with open(path) as fh:
            rec["previous"] = json.load(fh)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(f"[infid-lock] {path}: {digest[:16]}")


def require_prereg(path: str, digest: str, debug: bool) -> dict | None:
    if debug:
        return None
    if not os.path.exists(path):
        raise SystemExit(f"bước này cần protocol đã khoá: bash scripts/run.sh infid-lock (chưa thấy {path})")
    with open(path) as fh:
        rec = json.load(fh)
    if rec["digest"] != digest:
        raise SystemExit(f"protocol đã sửa sau khi khoá ({path}); bước xác nhận phải dùng đúng protocol đã khoá")
    return rec
