"""
Config: đọc YAML, khoá bằng SHA-256, phân tích spec phương pháp.

Mọi khoá ngoài mục `local:` (đường dẫn, thiết bị) được băm. Lệnh nào chạy trên dữ liệu
thật đều gọi `require_lock`: config phải khớp hash đã ghi trong `<config>.lock`.

Spec phương pháp: "tên:k=v,k=v". Giá trị giữ nguyên dạng chuỗi (ví dụ lam=1/11) để
khoá lưu trữ ổn định; dùng `num()` khi cần số.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from fractions import Fraction

import yaml

LOCAL_KEYS = ("local",)


def load(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = os.path.abspath(path)
    return cfg


def _locked_part(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in LOCAL_KEYS and not k.startswith("_")}


def digest(cfg: dict) -> str:
    blob = json.dumps(_locked_part(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def lock_file(cfg: dict) -> str:
    return cfg["_path"] + ".lock"


def write_lock(cfg: dict, force: bool = False, reason: str = "") -> str:
    p = lock_file(cfg)
    new = digest(cfg)
    history = []
    if os.path.exists(p):
        with open(p) as fh:
            old = json.load(fh)
        history = old.get("history", [])
        if old["sha256"] == new:
            print(f"[lock] config đã khoá, hash không đổi: {new[:16]}")
            return new
        if not force:
            raise SystemExit(f"[lock] config đã khoá với hash {old['sha256'][:16]}, hiện tại {new[:16]}.\n"
                             "Nếu thật sự muốn đổi: --force --reason '...' (lý do được ghi vào file lock).")
        history.append({"sha256": old["sha256"], "time": old.get("time")})
    rec = {"sha256": new, "time": _dt.datetime.now().isoformat(timespec="seconds"),
           "reason": reason, "history": history}
    with open(p, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"[lock] đã khoá {cfg['_path']} -> {new[:16]}")
    return new


def require_lock(cfg: dict) -> str:
    p = lock_file(cfg)
    if not os.path.exists(p):
        raise SystemExit(f"[lock] chưa khoá config. Chạy: python -m spi_study --config {cfg['_path']} lock")
    with open(p) as fh:
        rec = json.load(fh)
    d = digest(cfg)
    if rec["sha256"] != d:
        raise SystemExit(f"[lock] config đã bị sửa sau khi khoá (lock {rec['sha256'][:16]}, hiện tại {d[:16]}).")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# số và spec
# ─────────────────────────────────────────────────────────────────────────────

def num(v) -> float:
    """1, 0.5, "1/11", "1e5" -> float."""
    if isinstance(v, bool):
        raise ValueError(f"không phải số: {v!r}")
    if isinstance(v, (int, float)):
        return float(v)
    return float(Fraction(str(v).strip()))


def parse_spec(spec: str) -> tuple[str, dict]:
    name, _, rest = str(spec).partition(":")
    params = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        params[k.strip()] = v.strip()
    return name.strip(), params


def canon(spec: str) -> str:
    """Chuẩn hoá: khoá sắp xếp, không khoảng trắng."""
    name, p = parse_spec(spec)
    if not p:
        return name
    return name + ":" + ",".join(f"{k}={p[k]}" for k in sorted(p))


def frac_str(x: Fraction) -> str:
    """Fraction(1, 11) -> '1/11', Fraction(1, 1) -> '1'."""
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def beta_spec(alpha) -> str:
    return canon(f"spi_dp:alpha={alpha},node=rand")


def mix_spec(alpha) -> str:
    return canon(f"coord_mix:lam={frac_str(Fraction(1, int(alpha) + 1))},node=rand")


def line_spec(alpha) -> str:
    return canon(f"line:s={frac_str(Fraction(2, int(alpha) + 1))}")
