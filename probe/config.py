"""Đọc config, chế độ debug, khoá cho bước xác nhận, seed theo hash."""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os

import torch
import yaml

UNHASHED = ("local", "debug", "mechanism", "cross", "e5", "e6", "e7", "e8")    # mechanism có hash riêng (mechanism.py), không làm mất cache pilot


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path: str, debug: bool = False) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = os.path.abspath(path)
    cfg["_debug"] = bool(debug)
    if debug:
        cfg = _merge(cfg, cfg.get("debug", {}) or {})
        cfg["_debug"] = True
    return cfg


def digest(cfg: dict) -> str:
    part = {k: v for k, v in cfg.items() if k not in UNHASHED and not k.startswith("_")}
    return hashlib.sha256(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()


def write_lock(cfg: dict, reason: str = "") -> None:
    p = cfg["_path"] + ".lock"
    rec = {"sha256": digest(cfg), "time": _dt.datetime.now().isoformat(timespec="seconds"), "reason": reason}
    if os.path.exists(p):
        with open(p) as fh:
            rec["previous"] = json.load(fh)
    with open(p, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[lock] {p}: {rec['sha256'][:16]}")


def require_lock(cfg: dict) -> None:
    p = cfg["_path"] + ".lock"
    if not os.path.exists(p):
        raise SystemExit("--split confirm cần khoá config trước: python -m probe lock --reason '...'")
    with open(p) as fh:
        h = json.load(fh)["sha256"]
    if h != digest(cfg):
        raise SystemExit("config đã sửa sau khi khoá; bước xác nhận phải dùng đúng config đã khoá")


def out_dir(cfg: dict, split: str) -> str:
    base = (cfg.get("local", {}) or {}).get("out", "runs_probe")
    return os.path.join(base + ("_debug" if cfg["_debug"] else ""), split)


def seed_of(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def generator(seed: int, device) -> torch.Generator:
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(int(seed))
    return g