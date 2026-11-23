"""Đọc config, chế độ debug, seed theo hash. Mỗi thí nghiệm tự băm phần config của nó để quản lý cache."""

from __future__ import annotations

import copy
import hashlib
import os

import torch
import yaml


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
