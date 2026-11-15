"""Seed suy từ hash của bộ (ảnh, phương pháp, tham số, mức thang, sub-estimate b)."""

from __future__ import annotations

import hashlib

import torch


def seed_of(*parts) -> int:
    blob = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little") & ((1 << 63) - 1)


def generator(seed: int, device="cpu") -> torch.Generator:
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(int(seed))
    return g
