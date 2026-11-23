"""Ngữ cảnh chạy chung: mô hình, dữ liệu, chia ảnh, generator theo hash, đo tốc độ GPU."""

from __future__ import annotations

import os
import time

import torch

from . import config as C
from .data import ImageNetSeg, splits
from .model import TargetFn, load_backbone
from .quad import Counter, fg


class Ctx:
    def __init__(self, cfg: dict, device=None):
        self.cfg = cfg
        dev = device or (cfg.get("local", {}) or {}).get("device", "cuda")
        if str(dev).startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("không thấy CUDA; nếu thật sự muốn chạy CPU: --device cpu")
        self.dev = torch.device(dev)
        self.out = C.out_dir(cfg, "pilot")
        os.makedirs(self.out, exist_ok=True)
        self.ds = ImageNetSeg(cfg["local"]["imagenet_mat"], cfg["data"]["size"])
        self.sp = splits(cfg, self.ds.n_total)
        self.bb = load_backbone(cfg, self.dev)
        self.batch = int(cfg["model"]["batch"])
        self.seed = int(cfg["seed"])

    def load(self, idx: int):
        it = self.ds.get(idx)
        x, mask = it.x.to(self.dev), it.mask.to(self.dev)
        with torch.no_grad():
            cls = int(self.bb(x)[0].argmax())
        return it, x, mask, cls

    def gen(self, *parts):
        return C.generator(C.seed_of(self.seed, *parts), self.dev)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(ctx: Ctx) -> dict:
    """Thời gian một lượt forward+backward theo lô (giây / ảnh)."""
    fn = TargetFn(ctx.bb, 0)
    xb = torch.randn(ctx.batch, 3, 224, 224, device=ctx.dev)
    fg(fn, xb, ctx.batch)
    sync()
    t = time.time()
    for _ in range(3):
        fg(fn, xb, ctx.batch)
    sync()
    per = (time.time() - t) / (3 * ctx.batch)
    Counter.n = 0
    return {"per_eval": per}
