"""
Ngân hàng đường SPI cho PATHSCORE (đầu vào của E0): sinh lịch trình theo khoá đường, tính attribution của từng đường.

Giữ NGUYÊN khoá ngẫu nhiên của P4 ("P4spi", "P4q") ⇒ đường k của (ảnh, seed) tái lập bit-by-bit cache attr.pt đã có.
Tham số đường lấy từ mục `sampling:` của config: α, N từ paths.<spi_from> (không ghi đè), M điểm/đoạn, quy tắc `rule`
(mặc định "rand" = canon_rand của cache cũ; khác "rand" thì đi vào hash PATHSCORE).
"""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import platform
import subprocess
import time

import torch

from . import sampling as SM
from .ctx import Ctx, sync
from .infid_core import IEnv
from .pathquad import aw_stream, integrate

F64 = torch.float64


def settings(cfg: dict) -> dict:
    if "sampling" not in cfg:
        raise SystemExit("config thiếu mục sampling:")
    s = copy.deepcopy(cfg["sampling"])
    src = s.get("spi_from", "spi10")
    pc = cfg["paths"][src]
    if pc.get("kind") != "spi":
        raise SystemExit(f"sampling.spi_from = {src} không phải đường spi")
    s["alpha"], s["N"] = float(pc["alpha"]), int(pc["steps"])
    s["M"] = int(s.get("M", 1))
    s.setdefault("rule", "rand")
    return s


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _load(p: str, dg: str):
    if os.path.exists(p):
        obj = torch.load(p, weights_only=False)
        if obj.get("digest") == dg:
            return obj
    return None


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a * b).sum()) / max(float(a.norm() * b.norm()), 1e-300)


class Env(IEnv):
    """IEnv + cấu hình đường S."""

    def __init__(self, ctx: Ctx, S: dict, idx: int):
        super().__init__(ctx, idx)
        self.S = S


class Bank:
    """Lịch trình theo khoá ("spi", k) / ("spiR", k) cho một (ảnh, seed r, tag)."""

    def __init__(self, E: Env, r: int, tag: str, n_paths: int | None = None):
        self.E, self.r, self.tag, self.S = E, int(r), tag, E.S

    def Z(self, key) -> torch.Tensor:
        E, S = self.E, self.S
        if key[0] == "spi":
            return SM.spi_schedule(S["alpha"], S["N"], E.shape, E.gen("P4spi", self.tag, self.r, E.image, key[1]))
        if key[0] == "spiR":
            return SM.reverse_nodes(self.Z(("spi", key[1])))
        raise ValueError(key)

    def eval_path(self, key, qtag: str = "", rule: str | None = None) -> dict:
        E, S = self.E, self.S
        Z = self.Z(key)
        X = SM.nodes(E.x0, E.dx, Z)
        rule = rule or S.get("rule", "rand")
        if rule == "rand":
            A, _ = aw_stream(E.fn, X, S["M"], E.gen("P4q", self.tag, self.r, E.image, qtag, *key), E.ctx.batch)
        else:
            A = integrate(E.fn, X, S["M"], rule, batch=E.ctx.batch)
        out = {"A": A, "d_hat": SM.d_hat_cols(Z)}
        del Z, X
        return out


def bench_fwd(ctx: Ctx) -> float:
    xb = torch.randn(ctx.batch, 3, 224, 224, device=ctx.dev)
    with torch.no_grad():
        ctx.bb(xb)
        sync()
        t = time.time()
        for _ in range(3):
            ctx.bb(xb)
        sync()
    return (time.time() - t) / (3 * ctx.batch)


def provenance(cfg: dict) -> dict:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git = {}
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        if h.returncode == 0:
            git["commit"] = h.stdout.strip()
            d = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10)
            git["dirty"] = bool(d.stdout.strip())
    except Exception:
        pass
    hs = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(root, "probe", "*.py"))) + [cfg["_path"]]:
        with open(f, "rb") as fh:
            hs.update(os.path.basename(f).encode())
            hs.update(fh.read())
    import torchvision
    return {"git": git or None, "source_sha256": hs.hexdigest(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "python": platform.python_version(), "platform": platform.platform(),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "model": f"torchvision.models.{cfg['model']['arch']} weights={cfg['model']['weights']}",
            "allow_tf32": cfg["model"]["allow_tf32"]}
