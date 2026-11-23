"""Mô hình (torchvision, đóng băng) và hàm mục tiêu f(x) = logit lớp đích."""

from __future__ import annotations

import torch
import torch.nn as nn


def set_precision(allow_tf32: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.benchmark = False


def freeze(m: nn.Module, device) -> nn.Module:
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_backbone(cfg: dict, device) -> nn.Module:
    import torchvision.models as tvm
    set_precision(cfg["model"]["allow_tf32"])
    w = tvm.get_model_weights(cfg["model"]["arch"])[cfg["model"]["weights"]]
    return freeze(tvm.get_model(cfg["model"]["arch"], weights=w), device)


class TargetFn:
    """f(x) = backbone(x)[:, c] + Σ extra(x). Nhận (B,3,H,W), trả (B,)."""

    def __init__(self, backbone: nn.Module, cls: int, extras=()):
        self.backbone, self.cls, self.extras = backbone, int(cls), list(extras)

    def with_extra(self, extra) -> "TargetFn":
        return TargetFn(self.backbone, self.cls, self.extras + [extra])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)[:, self.cls]
        for e in self.extras:
            out = out + e(x)
        return out
