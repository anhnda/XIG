"""
Mô hình và grad_F.

grad_F(Zb): x = x′ + Zb ⊙ Δx, trả Δx ⊙ ∂f_c/∂x (c cố định), model.eval(), float32, không AMP.
TF32 bị tắt khi `model.allow_tf32: false` (mặc định trên A100 PyTorch bật TF32 cho conv).
"""

from __future__ import annotations

import torch
import torch.nn as nn

PET_CLASSES = 37


def set_precision(allow_tf32: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.benchmark = False


def _freeze(m: nn.Module, device) -> nn.Module:
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_imagenet(arch: str, weights: str, device) -> nn.Module:
    import torchvision.models as tvm
    w = tvm.get_model_weights(arch)[weights]
    return _freeze(tvm.get_model(arch, weights=w), device)


def pet_resnet18(init: str = "imagenet") -> nn.Module:
    import torchvision.models as tvm
    w = tvm.ResNet18_Weights.IMAGENET1K_V1 if init == "imagenet" else None
    m = tvm.resnet18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, PET_CLASSES)
    return m


def load_pet(ckpt: str, device) -> nn.Module:
    ck = torch.load(ckpt, map_location="cpu")
    m = pet_resnet18(init="scratch")
    m.load_state_dict(ck["state_dict"])
    return _freeze(m, device)


class TargetLogit(nn.Module):
    def __init__(self, backbone: nn.Module, target: int):
        super().__init__()
        self.backbone, self.target = backbone, int(target)

    def forward(self, x):
        return self.backbone(x)[:, self.target]


class TargetProb(nn.Module):
    def __init__(self, backbone: nn.Module, target: int):
        super().__init__()
        self.backbone, self.target = backbone, int(target)

    def forward(self, x):
        return torch.softmax(self.backbone(x), dim=-1)[:, self.target]


class ImageGradF:
    """grad_F trên ảnh. x, xb: (1, 3, H, W) đã chuẩn hoá ImageNet, trên cùng thiết bị với backbone."""

    def __init__(self, backbone: nn.Module, x: torch.Tensor, xb: torch.Tensor, target: int):
        self.backbone, self.target = backbone, int(target)
        self.xb = xb.detach()
        self.dx = (x - xb).detach()
        self.device = x.device
        self.n_bwd = 0

    def __call__(self, Zb: torch.Tensor) -> torch.Tensor:
        Zb = Zb.to(device=self.device, dtype=self.dx.dtype)
        xin = (self.xb + Zb * self.dx).detach().requires_grad_(True)       # (B, 3, H, W)
        with torch.enable_grad():
            out = self.backbone(xin)[:, self.target]
            (g,) = torch.autograd.grad(out.sum(), xin)
        self.n_bwd += int(xin.shape[0])
        return g * self.dx


@torch.no_grad()
def endpoint(backbone: nn.Module, x: torch.Tensor, xb: torch.Tensor, target: int) -> tuple[float, float]:
    out = backbone(torch.cat([x, xb], dim=0))[:, int(target)]
    return float(out[0]), float(out[1])
