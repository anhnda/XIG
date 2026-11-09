"""Tiny float64 models so the tests run in seconds on CPU."""

import torch
import torch.nn as nn

from experiments.common import TargetLogit
from pathattr import Oracle


class ToyBackbone(nn.Module):
    """(B, 3, 8, 8) -> (B, 3) logits. act='tanh' (smooth) or 'relu'."""

    def __init__(self, act: str = "tanh", dummy_channel: int = -1, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = nn.Conv2d(3, 6, 3, padding=1).double()
        self.lin = nn.Linear(6 * 4, 3).double()
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64) * 0.5)
            if dummy_channel >= 0:
                self.conv.weight[:, dummy_channel] = 0.0
        self.act = torch.tanh if act == "tanh" else torch.relu
        self.pool = nn.AdaptiveAvgPool2d(2)

    def forward(self, x):
        h = self.pool(self.act(self.conv(x)))
        return self.lin(h.flatten(1))


def setup(act: str = "tanh", dummy_channel: int = -1, seed: int = 0):
    torch.manual_seed(seed)
    bb = ToyBackbone(act, dummy_channel, seed).eval()
    model = TargetLogit(bb, 1)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(1, 3, 8, 8, generator=g, dtype=torch.float64)
    xb = torch.zeros_like(x)
    with torch.no_grad():
        D = float(model(x)[0] - model(xb)[0])
    return model, Oracle(model), x, xb, D
