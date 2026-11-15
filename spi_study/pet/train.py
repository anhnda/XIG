"""
Huấn luyện ResNet-18 trên Oxford-IIIT Pet, cùng quy trình, chỉ khác seed (protocol Giai đoạn 2).

Định nghĩa "tương đương" phải chốt trước trong config (`pet.init`):
    imagenet : tinh chỉnh từ cùng weights ImageNet (IMAGENET1K_V1); seed đổi head, thứ tự dữ liệu, augmentation
    scratch  : huấn luyện từ đầu; seed đổi toàn bộ khởi tạo, thứ tự dữ liệu, augmentation
Tiền xử lý test giống hệt bước attribution: resize thẳng 224×224 + chuẩn hoá ImageNet.
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from ..data import IMAGENET_MEAN, IMAGENET_STD
from ..model import pet_resnet18
from ..setting import pet_ckpt


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def test_transform(size: int = 224):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def train_transform(size: int, scale):
    return T.Compose([T.RandomResizedCrop(size, scale=tuple(scale)), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


@torch.no_grad()
def predict(model: nn.Module, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    for xb, yb in loader:
        logits.append(model(xb.to(device)).float().cpu())
        labels.append(yb)
    return torch.cat(logits), torch.cat(labels)


def run_train(cfg: dict, seed: int, out: str, root: str, device) -> dict:
    from torchvision.datasets import OxfordIIITPet
    pc = cfg["pet"]
    tc = pc["train"]
    size = int(cfg["data"]["pet"]["size"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = False
    _seed_all(seed)

    tr = OxfordIIITPet(root, split="trainval", target_types="category",
                       transform=train_transform(size, tc["rrc_scale"]), download=True)
    te = OxfordIIITPet(root, split="test", target_types="category", transform=test_transform(size), download=True)
    g = torch.Generator().manual_seed(seed)
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=int(tc["batch"]), shuffle=True, generator=g,
                                        num_workers=int(tc["num_workers"]), worker_init_fn=_worker_init,
                                        drop_last=True, pin_memory=dev.type == "cuda")
    dl_te = torch.utils.data.DataLoader(te, batch_size=128, shuffle=False, num_workers=int(tc["num_workers"]))

    model = pet_resnet18(pc["init"]).to(dev)          # head (và cả mạng nếu scratch) khởi tạo theo seed
    params = [{"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": float(tc["lr"])},
              {"params": model.fc.parameters(), "lr": float(tc["lr"]) * float(tc.get("head_lr_mult", 1.0))}]
    opt = torch.optim.SGD(params, momentum=float(tc["momentum"]), weight_decay=float(tc["weight_decay"]))
    epochs = int(tc["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl_tr))
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            loss = lossf(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(yb)
            n += len(yb)
        print(f"[pet-train s{seed}] epoch {ep + 1}/{epochs} loss={tot / n:.4f} ({time.time() - t0:.0f}s)")
    logits, labels = predict(model, dl_te, dev)
    acc = float((logits.argmax(1) == labels).float().mean())
    path = pet_ckpt(out, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "seed": int(seed),
                "init": pc["init"], "test_acc": acc, "epochs": epochs}, path)
    print(f"[pet-train s{seed}] test acc = {acc:.4f} -> {path}")
    return {"seed": seed, "acc": acc, "path": path}
