"""
Một "setting" = (bộ dữ liệu, mô hình, quy tắc lớp đích) và thư mục kết quả của nó.

imagenet          -> <out>/imagenet            ResNet-50 torchvision, lớp dự đoán
pet, seed s       -> <out>/pet_s<s>            ResNet-18 Pet seed s, lớp dự đoán (Cửa 1 trên Pet)
                  -> <out>/pet_s<s>_label      cùng mô hình, lớp đích = nhãn thật (Giai đoạn 2)
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from . import data as DATA
from . import model as MODEL
from .storage import Store


class Setting:
    def __init__(self, cfg: dict, kind: str = "imagenet", out: Optional[str] = None,
                 device: Optional[str] = None, pet_seed: Optional[int] = None,
                 target: str = "predicted", mat: Optional[str] = None, pet_root: Optional[str] = None,
                 limit: Optional[int] = None):
        loc = cfg.get("local", {}) or {}
        self.cfg, self.kind, self.target = cfg, kind, target
        self.out = out or loc.get("out", "runs")
        dev = device or loc.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.mat = mat or loc.get("imagenet_mat")
        self.pet_root = pet_root or loc.get("pet_root", "data/pet")
        self.pet_seed = pet_seed
        self.limit = limit
        if target not in ("predicted", "label"):
            raise ValueError(f"target phải là predicted|label, nhận {target}")
        if kind == "imagenet":
            if target != "predicted":
                raise SystemExit("ImageNet-Seg chỉ dùng lớp dự đoán")
            self.tag = "imagenet"
        elif kind == "pet":
            if pet_seed is None:
                raise SystemExit("--setting pet cần --pet-seed")
            self.tag = f"pet_s{pet_seed}" + ("_label" if target == "label" else "")
        else:
            raise ValueError(kind)
        self.root = os.path.join(self.out, self.tag)
        self.store = Store(self.root)
        self._ds = None
        self._bb = None

    # dữ liệu ----------------------------------------------------------------
    def dataset(self):
        if self._ds is None:
            if self.kind == "imagenet":
                self._ds = DATA.ImageNetSeg(self.mat, self.cfg["data"]["imagenet"]["size"])
            else:
                self._ds = DATA.PetTest(self.pet_root, self.cfg["data"]["pet"]["size"])
        return self._ds

    def index_file(self) -> str:
        if self.kind == "imagenet":
            return os.path.join(self.out, "imagenet", "index.json")
        return os.path.join(self.out, "pet_index.json")       # cùng 50 ảnh cho mọi seed

    def indices(self) -> list[int]:
        idx = DATA.read_index(self.index_file())
        return idx[: self.limit] if self.limit else idx

    def image_ids(self) -> list[str]:
        return [DATA.image_id(self.kind, i) for i in self.indices()]

    # mô hình ----------------------------------------------------------------
    def backbone(self):
        if self._bb is None:
            MODEL.set_precision(self.cfg["model"]["allow_tf32"])
            if self.kind == "imagenet":
                m = self.cfg["model"]
                self._bb = MODEL.load_imagenet(m["arch"], m["weights"], self.device)
            else:
                self._bb = MODEL.load_pet(pet_ckpt(self.out, self.pet_seed), self.device)
        return self._bb

    def model_desc(self) -> str:
        if self.kind == "imagenet":
            return f"{self.cfg['model']['arch']}/{self.cfg['model']['weights']}"
        return f"pet_resnet18/seed{self.pet_seed}"


def pet_ckpt(out: str, seed: int) -> str:
    return os.path.join(out, "pet", "models", f"seed{int(seed)}.pt")
