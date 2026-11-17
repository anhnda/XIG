"""
ImageNet-Segmentation (gtsegs_ijcv.mat), tiền xử lý giống các đợt trước (resize thẳng 224, chuẩn hoá ImageNet).

Chia ảnh: loại mọi chỉ số trong local.exclude_index_files (200 ảnh báo cáo gốc + 50 ảnh Cửa 1/1B),
xáo phần còn lại bằng data.index_seed, rồi cắt liên tiếp: pilot | confirm | other.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
_TF = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


@dataclass
class Item:
    x: torch.Tensor          # (1, 3, H, W)
    mask: torch.Tensor       # (H, W) bool
    image: str
    index: int


class ImageNetSeg:
    def __init__(self, path: str, size: int = 224):
        import h5py
        if not path or not os.path.exists(path):
            raise SystemExit(f"không thấy gtsegs_ijcv.mat: {path!r} (local.imagenet_mat)")
        self.h = h5py.File(path, "r")
        self.size = int(size)
        self.n_total = len(self.h["/value/img"])

    def get(self, i: int) -> Item:
        h = self.h
        img = np.array(h[h["/value/img"][i, 0]]).transpose((2, 1, 0))
        gt = np.array(h[h[h["/value/gt"][i, 0]][0, 0]]).transpose((1, 0))
        pil = Image.fromarray(img.astype(np.uint8)).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        m = Image.fromarray((gt > 0).astype(np.uint8)).resize((self.size, self.size), Image.NEAREST)
        return Item(_TF(pil).unsqueeze(0), torch.from_numpy(np.array(m)) > 0, f"seg{i:05d}", int(i))


def _read(path: str) -> list[int]:
    with open(path) as fh:
        obj = json.load(fh)
    return [int(i) for i in (obj["indices"] if isinstance(obj, dict) else obj)]


def splits(cfg: dict, n_total: int) -> dict:
    excl = set()
    for p in (cfg.get("local", {}) or {}).get("exclude_index_files", []) or []:
        if os.path.exists(p):
            excl |= set(_read(p))
        else:
            print(f"[data] CẢNH BÁO: không thấy {p}; các ảnh trong đó có thể lọt vào pilot")
    rest = [i for i in range(n_total) if i not in excl]
    random.Random(int(cfg["data"]["index_seed"])).shuffle(rest)
    a, b, c = int(cfg["data"]["n_pilot"]), int(cfg["data"]["n_confirm"]), int(cfg["data"]["n_other"])
    return {"pilot": rest[:a], "confirm": rest[a:a + b], "other": rest[a + b:a + b + c], "n_excluded": len(excl),
            "fresh": rest[a + b + c:]}            # phần còn lại, theo đúng thứ tự xáo; chưa dùng ở bất kỳ lượt nào
