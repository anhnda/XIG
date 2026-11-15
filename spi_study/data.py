"""
Dữ liệu.

ImageNetSeg : gtsegs_ijcv.mat (Guillaumin et al.), đọc giống loader cũ, resize thẳng 224×224
              (ảnh bilinear, mask nearest) — cùng tiền xử lý với báo cáo gốc.
PetTest     : Oxford-IIIT Pet, split test, trimap (1 = vật, 2 = nền, 3 = biên); ảnh và trimap
              resize cùng một phép 224×224 (không crop) nên luôn khớp nhau.

Danh sách ảnh mới (Giai đoạn 0): loại 200 ảnh cũ. File index cũ (runs/seg200.json) được đọc nếu có;
nếu không, tái tạo đúng cách code cũ làm: random.Random(0).sample(range(n_total), 200).
Danh sách mới là một hoán vị cố định của phần còn lại, nên tăng n chỉ nối thêm ảnh, không đổi ảnh cũ.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_TO_TENSOR = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


@dataclass
class Item:
    x: torch.Tensor                  # (1, 3, H, W), chuẩn hoá ImageNet
    mask: torch.Tensor               # (H, W) bool
    valid: Optional[torch.Tensor]    # (H, W) bool, None = mọi pixel hợp lệ
    image_id: str
    index: int
    label: Optional[int]
    source: str


def image_id(kind: str, i: int) -> str:
    return f"seg{i:05d}" if kind == "imagenet" else f"pet{i:05d}"


class ImageNetSeg:
    kind = "imagenet"

    def __init__(self, mat_path: str, size: int = 224):
        import h5py
        if not mat_path or not os.path.exists(mat_path):
            raise SystemExit(f"không thấy gtsegs_ijcv.mat: {mat_path!r} (đặt local.imagenet_mat hoặc --mat)")
        self.path, self.size = mat_path, int(size)
        self._h = h5py.File(mat_path, "r")
        self.n_total = len(self._h["/value/img"])

    def __len__(self):
        return self.n_total

    def get(self, i: int) -> Item:
        h = self._h
        img = np.array(h[h["/value/img"][i, 0]]).transpose((2, 1, 0))
        gt = np.array(h[h[h["/value/gt"][i, 0]][0, 0]]).transpose((1, 0))
        pil = Image.fromarray(img.astype(np.uint8)).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        m = Image.fromarray((gt > 0).astype(np.uint8)).resize((self.size, self.size), Image.NEAREST)
        x = _TO_TENSOR(pil).unsqueeze(0)
        mask = torch.from_numpy(np.array(m)) > 0
        return Item(x, mask, None, image_id(self.kind, i), int(i), None, f"{self.path}#{i}")


class PetTest:
    kind = "pet"

    def __init__(self, root: str, size: int = 224, download: bool = True):
        from torchvision.datasets import OxfordIIITPet
        self.ds = OxfordIIITPet(root, split="test", target_types=["category", "segmentation"],
                                download=download)
        self.size = int(size)
        self.n_total = len(self.ds)

    def __len__(self):
        return self.n_total

    def get(self, i: int) -> Item:
        img, (label, tri) = self.ds[i]
        img = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        tri = np.array(tri.resize((self.size, self.size), Image.NEAREST))
        x = _TO_TENSOR(img).unsqueeze(0)
        mask = torch.from_numpy(tri == 1)
        valid = torch.from_numpy(tri != 3)
        return Item(x, mask, valid, image_id(self.kind, i), int(i), int(label), f"pet-test#{i}")


# ─────────────────────────────────────────────────────────────────────────────
# danh sách ảnh
# ─────────────────────────────────────────────────────────────────────────────

def _read_index(path: str) -> list[int]:
    with open(path) as fh:
        obj = json.load(fh)
    return [int(i) for i in (obj["indices"] if isinstance(obj, dict) else obj)]


def old_indices(n_total: int, seed: int, size: int, old_file: Optional[str]) -> tuple[list[int], str]:
    if old_file and os.path.exists(old_file):
        return _read_index(old_file), os.path.abspath(old_file)
    return random.Random(seed).sample(range(n_total), size), f"regenerated: random.Random({seed}).sample(range({n_total}), {size})"


def make_imagenet_index(n_total: int, dcfg: dict, old_file: Optional[str], path: str) -> dict:
    old, src = old_indices(n_total, int(dcfg["old_index_seed"]), int(dcfg["old_index_size"]), old_file)
    n = int(dcfg["n_images"])
    if dcfg.get("reuse_old", False):
        idx, pilot = old[:n], True
        if n > len(old):
            raise SystemExit(f"reuse_old: chỉ có {len(old)} ảnh cũ, cần {n}")
    else:
        used = set(old)
        rest = [i for i in range(n_total) if i not in used]
        random.Random(int(dcfg["new_index_seed"])).shuffle(rest)
        idx, pilot = rest[:n], False
    rec = {"indices": idx, "n": n, "pilot_reused_old_images": pilot, "excluded_old": 0 if pilot else len(old),
           "old_source": src, "new_index_seed": int(dcfg["new_index_seed"]), "n_total": n_total}
    _write_index(path, rec)
    return rec


def make_pet_index(n_total: int, pcfg: dict, path: str) -> dict:
    perm = list(range(n_total))
    random.Random(int(pcfg["index_seed"])).shuffle(perm)
    n = int(pcfg["n_test"])
    rec = {"indices": perm[:n], "n": n, "index_seed": int(pcfg["index_seed"]), "n_total": n_total}
    _write_index(path, rec)
    return rec


def _write_index(path: str, rec: dict) -> None:
    if os.path.exists(path):
        prev = _read_index(path)
        new = rec["indices"]
        k = min(len(prev), len(new))
        if prev[:k] != new[:k]:
            raise SystemExit(f"index {path} đã tồn tại và khác danh sách mới (không phải tiền tố). Xoá file nếu chắc chắn.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[index] {path}: {rec['n']} ảnh" + (" (PILOT: dùng lại ảnh cũ)" if rec.get("pilot_reused_old_images") else ""))


def read_index(path: str) -> list[int]:
    if not os.path.exists(path):
        raise SystemExit(f"chưa có danh sách ảnh {path}. Chạy lệnh `index` trước.")
    return _read_index(path)
