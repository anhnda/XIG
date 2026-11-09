"""
Image sources.

ImageNetSeg  — ImageNet-Segmentation (Guillaumin et al.), the `gtsegs_ijcv.mat`
               file (4,276 images with object masks) used by Chefer et al.
               Read exactly as in their loader; image and mask resized to 224×224.
FolderImages — a folder of images, optional masks folder with the same stems.

The subset of images is fixed by an index file (JSON list) so every method,
seed and rerun sees the same images; no confidence filtering by default.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterator, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .metrics import IMAGENET_MEAN, IMAGENET_STD

_TO_TENSOR = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _prep(img: Image.Image, mask: Optional[Image.Image], size: int):
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    x = _TO_TENSOR(img).unsqueeze(0)
    m = None
    if mask is not None:
        m = torch.from_numpy(np.array(mask.resize((size, size), Image.NEAREST))) > 0
    return x, m


def select_indices(n_total: int, n: int, seed: int, index_file: Optional[str]) -> list[int]:
    """Load the index list if the file exists, otherwise draw it once and save it."""
    if index_file and os.path.exists(index_file):
        with open(index_file) as fh:
            idx = json.load(fh)
        return [int(i) for i in idx][:n] if n else [int(i) for i in idx]
    n = n_total if not n else min(n, n_total)
    idx = sorted(random.Random(seed).sample(range(n_total), n))
    if index_file:
        os.makedirs(os.path.dirname(os.path.abspath(index_file)), exist_ok=True)
        with open(index_file, "w") as fh:
            json.dump(idx, fh)
    return idx


class ImageNetSeg:
    def __init__(self, mat_path: str, size: int = 224):
        import h5py  # optional dependency
        self.path = mat_path
        self.size = size
        self._h = h5py.File(mat_path, "r")
        self.n_total = len(self._h["/value/img"])

    def __len__(self) -> int:
        return self.n_total

    def get(self, i: int):
        h = self._h
        img = np.array(h[h["/value/img"][i, 0]]).transpose((2, 1, 0))
        gt = np.array(h[h[h["/value/gt"][i, 0]][0, 0]]).transpose((1, 0))
        x, m = _prep(Image.fromarray(img.astype(np.uint8)),
                     Image.fromarray((gt > 0).astype(np.uint8)), self.size)
        return x, m, f"seg{i:05d}"


class FolderImages:
    EXT = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, image_dir: str, mask_dir: Optional[str] = None, size: int = 224):
        self.image_dir, self.mask_dir, self.size = image_dir, mask_dir, size
        self.files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(self.EXT))
        self.n_total = len(self.files)

    def __len__(self) -> int:
        return self.n_total

    def get(self, i: int):
        fname = self.files[i]
        img = Image.open(os.path.join(self.image_dir, fname))
        mask = None
        if self.mask_dir:
            stem = os.path.splitext(fname)[0]
            for ext in (".png", ".jpg", ".bmp"):
                p = os.path.join(self.mask_dir, stem + ext)
                if os.path.exists(p):
                    mask = Image.open(p).convert("L")
                    break
        x, m = _prep(img, mask, self.size)
        return x, m, fname


def iterate(ds, indices: list[int]) -> Iterator[tuple[int, torch.Tensor, Optional[torch.Tensor], str]]:
    for i in indices:
        x, m, name = ds.get(i)
        yield i, x, m, name
