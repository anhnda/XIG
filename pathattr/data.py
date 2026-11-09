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


def _prep(img: Image.Image, mask: Optional[Image.Image], size: int, prep: str = "resize"):
    """prep='resize': straight to size×size (image and mask stay aligned).
    prep='crop'  : Resize(256) + CenterCrop(224), exactly as the legacy code."""
    img = img.convert("RGB")
    if prep == "crop":
        short = int(round(size * 256 / 224))
        img = T.CenterCrop(size)(T.Resize(short)(img))
        if mask is not None:
            mask = T.CenterCrop(size)(T.Resize(short, interpolation=T.InterpolationMode.NEAREST)(mask))
    elif prep == "resize":
        img = img.resize((size, size), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((size, size), Image.NEAREST)
    else:
        raise ValueError(f"unknown prep '{prep}'")
    x = _TO_TENSOR(img).unsqueeze(0)
    m = torch.from_numpy(np.array(mask)) > 0 if mask is not None else None
    return x, m


def select_indices(n_total: int, n: int, seed: int, index_file: Optional[str],
                   size: int = 200) -> list[int]:
    """Fixed random image list.

    If `index_file` exists it is loaded; otherwise max(n, size) indices are drawn
    once (random order, so any prefix is itself a random subset) and saved.
    Returns the first n (all if n = 0). Asking for more than the file holds is an error.
    """
    if index_file and os.path.exists(index_file):
        with open(index_file) as fh:
            idx = [int(i) for i in json.load(fh)]
    else:
        m = n_total if not n else min(max(n, size), n_total)
        idx = random.Random(seed).sample(range(n_total), m)
        if index_file:
            os.makedirs(os.path.dirname(os.path.abspath(index_file)), exist_ok=True)
            with open(index_file, "w") as fh:
                json.dump(idx, fh)
    if n and n > len(idx):
        raise ValueError(f"index file {index_file} holds {len(idx)} images, {n} requested; "
                         f"delete it or use another --index-file")
    return idx[:n] if n else idx


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

    def __init__(self, image_dir: str, mask_dir: Optional[str] = None, size: int = 224,
                 prep: str = "resize"):
        self.image_dir, self.mask_dir, self.size, self.prep = image_dir, mask_dir, size, prep
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
        x, m = _prep(img, mask, self.size, self.prep)
        return x, m, fname


def iterate(ds, indices: list[int]) -> Iterator[tuple[int, torch.Tensor, Optional[torch.Tensor], str]]:
    for i in indices:
        x, m, name = ds.get(i)
        yield i, x, m, name
