"""
Rebuild the image list the legacy batch benchmark used
(compare_methods.py --n-test 50, seed 42, min-conf 0.70):

    random.seed(seed); shuffle(sorted file list); keep images whose top-1
    softmax ≥ min_conf (legacy Resize256 + CenterCrop224); first n.

Writes the indices into FolderImages' sorted file list, for use with
    python -m experiments.run --data folder --images DIR --prep crop \
        --n 50 --index-file runs/old50.json ...

python scripts/old50_index.py --images /path/sample_imagenet1k --out runs/old50.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.common import load_backbone          # noqa: E402
from pathattr.data import FolderImages                 # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="runs/old50.json")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-conf", type=float, default=0.70)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone = load_backbone(args.model, dev)
    ds = FolderImages(args.images, prep="crop")
    pos = {f: i for i, f in enumerate(ds.files)}

    # same list + shuffle as utility.load_image_batch after set_seed(seed)
    jpegs = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith((".jpeg", ".jpg", ".png")))
    random.seed(args.seed)
    random.shuffle(jpegs)

    chosen = []
    for fname in jpegs:
        if len(chosen) >= args.n:
            break
        x, _, _ = ds.get(pos[fname])
        with torch.no_grad():
            c, k = torch.softmax(backbone(x.to(dev)), dim=-1)[0].max(0)
        if float(c) >= args.min_conf:
            chosen.append(pos[fname])
            print(f"  [{len(chosen)}] {fname} → class={int(k)}, conf={float(c):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(chosen, fh)
    print(f"\n{len(chosen)} images → {args.out}")


if __name__ == "__main__":
    main()
