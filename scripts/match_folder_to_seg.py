"""
Do the images of an old folder (e.g. sample_imagenet1k, 50 images) appear in
ImageNet-Seg (gtsegs_ijcv.mat)? If yes, export their masks so the folder can be
run with --data folder --masks.

Matching is by content: each image → grayscale, two signatures (whole image and
centre square, 24×24, zero-mean, unit-norm); score = best correlation over the
two. A pair is a match when score ≥ --thr and it beats the runner-up by --margin.
Same original size is reported as extra evidence.

python scripts/match_folder_to_seg.py --mat /path/gtsegs_ijcv.mat \
    --images ./sample_imagenet1k --out-csv runs/folder_seg_match.csv \
    --out-masks ./sample_imagenet1k_masks
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from PIL import Image

EXT = (".jpg", ".jpeg", ".png", ".bmp")
S = 24


def _vec(img: Image.Image) -> np.ndarray:
    v = np.asarray(img.convert("L").resize((S, S), Image.BILINEAR), dtype=np.float32).ravel()
    v = v - v.mean()
    return v / max(float(np.linalg.norm(v)), 1e-6)


def signatures(img: Image.Image) -> np.ndarray:
    w, h = img.size
    m = min(w, h)
    left, top = (w - m) // 2, (h - m) // 2
    return np.stack([_vec(img), _vec(img.crop((left, top, left + m, top + m)))])   # (2, S*S)


def read_seg(h, i: int):
    img = np.array(h[h["/value/img"][i, 0]]).transpose((2, 1, 0)).astype(np.uint8)
    gt = np.array(h[h[h["/value/gt"][i, 0]][0, 0]]).transpose((1, 0))
    return Image.fromarray(img), gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out-csv", default="runs/folder_seg_match.csv")
    ap.add_argument("--out-masks", default=None, help="write <stem>.png masks for matched images")
    ap.add_argument("--thr", type=float, default=0.97)
    ap.add_argument("--margin", type=float, default=0.02)
    args = ap.parse_args()

    import h5py
    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(EXT))
    F, F_size = [], []
    for f in files:
        im = Image.open(os.path.join(args.images, f))
        F.append(signatures(im))
        F_size.append(im.size)
    F = np.stack(F)                                         # (nF, 2, D)

    h = h5py.File(args.mat, "r")
    n_seg = len(h["/value/img"])
    G, G_size = [], []
    for i in range(n_seg):
        im, _ = read_seg(h, i)
        G.append(signatures(im))
        G_size.append(im.size)
        if (i + 1) % 500 == 0:
            print(f"  signatures {i + 1}/{n_seg}")
    G = np.stack(G)                                         # (nG, 2, D)

    # best of whole-vs-whole and centre-vs-centre correlation
    score = np.maximum(F[:, 0] @ G[:, 0].T, F[:, 1] @ G[:, 1].T)   # (nF, nG)
    order = np.argsort(-score, axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    if args.out_masks:
        os.makedirs(args.out_masks, exist_ok=True)
    n_match = 0
    with open(args.out_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "seg_index", "score", "runner_up", "match", "same_size",
                     "file_size", "seg_size"])
        for k, f in enumerate(files):
            j, j2 = int(order[k, 0]), int(order[k, 1])
            s1, s2 = float(score[k, j]), float(score[k, j2])
            match = s1 >= args.thr and s1 - s2 >= args.margin
            same = tuple(F_size[k]) == tuple(G_size[j])
            wr.writerow([f, j, f"{s1:.4f}", f"{s2:.4f}", int(match), int(same),
                         f"{F_size[k][0]}x{F_size[k][1]}", f"{G_size[j][0]}x{G_size[j][1]}"])
            if match:
                n_match += 1
                if args.out_masks:
                    _, gt = read_seg(h, j)
                    m = Image.fromarray(((gt > 0) * 255).astype(np.uint8))
                    m = m.resize(F_size[k], Image.NEAREST)     # to the folder image's size
                    m.save(os.path.join(args.out_masks, os.path.splitext(f)[0] + ".png"))
            print(f"{f}: best seg{j:05d} score={s1:.3f} (2nd {s2:.3f}) "
                  f"{'MATCH' if match else '-'}{' same-size' if same else ''}")
    print(f"\n{n_match}/{len(files)} folder images matched → {args.out_csv}")


if __name__ == "__main__":
    main()
