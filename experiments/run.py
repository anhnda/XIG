"""
Paired benchmark runner: every method on the same images, one JSONL row per
(image, method, seed). Re-running the same command resumes (finished rows are
skipped). Failures are written as rows with "error" instead of being dropped.

Examples
--------
# ImageNet-Seg, 200 fixed images, black path baseline, ins/del with black and blur
python -m experiments.run --data seg --seg-mat /path/gtsegs_ijcv.mat --n 200 \
    --index-file runs/seg200.json --baseline black --insdel black blur \
    --methods ig guided_ig idgi dnoise "spi:alpha=1,K=8,est=mid" \
              "tpa:alpha=1,mode=core,eps=0.1" "kernelshap:M=256" \
    --infd-n 64 --road --seeds 0 1 2 --out runs/p2.jsonl

# folder of images (masks optional)
python -m experiments.run --data folder --images ./sample_imagenet1k --n 50 \
    --index-file runs/folder50.json --methods ig dnoise --out runs/quick.jsonl
"""

from __future__ import annotations

import argparse
import os

import torch

from experiments.common import (TargetLogit, TargetProb, append_row, evaluate, input_size,
                                load_backbone, load_done, resolve, run_one)
from pathattr import Oracle
from pathattr import data as DATA
from pathattr import metrics as MET


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["seg", "folder"], required=True)
    ap.add_argument("--seg-mat", default=None, help="path to gtsegs_ijcv.mat")
    ap.add_argument("--images", default=None, help="image folder (--data folder)")
    ap.add_argument("--masks", default=None, help="mask folder, same file stems (optional)")
    ap.add_argument("--prep", default="resize", choices=["resize", "crop"],
                    help="folder images: resize to 224 (default) or Resize256+CenterCrop224 (legacy)")
    ap.add_argument("--n", type=int, default=200, help="number of images (0 = all)")
    ap.add_argument("--start", type=int, default=0, help="first position in the index list")
    ap.add_argument("--index-file", default=None, help="JSON list of image indices (created if missing)")
    ap.add_argument("--index-seed", type=int, default=0)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--baseline", default="black", choices=["black", "gray", "white", "blur"],
                    help="path baseline x' used by the attribution methods")
    ap.add_argument("--insdel", nargs="+", default=["black"],
                    choices=["black", "gray", "white", "blur"],
                    help="baselines for insertion/deletion (one score set each)")
    ap.add_argument("--insdel-steps", type=int, default=100)
    ap.add_argument("--no-prob", action="store_true",
                    help="skip the softmax-probability ins/del (insdelp_*) (ROAD needs it)")
    ap.add_argument("--infd-n", type=int, default=0, help="infidelity samples (0 = off)")
    ap.add_argument("--infd-sigma", type=float, default=0.1)
    ap.add_argument("--road", action="store_true", help="approximate ROAD (MoRF/LeRF)")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="seeds for stochastic methods (deterministic ones use the first)")
    ap.add_argument("--chunk", type=int, default=32, help="batch size for gradient calls")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-maps", default=None, help="folder to save (H, W) attribution maps (fp16)")
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone = load_backbone(args.model, dev)

    if args.data == "seg":
        if not args.seg_mat:
            ap.error("--seg-mat is required for --data seg")
        ds = DATA.ImageNetSeg(args.seg_mat, size=input_size(args.model))
    else:
        if not args.images:
            ap.error("--images is required for --data folder")
        ds = DATA.FolderImages(args.images, args.masks, size=input_size(args.model), prep=args.prep)
    indices = DATA.select_indices(len(ds), args.n, args.index_seed, args.index_file,
                                  start=args.start)
    if args.road and args.no_prob:
        ap.error("--road needs the probability model (drop --no-prob)")
    print(f"[run] {len(indices)} images, {len(args.methods)} methods, device={dev}")

    specs = [(s, *resolve(s)) for s in args.methods]
    done = load_done(args.out)
    if args.save_maps:
        os.makedirs(args.save_maps, exist_ok=True)

    for n_img, (idx, x, mask, name) in enumerate(DATA.iterate(ds, indices)):
        x = x.to(dev)
        with torch.no_grad():
            prob = torch.softmax(backbone(x), dim=-1)[0]
        cls = int(prob.argmax())
        model = TargetLogit(backbone, cls)
        prob_model = None if args.no_prob else TargetProb(backbone, cls)
        oracle = Oracle(model, chunk=args.chunk)
        xb = MET.make_baseline(x, args.baseline)
        bases = {b: MET.make_baseline(x, b) for b in args.insdel}
        with torch.no_grad():
            fx, fb = float(model(x)[0]), float(model(xb)[0])
        D = fx - fb

        for spec, fn, params, stoch in specs:
            for seed in (args.seeds if stoch else args.seeds[:1]):
                if (name, spec, seed) in done:
                    continue
                attr, meta = run_one(fn, params, oracle, xb, x, seed + 1_000_003 * idx)
                row = {"image": name, "index": idx, "class": cls, "conf": float(prob[cls]),
                       "model": args.model, "baseline": args.baseline, "method": spec,
                       "seed": seed, "f_x": fx, "f_base": fb, "D": D, **meta}
                if attr is not None:
                    s = float(attr.sum())
                    row.update({"attr_sum": s, "residual": D - s})
                    egen = torch.Generator().manual_seed(7_919 + idx)   # same perturbations for every method
                    row.update(evaluate(model, x, mask, attr, bases, args.insdel_steps, prob_model,
                                        infd_n=args.infd_n, infd_sigma=args.infd_sigma,
                                        road=args.road, gen=egen))
                    if args.save_maps:
                        fn_map = os.path.join(args.save_maps,
                                              f"{name}__{spec.replace(':', '_').replace(',', '_')}__s{seed}.pt")
                        torch.save(MET.attribution_map(attr).half().cpu(), fn_map)
                append_row(args.out, row)
                tag = "ERR " + row["error"] if row["error"] else \
                    f"insdel={row.get('insdel_' + args.insdel[0], float('nan')):.3f} " \
                    f"iou={row.get('iou', float('nan')):.3f} bwd={row['n_bwd']} t={row['time']:.1f}s"
                print(f"  [{n_img + 1}/{len(indices)}] {name} {spec} s{seed}: {tag}")


if __name__ == "__main__":
    main()
