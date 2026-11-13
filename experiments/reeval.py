"""
Recompute metrics from saved attributions (experiments.run --save-attr DIR),
without re-running the attribution methods.

python -m experiments.reeval --data seg --seg-mat M.mat --index-file runs/seg200.json --n 200 \
    --attr-dir runs/attr_p2 --insdel black blur --infd-n 64 --infd-mode delta --road \
    --out runs/p2_reeval.jsonl
Rows have the same keys as the runner (image, method, seed, metrics), so
experiments.summarize / metric_agreement work on the output.
"""

from __future__ import annotations

import argparse
import glob
import os

import torch

from experiments.common import (TargetProb, add_data_args, append_row, evaluate, load_done,
                                open_data, target_model)
from pathattr import metrics as MET


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--attr-dir", required=True)
    ap.add_argument("--methods", nargs="*", default=None, help="only these method specs")
    ap.add_argument("--insdel", nargs="*", default=["black"], choices=["black", "gray", "white", "blur"])
    ap.add_argument("--insdel-steps", type=int, default=100)
    ap.add_argument("--no-prob", action="store_true")
    ap.add_argument("--infd-n", type=int, default=0)
    ap.add_argument("--infd-sigma", type=float, default=0.1)
    ap.add_argument("--infd-mode", default="delta", choices=["delta", "abs"])
    ap.add_argument("--infd-cell", type=int, default=16, help="patch size for infidelity noise (0 = per element)")
    ap.add_argument("--road", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.road and args.no_prob:
        ap.error("--road needs the probability model")

    done = load_done(args.out)
    dev, backbone, it, n = open_data(args)
    for j, (idx, x, mask, name) in enumerate(it):
        files = sorted(glob.glob(os.path.join(args.attr_dir, f"{name}__*.pt")))
        if not files:
            continue
        x = x.to(dev)
        model, cls, conf = target_model(backbone, x)
        prob_model = None if args.no_prob else TargetProb(backbone, cls)
        bases = {b: MET.make_baseline(x, b) for b in args.insdel}
        for f in files:
            rec = torch.load(f)
            spec, seed = rec["method"], int(rec["seed"])
            if args.methods and spec not in args.methods:
                continue
            if (name, spec, seed) in done:
                continue
            attr = rec["attr"].to(device=dev, dtype=x.dtype)
            xb = MET.make_baseline(x, rec.get("baseline", args.baseline))
            row = {"image": name, "index": idx, "class": cls, "conf": conf, "method": spec, "seed": seed,
                   "baseline": rec.get("baseline"), "error": None}
            try:
                egen = torch.Generator().manual_seed(7_919 + idx)
                row.update(evaluate(model, x, mask, attr, bases, args.insdel_steps, prob_model,
                                    infd_n=args.infd_n, infd_sigma=args.infd_sigma, road=args.road,
                                    gen=egen, infd_base=xb, infd_mode=args.infd_mode,
                                    infd_cell=args.infd_cell))
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
            append_row(args.out, row)
        print(f"[{j + 1}/{n}] {name}: {len(files)} attributions")


if __name__ == "__main__":
    main()
