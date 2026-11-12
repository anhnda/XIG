"""
Where do symmetric path methods disagree?  (replaces the "holonomy map", E1)

Two different quantities, kept apart:
  within-α spread   per-path variability under one prior P_α. It contains the
                    multilinear interactions too (F = z1 z2: single paths give
                    (1,0) or (0,1) although every symmetric method gives (½,½)),
                    so it is NOT ambiguity between methods (SPI-P already maps it).
  between-α gap     difference between the α-means: what changes when the
                    symmetric prior changes — only the non-multilinear part of F
                    (theorem T2). Reported with its Monte-Carlo noise level, so
                    a gap is only real when snr = gap / se ≫ 1.
Saturation of the straight line (IG) is logged per image: first-step share and
t90, to test "disagreement is large where the model saturates".

python -m experiments.alpha_disagreement --data seg --seg-mat M.mat --index-file runs/seg200.json \
    --n 200 --alphas 0.1 1 10 100 --K 16 --out runs/alpha.jsonl --save-maps runs/alpha_maps
"""

from __future__ import annotations

import argparse
import itertools
import os

import torch

from experiments.common import add_data_args, append_row, open_data, target_model
from pathattr import Oracle
from pathattr import metrics as MET
from pathattr import paths
from pathattr import tpa as T
from pathattr.complete import complete_stack, support_scale


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.flatten().argsort().argsort().double()
    rb = b.flatten().argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-30))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--est", default="mid", choices=["left", "mid", "gl"])
    ap.add_argument("--Q", type=int, default=1)
    ap.add_argument("--completion", default="support", choices=["support", "rescale", "none"])
    ap.add_argument("--N-sat", type=int, default=24, help="straight-line steps for the saturation profile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-maps", default=None, help="folder for per-α mean / spread maps (fp16)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.save_maps:
        os.makedirs(args.save_maps, exist_ok=True)
    dev, backbone, it, n = open_data(args)
    for j, (idx, x, mask, name) in enumerate(it):
        x = x.to(dev)
        model, cls, conf = target_model(backbone, x)
        oracle = Oracle(model)
        xb = MET.make_baseline(x, args.baseline)
        D = oracle.endpoint_gap(xb, x)
        _, C, H, W = x.shape

        # saturation profile of the straight line
        f = oracle.f(paths.straight(xb, x, args.N_sat)).double().cpu()
        prog = (f - f[0]) / (D if abs(D) > 1e-12 else 1.0)
        first_share = float(prog[1])
        reach = (prog >= 0.9).nonzero()
        t90 = float(reach[0, 0]) / args.N_sat if reach.numel() else 1.0

        means, se2, spread = {}, {}, {}
        for a_i, alpha in enumerate(args.alphas):
            gen = torch.Generator().manual_seed(args.seed + 1_000_003 * idx + 97 * a_i)
            A_raw = T.spi_image_candidates(oracle, xb, x, gen, alpha, args.K, args.N, args.est, args.Q,
                                           True, 8, D)
            A, _ = complete_stack(A_raw, D, args.completion, 0.0, support_scale(A_raw))
            m = A.mean(dim=0)
            dev2 = ((A - m[None]) ** 2).sum(dim=1)                       # (K,)
            means[alpha] = m
            se2[alpha] = float(dev2.sum()) / (args.K * (args.K - 1)) if args.K > 1 else float("nan")
            spread[alpha] = float(dev2.mean().sqrt()) / float(m.norm().clamp_min(1e-30))
            if args.save_maps:
                pm = MET.attribution_map(m.view(1, C, H, W))
                ps = (A - m[None]).view(args.K, C, H, W).pow(2).sum(dim=1).mean(dim=0).sqrt()
                torch.save({"mean": pm.half(), "spread": ps.half()},
                           os.path.join(args.save_maps, f"{name}__a{alpha:g}.pt"))

        row = {"image": name, "index": idx, "class": cls, "conf": conf, "D": D,
               "first_step_share": first_share, "t90": t90, "K": args.K, "N": args.N,
               "est": args.est, "within_rel": {f"{a:g}": spread[a] for a in args.alphas}, "pairs": {},
               "method": "alpha_disagreement", "seed": args.seed, "error": None}
        gaps = []
        for a1, a2 in itertools.combinations(args.alphas, 2):
            diff = means[a1] - means[a2]
            gap = float(diff.norm())
            se = (se2[a1] + se2[a2]) ** 0.5
            scale = 0.5 * float(means[a1].norm() + means[a2].norm())
            rho = spearman(MET.attribution_map(means[a1].view(1, C, H, W)),
                           MET.attribution_map(means[a2].view(1, C, H, W)))
            row["pairs"][f"{a1:g}-{a2:g}"] = {"gap_rel": gap / max(scale, 1e-30), "snr": gap / max(se, 1e-30),
                                             "spearman_abs": rho}
            gaps.append(gap / max(scale, 1e-30))
        row["gap_rel_max"] = max(gaps) if gaps else 0.0
        row["within_rel_mean"] = sum(spread.values()) / len(spread)
        append_row(args.out, row)
        print(f"[{j + 1}/{n}] {name}: first_share={first_share:.3f} gap_max={row['gap_rel_max']:.3f}")

    # cross-image summary: does disagreement follow saturation?
    import json
    rows = [json.loads(l) for l in open(args.out) if l.strip()]
    if len(rows) >= 3:
        fs = torch.tensor([r["first_step_share"] for r in rows], dtype=torch.float64)
        gm = torch.tensor([r["gap_rel_max"] for r in rows], dtype=torch.float64)
        wm = torch.tensor([r["within_rel_mean"] for r in rows], dtype=torch.float64)
        print(f"\nSpearman(first_step_share, between-α gap) = {spearman(fs, gm):+.3f}  (E1 refuted if < 0.2)")
        print(f"Spearman(first_step_share, within-α spread) = {spearman(fs, wm):+.3f}")


if __name__ == "__main__":
    main()
