"""
Where do symmetric path methods disagree?  (replaces the "holonomy map", E1)

Measured at several resolutions (element, pixel, grid cells): single SPI paths are
noise-dominated per element, so a between-α gap is only meaningful where its
debiased signal  ‖μ1 − μ2‖² ≈ ‖m1 − m2‖² − se1² − se2²  clears the Monte-Carlo noise.

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
    ap.add_argument("--resolutions", nargs="+", default=["element", "pixel", "grid:16", "grid:32"],
                    help="element | pixel | grid:<cell px>; signal/noise is measured at each")
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

        A_by_alpha = {}
        for a_i, alpha in enumerate(args.alphas):
            gen = torch.Generator().manual_seed(args.seed + 1_000_003 * idx + 97 * a_i)
            A_raw = T.spi_image_candidates(oracle, xb, x, gen, alpha, args.K, args.N, args.est, args.Q,
                                           True, 8, D)
            A, _ = complete_stack(A_raw, D, args.completion, 0.0, support_scale(A_raw))
            A_by_alpha[alpha] = A
            if args.save_maps:
                m = A.mean(dim=0)
                pm = MET.attribution_map(m.view(1, C, H, W))
                ps = (A - m[None]).view(args.K, C, H, W).pow(2).sum(dim=1).mean(dim=0).sqrt()
                torch.save({"mean": pm.half(), "spread": ps.half()},
                           os.path.join(args.save_maps, f"{name}__a{alpha:g}.pt"))

        row = {"image": name, "index": idx, "class": cls, "conf": conf, "D": D,
               "first_step_share": first_share, "t90": t90, "K": args.K, "N": args.N,
               "est": args.est, "res": {}, "method": "alpha_disagreement", "seed": args.seed, "error": None}
        for res in args.resolutions:
            st = {a: stats(aggregate(A_by_alpha[a], res, C, H, W)) for a in args.alphas}
            pairs = {}
            for a1, a2 in itertools.combinations(args.alphas, 2):
                m1, m2 = st[a1]["mean"], st[a2]["mean"]
                gap2 = float(((m1 - m2) ** 2).sum())
                noise2 = st[a1]["se2"] + st[a2]["se2"]
                scale = 0.5 * (float(m1.norm()) + float(m2.norm()))
                pairs[f"{a1:g}-{a2:g}"] = {
                    "gap_rel": gap2 ** 0.5 / max(scale, 1e-30),
                    "snr": (gap2 / max(noise2, 1e-300)) ** 0.5,
                    # E‖m1 − m2‖² = ‖μ1 − μ2‖² + se1² + se2²  →  debiased signal
                    "signal_rel": max(gap2 - noise2, 0.0) ** 0.5 / max(scale, 1e-30),
                    "spearman": spearman(m1, m2)}
            row["res"][res] = {"within_rel": {f"{a:g}": st[a]["within_rel"] for a in args.alphas},
                               "noise_rel": {f"{a:g}": st[a]["se2"] ** 0.5 / max(float(st[a]["mean"].norm()), 1e-30)
                                             for a in args.alphas},
                               "pairs": pairs,
                               "signal_max": max(p["signal_rel"] for p in pairs.values()) if pairs else 0.0,
                               "within_mean": sum(st[a]["within_rel"] for a in args.alphas) / len(args.alphas)}
        append_row(args.out, row)
        best = args.resolutions[-1]
        print(f"[{j + 1}/{n}] {name}: first_share={first_share:.3f} "
              f"signal_max[{best}]={row['res'][best]['signal_max']:.3f}")

    summarize(args.out, args.resolutions)


def aggregate(A: torch.Tensor, res: str, C: int, H: int, W: int) -> torch.Tensor:
    """(K, C·H·W) → (K, R): element | pixel (signed channel sum) | grid:<c> (sum over c×c cells)."""
    K = A.shape[0]
    if res == "element":
        return A
    pix = A.view(K, C, H, W).sum(dim=1)
    if res == "pixel":
        return pix.reshape(K, -1)
    if res.startswith("grid:"):
        c = int(res.split(":")[1])
        return (torch.nn.functional.avg_pool2d(pix[:, None], c, c) * (c * c)).reshape(K, -1)
    raise ValueError(f"unknown resolution '{res}'")


def stats(V: torch.Tensor) -> dict:
    """mean, se² = E‖mean − μ‖² estimate, within-α relative spread of single paths."""
    K = V.shape[0]
    m = V.mean(dim=0)
    dev2 = ((V - m[None]) ** 2).sum(dim=1)
    se2 = float(dev2.sum()) / (K * (K - 1)) if K > 1 else float("nan")
    within = float(dev2.mean().sqrt()) / max(float(m.norm()), 1e-30)
    return {"mean": m, "se2": se2, "within_rel": within}


def summarize(path: str, resolutions) -> None:
    """Per resolution: is the between-α signal above Monte-Carlo noise, and does it follow saturation?"""
    import json
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "res" in r]
    if len(rows) < 3:
        return
    fs = torch.tensor([r["first_step_share"] for r in rows], dtype=torch.float64)
    for res in resolutions:
        rs = [r for r in rows if res in r["res"]]
        if len(rs) < 3:
            continue
        fsr = torch.tensor([r["first_step_share"] for r in rs], dtype=torch.float64)
        print(f"\n== resolution {res} ({len(rs)} images) ==")
        print(f"{'pair':10s} {'med snr':>8s} {'%snr>3':>7s} {'med signal_rel':>15s} {'rho(fs,signal)':>15s}")
        for p in rs[0]["res"][res]["pairs"]:
            snr = torch.tensor([r["res"][res]["pairs"][p]["snr"] for r in rs], dtype=torch.float64)
            sig = torch.tensor([r["res"][res]["pairs"][p]["signal_rel"] for r in rs], dtype=torch.float64)
            print(f"{p:10s} {float(snr.median()):8.2f} {100 * float((snr > 3).double().mean()):6.0f}% "
                  f"{float(sig.median()):15.3f} {spearman(fsr, sig):+15.2f}")
        wm = torch.tensor([r["res"][res]["within_mean"] for r in rs], dtype=torch.float64)
        sm = torch.tensor([r["res"][res]["signal_max"] for r in rs], dtype=torch.float64)
        print(f"rho(first_share, max debiased between-α signal) = {spearman(fsr, sm):+.3f}  "
              f"(E1 needs snr > 3 on most images AND rho ≥ 0.2)")
        print(f"rho(first_share, within-α spread)               = {spearman(fsr, wm):+.3f}")


if __name__ == "__main__":
    main()