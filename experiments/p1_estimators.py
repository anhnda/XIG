"""
P1 — keep the path fixed, change the estimator.

For each image and seed, ONE path is drawn (e.g. the dnoise path) and the same
path is integrated with every estimator:
    left        (dnoise as on the slide: gradient at the start of each step)
    mid
    random      (one random interior node per segment; unbiased)
    random_cv   (K pilot draws + 1, control-variate completion)
    gl          (Gauss–Legendre, Q nodes per segment: near-exact reference)
Logged per estimator: backward passes, completeness residual, relative error
and Spearman correlation of the attribution map vs the GL reference, and the
usual ins/del + localisation metrics.

Decision rule (REVISION.md / doc §4): if gl ≥ left on the metrics → branch A
(estimate the path integral); if left ≫ gl → branch B ("first-order Shapley").

python -m experiments.p1_estimators --data seg --seg-mat gtsegs_ijcv.mat --n 200 \
    --index-file runs/seg200.json --path dnoise --seeds 0 1 2 --out runs/p1.jsonl
"""

from __future__ import annotations

import argparse

import torch

from experiments.common import (add_data_args, append_row, evaluate, open_data,
                                parse_spec, target_model)
from pathattr import Oracle
from pathattr import estimators as EST
from pathattr import metrics as MET
from pathattr import methods as M
from pathattr import paths


def build_path(spec: str, xb, x, gen) -> torch.Tensor:
    name, params = parse_spec(spec)
    if name == "straight":
        return paths.straight(xb, x, int(params.get("N", 50)))
    if name in M.PRESETS:
        base, p = M.PRESETS[name]
        p = dict(p)
        p.update(params)
        name, params = base, p
    if name == "dither":
        return paths.dither(xb, x, N=int(params.get("N", 24)), kind=params.get("thresh", "bayer"),
                            tile=int(params.get("tile", 8)), block=int(params.get("block", 14)),
                            spread=float(params.get("spread", 0.5)),
                            noise_std=float(params.get("noise", 0.0)),
                            per_channel=bool(params.get("per_channel", False)), gen=gen)
    if name == "spi":
        return paths.spi(xb, x, N=int(params.get("N", 24)), alpha=float(params.get("alpha", 10.0)),
                         per_channel=bool(params.get("per_channel", True)), gen=gen)
    raise ValueError(f"unsupported path spec '{spec}'")


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.flatten().argsort().argsort().double()
    rb = b.flatten().argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    ap.add_argument("--path", default="dnoise", help="path spec: dnoise, pixel_random, spi:alpha=10, straight:N=24 …")
    ap.add_argument("--estimators", nargs="+", default=["left", "mid", "random", "random_cv"])
    ap.add_argument("--gl-Q", type=int, default=16)
    ap.add_argument("--K-cv", type=int, default=8)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--insdel", nargs="+", default=["black"], choices=["black", "gray", "white", "blur"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev, backbone, it, n = open_data(args)
    for j, (idx, x, mask, name) in enumerate(it):
        x = x.to(dev)
        model, cls, conf = target_model(backbone, x)
        oracle = Oracle(model)
        xb = MET.make_baseline(x, args.baseline)
        bases = {b: MET.make_baseline(x, b) for b in args.insdel}
        with torch.no_grad():
            D = float(model(x)[0] - model(xb)[0])

        for seed in args.seeds:
            gen = torch.Generator().manual_seed(seed + 1_000_003 * idx)
            pts = build_path(args.path, xb, x, gen)
            N = pts.shape[0] - 1

            oracle.reset()
            ref = EST.estimate(oracle, pts, "gl", args.gl_Q)
            ref_bwd = oracle.counts()[1]
            ref_map = MET.attribution_map(ref)

            for e in ["gl"] + list(args.estimators):
                oracle.reset()
                if e == "gl":
                    a, extra = ref, {}
                    n_bwd = ref_bwd
                else:
                    a, extra = M.integrate_path(oracle, pts, e, 1, args.K_cv, xb, x, gen)
                    n_bwd = oracle.counts()[1]
                amap = MET.attribution_map(a)
                row = {"image": name, "index": idx, "class": cls, "conf": conf,
                       "path": args.path, "N": N, "seed": seed, "estimator": e,
                       "method": f"{args.path}|{e}", "n_bwd": n_bwd, "D": D,
                       "attr_sum": float(a.sum()), "residual": D - float(a.sum()),
                       "rel_err_vs_gl": float((a - ref).norm() / ref.norm().clamp_min(1e-30)),
                       "spearman_vs_gl": spearman(amap, ref_map), "error": None, **extra}
                row.update(evaluate(model, x, mask, a, bases))
                append_row(args.out, row)
            print(f"[{j + 1}/{n}] {name} seed={seed} done")


if __name__ == "__main__":
    main()
