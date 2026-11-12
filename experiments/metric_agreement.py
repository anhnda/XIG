"""
Do metric families rank methods the same way?  (E5 — "the metric is a kernel")

For every metric: mean over images (seeds averaged first, only images that every
method has), oriented so that higher = better, then Kendall τ-b between every
pair of metric rankings. A low τ between the binary-mask family (ins/del, ROAD,
hold_rel) and the Gaussian family (infd) supports metric–kernel alignment.

python -m experiments.metric_agreement runs/p2.jsonl \
    --metrics insdel_black insdelp_black iou overlap pointing infd road
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from experiments.summarize import load, per_image

LOWER_IS_BETTER = {"infd", "infd_raw", "road_morf", "checker", "hold_rel", "tr_rel", "fit_rel",
                   "del_black", "del_gray", "del_blur", "delp_black", "delp_gray", "delp_blur"}


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    den = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / den) if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--methods", nargs="*", default=None, help="restrict to these methods")
    args = ap.parse_args()

    rows, _ = load(args.jsonl)
    scores = {}
    common_methods = None
    for metric in args.metrics:
        pim = per_image(rows, metric)
        if args.methods:
            pim = {m: d for m, d in pim.items() if m in args.methods}
        if not pim:
            print(f"(no data for {metric})")
            continue
        imgs = set.intersection(*[set(d) for d in pim.values()])
        sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
        scores[metric] = {m: sign * float(np.mean([d[i][0] for i in imgs])) for m, d in pim.items()}
        ms = set(scores[metric])
        common_methods = ms if common_methods is None else common_methods & ms
    methods = sorted(common_methods or [])
    if len(methods) < 3:
        raise SystemExit("need ≥ 3 methods shared by all metrics")

    print(f"{len(methods)} methods: {methods}\n")
    print("ranking per metric (best first):")
    for metric, sc in scores.items():
        order = sorted(methods, key=lambda m: -sc[m])
        print(f"  {metric:16s} " + " > ".join(order))
    print("\nKendall τ-b between metric rankings:")
    names = list(scores)
    for a, b in itertools.combinations(names, 2):
        xa = np.array([scores[a][m] for m in methods])
        xb = np.array([scores[b][m] for m in methods])
        print(f"  {a:16s} vs {b:16s}  τ = {kendall_tau_b(xa, xb):+.3f}")


if __name__ == "__main__":
    main()
