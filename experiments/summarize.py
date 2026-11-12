"""
Paired summary of a runner JSONL.

For each method: mean ± std over images (seeds averaged per image first),
seed spread, mean backward passes, and the paired difference to a reference
method on the images both have: mean Δ and median Δ with 95% bootstrap CIs over
images, the win rate (share of images where the method beats the reference;
ties count ½; for metrics where lower is better pass --lower), and the Wilcoxon
signed-rank p-value (if scipy is installed). Heavy-tailed metrics (insdel_blur)
should be read on the median Δ / win rate, not on the mean.

python -m experiments.summarize runs/p2.jsonl --ref "ig" \
    --metrics insdel_black insdel_blur overlap iou pointing checker
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np


def load(path: str):
    rows, errors = [], defaultdict(int)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error"):
                errors[r["method"]] += 1
            else:
                rows.append(r)
    return rows, errors


def per_image(rows, metric):
    """method -> {image: (mean over seeds, std over seeds, n seeds)}"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(metric)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        acc[r["method"]][r["image"]].append(float(v))
    return {m: {im: (float(np.mean(v)), float(np.std(v)), len(v)) for im, v in d.items()}
            for m, d in acc.items()}


def boot_ci(d: np.ndarray, n_boot: int = 5000, seed: int = 0, stat=np.mean):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    vals = stat(d[idx], axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilcoxon_p(d: np.ndarray):
    try:
        from scipy.stats import wilcoxon
        if np.allclose(d, 0):
            return 1.0
        return float(wilcoxon(d).pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--ref", default=None, help="reference method spec for paired Δ")
    ap.add_argument("--metrics", nargs="+",
                    default=["insdel_black", "insdelp_black", "insdelp_blur", "overlap", "iou",
                             "pointing", "checker"])
    ap.add_argument("--lower", nargs="*", default=["checker", "infd", "infd_raw", "road_morf",
                                                   "hold_rel", "tr_rel", "fit_rel", "suff", "ref_l1"],
                    help="metrics where lower is better (only affects the win rate)")
    args = ap.parse_args()

    rows, errors = load(args.jsonl)
    methods = sorted({r["method"] for r in rows})
    bwd = defaultdict(list)
    for r in rows:
        if r.get("n_bwd") is not None:
            bwd[r["method"]].append(r["n_bwd"])

    for metric in args.metrics:
        pim = per_image(rows, metric)
        if not pim:
            continue
        print(f"\n### {metric}\n")
        hdr = "| method | n img | mean ± std | median | seed std | backward |"
        if args.ref:
            hdr += f" Δ vs {args.ref} | 95% CI | Δ median | 95% CI | win % | Wilcoxon p |"
        print(hdr)
        print("|" + "---|" * (hdr.count("|") - 1))
        ref = pim.get(args.ref, {}) if args.ref else {}
        for m in methods:
            if m not in pim:
                continue
            vals = np.array([v[0] for v in pim[m].values()])
            sstd = np.array([v[1] for v in pim[m].values() if v[2] > 1])
            sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
            seed_col = f"{sstd.mean():.4f}" if len(sstd) else "–"
            bwd_col = f"{np.mean(bwd[m]):.0f}" if bwd[m] else "–"
            line = (f"| {m} | {len(vals)} | {vals.mean():.4f} ± {sd:.4f} | {np.median(vals):.4f} "
                    f"| {seed_col} | {bwd_col} |")
            if args.ref:
                common = sorted(set(pim[m]) & set(ref))
                if m != args.ref and len(common) >= 2:
                    d = np.array([pim[m][i][0] - ref[i][0] for i in common])
                    lo, hi = boot_ci(d)
                    mlo, mhi = boot_ci(d, stat=np.median)
                    sgn = -1.0 if metric in args.lower else 1.0
                    win = 100.0 * (np.mean(sgn * d > 0) + 0.5 * np.mean(d == 0))
                    line += (f" {d.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | {np.median(d):+.4f} "
                             f"| [{mlo:+.4f}, {mhi:+.4f}] | {win:.0f} | {wilcoxon_p(d):.2g} |")
                else:
                    line += " – | – | – | – | – | – |"
            print(line)

    if errors:
        print("\nErrors (rows excluded):", dict(errors))


if __name__ == "__main__":
    main()