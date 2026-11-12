"""
Choose TPA hyper-parameters on the calibration slice by HELD-OUT mask fidelity
(hold_rel, same mask family as the fit, fresh masks). Never by ins/del, never by
the transfer family (tr_rel), which stays an independent test.

One winner per family:
    uniform            → α                        (SPI baseline, tuned fairly)
    gibbs              → (α, ε)
    core               → (α, ε)
    r, for each κ      → (α, ε)                   (κ is NOT selected: fixed rule)
    kernelshap         → kept as is
Writes the chosen variant names (JSON list) for `tpa_sweep --variants-file`.

python -m experiments.select_tpa runs/tpa_calib.jsonl --out runs/tpa_chosen.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from experiments.summarize import load, per_image


def family_of(row: dict) -> str:
    mode = row.get("mode")
    if mode == "r":
        return f"r|k{row.get('kappa'):g}"
    return str(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--criterion", default="hold_rel", choices=["hold_rel"],
                    help="selection metric (lower is better); only held-out fidelity is allowed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, errors = load(args.jsonl)
    fam = {}
    for r in rows:
        fam[r["method"]] = family_of(r)
    pim = per_image(rows, args.criterion)                # method -> {image: (mean, std, n)}
    images = None
    for m, d in pim.items():
        images = set(d) if images is None else images & set(d)
    if not images:
        raise SystemExit("no image has every variant; check the JSONL")

    by_family = defaultdict(list)
    for m, d in pim.items():
        vals = np.array([d[i][0] for i in sorted(images)])
        by_family[fam[m]].append((float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0), m))

    chosen = []
    print(f"criterion: {args.criterion} (lower is better), {len(images)} calibration images\n")
    for f in sorted(by_family):
        cands = sorted(by_family[f])
        best = cands[0]
        chosen.append(best[2])
        print(f"## {f}: chosen {best[2]}  ({best[0]:.4f} ± {best[1]:.4f})")
        for mean, sd, m in cands[:6]:
            print(f"     {m:40s} {mean:.4f} ± {sd:.4f}")
    if errors:
        print("\nErrors:", dict(errors))
    with open(args.out, "w") as fh:
        json.dump(chosen, fh, indent=1)
    print(f"\nwrote {args.out}: {chosen}")


if __name__ == "__main__":
    main()
