"""
Deterministic sweep (no Monte Carlo): does the large-α SPI correction point towards Shapley?

For many random toys F (family × d × seed):
    IG = exact,  Shapley = exact (2^d vertices),  C1 = (α+1)·C  (α-free first-order SPI correction)
    cos      = cos(C1, Shapley − IG)        > 0: SPI(α large) moves towards Shapley, < 0: away
    ratio    = |C1| / |Shapley − IG|        size of the correction against the IG–Shapley gap
    sign     = share of features with sign(C1_i) = sign(Shapley_i − IG_i)
    gap      = |Shapley − IG| / |Shapley|   how far apart IG and Shapley are at all
Toys whose IG–Shapley gap or correction is ~0 (relative 1e-8) are counted but left out of cos/sign.

python -m experiments.sweep_spi_theory --families mlp mlpn saturating satint --ds 5 10 15 --seeds 200
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from pathattr import theory as TH


def q(v: list, p: float) -> float:
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["mlp", "mlpn", "saturating", "satint"],
                    choices=sorted(TH.TOYS))
    ap.add_argument("--ds", nargs="+", type=int, default=[5, 10, 15])
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--n-t", type=int, default=32, help="Gauss–Legendre nodes in t for IG and C")
    ap.add_argument("--out", default=None, help="optional JSONL, one row per toy")
    args = ap.parse_args()
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").close()

    print(f"{'family':>11} {'d':>3} {'n':>4} {'skip':>4} | {'cos q25':>8} {'median':>8} {'q75':>8} {'cos<0 %':>8}"
          f" | {'sign %':>7} | {'ratio':>7} | {'gap':>7}")
    for fam in args.families:
        for d in args.ds:
            cs, rs, sg, gp, skip = [], [], [], [], 0
            for s in range(args.seeds):
                F = TH.TOYS[fam](d, s)
                ig = TH.ig_exact(F, d, args.n_t)
                sh = TH.shapley_exact(F, d)
                C1 = TH.spi_correction(F, d, alpha=0.0, n_t=args.n_t)
                diff = sh - ig
                nd, nc, ns = float(diff.norm()), float(C1.norm()), float(sh.norm())
                row = {"family": fam, "d": d, "seed": s, "gap_norm": nd, "c1_norm": nc, "sh_norm": ns}
                if nd <= 1e-8 * max(ns, 1e-30) or nc <= 1e-8 * max(float(ig.norm()), 1e-30):
                    skip += 1
                else:
                    c = float(C1 @ diff) / (nd * nc)
                    agree = float(((torch.sign(C1) == torch.sign(diff)) & (diff.abs() > 1e-12)).double().mean())
                    cs.append(c)
                    rs.append(nc / nd)
                    sg.append(agree)
                    row.update({"cos": c, "ratio": nc / nd, "sign": agree})
                gp.append(nd / max(ns, 1e-30))
                if args.out:
                    with open(args.out, "a") as fh:
                        fh.write(json.dumps(row) + "\n")
            neg = 100.0 * sum(c < 0 for c in cs) / len(cs) if cs else float("nan")
            print(f"{fam:>11} {d:>3} {len(cs):>4} {skip:>4} | {q(cs, .25):+8.3f} {q(cs, .5):+8.3f} {q(cs, .75):+8.3f}"
                  f" {neg:8.1f} | {100 * q(sg, .5):7.1f} | {q(rs, .5):7.3f} | {q(gp, .5):7.3f}", flush=True)

    print("\nRead: median cos < 0 with cos<0 % well above 50 ⇒ the first-order SPI correction points away from"
          "\n      Shapley in general (SPI(α) is not an IG→Shapley interpolation); ≈ 0 ⇒ unrelated; > 0 ⇒ towards.")


if __name__ == "__main__":
    main()
