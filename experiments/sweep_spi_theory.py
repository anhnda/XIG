"""
Deterministic sweep (no Monte Carlo): does the large-α SPI correction point towards Shapley?

For many random toys F (family × d × seed), all seeds of one (family, d) in one vmapped batch:
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
import time

import torch

from pathattr import theory as TH


def q(v: torch.Tensor, p: float) -> float:
    return float(torch.quantile(v, p)) if v.numel() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["mlp", "mlpn", "saturating", "satint"],
                    choices=sorted(TH.FAMILIES))
    ap.add_argument("--ds", nargs="+", type=int, default=[5, 10, 15])
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--n-t", type=int, default=32, help="Gauss–Legendre nodes in t for IG and C")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--out", default=None, help="optional JSONL, one row per toy")
    args = ap.parse_args()
    print(f"[sweep] device={TH.set_device(args.device)}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").close()

    print(f"{'family':>11} {'d':>3} {'n':>4} {'skip':>4} | {'cos q25':>8} {'median':>8} {'q75':>8} {'cos<0 %':>8}"
          f" | {'sign %':>7} | {'ratio':>7} | {'gap':>7} | {'sec':>5}")
    for fam in args.families:
        for d in args.ds:
            t0 = time.time()
            seeds = list(range(args.seeds))
            fn, P = TH.stack_params(fam, d, seeds)
            ig = TH.ig_batch(fn, P, d, args.n_t)
            sh = TH.shapley_batch(fn, P, d)
            C1 = TH.correction_batch(fn, P, d, 0.0, args.n_t)
            diff = sh - ig
            nd, nc, ns, ni = diff.norm(dim=1), C1.norm(dim=1), sh.norm(dim=1), ig.norm(dim=1)
            ok = (nd > 1e-8 * ns.clamp_min(1e-30)) & (nc > 1e-8 * ni.clamp_min(1e-30))
            cosv = (C1 * diff).sum(dim=1) / (nd * nc).clamp_min(1e-300)
            sign = ((torch.sign(C1) == torch.sign(diff)) & (diff.abs() > 1e-12)).double().mean(dim=1)
            ratio = nc / nd.clamp_min(1e-300)
            gap = nd / ns.clamp_min(1e-30)
            c, s, r = cosv[ok].cpu(), sign[ok].cpu(), ratio[ok].cpu()
            if args.out:
                with open(args.out, "a") as fh:
                    for k, sd in enumerate(seeds):
                        row = {"family": fam, "d": d, "seed": sd, "gap_norm": float(nd[k]),
                               "c1_norm": float(nc[k]), "sh_norm": float(ns[k]), "gap": float(gap[k])}
                        if bool(ok[k]):
                            row.update({"cos": float(cosv[k]), "ratio": float(ratio[k]), "sign": float(sign[k])})
                        fh.write(json.dumps(row) + "\n")
            neg = 100.0 * float((c < 0).double().mean()) if c.numel() else float("nan")
            print(f"{fam:>11} {d:>3} {c.numel():>4} {len(seeds) - c.numel():>4} | {q(c, .25):+8.3f} {q(c, .5):+8.3f}"
                  f" {q(c, .75):+8.3f} {neg:8.1f} | {100 * q(s, .5):7.1f} | {q(r, .5):7.3f} | {q(gap.cpu(), .5):7.3f}"
                  f" | {time.time() - t0:5.1f}", flush=True)

    print("\nRead: median cos < 0 with cos<0 % well above 50 ⇒ the first-order SPI correction points away from"
          "\n      Shapley in general (SPI(α) is not an IG→Shapley interpolation); ≈ 0 ⇒ unrelated; > 0 ⇒ towards.")


if __name__ == "__main__":
    main()
