"""
Numerical check of the SPI small-noise expansion  SPI_i = IG_i + C_i + O(1/α²)  (CPU, toys).

For each toy F and α:
    Δ  = SPI_MC − IG_exact      (SPI by Monte Carlo over DP paths, exact line integrals)
    C  = predicted correction   (pathattr.theory.spi_correction)
Reported:
    |Δ|, |C|            norms
    cos(Δ, C)           direction agreement (→ 1 if the formula is right)
    rel = |Δ − C|/|C|   should shrink like O(1/α) until the Monte-Carlo floor
    snr = |C| / |se|    Monte-Carlo resolution (rel is meaningful only if snr ≫ 1)
    (α+1)·Δ vs (α+1)·C  should both converge to the same α-free vector
Controls: multilinear F → C = 0 exactly and Δ must be pure noise (|Δ| ≈ |se|).
Also reported at small α: SPI vs the exact Shapley value of the toy game z ∈ {0,1}^d.

python -m experiments.verify_spi_expansion --alphas 3 10 30 100 300 --n 20000
"""

from __future__ import annotations

import argparse
import itertools
import math

import torch

from pathattr import theory as TH


def shapley_exact(F, d: int) -> torch.Tensor:
    """Shapley of v(S) = F(1_S) by enumerating all orders (d ≤ 7)."""
    phi = torch.zeros(d, dtype=TH.DT)
    perms = list(itertools.permutations(range(d)))
    for p in perms:
        z = torch.zeros(d, dtype=TH.DT)
        prev = float(F(z))
        for i in p:
            z[i] = 1.0
            cur = float(F(z))
            phi[i] += cur - prev
            prev = cur
    return phi / len(perms)


def fmt(v: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(a):+.4f}" for a in v) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphas", nargs="+", type=float, default=[3, 10, 30, 100, 300])
    ap.add_argument("--small-alphas", nargs="*", type=float, default=[0.1, 1.0])
    ap.add_argument("--n", type=int, default=20000, help="paths per α (antithetic pairs)")
    ap.add_argument("--N", type=int, default=128, help="polygon bins per path")
    ap.add_argument("--Q", type=int, default=8, help="Gauss–Legendre nodes per segment")
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    toys = {"mlp": TH.toy_mlp(args.d, seed=args.seed),
            "saturating": TH.toy_saturating(args.d, seed=args.seed),
            "multilinear": TH.toy_multilinear(args.d, seed=args.seed)}

    for name, F in toys.items():
        ig = TH.ig_exact(F, args.d)
        sh = shapley_exact(F, args.d) if args.d <= 7 else None
        D = float(F(torch.ones(args.d, dtype=TH.DT)) - F(torch.zeros(args.d, dtype=TH.DT)))
        print(f"\n=== toy {name}  (d={args.d}, D={D:+.4f})")
        print(f"IG      {fmt(ig)}")
        if sh is not None:
            print(f"Shapley {fmt(sh)}   |IG − Shapley| = {float((ig - sh).norm()):.4f}")
        print(f"{'alpha':>7} {'|Δ|':>9} {'|C|':>9} {'cos':>7} {'rel':>7} {'snr':>7}   (α+1)Δ  /  (α+1)C")
        for k, a in enumerate(args.alphas):
            gen = torch.Generator().manual_seed(1000 * args.seed + k)
            mc = TH.spi_monte_carlo(F, args.d, a, args.n, args.N, args.Q, gen)
            C = TH.spi_correction(F, args.d, a)
            dlt = mc["mean"] - ig
            nC, nD, nse = float(C.norm()), float(dlt.norm()), float(mc["se"].norm())
            cos = float(dlt @ C) / (nD * nC) if nD > 0 and nC > 0 else float("nan")
            rel = float((dlt - C).norm()) / nC if nC > 0 else float("nan")
            snr = nC / nse if nse > 0 else float("nan")
            print(f"{a:7g} {nD:9.2e} {nC:9.2e} {cos:7.3f} {rel:7.3f} {snr:7.1f}   "
                  f"{fmt((a + 1) * dlt)}  /  {fmt((a + 1) * C)}   (|se|={nse:.1e}, resid={mc['residual']:+.1e})")
        for k, a in enumerate(args.small_alphas):
            gen = torch.Generator().manual_seed(5000 + 1000 * args.seed + k)
            mc = TH.spi_monte_carlo(F, args.d, a, args.n, args.N, args.Q, gen)
            extra = f"  |SPI − Shapley| = {float((mc['mean'] - sh).norm()):.4f}" if sh is not None else ""
            print(f"  small α={a:g}: SPI {fmt(mc['mean'])}  |SPI − IG| = {float((mc['mean'] - ig).norm()):.4f}{extra}"
                  f"  (|se|={float(mc['se'].norm()):.1e})")

    print("\nRead: for mlp / saturating, cos → 1 and rel shrinks with α (≈ ∝ 1/α) until snr limits it;"
          "\n      (α+1)Δ ≈ (α+1)C at large α. For multilinear, |C| = 0 and |Δ| ≈ |se|."
          "\n      At small α, SPI should approach Shapley (the expansion no longer applies).")


if __name__ == "__main__":
    main()
