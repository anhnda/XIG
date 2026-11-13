"""
Numerical check of the SPI small-noise expansion  SPI_i = IG_i + C_i + O(1/α²)  (CPU, toys).

For each toy F, dimension d and α:
    Δ  = SPI_MC − IG_exact      (SPI by Monte Carlo over DP paths, exact line integrals)
    C  = predicted correction   (pathattr.theory.spi_correction)
Reported:
    |Δ|, |C|            norms
    cos(Δ, C)           direction agreement (→ 1 if the formula is right)
    rel = |Δ − C|/|C|   should shrink like O(1/α) until the Monte-Carlo floor (≈ 1/snr)
    snr = |C| / |se|    Monte-Carlo resolution (rel is meaningful only if snr ≫ 1)
    (α+1)·Δ vs (α+1)·C  should both converge to the same α-free vector
Per toy also: cos(C, Shapley − IG): > 0 ⇒ large-α SPI moves from IG towards Shapley, < 0 ⇒ away.
Controls: multilinear F → C = 0 exactly and Δ must be pure noise (|Δ| ≈ |se|).
At small α: SPI vs the exact Shapley value of the vertex game.
d-scaling: --toys mlp mlpn --ds 5 20 80. mlpn keeps pre-activation noise O(1) in d
(rel should stay flat); mlp lets it grow like √d (rel should worsen with d).

python -m experiments.verify_spi_expansion --toys mlp saturating satint multilinear --ds 5
"""

from __future__ import annotations

import argparse

import torch

from pathattr import theory as TH


def fmt(v: torch.Tensor, k: int = 6) -> str:
    s = ", ".join(f"{float(a):+.4f}" for a in v[:k])
    return "[" + s + (", …" if v.numel() > k else "") + "]"


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    return float(a @ b) / (na * nb) if na > 0 and nb > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toys", nargs="+", default=["mlp", "saturating", "satint", "multilinear"],
                    choices=sorted(TH.TOYS))
    ap.add_argument("--ds", nargs="+", type=int, default=[5])
    ap.add_argument("--alphas", nargs="+", type=float, default=[3, 10, 30, 100, 300])
    ap.add_argument("--small-alphas", nargs="*", type=float, default=[0.1, 1.0])
    ap.add_argument("--n", type=int, default=20000, help="paths per α (antithetic pairs)")
    ap.add_argument("--N", type=int, default=128, help="polygon bins per path")
    ap.add_argument("--Q", type=int, default=8, help="Gauss–Legendre nodes per segment")
    ap.add_argument("--max-shapley-d", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="default: cuda if available")
    args = ap.parse_args()
    print(f"[theory] device={TH.set_device(args.device)}")

    for d in args.ds:
        for name in args.toys:
            F = TH.TOYS[name](d, args.seed)
            ig = TH.ig_exact(F, d)
            sh = TH.shapley_exact(F, d) if d <= args.max_shapley_d else None
            C1 = TH.spi_correction(F, d, alpha=0.0)              # (α+1)·C, α-free
            D = float(F(TH.ones(d)) - F(0 * TH.ones(d)))
            print(f"\n=== toy {name}  (d={d}, seed={args.seed}, D={D:+.4f})")
            print(f"IG      {fmt(ig)}   |IG|={float(ig.norm()):.4f}")
            if sh is not None:
                print(f"Shapley {fmt(sh)}   |IG − Shapley| = {float((ig - sh).norm()):.4f}"
                      f"   cos(C, Shapley − IG) = {cos(C1, sh - ig):+.3f}"
                      f"   |(α+1)C| / |Shapley − IG| = {float(C1.norm()) / max(float((sh - ig).norm()), 1e-30):.3f}")
            print(f"{'alpha':>7} {'|Δ|':>9} {'|C|':>9} {'cos':>7} {'rel':>7} {'snr':>7}   (α+1)Δ  /  (α+1)C")
            for k, a in enumerate(args.alphas):
                gen = torch.Generator().manual_seed(1000 * args.seed + 97 * d + k)
                mc = TH.spi_monte_carlo(F, d, a, args.n, args.N, args.Q, gen)
                C = C1 / (a + 1.0)
                dlt = mc["mean"] - ig
                nC, nD, nse = float(C.norm()), float(dlt.norm()), float(mc["se"].norm())
                rel = float((dlt - C).norm()) / nC if nC > 0 else float("nan")
                snr = nC / nse if nse > 0 else float("nan")
                print(f"{a:7g} {nD:9.2e} {nC:9.2e} {cos(dlt, C):7.3f} {rel:7.3f} {snr:7.1f}   "
                      f"{fmt((a + 1) * dlt)}  /  {fmt(C1)}   (|se|={nse:.1e}, resid={mc['residual']:+.1e})")
            for k, a in enumerate(args.small_alphas):
                gen = torch.Generator().manual_seed(5000 + 1000 * args.seed + 97 * d + k)
                mc = TH.spi_monte_carlo(F, d, a, args.n, args.N, args.Q, gen)
                extra = (f"  |SPI − Shapley| = {float((mc['mean'] - sh).norm()):.4f}"
                         f"  cos(SPI − IG, Shapley − IG) = {cos(mc['mean'] - ig, sh - ig):+.3f}") if sh is not None else ""
                print(f"  small α={a:g}: |SPI − IG| = {float((mc['mean'] - ig).norm()):.4f}{extra}"
                      f"  (|se|={float(mc['se'].norm()):.1e})")

    print("\nRead: cos → 1 and rel shrinking with α (until ≈ 1/snr) confirm the formula; multilinear: |Δ| ≈ |se|."
          "\n      cos(C, Shapley − IG) < 0: large-α SPI moves away from Shapley (not an IG→Shapley interpolation)."
          "\n      d-scaling: compare rel at the same α across d for mlp vs mlpn.")


if __name__ == "__main__":
    main()
