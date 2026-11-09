"""
P0 — is the Phase-1 collapse of μ explained by the closed form?

Per image, on the straight line (N steps, aligned Δf):
  a_k = |d_k|, gap = a(1) − a(2)
  μ_proj = Proj_simplex((λ/τ)·a)          (optimum without the Var term)
  theorem: gap ≥ τ/λ and Δf_k* ≠ 0  ⇒  e_k* is the GLOBAL minimiser of the full J
  μ_adam      = utility.optimize_mu_signal_harvesting(d, Δf)          (fixed Δf)
  μ_adam_leg  = same with the legacy shifted Δf (what the paper ran)
Logged: gap, supports, argmax steps, J of each μ, L1 distances.

python -m experiments.p0_mu_projection --data seg --seg-mat gtsegs_ijcv.mat \
    --n 200 --index-file runs/seg200.json --out runs/p0_mu.jsonl
"""

from __future__ import annotations

import argparse

import torch

from experiments.common import add_data_args, append_row, open_data, target_model
from pathattr import Oracle, eval_path, legacy_df
from pathattr import metrics as MET
from pathattr import paths


def project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto {μ ≥ 0, Σμ = 1}."""
    u, _ = torch.sort(v, descending=True)
    css = u.cumsum(0) - 1.0
    k = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    rho = int((u - css / k > 0).nonzero().max())
    eta = css[rho] / (rho + 1)
    return (v - eta).clamp(min=0.0)


def summary(mu: torch.Tensor) -> dict:
    return {"argmax": int(mu.argmax()), "support": int((mu > 1e-3).sum()),
            "max": float(mu.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    add_data_args(ap)
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--mu-iter", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import utility  # legacy optimiser and objective
    dev, backbone, it, n = open_data(args)
    for j, (idx, x, mask, name) in enumerate(it):
        x = x.to(dev)
        model, cls, conf = target_model(backbone, x)
        oracle = Oracle(model)
        xb = MET.make_baseline(x, args.baseline)
        pe = eval_path(oracle, paths.straight(xb, x, args.N))
        d, df = pe.d.double().cpu(), pe.df.double().cpu()
        df_leg = legacy_df(pe.f.double().cpu())

        a = d.abs()
        top2 = torch.topk(a, 2).values
        gap = float(top2[0] - top2[1])
        k_star = int(a.argmax())
        mu_proj = project_simplex((args.lam / args.tau) * a)
        e_star = torch.zeros_like(a)
        e_star[k_star] = 1.0
        theorem = gap >= args.tau / args.lam and abs(float(df[k_star])) > 1e-12

        J = lambda mu, dfx: utility.compute_signal_harvesting_objective(
            d.float(), dfx.float(), mu.float(), lam=args.lam, tau=args.tau)[0]
        mu_adam = utility.optimize_mu_signal_harvesting(
            d.float(), df.float(), lam=args.lam, tau=args.tau, n_iter=args.mu_iter).double()
        mu_adam_leg = utility.optimize_mu_signal_harvesting(
            d.float(), df_leg.float(), lam=args.lam, tau=args.tau, n_iter=args.mu_iter).double()

        row = {
            "image": name, "index": idx, "class": cls, "conf": conf, "N": args.N,
            "lam": args.lam, "tau": args.tau, "gap": gap, "threshold": args.tau / args.lam,
            "theorem_applies": bool(theorem), "k_star": k_star,
            "df_k_star": float(df[k_star]), "df_leg_k_star": float(df_leg[k_star]),
            "transition_step": int(df.abs().argmax()),
            "sum_df": float(df.sum()), "sum_df_legacy": float(df_leg.sum()),
            "D": float(pe.f[-1] - pe.f[0]),
            "proj": summary(mu_proj), "adam": summary(mu_adam), "adam_legacy": summary(mu_adam_leg),
            "J_e_star": J(e_star, df), "J_proj": J(mu_proj, df),
            "J_adam": J(mu_adam, df), "J_adam_legacy_on_legacy_df": J(mu_adam_leg, df_leg),
            "L1_adam_vs_proj": float((mu_adam - mu_proj).abs().sum()),
            "L1_adam_vs_e_star": float((mu_adam - e_star).abs().sum()),
            "L1_adam_leg_vs_e0": float((mu_adam_leg - torch.eye(args.N, dtype=torch.float64)[0]).abs().sum()),
        }
        append_row(args.out, row)
        print(f"[{j + 1}/{n}] {name}: gap={gap:.4f} thm={theorem} k*={k_star} "
              f"proj_supp={row['proj']['support']} adam_argmax={row['adam']['argmax']} "
              f"adam_leg_argmax={row['adam_legacy']['argmax']} J(e*)={row['J_e_star']:.4f} "
              f"J(adam)={row['J_adam']:.4f}")


if __name__ == "__main__":
    main()
