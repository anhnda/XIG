"""
Token-level attribution on text classifiers (E7), with a Shapley reference.

Methods (spec syntax as in experiments.run, "name:k=v,k=v"):
    ig          N=50, est=mid                 straight line in embedding space
    spi         alpha, K, N, est, completion  SPI over tokens (one schedule per token)
    smooth      sigma, K, N                   smooth IG, σ in units of the embedding RMS
    tpa         alpha, K, N, est, completion, mode (uniform|gibbs|core|r), eps, kappa, M
    kernelshap  M                             KernelSHAP on token coalitions
Reference: Shapley of the [MASK] game, antithetic permutation sampling with a
standard-error stopping rule (cached per example with --ref-cache).
Metrics: distance to the reference (Spearman, Kendall, rel. L1, top-20% overlap),
ERASER comprehensiveness / sufficiency (AOPC over 1/5/10/20/50%), held-out mask
fidelity hold_rel, completeness residual, forward/backward counts.

python -m experiments.nlp.run_nlp --dataset sst2 --model textattack/bert-base-uncased-SST-2 \
    --n 872 --methods "ig:N=50" "spi:alpha=0.1" "spi:alpha=1" "spi:alpha=10" "smooth:sigma=0.1" \
    "tpa:alpha=1,mode=uniform" "tpa:alpha=1,mode=gibbs,eps=0.1" "tpa:alpha=1,mode=core,eps=0.1" \
    "tpa:alpha=1,mode=r,eps=0.1,kappa=1" "kernelshap:M=256" \
    --ref-cache runs/nlp_ref_sst2_bert --out runs/nlp_sst2_bert.jsonl
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from experiments.common import append_row, load_done, parse_spec
from experiments.nlp.hf import EmbTarget, load_model, load_texts, prepare
from pathattr import Oracle
from pathattr import tokens as TK
from pathattr import tpa as T
from pathattr.complete import complete_stack, support_scale
from pathattr.masks import shapley_kernel_masks

STOCHASTIC = {"spi", "smooth", "tpa", "kernelshap"}


def run_method(name: str, p: dict, oracle: Oracle, ti: TK.TokenInput, D: float,
               gen: torch.Generator) -> tuple[torch.Tensor, dict]:
    """→ token attributions φ (F,) float64, info."""
    info: dict = {}
    if name in ("ig", "spi", "tpa"):
        N = int(p.get("N", 50 if name == "ig" else 24))
        est = str(p.get("est", "mid"))
        completion = str(p.get("completion", "none" if name != "tpa" else "support"))
        if completion == "cv":
            est = "random_cv"
        if name == "ig":
            make = lambda: TK.build_path(ti, TK.straight_sched(ti.F, N))
            K = 1
        else:
            alpha = float(p.get("alpha", 1.0))
            make = lambda: TK.build_path(ti, TK.spi_sched(ti.F, N, alpha, gen))
            K = int(p.get("K", 8))
        A_raw = ti.restrict(T.path_candidates(oracle, make, K, est, int(p.get("Q", 1)),
                                              int(p.get("K_cv", 8)), D, gen))     # (K, F·h)
        lam = support_scale(A_raw)
        A, cinfo = complete_stack(A_raw, D, "none" if completion == "cv" else completion,
                                  float(p.get("eta", 0.0)), lam)
        info["max_abs_residual_before"] = max(abs(v) for v in cinfo["residual_before"])
        if name in ("ig", "spi"):
            return ti.token_sum(A.mean(dim=0)), info
        M = int(p.get("M", 256))
        m = shapley_kernel_masks(ti.F, M, gen)
        y = TK.game_values(oracle, ti, m)
        game = T.Game(B=m.double(), y=y, feat=ti.feat(), F=ti.F, D=D)
        a, q, finfo = T.fit(A, game, str(p.get("mode", "core")), float(p.get("eps", 0.1)),
                            float(p.get("kappa", 1.0)), lam)
        info.update(finfo)
        return ti.token_sum(a), info
    if name == "smooth":
        A = TK.smooth_candidates(oracle, ti, int(p.get("N", 24)), float(p.get("sigma", 0.1)),
                                 int(p.get("K", 8)), gen)
        return ti.token_sum(A.mean(dim=0)), info
    if name == "kernelshap":
        M = int(p.get("M", 256))
        m = shapley_kernel_masks(ti.F, M, gen)
        y = TK.game_values(oracle, ti, m)
        return T.kernelshap(m.double(), y, D, ridge=float(p.get("ridge", 1e-10))), info
    raise ValueError(f"unknown method '{name}'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["sst2", "ag_news", "imdb"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--ref-max-perm", type=int, default=5000)
    ap.add_argument("--ref-tol", type=float, default=0.02)
    ap.add_argument("--ref-cache", default=None)
    ap.add_argument("--M-hold", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tok, hf_model = load_model(args.model, dev)
    texts = load_texts(args.dataset, args.n, args.data_seed)
    specs = [(s, *parse_spec(s)) for s in args.methods]
    done = load_done(args.out)
    if args.ref_cache:
        os.makedirs(args.ref_cache, exist_ok=True)

    for j, (ex_id, text) in enumerate(texts):
        name = f"{args.dataset}_{ex_id}"
        ti, cls, conf = prepare(tok, hf_model, text, args.max_len, dev)
        if ti.F < 2:
            continue
        oracle = Oracle(EmbTarget(hf_model, cls), chunk=args.chunk)
        prob_oracle = Oracle(EmbTarget(hf_model, cls, prob=True), chunk=args.chunk)
        D = oracle.endpoint_gap(ti.Eb, ti.E)

        # Shapley reference (cached)
        cache = os.path.join(args.ref_cache, f"{name}.pt") if args.ref_cache else None
        if cache and os.path.exists(cache):
            ref, rinfo = torch.load(cache)
        else:
            oracle.reset()
            t0 = time.time()
            ref, rinfo = TK.shapley_permutation(oracle, ti, args.ref_max_perm, tol_rel=args.ref_tol,
                                                gen=torch.Generator().manual_seed(17 + ex_id),
                                                chunk=args.chunk * 2)
            rinfo["time"] = time.time() - t0
            if cache:
                torch.save((ref, rinfo), cache)

        # held-out masks, shared by all methods of this example
        g_hold = torch.Generator().manual_seed(1_000_003 * ex_id + 5)
        mh = shapley_kernel_masks(ti.F, args.M_hold, g_hold)
        hold = T.Game(B=mh.double(), y=TK.game_values(oracle, ti, mh), feat=torch.arange(ti.F), F=ti.F, D=D)

        common = {"image": name, "dataset": args.dataset, "model": args.model, "class": cls, "conf": conf,
                  "F": ti.F, "D": D, "ref_se_max": rinfo["se_max"], "ref_n_perm": rinfo["n_perm"]}
        if (name, "shapley_ref", 0) not in done:
            row = dict(common, method="shapley_ref", seed=0, error=None, **TK.reference_distances(ref, ref),
                       **TK.eraser_scores(prob_oracle, ti, ref), hold_rel=hold.rel_loss(ref),
                       residual=D - float(ref.sum()), n_fwd=rinfo["n_fwd"], n_bwd=0)
            append_row(args.out, row)

        for spec, mname, params in specs:
            for seed in (args.seeds if mname in STOCHASTIC else args.seeds[:1]):
                if (name, spec, seed) in done:
                    continue
                gen = torch.Generator().manual_seed(seed + 1_000_003 * ex_id)
                oracle.reset()
                t0 = time.time()
                row = dict(common, method=spec, seed=seed)
                try:
                    phi, info = run_method(mname, params, oracle, ti, D, gen)
                    n_fwd, n_bwd = oracle.counts()
                    row.update({"error": None, "time": time.time() - t0, "n_fwd": n_fwd, "n_bwd": n_bwd,
                                "residual": D - float(phi.sum()), "hold_rel": hold.rel_loss(phi),
                                "info": info})
                    row.update(TK.reference_distances(phi, ref))
                    row.update(TK.eraser_scores(prob_oracle, ti, phi))
                except Exception as e:  # noqa: BLE001
                    row.update({"error": f"{type(e).__name__}: {e}"})
                append_row(args.out, row)
        print(f"[{j + 1}/{len(texts)}] {name} F={ti.F} ref_perm={rinfo['n_perm']} se={rinfo['se_max']:.3g}")


if __name__ == "__main__":
    main()
