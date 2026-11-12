"""
TPA sweep: SPI(α) candidates → {uniform, gibbs, core, r} × ε × κ, plus KernelSHAP.

For each image and seed:
  * fit masks (family A: Shapley kernel on --parts), held-out masks (same family,
    fresh draw) and transfer masks (family B: --transfer-family on
    --transfer-parts) are drawn ONCE and shared by every α and variant;
  * for each α, K candidate paths are drawn once and completed (--completion),
    then every variant is fitted on the fit masks;
  * logged per variant: fit_rel / hold_rel / tr_rel  (loss / Var(y), lower is
    better), q, G/D/L (ambiguity), equidistance, residual; with --eval also the
    usual metrics (ins/del, localisation, optional infidelity / ROAD).
Variant names:  spi_a{α}|uniform · tpa_a{α}|gibbs|e{ε} · tpa_a{α}|core|e{ε}
                · tpa_a{α}|r|e{ε}|k{κ} · kernelshap

Protocol (no test information is used for any choice):
  1) calibration slice, no metrics:
     python -m experiments.tpa_sweep --data seg --seg-mat M.mat --index-file runs/seg200.json \
        --start 0 --n 50 --out runs/tpa_calib.jsonl
  2) python -m experiments.select_tpa runs/tpa_calib.jsonl --out runs/tpa_chosen.json
  3) test slice, chosen variants only, with metrics:
     python -m experiments.tpa_sweep ... --start 50 --n 150 \
        --variants-file runs/tpa_chosen.json --eval --infd-n 64 --road --out runs/tpa_test.jsonl
  4) python -m experiments.summarize runs/tpa_test.jsonl --ref "<chosen uniform>" \
        --metrics hold_rel tr_rel insdel_black insdelp_black iou infd road
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from experiments.common import (TargetProb, add_data_args, append_row, evaluate, load_done,
                                open_data, target_model)
from pathattr import Oracle
from pathattr import masks as MK
from pathattr import metrics as MET
from pathattr import tpa as T
from pathattr.complete import complete_stack, support_scale
from pathattr.methods import _specs, kernelshap_from_masks


def fmt(v: float) -> str:
    return "inf" if math.isinf(v) else f"{v:g}"


def variant_grid(alphas, modes, epss, kappas):
    """[(name, alpha, mode, eps, kappa)]"""
    out = []
    for a in alphas:
        if "uniform" in modes:
            out.append((f"spi_a{fmt(a)}|uniform", a, "uniform", float("inf"), None))
        for e in epss:
            if "gibbs" in modes:
                out.append((f"tpa_a{fmt(a)}|gibbs|e{fmt(e)}", a, "gibbs", e, None))
            if "core" in modes:
                out.append((f"tpa_a{fmt(a)}|core|e{fmt(e)}", a, "core", e, None))
            if "r" in modes:
                for k in kappas:
                    out.append((f"tpa_a{fmt(a)}|r|e{fmt(e)}|k{fmt(k)}", a, "r", e, k))
    if "kernelshap" in modes:
        out.append(("kernelshap", None, "kernelshap", None, None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--est", default="mid", choices=["left", "mid", "random", "gl", "random_cv"])
    ap.add_argument("--Q", type=int, default=1)
    ap.add_argument("--completion", default="support", choices=["support", "rescale", "none", "cv"])
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--parts", default="slic:64", help="fit/hold partitions, '+'-joined")
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--M-hold", type=int, default=256)
    ap.add_argument("--transfer-parts", default="grid:28+grid:28:14:14")
    ap.add_argument("--transfer-family", default="bernoulli", choices=["bernoulli", "shapley"])
    ap.add_argument("--M-transfer", type=int, default=256)
    ap.add_argument("--modes", nargs="+", default=["uniform", "gibbs", "core", "r", "kernelshap"])
    ap.add_argument("--eps", nargs="+", type=float, default=[1.0, 0.1, 0.01, 0.0])
    ap.add_argument("--kappas", nargs="+", type=float, default=[1.0])
    ap.add_argument("--variants-file", default=None, help="JSON list of variant names to run")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--eval", action="store_true", help="also compute ins/del, localisation, …")
    ap.add_argument("--insdel", nargs="+", default=["black"], choices=["black", "gray", "white", "blur"])
    ap.add_argument("--infd-n", type=int, default=0)
    ap.add_argument("--infd-sigma", type=float, default=0.1)
    ap.add_argument("--road", action="store_true")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grid = variant_grid(args.alphas, args.modes, args.eps, args.kappas)
    if args.variants_file:
        with open(args.variants_file) as fh:
            keep = set(json.load(fh))
        grid = [g for g in grid if g[0] in keep]
        missing = keep - {g[0] for g in grid}
        if missing:
            raise SystemExit(f"variants not in the grid (check --alphas/--eps/--kappas): {sorted(missing)}")
    alphas_needed = sorted({g[1] for g in grid if g[1] is not None})
    print(f"[sweep] {len(grid)} variants, alphas {alphas_needed}")

    done = load_done(args.out)
    dev, backbone, it, n = open_data(args)
    fit_specs, tr_specs = _specs(args.parts), _specs(args.transfer_parts)

    for j, (idx, x, mask, name) in enumerate(it):
        x = x.to(dev)
        model, cls, conf = target_model(backbone, x)
        prob_model = TargetProb(backbone, cls)
        oracle = Oracle(model, chunk=args.chunk)
        xb = MET.make_baseline(x, args.baseline)
        bases = {b: MET.make_baseline(x, b) for b in args.insdel}
        D = oracle.endpoint_gap(xb, x)
        feat, F = T.image_feat(x)
        fit_lbl = [MK.partition(x, s) for s in fit_specs]
        tr_lbl = [MK.partition(x, s) for s in tr_specs]

        for seed in args.seeds:
            todo = [g for g in grid if (name, g[0], seed) not in done]
            if not todo:
                continue
            base_seed = seed + 1_000_003 * idx
            g_fit = torch.Generator().manual_seed(base_seed + 1)
            g_hold = torch.Generator().manual_seed(base_seed + 2)
            g_tr = torch.Generator().manual_seed(base_seed + 3)
            oracle.reset()
            ms_fit = MK.build_masks(oracle, x, xb, fit_specs, args.M, g_fit, "shapley", labels=fit_lbl)
            ms_hold = MK.build_masks(oracle, x, xb, fit_specs, args.M_hold, g_hold, "shapley", labels=fit_lbl)
            ms_tr = MK.build_masks(oracle, x, xb, tr_specs, args.M_transfer, g_tr,
                                   args.transfer_family, labels=tr_lbl)
            mask_fwd = oracle.counts()[0]
            game = T.Game(ms_fit.B(), ms_fit.y, feat, F, D)
            hold = T.Game(ms_hold.B(), ms_hold.y, feat, F, D)
            trg = T.Game(ms_tr.B(), ms_tr.y, feat, F, D)

            common = {"image": name, "index": idx, "rank": args.start + j, "class": cls, "conf": conf,
                      "model": args.model, "baseline": args.baseline, "seed": seed, "D": D,
                      "M": args.M, "parts": args.parts, "mask_fwd": mask_fwd, "error": None}

            def emit(vname, attr_flat, extra):
                a_img = attr_flat.view_as(x).to(device=x.device, dtype=x.dtype)
                row = dict(common)
                row.update({"method": vname, "fit_rel": game.rel_loss(attr_flat),
                            "hold_rel": hold.rel_loss(attr_flat), "tr_rel": trg.rel_loss(attr_flat),
                            "attr_sum": float(attr_flat.sum()), "residual": D - float(attr_flat.sum())})
                row.update(extra)
                if args.eval:
                    egen = torch.Generator().manual_seed(7_919 + idx)
                    row.update(evaluate(model, x, mask, a_img, bases, 100, prob_model,
                                        infd_n=args.infd_n, infd_sigma=args.infd_sigma,
                                        road=args.road, gen=egen))
                append_row(args.out, row)

            for alpha in alphas_needed:
                sub = [g for g in todo if g[1] == alpha]
                if not sub:
                    continue
                g_path = torch.Generator().manual_seed(base_seed + 10 + int(round(1000 * alpha)))
                oracle.reset()
                est = "random_cv" if args.completion == "cv" else args.est
                A_raw = T.spi_image_candidates(oracle, xb, x, g_path, alpha, args.K, args.N, est,
                                               args.Q, True, 8, D)
                n_bwd = oracle.counts()[1]
                lam = support_scale(A_raw)
                A, cinfo = complete_stack(A_raw, D, "none" if args.completion == "cv" else args.completion,
                                          args.eta, lam)
                cand = {"alpha": alpha, "K": args.K, "N": args.N, "est": est, "completion": args.completion,
                        "n_bwd": n_bwd, "support_size": cinfo["support_size"],
                        "max_abs_residual_before": max(abs(v) for v in cinfo["residual_before"]),
                        "cand_hold_rel": [hold.rel_loss(A[k]) for k in range(A.shape[0])]}
                for vname, _, mode, eps, kappa in sub:
                    try:
                        a, q, info = T.fit(A, game, mode, eps, kappa if kappa is not None else 1.0, lam)
                        extra = dict(cand)
                        extra.update({"mode": mode, "eps": eps, "kappa": kappa, "info": info})
                        emit(vname, a, extra)
                    except Exception as e:  # noqa: BLE001
                        row = dict(common)
                        row.update({"method": vname, "error": f"{type(e).__name__}: {e}"})
                        append_row(args.out, row)

            for vname, _, mode, _, _ in todo:
                if mode != "kernelshap":
                    continue
                attr, G = kernelshap_from_masks(ms_fit, x, D)
                emit(vname, attr.reshape(-1).double().cpu(), {"mode": "kernelshap", "G": G})
        print(f"[{j + 1}/{n}] {name} done")


if __name__ == "__main__":
    main()
