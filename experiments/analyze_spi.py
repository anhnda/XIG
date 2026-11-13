"""
Why does SPI rank near the top on many metrics? Tier-2 analysis on the saved 'axis' attributions
plus forward-only profiles of f along the paths. Nothing here computes new attributions.

Per image (one JSONL row):
  straight line (IG path), T = N+1 forwards:
      line_t50, line_t90   first t where (f(t) − f(x'))/D ≥ 0.5 / 0.9   (small = early saturation)
      line_area            ∫ (f(t) − f(x'))/D dt                         (large = early saturation)
      line_tv              Σ|Δf| / |D|                                   (1 = monotone)
  SPI paths for each α (--prof-alphas, --prof-K paths, N steps, forwards only):
      prof_<a>_tv / _neg / _t50 / _t90 / _ent / _max
        tv   Σ|Δf|/|D|, neg = share of the variation that goes against D,
        ent  exp(entropy of |Δf_k|)/N (1 = Δf spread evenly over the steps), max = largest step share
  saved attributions (mean over seeds) for each α found in --attr-dir, IG = α ∞:
      map_<a>_overlap / _iou / _gini / _top5 / _checker
      gap_<a>     debiased ‖Ā_α − A_IG‖₂ / ‖A_IG‖₂ (seed noise of Ā removed)
      gapx_<a>    (α+1)·gap_<a>   — constant in α if SPI − IG ∝ 1/(α+1) (the first-order theory)
      enr_<a>     where the extra |A| of SPI goes: (mass of (map_α − map_IG)⁺ in mask / its total) / mask area
                  (> 1: SPI moves mass into the object)
      coarse_<a>  share of ‖map_α − map_IG‖² carried at 16 px scale (1 = smooth, 0 = speckle)
The summary (also with --summary-only) prints medians per α and Spearman correlations between
the per-image saturation of the line and the per-image advantage of SPI(--adv-alpha) over IG in
the axis JSONL.

bash scripts/run_all.sh $MAT analyze        (NIMG=2 SEEDS=0 first)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import torch
import torch.nn.functional as Fnn

from experiments.common import add_data_args, append_row, open_data, target_model
from pathattr import Oracle
from pathattr import metrics as MET
from pathattr import paths as PATHS

INF = float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# helpers (pure, tested)
# ─────────────────────────────────────────────────────────────────────────────

def alpha_of(spec: str) -> float:
    if spec.startswith("ig"):
        return INF
    m = re.search(r"alpha=([0-9.eE+-]+)", spec)
    return float(m.group(1)) if m else float("nan")


def akey(a: float) -> str:
    return "inf" if math.isinf(a) else f"{a:g}"


def profile_stats(f: torch.Tensor) -> dict:
    """f: (N+1,) values along a path from x' to x."""
    f = f.double().cpu()
    N = f.numel() - 1
    D = float(f[-1] - f[0])
    df = f[1:] - f[:-1]
    var = float(df.abs().sum())
    out = {"tv": var / abs(D) if abs(D) > 1e-12 else float("nan")}
    sgn = 1.0 if D >= 0 else -1.0
    out["neg"] = float((-(sgn * df)).clamp_min(0).sum()) / var if var > 0 else float("nan")
    p = df.abs() / var if var > 0 else torch.full((N,), 1.0 / N, dtype=torch.float64)
    ent = float(-(p * torch.log(p.clamp_min(1e-300))).sum())
    out["ent"] = math.exp(ent) / N
    out["max"] = float(p.max())
    g = (f - f[0]) / D if abs(D) > 1e-12 else torch.zeros_like(f)
    t = torch.linspace(0, 1, N + 1, dtype=torch.float64)
    for q in (0.5, 0.9):
        hit = torch.nonzero(g >= q)
        out[f"t{int(q * 100)}"] = float(t[int(hit[0])]) if hit.numel() else 1.0
    out["area"] = float(((g[1:] + g[:-1]) / 2).sum() / N)
    return out


def gini(v: torch.Tensor) -> float:
    v = v.flatten().double().sort().values
    n = v.numel()
    s = float(v.sum())
    if s <= 0:
        return float("nan")
    i = torch.arange(1, n + 1, dtype=torch.float64)
    return float(((2 * i - n - 1) * v).sum() / (n * s))


def top_share(v: torch.Tensor, q: float = 0.05) -> float:
    v = v.flatten().double()
    k = max(1, int(round(q * v.numel())))
    s = float(v.sum())
    return float(torch.topk(v, k).values.sum()) / s if s > 0 else float("nan")


def debiased_gap(A: torch.Tensor, ref: torch.Tensor) -> float:
    """A: (S, ...) seeds of one method; ref: (...) deterministic. ‖mean − ref‖ with the
    seed-noise of the mean removed (S ≥ 2), relative to ‖ref‖."""
    S = A.shape[0]
    m = A.mean(dim=0)
    g2 = float(((m - ref) ** 2).sum())
    if S >= 2:
        g2 -= float(((A - m) ** 2).sum()) / (S - 1) / S
    return math.sqrt(max(g2, 0.0)) / max(float(ref.norm()), 1e-30)


def coarse_share(d: torch.Tensor, cell: int = 16) -> float:
    e = float((d ** 2).sum())
    if e <= 0:
        return float("nan")
    p = Fnn.avg_pool2d(d[None, None], cell, stride=cell, ceil_mode=True)
    return float((p ** 2).sum()) * cell * cell / e


def spearman(a, b) -> tuple[float, int]:
    pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y and abs(x) != INF and abs(y) != INF]
    n = len(pairs)
    if n < 4:
        return float("nan"), n

    def rank(v):
        v = torch.tensor(v, dtype=torch.float64)
        o = torch.argsort(v)
        r = torch.empty_like(v)
        r[o] = torch.arange(n, dtype=torch.float64)
        for val in torch.unique(v):                     # average ties
            idx = v == val
            if int(idx.sum()) > 1:
                r[idx] = r[idx].mean()
        return r
    ra, rb = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(ra.norm() * rb.norm())
    return (float((ra * rb).sum()) / den if den > 0 else float("nan")), n


# ─────────────────────────────────────────────────────────────────────────────
# per image
# ─────────────────────────────────────────────────────────────────────────────

def load_attrs(folder: str, name: str) -> dict:
    """{alpha: (S, C, H, W) float32} from experiments.run --save-attr files of one image."""
    by = defaultdict(list)
    for fn in sorted(glob.glob(os.path.join(folder, f"{glob.escape(name)}__*__s*.pt"))):
        d = torch.load(fn, map_location="cpu")
        a = alpha_of(d["method"])
        if a == a:
            by[a].append(d["attr"].float()[0])
    return {a: torch.stack(v) for a, v in by.items()}


@torch.no_grad()
def path_values(oracle: Oracle, pts: torch.Tensor) -> torch.Tensor:
    return oracle.f(pts).double().cpu()


def analyse_image(args, oracle, x, xb, mask, name, gen) -> dict:
    row = {}
    # straight line
    t = torch.linspace(0, 1, args.N + 1, device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
    fl = path_values(oracle, xb + t * (x - xb))
    for k, v in profile_stats(fl).items():
        row[f"line_{k}"] = v
    # SPI paths
    for a in args.prof_alphas:
        acc = defaultdict(list)
        for _ in range(args.prof_K):
            pts = PATHS.spi(xb, x, N=args.N, alpha=a, gen=gen)
            for k, v in profile_stats(path_values(oracle, pts)).items():
                acc[k].append(v)
        for k, v in acc.items():
            row[f"prof_{akey(a)}_{k}"] = sum(v) / len(v)
    # saved attributions
    if args.attr_dir:
        A = load_attrs(args.attr_dir, name)
        if INF not in A:
            row["attr_missing"] = True
            return row
        ig = A[INF][0]
        m_ig = ig.abs().sum(dim=0)
        msk = None if mask is None else mask.cpu().bool()
        area = float(msk.double().mean()) if msk is not None else float("nan")
        row["mask_area"] = area
        for a, As in sorted(A.items()):
            mean = As.mean(dim=0)
            m = mean.abs().sum(dim=0)
            key = akey(a)
            if msk is not None:
                loc = MET.localisation(mean[None], msk)
                row[f"map_{key}_overlap"], row[f"map_{key}_iou"] = loc["overlap"], loc["iou"]
            row[f"map_{key}_gini"] = gini(m)
            row[f"map_{key}_top5"] = top_share(m)
            row[f"map_{key}_checker"] = MET.checker_energy(mean[None], msk)
            row[f"nseed_{key}"] = As.shape[0]
            if math.isinf(a):
                continue
            g = debiased_gap(As, ig)
            row[f"gap_{key}"], row[f"gapx_{key}"] = g, g * (a + 1)
            dm = m - m_ig
            row[f"coarse_{key}"] = coarse_share(dm)
            pos = dm.clamp_min(0)
            if msk is not None and float(pos.sum()) > 0 and area > 0:
                row[f"enr_{key}"] = float(pos[msk].sum()) / float(pos.sum()) / area
    return row


# ─────────────────────────────────────────────────────────────────────────────
# summary
# ─────────────────────────────────────────────────────────────────────────────

def median(v):
    v = sorted(x for x in v if x == x and abs(x) != INF)
    if not v:
        return float("nan")
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def load_axis(path: str, metrics: list[str]) -> dict:
    """{image: {alpha: {metric: mean over seeds}}}"""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error"):
                continue
            a = alpha_of(r["method"])
            for m in metrics:
                if r.get(m) is not None:
                    acc[r["image"]][a][m].append(float(r[m]))
    return {im: {a: {m: sum(v) / len(v) for m, v in d.items()} for a, d in da.items()}
            for im, da in acc.items()}


def summarize(out: str, axis: str | None, adv_alpha: float, metrics: list[str]) -> None:
    rows = []
    with open(out) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"\n[analyze] {len(rows)} images  ({out})")
    keys = sorted({k for r in rows for k in r if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)})

    def show(prefix: str, title: str):
        groups = defaultdict(dict)
        for k in keys:
            m = re.match(rf"{prefix}_([^_]+)_(.+)$", k)
            if m:
                groups[m.group(1)][m.group(2)] = median([r.get(k, float("nan")) for r in rows])
        if not groups:
            return
        cols = sorted({c for g in groups.values() for c in g})
        print(f"\n{title} (medians)")
        print(f"{'alpha':>8} " + " ".join(f"{c:>9}" for c in cols))
        order = sorted(groups, key=lambda s: INF if s == "inf" else float(s))
        for a in order:
            print(f"{a:>8} " + " ".join(f"{groups[a].get(c, float('nan')):9.4f}" for c in cols))

    line = {k[5:]: median([r.get(k, float("nan")) for r in rows]) for k in keys if k.startswith("line_")}
    if line:
        print("\nstraight line (medians): " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(line.items())))
    show("prof", "f along SPI paths")
    show("map", "attribution maps")
    per_a = defaultdict(dict)
    for k in keys:
        m = re.match(r"(gap|gapx|enr|coarse)_(.+)$", k)
        if m:
            per_a[m.group(2)][m.group(1)] = median([r.get(k, float("nan")) for r in rows])
    if per_a:
        print("\nSPI − IG (medians): gap = debiased rel. L2, gapx = (α+1)·gap (flat ⇒ 1/(α+1) law),"
              " enr > 1 ⇒ extra mass inside the object, coarse = 16 px share of the map difference")
        print(f"{'alpha':>8} {'gap':>9} {'gapx':>9} {'enr':>9} {'coarse':>9}")
        for a in sorted(per_a, key=float):
            d = per_a[a]
            print(f"{a:>8} " + " ".join(f"{d.get(c, float('nan')):9.4f}" for c in ("gap", "gapx", "enr", "coarse")))

    if not axis or not os.path.exists(axis):
        return
    ax = load_axis(axis, metrics)
    adv = {m: {} for m in metrics}
    for im, da in ax.items():
        if adv_alpha in da and INF in da:
            for m in metrics:
                if m in da[adv_alpha] and m in da[INF]:
                    adv[m][im] = da[adv_alpha][m] - da[INF][m]
    ka = akey(adv_alpha)
    preds = [k for k in ("line_area", "line_t90", "line_t50", "line_tv", f"prof_{ka}_ent",
                         f"prof_{ka}_tv", f"enr_{ka}", f"coarse_{ka}", f"gap_{ka}", "mask_area") if k in keys]
    print(f"\nadvantage of SPI α={ka} over IG per image (axis JSONL): median Δ and Spearman with predictors")
    print(f"{'metric':>16} {'medΔ':>8} {'win%':>6} " + " ".join(f"{p:>14}" for p in preds))
    by_img = {r["image"]: r for r in rows}
    for m in metrics:
        ims = [im for im in adv[m] if im in by_img]
        if not ims:
            continue
        d = [adv[m][im] for im in ims]
        win = 100.0 * sum(v > 0 for v in d) / len(d)
        cells = []
        for p in preds:
            rho, n = spearman([by_img[im].get(p, float("nan")) for im in ims], d)
            cells.append(f"{rho:+8.3f} (n={n:3d})")
        print(f"{m:>16} {median(d):+8.4f} {win:6.1f} " + " ".join(f"{c:>14}" for c in cells))
    print("Read: line_area ↑ / line_t90 ↓ = earlier saturation. If SPI's edge is a saturation effect,"
          "\n      rho(line_area, Δ) > 0 and rho(line_t90, Δ) < 0 on the deletion metrics.")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--attr-dir", default=None, help="experiments.run --save-attr folder of the axis stage")
    ap.add_argument("--axis", default=None, help="axis JSONL (per-image metrics) for the advantage correlations")
    ap.add_argument("--N", type=int, default=24, help="steps of the profiled paths")
    ap.add_argument("--prof-alphas", nargs="*", type=float, default=[0.1, 1, 10, 100, 1000])
    ap.add_argument("--prof-K", type=int, default=4, help="SPI paths profiled per α")
    ap.add_argument("--adv-alpha", type=float, default=100.0)
    ap.add_argument("--metrics", nargs="+",
                    default=["insdelp_black", "insdelp_blur", "insdel_black", "iou", "overlap", "road"])
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not args.summary_only:
        done = set()
        if os.path.exists(args.out):
            with open(args.out) as fh:
                for line in fh:
                    try:
                        done.add(json.loads(line)["image"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        dev, backbone, it, n = open_data(args)
        print(f"[analyze] {n} images, device={dev}, attr_dir={args.attr_dir}")
        for k, (idx, x, mask, name) in enumerate(it):
            if name in done:
                continue
            x = x.to(dev)
            model, cls, conf = target_model(backbone, x)
            oracle = Oracle(model, chunk=args.chunk)
            xb = MET.make_baseline(x, args.baseline)
            gen = torch.Generator().manual_seed(args.seed * 1_000_003 + idx)
            row = {"image": name, "index": idx, "class": cls, "conf": conf}
            row.update(analyse_image(args, oracle, x, xb, mask, name, gen))
            append_row(args.out, row)
            print(f"  [{k + 1}/{n}] {name} line_t90={row['line_t90']:.2f} line_area={row['line_area']:.2f}"
                  f" gapx_{akey(args.adv_alpha)}={row.get('gapx_' + akey(args.adv_alpha), float('nan')):.3f}")
    summarize(args.out, args.axis, args.adv_alpha, args.metrics)


if __name__ == "__main__":
    main()
