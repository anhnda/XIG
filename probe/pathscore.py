"""
PATHSCORE — eval riêng (tách khỏi P4): InsDel của TỪNG đường SPI trong K đường, đánh giá chéo giữa nhiều kiểu nền mask.

Câu hỏi
  1. Trong K đường SPI, đường có InsDel cao nhất / thấp nhất chênh bao nhiêu so với trung bình các đường?
  2. SPI (= trung bình cộng có dấu của K bản đồ) so với một đường điển hình?
  3. Đường tốt nhất theo metric a (InsDel với nền mask a) đứng ở đâu theo metric b? Còn hơn trung bình các đường không?
Metric: InsDel = AUC_ins − AUC_del (xác suất softmax lớp đích, 50 bước, hạng pixel Σ_c|A|) với nền mask (thứ thay cho pixel
bị xoá / chưa chèn) ∈ pathscore.mask_bases:
  blur  Gaussian blur của ảnh (kernel/sigma như P0–P3)     black  ảnh đen
  white ảnh trắng                                          grey   xám 0.5
  mean  màu trung bình (theo kênh) của chính ảnh
Baseline của TÍCH PHÂN đường luôn là ảnh đen (không đổi); chỉ nền mask của InsDel thay đổi.

Đường = đúng sampler SPI hiện tại (α, N từ paths.<sampling.spi_from>), attribution canonical canon_rand (M = sampling.M) của
TỪNG đường; ảnh = tiền tố split pilot; mô hình / target / tiền xử lý như P0–P3.
Cache theo ảnh: attr.pt (bản đồ từng đường, float32 — đổi metric không phải tính lại gradient) và scores.pt.
Output: <out>/pilot/pathscore/ và <out>/pilot/ATTR_PROBE_PATHSCORE_REPORT.md.
Lệnh: bash scripts/run.sh pathscore | pathscore-debug | pathscore-report
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import time

import numpy as np
import torch

from . import config as C
from . import sampling as SM
from .data import MEAN, STD
from .metrics import iou, make_base, spearman
from .pilot import Ctx, sync
from .quad import Counter
from .report import fmt, md
from .sampling_benchmark import Bank, Env, _cos, _h, _load, bench_fwd, provenance, settings

VERSION = 2
MASK_KINDS = ("blur", "black", "white", "grey", "mean")
PROXIES = {          # proxy tính được KHÔNG cần thêm forward; hướng chọn đặt trước (không chọn theo kết quả)
    "loo_cos": "max",            # cosine với trung bình K−1 đường còn lại (đồng thuận)
    "loo_spearman": "max",       # Spearman Σ_c|A| với trung bình K−1 đường còn lại
    "abs_resid_rel": "min",      # |ΣA − D| / |D| (completeness của riêng đường)
    "top_mass": "max",           # tỷ lệ Σ_c|A| nằm trong top `top_frac` pixel (độ tập trung)
    "l1_rel": "min",             # ‖A‖₁ / |D|
    "d_hat": "min",              # ∫(Z − t)² — mức exploration của đường
}
# màu cố định theo metric (bảng categorical mặc định, thứ tự cố định — không xoay vòng)
MCOLORS = {"blur": "#2a78d6", "black": "#eb6834", "white": "#1baf7a", "grey": "#eda100", "mean": "#e87ba4"}
ENS_COLOR, MUTED = "#0b0b0b", "#6b6a63"


# ─────────────────────────────────────────────────────────────────────────────
# cấu hình
# ─────────────────────────────────────────────────────────────────────────────

def pcfg(cfg: dict) -> dict:
    pc = cfg.get("pathscore")
    if not pc:
        raise SystemExit("config thiếu mục pathscore:")
    bad = [b for b in pc["mask_bases"] if b not in MASK_KINDS]
    if bad:
        raise SystemExit(f"pathscore.mask_bases không biết: {bad} (cho phép {MASK_KINDS})")
    if int(pc["K"]) < 2:
        raise SystemExit("pathscore.K ≥ 2")
    return pc


def out_dir(cfg: dict) -> str:
    d = os.path.join(C.out_dir(cfg, "pilot"), "pathscore")
    os.makedirs(d, exist_ok=True)
    return d


def report_path(cfg: dict) -> str:
    return os.path.join(C.out_dir(cfg, "pilot"), "ATTR_PROBE_PATHSCORE_REPORT.md")


def digests(cfg: dict, S: dict, pc: dict) -> dict:
    a = _h({"v": VERSION, "seed": cfg["seed"], "model": cfg["model"], "data": cfg["data"], "alpha": S["alpha"], "N": S["N"],
            "M": S["M"], "spi_from": S.get("spi_from"), "K": int(pc["K"]), "seeds": int(pc["seeds"])})
    s = _h({"attr": a, "bases": list(pc["mask_bases"]), "insdel": cfg["metrics"]["insdel"], "steps": int(S["insdel_steps"]),
            "top_frac": float(pc.get("top_frac", 0.05)), "proxies": PROXIES})
    return {"attr": a, "scores": s}


def mask_base(x: torch.Tensor, kind: str, ic: dict) -> torch.Tensor:
    """Nền mask cho InsDel, trong không gian đã chuẩn hoá ImageNet."""
    if kind in ("black", "blur"):
        return make_base(x, kind, ic["blur_kernel"], ic["blur_sigma"])
    m = torch.tensor(MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    sd = torch.tensor(STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    if kind == "white":
        return ((1.0 - m) / sd).expand_as(x).clone()
    if kind == "grey":
        return ((0.5 - m) / sd).expand_as(x).clone()
    if kind == "mean":
        return x.mean(dim=(2, 3), keepdim=True).expand_as(x).clone()
    raise ValueError(kind)


# ─────────────────────────────────────────────────────────────────────────────
# tính
# ─────────────────────────────────────────────────────────────────────────────

def compute_attr(E: Env, pc: dict) -> dict:
    """Bản đồ canonical của từng đường SPI: A (seeds, K, C, H, W) float32 CPU. Seed r dùng đúng khoá đường S0 của P4 (đường
    k < K_max của P4 trùng nhau) — chỉ để tái lập, không phụ thuộc P4."""
    K, R = int(pc["K"]), int(pc["seeds"])
    As, dh = [], []
    for r in range(R):
        bank = Bank(E, r, "ens", n_paths=K)
        Ar, dr = [], []
        for k in range(K):
            p = bank.eval_path(("spi", k))
            Ar.append(p["A"].float().cpu())
            dr.append(p["d_hat"])
        As.append(torch.stack(Ar))
        dh.append(dr)
    sync()
    return {"A": torch.stack(As), "d_hat": torch.tensor(dh, dtype=torch.float64)}


def compute_scores(E: Env, attr: dict, pc: dict) -> dict:
    S = E.S
    ic = E.ctx.cfg["metrics"]["insdel"]
    bases = {b: mask_base(E.x, b, ic) for b in pc["mask_bases"]}
    steps = int(S["insdel_steps"])
    tf = float(pc.get("top_frac", 0.05))
    kk = max(1, int(tf * E.H * E.W))
    R, K = attr["A"].shape[:2]
    paths, ens = [], []
    for r in range(R):
        As = attr["A"][r].to(E.ctx.dev).double()
        tot = As.sum(0)
        spi = tot / K
        er = {"image": E.image, "seed": r, "K": K}
        for b, base in bases.items():
            q = SM.insdel_full(E.ctx.bb, E.cls, E.x, base, spi.float(), steps, E.ins_batch)
            er[f"ins_{b}"], er[f"del_{b}"], er[f"insdel_{b}"] = q["ins"], q["del"], q["diff"]
        ens.append(er)
        for k in range(K):
            A = As[k]
            loo = (tot - A) / (K - 1)
            am = A.abs().sum(0).flatten()
            s = float(A.sum())
            row = {"image": E.image, "seed": r, "path": k, "K": K, "sumA": s, "D": E.D,
                   "abs_resid_rel": abs(s - E.D) / abs(E.D) if E.D else float("nan"),
                   "l1_rel": float(A.abs().sum()) / abs(E.D) if E.D else float("nan"),
                   "top_mass": float(am.topk(kk).values.sum() / am.sum().clamp_min(1e-300)),
                   "loo_cos": _cos(A, loo), "loo_spearman": spearman(A.float(), loo.float()),
                   "d_hat": float(attr["d_hat"][r, k]), "iou": iou(A.float(), E.mask, E.perm)}
            for b, base in bases.items():
                q = SM.insdel_full(E.ctx.bb, E.cls, E.x, base, A.float(), steps, E.ins_batch)
                row[f"ins_{b}"], row[f"del_{b}"], row[f"insdel_{b}"] = q["ins"], q["del"], q["diff"]
            paths.append(row)
        del As
    sync()
    return {"paths": paths, "ens": ens}


def estimate(S: dict, pc: dict, per_eval: float, per_fwd: float) -> dict:
    K, R, nb = int(pc["K"]), int(pc["seeds"]), len(pc["mask_bases"])
    grad = K * R * S["N"] * S["M"]
    fwd = (K + 1) * R * nb * 2 * (int(S["insdel_steps"]) + 1)
    n = int(pc["n_images"])
    return {"grad_per_image": grad, "fwd_per_image": fwd, "n_images": n,
            "minutes_full": n * (grad * per_eval + fwd * per_fwd) / 60, "minutes_scores_only": n * fwd * per_fwd / 60}


def process_image(ctx: Ctx, S: dict, pc: dict, out: str, idx: int, dg: dict) -> dict:
    image = f"seg{idx:05d}"
    cdir = os.path.join(out, "cache", image)
    os.makedirs(cdir, exist_ok=True)
    sp, ap = os.path.join(cdir, "scores.pt"), os.path.join(cdir, "attr.pt")
    if _load(sp, dg["scores"]) is not None:
        print(f"    [pathscore] {image}: đã có, bỏ qua")
        return {"grad": 0, "fwd": 0}
    E = Env(ctx, S, idx)
    n0, f0, t0 = Counter.n, SM.FwdCounter.n, time.time()
    attr = _load(ap, dg["attr"])
    if attr is None:
        attr = compute_attr(E, pc)
        attr["digest"] = dg["attr"]
        torch.save(attr, ap)
    sc = compute_scores(E, attr, pc)
    sc.update({"digest": dg["scores"], "info": E.info(), "x": E.it.x.half(), "mask": E.mask.cpu()})
    torch.save(sc, sp)
    print(f"    [pathscore] {image}: {Counter.n - n0} grad eval, {SM.FwdCounter.n - f0} forward, {time.time() - t0:.0f}s", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"grad": Counter.n - n0, "fwd": SM.FwdCounter.n - f0}


def write_csvs(out: str, names: list) -> None:
    import pandas as pd
    P, Ens, I = [], [], []
    for im in names:
        p = os.path.join(out, "cache", im, "scores.pt")
        if os.path.exists(p):
            o = torch.load(p, weights_only=False)
            P += o["paths"]
            Ens += o["ens"]
            I.append(o["info"])
    for rows, name in ((P, "pathscore_paths"), (Ens, "pathscore_spi"), (I, "pathscore_images")):
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out, f"{name}.csv"), index=False)
            print(f"[pathscore] {name}.csv: {len(rows)} dòng")


def run(ctx: Ctx) -> None:
    from .pilot import bench
    cfg = ctx.cfg
    S = settings(cfg)
    pc = pcfg(cfg)
    out = out_dir(cfg)
    dg = digests(cfg, S, pc)
    idxs = ctx.sp["pilot"][:int(pc["n_images"])]
    if len(idxs) < int(pc["n_images"]):
        raise SystemExit(f"split pilot chỉ có {len(idxs)} ảnh < pathscore.n_images")
    b = bench(ctx)
    pf = bench_fwd(ctx)
    est = estimate(S, pc, b["per_eval"], pf)
    print(f"[pathscore] K = {pc['K']} đường SPI × {pc['seeds']} seed, {len(idxs)} ảnh, nền mask {pc['mask_bases']}: "
          f"≈ {est['minutes_full']:.1f} phút nếu chưa có gì ({est['minutes_scores_only']:.1f} phút nếu bản đồ đã có cache)")
    lim = float(pc.get("budget_minutes", 30))
    if est["minutes_full"] > lim and pc.get("enforce_budget", True):
        raise SystemExit(f"[pathscore] ước tính {est['minutes_full']:.1f} phút > pathscore.budget_minutes = {lim}")
    t0, n0, f0 = time.time(), Counter.n, SM.FwdCounter.n
    for j, idx in enumerate(idxs):
        print(f"  [pathscore] ảnh {j + 1}/{len(idxs)} seg{idx:05d}", flush=True)
        process_image(ctx, S, pc, out, idx, dg)
    names = [f"seg{i:05d}" for i in idxs]
    write_csvs(out, names)
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"time": _dt.datetime.now().isoformat(timespec="seconds"), "images": names, "indices": [int(i) for i in idxs],
                   "pathscore": pc, "spi": {"alpha": S["alpha"], "N": S["N"], "M": S["M"], "spi_from": S.get("spi_from")},
                   "insdel": {"steps": S["insdel_steps"], **cfg["metrics"]["insdel"]}, "estimate": est,
                   "bench": {**b, "per_fwd": pf}, "minutes_this_run": (time.time() - t0) / 60,
                   "grad_evals_this_run": Counter.n - n0, "fwd_evals_this_run": SM.FwdCounter.n - f0, "digests": dg,
                   "global_seed": cfg["seed"], "debug": cfg["_debug"], "provenance": provenance(cfg)},
                  fh, indent=1, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# phân tích (chỉ từ CSV)
# ─────────────────────────────────────────────────────────────────────────────

def best_of_m(scores, m: int) -> float:
    """E[max của m đường lấy ngẫu nhiên không hoàn lại từ K đường] — chính xác theo thống kê thứ tự."""
    x = sorted(scores)
    n = len(x)
    if m >= n:
        return x[-1]
    return sum(x[i] * math.comb(i, m - 1) for i in range(m - 1, n)) / math.comb(n, m)


def _rank_desc(v: np.ndarray) -> np.ndarray:
    """Hạng giảm dần 1..K (1 = cao nhất), hoà lấy trung bình."""
    import pandas as pd
    return pd.Series(v).rank(ascending=False, method="average").values


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank_desc(a), _rank_desc(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def per_group_stats(paths, spi, bases: list) -> dict:
    """Mỗi (ảnh, seed): ma trận điểm K × |metric| → thống kê chéo. Trả các bảng dài (một dòng / (ảnh, seed, ...))."""
    import pandas as pd
    mets = [f"insdel_{b}" for b in bases]
    if spi is None:
        spi = pd.DataFrame(columns=["image", "seed"] + mets)
    spread, cross, corr, bom, prox = [], [], [], [], []
    for (im, r), g in paths.groupby(["image", "seed"]):
        g = g.sort_values("path")
        X = g[mets].values                      # (K, nm)
        K = X.shape[0]
        mu = X.mean(0)
        e = spi[(spi.image == im) & (spi.seed == r)]
        ens = e[mets].values[0] if len(e) else np.full(len(mets), np.nan)
        R = np.stack([_rank_desc(X[:, j]) for j in range(len(mets))], 1)
        for j, b in enumerate(bases):
            col = X[:, j]
            spread.append({"image": im, "seed": r, "metric": b, "K": K, "mean_path": mu[j], "sd_path": col.std(ddof=1),
                           "best_minus_mean": col.max() - mu[j], "mean_minus_worst": mu[j] - col.min(),
                           "spi": ens[j], "spi_minus_mean": ens[j] - mu[j], "best_minus_spi": col.max() - ens[j],
                           "frac_paths_above_spi": float((col > ens[j]).mean())})
            mm = 1
            while mm <= K:
                bom.append({"image": im, "seed": r, "metric": b, "m": mm, "gain": best_of_m(col, mm) - mu[j]})
                mm *= 2
            for px, direction in PROXIES.items():
                v = g[px].values
                if not np.all(np.isfinite(v)):
                    continue
                pick = int(np.argmax(v) if direction == "max" else np.argmin(v))
                prox.append({"image": im, "seed": r, "metric": b, "proxy": px, "rho": _spearman(v, col),
                             "select_gain": col[pick] - mu[j], "oracle_gain": col.max() - mu[j], "spi_gain": ens[j] - mu[j]})
        for a, ba in enumerate(bases):
            ib, iw = int(np.argmax(X[:, a])), int(np.argmin(X[:, a]))
            for j, bb in enumerate(bases):
                cross.append({"image": im, "seed": r, "select_on": ba, "eval_on": bb, "K": K,
                              "best_gain": X[ib, j] - mu[j], "best_rank": R[ib, j],
                              "worst_gain": X[iw, j] - mu[j], "worst_rank": R[iw, j],
                              "spi_gain": ens[j] - mu[j], "sd_path": X[:, j].std(ddof=1)})
                corr.append({"image": im, "seed": r, "a": ba, "b": bb, "spearman": _spearman(X[:, a], X[:, j])})
    return {k: pd.DataFrame(v) for k, v in (("spread", spread), ("cross", cross), ("corr", corr), ("bestofm", bom),
                                             ("proxy", prox))}


def _boot(v, B: int, seed: int) -> tuple:
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return (np.nan,) * 3
    if len(v) == 1:
        return float(v[0]), np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def summarize(df, keys: list, cols: list, B: int, seed: int):
    """Trung bình seed trong ảnh trước, rồi trung bình + CI bootstrap theo ảnh."""
    import pandas as pd
    pim = df.groupby(keys + ["image"])[cols].mean().reset_index()
    rows = []
    for kv, g in pim.groupby(keys):
        kv = kv if isinstance(kv, tuple) else (kv,)
        r = dict(zip(keys, kv))
        r["n_images"] = len(g)
        for c in cols:
            m, lo, hi = _boot(g[c].values, B, seed)
            r[c], r[f"{c}_lo95"], r[f"{c}_hi95"] = m, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# hình
# ─────────────────────────────────────────────────────────────────────────────

def _matrix(ax, M, labels, title, cmap, vmin, vmax, fmt_s, star=None, diag_box=True, cbar_label=""):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    n = len(labels)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("chấm theo (eval_on)"); ax.set_ylabel("chọn theo (select_on)")
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            rgba = im.cmap(im.norm(v))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt = fmt_s.format(v) + ("*" if star is not None and star[i, j] else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if lum < 0.45 else "#0b0b0b")
    if diag_box:
        import matplotlib.patches as mpatches
        for i in range(n):
            ax.add_patch(mpatches.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#0b0b0b", lw=1.6))
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=8)


def make_figures(out: str, bases: list, cs, cr, sp, bom, paths, K: int) -> list:
    from .sampling_report import _plt
    plt = _plt()
    files = []
    n = len(bases)

    def mat(df, col):
        M = np.full((n, n), np.nan)
        for i, a in enumerate(bases):
            for j, b in enumerate(bases):
                q = df[(df.select_on == a) & (df.eval_on == b)] if "select_on" in df else df[(df.a == a) & (df.b == b)]
                if len(q):
                    M[i, j] = float(q[col].iloc[0])
        return M

    # 1. ba ma trận
    try:
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
        G = mat(cs, "best_gain")
        star = (mat(cs, "best_gain_lo95") > 0) | (mat(cs, "best_gain_hi95") < 0)
        v = np.nanmax(np.abs(G))
        _matrix(ax[0], G, bases, "Đường tốt nhất theo hàng, chấm theo cột:\nInsDel − trung bình các đường (* = CI 95% không chứa 0)",
                "RdBu", -v, v, "{:+.3f}", star, cbar_label="ΔInsDel so với trung bình đường")
        Rk = mat(cs, "best_rank")
        mid = (K + 1) / 2
        _matrix(ax[1], Rk, bases, f"Hạng của đường đó theo cột (1 = tốt nhất / {K};\nngẫu nhiên = {mid:g})",
                "RdBu_r", 1, K, "{:.1f}", None, cbar_label="hạng trung bình (xanh = tốt)")
        Cm = mat(cr, "spearman")
        _matrix(ax[2], Cm, bases, f"Tương quan hạng điểm của {K} đường\ngiữa hai metric (Spearman)", "RdBu",
                -1, 1, "{:.2f}", None, diag_box=False, cbar_label="Spearman")
        fig.tight_layout()
        f = os.path.join(out, "pathscore_cross_matrix.png")
        fig.savefig(f, dpi=150); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[pathscore plots] matrix: {type(ex).__name__}: {ex}")

    # 2. dot-plot: mỗi cột = metric chấm; mỗi chấm = metric dùng để chọn; kèm SPI và 0
    try:
        fig, ax = plt.subplots(figsize=(max(8, 1.9 * n + 3), 4.6))
        w = 0.13
        for j, b in enumerate(bases):
            for i, a in enumerate(bases):
                q = cs[(cs.select_on == a) & (cs.eval_on == b)]
                if not len(q):
                    continue
                q = q.iloc[0]
                x = j + (i - (n - 1) / 2) * w
                own = a == b
                ax.errorbar(x, q.best_gain, yerr=[[q.best_gain - q.best_gain_lo95], [q.best_gain_hi95 - q.best_gain]],
                            fmt="o", ms=9 if own else 7, mfc="white" if own else MCOLORS[a], mec=MCOLORS[a], mew=2,
                            ecolor=MCOLORS[a], elinewidth=1.4, capsize=0,
                            label=f"chọn theo {a}" if j == 0 else None, zorder=3)
            s = sp[sp.metric == b]
            if len(s):
                s = s.iloc[0]
                ax.errorbar(j + (n / 2 + 0.3) * w, s.spi_minus_mean, yerr=[[s.spi_minus_mean - s.spi_minus_mean_lo95],
                            [s.spi_minus_mean_hi95 - s.spi_minus_mean]], fmt="D", ms=7, color=ENS_COLOR, capsize=0,
                            label=f"SPI {K} đường" if j == 0 else None, zorder=3)
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.text(n - 0.45, 0, " trung bình\n các đường", va="center", fontsize=7, color=MUTED)
        ax.set_xticks(range(n)); ax.set_xticklabels([f"chấm theo\n{b}" for b in bases])
        ax.set_ylabel("InsDel − trung bình các đường (CI 95% theo ảnh)")
        ax.set_title("Đường tốt nhất theo một metric, chấm theo từng metric. Chấm rỗng = cùng metric (oracle, lạc quan).", fontsize=9)
        ax.legend(fontsize=7, ncol=n + 1, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
        ax.set_xlim(-0.6, n - 0.2)
        fig.tight_layout()
        f = os.path.join(out, "pathscore_cross_dots.png")
        fig.savefig(f, dpi=150); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[pathscore plots] dots: {type(ex).__name__}: {ex}")

    # 3. ví dụ: hạng của từng đường theo từng metric, 4 ảnh
    try:
        ims = list(dict.fromkeys(paths.image))[:4]
        r0 = paths.seed.min()
        mets = [f"insdel_{b}" for b in bases]
        fig, ax = plt.subplots(1, len(ims), figsize=(3.2 * len(ims) + 1, 5.2), squeeze=False)
        for t, im in enumerate(ims):
            g = paths[(paths.image == im) & (paths.seed == r0)].sort_values("path")
            X = g[mets].values
            Rk = np.stack([_rank_desc(X[:, j]) for j in range(n)], 1)
            order = np.argsort(Rk[:, 0])                     # sắp theo hạng của metric đầu
            a = ax[0, t]
            img = a.imshow(Rk[order], cmap="Blues_r", vmin=1, vmax=K, aspect="auto")
            a.set_xticks(range(n)); a.set_xticklabels(bases, rotation=30)
            a.set_yticks(range(len(order))); a.set_yticklabels([f"đường {int(g.path.values[o])}" for o in order], fontsize=6)
            a.set_title(f"{im} (seed {r0})", fontsize=8)
            a.grid(False)
            for j in range(n):
                i = int(np.where(Rk[order, j] == Rk[:, j].min())[0][0])
                a.scatter([j], [i], marker="*", s=90, color=MCOLORS[bases[j]], edgecolor="white", linewidth=0.8, zorder=3)
        cb = fig.colorbar(img, ax=ax[0, :].tolist(), fraction=0.02, pad=0.02)
        cb.set_label(f"hạng (1 = tốt nhất / {K})", fontsize=8)
        fig.suptitle(f"Hạng của từng đường theo từng metric (hàng sắp theo {bases[0]}; ★ = đường tốt nhất của cột)", fontsize=9)
        f = os.path.join(out, "pathscore_rank_examples.png")
        fig.savefig(f, dpi=150); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[pathscore plots] examples: {type(ex).__name__}: {ex}")

    # 4. best-of-m (oracle, cùng metric) so với SPI
    try:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for b in bases:
            q = bom[bom.metric == b].sort_values("m")
            ax.plot(q.m, q.gain, marker="o", color=MCOLORS[b], label=f"{b}: tốt nhất trong m đường (oracle)")
            s = sp[sp.metric == b]
            if len(s):
                ax.scatter([K * 1.35], [s.spi_minus_mean.iloc[0]], marker="D", s=40, color=MCOLORS[b], edgecolor=ENS_COLOR, zorder=3)
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(f"m (số đường để chọn); ◆ bên phải = SPI {K} đường")
        ax.set_ylabel("InsDel − trung bình các đường")
        ax.legend(fontsize=7)
        fig.tight_layout()
        f = os.path.join(out, "pathscore_bestofm.png")
        fig.savefig(f, dpi=150); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[pathscore plots] bestofm: {type(ex).__name__}: {ex}")
    return files


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo
# ─────────────────────────────────────────────────────────────────────────────

def write_report(cfg: dict) -> str:
    import pandas as pd
    from .report import _read
    out = out_dir(cfg)
    rp = report_path(cfg)
    S = settings(cfg)
    pc = pcfg(cfg)
    st = S.get("stats", {}) or {}
    B, seed = int(st.get("bootstrap", 10000)), int(st.get("seed", 4))
    paths, spi = _read(out, "pathscore_paths"), _read(out, "pathscore_spi")
    mp = os.path.join(out, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    L = ["# ATTR_PROBE_PATHSCORE_REPORT — InsDel của từng đường SPI, đánh giá chéo giữa các nền mask\n"]
    if meta.get("debug") or cfg.get("_debug"):
        L.append("> **DEBUG** — chỉ để kiểm pipeline, không dùng để kết luận.\n")
    if paths is None or not len(paths):
        L.append("_Chưa có kết quả — `bash scripts/run.sh pathscore`._\n")
        with open(rp, "w") as fh:
            fh.write("\n".join(L) + "\n")
        print(f"[pathscore report] {rp} (trống)")
        return rp
    bases = [b for b in pc["mask_bases"] if f"insdel_{b}" in paths]
    K = int(paths.K.iloc[0])
    st_ = per_group_stats(paths, spi, bases)
    sp = summarize(st_["spread"], ["metric"], ["mean_path", "sd_path", "best_minus_mean", "mean_minus_worst", "spi",
                                                "spi_minus_mean", "best_minus_spi", "frac_paths_above_spi"], B, seed)
    sp["metric"] = pd.Categorical(sp.metric, bases, ordered=True)
    sp = sp.sort_values("metric")
    cs = summarize(st_["cross"], ["select_on", "eval_on"], ["best_gain", "best_rank", "worst_gain", "worst_rank", "spi_gain"], B, seed)
    cr = summarize(st_["corr"], ["a", "b"], ["spearman"], B, seed)
    bom = st_["bestofm"].groupby(["metric", "m", "image"]).gain.mean().groupby(["metric", "m"]).mean().reset_index()
    px = summarize(st_["proxy"], ["metric", "proxy"], ["rho", "select_gain", "oracle_gain", "spi_gain"], B, seed) \
        if len(st_["proxy"]) else None
    for name, df in (("pathscore_spread", sp), ("pathscore_cross", cs), ("pathscore_corr", cr), ("pathscore_bestofm", bom),
                     ("pathscore_proxy", px)):
        if df is not None and len(df):
            df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    files = make_figures(out, bases, cs, cr, sp, bom, paths, K)
    n_img, n_seed = int(paths.image.nunique()), int(paths.seed.nunique())

    # tóm tắt đầu báo cáo
    off = cs[cs.select_on != cs.eval_on]
    diag = cs[cs.select_on == cs.eval_on]
    L.append("## Tóm tắt\n")
    L.append(f"- {n_img} ảnh × {n_seed} seed × K = {K} đường SPI; metric = InsDel với nền mask {bases}.")
    L.append(f"- Đường tốt nhất **trên chính metric dùng để chọn**: hơn trung bình các đường {fmt(diag.best_gain.mean())} "
             f"(trung bình {len(diag)} metric).")
    L.append(f"- Cùng đường đó **chấm theo metric khác**: hơn trung bình {fmt(off.best_gain.mean())} (trung bình {len(off)} cặp; "
             f"min {fmt(off.best_gain.min())}, max {fmt(off.best_gain.max())}); hạng trung bình {fmt(off.best_rank.mean())}/{K} "
             f"(ngẫu nhiên = {(K + 1) / 2:g}). Số cặp có CI 95% > 0: {int((off.best_gain_lo95 > 0).sum())}/{len(off)}.")
    L.append(f"- SPI {K} đường (trung bình cộng các bản đồ) hơn trung bình các đường {fmt(sp.spi_minus_mean.mean())} "
             f"(trung bình các metric; từng metric ở mục 2).")
    L.append(f"- Tương quan hạng điểm của {K} đường giữa hai metric khác nhau: trung bình "
             f"{fmt(cr[cr.a != cr.b].spearman.mean())}.\n")
    L.append("![ma trận](pathscore/pathscore_cross_matrix.png)\n")
    L.append("![dot](pathscore/pathscore_cross_dots.png)\n")

    L.append("## 1. Thiết lập\n")
    L.append(f"- Đường = đúng sampler SPI hiện tại (α = {S['alpha']:g}, N = {S['N']}); attribution canonical (canon_rand, M = {S['M']}) "
             "của TỪNG đường; SPI = trung bình cộng có dấu của K bản đồ. Baseline của tích phân luôn là ảnh đen.")
    L.append("- InsDel = AUC_ins − AUC_del, 50 bước, xác suất softmax lớp dự đoán, hạng pixel Σ_c|A| (quy ước P0–P3). Nền mask (thay cho "
             "pixel chưa chèn / đã xoá): blur = Gaussian blur của ảnh; black / white / grey = ảnh đen / trắng / xám 0,5; mean = màu "
             "trung bình theo kênh của chính ảnh.")
    L.append(f"- ResNet-50 IMAGENET1K_V2; {n_img} ảnh tiền tố split pilot; thống kê: ảnh là đơn vị (trung bình seed trước), CI 95% "
             f"bootstrap theo ảnh ({B} lần).")
    if meta:
        pr = meta.get("provenance", {}) or {}
        L.append(f"- Ảnh: {meta.get('indices')}; GPU `{pr.get('gpu')}`; torch {pr.get('torch')}; hash "
                 f"`{str((meta.get('digests') or {}).get('scores', ''))[:12]}`; seed toàn cục {meta.get('global_seed')}.")
    L.append("- 'Đường tốt nhất' chọn bằng CHÍNH metric trên chính ảnh = oracle (cận trên lạc quan, không phải phương pháp).\n")

    L.append("## 2. Từng metric: đường tốt nhất / kém nhất / SPI so với trung bình các đường\n")
    L.append(md(sp, [c for c in ("metric", "n_images", "mean_path", "sd_path", "best_minus_mean", "best_minus_mean_lo95",
                                 "best_minus_mean_hi95", "mean_minus_worst", "mean_minus_worst_lo95", "mean_minus_worst_hi95",
                                 "spi", "spi_minus_mean", "spi_minus_mean_lo95", "spi_minus_mean_hi95", "best_minus_spi",
                                 "frac_paths_above_spi") if c in sp]))

    L.append("## 3. Đánh giá chéo: đường tốt nhất theo metric này, chấm theo metric kia\n")
    L.append("`best_gain` = InsDel(metric chấm) của đường tốt nhất theo metric chọn − trung bình các đường (metric chấm). `best_rank` = hạng "
             f"của nó theo metric chấm (1..{K}; ngẫu nhiên {(K + 1) / 2:g}). `worst_*` = tương tự cho đường kém nhất. `spi_gain` = SPI "
             "− trung bình (metric chấm), để so. Đường chéo = oracle.\n")
    L.append("Ma trận `best_gain` (hàng = chọn theo, cột = chấm theo; * = CI 95% không chứa 0):\n")
    tab = []
    for a in bases:
        r = {"chọn theo \\ chấm theo": a}
        for b in bases:
            q = cs[(cs.select_on == a) & (cs.eval_on == b)]
            if len(q):
                q = q.iloc[0]
                sig = "*" if (q.best_gain_lo95 > 0 or q.best_gain_hi95 < 0) else ""
                r[b] = f"{q.best_gain:+.3f}{sig}"
        tab.append(r)
    sp_row = {"chọn theo \\ chấm theo": f"(SPI {K} đường)"}
    for b in bases:
        q = sp[sp.metric == b]
        if len(q):
            sp_row[b] = f"{q.spi_minus_mean.iloc[0]:+.3f}"
    tab.append(sp_row)
    L.append(md(pd.DataFrame(tab)))
    L.append("Ma trận hạng trung bình `best_rank`:\n")
    tab = []
    for a in bases:
        r = {"chọn theo \\ chấm theo": a}
        for b in bases:
            q = cs[(cs.select_on == a) & (cs.eval_on == b)]
            if len(q):
                r[b] = f"{q.best_rank.iloc[0]:.1f}"
        tab.append(r)
    L.append(md(pd.DataFrame(tab)))
    L.append("Đầy đủ với CI:\n")
    L.append(md(cs, [c for c in ("select_on", "eval_on", "n_images", "best_gain", "best_gain_lo95", "best_gain_hi95", "best_rank",
                                 "worst_gain", "worst_gain_lo95", "worst_gain_hi95", "worst_rank", "spi_gain") if c in cs]))

    L.append(f"## 4. Tương quan hạng điểm của {K} đường giữa các metric\n")
    tab = []
    for a in bases:
        r = {"": a}
        for b in bases:
            q = cr[(cr.a == a) & (cr.b == b)]
            if len(q):
                r[b] = f"{q.spearman.iloc[0]:.2f}"
        tab.append(r)
    L.append(md(pd.DataFrame(tab)))
    L.append("Gần 0 nghĩa là thứ tự 'đường nào tốt hơn' của metric này không nói gì về metric kia.\n")

    L.append("## 5. Best-of-m (oracle, cùng metric)\n")
    L.append(md(bom.pivot(index="metric", columns="m", values="gain").reset_index()))
    L.append("![best-of-m](pathscore/pathscore_bestofm.png)\n")

    if px is not None and len(px):
        L.append("## 6. Proxy chọn đường không tốn thêm forward\n")
        L.append("Hướng chọn đặt trước (loo_cos / loo_spearman / top_mass → lớn nhất; abs_resid_rel / l1_rel / d_hat → nhỏ nhất). "
                 "`rho` = Spearman proxy–điểm trong một ảnh; `select_gain` = đường proxy chọn − trung bình; so với `spi_gain`.\n")
        L.append(md(px, [c for c in ("metric", "proxy", "n_images", "rho", "rho_lo95", "rho_hi95", "select_gain", "select_gain_lo95",
                                     "select_gain_hi95", "oracle_gain", "spi_gain") if c in px]))

    L.append("## 7. Ví dụ từng ảnh\n")
    L.append("![ví dụ](pathscore/pathscore_rank_examples.png)\n")
    L.append("## Tệp\n")
    for f in sorted(os.listdir(out)):
        if f.endswith(".csv") or f.endswith(".png") or f == "meta.json":
            L.append(f"- `pathscore/{f}`")
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[pathscore report] {rp}; {len(files)} hình")
    return rp
