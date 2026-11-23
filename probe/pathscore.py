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

Metric bổ sung bên cạnh InsDel (pathscore.extra_metrics; định nghĩa ở probe/pathscore_metrics.py), cho từng đường và cho SPI:
  định vị theo mask GT   iou, mass_in_mask, energy_mask, pointing, prec_top, auroc_mask, ap_mask      (không forward)
  faithfulness           road (= road_lerf − road_morf), road_morf, road_lerf, sens_n, infid          (forward)
Metric "nhỏ = tốt" (road_morf, infid) được ĐỔI DẤU trong mọi phân tích chéo / hạng / best-of-m / proxy ⇒ "gain > 0" luôn là tốt hơn.
Đơn vị z = chênh lệch chia cho độ lệch chuẩn giữa K đường trong ảnh (để so được giữa các metric khác đơn vị).

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
from . import pathscore_metrics as PM
from . import sampling as SM
from .data import MEAN, STD
from .metrics import iou, make_base, spearman
from .ctx import Ctx, sync
from .quad import Counter
from .report import fmt, md
from .pathbank import Bank, Env, _cos, _h, _load, bench_fwd, provenance, settings

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
SCORES_VERSION = 3   # đổi khi đổi cách tính điểm (scores.pt); KHÔNG đụng digest bản đồ (attr.pt)
# metric bổ sung: cột → (mô tả, hướng: +1 lớn = tốt / −1 nhỏ = tốt, nhóm). Thứ tự = thứ tự trong báo cáo.
EXTRA = {
    "iou": ("IoU top-|mask| với mask", 1, "loc"),
    "mass_in_mask": ("tỷ lệ Σ_c|A| trong mask", 1, "loc"),
    "energy_mask": ("tỷ lệ Σ_c A² trong mask", 1, "loc"),
    "pointing": ("pointing game (argmax trong mask nới)", 1, "loc"),
    "prec_top": ("precision của top_frac pixel", 1, "loc"),
    "auroc_mask": ("AUC-ROC của Σ_c|A| so với mask", 1, "loc"),
    "ap_mask": ("average precision so với mask", 1, "loc"),
    "road": ("ROAD: LeRF − MoRF", 1, "faith"),
    "road_morf": ("ROAD MoRF (xác suất, nhỏ = tốt)", -1, "faith"),
    "road_lerf": ("ROAD LeRF (xác suất)", 1, "faith"),
    "sens_n": ("sensitivity-n (Pearson Σ_S A ~ ΔF_S)", 1, "faith"),
    "infid": ("infidelity chuẩn hoá (nhỏ = tốt)", -1, "faith"),
}
GROUPS = {"insdel": "InsDel (các nền mask)", "loc": "định vị theo mask", "faith": "faithfulness khác"}
EXTRA_DEFAULT = {
    "loc": True,
    "pointing_tol": 15,
    "road": {"enabled": True, "fracs": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "noise": 0.01, "iters": 1000, "tol": 1.0e-5},
    "subset": {"enabled": True, "n": 96, "sizes": [16, 32, 56]},
}
# màu cố định theo metric (bảng categorical mặc định, thứ tự cố định — không xoay vòng)
MCOLORS = {"blur": "#2a78d6", "black": "#eb6834", "white": "#1baf7a", "grey": "#eda100", "mean": "#e87ba4"}
CAT8 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
# metric bổ sung: mỗi nhóm vẽ ở panel riêng ⇒ trong một panel các màu theo đúng thứ tự slot, mỗi metric giữ một màu cố định
XCOLORS = {**{m: CAT8[i] for i, m in enumerate([k for k, v in EXTRA.items() if v[2] == "loc"])},
           **{m: CAT8[i] for i, m in enumerate([k for k, v in EXTRA.items() if v[2] == "faith"])}}
ENS_COLOR, MUTED = "#0b0b0b", "#6b6a63"


def mcol(k: str) -> str:
    """Khoá metric → cột CSV: nền mask InsDel giữ tên ngắn (blur, black, …), metric bổ sung là chính tên cột."""
    return f"insdel_{k}" if k in MASK_KINDS else k


def mdir(k: str) -> int:
    return int(EXTRA[k][1]) if k in EXTRA else 1


def mgroup(k: str) -> str:
    return EXTRA[k][2] if k in EXTRA else "insdel"


def mcolor(k: str) -> str:
    return MCOLORS.get(k) or XCOLORS.get(k, ENS_COLOR)


def xcfg(pc: dict) -> dict:
    return C._merge(EXTRA_DEFAULT, pc.get("extra_metrics") or {})


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
    xm = xcfg(pc)
    if (xm.get("road") or {}).get("enabled") and not all(0 < float(f) < 1 for f in xm["road"]["fracs"]):
        raise SystemExit("pathscore.extra_metrics.road.fracs phải nằm trong (0, 1)")
    if (xm.get("subset") or {}).get("enabled") and int(xm["subset"]["n"]) < 3:
        raise SystemExit("pathscore.extra_metrics.subset.n ≥ 3")
    bad = [m for m in (pc.get("report_extra") or []) if m not in EXTRA]
    if bad:
        raise SystemExit(f"pathscore.report_extra không biết: {bad} (cho phép {list(EXTRA)})")
    return pc


def out_dir(cfg: dict) -> str:
    d = os.path.join(C.out_dir(cfg, "pilot"), "pathscore")
    os.makedirs(d, exist_ok=True)
    return d


def report_path(cfg: dict) -> str:
    return os.path.join(C.out_dir(cfg, "pilot"), "ATTR_PROBE_PATHSCORE_REPORT.md")


def digests(cfg: dict, S: dict, pc: dict) -> dict:
    a = _h({"v": VERSION, "seed": cfg["seed"], "model": cfg["model"], "data": cfg["data"], "alpha": S["alpha"], "N": S["N"],
            "M": S["M"], "spi_from": S.get("spi_from"), "K": int(pc["K"]), "seeds": int(pc["seeds"]),
            **({"rule": S["rule"]} if S.get("rule", "rand") != "rand" else {})})
    s = _h({"attr": a, "bases": list(pc["mask_bases"]), "insdel": cfg["metrics"]["insdel"], "steps": int(S["insdel_steps"]),
            "top_frac": float(pc.get("top_frac", 0.05)), "proxies": PROXIES, "sv": SCORES_VERSION, "extra": xcfg(pc)})
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


def prepare_extra(E: Env, pc: dict) -> dict:
    """Phần chỉ phụ thuộc ảnh của metric bổ sung (nhiễu ROAD, các ô vuông S và ΔF_S): tính MỘT lần, dùng chung cho mọi đường."""
    xm = xcfg(pc)
    P = {"xm": xm}
    rc = xm.get("road") or {}
    if rc.get("enabled"):
        fr = [float(f) for f in rc["fracs"]]
        g = E.cgen("pathscore-road", E.image)
        P["fracs"] = fr
        P["road_z"] = torch.randn((2 * len(fr), E.x.shape[1], E.H, E.W), generator=g, dtype=torch.float64).to(E.ctx.dev)
        P["road_sd"] = PM.road_noise_sd(float(rc["noise"]), E.ctx.dev)
        bb, cls = E.ctx.bb, E.cls

        def prob(t):
            SM.FwdCounter.n += int(t.shape[0])
            return torch.softmax(bb(t), dim=-1)[:, cls]
        P["prob"] = prob
    sc = xm.get("subset") or {}
    if sc.get("enabled"):
        Sm = PM.subset_masks(E.H, E.W, int(sc["n"]), [int(s) for s in sc["sizes"]], E.cgen("pathscore-subset", E.image))
        P["S"] = Sm.to(E.ctx.dev)
        P["dF"] = PM.subset_deltas(E.fn, E.x, E.x0, P["S"], E.F1, E.ins_batch)
        SM.FwdCounter.n += int(Sm.shape[0])
    return P


def extra_scores(E: Env, P: dict, A: torch.Tensor, top_frac: float) -> dict:
    xm, q = P["xm"], {}
    if xm.get("loc"):
        q.update(PM.localization(A, E.mask, E.perm, top_frac, int(xm.get("pointing_tol", 15))))
    if "prob" in P:
        rc = xm["road"]
        q.update(PM.road(P["prob"], E.x, A, E.perm, P["fracs"], P["road_z"], P["road_sd"], E.ins_batch,
                         int(rc.get("iters", 1000)), float(rc.get("tol", 1e-5))))
    if "S" in P:
        q.update(PM.subset_scores(A, P["S"], P["dF"]))
    return q


def compute_scores(E: Env, attr: dict, pc: dict) -> dict:
    S = E.S
    ic = E.ctx.cfg["metrics"]["insdel"]
    bases = {b: mask_base(E.x, b, ic) for b in pc["mask_bases"]}
    steps = int(S["insdel_steps"])
    tf = float(pc.get("top_frac", 0.05))
    kk = max(1, int(tf * E.H * E.W))
    R, K = attr["A"].shape[:2]
    P = prepare_extra(E, pc)
    paths, ens = [], []
    for r in range(R):
        As = attr["A"][r].to(E.ctx.dev).double()
        tot = As.sum(0)
        spi = tot / K
        er = {"image": E.image, "seed": r, "K": K}
        for b, base in bases.items():
            q = SM.insdel_full(E.ctx.bb, E.cls, E.x, base, spi.float(), steps, E.ins_batch)
            er[f"ins_{b}"], er[f"del_{b}"], er[f"insdel_{b}"] = q["ins"], q["del"], q["diff"]
        er.update(extra_scores(E, P, spi, tf))
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
            row.update(extra_scores(E, P, A, tf))
            paths.append(row)
        del As
    sync()
    return {"paths": paths, "ens": ens}


def estimate(S: dict, pc: dict, per_eval: float, per_fwd: float) -> dict:
    K, R, nb = int(pc["K"]), int(pc["seeds"]), len(pc["mask_bases"])
    grad = K * R * S["N"] * S["M"]
    fwd = (K + 1) * R * nb * 2 * (int(S["insdel_steps"]) + 1)
    xm = xcfg(pc)
    if (xm.get("road") or {}).get("enabled"):
        fwd += (K + 1) * R * 2 * len(xm["road"]["fracs"])      # + thời gian PCG của nội suy ROAD (không tính ở đây)
    if (xm.get("subset") or {}).get("enabled"):
        fwd += int(xm["subset"]["n"])
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
    from .ctx import bench
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


def _z(v: float, sd: float) -> float:
    return v / sd if np.isfinite(sd) and sd > 0 else np.nan


def per_group_stats(paths, spi, bases: list, extra=()) -> dict:
    """Mỗi (ảnh, seed): ma trận điểm K × |metric| → thống kê chéo. Trả các bảng dài (một dòng / (ảnh, seed, ...)).
    Khoá metric = tên nền mask InsDel (cột insdel_<nền>) + tên cột metric bổ sung. Metric "nhỏ = tốt" được đổi dấu trước mọi
    phép chọn / hạng / gain (gain > 0 luôn = tốt hơn); `mean_path`, `spi`, `sd_path` vẫn ở đơn vị gốc."""
    import pandas as pd
    keys = list(bases) + [m for m in extra if m not in bases]
    mets = [mcol(k) for k in keys]
    sgn = np.array([mdir(k) for k in keys], dtype=float)
    if spi is None:
        spi = pd.DataFrame(columns=["image", "seed"] + mets)
    spread, cross, corr, bom, prox = [], [], [], [], []
    for (im, r), g in paths.groupby(["image", "seed"]):
        g = g.sort_values("path")
        X = g.reindex(columns=mets).values.astype(float) * sgn      # (K, nm), hướng: lớn = tốt
        K = X.shape[0]
        ok = np.isfinite(X).all(0)                                  # metric thiếu / NaN ở ảnh này → bỏ metric đó ở ảnh này
        mu = np.where(ok, X.mean(0), np.nan)
        sd = np.array([X[:, j].std(ddof=1) if ok[j] and K > 1 else np.nan for j in range(len(keys))])
        e = spi[(spi.image == im) & (spi.seed == r)]
        ens = (e.reindex(columns=mets).values[0].astype(float) * sgn) if len(e) else np.full(len(mets), np.nan)
        R = np.stack([_rank_desc(X[:, j]) if ok[j] else np.full(K, np.nan) for j in range(len(keys))], 1)
        for j, b in enumerate(keys):
            if not ok[j]:
                continue
            col = X[:, j]
            has_e = bool(np.isfinite(ens[j]))
            spread.append({"image": im, "seed": r, "metric": b, "K": K, "mean_path": mu[j] * sgn[j], "sd_path": sd[j],
                           "best_minus_mean": col.max() - mu[j], "mean_minus_worst": mu[j] - col.min(),
                           "spi": ens[j] * sgn[j], "spi_minus_mean": ens[j] - mu[j], "best_minus_spi": col.max() - ens[j],
                           "frac_paths_above_spi": float((col > ens[j]).mean()) if has_e else np.nan,
                           "best_z": _z(col.max() - mu[j], sd[j]), "spi_z": _z(ens[j] - mu[j], sd[j])})
            mm = 1
            while mm <= K:
                gm = best_of_m(col, mm) - mu[j]
                bom.append({"image": im, "seed": r, "metric": b, "m": mm, "gain": gm, "gain_z": _z(gm, sd[j])})
                mm *= 2
            for px, direction in PROXIES.items():
                v = g[px].values if px in g else np.full(K, np.nan)
                if not np.all(np.isfinite(v)):
                    continue
                pick = int(np.argmax(v) if direction == "max" else np.argmin(v))
                prox.append({"image": im, "seed": r, "metric": b, "proxy": px, "rho": _spearman(v, col),
                             "select_gain": col[pick] - mu[j], "oracle_gain": col.max() - mu[j], "spi_gain": ens[j] - mu[j],
                             "select_z": _z(col[pick] - mu[j], sd[j])})
        for a, ba in enumerate(keys):
            if not ok[a]:
                continue
            ib, iw = int(np.argmax(X[:, a])), int(np.argmin(X[:, a]))
            for j, bb in enumerate(keys):
                if not ok[j]:
                    continue
                cross.append({"image": im, "seed": r, "select_on": ba, "eval_on": bb, "K": K,
                              "best_gain": X[ib, j] - mu[j], "best_rank": R[ib, j],
                              "worst_gain": X[iw, j] - mu[j], "worst_rank": R[iw, j],
                              "spi_gain": ens[j] - mu[j], "sd_path": sd[j],
                              "best_z": _z(X[ib, j] - mu[j], sd[j]), "worst_z": _z(X[iw, j] - mu[j], sd[j]),
                              "spi_z": _z(ens[j] - mu[j], sd[j])})
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
    """Trung bình seed trong ảnh trước; rồi qua các ảnh: mean, std (ddof = 1), min, max, CI 95% bootstrap của mean."""
    import pandas as pd
    pim = df.groupby(keys + ["image"])[cols].mean().reset_index()
    rows = []
    for kv, g in pim.groupby(keys):
        kv = kv if isinstance(kv, tuple) else (kv,)
        r = dict(zip(keys, kv))
        r["n_images"] = len(g)
        for c in cols:
            v = g[c].values.astype(float)
            v = v[np.isfinite(v)]
            m, lo, hi = _boot(v, B, seed)
            r[c], r[f"{c}_lo95"], r[f"{c}_hi95"] = m, lo, hi
            r[f"{c}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
            r[f"{c}_min"] = float(np.min(v)) if len(v) else np.nan
            r[f"{c}_max"] = float(np.max(v)) if len(v) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def long_stats(df, keys: list, cols: list, names: dict | None = None):
    """Bảng dài: một dòng / (khoá, đại lượng) với mean, std, min, max, lo95, hi95 (từ kết quả của summarize)."""
    import pandas as pd
    rows = []
    for _, r in df.iterrows():
        for c in cols:
            if c not in df:
                continue
            row = {k: r[k] for k in keys}
            row["đại lượng"] = (names or {}).get(c, c)
            row["n_images"] = r.get("n_images")
            for s, suf in (("mean", ""), ("std", "_std"), ("min", "_min"), ("max", "_max"), ("lo95", "_lo95"), ("hi95", "_hi95")):
                row[s] = r.get(f"{c}{suf}", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# hình
# ─────────────────────────────────────────────────────────────────────────────

def _matrix(ax, M, labels, title, cmap, vmin, vmax, fmt_s, star=None, diag_box=True, cbar_label="", fs=8, seps=()):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    n = len(labels)
    rot = 45 if n > 6 else 0
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center", fontsize=fs)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=fs)
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
            ax.text(j, i, txt, ha="center", va="center", fontsize=fs, color="white" if lum < 0.45 else "#0b0b0b")
    for s in seps:                                    # ranh giới nhóm metric (InsDel | định vị | faithfulness khác)
        ax.axhline(s - 0.5, color="#0b0b0b", lw=1.0)
        ax.axvline(s - 0.5, color="#0b0b0b", lw=1.0)
    if diag_box:
        import matplotlib.patches as mpatches
        for i in range(n):
            ax.add_patch(mpatches.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#0b0b0b", lw=1.6))
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=8)


def _seps(keys: list) -> list:
    g = [mgroup(k) for k in keys]
    return [i for i in range(1, len(g)) if g[i] != g[i - 1]]


def _yerr(v, lo, hi):
    lo_ = v - lo if np.isfinite(lo) else 0.0
    hi_ = hi - v if np.isfinite(hi) else 0.0
    return [[max(lo_, 0.0)], [max(hi_, 0.0)]]


def make_figures(out: str, bases: list, cs, cr, sp, bom, paths, K: int, extra=()) -> list:
    from .plots import _setup
    plt = _setup()
    files = []
    keys = list(bases) + [m for m in extra if m not in bases]
    n = len(keys)
    seps = _seps(keys)
    fs = 8 if n <= 8 else 6

    def mat(df, col):
        M = np.full((n, n), np.nan)
        if col not in df:
            return M
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                q = df[(df.select_on == a) & (df.eval_on == b)] if "select_on" in df else df[(df.a == a) & (df.b == b)]
                if len(q):
                    M[i, j] = float(q[col].iloc[0])
        return M

    # 1. ba ma trận (màu theo z khi có: các metric khác đơn vị)
    try:
        if n > 6:
            s = 0.55 * n + 2.5
            fig, ax = plt.subplots(3, 1, figsize=(s + 1.5, 3 * s))
        else:
            fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
        zc = "best_z" if "best_z" in cs else "best_gain"
        G = mat(cs, zc)
        star = (mat(cs, f"{zc}_lo95") > 0) | (mat(cs, f"{zc}_hi95") < 0)
        v = np.nanmax(np.abs(G)) if np.isfinite(G).any() else 1.0
        if zc == "best_z":
            _matrix(ax[0], G, keys, "Đường tốt nhất theo hàng, chấm theo cột: chênh so với trung bình các đường,\n"
                    "đơn vị z = sd giữa các đường trong ảnh (* = CI 95% không chứa 0)", "RdBu", -v, v, "{:+.2f}", star,
                    cbar_label="Δ / sd giữa các đường (dương = tốt hơn)", fs=fs, seps=seps)
        else:
            _matrix(ax[0], G, keys, "Đường tốt nhất theo hàng, chấm theo cột:\nInsDel − trung bình các đường (* = CI 95% không chứa 0)",
                    "RdBu", -v, v, "{:+.3f}", star, cbar_label="ΔInsDel so với trung bình đường", fs=fs, seps=seps)
        Rk = mat(cs, "best_rank")
        mid = (K + 1) / 2
        _matrix(ax[1], Rk, keys, f"Hạng của đường đó theo cột (1 = tốt nhất / {K};\nngẫu nhiên = {mid:g})",
                "RdBu_r", 1, K, "{:.1f}", None, cbar_label="hạng trung bình (xanh = tốt)", fs=fs, seps=seps)
        Cm = mat(cr, "spearman")
        _matrix(ax[2], Cm, keys, f"Tương quan hạng điểm của {K} đường\ngiữa hai metric (Spearman, đã đổi dấu metric nhỏ = tốt)",
                "RdBu", -1, 1, "{:.2f}", None, diag_box=False, cbar_label="Spearman", fs=fs, seps=seps)
        fig.tight_layout()
        f = os.path.join(out, "pathscore_cross_matrix.png")
        fig.savefig(f, dpi=150); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[pathscore plots] matrix: {type(ex).__name__}: {ex}")

    # 2. small multiples: mỗi ô = một metric chấm (trục y riêng, đơn vị gốc); trục x = nền InsDel dùng để chọn, oracle, SPI
    try:
        ncol = min(4, n)
        nrow = int(math.ceil(n / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.9 * nrow + 1.0), squeeze=False)
        for t, b in enumerate(keys):
            a_ = axs[t // ncol, t % ncol]
            labs = []
            for a in list(bases) + ([b] if b not in bases else []):
                q = cs[(cs.select_on == a) & (cs.eval_on == b)]
                pos = len(labs)
                labs.append(a if a != b else f"{a}\n(oracle)")
                if not len(q):
                    continue
                q = q.iloc[0]
                own = a == b
                c = MCOLORS.get(a, ENS_COLOR)
                a_.errorbar(pos, q.best_gain, yerr=_yerr(q.best_gain, q.get("best_gain_lo95", np.nan), q.get("best_gain_hi95", np.nan)),
                            fmt="o", ms=8, mfc="white" if own else c, mec=c, mew=2, ecolor=c, elinewidth=1.4, capsize=0, zorder=3)
            s = sp[sp.metric == b]
            if len(s):
                s = s.iloc[0]
                a_.errorbar(len(labs), s.spi_minus_mean, yerr=_yerr(s.spi_minus_mean, s.get("spi_minus_mean_lo95", np.nan),
                            s.get("spi_minus_mean_hi95", np.nan)), fmt="D", ms=7, color=ENS_COLOR, capsize=0, zorder=3)
                labs.append(f"SPI {K}")
            a_.axhline(0, color=MUTED, lw=1, ls="--")
            a_.set_xticks(range(len(labs))); a_.set_xticklabels(labs, rotation=40, ha="right", fontsize=7)
            a_.set_xlim(-0.6, len(labs) - 0.4)
            a_.tick_params(axis="y", labelsize=7)
            a_.set_title(f"chấm theo {b}" + (" (đã đổi dấu)" if mdir(b) < 0 else ""), fontsize=8)
        for t in range(n, nrow * ncol):
            axs[t // ncol, t % ncol].axis("off")
        fig.supylabel("Δ so với trung bình các đường (dương = tốt hơn; CI 95% theo ảnh)", fontsize=8)
        fig.suptitle("Đường tốt nhất theo từng nền InsDel (trục x), chấm theo từng metric (mỗi ô, trục y riêng). "
                     "Chấm rỗng = chọn và chấm cùng metric (oracle, lạc quan); ◆ = SPI.", fontsize=9)
        fig.tight_layout()
        f = os.path.join(out, "pathscore_cross_dots.png")
        fig.savefig(f, dpi=150); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[pathscore plots] dots: {type(ex).__name__}: {ex}")

    # 3. ví dụ: hạng của từng đường theo từng metric, 4 ảnh
    try:
        ims = list(dict.fromkeys(paths.image))[:4]
        r0 = paths.seed.min()
        mets = [mcol(k) for k in keys]
        sgn = np.array([mdir(k) for k in keys], dtype=float)
        fig, ax = plt.subplots(1, len(ims), figsize=((3.2 + 0.18 * max(n - 5, 0)) * len(ims) + 1, 5.2), squeeze=False)
        img = None
        for t, im in enumerate(ims):
            g = paths[(paths.image == im) & (paths.seed == r0)].sort_values("path")
            X = g.reindex(columns=mets).values.astype(float) * sgn
            Rk = np.stack([_rank_desc(X[:, j]) for j in range(n)], 1)
            order = np.argsort(Rk[:, 0], kind="stable")      # sắp theo hạng của metric đầu
            a = ax[0, t]
            img = a.imshow(Rk[order], cmap="Blues_r", vmin=1, vmax=K, aspect="auto")
            a.set_xticks(range(n)); a.set_xticklabels(keys, rotation=45, ha="right", fontsize=fs)
            a.set_yticks(range(len(order))); a.set_yticklabels([f"đường {int(g.path.values[o])}" for o in order], fontsize=6)
            a.set_title(f"{im} (seed {r0})", fontsize=8)
            a.grid(False)
            for s_ in seps:
                a.axvline(s_ - 0.5, color="#0b0b0b", lw=1.0)
            for j in range(n):
                col = Rk[order, j]
                if not np.isfinite(col).any():
                    continue
                i = int(np.nanargmin(col))
                a.scatter([j], [i], marker="*", s=90, color=mcolor(keys[j]), edgecolor="white", linewidth=0.8, zorder=3)
        cb = fig.colorbar(img, ax=ax[0, :].tolist(), fraction=0.02, pad=0.02)
        cb.set_label(f"hạng (1 = tốt nhất / {K})", fontsize=8)
        fig.suptitle(f"Hạng của từng đường theo từng metric (hàng sắp theo {keys[0]}; ★ = đường tốt nhất của cột)", fontsize=9)
        f = os.path.join(out, "pathscore_rank_examples.png")
        fig.savefig(f, dpi=150); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[pathscore plots] examples: {type(ex).__name__}: {ex}")

    # 4. best-of-m (oracle, cùng metric) so với SPI: InsDel ở đơn vị gốc; nhóm metric bổ sung ở panel riêng, đơn vị z
    try:
        panels = [("insdel", list(bases), "gain", "spi_minus_mean", "InsDel − trung bình các đường")]
        for grp in ("loc", "faith"):
            ms = [k for k in keys if k not in bases and mgroup(k) == grp]
            if ms and "gain_z" in bom:
                panels.append((grp, ms, "gain_z", "spi_z", "Δ / sd giữa các đường (z)"))
        fig, axs = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.4), squeeze=False)
        for a_, (grp, ms, gc, sc, ylab) in zip(axs[0], panels):
            for b in ms:
                q = bom[bom.metric == b].sort_values("m")
                c = mcolor(b)
                a_.plot(q.m, q[gc], marker="o", color=c, lw=2, ms=6, label=f"{b}: tốt nhất trong m đường (oracle)")
                s = sp[sp.metric == b]
                if len(s) and sc in s:
                    a_.scatter([K * 1.35], [s[sc].iloc[0]], marker="D", s=40, color=c, edgecolor=ENS_COLOR, zorder=3)
            a_.axhline(0, color=MUTED, lw=1, ls="--")
            a_.set_xscale("log", base=2)
            a_.set_xlabel(f"m (số đường để chọn); ◆ bên phải = SPI {K} đường")
            a_.set_ylabel(ylab)
            a_.set_title(GROUPS[grp], fontsize=9)
            a_.legend(fontsize=7)
        fig.tight_layout()
        f = os.path.join(out, "pathscore_bestofm.png")
        fig.savefig(f, dpi=150); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[pathscore plots] bestofm: {type(ex).__name__}: {ex}")
    return files


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo
# ─────────────────────────────────────────────────────────────────────────────

def report_extra(pc: dict, paths) -> list:
    """Metric bổ sung đưa vào báo cáo: có trong CSV, có giá trị hữu hạn, và nằm trong pathscore.report_extra (null = tất cả)."""
    want = pc.get("report_extra") or list(EXTRA)
    out = []
    for m in EXTRA:
        if m in want and m in paths:
            v = pd_numeric(paths[m])
            if np.isfinite(v).any():
                out.append(m)
    return out


def pd_numeric(s) -> np.ndarray:
    import pandas as pd
    return pd.to_numeric(s, errors="coerce").values.astype(float)


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
    L = ["# ATTR_PROBE_PATHSCORE_REPORT — điểm của từng đường SPI (InsDel + metric bổ sung), đánh giá chéo giữa các metric\n"]
    if meta.get("debug") or cfg.get("_debug"):
        L.append("> **DEBUG** — chỉ để kiểm pipeline, không dùng để kết luận.\n")
    if paths is None or not len(paths):
        L.append("_Chưa có kết quả — `bash scripts/run.sh pathscore`._\n")
        with open(rp, "w") as fh:
            fh.write("\n".join(L) + "\n")
        print(f"[pathscore report] {rp} (trống)")
        return rp
    bases = [b for b in pc["mask_bases"] if f"insdel_{b}" in paths]
    extra = report_extra(pc, paths)
    keys = bases + extra
    K = int(paths.K.iloc[0])
    st_ = per_group_stats(paths, spi, bases, extra)
    sp = summarize(st_["spread"], ["metric"], ["mean_path", "sd_path", "best_minus_mean", "mean_minus_worst", "spi",
                                                "spi_minus_mean", "best_minus_spi", "frac_paths_above_spi", "best_z", "spi_z"], B, seed)
    sp["metric"] = pd.Categorical(sp.metric, keys, ordered=True)
    sp = sp.sort_values("metric")
    sp["metric"] = sp.metric.astype(str)
    cs = summarize(st_["cross"], ["select_on", "eval_on"], ["best_gain", "best_rank", "worst_gain", "worst_rank", "spi_gain",
                                                             "best_z", "worst_z"], B, seed)
    cr = summarize(st_["corr"], ["a", "b"], ["spearman"], B, seed)
    bom = summarize(st_["bestofm"], ["metric", "m"], ["gain", "gain_z"], B, seed)
    px = summarize(st_["proxy"], ["metric", "proxy"], ["rho", "select_gain", "oracle_gain", "spi_gain", "select_z"], B, seed) \
        if len(st_["proxy"]) else None
    for name, df in (("pathscore_spread", sp), ("pathscore_cross", cs), ("pathscore_corr", cr), ("pathscore_bestofm", bom),
                     ("pathscore_proxy", px)):
        if df is not None and len(df):
            df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    files = make_figures(out, bases, cs, cr, sp, bom, paths, K, extra)
    n_img, n_seed = int(paths.image.nunique()), int(paths.seed.nunique())
    STAT = ["mean", "std", "min", "max", "lo95", "hi95"]

    def mm(v) -> str:
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            return "—"
        sd = np.std(v, ddof=1) if len(v) > 1 else float("nan")
        return f"mean {fmt(v.mean())}, std {fmt(sd)}, min {fmt(v.min())}, max {fmt(v.max())}"

    def pm(v) -> str:
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            return "—"
        return f"{v.mean():+.2f} ± {np.std(v, ddof=1):.2f}" if len(v) > 1 else f"{v.mean():+.2f}"

    def sqmat(df, col, fmt_s, sig=False, with_std=True):
        tab = []
        for a in keys:
            r = {"chọn theo \\ chấm theo": a}
            for b in keys:
                q = df[(df.select_on == a) & (df.eval_on == b)]
                if len(q):
                    q = q.iloc[0]
                    s_ = "*" if sig and (q[f"{col}_lo95"] > 0 or q[f"{col}_hi95"] < 0) else ""
                    r[b] = fmt_s.format(q[col]) + (f" ± {q[f'{col}_std']:.2f}" if with_std and np.isfinite(q[f"{col}_std"]) else "") + s_
            tab.append(r)
        return pd.DataFrame(tab)

    # dữ liệu theo ảnh cho phần tóm tắt (mỗi ảnh: trung bình seed)
    xs = st_["cross"].groupby(["select_on", "eval_on", "image"])[["best_gain", "best_rank", "best_z"]].mean().reset_index()
    xs["gs"], xs["ge"] = xs.select_on.map(mgroup), xs.eval_on.map(mgroup)
    is_off = xs.select_on != xs.eval_on
    ins_pair = xs.select_on.isin(bases) & xs.eval_on.isin(bases)
    off = cs[(cs.select_on != cs.eval_on) & cs.select_on.isin(bases) & cs.eval_on.isin(bases)]
    sp_im = st_["spread"].groupby(["metric", "image"])[["spi_minus_mean", "spi_z"]].mean().reset_index()
    sp_im["g"] = sp_im.metric.map(mgroup)
    cr_im = st_["corr"].groupby(["a", "b", "image"]).spearman.mean().reset_index()
    cr_im["ga"], cr_im["gb"] = cr_im.a.map(mgroup), cr_im.b.map(mgroup)

    # ── tóm tắt
    L.append("## Tóm tắt\n")
    L.append(f"- {n_img} ảnh × {n_seed} seed × K = {K} đường SPI. Metric: InsDel với nền mask {bases}"
             + (f"; bổ sung {extra}" if extra else "") + ". Trong các dòng dưới, mean / std / min / max là **qua các ảnh** "
             "(mỗi ảnh: trung bình seed, rồi trung bình các ô / metric được nêu).")
    L.append("- **InsDel (cùng đơn vị)** — đường tốt nhất **trên chính metric dùng để chọn** (đường chéo), hơn trung bình các đường: "
             f"{mm(xs[~is_off & ins_pair].groupby('image').best_gain.mean())}.")
    if len(off):
        L.append(f"- InsDel — cùng đường đó **chấm theo nền mask khác**, hơn trung bình các đường: "
                 f"{mm(xs[is_off & ins_pair].groupby('image').best_gain.mean())}; hạng ({K} đường, ngẫu nhiên = {(K + 1) / 2:g}): "
                 f"{mm(xs[is_off & ins_pair].groupby('image').best_rank.mean())}. Theo từng cặp (trung bình ảnh): min "
                 f"{fmt(off.best_gain.min())}, max {fmt(off.best_gain.max())}; số cặp có CI 95% > 0: "
                 f"{int((off.best_gain_lo95 > 0).sum())}/{len(off)}.")
    L.append(f"- InsDel — SPI {K} đường (trung bình cộng các bản đồ) hơn trung bình các đường: "
             f"{mm(sp_im[sp_im.metric.isin(bases)].groupby('image').spi_minus_mean.mean())}.")
    if len(bases) > 1:
        L.append(f"- InsDel — tương quan hạng điểm của {K} đường giữa hai nền mask khác nhau: "
                 f"{mm(cr_im[(cr_im.a != cr_im.b) & cr_im.a.isin(bases) & cr_im.b.isin(bases)].groupby('image').spearman.mean())}.")
    L.append("")
    if extra:
        grps = [g for g in GROUPS if g in {mgroup(k) for k in keys}]
        L.append("**Theo nhóm metric** (đơn vị z = chênh lệch / sd giữa các đường trong ảnh; metric nhỏ = tốt đã đổi dấu; chỉ các cặp "
                 "select ≠ eval). Ô = mean ± std qua các ảnh của: z của đường tốt nhất theo một metric nhóm hàng, chấm theo "
                 f"metric nhóm cột | hạng trung bình (1..{K}, ngẫu nhiên {(K + 1) / 2:g}):\n")
        tab = []
        for gs in grps:
            r = {"chọn theo \\ chấm theo": GROUPS[gs]}
            for ge in grps:
                q = xs[is_off & (xs.gs == gs) & (xs.ge == ge)].groupby("image")[["best_z", "best_rank"]].mean()
                r[GROUPS[ge]] = f"{pm(q.best_z)} \\| {q.best_rank.mean():.1f}" if len(q) else "—"
            tab.append(r)
        r = {"chọn theo \\ chấm theo": f"(SPI {K} đường, z)"}
        for ge in grps:
            r[GROUPS[ge]] = pm(sp_im[sp_im.g == ge].groupby("image").spi_z.mean())
        tab.append(r)
        L.append(md(pd.DataFrame(tab)))
        L.append("Tương quan hạng (Spearman) điểm của các đường giữa hai metric, trung bình theo cặp nhóm (cặp khác metric), qua các ảnh:\n")
        tab = []
        for ga in grps:
            r = {"": GROUPS[ga]}
            for gb in grps:
                q = cr_im[(cr_im.a != cr_im.b) & (cr_im.ga == ga) & (cr_im.gb == gb)].groupby("image").spearman.mean()
                r[GROUPS[gb]] = pm(q)
            tab.append(r)
        L.append(md(pd.DataFrame(tab)))
    L.append("![ma trận](pathscore/pathscore_cross_matrix.png)\n")
    L.append("![dot](pathscore/pathscore_cross_dots.png)\n")

    # ── 1. thiết lập
    L.append("## 1. Thiết lập\n")
    L.append(f"- Đường = đúng sampler SPI hiện tại (α = {S['alpha']:g}, N = {S['N']}); attribution canonical (canon_rand, M = {S['M']}) "
             "của TỪNG đường; SPI = trung bình cộng có dấu của K bản đồ. Baseline của tích phân luôn là ảnh đen.")
    L.append("- InsDel = AUC_ins − AUC_del, xác suất softmax lớp dự đoán, hạng pixel Σ_c|A| (quy ước P0–P3). Nền mask (thay cho "
             "pixel chưa chèn / đã xoá): blur = Gaussian blur của ảnh; black / white / grey = ảnh đen / trắng / xám 0,5; mean = màu "
             "trung bình theo kênh của chính ảnh.")
    L.append(f"- ResNet-50 IMAGENET1K_V2; {n_img} ảnh tiền tố split pilot; thống kê: ảnh là đơn vị (trung bình seed trước), CI 95% "
             f"bootstrap theo ảnh ({B} lần).")
    if meta:
        pr = meta.get("provenance", {}) or {}
        L.append(f"- Ảnh: {meta.get('indices')}; GPU `{pr.get('gpu')}`; torch {pr.get('torch')}; hash "
                 f"`{str((meta.get('digests') or {}).get('scores', ''))[:12]}`; seed toàn cục {meta.get('global_seed')}.")
    L.append("- 'Đường tốt nhất' chọn bằng CHÍNH metric trên chính ảnh = oracle (cận trên lạc quan, không phải phương pháp).\n")
    if extra:
        xm = (meta.get("pathscore") or {}).get("extra_metrics") or xcfg(pc)
        xm = C._merge(EXTRA_DEFAULT, xm)
        L.append("**Metric bổ sung** (cùng hạng pixel Σ_c|A|, hoà phá bằng cùng hoán vị cho mọi đường). Hướng −1 = nhỏ là tốt "
                 "(được đổi dấu trong mọi phân tích chéo / hạng / gain):\n")
        L.append(md(pd.DataFrame([{"metric": k, "nhóm": GROUPS[mgroup(k)], "hướng": mdir(k), "mô tả": EXTRA[k][0]} for k in extra])))
        L.append(f"- Định vị: mask GT của ImageNet-Segmentation; pointing game nới mask {xm['pointing_tol']} px; prec_top dùng "
                 f"top {fmt(float(pc.get('top_frac', 0.05)))} pixel; iou dùng k = diện tích mask.")
        L.append(f"- ROAD: xoá p ∈ {xm['road']['fracs']} pixel (MoRF / LeRF), điền bằng nội suy tuyến tính có nhiễu (láng giềng cạnh 1/6, "
                 f"chéo 1/12, giải đúng hệ tuyến tính bằng PCG, tol {xm['road']['tol']}), nhiễu σ = {xm['road']['noise']} (pixel [0,1]) "
                 "dùng chung cho mọi đường của ảnh; điểm = xác suất lớp đích trung bình theo p.")
        L.append(f"- sens_n / infid: {xm['subset']['n']} ô vuông cạnh {xm['subset']['sizes']} px mỗi ảnh (dùng chung cho mọi đường), "
                 "thay bằng baseline tích phân đen; ΔF_S = logit(x) − logit(x với S xoá); sens_n = Pearson(Σ_S A, ΔF_S); "
                 "infid = Σ(Σ_S A − ΔF_S)² / ΣΔF_S² (không co giãn tối ưu).\n")

    # ── 2. từng metric
    L.append("## 2. Từng metric: đường tốt nhất / kém nhất / SPI so với trung bình các đường\n")
    L.append("Bảng nhanh (mean qua các ảnh; đơn vị gốc cho `TB các đường` và `SPI`, z cho các cột Δ; CI 95% bootstrap theo ảnh "
             "trong ngoặc):\n")
    tab = []
    for _, q in sp.iterrows():
        k = q.metric
        tab.append({"metric": k, "hướng": mdir(k), "TB các đường": fmt(q.mean_path), "SPI": fmt(q.spi),
                    "SPI − TB (z)": f"{q.spi_z:+.2f} [{q.spi_z_lo95:+.2f}, {q.spi_z_hi95:+.2f}]" if np.isfinite(q.spi_z) else "—",
                    "tốt nhất − TB (z)": f"{q.best_z:+.2f}" if np.isfinite(q.best_z) else "—",
                    "tỷ lệ đường vượt SPI": fmt(q.frac_paths_above_spi)})
    L.append(md(pd.DataFrame(tab)))
    L.append("Đầy đủ — mỗi dòng: một đại lượng của một metric; mean / std / min / max qua các ảnh; lo95 / hi95 = CI 95% bootstrap "
             f"của mean. `sd_path` = độ lệch chuẩn giữa {K} đường trong một ảnh. Các Δ đã đổi dấu cho metric nhỏ = tốt.\n")
    names2 = {"mean_path": "trung bình các đường", "sd_path": "sd giữa các đường (trong ảnh)",
              "best_minus_mean": "tốt nhất − trung bình", "mean_minus_worst": "trung bình − kém nhất",
              "spi": f"SPI {K} đường", "spi_minus_mean": "SPI − trung bình", "best_minus_spi": "tốt nhất − SPI",
              "frac_paths_above_spi": "tỷ lệ đường vượt SPI", "best_z": "tốt nhất − trung bình (z)", "spi_z": "SPI − trung bình (z)"}
    L.append(md(long_stats(sp, ["metric"], list(names2), names2), ["metric", "đại lượng", "n_images"] + STAT))

    # ── 3. chéo
    L.append("## 3. Đánh giá chéo: đường tốt nhất theo metric này, chấm theo metric kia\n")
    L.append("`best_gain` = điểm (metric chấm) của đường tốt nhất theo metric chọn − trung bình các đường (đơn vị gốc của metric chấm, "
             f"đã đổi dấu nếu nhỏ = tốt). `best_z` = best_gain / sd giữa các đường. `best_rank` = hạng theo metric chấm (1..{K}; "
             f"ngẫu nhiên {(K + 1) / 2:g}). `worst_*` = tương tự cho đường kém nhất. Đường chéo = oracle.\n")
    if len(bases) > 1:
        L.append("Ma trận `best_gain` giữa các nền InsDel (cùng đơn vị), ô = mean ± std qua các ảnh (* = CI 95% không chứa 0):\n")
        tab = []
        for a in bases:
            r = {"chọn theo \\ chấm theo": a}
            for b in bases:
                q = cs[(cs.select_on == a) & (cs.eval_on == b)]
                if len(q):
                    q = q.iloc[0]
                    sig = "*" if (q.best_gain_lo95 > 0 or q.best_gain_hi95 < 0) else ""
                    r[b] = f"{q.best_gain:+.3f} ± {q.best_gain_std:.3f}{sig}"
            tab.append(r)
        sp_row = {"chọn theo \\ chấm theo": f"(SPI {K} đường)"}
        for b in bases:
            q = sp[sp.metric == b]
            if len(q):
                sp_row[b] = f"{q.spi_minus_mean.iloc[0]:+.3f} ± {q.spi_minus_mean_std.iloc[0]:.3f}"
        tab.append(sp_row)
        L.append(md(pd.DataFrame(tab)))
    if extra:
        L.append("Ma trận `best_z` cho mọi metric (ô = mean ± std qua các ảnh; * = CI 95% không chứa 0):\n")
        L.append(md(sqmat(cs, "best_z", "{:+.2f}", sig=True)))
        L.append("Hàng SPI (z), theo metric chấm: " + "; ".join(
            f"{k} {pm(sp_im[sp_im.metric == k].spi_z)}" for k in keys) + ".\n")
    L.append(f"Ma trận hạng `best_rank` (1..{K}), ô = mean ± std qua các ảnh:\n")
    L.append(md(sqmat(cs, "best_rank", "{:.1f}")))
    L.append("Đầy đủ (mean / std / min / max qua các ảnh, CI 95% của mean):\n")
    names3 = {"best_gain": "tốt nhất: Δ", "best_z": "tốt nhất: Δ (z)", "best_rank": "tốt nhất: hạng",
              "worst_gain": "kém nhất: Δ", "worst_rank": "kém nhất: hạng"}
    L.append(md(long_stats(cs, ["select_on", "eval_on"], list(names3), names3),
                ["select_on", "eval_on", "đại lượng", "n_images"] + STAT))

    # ── 4. tương quan
    L.append(f"## 4. Tương quan hạng điểm của {K} đường giữa các metric\n")
    tab = []
    for a in keys:
        r = {"": a}
        for b in keys:
            q = cr[(cr.a == a) & (cr.b == b)]
            if len(q):
                r[b] = f"{q.spearman.iloc[0]:.2f}"
        tab.append(r)
    L.append(md(pd.DataFrame(tab)))
    L.append("Gần 0 nghĩa là thứ tự 'đường nào tốt hơn' của metric này không nói gì về metric kia (metric nhỏ = tốt đã đổi dấu, nên "
             "dương = đồng thuận). Theo cặp, qua các ảnh:\n")
    crp = cr[cr.a.map(keys.index) < cr.b.map(keys.index)]
    L.append(md(long_stats(crp, ["a", "b"], ["spearman"]), ["a", "b", "n_images"] + STAT))

    # ── 5. best-of-m
    L.append("## 5. Best-of-m (oracle, cùng metric)\n")
    L.append("Kỳ vọng điểm của đường tốt nhất trong m đường rút ngẫu nhiên từ K, trừ trung bình các đường (m = 1 ≡ 0); InsDel ở đơn vị "
             "gốc, metric bổ sung ở đơn vị z; qua các ảnh:\n")
    L.append(md(long_stats(bom[bom.metric.isin(bases)], ["metric", "m"], ["gain"]), ["metric", "m", "n_images"] + STAT))
    if extra:
        L.append(md(long_stats(bom[bom.metric.isin(extra)], ["metric", "m"], ["gain_z"], {"gain_z": "gain (z)"}),
                    ["metric", "m", "đại lượng", "n_images"] + STAT))
    L.append("![best-of-m](pathscore/pathscore_bestofm.png)\n")

    # ── 6. proxy
    if px is not None and len(px):
        L.append("## 6. Proxy chọn đường không tốn thêm forward\n")
        L.append("Hướng chọn đặt trước (loo_cos / loo_spearman / top_mass → lớn nhất; abs_resid_rel / l1_rel / d_hat → nhỏ nhất). "
                 "`rho` = Spearman proxy–điểm trong một ảnh; `select_gain` / `select_z` = đường proxy chọn − trung bình (đơn vị gốc / "
                 "z). So với `tốt nhất − trung bình` và `SPI − trung bình` ở mục 2. Qua các ảnh:\n")
        names6 = {"rho": "rho (proxy ~ điểm)", "select_gain": "đường proxy chọn − trung bình"}
        L.append(md(long_stats(px[px.metric.isin(bases)], ["metric", "proxy"], list(names6), names6),
                    ["metric", "proxy", "đại lượng", "n_images"] + STAT))
        if extra:
            names6x = {"rho": "rho (proxy ~ điểm)", "select_z": "đường proxy chọn − trung bình (z)"}
            L.append(md(long_stats(px[px.metric.isin(extra)], ["metric", "proxy"], list(names6x), names6x),
                        ["metric", "proxy", "đại lượng", "n_images"] + STAT))

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
