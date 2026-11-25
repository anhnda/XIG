"""
PATHFAM — chẩn đoán có dấu / không dấu, VÒNG 2 (hậu kiểm, mô tả; không chọn lại, không đổi endpoint, không tạo tuyên bố xác nhận mới).

Vòng 1 (`signdiag`) chỉ lưu thống kê tổng hợp của bản đồ, KHÔNG lưu attribution ⇒ thang đo β, sai số completeness tuyệt đối và so sánh có
kiểm soát (cùng h, N, K) không tính lại được từ cache. Vòng 2 dựng lại bản đồ một lần và lưu đủ đại lượng cho 4 kiểm toán:

  1. Thang đo: yᵀy, yᵀd, dᵀd trên bộ ô CHỌN và bộ ô ĐÁNH GIÁ (y = P_S A, d = ΔF_S) ⇒ β_opt = yᵀd/yᵀy trong mẫu (mỗi bộ) và
     ngoài mẫu (β fit trên bộ chọn, đánh giá trên bộ đánh giá); so với β = 0 (L = 1) và β = 1 (L = infidelity).
  2. So sánh có kiểm soát: thêm các cấu hình đối ứng spi / ln / cmix (và cả smooth khi có) CÙNG h, CÙNG N = 32, so ở CÙNG K.
  3. Metric: InsDel có dấu / không dấu với nền blur (như metric ngoài đã khoá) VÀ nền đen (CÙNG nền với attribution).
  4. Completeness: F(x), F(x′), ΔF, ΣA cho mọi bản đồ lồng nhau K ∈ {2, 4, 8, 16, 32} và Ē ⇒ sai số tuyệt đối, chuẩn hoá bền vững, xu hướng theo K.

Lệnh: bash scripts/run.sh pathfam-signdiag2 (GPU) → runs_probe/pilot/infid/pathfam/signdiag2/ + PATHFAM_SIGNDIAG2_AUDIT.md
       bash scripts/run.sh pathfam-signdiag2-report (CPU)
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from . import budget_opt as BO
from . import infid_core as IC
from . import pathfam as PFM
from . import sampling as SM
from .quad import Counter
from .sampling_infid import build_dsets, cname, norm_conf
from .signdiag import map_stats

DEFAULT = {"m": 16, "K_levels": [2, 4, 8, 16, 32], "metric_budgets": [256, 1024], "matched": True, "insdel_steps": 50,
           "n_images": None, "gpu_minutes": 240}


def sd2(cfg: dict) -> dict:
    from . import config as C
    return C._merge(DEFAULT, (PFM.pc(cfg).get("signdiag2") or {}))


def out_dir(cfg: dict) -> str:
    d = os.path.join(PFM.root(cfg), "signdiag2")
    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# hàm thuần
# ─────────────────────────────────────────────────────────────────────────────

def scale_terms(yy: float, yd: float, dd: float) -> dict:
    """β_opt = yᵀd/yᵀy; L(β) = (β²yy − 2β yd + dd)/dd. L(0) = 1, L(1) = infidelity, L(β_opt) = 1 − yd²/(yy·dd)."""
    b = yd / yy if yy > 0 else math.nan
    return {"beta_opt": b, "L_1": (yy - 2 * yd + dd) / dd, "L_opt": 1 - yd * yd / (yy * dd) if yy > 0 else math.nan}


def L_at(beta: float, yy: float, yd: float, dd: float) -> float:
    return (beta * beta * yy - 2 * beta * yd + dd) / dd


def matched_configs(cfg: dict, confs: list) -> list:
    """Cho mỗi h xuất hiện ở các cấu hình test N = 32 thuộc spi / ln / cmix / smooth: thêm spi, ln, cmix cùng h, N = 32."""
    N = int(PFM.pc(cfg)["grid"]["N"])
    alphas = set()
    for c in confs:
        if c["N"] != N:
            continue
        if c["family"] == "smooth":
            alphas.add(round(5.0 / c["delta"] ** 2 - 1.0, 6))                  # h = δ²/5 ⇔ α = 1/h − 1
        elif c["family"] in ("spi", "ln", "cmix"):
            alphas.add(float(c["alpha"]))
    out, seen = [], {cname(c) for c in confs}
    for a in sorted(alphas):
        for f in ("spi", "ln", "cmix"):
            c = norm_conf({"family": f, "alpha": a, "N": N})
            if cname(c) not in seen:
                seen.add(cname(c))
                out.append(c)
    return out


def robust_tau(dF: np.ndarray) -> float:
    """Ngưỡng chuẩn hoá bền vững: trung vị |ΔF| qua ảnh (tránh chia cho ΔF ≈ 0)."""
    return float(np.median(np.abs(dF)))


# ─────────────────────────────────────────────────────────────────────────────
# chạy (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def run(ctx) -> None:
    import pandas as pd
    cfg = ctx.cfg
    v = PFM.view(cfg)
    c = sd2(cfg)
    P = PFM.pr(cfg)
    if not cfg["_debug"] and not os.path.exists(PFM.lock_path(cfg)):
        raise SystemExit("pathfam-signdiag2 là hậu kiểm trên ảnh test ĐÃ KHOÁ: cần pathfam-lock và pathfam-test trước")
    imgs, confs = PFM.posthoc_targets(ctx, cfg, c["n_images"])
    extra = matched_configs(cfg, confs) if c["matched"] else []
    allc = confs + extra
    Dp = PFM.pc(cfg)["sets"][0]
    m = int(c["m"])
    Ks = [int(k) for k in c["K_levels"] if int(k) % 2 == 0 and int(k) <= 2 * m]
    grads = sum(2 * m * cf["N"] * int(P["M"]) for cf in allc)
    from .ctx import bench
    bm = bench(ctx)
    est = len(imgs) * grads * bm["per_eval"] / 60
    print(f"[pathfam-signdiag2] {len(imgs)} ảnh × {len(allc)} cấu hình ({len(extra)} đối ứng cùng h): {grads} grad/ảnh ≈ {est:.1f} phút "
          "(chưa kể InsDel)", flush=True)
    if c["gpu_minutes"] is not None and est > float(c["gpu_minutes"]) and not cfg["_debug"]:
        raise SystemExit(f"ước tính {est:.1f} phút > pathfam.signdiag2.gpu_minutes = {c['gpu_minutes']}")
    od = out_dir(cfg)
    rows = []
    t0, n0 = time.time(), Counter.n
    for j, im in enumerate(imgs):
        cp = os.path.join(od, f"{im}.json")
        if os.path.exists(cp):
            rows += json.load(open(cp))
            continue
        print(f"  [pathfam-signdiag2] ảnh {j + 1}/{len(imgs)} {im}", flush=True)
        E = IC.IEnv(ctx, int(im.replace("seg", "")))
        ds = build_dsets(E, v, {"sets": [Dp]}, "budget")
        rr = []
        for cf in allc:
            keep = [(f"rev_K{K}", "rev", K) for K in Ks] + [("mean_all", "mean", 0)]
            res = BO.pair_maps(E, cf, m, P, ds, "pfsd2", keep)
            metric_K = {B // cf["N"] for B in c["metric_budgets"] if B % cf["N"] == 0}
            for lab, A in res["maps"].items():
                K = 2 * m if lab == "mean_all" else int(lab.split("K")[1])
                r = {"image": im, "config": cname(cf), "family": cf["family"], "N": cf["N"], "h": PFM.h_of(cf), "estimator": lab, "K": K,
                     "B": K * cf["N"] * int(P["M"]), "matched_extra": cf in extra, "F0": E.F0, "F1": E.F1, "dF": E.D, "sumA": float(A.sum()),
                     "l1": float(A.abs().sum())}
                for st in ("sel", "eval"):
                    y = ds[Dp][st].project(A)[0]
                    d = ds[Dp][st].dF.to(y.device, torch.float64)
                    r[f"yy_{st}"], r[f"yd_{st}"], r[f"dd_{st}"] = float(y @ y), float(y @ d), float(d @ d)
                r.update(map_stats(A, E.D, E.mask, E.perm))
                if lab == "mean_all" or K in metric_K:
                    for bgn in ("blur", "black"):
                        for sg in (False, True):
                            r[f"insdel_{'signed' if sg else 'abs'}_{bgn}"] = float(
                                SM.insdel_full(E.ctx.bb, E.cls, E.x, E.bg(bgn), A.float(), int(c["insdel_steps"]), E.ins_batch, signed=sg)["diff"])
                rr.append(r)
            del res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with open(cp, "w") as fh:
            json.dump(rr, fh)
        rows += rr
    pd.DataFrame(rows).to_csv(os.path.join(od, "signdiag2.csv"), index=False)
    with open(os.path.join(od, "meta.json"), "w") as fh:
        json.dump({"images": imgs, "configs": [cname(x) for x in allc], "matched_extra": [cname(x) for x in extra], "settings": c,
                   "minutes": (time.time() - t0) / 60, "grad_evals": Counter.n - n0, "post_hoc": True, "debug": cfg["_debug"]}, fh, indent=1)
    report(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def _q(v) -> str:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "—"
    q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
    return " / ".join(f"{x:.3g}" for x in q)


def report(cfg: dict) -> str:
    import pandas as pd
    from .infid_posthoc import boot_idx, ci_of, fmt_ci
    from .pathfam_verify import kendall
    from .report import md
    P = PFM.pr(cfg)
    nb, seed, al = int(P["bootstrap"]), int(P["seed"]), float(P["alpha"])
    od = out_dir(cfg)
    t = pd.read_csv(os.path.join(od, "signdiag2.csv"))
    L = ["# PATHFAM — chẩn đoán vòng 2: thang đo, so sánh có kiểm soát, metric, completeness (hậu kiểm, mô tả)\n",
         f"{t.image.nunique()} ảnh test đã khoá; {t.config.nunique()} cấu hình ({t[t.matched_extra].config.nunique()} cấu hình đối ứng thêm vào chỉ "
         "để so sánh có kiểm soát). Không chọn lại, không đổi endpoint; mọi CI theo ảnh, chưa hiệu chỉnh đa kiểm định. Phân vị in dạng "
         "q10 / q25 / q50 / q75 / q90 theo ảnh.\n"]
    # 1. thang đo
    rows = []
    for (cfgn, est), g in t.groupby(["config", "estimator"]):
        if est not in ("mean_all",) and not any(est == f"rev_K{B // int(g.N.iloc[0])}" for B in (256, 1024)):
            continue
        bs = g.yd_sel / g.yy_sel
        be = g.yd_eval / g.yy_eval
        L1 = (g.yy_eval - 2 * g.yd_eval + g.dd_eval) / g.dd_eval
        Lopt_in = 1 - g.yd_eval ** 2 / (g.yy_eval * g.dd_eval)
        Lout = (bs ** 2 * g.yy_eval - 2 * bs * g.yd_eval + g.dd_eval) / g.dd_eval
        rows.append({"config": cfgn, "bản đồ": est, "β_opt bộ chọn (phân vị)": _q(bs), "β_opt bộ đánh giá (phân vị)": _q(be),
                     "tỷ lệ ảnh β_eval < 1": float((be < 1).mean()), "TB L(0)": 1.0, "TB L(1) = infid": float(L1.mean()),
                     "TB L(β_eval) trong mẫu": float(Lopt_in.mean()), "TB L(β_sel) ngoài mẫu": float(Lout.mean()),
                     "TB [L(1) − L(β_sel)] ngoài mẫu": float((L1 - Lout).mean())})
    L += ["## 1. Thang đo β (y = P_S A, d = ΔF_S; D chính)\n",
          "β_opt trong mẫu = yᵀd/yᵀy trên cùng bộ ô (lạc quan); ngoài mẫu: β fit trên bộ CHỌN rồi đánh giá L trên bộ ĐÁNH GIÁ. L(0) = 1 "
          "(bản đồ 0), L(1) = infidelity đã khoá. Nếu L(β_sel) ngoài mẫu ≈ L(1) và β ≈ 1 ⇒ thang đo không phải nguồn lỗi; β < 1 có hệ thống ⇒ bản đồ "
          "\"quá lớn\" trên các ô (thường do nhiễu phương sai cộng vào yᵀy).\n", md(pd.DataFrame(rows))]
    # 2. so sánh có kiểm soát
    L.append("## 2. So sánh có kiểm soát: cùng h, cùng N = 32, cùng K\n")
    tt = t[t.N == int(PFM.pc(cfg)["grid"]["N"])].copy()
    tt["hk"] = tt.h.round(4)
    rows, rel = [], []
    for (h, est), g in tt.groupby(["hk", "estimator"]):
        fams = sorted(g.family.unique())
        if len(fams) < 2:
            continue
        for fam in fams:
            gf = g[g.family == fam]
            rows.append({"h": h, "bản đồ": est, "họ": fam, "n": gf.image.nunique(), "infid": float(((gf.yy_eval - 2 * gf.yd_eval + gf.dd_eval) / gf.dd_eval).mean()),
                         "chan_cancel": float(gf.chan_cancel.mean()), "cancel": float(gf.cancel.replace(np.inf, np.nan).median()),
                         "iou_abs": float(gf.iou_abs.mean()), "energy_abs": float(gf.energy_abs.mean()),
                         **{k: float(gf[k].mean()) for k in ("insdel_abs_blur", "insdel_abs_black", "insdel_signed_black") if k in gf and gf[k].notna().any()}})
        g = g.assign(infid=(g.yy_eval - 2 * g.yd_eval + g.dd_eval) / g.dd_eval)
        for a, b in (("chan_cancel", "infid"), ("chan_cancel", "iou_abs"), ("chan_cancel", "energy_abs"), ("infid", "iou_abs"),
                     ("chan_cancel", "insdel_abs_black"), ("infid", "insdel_signed_black")):
            if a not in g or b not in g:
                continue
            taus = [kendall(gi[a].values, gi[b].values) for _, gi in g.groupby("image") if gi.family.nunique() >= 3 and gi[[a, b]].notna().all().all()]
            taus = np.array([x for x in taus if np.isfinite(x)])
            if len(taus) >= 3:
                lo, hi = ci_of(taus[boot_idx(len(taus), nb, seed)].mean(1), al)
                rel.append({"h": h, "bản đồ": est, "cặp": f"τ({a}, {b})", "n ảnh": len(taus), "TB [CI]": fmt_ci(float(taus.mean()), lo, hi, "{:.3f}")})
    L += ["Trung bình theo họ trong cùng (h, N, K):\n", md(pd.DataFrame(rows)),
          "Quan hệ còn lại khi đã cố định h, N, K (Kendall τ qua các họ, theo ảnh; cần ≥ 3 họ ở cùng h):\n", md(pd.DataFrame(rel)),
          "Đọc: nếu τ(chan_cancel, infid) và τ(chan_cancel, localization) biến mất khi cố định h, N, K thì quan hệ gộp (vòng 1) là do exploration "
          "và K gây ra, không phải do họ đường.\n"]
    # 3. metric
    rows = []
    for est, g in t.groupby("estimator"):
        cols = [c for c in ("insdel_abs_blur", "insdel_signed_blur", "insdel_abs_black", "insdel_signed_black") if c in g and g[c].notna().any()]
        if not cols:
            continue
        r = {"bản đồ": est, "ghi chú": "insdel_signed_blur LOẠI khỏi diễn giải (lệch nền) — xem DIAG_LOCK.json"}
        for c in cols:
            v = g[c].dropna().values
            lo, hi = ci_of(v[boot_idx(len(v), nb, seed)].mean(1), al)
            r[c] = fmt_ci(float(v.mean()), lo, hi, "{:.3f}")
        rows.append(r)
    L += ["## 3. InsDel có dấu / không dấu theo nền (TB qua ảnh × cấu hình)\n",
          "Định nghĩa đúng theo mã: xem docs/PATHFAM_METRIC_AUDIT.md. Nền blur khác nền của attribution (đen) ⇒ InsDel có dấu với nền blur không "
          "có diễn giải nhất quán; nền đen là phiên bản nhất quán.\n", md(pd.DataFrame(rows))]
    # 4. completeness
    tau = robust_tau(t.groupby("image").dF.first().values)
    t = t.assign(err=t.sumA - t.dF)
    t = t.assign(abs_err=t.err.abs(), rob=t.err.abs() / (t.dF.abs() + tau), rel_l1=t.err.abs() / t.l1.replace(0, np.nan))
    rows = []
    for (cfgn, K), g in t.groupby(["config", "K"]):
        rows.append({"config": cfgn, "K": int(K), "TB |ΣA − ΔF|": float(g.abs_err.mean()), "trung vị |ΣA − ΔF|": float(g.abs_err.median()),
                     "TB |err|/(|ΔF| + τ)": float(g.rob.mean()), "TB |err|/‖A‖₁": float(g.rel_l1.mean()), "TB err (có dấu)": float(g.err.mean())})
    C4 = pd.DataFrame(rows)
    slopes = []
    for cfgn, g in C4.groupby("config"):
        g = g[g.K > 0]
        if len(g) >= 3:
            sl = float(np.polyfit(np.log(g.K), np.log(g["TB |ΣA − ΔF|"]), 1)[0])
            slopes.append({"config": cfgn, "độ dốc log|err| theo log K": sl, "lý thuyết (nhiễu độc lập)": -0.5})
    L += ["## 4. Completeness\n",
          f"err = ΣA − ΔF. Chuẩn hoá bền vững chia cho |ΔF| + τ với τ = trung vị |ΔF| qua ảnh = {tau:.4g} (tránh nổ khi ΔF ≈ 0). Với cầu phương "
          "`rand` không chệch, E err = 0 và |err| ∝ K^{−1/2} nếu nhiễu giữa các đường độc lập (các đường trong cặp `rev` tương quan nên độ dốc "
          "có thể lệch −0,5).\n", md(C4), md(pd.DataFrame(slopes))]
    rp = os.path.join(PFM.root(cfg), "PATHFAM_SIGNDIAG2_AUDIT.md")
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[pathfam-signdiag2] {rp}")
    return rp
