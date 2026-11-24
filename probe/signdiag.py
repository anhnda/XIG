"""
PATHFAM — chẩn đoán CÓ DẤU / KHÔNG DẤU (hậu kiểm, GPU; không chọn lại, không đổi objective, không đổi kết quả đã khoá).

Câu hỏi: vì sao J (infidelity, dùng A có dấu cộng trong ô) và metric ngoài (xếp hạng Σ_c|A_c|) mâu thuẫn? Trên CHÍNH các ảnh test đã khoá
và CÁC cấu hình test đã khoá, dựng lại bản đồ (luồng ngẫu nhiên riêng `pfsd`) ở K của từng ngân sách và Ē của 2m đường, rồi đo trên
cùng một bản đồ:
  infid (del_sq, bộ ô đánh giá) và infid_β (thang đo tối ưu — chẩn đoán thang đo, không phải objective);
  completeness: (ΣA − ΔF)/|ΔF|; triệt tiêu: ‖A‖₁/|ΣA|, khối lượng âm, triệt tiêu giữa kênh Σ_p|Σ_c A| / Σ_p Σ_c|A|;
  InsDel nền blur xếp hạng KHÔNG DẤU (như metric ngoài đã khoá) và CÓ DẤU; năng lượng trong mask |A| và A⁺; IoU |A| và A⁺;
  Spearman giữa bản đồ có dấu và không dấu.
Báo cáo: đồng thuận thứ hạng theo ảnh (Kendall τ) của −infid với từng metric, so cặp có dấu − không dấu; cơ chế (triệt tiêu ↔ infid,
↔ metric); smooth; completeness.

Lệnh: bash scripts/run.sh pathfam-signdiag  (GPU)  → runs_probe/pilot/infid/pathfam/signdiag/ + PATHFAM_SIGNED_UNSIGNED.md
       bash scripts/run.sh pathfam-signdiag-report (CPU, dựng lại báo cáo)
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
from .sampling_infid import build_dsets, cname

DEFAULT = {"m": 16, "budgets": [256, 1024], "insdel_steps": 50, "n_images": None, "gpu_minutes": 180}
METRICS = {  # tên: (chiều tốt, nhóm)
    "insdel_abs": (+1, "không dấu"), "insdel_signed": (+1, "có dấu"), "energy_abs": (+1, "không dấu"), "energy_pos": (+1, "có dấu"),
    "iou_abs": (+1, "không dấu"), "iou_pos": (+1, "có dấu")}
PAIRS = [("insdel_signed", "insdel_abs"), ("energy_pos", "energy_abs"), ("iou_pos", "iou_abs")]


def sdc(cfg: dict) -> dict:
    from . import config as C
    return C._merge(DEFAULT, (PFM.pc(cfg).get("signdiag") or {}))


def out_dir(cfg: dict) -> str:
    d = os.path.join(PFM.root(cfg), "signdiag")
    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# đại lượng trên một bản đồ (thuần, unit test được trên CPU)
# ─────────────────────────────────────────────────────────────────────────────

def _rank_top(m: torch.Tensor, k: int, perm: torch.Tensor) -> torch.Tensor:
    m = m.flatten()
    order = perm[torch.argsort(m[perm], descending=True, stable=True)]
    top = torch.zeros_like(m, dtype=torch.bool)
    top[order[:k]] = True
    return top


def _rank(m: torch.Tensor) -> torch.Tensor:
    r = torch.empty_like(m, dtype=torch.float64)
    r[torch.argsort(m)] = torch.arange(m.numel(), dtype=torch.float64, device=m.device)
    return r - r.mean()


def map_stats(A: torch.Tensor, dF_total: float, mask: torch.Tensor, perm: torch.Tensor) -> dict:
    """A (C, H, W). Không cần mô hình."""
    A = A.double()
    s = float(A.sum())
    l1 = float(A.abs().sum())
    ps, pa = A.sum(0), A.abs().sum(0)
    pos = ps.clamp_min(0)
    mk = mask.to(A.device).bool()
    k = int(mk.sum())
    out = {"compl_err": (s - dF_total) / abs(dF_total) if dF_total != 0 else math.nan, "cancel": l1 / abs(s) if s != 0 else math.inf,
           "neg_frac": float(A.clamp_max(0).abs().sum()) / l1 if l1 > 0 else math.nan,
           "chan_cancel": float(ps.abs().sum()) / float(pa.sum()) if float(pa.sum()) > 0 else math.nan,
           "energy_abs": float(pa[mk].sum()) / float(pa.sum()) if float(pa.sum()) > 0 else math.nan,
           "energy_pos": float(pos[mk].sum()) / float(pos.sum()) if float(pos.sum()) > 0 else math.nan}
    if k > 0:
        mf = mk.flatten()
        for nm, m in (("iou_abs", pa), ("iou_pos", ps)):
            top = _rank_top(m, k, perm)
            out[nm] = int((top & mf).sum()) / int((top | mf).sum())
    else:
        out["iou_abs"] = out["iou_pos"] = math.nan
    a, b = _rank(ps.flatten()), _rank(pa.flatten())
    out["spearman_signed_abs"] = float((a * b).sum() / (a.norm() * b.norm()))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# chạy (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def run(ctx) -> None:
    import pandas as pd
    cfg = ctx.cfg
    v = PFM.view(cfg)
    c = sdc(cfg)
    P = PFM.pr(cfg)
    if not cfg["_debug"] and not os.path.exists(PFM.lock_path(cfg)):
        raise SystemExit("pathfam-signdiag là hậu kiểm trên ảnh test ĐÃ KHOÁ: cần pathfam-lock và pathfam-test trước")
    mt_p = os.path.join(PFM.root(cfg), "test", "meta.json")
    if not os.path.exists(mt_p):
        raise SystemExit(f"chưa có {mt_p} — chạy pathfam-test trước")
    imgs = json.load(open(mt_p))["images"]
    if c["n_images"]:
        imgs = imgs[: int(c["n_images"])]
    sel = PFM.load_sel(cfg)
    confs = PFM.test_configs(cfg, sel)
    Dp = PFM.pc(cfg)["sets"][0]
    sc = {"sets": [Dp]}
    budgets = [int(B) for B in c["budgets"]]
    m_need = {cname(cf): max([int(c["m"])] + [B // cf["N"] // 2 for B in budgets if B % cf["N"] == 0]) for cf in confs}
    grads = sum(2 * m_need[cname(cf)] * cf["N"] * int(P["M"]) for cf in confs)
    from .ctx import bench
    bm = bench(ctx)
    est = len(imgs) * grads * bm["per_eval"] / 60
    print(f"[pathfam-signdiag] {len(imgs)} ảnh × {len(confs)} cấu hình: {grads} grad/ảnh ≈ {est:.1f} phút (chưa kể InsDel)", flush=True)
    if c["gpu_minutes"] is not None and est > float(c["gpu_minutes"]) and not cfg["_debug"]:
        raise SystemExit(f"ước tính {est:.1f} phút > pathfam.signdiag.gpu_minutes = {c['gpu_minutes']}")
    od = out_dir(cfg)
    rows = []
    t0, n0 = time.time(), Counter.n
    for j, im in enumerate(imgs):
        cp = os.path.join(od, f"{im}.json")
        if os.path.exists(cp):
            rows += json.load(open(cp))
            continue
        print(f"  [pathfam-signdiag] ảnh {j + 1}/{len(imgs)} {im}", flush=True)
        E = IC.IEnv(ctx, int(im.replace("seg", "")))
        ds = build_dsets(E, v, sc, "budget")
        ev = ds[Dp]["eval"]
        rr = []
        for cf in confs:
            keep = [(f"rev_B{B}_K{B // cf['N']}", "rev", B // cf["N"]) for B in budgets
                    if B % cf["N"] == 0 and (B // cf["N"]) % 2 == 0] + [("mean_all", "mean", 0)]
            res = BO.pair_maps(E, cf, m_need[cname(cf)], P, ds, "pfsd", keep)
            for lab, A in res["maps"].items():
                y = ev.project(A)
                d = ev.dF.to(y.device)
                r = {"image": im, "config": cname(cf), "family": cf["family"], "N": cf["N"], "h": PFM.h_of(cf), "estimator": lab,
                     "infid": float(IC.infid(y, d)[0]), "infid_beta": float(IC.infid_beta(y, d)[0]), **map_stats(A, E.D, E.mask, E.perm)}
                for nm, sg in (("insdel_abs", False), ("insdel_signed", True)):
                    r[nm] = float(SM.insdel_full(E.ctx.bb, E.cls, E.x, E.bg("blur"), A.float(), int(c["insdel_steps"]), E.ins_batch, signed=sg)["diff"])
                rr.append(r)
            del res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with open(cp, "w") as fh:
            json.dump(rr, fh)
        rows += rr
    pd.DataFrame(rows).to_csv(os.path.join(od, "signdiag.csv"), index=False)
    with open(os.path.join(od, "meta.json"), "w") as fh:
        json.dump({"images": imgs, "configs": [cname(cf) for cf in confs], "settings": c, "minutes": (time.time() - t0) / 60,
                   "grad_evals": Counter.n - n0, "post_hoc": True, "debug": cfg["_debug"]}, fh, indent=1, default=str)
    report(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def report(cfg: dict) -> str:
    import pandas as pd
    from .infid_posthoc import boot_idx, ci_of, fmt_ci
    from .pathfam_verify import kendall
    from .report import md
    P = PFM.pr(cfg)
    nb, seed, al = int(P["bootstrap"]), int(P["seed"]), float(P["alpha"])
    od = out_dir(cfg)
    t = pd.read_csv(os.path.join(od, "signdiag.csv"))
    inc = PFM.cname(PFM.pc(cfg)["incumbent"])
    L = ["# PATHFAM — chẩn đoán có dấu / không dấu (hậu kiểm, mô tả; không chọn lại, không đổi objective)\n",
         f"{t.image.nunique()} ảnh test đã khoá × {t.config.nunique()} cấu hình test đã khoá; bản đồ dựng lại với luồng ngẫu nhiên riêng. "
         "Mọi CI theo ảnh, chưa hiệu chỉnh đa kiểm định (mô tả).\n"]
    cols = ["infid", "infid_beta", "compl_err", "cancel", "neg_frac", "chan_cancel", "insdel_abs", "insdel_signed", "energy_abs", "energy_pos",
            "iou_abs", "iou_pos", "spearman_signed_abs"]
    g = t.groupby(["estimator", "config"])[cols].mean().reset_index()
    g["|compl_err| TB"] = t.assign(a=t.compl_err.abs()).groupby(["estimator", "config"]).a.mean().values
    L += ["## 1. Trung bình theo cấu hình và mức bản đồ\n",
          "`compl_err` = (ΣA − ΔF)/|ΔF| (kỳ vọng 0 với mọi họ — cầu phương `rand` không chệch; |·| giảm theo K). `cancel` = ‖A‖₁/|ΣA| (1 ⇔ không "
          "triệt tiêu). `chan_cancel` = Σ_p|Σ_c A| / Σ_p Σ_c|A|. `infid_beta` = infidelity sau khi chọn thang đo tối ưu — nếu khác biệt giữa họ "
          "biến mất ở `infid_beta` thì đó là khác biệt về thang đo, không phải về hình dạng.\n", md(g)]
    # 2. đồng thuận
    L.append("## 2. Đồng thuận thứ hạng theo ảnh giữa −infid và metric (Kendall τ qua cấu hình), có dấu so với không dấu\n")
    rows, prow = [], []
    for est, ge in t.groupby("estimator"):
        taus = {m: [] for m in METRICS}
        for im, gi in ge.groupby("image"):
            if gi.config.nunique() < 3:
                continue
            x = -gi.infid.values
            for m, (sg, _) in METRICS.items():
                taus[m].append(kendall(x, sg * gi[m].values))
        for m in METRICS:
            v = np.array([z for z in taus[m] if np.isfinite(z)])
            if len(v) < 3:
                continue
            lo, hi = ci_of(v[boot_idx(len(v), nb, seed)].mean(1), al)
            rows.append({"mức": est, "metric": m, "nhóm": METRICS[m][1], "n ảnh": len(v), "τ TB [CI]": fmt_ci(float(v.mean()), lo, hi, "{:.3f}")})
        for a, b in PAIRS:
            va, vb = np.array(taus[a]), np.array(taus[b])
            ok = np.isfinite(va) & np.isfinite(vb)
            dv = va[ok] - vb[ok]
            if len(dv) >= 3:
                lo, hi = ci_of(dv[boot_idx(len(dv), nb, seed)].mean(1), al)
                prow.append({"mức": est, "so sánh": f"τ({a}) − τ({b})", "TB [CI]": fmt_ci(float(dv.mean()), lo, hi, "{:.3f}"),
                             "đọc": "có dấu đồng thuận với J hơn" if lo > 0 else "không dấu đồng thuận hơn" if hi < 0 else "không rõ"})
    L += [md(pd.DataFrame(rows)), "**Giả thuyết H-SU** (đặt trước khi xem dữ liệu này): metric CÓ DẤU đồng thuận với J nhiều hơn metric KHÔNG DẤU.\n",
          md(pd.DataFrame(prow))]
    # 3. cơ chế
    L.append("## 3. Cơ chế: triệt tiêu có đi kèm infid thấp và metric không dấu kém không? (Kendall τ theo ảnh qua cấu hình)\n")
    rows = []
    for est, ge in t.groupby("estimator"):
        for a, b in (("cancel", "infid"), ("cancel", "insdel_abs"), ("cancel", "insdel_signed"), ("chan_cancel", "infid"), ("neg_frac", "iou_abs")):
            v = np.array([kendall(gi[a].values, gi[b].values) for _, gi in ge.groupby("image") if gi.config.nunique() >= 3])
            v = v[np.isfinite(v)]
            if len(v) >= 3:
                lo, hi = ci_of(v[boot_idx(len(v), nb, seed)].mean(1), al)
                rows.append({"mức": est, "cặp": f"τ({a}, {b})", "TB [CI]": fmt_ci(float(v.mean()), lo, hi, "{:.3f}")})
    L.append(md(pd.DataFrame(rows)))
    # 4. smooth
    rows = []
    for est, ge in t.groupby("estimator"):
        W = {m: ge.pivot_table(index="image", columns="config", values=m, aggfunc="first") for m in ["infid", "infid_beta", "cancel"] + list(METRICS)}
        for sm in [c for c in ge.config.unique() if c.startswith("smooth_")]:
            for rf in [inc] + [c for c in ge.config.unique() if c.startswith("spi_a10_")]:
                r = {"mức": est, "smooth": sm, "so với": rf}
                for m, Wm in W.items():
                    if sm in Wm and rf in Wm:
                        dv = (Wm[sm] - Wm[rf]).dropna().values
                        if len(dv) >= 3:
                            lo, hi = ci_of(dv[boot_idx(len(dv), nb, seed)].mean(1), al)
                            r[f"Δ{m}"] = fmt_ci(float(dv.mean()), lo, hi, "{:.3g}")
                rows.append(r)
    L += ["## 4. smooth − tham chiếu (incumbent; SPI α = 10 cùng D)\n", md(pd.DataFrame(rows)) if rows else "_(không có smooth)_\n"]
    rp = os.path.join(PFM.root(cfg), "PATHFAM_SIGNED_UNSIGNED.md")
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[pathfam-signdiag] {rp}")
    return rp
