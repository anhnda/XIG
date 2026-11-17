"""E9_REPORT.md + đồ thị (đọc CSV trong <out>/e9)."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .report import _read, boot_ci, fmt, md

MUTED = "#6b6a63"


def _plt():
    from .plots import _setup
    return _setup()


def partial_spearman(df, x, y, controls):
    d = df[[x, y] + controls].dropna()
    if len(d) < len(controls) + 4:
        return float("nan")
    R = d.rank()
    Z = np.column_stack([np.ones(len(R))] + [R[c].values for c in controls])
    rx = R[x].values - Z @ np.linalg.lstsq(Z, R[x].values, rcond=None)[0]
    ry = R[y].values - Z @ np.linalg.lstsq(Z, R[y].values, rcond=None)[0]
    den = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / den) if den > 0 else float("nan")


def within(df, f, group="image"):
    vals = [f(g) for _, g in df.groupby(group)]
    return boot_ci([v for v in vals if np.isfinite(v)])


def paired_means(df, key, a, b, col, by="image"):
    m = df.groupby([by, key])[col].mean().unstack()
    if a not in m or b not in m:
        return (float("nan"),) * 3 + (0,)
    return boot_ci((m[a] - m[b]).dropna().values)


def write(out: str, cfg: dict) -> str:
    import pandas as pd
    warnings.filterwarnings("ignore")
    meta = json.load(open(os.path.join(out, "meta.json"))) if os.path.exists(os.path.join(out, "meta.json")) else {}
    A, H, qp, qe, loc, pt = (_read(out, f"e9_{n}") for n in ("A", "H1", "q_paths", "q_ens", "locality", "patches"))
    c = cfg["e9"]
    L = ["# E9_REPORT — bất biến global ≡ IG, độ bền H1, và cơ chế ghép kênh RGB\n"]
    if meta:
        L.append(f"- {meta['minutes']:.1f} phút (ước tính {meta['estimate']['e9_min']:.1f}), GPU `{meta['gpu']}`"
                 + (" — **DEBUG**" if meta.get("debug") else ""))
    files = []
    plt = None
    try:
        plt = _plt()
    except Exception:
        pass

    # ── A1 ────────────────────────────────────────────────────────────────
    L.append("## A1. Bất biến global-shared ≡ IG\n")
    if A is not None and len(A):
        s = A[(A.estimator == "s_uniform") & (A.family == "global")]
        if len(s):
            L.append(f"**Tích phân theo toạ độ tiến độ s** (điểm phân tầng đều theo s, ánh xạ ngược qua lịch trình global gấp khúc): "
                     f"max |Z(t(s)) − s| = {fmt(s.max_abs_s_err.max())}, max |điểm global − điểm IG| = {fmt(s.max_abs_point_diff.max())}, "
                     f"max ‖A_global − A_IG‖₁/‖A_IG‖₁ trên cùng điểm = {fmt(s.A_rel_diff_vs_ig_same_points.max())} "
                     f"({len(s)} đường, {s.image.nunique()} ảnh). Đường global và IG cho cùng một tích phân; khác biệt chỉ có thể đến "
                     "từ cách đặt điểm cầu phương.\n")
        L.append("**Hội tụ về tham chiếu IG dày** (`dist_ref_l1` = ‖A − A_ref‖₁/‖A_ref‖₁; `d_*` = metric − metric của tham chiếu). "
                 "`e8_*` tái tạo đúng ước lượng của E8 (phân tầng theo đoạn của lưới Z, M điểm/đoạn); `node_M*` cùng kiểu với M lớn hơn.\n")
        cols = [x for x in ("points", "dist_ref_l1", "dist_ref_l2", "d_iou", "d_insdel_blur", "d_insdel_black") if x in A]
        t = A[A.estimator != "ig_ref"].groupby(["estimator", "family"])[cols].median()
        L.append(md(t, index=True))
        rows = []
        for est in sorted(set(A.estimator) - {"ig_ref"}):
            g = A[(A.estimator == est) & (A.family == "global")]
            i = A[(A.estimator == est) & (A.family == "ig")]
            if len(g) and len(i):
                ig_d = i.groupby("image").dist_ref_l1.mean()
                r = (g.groupby("image").dist_ref_l1.mean() / ig_d).dropna()
                rows.append({"estimator": est, "median dist(global)/dist(IG) cùng số điểm": float(r.median()), "n_ảnh": len(r)})
        if rows:
            L.append(md(pd.DataFrame(rows)))
        e8g = A[A.estimator == "e8_global"]
        if "max_seg_len" in e8g and len(e8g):
            L.append(f"Đường global của E8 (N = {cfg['e8']['N']}): đoạn dài nhất của lịch trình median {fmt(e8g.max_seg_len.median())} "
                     f"(đều = {fmt(1 / cfg['e8']['N'])}), hệ số biến thiên độ dài đoạn median {fmt(e8g.seg_len_cv.median())}. "
                     "Đoạn dài chỉ nhận M điểm → nhiễu cầu phương lớn hơn IG cùng số điểm.\n")
        rows = []
        for m_ in ("iou", "insdel_blur", "insdel_black"):
            for est in [e for e in ("e8_global", "node_M16", "node_M64") if e in set(A.estimator)]:
                d = (A[(A.estimator == est) & (A.family == "global")].groupby("image")[m_].mean()
                     - A[(A.estimator == "e8_ig_a")].groupby("image")[m_].mean()).dropna()
                e = boot_ci(d.values)
                rows.append({"so sánh": f"global[{est}] − IG[e8_ig_a]", "metric": m_, "mean": e[0], "lo95": e[1], "hi95": e[2], "n": e[3]})
        L.append("Hiệu metric global − IG (IG theo đúng ước lượng E8):\n")
        L.append(md(pd.DataFrame(rows)))

    # ── A2 ────────────────────────────────────────────────────────────────
    L.append("## A2. H1 (rgb − indep) với nhiều eval hơn, cùng realization và seed của E8\n")
    if H is not None and len(H):
        igW = H[H.family == "ig"].set_index(["image", "level"]).W
        H = H.copy()
        H["exposure_ratio"] = [r.W / igW.get((r.image, r.level), np.nan) for r in H.itertuples()]
        rows = []
        for lev in sorted(set(H.level)):
            h = H[H.level == lev]
            for m_ in ("iou", "insdel_blur", "insdel_black", "exposure_ratio"):
                e = paired_means(h, "family", "rgb", "indep", m_)
                rows.append({"mức": lev, "evals/đường": int(h.evals.iloc[0]), "metric": m_, "rgb − indep": e[0], "lo95": e[1],
                             "hi95": e[2], "n_ảnh": e[3]})
        L.append(md(pd.DataFrame(rows)))
        L.append("Cùng các đường (lưới mịn là chính realization trước khi gộp). Nếu hiệu giữ dấu và độ lớn ở mức mịn, H1 không phải do "
                 "gộp bước nhảy hay nhiễu cầu phương.\n")

    # ── B ─────────────────────────────────────────────────────────────────
    L.append("## B. Ghép kênh RGB theo q\n")
    b = c["B"]
    if qp is not None and len(qp):
        q = qp[qp.family == "rgbq"].copy()
        igW = qp[qp.family == "ig"].set_index("image").W
        q["exposure_ratio"] = q.W / q.image.map(igW)
        q["within_frac"] = q.within / q.W
        L.append(f"{q.image.nunique()} ảnh, K = {b['K']} đường/q, N = {b['N']}, M = {b['M']}; common random numbers qua q (cùng Z_ind, "
                 "Z_pix, U_pix, cùng điểm cầu phương). Cov(Z_c, Z_c′) = q·t(1−t)/(α+1).\n")
        cols = [x for x in ("shared_frac", "exposure_ratio", "within_frac", "dist_geom", "iou", "energy_mask", "insdel_blur",
                            "insdel_black") if x in q]
        L.append("**K = 1** (median qua ảnh của trung bình trong ảnh):\n")
        L.append(md(q.groupby(["image", "q"])[cols].mean().groupby("q").median(), index=True))
        qs = sorted(set(q.q))
        rows = []
        for qq in qs[1:]:
            for m_ in [x for x in ("iou", "insdel_blur", "insdel_black", "exposure_ratio") if x in q]:
                e = paired_means(q, "q", qq, qs[0], m_)
                rows.append({"so sánh": f"q={qq:g} − q={qs[0]:g}", "metric": m_, "mean": e[0], "lo95": e[1], "hi95": e[2], "n": e[3]})
        L.append("**Hiệu ghép cặp theo ảnh so với q = 0 (K = 1)**:\n")
        L.append(md(pd.DataFrame(rows)))
        rows = []
        for m_ in [x for x in ("iou", "insdel_blur", "insdel_black", "exposure_ratio") if x in q]:
            e = within(q.groupby(["image", "q"])[m_].mean().reset_index(), lambda g: g.q.rank().corr(g[m_].rank()))
            mm = q.groupby(["image", "q"])[m_].mean().unstack()
            mid = 0.5 if 0.5 in mm else qs[len(qs) // 2]
            conv = boot_ci((mm[mid] - (mm[qs[0]] + mm[qs[-1]]) / 2).dropna().values) if mid in mm else (np.nan,) * 4
            rows.append({"metric": m_, "Spearman(q, metric) trong ảnh": e[0], "lo95": e[1], "hi95": e[2],
                         f"m({mid:g}) − [m(0)+m(1)]/2": conv[0], "lo95 ": conv[1], "hi95 ": conv[2]})
        L.append("**Hình dạng theo q** (liên tục hay ngưỡng): xu hướng trong ảnh và độ lệch của điểm giữa khỏi đường nối hai đầu:\n")
        L.append(md(pd.DataFrame(rows)))
        if plt is not None:
            try:
                fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
                g = q.groupby(["image", "q"])[["iou", "insdel_blur", "insdel_black", "exposure_ratio"]].mean().reset_index()
                for a, m_ in zip(ax, ["iou", "insdel_blur", "insdel_black", "exposure_ratio"]):
                    base = g[g.q == qs[0]].set_index("image")[m_]
                    for im, gg in g.groupby("image"):
                        a.plot(gg.q, gg[m_] - base[im], color=MUTED, lw=0.6, alpha=0.5)
                    mm = g.assign(d=g[m_] - g.image.map(base)).groupby("q").d.mean()
                    a.plot(mm.index, mm.values, color="#eb6834", lw=2.5, marker="o")
                    a.axhline(0, color=MUTED, lw=0.8); a.set_xlabel("q"); a.set_title(f"Δ{m_} so với q=0 (K=1)", fontsize=9)
                fig.tight_layout(); f = os.path.join(out, "e9_q.png"); fig.savefig(f); plt.close(fig); files.append(f)
            except Exception as ex:
                print(f"[e9 plots] {ex}")
    if qe is not None and len(qe):
        e2 = qe.copy()
        for k in ("within", "between", "across"):
            e2[f"{k}_frac"] = e2[k] / e2.EW
        L.append("**Ensemble theo q** và `linearity_resid` = ‖Ā(q) − [(1−q)Ā(0) + qĀ(1)]‖₁ / ‖Ā(1) − Ā(0)‖₁:\n")
        L.append(md(e2.groupby("q")[["iou", "insdel_blur", "insdel_black", "between_frac", "within_frac", "linearity_resid"]].median(),
                    index=True))
    if loc is not None and len(loc):
        cols = [x for x in loc.columns if x.startswith("locality_resid")]
        L.append("**Locality** (từng đường): R(q) = ‖A(q) − [s⊙A(1) + (1−s)⊙A(0)]‖₁/‖A(1) − A(0)‖₁, s = pixel chia sẻ ở mức q. "
                 "R ≈ mức nhiễu cầu phương (`quad_noise_ratio`) nghĩa là việc ghép kênh ở pixel KHÁC hầu như không đổi credit của pixel "
                 "này — hiệu ứng của RGB coupling là cục bộ theo pixel.\n")
        L.append(md(loc[cols + ["quad_noise_ratio"]].median().to_frame("median").T))
    if pt is not None and len(pt):
        p = pt.copy()
        p["I_rgb_rel"] = p.I_rgb / p["first"].replace(0, np.nan)
        L.append("**Tương tác hữu hạn sai phân và thay đổi credit theo patch** "
                 f"({b['interaction']['grid']}×{b['interaction']['grid']} patch, t ∈ {b['interaction']['t']}, bước {b['interaction']['step']}·Δx). "
                 "`dA` = Σ_patch |Ā(1) − Ā(0)| (Σ kênh). `I_rgb` = |I| của cặp kênh trong patch; `I_adj` / `I_far` = cùng kênh với patch kề / "
                 "xa; `first` = |F(z+u) − F(z)|. Spearman trong ảnh qua các patch, trung bình qua ảnh (bootstrap theo ảnh).\n")
        rows = []
        rel = within(p, lambda g: g.dA_h1.rank().corr(g.dA_h2.rank()))
        rows.append({"quan hệ": "độ tin cậy dA (nửa đường chẵn vs lẻ)", "mean": rel[0], "lo95": rel[1], "hi95": rel[2], "n_ảnh": rel[3]})
        for xcol in ("I_rgb", "I_rgb_rel", "I_adj", "I_far", "first", "absA0"):
            e = within(p, lambda g, xc=xcol: g[xc].rank().corr(g.dA.rank()))
            rows.append({"quan hệ": f"Spearman({xcol}, dA)", "mean": e[0], "lo95": e[1], "hi95": e[2], "n_ảnh": e[3]})
        for xcol in ("I_rgb", "I_adj", "I_far"):
            e = within(p, lambda g, xc=xcol: partial_spearman(g, xc, "dA", ["first", "absA0"]))
            rows.append({"quan hệ": f"partial({xcol}, dA | first, |Ā(0)|)", "mean": e[0], "lo95": e[1], "hi95": e[2], "n_ảnh": e[3]})
        e = within(p, lambda g: partial_spearman(g, "I_rgb", "dA", ["first", "absA0", "I_adj", "I_far"]))
        rows.append({"quan hệ": "partial(I_rgb, dA | first, |Ā(0)|, I_adj, I_far)", "mean": e[0], "lo95": e[1], "hi95": e[2], "n_ảnh": e[3]})
        L.append(md(pd.DataFrame(rows)))
        if qp is not None and len(qp):
            q = qp[qp.family == "rgbq"]
            mm = q.groupby(["image", "q"])[["iou", "insdel_blur", "insdel_black"]].mean().unstack()
            qs = sorted(set(q.q))
            img = p.groupby("image").agg(I_rgb_rel=("I_rgb_rel", "median"), I_rgb=("I_rgb", "median"),
                                         ratio_rgb_far=("I_rgb", "sum")).reset_index()
            img["ratio_rgb_far"] = img.image.map(p.groupby("image").I_rgb.sum() / p.groupby("image").I_far.sum())
            rows = []
            for m_ in ("iou", "insdel_blur", "insdel_black"):
                d = (mm[(m_, qs[-1])] - mm[(m_, qs[0])]).rename("d")
                j = img.set_index("image").join(d)
                for xc in ("I_rgb_rel", "ratio_rgb_far"):
                    from .report import boot_spearman
                    e = boot_spearman(j[xc], j.d)
                    rows.append({"x (theo ảnh)": xc, "y": f"{m_}(q=1) − {m_}(q=0)", "spearman": e[0], "lo95": e[1], "hi95": e[2], "n": e[3]})
            L.append("Giữa các ảnh: mức tương tác RGB của ảnh có dự đoán độ lớn hiệu ứng ghép kênh không?\n")
            L.append(md(pd.DataFrame(rows)))
        L.append("**Đọc**: cơ chế \"schedule coupling × functional interaction → credit allocation\" được ủng hộ nếu dA tăng theo I_rgb "
                 "**sau khi** khống chế độ lớn bậc một và |Ā(0)| (partial CI > 0), mạnh hơn so với I_adj / I_far, và locality ≈ nhiễu.\n")
    L.append("\n## Tệp\n")
    L.append("`e9_A.csv`, `e9_H1.csv`, `e9_q_paths.csv`, `e9_q_ens.csv`, `e9_locality.csv`, `e9_patches.csv`, `e9_images.csv`, `meta.json`.\n")
    for f in files:
        L.append(f"- `{os.path.basename(f)}`")
    path = os.path.join(out, "E9_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[e9 report] {path}")
    return path
