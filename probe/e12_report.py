"""E12_REPORT.md — chỉ số phản ứng đường có dự đoán over-exploration gây hại không (đọc CSV trong <out>/e12_<split>)."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .e9_report import partial_spearman
from .report import _read, boot_ci, boot_spearman, fmt, md

Q3 = ("iou", "insdel_blur", "insdel_black")
MUTED = "#6b6a63"


def _plt():
    from .plots import _setup
    return _setup()


def diff_by_image(df, fam, lo, ref, col, extra=None):
    d = df[(df.family == fam)]
    if extra:
        for k, v in extra.items():
            d = d[d[k] == v]
    a = d[d.level == lo].groupby("image")[col].mean()
    b = d[d.level == ref].groupby("image")[col].mean()
    return (a - b).dropna()


def write(out: str, cfg: dict) -> str:
    import pandas as pd
    warnings.filterwarnings("ignore")
    c = cfg["e12"]
    dc = c["decision"]
    meta = json.load(open(os.path.join(out, "meta.json"))) if os.path.exists(os.path.join(out, "meta.json")) else {}
    p, en, g, ge = (_read(out, f"e12_{n}") for n in ("paths", "ens", "gate", "gate_ens"))
    L = ["# E12_REPORT — chỉ số phản ứng của đường có dự đoán khi nào exploration quá mạnh gây hại?\n"]
    if meta:
        L.append(f"- split **{meta['split']}**, {len(meta['images'])} ảnh, {meta['minutes']:.1f} phút (ước tính {meta['estimate']['e12_min']:.1f}), "
                 f"GPU `{meta['gpu']}`" + (" — **DEBUG**" if meta.get("debug") else "")
                 + (" — tiền đăng ký `e12_prereg.json`" if meta.get("split") != "pilot" else " — exploratory"))
    lo, ref = float(dc["low_level"]), float(dc["ref_level"])
    L.append(f"""
## Thiết kế và luật đặt trước

DP α ∈ {c['levels']['dp']} (họ: {', '.join(c['families'])}), K = {c['K']} đường/mức, N = {c['N']}, M = {c['M']}, canonical (μ đều), không LIG.
**S** (bất đồng bộ): `dist_geom`, `D_hat`, `max_jump`. **R_nl** = Σ_k |F(m_k) − (F(z_k)+F(z_{{k+1}}))/2| / Σ_k |ΔF_k| (= 0 với mọi F
affine, kể cả khi bước nhảy lớn); `jump` = max|ΔF_k|/Σ|ΔF_k| chỉ để đối chiếu. **Cổng số học**: {c['gate']['n_images']} ảnh, {c['gate']['K']}
đường/mức, M lồng nhau {c['gate']['M']} trên cùng đường; R_nl tính lại ở lưới 2N.

Phép so chính: α = {lo:g} (exploration mạnh) với α = {ref:g}; metric {list(Q3)}; "kém" = CI 95% của hiệu ghép theo ảnh < 0 ở
≥ {dc['metrics_min']}/3 metric. Quyết định:
1. α nhỏ kém ở M = {c['M']} nhưng **hết kém** ở M = {c['gate']['M_hi']} (cùng đường, ảnh gate) → **lỗi số học**.
2. α nhỏ vẫn kém, và R_nl dự đoán được mức hại — Spearman giữa các ảnh (ΔR_nl, hiệu chất lượng) CI < 0 **và** vẫn < 0 khi khống chế
   ΔS (partial), **và** Spearman mức đường (R_nl, chất lượng K = 1) trong từng ô (ảnh × α) CI < 0, ở ≥ {dc['metrics_min']}/3 metric →
   **có cơ sở phát triển adaptive exploration**.
3. α nhỏ vẫn kém nhưng R_nl không dự đoán được → **trade-off thực nghiệm, chưa biết cách chọn α → DỪNG** (không thay bằng chỉ số khác).
4. α nhỏ không kém → không thấy over-exploration gây hại trong phạm vi đã thử.
""")
    if p is None or not len(p):
        L.append("_(không có dữ liệu)_")
        path = os.path.join(out, "E12_REPORT.md")
        open(path, "w").write("\n".join(L))
        return path
    fams = list(dict.fromkeys(p.family))

    # ── 1. cổng số học ─────────────────────────────────────────────────────
    L.append("## 1. Cổng số học\n")
    numeric_artifact = None
    if g is not None and len(g):
        L.append("Sai khác L1 so với M_hi trên cùng đường, và chất lượng K = 1 theo M (median):\n")
        L.append(md(g.groupby(["family", "level", "M"])[["rel_l1_vs_Mhi", "iou", "energy_mask"]].median(), index=True))
    if ge is not None and len(ge):
        rows = []
        Ms = sorted(set(ge.M))
        for fam in fams:
            for M_ in Ms:
                for col in [q for q in Q3 if q in ge]:
                    d = diff_by_image(ge, fam, lo, ref, col, {"M": M_})
                    e = boot_ci(d.values)
                    rows.append({"họ": fam, "M": M_, "metric": col, f"α{lo:g} − α{ref:g}": e[0], "lo95": e[1], "hi95": e[2], "n_ảnh": e[3]})
        gt = pd.DataFrame(rows)
        L.append(f"Hiệu chất lượng ensemble {c['gate']['K']} đường giữa hai mức, theo M (ảnh gate, cùng đường):\n")
        L.append(md(gt))
        stab = ge.pivot_table(index=["image", "family", "level"], columns="M", values=[q for q in Q3 if q in ge])
        Mh = max(Ms)
        Mp = sorted(Ms)[-2] if len(Ms) > 1 else Mh
        conv = {q: float((stab[(q, Mp)] - stab[(q, Mh)]).abs().median()) for q in Q3 if (q, Mh) in stab}
        L.append(f"Ổn định theo M: median |metric(M = {Mp}) − metric(M = {Mh})| = " + ", ".join(f"{k} {fmt(v)}" for k, v in conv.items())
                 + f" (dung sai {dc['gate_tol']}) → cổng {'**ĐẠT**' if all(v <= dc['gate_tol'] for v in conv.values()) else '**CHƯA ĐẠT**'}.\n")
        fam0 = fams[0]
        lowM = gt[(gt["họ"] == fam0) & (gt.M == c["M"])] if c["M"] in Ms else gt[(gt["họ"] == fam0) & (gt.M == min(Ms))]
        hiM = gt[(gt["họ"] == fam0) & (gt.M == Mh)]
        worse_lo = int((lowM.hi95 < 0).sum())
        worse_hi = int((hiM.hi95 < 0).sum())
        numeric_artifact = worse_lo >= dc["metrics_min"] and worse_hi < dc["metrics_min"]
        L.append(f"- α nhỏ kém rõ ở M thô: {worse_lo}/3 metric; ở M = {Mh}: {worse_hi}/3 → lỗi số học: **{numeric_artifact}**.\n")

    # ── 2. chất lượng theo mức ─────────────────────────────────────────────
    L.append("## 2. Chất lượng theo mức exploration (ensemble K đường, median qua ảnh)\n")
    qc = [x for x in Q3 + ("energy_mask",) if x in en]
    L.append(md(en.groupby(["family", "level"])[qc].median(), index=True))
    rows = []
    for fam in fams:
        for lev in sorted(set(en[en.family == fam].level)):
            if lev == ref:
                continue
            for col in [q for q in Q3 if q in en]:
                d = diff_by_image(en, fam, lev, ref, col)
                e = boot_ci(d.values)
                rows.append({"họ": fam, "so sánh": f"α{lev:g} − α{ref:g}", "metric": col, "mean": e[0], "lo95": e[1], "hi95": e[2],
                             "số ảnh âm": int((d < 0).sum()), "n": e[3]})
    main_t = pd.DataFrame(rows)
    L.append(md(main_t))

    # ── 3. chẩn đoán đường ─────────────────────────────────────────────────
    L.append("## 3. Bất đồng bộ và phản ứng của mô hình theo mức (từng đường, median)\n")
    dcols = [x for x in ("dist_geom", "D_hat", "max_jump", "R_nl", "R_nl_max", "jump", "tv_over_D", "R_nl_fine") if x in p]
    L.append(md(p.groupby(["family", "level"])[dcols].median(), index=True))
    if "R_nl_fine" in p:
        q_ = p.dropna(subset=["R_nl_fine"])
        L.append(f"Độ nhạy theo lưới: median R_nl(2N)/R_nl(N) = {fmt((q_.R_nl_fine / q_.R_nl).median())}; Spearman(R_nl(N), R_nl(2N)) "
                 f"qua các đường = {fmt(q_.R_nl.rank().corr(q_.R_nl_fine.rank()))}.\n")
    L.append(f"Đối chiếu: Spearman(jump, R_nl) qua mọi đường = {fmt(p.jump.rank().corr(p.R_nl.rank()))} "
             "(nếu thấp: output nhảy lớn không đồng nghĩa phi tuyến).\n")

    # ── 4. dự đoán ─────────────────────────────────────────────────────────
    L.append("## 4. R_nl có dự đoán mức hại không?\n")
    rows, pred_ok = [], {}
    fam0 = fams[0]
    pm = p[p.family == fam0].groupby(["image", "level"])[["R_nl", "dist_geom"]].mean().reset_index()
    dR = (pm[pm.level == lo].set_index("image").R_nl - pm[pm.level == ref].set_index("image").R_nl).rename("dR")
    dS = (pm[pm.level == lo].set_index("image").dist_geom - pm[pm.level == ref].set_index("image").dist_geom).rename("dS")
    for col in [q for q in Q3 if q in en]:
        harm = diff_by_image(en, fam0, lo, ref, col).rename("harm")
        j = pd.concat([dR, dS, harm], axis=1).dropna().reset_index()
        e = boot_spearman(j.dR, j.harm)
        part = []
        rng = np.random.default_rng(0)
        for _ in range(2000):
            s_ = j.iloc[rng.integers(0, len(j), len(j))]
            v = partial_spearman(s_, "dR", "harm", ["dS"])
            if np.isfinite(v):
                part.append(v)
        pe = partial_spearman(j, "dR", "harm", ["dS"])
        plo, phi = (np.quantile(part, [0.025, 0.975]) if part else (np.nan, np.nan))
        cells = []
        for (im, lev), gg in p[p.family == fam0].groupby(["image", "level"]):
            if col in gg and len(gg) >= 4:
                v = gg.R_nl.rank().corr(gg[col].rank())
                if np.isfinite(v):
                    cells.append((im, v))
        cdf = pd.DataFrame(cells, columns=["image", "rho"])
        ce = boot_ci(cdf.groupby("image").rho.mean().values) if len(cdf) else (np.nan,) * 4
        rows.append({"metric": col, "Spearman(ΔR_nl, hại) giữa ảnh": e[0], "lo95": e[1], "hi95": e[2],
                     "partial | ΔS": pe, "p_lo95": plo, "p_hi95": phi,
                     "Spearman(R_nl, chất lượng) mức đường": ce[0], "c_lo95": ce[1], "c_hi95": ce[2], "n_ảnh": e[3]})
        pred_ok[col] = (e[2] < 0) and (phi < 0) and (ce[2] < 0)
    L.append(f"`hại` = chất lượng(α{lo:g}) − chất lượng(α{ref:g}) theo ảnh (âm = có hại); ΔR_nl, ΔS cùng hiệu. Mức đường: Spearman trong từng ô "
             "(ảnh × α) giữa R_nl và chất lượng K = 1, trung bình theo ảnh.\n")
    L.append(md(pd.DataFrame(rows)))
    rows = []
    for col in [q for q in Q3 if q in en]:
        per = []
        for im, gg in p[p.family == fam0].groupby("image"):
            lv = gg.groupby("level")[["R_nl", "dist_geom"]].mean()
            qv = en[(en.family == fam0) & (en.image == im)].set_index("level")[col]
            jj = lv.join(qv).dropna()
            if len(jj) >= 3:
                per.append((jj.R_nl.rank().corr(jj[col].rank()), jj.dist_geom.rank().corr(jj[col].rank())))
        if per:
            a = boot_ci([x[0] for x in per]); b = boot_ci([x[1] for x in per])
            rows.append({"metric": col, "trong ảnh qua α: Spearman(R_nl, Q)": a[0], "lo95": a[1], "hi95": a[2],
                         "Spearman(S, Q)": b[0], "s_lo95": b[1], "s_hi95": b[2]})
    L.append("Trong từng ảnh qua các mức α (R_nl và S cùng tăng theo exploration nên bị gây nhiễu lẫn nhau — chỉ để mô tả):\n")
    L.append(md(pd.DataFrame(rows)))

    # ── 5. quyết định ──────────────────────────────────────────────────────
    L.append("## 5. Quyết định theo luật\n")
    sub = main_t[(main_t["họ"] == fam0) & (main_t["so sánh"] == f"α{lo:g} − α{ref:g}")]
    inferior = int((sub.hi95 < 0).sum()) >= dc["metrics_min"]
    n_pred = sum(bool(v) for v in pred_ok.values())
    if numeric_artifact:
        case = "1 — LỖI SỐ HỌC: α nhỏ hết kém khi tích phân chính xác hơn; không phải bằng chứng trạng thái trung gian bị vỡ"
    elif not inferior:
        case = "4 — α nhỏ không kém rõ: không thấy over-exploration gây hại trong phạm vi đã thử"
    elif n_pred >= dc["metrics_min"]:
        case = "2 — α nhỏ kém và R_nl dự đoán được mức hại: có cơ sở phát triển adaptive (risk-constrained) exploration"
    else:
        case = "3 — trade-off thực nghiệm nhưng R_nl không dự đoán được: chưa biết cách chọn α → DỪNG"
    L.append(f"- α{lo:g} kém α{ref:g} rõ (≥ {dc['metrics_min']}/3 metric): **{inferior}**; R_nl dự đoán đạt ở {n_pred}/3 metric; "
             f"lỗi số học: **{numeric_artifact}**.")
    L.append(f"- **Kết luận: trường hợp {case}.**")
    if "lognormal" in fams:
        L.append("- Kiểm tổng quát ngoài DP (lognormal khớp D): xem các dòng `lognormal` ở mục 2–3.")
    elif case.startswith("2"):
        L.append("- Bước tiếp theo đã đăng ký: lặp lại trên lognormal ở các mức D tương ứng (`e12.families: [dp, lognormal]`).")
    if meta.get("split") == "pilot":
        L.append("- Split pilot = exploratory; kết luận cần chạy lại với `e12.split: fresh` (tiền đăng ký).")

    files = []
    try:
        plt = _plt()
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        q_ = en[en.family == fam0].groupby("level")[[x for x in Q3 if x in en]].median()
        for col in q_.columns:
            ax[0].plot(q_.index, q_[col], marker="o", label=col)
        ax[0].set_xscale("log"); ax[0].set_xlabel("α (nhỏ = exploration mạnh)"); ax[0].legend(fontsize=7); ax[0].set_title("chất lượng", fontsize=9)
        r_ = p[p.family == fam0].groupby("level")[["R_nl", "dist_geom"]].median()
        ax[1].plot(r_.index, r_.R_nl, marker="o", color="#eb6834", label="R_nl")
        ax[1].plot(r_.index, r_.dist_geom, marker="s", color="#2a78d6", label="S = dist_geom")
        ax[1].set_xscale("log"); ax[1].legend(fontsize=7); ax[1].set_title("phản ứng và bất đồng bộ", fontsize=9)
        if "insdel_blur" in en:
            harm = diff_by_image(en, fam0, lo, ref, "insdel_blur")
            j = pd.concat([dR, harm.rename("h")], axis=1).dropna()
            ax[2].scatter(j.dR, j.h, s=24, color="#eb6834", edgecolor="white", linewidth=1)
            ax[2].axhline(0, color=MUTED, lw=0.8)
            ax[2].set_xlabel(f"ΔR_nl (α{lo:g} − α{ref:g})"); ax[2].set_ylabel("Δinsdel_blur"); ax[2].set_title("dự đoán giữa ảnh", fontsize=9)
        fig.tight_layout(); f = os.path.join(out, "e12.png"); fig.savefig(f); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[e12 plots] lỗi: {type(ex).__name__}: {ex}")
    L.append("\n## Tệp\n")
    L.append("`e12_paths.csv`, `e12_ens.csv`, `e12_gate.csv`, `e12_gate_ens.csv`, `e12_images.csv`, `meta.json`.\n")
    for f in files:
        L.append(f"- `{os.path.basename(f)}`")
    path = os.path.join(out, "E12_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[e12 report] {path}")
    return path
