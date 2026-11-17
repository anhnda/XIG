"""E11_REPORT.md — LIG Phase 1 trên đường cho trước (đọc CSV trong <out>/e11_<split>)."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .report import _read, boot_ci, fmt, md

MUTED = "#6b6a63"
Q3 = ("iou", "insdel_blur", "insdel_black")


def _plt():
    from .plots import _setup
    return _setup()


def paired_by_path(p, path, a, b, col):
    """Hiệu a − b trên cùng mẫu đường (image, p), trung bình trong ảnh."""
    x = p[(p.path == path) & (p.estimator == a)].set_index(["image", "p"])[col]
    y = p[(p.path == path) & (p.estimator == b)].set_index(["image", "p"])[col]
    d = (x - y).dropna()
    return d, d.groupby(level=0).mean()


def write(out: str, cfg: dict) -> str:
    import pandas as pd
    warnings.filterwarnings("ignore")
    c = cfg["e11"]
    meta = json.load(open(os.path.join(out, "meta.json"))) if os.path.exists(os.path.join(out, "meta.json")) else {}
    p, en = _read(out, "e11_paths"), _read(out, "e11_ens")
    L = ["# E11_REPORT — LIG chỉ Phase 1 (tối ưu μ) trên một đường cho trước: có luôn hiệu quả?\n"]
    if meta:
        L.append(f"- split **{meta['split']}**, {len(meta['images'])} ảnh, {meta['minutes']:.1f} phút, GPU `{meta['gpu']}`"
                 + (" — **DEBUG**" if meta.get("debug") else "") + (" — exploratory" if meta.get("split") == "pilot" else ""))
    L.append(f"""
## Thiết kế

Mỗi họ đường sinh K = {c['K']} mẫu (IG: 1). Trên **cùng** đường và cùng gradient tại {c['N']} nút trái, chỉ đổi trọng số bước μ:
`uniform` (= canonical Riemann trái, cách SPI tổng hợp), `uniform_rescale`, `lig` (Phase 1 LIG, λ = {c['lam']}, τ = {c['tau']},
{c['mu_iter']} bước Adam, Δf căn đúng, rescale về D), `lig_norescale`, `lig_legacy` (Δf lệch một bước như bài đã chạy), `lig_tau`
(τ = {c['tau_alt']}), `mu_df` (μ ∝ |Δf|), `mu_proj` (Proj_simplex((λ/τ)|d|)). Không có Phase 2 (đường giữ nguyên).
Baseline đen, logit lớp dự đoán. `ig` + `lig` chính là `compute_lig_idig` của repo LIG (với Δf căn đúng).
Họ đường: {', '.join(c['paths'])} (smooth / lognormal khớp mức exploration với DP α = {c['alpha']} như E10).

**Luật trả lời (đặt trước):** Phase 1 **hiệu quả trên một họ** nếu `lig − uniform` (so với đúng cách tổng hợp của họ đó) có CI 95% > 0
ở ≥ {c['rule']['metrics_min']}/3 metric {list(Q3)} (hiệu ghép theo mẫu đường, trung bình trong ảnh, bootstrap theo ảnh). **Luôn hiệu quả**
nếu hiệu quả trên mọi họ **và** tỉ lệ mẫu đường mà `lig` hơn `uniform` (insdel_blur) ≥ {c['rule']['win_rate_min']} ở mọi họ.
`lig − uniform_rescale` tách phần do riêng μ khỏi phần do rescale.
""")
    if p is None or not len(p):
        L.append("_(không có dữ liệu)_")
        path = os.path.join(out, "E11_REPORT.md")
        open(path, "w").write("\n".join(L))
        return path
    paths = list(dict.fromkeys(p.path))
    ests = list(dict.fromkeys(p.estimator))
    qc = [x for x in Q3 + ("energy_mask",) if x in p]

    L.append("## 1. Chất lượng từng mẫu đường (K = 1; trung bình trong ảnh, median qua ảnh)\n")
    t = p.groupby(["path", "estimator", "image"])[qc].mean().groupby(["path", "estimator"]).median()
    L.append(md(t, index=True))

    L.append("## 2. μ của Phase 1 trông như thế nào?\n")
    L.append("`collapsed` = μ_max > 0.9; `coverage_support` = phần phần tử được di chuyển ở các bước có μ > 1e-3; `theorem_applies` = "
             "khoảng cách hai |d_k| lớn nhất ≥ τ/λ (khi đó μ = e_k* là nghiệm toàn cục); `resid_raw_rel` = (ΣA trước rescale − D)/|D|.\n")
    dcols = ["mu_max", "mu_support", "mu_entropy", "collapsed", "coverage_support", "coverage_weighted", "theorem_applies", "Q",
             "resid_raw_rel"]
    t = p[p.estimator.isin(["uniform", "lig", "lig_legacy", "lig_tau", "mu_df", "mu_proj"])].groupby(["path", "estimator"])[
        [x for x in dcols if x in p]].mean()
    L.append(md(t, index=True))

    L.append("## 3. Phase 1 có cải thiện không? (hiệu ghép theo mẫu đường)\n")
    rows, verdict = [], {}
    comps = [("lig", "uniform"), ("lig", "uniform_rescale"), ("uniform_rescale", "uniform"), ("lig_tau", "uniform"),
             ("lig_legacy", "uniform"), ("mu_df", "uniform"), ("mu_proj", "uniform")]
    for path in paths:
        for a, b in comps:
            if a not in ests or b not in ests:
                continue
            for col in Q3:
                if col not in p:
                    continue
                dpath, dimg = paired_by_path(p, path, a, b, col)
                e = boot_ci(dimg.values)
                win = float((dpath > 0).mean()) if len(dpath) else float("nan")
                wci = boot_ci((dpath > 0).groupby(level=0).mean().values)
                rows.append({"đường": path, "so sánh": f"{a} − {b}", "metric": col, "mean": e[0], "lo95": e[1], "hi95": e[2],
                             "tỉ lệ mẫu đường thắng": win, "win_lo95": wci[1], "win_hi95": wci[2], "n_ảnh": e[3], "n_mẫu": len(dpath)})
    tab = pd.DataFrame(rows)
    L.append(md(tab))

    L.append("## 4. Ensemble K mẫu (trung bình bản đồ, median qua ảnh)\n")
    if en is not None and len(en):
        L.append(md(en.groupby(["path", "estimator"])[[x for x in Q3 + ("energy_mask",) if x in en]].median(), index=True))
        rows = []
        for path in paths:
            for col in Q3:
                if col not in en:
                    continue
                x = en[(en.path == path) & (en.estimator == "lig")].set_index("image")[col]
                y = en[(en.path == path) & (en.estimator == "uniform")].set_index("image")[col]
                e = boot_ci((x - y).dropna().values)
                rows.append({"đường": path, "metric": col, "ensemble lig − uniform": e[0], "lo95": e[1], "hi95": e[2]})
        L.append(md(pd.DataFrame(rows)))

    L.append("## 5. Mục tiêu của LIG có dự đoán chất lượng không?\n")
    rows = []
    for path in paths:
        a = p[(p.path == path) & (p.estimator == "lig")].set_index(["image", "p"])
        b = p[(p.path == path) & (p.estimator == "uniform")].set_index(["image", "p"])
        j = pd.DataFrame({"dQ": a.Q - b.Q, "dJ": a.J - b.J, "d_insdel_blur": a.get("insdel_blur") - b.get("insdel_blur")
                          if "insdel_blur" in a else np.nan, "d_iou": a.iou - b.iou}).dropna()
        if len(j) > 4:
            rows.append({"đường": path, "Spearman(ΔQ, Δinsdel_blur)": j.dQ.rank().corr(j.d_insdel_blur.rank()),
                         "Spearman(−ΔJ, Δinsdel_blur)": (-j.dJ).rank().corr(j.d_insdel_blur.rank()),
                         "Spearman(ΔQ, ΔIoU)": j.dQ.rank().corr(j.d_iou.rank()), "n": len(j)})
    L.append(md(pd.DataFrame(rows)))

    L.append("## 6. Trả lời theo luật đặt trước\n")
    eff = {}
    for path in paths:
        sub = tab[(tab["đường"] == path) & (tab["so sánh"] == "lig − uniform")]
        n_pos = int((sub.lo95 > 0).sum())
        n_neg = int((sub.hi95 < 0).sum())
        wr = sub[sub.metric == "insdel_blur"]["tỉ lệ mẫu đường thắng"]
        wr = float(wr.iloc[0]) if len(wr) else float("nan")
        eff[path] = (n_pos >= int(c["rule"]["metrics_min"]), n_pos, n_neg, wr)
        L.append(f"- **{path}**: `lig − uniform` CI > 0 ở {n_pos}/3 metric, CI < 0 ở {n_neg}/3; tỉ lệ mẫu đường thắng (insdel_blur) "
                 f"{fmt(wr)} → {'**hiệu quả**' if eff[path][0] else ('**có hại**' if n_neg >= int(c['rule']['metrics_min']) else 'không rõ')}.")
    always = all(v[0] and np.isfinite(v[3]) and v[3] >= float(c["rule"]["win_rate_min"]) for v in eff.values())
    L.append(f"- **Kết luận: Phase 1 của LIG {'LUÔN' if always else 'KHÔNG luôn'} hiệu quả khi cho trước một mẫu đường** "
             f"(theo luật: hiệu quả trên mọi họ và tỉ lệ thắng ≥ {c['rule']['win_rate_min']}).")

    files = []
    try:
        plt = _plt()
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
        for a_, col in zip(ax, Q3):
            sub = tab[(tab["so sánh"] == "lig − uniform") & (tab.metric == col)]
            y = np.arange(len(sub))
            a_.errorbar(sub["mean"], y, xerr=[sub["mean"] - sub.lo95, sub.hi95 - sub["mean"]], fmt="o", color="#eb6834", ecolor=MUTED)
            a_.axvline(0, color=MUTED, lw=0.8)
            a_.set_yticks(y, sub["đường"], fontsize=7)
            a_.set_title(f"lig − uniform: {col} (95% CI)", fontsize=9)
        fig.tight_layout(); f = os.path.join(out, "e11_lig_vs_uniform.png"); fig.savefig(f); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[e11 plots] lỗi: {type(ex).__name__}: {ex}")
    L.append("\n## Tệp\n")
    L.append("`e11_paths.csv` (mẫu đường × ước lượng: chất lượng + chẩn đoán μ), `e11_ens.csv`, `e11_timing.csv`, `meta.json`.\n")
    for f in files:
        L.append(f"- `{os.path.basename(f)}`")
    path = os.path.join(out, "E11_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[e11 report] {path}")
    return path
