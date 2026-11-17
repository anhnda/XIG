"""E6_REPORT.md + đồ thị (đọc CSV trong <out>/e6_<split>)."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .report import _read, boot_ci, fmt, md

FCOL = {"ig": "#2a78d6", "guided": "#1baf7a", "guided_perm": "#008300", "guided_xfer": "#7cc47c", "spi": "#eb6834",
        "spi_perm": "#e87ba4", "spi_lam": "#eda100"}
MUTED = "#6b6a63"
Q = ("iou", "insdel_blur", "insdel_black", "energy_mask")


def _plt():
    from .plots import _setup
    return _setup()


def image_means(p, exp):
    """Trung bình metric theo realization (K = 1) trong từng ảnh × họ; kèm exposure/IG cùng ảnh."""
    d = p[p.exp == exp].copy()
    igW = d[d.family == "ig"].set_index("image").W
    d["exposure_ratio"] = d.W / d.image.map(igW)
    d["within_frac"] = d.within / d.W
    d["in_frac_W"] = d.W_in / d.W
    cols = ["exposure_ratio", "within_frac", "in_frac_W", "dist_geom", *[q for q in Q if q in d]]
    extra = [c for c in ("path_cost", "groups", "lambda") if c in d]
    g = d.groupby(["image", "family"])
    m = g[cols + extra].mean()
    m["n_real"] = g.size()
    sd = g[[q for q in Q if q in d]].std()
    sd.columns = [f"{c}_sd" for c in sd.columns]
    return m.join(sd).reset_index(), d


def paired(m, a, b, col):
    x = m[m.family == a].set_index("image")[col]
    y = m[m.family == b].set_index("image")[col]
    return boot_ci((x - y).dropna().values)


def write(out: str, cfg: dict) -> str:
    import pandas as pd
    warnings.filterwarnings("ignore")
    meta = json.load(open(os.path.join(out, "meta.json"))) if os.path.exists(os.path.join(out, "meta.json")) else {}
    p, en = _read(out, "e6_paths"), _read(out, "e6_ens")
    c = cfg["e6"]
    L = ["# E6_REPORT — cô lập gradient guidance (A) và cấu trúc đồng bộ (B)\n"]
    if meta:
        L.append(f"- split **{meta['split']}**, {len(meta['images'])} ảnh, {meta['minutes']:.1f} phút (ước tính "
                 f"{meta['estimate']['e6_min']:.1f}), GPU `{meta['gpu']}`, ảnh cho lịch trình: {', '.join(meta.get('donors', []))}"
                 + (" — **DEBUG, không dùng để kết luận**" if meta.get("debug") else ""))
        if meta["split"] == "pilot":
            L.append("- **Exploratory**: cùng 16 ảnh đã dùng để hình thành giả thuyết. Kết luận phải chờ split confirm.")
    a, b = c["A"], c["B"]
    L.append(f"""
## Thiết kế

Mọi đường: đen → x, cùng cầu phương phân tầng. **Phép so chính ở mức K = 1**: metric tính trên từng realization
(một đường / một seed hoán vị / một ảnh cho), trung bình trong ảnh, rồi hiệu ghép cặp theo ảnh với bootstrap theo ảnh.
Ensemble (bản đồ trung bình nhiều đường) chỉ là bảng phụ.

**E6-A** (N = {a['N']}, M = {a['M']}): `ig`; `guided` (Guided IG gốc); `guided_perm` (lịch trình Guided hoán vị trong
kênh × thập phân vị |Δx|, {a['perm_seeds']} seed); `guided_xfer` (lịch trình Guided của {a['donors']} ảnh khác thuộc split
other, gán theo hạng |Δx| trong từng kênh — giữ đặc tính lịch trình Guided, bỏ phụ thuộc vào gradient của ảnh đang giải thích);
`spi` (DP α độc lập, {a['spi_paths']} đường); `spi_perm` (chính các đường đó hoán vị, đối chứng âm ghép cặp);
`spi_lam` (λ khớp độ lệch hình học của Guided). `path_cost` = số eval gradient chỉ để dựng đường.

**E6-B** (N = {b['N']}, M = {b['M']}, {b['paths']} đường mỗi kiểu): DP α, cùng phân phối biên Beta(αt, α(1−t)) cho mọi phần tử;
chỉ đổi việc chia sẻ lịch trình: {', '.join(b['kinds'])}. `global` là đường thẳng IG tham số hoá lại (canonical chính xác = IG):
đối chứng.
""")
    if p is None or not len(p):
        L.append("_(không có dữ liệu)_")
    else:
        # ── A ──────────────────────────────────────────────────────────────
        mA, dA = image_means(p, "A")
        L.append("## E6-A. Gradient guidance\n")
        cols = ["n_real", "path_cost", "exposure_ratio", "dist_geom", "within_frac", "iou", "insdel_blur", "insdel_black",
                "energy_mask"]
        L.append("Median qua ảnh của trung bình K = 1 trong ảnh:\n")
        L.append(md(mA.groupby("family")[[x for x in cols if x in mA]].median(), index=True))
        comps = [("guided", "ig"), ("guided_perm", "guided"), ("guided_xfer", "guided"), ("spi_lam", "guided"),
                 ("spi", "guided"), ("guided_perm", "ig"), ("guided_xfer", "ig"), ("spi_lam", "ig"), ("spi", "ig")]
        rows = []
        for x, y in comps:
            if x in set(mA.family) and y in set(mA.family):
                for col in ("exposure_ratio", "iou", "insdel_blur", "insdel_black", "energy_mask"):
                    e = paired(mA, x, y, col)
                    rows.append({"so sánh": f"{x} − {y}", "metric": col, "mean": e[0], "lo95": e[1], "hi95": e[2], "n": e[3]})
        L.append("**Hiệu ghép cặp theo ảnh (K = 1)**:\n")
        L.append(md(pd.DataFrame(rows)))
        # đối chứng âm ghép cặp theo đường
        sp = dA[dA.family == "spi"].set_index(["image", "r"])
        spp = dA[dA.family == "spi_perm"].set_index(["image", "r"])
        rows = []
        for col in ("exposure_ratio", "iou", "insdel_blur", "insdel_black", "energy_mask"):
            dd = (spp[col] - sp[col]).groupby(level=0).mean()
            e = boot_ci(dd.values)
            rows.append({"metric": col, "spi_perm − spi (ghép theo đường)": e[0], "lo95": e[1], "hi95": e[2], "n_ảnh": e[3],
                         "sd giữa đường của hiệu": float((spp[col] - sp[col]).std())})
        L.append("**Đối chứng âm** (hoán vị các lịch trình DP độc lập cùng phân phối; kỳ vọng hiệu = 0):\n")
        L.append(md(pd.DataFrame(rows)))
        # kiểm định hoán vị trong từng ảnh
        rows = []
        g0 = dA[dA.family == "guided"].set_index("image")
        for col in ("iou", "insdel_blur", "insdel_black", "exposure_ratio"):
            pct = []
            for im, g in dA[dA.family == "guided_perm"].groupby("image"):
                v = float(g0.loc[im, col])
                pct.append(float((g[col] < v).mean() + 0.5 * (g[col] == v).mean()))
            pct = np.array(pct)
            rows.append({"metric": col, "median phân vị của Guided trong phân phối hoán vị": float(np.median(pct)),
                         "tỉ lệ ảnh Guided > mọi hoán vị": float((pct >= 1).mean()),
                         "tỉ lệ ảnh Guided < mọi hoán vị": float((pct <= 0).mean()), "n_ảnh": len(pct)})
        L.append(f"**Hoán vị trong từng ảnh** ({a['perm_seeds']} seed/ảnh): Guided gốc nằm ở phân vị nào của phân phối hoán vị "
                 "(0.5 = không khác; 1 = cao hơn mọi hoán vị):\n")
        L.append(md(pd.DataFrame(rows)))
        rows = []
        for fam in ("guided_perm", "guided_xfer", "spi_lam", "spi"):
            if fam not in set(mA.family):
                continue
            for col in ("iou", "insdel_blur", "insdel_black"):
                gg = mA[mA.family == "guided"][col].mean() - mA[mA.family == "ig"][col].mean()
                pp = mA[mA.family == fam][col].mean() - mA[mA.family == "ig"][col].mean()
                rows.append({"họ": fam, "metric": col, "lợi ích Guided vs IG": gg, "lợi ích của họ vs IG": pp,
                             "tỉ lệ": pp / gg if abs(gg) > 1e-12 else float("nan")})
        L.append("**Tỉ lệ lợi ích của Guided đạt được không cần gradient của chính ảnh** (tỉ số của trung bình; không ổn định khi "
                 "lợi ích gốc nhỏ):\n")
        L.append(md(pd.DataFrame(rows)))
        if "lambda" in dA:
            lm = dA[dA.family == "spi_lam"]
            L.append(f"Khớp hình học: dist Guided median {fmt(mA[mA.family == 'guided'].dist_geom.median())}, dist spi_lam median "
                     f"{fmt(lm.dist_geom.median())}, λ median {fmt(lm['lambda'].median())}.\n")

        # ── B ──────────────────────────────────────────────────────────────
        mB, dB = image_means(p, "B")
        L.append("## E6-B. Cấu trúc đồng bộ (topology chia sẻ lịch trình)\n")
        cols = ["groups", "n_real", "exposure_ratio", "dist_geom", "within_frac", "in_frac_W", "iou", "insdel_blur",
                "insdel_black", "energy_mask"]
        order = ["ig"] + list(b["kinds"])
        t = mB.groupby("family")[[x for x in cols if x in mB]].median().reindex([o for o in order if o in set(mB.family)])
        L.append("Median qua ảnh của trung bình K = 1 trong ảnh:\n")
        L.append(md(t, index=True))
        rows = []
        for k in b["kinds"]:
            for ref in ("indep", "ig"):
                if k == ref or k not in set(mB.family) or ref not in set(mB.family):
                    continue
                for col in ("exposure_ratio", "iou", "insdel_blur", "insdel_black", "energy_mask"):
                    e = paired(mB, k, ref, col)
                    rows.append({"so sánh": f"{k} − {ref}", "metric": col, "mean": e[0], "lo95": e[1], "hi95": e[2], "n": e[3]})
        L.append("**Hiệu ghép cặp theo ảnh (K = 1)**. `global − ig` là đối chứng (kỳ vọng ≈ 0, chỉ khác điểm cầu phương):\n")
        L.append(md(pd.DataFrame(rows)))
        rows = []
        sub = mB[mB.family.isin(list(b["kinds"]))].copy()
        sub["log_groups"] = np.log(sub.groups.clip(lower=1))
        for col in ("exposure_ratio", "iou", "insdel_blur", "insdel_black", "energy_mask", "dist_geom"):
            vals = []
            for _, g in sub.groupby("image"):
                if g[col].notna().sum() >= 3:
                    v = g.log_groups.rank().corr(g[col].rank())
                    if np.isfinite(v):
                        vals.append(v)
            e = boot_ci(vals)
            rows.append({"metric": col, "Spearman(log số nhóm, metric) trong ảnh": e[0], "lo95": e[1], "hi95": e[2], "n_ảnh": e[3]})
        L.append("**Xu hướng theo mức chia sẻ** (số nhóm lịch trình độc lập: nhiều nhóm = ít đồng bộ):\n")
        L.append(md(pd.DataFrame(rows)))

        # ── ensemble phụ ───────────────────────────────────────────────────
        if en is not None and len(en):
            L.append("## Phụ: ensemble (bản đồ trung bình mọi realization)\n")
            e2 = en.copy()
            for k in ("within", "between", "across"):
                e2[f"{k}_frac"] = e2[k] / e2.EW
            L.append(md(e2.groupby(["exp", "family"])[["K", "within_frac", "between_frac", "across_frac", "iou", "insdel_blur",
                                                      "insdel_black", "energy_mask"]].median(), index=True))

        # ── tiêu chí ───────────────────────────────────────────────────────
        L.append("## Đối chiếu với tiêu chí (ChatGPT đặt trước)\n")
        e1 = paired(mA, "guided_perm", "guided", "iou") if "guided_perm" in set(mA.family) else None
        e2_ = paired(mA, "guided_xfer", "guided", "iou") if "guided_xfer" in set(mA.family) else None
        e3 = paired(mA, "spi_lam", "guided", "iou") if "spi_lam" in set(mA.family) else None
        for nm, e in (("guided_perm − guided", e1), ("guided_xfer − guided", e2_), ("spi_lam − guided", e3)):
            if e:
                L.append(f"- ΔIoU {nm} (K=1): {fmt(e[0])} [{fmt(e[1])}, {fmt(e[2])}]")
        if "indep" in set(mB.family) and "global" in set(mB.family):
            e = paired(mB, "global", "indep", "iou")
            L.append(f"- ΔIoU global − indep (K=1): {fmt(e[0])} [{fmt(e[1])}, {fmt(e[2])}]")
        L.append("- Desynchronization được hỗ trợ nếu: phá guidance (perm/xfer/λ) giữ chất lượng **và** tăng đồng bộ (ít nhóm hơn) "
                 "làm chất lượng giảm có hệ thống. Guidance thiết yếu nếu chỉ `guided` gốc tốt hơn mọi họ phá ghép cặp. "
                 "Nếu đồng bộ chỉ đổi exposure mà không đổi IoU/insdel: nó giải thích cách lấy mẫu gradient, chưa giải thích "
                 "attribution tốt.")
        L.append("- Mọi kết luận trên split pilot là exploratory; bước tiếp theo là khoá các phép so trên và chạy split confirm "
                 "(`e6.split: confirm`, lần chạy đầu ghi `e6_prereg.json`).")

        # ── đồ thị ─────────────────────────────────────────────────────────
        files = []
        try:
            plt = _plt()
            fams = [f for f in ("guided", "guided_perm", "guided_xfer", "spi_lam", "spi", "spi_perm") if f in set(mA.family)]
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
            for axx, col in zip(ax, ["exposure_ratio", "iou", "insdel_blur"]):
                vals = [paired(mA, f, "ig", col) for f in fams]
                axx.bar(range(len(fams)), [v[0] for v in vals], 0.7, color=[FCOL.get(f, MUTED) for f in fams],
                        edgecolor="white", linewidth=1)
                axx.errorbar(range(len(fams)), [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                             fmt="none", ecolor=MUTED, lw=1)
                axx.axhline(0, color=MUTED, lw=0.8)
                axx.set_xticks(range(len(fams)), fams, rotation=40, ha="right", fontsize=7)
                axx.set_title(f"{col}: họ − IG (K=1, 95% CI)", fontsize=9)
            fig.tight_layout(); f = os.path.join(out, "e6A.png"); fig.savefig(f); plt.close(fig); files.append(f)
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
            kinds = [k for k in b["kinds"] if k in set(mB.family)]
            xs = [float(mB[mB.family == k].groups.median()) for k in kinds]
            for axx, col in zip(ax, ["exposure_ratio", "iou", "insdel_blur"]):
                med = [mB[mB.family == k][col].median() for k in kinds]
                lo = [mB[mB.family == k][col].quantile(0.25) for k in kinds]
                hi = [mB[mB.family == k][col].quantile(0.75) for k in kinds]
                axx.fill_between(xs, lo, hi, color="#eb6834", alpha=0.2, linewidth=0)
                axx.plot(xs, med, marker="o", color="#eb6834")
                for xv, k, mv in zip(xs, kinds, med):
                    axx.annotate(k, (xv, mv), fontsize=7, xytext=(3, 3), textcoords="offset points", color=MUTED)
                if col != "exposure_ratio":
                    axx.axhline(mB[mB.family == "ig"][col].median(), color="#2a78d6", ls="--", lw=1.2, label="IG")
                    axx.legend(fontsize=7)
                axx.set_xscale("log"); axx.set_xlabel("số nhóm lịch trình độc lập"); axx.set_title(f"{col} (K=1, median, IQR)", fontsize=9)
            fig.tight_layout(); f = os.path.join(out, "e6B.png"); fig.savefig(f); plt.close(fig); files.append(f)
        except Exception as ex:
            print(f"[e6 plots] lỗi: {type(ex).__name__}: {ex}")
        L.append("\n## Tệp\n")
        L.append("`e6_paths.csv` (từng realization), `e6_ens.csv`, `e6_images.csv`, `meta.json`.\n")
        for f in files:
            L.append(f"- `{os.path.basename(f)}`")
    path = os.path.join(out, "E6_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[e6 report] {path}")
    return path
