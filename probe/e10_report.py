"""E10_REPORT.md — Random Path Family Ablation (đọc CSV trong <out>/e10_<split>)."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .e8_report import analyse
from .report import _read, fmt, md

FAMS = ["ig", "dp", "smooth", "lognormal"]
COL = {"ig": "#2a78d6", "dp": "#eb6834", "smooth": "#1baf7a", "lognormal": "#eda100"}
MUTED = "#6b6a63"


def _plt():
    from .plots import _setup
    return _setup()


def write(out: str, cfg: dict) -> str:
    import pandas as pd
    warnings.filterwarnings("ignore")
    c = cfg["e10"]
    meta = json.load(open(os.path.join(out, "meta.json"))) if os.path.exists(os.path.join(out, "meta.json")) else {}
    fam, paths, gate = (_read(out, f"e10_{n}") for n in ("fam", "paths", "gate"))
    L = ["# E10_REPORT — hiệu quả SPI đến từ mức exploration hay từ cấu trúc của Dirichlet Process?\n"]
    if meta:
        L.append(f"- split **{meta['split']}**, {len(meta['images'])} ảnh, {meta['minutes']:.1f} phút (ước tính {meta['estimate']['e10_min']:.1f}), "
                 f"GPU `{meta['gpu']}`" + (" — **DEBUG**" if meta.get("debug") else "")
                 + (" — tiền đăng ký: `e10_prereg.json`" if meta.get("split") != "pilot" else " — **exploratory**"))
    cal = meta.get("calibration", {})
    L.append(f"""
## Thiết kế

Bốn họ đường đơn điệu, cùng E Z_i(t) = t, độc lập giữa phần tử: `ig` (Z = t), `dp` (SPI, α = {c['alpha']}), `smooth`
(Z = t + δ s t(1−t), s = ±1), `lognormal` (số gia softmax(σ ε)). Mức exploration D(P) = E ∫(Z − t)² dt được KHỚP giữa dp, smooth,
lognormal; hiệu chuẩn chỉ dùng mô phỏng lịch trình, không dùng ảnh hay mô hình. Cùng baseline đen, lớp đích, canonical attribution,
N = {c['N']}, M = {c['M']}, 2×{c['K_half']} đường (hai tập rời nhau), cùng số eval. Đầu ra chính = bản đồ trung bình {2 * c['K_half']} đường;
**ổn định** = tương đồng giữa hai tập {c['K_half']} đường (IG: hai bộ điểm cầu phương).
""")
    if cal:
        L.append("**Hiệu chuẩn** (D trên lưới N nút, hình thang):\n")
        L.append(md(pd.DataFrame([{"đại lượng": k, "giá trị": v} for k, v in cal.items()])))
    if fam is None or not len(fam):
        L.append("_(không có dữ liệu)_")
        path = os.path.join(out, "E10_REPORT.md")
        open(path, "w").write("\n".join(L))
        return path

    L.append("## 1. Exploration thực tế (từng đường, median)\n")
    if paths is not None and len(paths):
        cols = [x for x in ("D_hat", "max_jump", "conc", "eff_steps", "dist_geom", "z_mid_q10", "z_mid_q90") if x in paths]
        L.append(md(paths.groupby("family")[cols].median().reindex([f for f in FAMS if f in set(paths.family)]), index=True))
        L.append("`max_jump` = bước tiến lớn nhất của một phần tử trong một ô lưới (trung bình qua phần tử); `conc` = Σ_k ΔZ² "
                 "(1/N nếu đều, → 1 nếu dồn vào một bước); `eff_steps` = 1/conc; `dist_geom` = độ lệch khỏi đường chéo trong không gian ảnh.\n")

    L.append("## 2. Attribution và ổn định\n")
    igW = fam[fam.family == "ig"].set_index("image").W_mean
    fam = fam.copy()
    fam["exposure_ratio"] = fam.W_mean / fam.image.map(igW)
    qcols = [x for x in ("iou", "insdel_blur", "insdel_black", "energy_mask") if x in fam]
    scols = [x for x in ("stab_cos_elem", "stab_spearman_abs", "stab_sign_agree_top10", "stab_jaccard_10", "stab_spearman_signed_pix")
             if x in fam]
    L.append(f"**Bản đồ {2 * c['K_half']} đường** và **ổn định giữa hai tập** (median qua ảnh):\n")
    L.append(md(fam.groupby("family")[qcols + scols + ["exposure_ratio"]].median().reindex([f for f in FAMS if f in set(fam.family)]),
                index=True))
    if paths is not None and len(paths):
        k1 = paths.groupby(["image", "family"])[[x for x in ("iou", "energy_mask", "insdel_blur") if x in paths]].mean()
        L.append("**K = 1** (trung bình các đường trong ảnh, median qua ảnh):\n")
        L.append(md(k1.groupby("family").median().reindex([f for f in FAMS if f in set(paths.family)]), index=True))

    L.append("## 3. Chi phí\n")
    cost = fam.groupby("family")[["exposure_ratio", "seconds", "evals"]].median()
    if gate is not None and len(gate):
        gate = gate.copy()
        gate["d_iou_M_vs_Mhi"] = gate.iou_M - gate.iou_Mhi
        cost = cost.join(gate.groupby("family")[["rel_l1_M_vs_Mhi", "d_iou_M_vs_Mhi", "max_jump"]].median())
    L.append(md(cost.reindex([f for f in FAMS if f in cost.index]), index=True))
    L.append("Exposure W chỉ mô tả, không phải tiêu chuẩn thắng. `rel_l1_M_vs_Mhi` = ‖A_M − A_Mhi‖₁/‖A_Mhi‖₁ trên cùng đường (sai số cầu phương).\n")

    L.append("## 4. Các phép so đã đăng ký\n")
    t = analyse(fam, c["hypotheses"], c["stats"])
    cols = ["ID", "so sánh", "metric", "test", "family", "margin", "n", "mean", "lo95", "hi95", "lo90", "hi90", "p_boot", "p_holm",
            "số ảnh âm", "số ảnh dương", "verdict"]
    L.append(md(t[[x for x in cols if x in t]]))

    L.append("## 5. Phân loại theo luật đặt trước\n")
    quality = ["iou", "insdel_blur", "insdel_black"]

    def comp(alt, met):
        r = t[(t["so sánh"] == f"{alt} − dp") & (t.metric == met)]
        return r.iloc[0] if len(r) else None
    eq_all, worse, better = True, {}, {}
    for alt in ("smooth", "lognormal"):
        w = b = 0
        for met in quality:
            r = comp(alt, met)
            if r is None:
                eq_all = False
                continue
            if not str(r.verdict).startswith("TƯƠNG"):
                eq_all = False
            w += int(r.hi95 < 0)
            b += int(r.lo95 > 0)
        worse[alt], better[alt] = w, b
    stab_better = {alt: (lambda r: r is not None and r.lo95 > 0)(comp(alt, "stab_cos_elem")) for alt in ("smooth", "lognormal")}
    if all(worse[a] >= 2 for a in worse):
        case = "B — DP tốt hơn rõ cả hai họ đối chứng (≥ 2/3 metric chất lượng, CI 95% < 0)"
    elif any(better[a] >= 2 for a in better) or (eq_all and any(stab_better.values())):
        who = [a for a in better if better[a] >= 2] or [a for a in stab_better if stab_better[a]]
        case = f"C — họ khác tốt hơn DP ({', '.join(who)}: chất lượng cao hơn, hoặc tương đương mà ổn định hơn)"
    elif eq_all:
        case = "A — các họ tương đương DP trong biên đặt trước (DP không thiết yếu)"
    else:
        case = "Chưa phân loại được (một số phép so CHƯA KẾT LUẬN)"
    L.append(f"- Số metric chất lượng (trong {quality}) mà họ đối chứng **kém** DP rõ: {worse}; **hơn** DP rõ: {better}; "
             f"ổn định (cosine) hơn DP rõ: {stab_better}; mọi phép tương đương đạt: {eq_all}.")
    L.append(f"- **Kết luận theo luật: {case}.**")
    L.append("- Luật: A nếu mọi phép so chất lượng smooth/lognormal − dp đều TƯƠNG ĐƯƠNG; B nếu cả hai họ kém DP rõ ở ≥ 2/3 metric; "
             "C nếu một họ hơn DP rõ ở ≥ 2/3 metric, hoặc tương đương mà ổn định hơn (CI > 0); còn lại chưa phân loại. "
             "Sau thí nghiệm này: **dừng**, không tự mở sweep mới.")

    files = []
    try:
        plt = _plt()
        fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
        for a_, met in zip(ax, ["iou", "insdel_blur", "insdel_black", "stab_cos_elem"]):
            if met not in fam:
                continue
            piv = fam.pivot_table(index="image", columns="family", values=met)
            fams = [f for f in ("ig", "smooth", "lognormal") if f in piv]
            vals = [piv[f] - piv["dp"] for f in fams]
            bp = a_.boxplot([v.dropna().values for v in vals], patch_artist=True, widths=0.55)
            for patch, f in zip(bp["boxes"], fams):
                patch.set_facecolor(COL[f]); patch.set_alpha(0.6)
            a_.axhline(0, color=MUTED, lw=0.8)
            a_.set_xticks(range(1, len(fams) + 1), [f"{f} − dp" for f in fams], fontsize=7)
            a_.set_title(met, fontsize=9)
        fig.tight_layout(); f = os.path.join(out, "e10_vs_dp.png"); fig.savefig(f); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[e10 plots] lỗi: {type(ex).__name__}: {ex}")
    L.append("\n## Tệp\n")
    L.append("`e10_fam.csv` (ảnh × họ: bản đồ đầy đủ, nửa 1, ổn định), `e10_paths.csv` (từng đường), `e10_gate.csv`, `e10_images.csv`, `meta.json`.\n")
    for f in files:
        L.append(f"- `{os.path.basename(f)}`")
    path = os.path.join(out, "E10_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[e10 report] {path}")
    return path
