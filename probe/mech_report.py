"""MECHANISM_REPORT.md + đồ thị cho E1–E3 (đọc CSV trong <out>/mechanism)."""

from __future__ import annotations

import json
import math
import os
import warnings

import numpy as np

from .report import _read, boot_ci, boot_spearman, fmt, md, q10, q90

COL = {"absSum": "#9a998f", "across": "#2a78d6", "between": "#eb6834", "within": "#1baf7a"}
FAM = {"ig": "#2a78d6", "spi10": "#eb6834"}
MUTED = "#6b6a63"


def _plt():
    from .plots import _setup
    return _setup()


# ─────────────────────────────────────────────────────────────────────────────
# bảng dẫn xuất
# ─────────────────────────────────────────────────────────────────────────────

def image_table(dm):
    """Một hàng mỗi ảnh: IG (K=1) và SPI (K lớn nhất), thành phần F."""
    import pandas as pd
    im = dm[(dm.level == "image") & (dm.component == "F")]
    ig = im[im.method == "ig"].set_index("image")
    sp = im[im.method == "spi10"]
    sp = sp[sp.K == sp.K.max()].set_index("image")
    t = pd.DataFrame(index=sp.index)
    t["K"] = sp.K
    t["W_ig"], t["EW_spi"] = ig.EW, sp.EW
    t["exposure_ratio"] = sp.EW / ig.EW
    t["L1_ig"], t["L1bar_spi"] = ig.L1bar, sp.L1bar
    for k in ("within", "across"):
        t[f"{k}_frac_ig"] = ig[k] / ig.EW
    for k in ("within", "between", "across"):
        t[f"{k}_frac_spi"] = sp[k] / sp.EW
    # ngân sách: L1_IG − L1bar_SPI = (W_IG − EW) + (within_SPI − within_IG) + between_SPI
    t["budget"] = ig.L1bar - sp.L1bar
    t["b_exposure"] = ig.EW - sp.EW
    t["b_within"] = sp.within - ig.within
    t["b_between"] = sp.between
    for reg, sfx in (("in", "_in"), ("out", None)):
        if sfx:
            W_i, W_s, L_i, L_s = ig.EW_in, sp.EW_in, ig.L1bar_in, sp.L1bar_in
            wi_i, wi_s, bt = ig.within_in, sp.within_in, sp.between_in
        else:
            W_i, W_s = ig.EW - ig.EW_in, sp.EW - sp.EW_in
            L_i, L_s = ig.L1bar - ig.L1bar_in, sp.L1bar - sp.L1bar_in
            wi_i, wi_s, bt = ig.within - ig.within_in, sp.within - sp.within_in, sp.between - sp.between_in
        t[f"budget_{reg}"] = L_i - L_s
        t[f"b_exposure_{reg}"] = W_i - W_s
        t[f"b_within_{reg}"] = wi_s - wi_i
        t[f"b_between_{reg}"] = bt
        t[f"L1share_{reg}_ig"] = L_i / ig.L1bar
        t[f"L1share_{reg}_spi"] = L_s / sp.L1bar
    for m in ("iou", "energy_mask", "insdel_blur", "insdel_black"):
        if m in sp:
            t[f"{m}_ig"], t[f"{m}_spi"] = ig[m], sp[m]
            t[f"d_{m}"] = sp[m] - ig[m]
    return t.reset_index()


def path_level(dm):
    p = dm[(dm.level == "path") & (dm.component == "F")]
    ig = p[p.method == "ig"].set_index("image").W
    s = p[p.method == "spi10"].copy()
    s["W_rel_ig"] = s.W / s.image.map(ig)
    s["within_frac"] = s.within / s.W
    s["L1_over_W"] = s.L1 / s.W
    s["W_in_frac"] = s.W_in / s.W
    return s


# ─────────────────────────────────────────────────────────────────────────────
# đồ thị
# ─────────────────────────────────────────────────────────────────────────────

def plot_e1(out, plt, q, dp, toy):
    files = []
    if q is not None and len(q):
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        qq = q[q.quad != "left"].copy()
        qq["M"] = qq.quad.astype(int)
        Mmax = qq.M.max()
        for fam, g in qq[qq.M < Mmax].groupby("path"):
            for a, col, lab in zip(ax[:2], ["A_dist_l1", "L1_relerr"], ["‖A_M − A_ref‖₁/‖A_ref‖₁", "|‖A_M‖₁ − ‖A_ref‖₁|/‖A_ref‖₁"]):
                m = g.groupby("M")[col].median()
                a.plot(m.index, m.values, marker="o", color=FAM.get(fam, MUTED), label=fam)
                a.set_ylabel(lab)
            m = g.groupby("M")["W_relerr"].median()
            ax[1].plot(m.index, m.values, marker="s", ls="--", color=FAM.get(fam, MUTED), label=f"{fam} W")
        for a in ax[:2]:
            a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel(f"M (điểm/đoạn; ref M={Mmax})"); a.legend(fontsize=7)
        ax[0].set_title("đường cố định: hội tụ vector A")
        ax[1].set_title("sai lệch ‖A‖₁ (nét liền) và W (nét đứt)")
        if dp is not None and len(dp):
            m = dp.groupby("N")[["W_ratio_fine", "within_ratio_fine", "A_dist_l1_fine"]].median()
            for col, c in zip(m.columns, ["#2a78d6", "#1baf7a", "#eb6834"]):
                ax[2].plot(m.index, m[col], marker="o", color=c, label=col)
            ax[2].axhline(1, color=MUTED, lw=0.8)
            ax[2].set_xscale("log", base=2); ax[2].set_xlabel("N (cùng đường DP, lưới gộp)")
            ax[2].set_title("gộp bước nhảy DP (so với N lớn nhất)"); ax[2].legend(fontsize=7)
        fig.tight_layout()
        f = os.path.join(out, "e1_numerical.png"); fig.savefig(f); plt.close(fig); files.append(f)
    if toy is not None and len(toy):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        t = toy.sort_values("N")
        ax[0].plot(t.N, t.perpath_A_dist_fine, marker="o", color="#eb6834", label="từng đường: ‖A_N − A_mịn‖₁/‖A_mịn‖₁")
        ax[0].plot(t.N, t.mean_A_dist_fine, marker="s", color="#2a78d6", label="trung bình: ‖Ā_N − Ā_mịn‖₁/‖Ā_mịn‖₁")
        ax[0].plot(t.N, t.mc_se_mean_A, color=MUTED, ls="--", lw=1, label="mức nhiễu MC của Ā")
        ax[0].set_xscale("log", base=2); ax[0].set_yscale("log"); ax[0].legend(fontsize=7); ax[0].set_xlabel("N")
        for col in ("within", "between", "across"):
            ax[1].plot(t.N, t[col] / t.EW, marker="o", color=COL[col], label=f"{col}/EW")
        ax[1].set_xscale("log", base=2); ax[1].legend(fontsize=7); ax[1].set_xlabel("N"); ax[1].set_title("toy, phân rã chính xác")
        fig.tight_layout()
        f = os.path.join(out, "e1_toy.png"); fig.savefig(f); plt.close(fig); files.append(f)
    return files


def plot_e2(out, plt, dm, it):
    files = []
    n = len(it)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n + 2), 4))
    segs = ["absSum", "across", "between", "within"]
    im = dm[(dm.level == "image") & (dm.component == "F")]
    for i, (_, r) in enumerate(it.iterrows()):
        ig = im[(im.image == r.image) & (im.method == "ig")].iloc[0]
        sp = im[(im.image == r.image) & (im.method == "spi10") & (im.K == r.K)].iloc[0]
        for off, row in ((-0.2, ig), (0.2, sp)):
            bottom = 0.0
            for sname in segs:
                v = (row[sname] if sname != "between" else (row["between"] if row.method == "spi10" else 0.0)) / ig.EW
                ax.bar(i + off, v, 0.36, bottom=bottom, color=COL[sname], edgecolor="white", linewidth=1,
                       label=sname if i == 0 and off > 0 else None)
                bottom += v
    ax.set_xticks(range(n), [f"{s}\nIG|SPI" for s in it.image], fontsize=6)
    ax.set_ylabel("÷ W_IG  (chiều cao cột = exposure)")
    ax.set_title("E W = |ΣĀ| + across + between + within; cột trái IG, cột phải SPI")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    f = os.path.join(out, "e2_decomposition.png"); fig.savefig(f); plt.close(fig); files.append(f)

    sp = im[im.method == "spi10"].copy()
    sp["between_frac"] = sp.between / sp.EW
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, col, lab in zip(ax, ["between_frac", "iou", "insdel_blur"], ["between/EW", "IoU(Ā_K)", "insdel_blur(Ā_K)"]):
        if col not in sp:
            continue
        g = sp.groupby("K")[col]
        med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        a.fill_between(med.index, lo, hi, color="#eb6834", alpha=0.2, linewidth=0)
        a.plot(med.index, med.values, marker="o", color="#eb6834", label="SPI (median, IQR)")
        if col != "between_frac" and f"{col}_ig" in it:
            a.axhline(it[f"{col}_ig"].median(), color="#2a78d6", lw=1.5, ls="--", label="IG (median)")
        a.set_xscale("log", base=2); a.set_xlabel("K đường"); a.set_ylabel(lab); a.legend(fontsize=7)
    fig.tight_layout()
    f = os.path.join(out, "e2_k_curve.png"); fig.savefig(f); plt.close(fig); files.append(f)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, xcol, lab in zip(ax[:2], ["between_frac_spi", "exposure_ratio"], ["between/EW (SPI)", "E W_SPI / W_IG"]):
        a.scatter(it[xcol], it.d_iou, s=30, color="#eb6834", edgecolor="white", linewidth=1)
        a.axhline(0, color=MUTED, lw=0.8); a.set_xlabel(lab); a.set_ylabel("IoU(Ā_SPI) − IoU(IG)")
    pl = path_level(dm)
    rs = []
    for imn, g in pl.groupby("image"):
        rs.append((imn, g.W.rank().corr(g.iou.rank()), g.L1_over_W.rank().corr(g.iou.rank())))
    ax[2].scatter(range(len(rs)), [r[1] for r in rs], color="#2a78d6", label="Spearman(W_γ, IoU(A^γ))", s=24)
    ax[2].scatter(range(len(rs)), [r[2] for r in rs], color="#1baf7a", marker="D", label="Spearman(‖A^γ‖₁/W_γ, IoU)", s=20)
    ax[2].axhline(0, color=MUTED, lw=0.8); ax[2].set_xlabel("ảnh"); ax[2].legend(fontsize=7)
    ax[2].set_title("trong từng ảnh, giữa các đường")
    fig.tight_layout()
    f = os.path.join(out, "e2_relations.png"); fig.savefig(f); plt.close(fig); files.append(f)
    return files


def plot_e3(out, plt, e3):
    if e3 is None or not len(e3):
        return []
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    h = e3[e3.u == "high"].groupby("rho")
    m = h[["W_ratio", "pred_att", "L1bar_ratio"]].median()
    ax[0].plot(m.index, m.W_ratio, marker="o", color="#eb6834", label="E W_H(SPI) / W_H(IG)")
    ax[0].plot(m.index, m.L1bar_ratio, marker="s", color="#1baf7a", label="‖Ā_H(SPI)‖₁ / ‖A_H(IG)‖₁")
    ax[0].plot(m.index, m.pred_att, color=MUTED, ls="--", lw=1, label="a(ρ) dự đoán")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("ρ"); ax[0].legend(fontsize=7)
    m = h[["within_frac_spi", "between_frac_spi", "within_frac_ig"]].median()
    for col, c in zip(m.columns, ["#1baf7a", "#eb6834", "#2a78d6"]):
        ax[1].plot(m.index, m[col], marker="o", color=c, label=col)
    ax[1].set_xscale("log"); ax[1].set_xlabel("ρ"); ax[1].set_title("phân rã của riêng H"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    f = os.path.join(out, "e3_trap.png"); fig.savefig(f); plt.close(fig)
    return [f]


def plot_maps(out, plt):
    import torch
    p = os.path.join(out, "e2_heatmaps.pt")
    if not os.path.exists(p):
        return []
    files = []
    from .data import MEAN, STD
    for im, h in torch.load(p, weights_only=False).items():
        fig, ax = plt.subplots(1, 5, figsize=(13, 2.9))
        img = (h["x"][0].float() * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)).clamp(0, 1)
        ax[0].imshow(img.permute(1, 2, 0).numpy())
        ax[0].contour(h["mask"].numpy().astype(float), levels=[0.5], colors="white", linewidths=0.8)
        vW = float(np.quantile(torch.cat([h["W_ig"].flatten(), h["EW_spi"].flatten()]).numpy(), 0.995))
        vA = float(np.quantile(torch.cat([h["A_ig"].flatten(), h["Abar_spi"].flatten()]).numpy(), 0.995))
        for a, k, v, t in zip(ax[1:], ["W_ig", "EW_spi", "A_ig", "Abar_spi"], [vW, vW, vA, vA],
                              ["W IG", "E W SPI", "|A| IG", "|Ā| SPI"]):
            a.imshow(h[k].numpy(), cmap="Blues", vmin=0, vmax=v + 1e-30)
            a.set_title(t + " (cùng thang trong cặp)", fontsize=8)
        for a in ax:
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
        fig.suptitle(im, fontsize=9)
        fig.tight_layout()
        f = os.path.join(out, f"e2_maps_{im}.png"); fig.savefig(f); plt.close(fig); files.append(f)
    return files


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo
# ─────────────────────────────────────────────────────────────────────────────

def e3_table(dm):
    import pandas as pd
    d = dm[(dm.level == "image") & (dm.component == "H")]
    if not len(d):
        return None
    ig = d[d.method == "ig"].set_index(["image", "trap"])
    sp = d[d.method == "spi10"]
    sp = sp[sp.K == sp.K.max()].set_index(["image", "trap"])
    t = pd.DataFrame(index=sp.index)
    for c in ("rho", "eta", "u", "pred_att"):
        t[c] = sp[c]
    t["W_ratio"] = sp.EW / ig.EW
    t["L1bar_ratio"] = sp.L1bar / ig.L1bar
    t["within_frac_spi"] = sp.within / sp.EW
    t["between_frac_spi"] = sp.between / sp.EW
    t["within_frac_ig"] = ig.within / ig.EW
    fh = dm[(dm.level == "image") & (dm.component == "F+H")]
    f0 = dm[(dm.level == "image") & (dm.component == "F")]
    for fam, tag in (("ig", "ig"), ("spi10", "spi")):
        a = fh[fh.method == fam]
        b = f0[f0.method == fam]
        if fam == "spi10":
            a = a[a.K == a.K.max()]
            b = b[b.K == b.K.max()]
        a = a.set_index(["image", "trap"])
        b = b.set_index("image")
        idx = a.index
        t[f"dEW_FH_{tag}"] = (a.EW.values - b.loc[[i for i, _ in idx]].EW.values) / b.loc[[i for i, _ in idx]].EW.values
        t[f"dL1bar_FH_{tag}"] = (a.L1bar.values - b.loc[[i for i, _ in idx]].L1bar.values) / b.loc[[i for i, _ in idx]].L1bar.values
    return t.reset_index()


def write(out: str, cfg: dict) -> str:
    import pandas as pd
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    plt = _plt()
    meta = json.load(open(os.path.join(out, "meta.json"))) if os.path.exists(os.path.join(out, "meta.json")) else {}
    q, dp, toy = _read(out, "e1_quad"), _read(out, "e1_dp"), _read(out, "e1_toy")
    dm, imgs = _read(out, "mechanism_decomposition"), _read(out, "e2_images")
    M2 = int(cfg["mechanism"]["e2"]["M"])
    L = ["# MECHANISM_REPORT — E1–E3: SPI tránh exposure, triệt tiêu trong đường, hay triệt tiêu giữa đường?\n"]
    for st in ("e1", "e23"):
        m = meta.get(st)
        if m:
            L.append(f"- **{st}**: {len(m['images'])} ảnh, {m['minutes']:.1f} phút (ước tính trước "
                     f"{m['estimate']['e1_min' if st == 'e1' else 'e2_min']:.1f}), GPU `{m['gpu']}`, "
                     f"hash `{m['digest'][:12]}`" + (" — **DEBUG, không dùng để kết luận**" if m.get("debug") else ""))
    L.append("")
    L.append("""## Định nghĩa

Trên **cùng** các điểm cầu phương của một đường γ: A^γ_i = ∫∂_iF dx_i, W_γ = Σ_i ∫|∂_iF||dx_i| (exposure).
Với Ā = trung bình K đường, D = F(x) − F(x′):

E W − |D| = **within** E[W − ‖A^γ‖₁] + **between** (E‖A^γ‖₁ − ‖Ā‖₁) + **across** (‖Ā‖₁ − |ΣĀ|) + **resid** (|ΣĀ| − |D|).

Ba số hạng đầu không âm chính xác với mọi cầu phương (bất đẳng thức tam giác trên cùng điểm); resid là sai số
completeness của ước lượng. Đây là đẳng thức kế toán, chưa phải định nghĩa nhiễu: phần bị triệt tiêu vẫn có thể là
tác động chức năng thật.

**Ngân sách so với IG** (IG một đường, between = 0), chính xác theo cấu tạo:
‖A_IG‖₁ − ‖Ā_SPI‖₁ = **exposure** (W_IG − E W_SPI) + **within thêm** (within_SPI − within_IG) + **between** (between_SPI).
Tách theo vùng (trong / ngoài mask) cũng chính xác vì ‖·‖₁ và W cộng theo phần tử.

W bất biến khi đổi thang từng toạ độ nên không phụ thuộc chuẩn hoá input. Mọi đại lượng tính trong không gian input
đã chuẩn hoá, gradient của logit lớp dự đoán, baseline đen; ảnh, seed và 32 đường SPI đầu trùng pilot.
Cầu phương: mỗi đoạn chia M tầng, một điểm đều mỗi tầng (dùng chung mọi phần tử), trọng số Δx/M.
""")

    # ── E1 ─────────────────────────────────────────────────────────────────
    L.append("## E1. Kiểm định số học\n")
    gate = {}
    if q is not None and len(q):
        qq = q.copy()
        Mmax = int(qq[qq.quad != "left"].quad.astype(int).max())
        ref = qq[qq.quad == str(Mmax)].set_index(["image", "path", "p"])
        qq = qq.join(ref[["W", "L1", "within"]], on=["image", "path", "p"], rsuffix="_ref")
        qq["within_bias_rel_W"] = (qq.within - qq.within_ref) / qq.W_ref
        qq["L1_bias_rel_W"] = (qq.L1 - qq.L1_ref) / qq.W_ref
        L.append(f"### E1a. Đường cố định, tăng số điểm cầu phương (ref: M = {Mmax}, `left` = nút trái)\n")
        L.append("`A_dist_l1` = ‖A − A_ref‖₁/‖A_ref‖₁. `L1_bias_rel_W` = (‖A‖₁ − ‖A_ref‖₁)/W_ref: độ chệch lên của "
                 "‖A‖₁ do nhiễu cầu phương (Jensen), quy về đơn vị W. `within_bias_rel_W` = chệch của số hạng within.\n")
        t = qq.groupby(["path", "quad"]).agg(points=("points", "median"), A_dist_l1=("A_dist_l1", "median"),
                                             A_dist_l2=("A_dist_l2", "median"), W_relerr=("W_relerr", "median"),
                                             L1_bias_rel_W=("L1_bias_rel_W", "median"),
                                             within_bias_rel_W=("within_bias_rel_W", "median"),
                                             within_frac=("within", "median"), abs_resid=("resid_rel", lambda s: s.abs().median()))
        L.append(md(t, index=True))
        sel = qq[qq.quad == str(M2)]
        gate["quad_L1_bias"] = float(sel.L1_bias_rel_W.abs().median()) if len(sel) else float("nan")
        gate["quad_within_bias"] = float(sel.within_bias_rel_W.abs().median()) if len(sel) else float("nan")
    if dp is not None and len(dp):
        L.append("### E1b. Gộp bước nhảy DP: cùng một đường DP ở nhiều độ phân giải, cùng tổng số điểm\n")
        L.append("Lưới thô = cộng gộp số gia của lưới mịn (bước nhảy của các phần tử trong cùng ô bị gộp thành một đoạn chéo). "
                 "Tỉ số so với N lớn nhất; `A_dist_l1_fine` = ‖A_N − A_mịn‖₁/‖A_mịn‖₁ trên cùng realization.\n")
        L.append(md(dp.groupby("N")[["M", "W_ratio_fine", "within_ratio_fine", "A_dist_l1_fine", "resid_rel"]].median(), index=True))
        Nm = int(cfg["paths"]["spi10"]["steps"])
        s = dp[dp.N == Nm]
        gate["dp_W"] = float((s.W_ratio_fine - 1).abs().median()) if len(s) else float("nan")
        gate["dp_within"] = float((s.within_ratio_fine - 1).abs().median()) if len(s) else float("nan")
    if toy is not None and len(toy):
        L.append("### E1c. Toy tương tác (thứ tự bước nhảy)\n")
        r0 = toy.iloc[0]
        L.append(f"F(z) = b·z + ½zᵀCz, d = {int(r0.d)}, C đối xứng có dấu ngẫu nhiên (gradient đổi dấu dọc đường → có triệt tiêu), "
                 f"{int(r0.paths)} đường DP (α như SPI), lưới mịn N = {int(toy.N.max())}. Trên đường gấp khúc, ∂F tuyến tính "
                 "trong mỗi đoạn nên A và W tính **chính xác** (không có sai số cầu phương). Lưới thô = gộp lưới mịn.\n")
        cols = ["N", "EW", "within", "between", "across", "resid", "perpath_A_dist_fine", "perpath_W_ratio_fine",
                "mean_A_dist_fine", "mc_se_mean_A", "W_ig", "L1_ig"]
        L.append(md(toy.sort_values("N")[cols]))
        L.append("Đọc: `perpath_A_dist_fine` lớn mà `mean_A_dist_fine` ≈ mức nhiễu MC → gộp bước nhảy đổi từng đường "
                 "nhưng không đổi kỳ vọng (với F bậc hai, độc lập giữa các phần tử làm E[A] bất biến). Khi đó within/between "
                 "theo từng đường phụ thuộc N, còn Ā thì không.\n")

    # ── E2 ─────────────────────────────────────────────────────────────────
    L.append("## E2. Cơ chế trên gradient ResNet thật\n")
    it = None
    if dm is not None and len(dm):
        it = image_table(dm)
        K = int(it.K.max())
        L.append(f"{len(it)} ảnh, SPI K = {K} đường/ảnh, M = {M2} điểm/đoạn. Bootstrap theo **ảnh** (đường không coi là "
                 "quan sát độc lập); phân tán giữa đường dùng để mô tả bất định trong ảnh.\n")
        L.append("### E2a. Theo ảnh\n")
        cols = ["image", "W_ig", "EW_spi", "exposure_ratio", "within_frac_ig", "within_frac_spi", "between_frac_spi",
                "across_frac_ig", "across_frac_spi", "L1_ig", "L1bar_spi", "d_iou"]
        L.append(md(it[[c for c in cols if c in it]]))
        L.append("### E2b. Ngân sách: ‖A_IG‖₁ − ‖Ā_SPI‖₁ = exposure + within thêm + between\n")
        rows = []
        for reg, sfx in (("toàn ảnh", ""), ("trong mask", "_in"), ("ngoài mask", "_out")):
            bud = it[f"budget{sfx}"]
            for comp in ("exposure", "within", "between"):
                share = it[f"b_{comp}{sfx}"] / bud
                est, lo, hi, n = boot_ci(share.values)
                rows.append({"vùng": reg, "thành phần": comp, "share_mean": est, "lo95": lo, "hi95": hi,
                             "abs_mean_÷W_IG": float((it[f"b_{comp}{sfx}"] / it.W_ig).mean()), "n": n})
            rows.append({"vùng": reg, "thành phần": "tổng ‖A_IG‖₁−‖Ā‖₁", "share_mean": 1.0, "lo95": float("nan"),
                         "hi95": float("nan"), "abs_mean_÷W_IG": float((bud / it.W_ig).mean()), "n": len(it)})
        bt = pd.DataFrame(rows)
        L.append("`share` = thành phần / tổng (trung bình theo ảnh, bootstrap 95%). Exposure âm nghĩa là SPI có exposure "
                 "**lớn hơn** IG. Một share có thể > 1 khi thành phần khác âm.\n")
        L.append(md(bt))
        L.append("**Phần ‖·‖₁ nằm trong mask:** IG " + fmt(it.L1share_in_ig.mean()) + ", SPI " + fmt(it.L1share_in_spi.mean())
                 + " (trung bình theo ảnh).\n")
        pl = path_level(dm)
        L.append("### E2c. Từng đường (K = 1)\n")
        L.append("Mỗi ảnh lấy median / q10 / q90 qua các đường, rồi trung bình qua ảnh. `W_rel_ig` = W_γ/W_IG; "
                 "`frac_paths_below_ig` = tỉ lệ đường có W_γ < W_IG.\n")
        per = pl.groupby("image").agg(W_rel_med=("W_rel_ig", "median"), W_rel_q10=("W_rel_ig", q10), W_rel_q90=("W_rel_ig", q90),
                                      frac_paths_below_ig=("W_rel_ig", lambda s: float((s < 1).mean())),
                                      within_frac_med=("within_frac", "median"), L1_over_W_med=("L1_over_W", "median"),
                                      iou_K1_med=("iou", "median"))
        L.append(md(per.mean().to_frame("trung bình qua ảnh").T))
        sp = dm[(dm.level == "image") & (dm.component == "F") & (dm.method == "spi10")].copy()
        sp["between_frac"] = sp.between / sp.EW
        L.append("### E2d. Theo K (Ā_K từ K đường đầu)\n")
        L.append("between_K dùng ‖Ā_K‖₁ hữu hạn K, vốn bị nhiễu MC thổi phồng → between_K **thấp hơn** giá trị K→∞; "
                 "đường cong theo K cho thấy nó đã bão hoà hay chưa.\n")
        kc = [c for c in ["between_frac", "iou", "energy_mask", "insdel_blur", "insdel_black"] if c in sp]
        L.append(md(sp.groupby("K")[kc].median(), index=True))
        L.append("### E2e. Quan hệ với chất lượng\n")
        rel = []
        for xc in ("exposure_ratio", "within_frac_spi", "between_frac_spi"):
            for yc in ("d_iou", "d_insdel_blur", "d_energy_mask"):
                if yc in it:
                    e, lo, hi, n = boot_spearman(it[xc], it[yc])
                    rel.append({"x (theo ảnh)": xc, "y": yc, "spearman": e, "lo95": lo, "hi95": hi, "n": n})
        L.append("Giữa các ảnh (y = SPI − IG):\n")
        L.append(md(pd.DataFrame(rel)))
        rel = []
        for xc in ("W", "L1_over_W", "W_in_frac", "within_frac"):
            for yc in ("iou", "energy_mask", "insdel_blur"):
                if yc not in pl:
                    continue
                per_img = [g[xc].rank().corr(g[yc].rank()) for _, g in pl.groupby("image")]
                e, lo, hi, n = boot_ci(per_img)
                rel.append({"x (từng đường)": xc, "y (A^γ)": yc, "mean_spearman_trong_ảnh": e, "lo95": lo, "hi95": hi, "n_ảnh": n})
        L.append("Trong từng ảnh, giữa các đường (Spearman tính trong ảnh, rồi trung bình qua ảnh, bootstrap theo ảnh):\n")
        L.append(md(pd.DataFrame(rel)))
        L.append("Tín hiệu cộng tính tuyến tính: canonical giữ đúng 100% trên mọi đường không phụ thuộc mô hình (pilot mục B3), "
                 "nên retention không phân biệt được các cơ chế ở đây; không đưa vào phép thử quan hệ.\n")
        pm = _read(C_out_pilot(cfg), "maps")
        if pm is not None and len(pm):
            a = pm[(pm.path == "spi10") & (pm.rule == "canon_rand")].set_index("image").iou
            b = pm[(pm.path == "ig") & (pm.rule == "canon_rand")].set_index("image").iou
            s32 = sp[sp.K == min(32, int(sp.K.max()))].set_index("image").iou
            ig2 = it.set_index("image").iou_ig if "iou_ig" in it else None
            d1 = (s32 - a).dropna().abs()
            d2 = (ig2 - b).dropna().abs() if ig2 is not None else None
            L.append(f"**Đối chiếu với pilot** (cùng ảnh, cùng 32 đường; khác cầu phương): |IoU SPI K=32 − pilot| median "
                     f"{fmt(d1.median())} (n={len(d1)}); |IoU IG − pilot| median {fmt(d2.median() if d2 is not None else float('nan'))}.\n")

    # ── E3 ─────────────────────────────────────────────────────────────────
    e3 = e3_table(dm) if dm is not None and len(dm) else None
    L.append("## E3. Bẫy gradient đã biết hình học\n")
    if e3 is not None and len(e3):
        L.append("Phân rã áp lên **riêng H** (cùng đường, cùng điểm cầu phương với E2). Hình học đã biết: trên đường IG, "
                 "∂_iH = κ sin²(πs)u_i giữ dấu → within_H(IG) ≈ 0; đường SPI lệch khỏi ống r = ρ‖Δx‖ nên kỳ vọng exposure "
                 "giảm theo a(ρ) và gần như không cần triệt tiêu. Nếu phép đo đúng, `W_ratio` ≈ `pred_att` và "
                 "within/between của H nhỏ. `dEW_FH_*` / `dL1bar_FH_*` = thay đổi tương đối của E W và ‖Ā‖₁ khi thêm H vào F.\n")
        L.append(md(e3.groupby(["u", "rho"])[["eta", "pred_att", "W_ratio", "L1bar_ratio", "within_frac_spi", "between_frac_spi",
                                             "within_frac_ig", "dEW_FH_ig", "dEW_FH_spi", "dL1bar_FH_ig", "dL1bar_FH_spi"]].median(),
                    index=True))
    else:
        L.append("_(không có dữ liệu bẫy)_\n")

    # ── chỉ báo ────────────────────────────────────────────────────────────
    L.append("## Tiêu chí quyết định (đặt trước) và số đo tương ứng\n")
    if it is not None:
        bud = it.budget
        sh = {c: boot_ci((it[f"b_{c}"] / bud).values) for c in ("exposure", "within", "between")}
        order = sorted(sh, key=lambda c: -sh[c][0])
        top, second = order[0], order[1]
        sep = sh[top][1] > sh[second][2]
        pl = path_level(dm)
        frac_below = pl.groupby("image").W_rel_ig.apply(lambda s: float((s < 1).mean())).mean()
        L.append(f"- **Geometric avoidance** (W_γ thấp ngay ở phần lớn đường K=1): tỉ lệ đường có W_γ < W_IG = "
                 f"**{fmt(frac_below)}**; E W_SPI/W_IG median **{fmt(it.exposure_ratio.median())}**; share exposure "
                 f"{fmt(sh['exposure'][0])} [{fmt(sh['exposure'][1])}, {fmt(sh['exposure'][2])}].")
        L.append(f"- **Within cancellation** (W lớn nhưng W − ‖A^γ‖₁ lớn): within/W của SPI median "
                 f"**{fmt(it.within_frac_spi.median())}** so với IG **{fmt(it.within_frac_ig.median())}**; share within thêm "
                 f"{fmt(sh['within'][0])} [{fmt(sh['within'][1])}, {fmt(sh['within'][2])}].")
        L.append(f"- **Between cancellation** (mỗi A^γ lớn nhưng E‖A^γ‖₁ − ‖Ā‖₁ lớn): between/EW median "
                 f"**{fmt(it.between_frac_spi.median())}**; share between {fmt(sh['between'][0])} "
                 f"[{fmt(sh['between'][1])}, {fmt(sh['between'][2])}].")
        L.append(f"- Thành phần lớn nhất trong ngân sách: **{top}**" + (f", tách khỏi {second} (CI không chồng)." if sep
                                                                            else f", CI chồng với {second} → chưa tách được."))
        L.append("- **Có dự đoán chất lượng không?** xem E2e: một chỉ số chỉ giải thích chất lượng nếu CI của Spearman "
                 "không chứa 0 ở mức ảnh hoặc mức đường; nếu không, nó chỉ mô tả gradient.")
    if gate:
        L.append(f"- **Cổng số học** (để đọc các share trên): ở M = {M2}, |chệch ‖A‖₁|/W ≈ {fmt(gate.get('quad_L1_bias'))}, "
                 f"|chệch within|/W ≈ {fmt(gate.get('quad_within_bias'))} (E1a); gộp bước nhảy DP ở N = "
                 f"{cfg['paths']['spi10']['steps']}: |W/W_mịn − 1| ≈ {fmt(gate.get('dp_W'))}, |within/within_mịn − 1| ≈ "
                 f"{fmt(gate.get('dp_within'))} (E1b). So các số này với within/EW và between/EW ở E2: nếu cùng cỡ thì "
                 "phân rã chưa đáng tin ở mức đó.")
    L.append("")
    files = []
    for fn_ in (lambda: plot_e1(out, plt, q, dp, toy), lambda: plot_e2(out, plt, dm, it) if it is not None else [],
                lambda: plot_e3(out, plt, e3), lambda: plot_maps(out, plt)):
        try:
            files += fn_()
        except Exception as e:
            print(f"[mech plots] lỗi: {type(e).__name__}: {e}")
    L.append("## Tệp\n")
    L.append("`mechanism_decomposition.csv` (level = path: từng đường; level = image: từng ảnh × họ đường × K × thành phần "
             "F / H / F+H), `e1_quad.csv`, `e1_dp.csv`, `e1_toy.csv`, `e2_images.csv`, `meta.json`.\n")
    for f in files:
        L.append(f"- `{os.path.basename(f)}`")
    path = os.path.join(out, "MECHANISM_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[mech report] {path}")
    return path


def C_out_pilot(cfg):
    from . import config as C
    return C.out_dir(cfg, "pilot")
