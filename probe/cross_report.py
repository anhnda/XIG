"""CROSS_REPORT.md + đồ thị cho E1′ (gate) và E4 (đọc CSV trong <out>/cross)."""

from __future__ import annotations

import json
import math
import os
import warnings

import numpy as np

from .report import _read, boot_ci, fmt, md, q10, q90

MODES = ("cm_low", "cm_high", "cd_low", "cd_high")
MCOL = {"ig": "#2a78d6", "spi10": "#eb6834", "guided": "#1baf7a", "blur": "#eda100", "line_blur": "#9a998f"}
MODECOL = {"cm_low": "#2a78d6", "cm_high": "#eb6834", "cd_low": "#1baf7a", "cd_high": "#eda100"}
MUTED = "#6b6a63"
REF = {"ig": "ig", "spi10": "ig", "guided": "ig", "blur": "line_blur", "line_blur": "line_blur"}


def _plt():
    from .plots import _setup
    return _setup()


def cluster_spearman(df, xcol, ycol, seed=0, B=2000):
    """Spearman gộp các hàng; bootstrap theo ảnh (giữ mọi hàng của ảnh được rút)."""
    d = df[["image", xcol, ycol]].dropna()
    if d.image.nunique() < 4:
        return float("nan"), float("nan"), float("nan"), len(d)
    est = float(d[xcol].rank().corr(d[ycol].rank()))
    groups = {k: g for k, g in d.groupby("image")}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    bs = []
    import pandas as pd
    for _ in range(B):
        s = pd.concat([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        v = s[xcol].rank().corr(s[ycol].rank())
        if np.isfinite(v):
            bs.append(v)
    return est, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), len(d)


def within_image_spearman(df, xcol, ycol, group="image"):
    vals = []
    for _, g in df.groupby(group):
        g = g[[xcol, ycol]].dropna()
        if len(g) >= 3:
            v = g[xcol].rank().corr(g[ycol].rank())
            if np.isfinite(v):
                vals.append(v)
    return boot_ci(vals)


# ─────────────────────────────────────────────────────────────────────────────
# bảng dẫn xuất
# ─────────────────────────────────────────────────────────────────────────────

def with_ref(m):
    """Thêm tỉ số so với tham chiếu (cùng ảnh) cho e4_methods."""
    m = m.copy()
    r = m.set_index(["image", "method"])
    def refval(row, col):
        try:
            return r.loc[(row.image, REF[row.method]), col]
        except KeyError:
            return float("nan")
    m["W_ref"] = m.apply(lambda q: refval(q, "EW"), axis=1)
    m["exposure_ratio"] = m.EW / m.W_ref
    m["r_in"] = m.EW_in / m.apply(lambda q: refval(q, "EW_in"), axis=1)
    m["r_out"] = (m.EW - m.EW_in) / m.apply(lambda q: refval(q, "EW") - refval(q, "EW_in"), axis=1)
    for md_ in MODES:
        m[f"r_{md_}"] = m[f"W_{md_}"] / m.apply(lambda q, c=f"W_{md_}": refval(q, c), axis=1)
    lo = m.W_cm_low + m.W_cd_low
    hi = m.W_cm_high + m.W_cd_high
    lo_r = m.apply(lambda q: refval(q, "W_cm_low") + refval(q, "W_cd_low"), axis=1)
    hi_r = m.apply(lambda q: refval(q, "W_cm_high") + refval(q, "W_cd_high"), axis=1)
    m["r_low"], m["r_high"] = lo / lo_r, hi / hi_r
    m["S_freq"] = np.log(m.r_low / m.r_high)                  # > 0: tránh tần số cao nhiều hơn tần số thấp
    m["S_cm_high"] = np.log(m.r_cm_low / m.r_cm_high)
    m["S_mask"] = np.log(m.r_in / m.r_out)                    # > 0: giữ exposure trong vật hơn ngoài vật (dùng mask)
    m["len_ratio"] = m.path_len / m.chord_l1
    for k in ("within", "between", "across"):
        m[f"{k}_frac"] = m[k] / m.EW
    for q in ("iou", "insdel_blur", "insdel_black", "energy_mask"):
        if q in m:
            m[f"d_{q}"] = m[q] - m.apply(lambda row, c=q: refval(row, c), axis=1)
            if f"idgi_{q}" in m:
                m[f"idgi_minus_canon_{q}"] = m[f"idgi_{q}"] - m[q]
    return m


def alpha_table(a, m):
    """E4c, mức ảnh, kèm tỉ số so với IG của E4 (cùng ảnh)."""
    im = a[a.level == "image"].copy()
    if m is None or not len(m):
        return im
    ig = m[m.method == "ig"].set_index("image")
    for col, src in (("W_ref", "EW"), ("W_ref_in", "EW_in")):
        im[col] = im.image.map(ig[src])
    im["exposure_ratio"] = im.EW / im.W_ref
    im["r_in"] = im.EW_in / im.W_ref_in
    im["r_out"] = (im.EW - im.EW_in) / (im.W_ref - im.W_ref_in)
    lo_r = im.image.map(ig.W_cm_low + ig.W_cd_low)
    hi_r = im.image.map(ig.W_cm_high + ig.W_cd_high)
    im["r_low"] = (im.W_cm_low + im.W_cd_low) / lo_r
    im["r_high"] = (im.W_cm_high + im.W_cd_high) / hi_r
    im["S_freq"] = np.log(im.r_low / im.r_high)
    im["S_mask"] = np.log(im.r_in / im.r_out)
    im["between_frac"] = im.between / im.EW
    im["within_frac"] = im.within / im.EW
    for q in ("iou", "insdel_blur", "insdel_black"):
        if q in im:
            im[f"d_{q}"] = im[q] - im.image.map(ig[q])
    return im


# ─────────────────────────────────────────────────────────────────────────────
# đồ thị
# ─────────────────────────────────────────────────────────────────────────────

def plots(out, plt, gate, toy, mt, st, at, ap):
    files = []

    def save(fig, name):
        f = os.path.join(out, name)
        fig.tight_layout(); fig.savefig(f); plt.close(fig); files.append(f)

    if gate is not None and len(gate):
        gi = gate[gate.level == "image"].copy()
        gi["between_frac"] = gi.between / gi.EW
        gi["within_frac"] = gi.within / gi.EW
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for im, g in gi[gi.method == "spi10"].groupby("image"):
            ax[0].plot(g.M, g.between_frac, marker="o", color=MCOL["spi10"], alpha=0.7)
            ax[0].plot(g.M, g.within_frac, marker="s", ls="--", color="#1baf7a", alpha=0.7)
        ax[0].plot([], [], color=MCOL["spi10"], marker="o", label="between/EW (SPI)")
        ax[0].plot([], [], color="#1baf7a", marker="s", ls="--", label="within/EW (SPI)")
        ax[0].set_xscale("log", base=2); ax[0].set_xlabel("M (điểm/đoạn)"); ax[0].legend(fontsize=7)
        ax[0].set_title("mỗi đường = một ảnh")
        bud = gate_budget(gate)
        if bud is not None:
            for comp, c in (("exposure", "#2a78d6"), ("within", "#1baf7a"), ("between", "#eb6834")):
                mm = bud.groupby("M")[f"share_{comp}"].mean()
                ax[1].plot(mm.index, mm.values, marker="o", color=c, label=comp)
            ax[1].axhline(0, color=MUTED, lw=0.8)
            ax[1].set_xscale("log", base=2); ax[1].set_xlabel("M"); ax[1].set_ylabel("share của ‖A_IG‖₁ − ‖Ā‖₁")
            ax[1].legend(fontsize=7); ax[1].set_title("ngân sách theo M (ảnh gate)")
        save(fig, "gate_quadrature.png")
    if toy is not None and len(toy):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for fam, g in toy[~toy.is_fine].groupby("family"):
            c = "#2a78d6" if fam == "quadratic" else "#eb6834"
            ax[0].plot(g.N, g.paired_z_rms, marker="o", color=c, label=fam)
            ax[1].plot(g.N, g.paired_bias_rel, marker="o", color=c, label=fam)
        ax[0].axhline(1, color=MUTED, lw=0.8, ls="--")
        ax[0].set_ylabel("rms z của chệch E[A] (≈1: không phát hiện chệch)")
        ax[1].set_ylabel("‖E[A_N − A_mịn]‖₁ / ‖Ā_mịn‖₁")
        for a in ax:
            a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("N"); a.legend(fontsize=7)
        save(fig, "gate_toy.png")
    if mt is not None and len(mt):
        meths = [m for m in ("spi10", "guided", "blur") if m in set(mt.method)]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        w = 0.8 / len(MODES)
        for i, md_ in enumerate(MODES):
            vals = [mt[mt.method == m][f"r_{md_}"].median() for m in meths]
            ax.bar(np.arange(len(meths)) + (i - 1.5) * w, vals, w * 0.92, color=MODECOL[md_], label=md_)
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(meths)), [f"{m}\n(ref {REF[m]})" for m in meths])
        ax.set_ylabel("W_mode / W_mode,ref (median)"); ax.legend(fontsize=7, ncol=4)
        save(fig, "e4_modes.png")
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for a, xcol, lab in zip(ax, ["S_freq", "S_cm_high", "S_mask"],
                                ["S_freq = log(r_low / r_high)", "S_cm_high = log(r_cm_low / r_cm_high)", "S_mask = log(r_in / r_out) [mask]"]):
            for m in meths:
                g = mt[mt.method == m]
                a.scatter(g[xcol], g.d_iou, s=28, color=MCOL[m], label=m, edgecolor="white", linewidth=1)
            a.axhline(0, color=MUTED, lw=0.8); a.axvline(0, color=MUTED, lw=0.8)
            a.set_xlabel(lab); a.set_ylabel("ΔIoU so với ref")
        ax[0].legend(fontsize=7)
        save(fig, "e4_selectivity.png")
    if st is not None and len(st):
        t = st[st.axis == "tau"].groupby(["method", "image", "bin"]).share.mean().reset_index()
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for m, g in t.groupby("method"):
            mm = g.groupby("bin").share.median()
            ax[0].plot((mm.index + 0.5) / (mm.index.max() + 1), mm.values, marker="o", color=MCOL.get(m, MUTED), label=m)
        ax[0].set_xlabel("τ (tỉ lệ L1 đã đi)"); ax[0].set_ylabel("phần exposure (median)"); ax[0].legend(fontsize=7)
        d = st[st.axis == "dist"].groupby(["method", "image", "bin"]).share.mean().reset_index()
        edges = st[st.axis == "dist"].groupby("bin").lo.first()
        for m, g in d.groupby("method"):
            mm = g.groupby("bin").share.median()
            ax[1].plot(mm.index, mm.values, marker="o", color=MCOL.get(m, MUTED), label=m)
        ax[1].set_xticks(list(edges.index), [f"≥{fmt(v)}" for v in edges.values], fontsize=7)
        ax[1].set_xlabel("khoảng cách tới dây cung / ‖Δ‖"); ax[1].set_ylabel("phần exposure (median)")
        save(fig, "e4_state.png")
    if at is not None and len(at) and "exposure_ratio" in at:
        fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
        for a, col in zip(ax, ["exposure_ratio", "S_freq", "iou", "insdel_blur"]):
            if col not in at:
                continue
            g = at.groupby("h")[col]
            med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
            a.fill_between(med.index, lo, hi, color="#eb6834", alpha=0.2, linewidth=0)
            a.plot(med.index, med.values, marker="o", color="#eb6834")
            if col in ("iou", "insdel_blur") and mt is not None:
                a.axhline(mt[mt.method == "ig"][col].median(), color=MCOL["ig"], ls="--", lw=1.2, label="IG")
                a.legend(fontsize=7)
            a.set_xscale("log"); a.set_xlabel("h = 1/(α+1)  (lớn = lệch nhịp hơn)"); a.set_title(col)
        save(fig, "e4c_alpha.png")
    p = os.path.join(out, "e4_heatmaps.pt")
    if os.path.exists(p):
        import torch
        from .data import MEAN, STD
        for im, h in torch.load(p, weights_only=False).items():
            ks = [k for k in ("dW_spi10", "dW_guided", "dW_blur") if k in h]
            fig, ax = plt.subplots(1, len(ks) + 1, figsize=(3 * (len(ks) + 1), 3))
            img = (h["x"][0].float() * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)).clamp(0, 1)
            ax[0].imshow(img.permute(1, 2, 0).numpy())
            ax[0].contour(h["mask"].numpy().astype(float), levels=[0.5], colors="white", linewidths=0.8)
            for a, k in zip(ax[1:], ks):
                v = h[k].numpy()
                lim = float(np.quantile(np.abs(v), 0.995)) + 1e-30
                a.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim)
                a.set_title(f"{k[3:]}: W_ref − W (đỏ = tránh)", fontsize=8)
            for a in ax:
                a.set_xticks([]); a.set_yticks([]); a.grid(False)
            fig.suptitle(im, fontsize=9)
            save(fig, f"e4_avoided_{im}.png")
    return files


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo
# ─────────────────────────────────────────────────────────────────────────────

def gate_budget(gate):
    import pandas as pd
    gi = gate[gate.level == "image"]
    rows = []
    for (im, M), g in gi.groupby(["image", "M"]):
        if set(g.method) < {"ig", "spi10"}:
            continue
        ig, sp = g[g.method == "ig"].iloc[0], g[g.method == "spi10"].iloc[0]
        bud = ig.L1bar - sp.L1bar
        rows.append({"image": im, "M": M, "budget_÷W_IG": bud / ig.EW,
                     "share_exposure": (ig.EW - sp.EW) / bud, "share_within": (sp.within - ig.within) / bud,
                     "share_between": sp.between / bud, "between_frac": sp.between / sp.EW,
                     "within_frac_spi": sp.within / sp.EW, "within_frac_ig": ig.within / ig.EW,
                     "EL1_spi": sp.EL1, "L1bar_spi": sp.L1bar, "L1_ig": ig.L1bar, "EW_spi": sp.EW, "W_ig": ig.EW})
    return pd.DataFrame(rows) if rows else None


def extrapolate(df, cols, by="image"):
    """Khớp y(M) = a + b/√M trên các M có được, trả a (giới hạn M → ∞, thô)."""
    import pandas as pd
    out = []
    for im, g in df.groupby(by):
        Ms = g.M.values.astype(float)
        if len(Ms) < 2:
            continue
        X = np.stack([np.ones_like(Ms), 1 / np.sqrt(Ms)], 1)
        row = {by: im}
        for c in cols:
            a, _ = np.linalg.lstsq(X, g[c].values.astype(float), rcond=None)[0]
            row[c] = a
        out.append(row)
    return pd.DataFrame(out)


def write(out: str, cfg: dict) -> str:
    import pandas as pd
    warnings.filterwarnings("ignore")
    meta = json.load(open(os.path.join(out, "meta.json"))) if os.path.exists(os.path.join(out, "meta.json")) else {}
    gate, toy = _read(out, "gate"), _read(out, "gate_toy")
    m0, pr, st, sp, a0 = (_read(out, n) for n in ("e4_methods", "e4_pairs", "e4_state", "e4_spi_paths", "e4c_alpha"))
    mt = with_ref(m0) if m0 is not None and len(m0) else None
    at = alpha_table(a0, m0) if a0 is not None and len(a0) else None
    focus = list(cfg["cross"].get("focus_images", []) or [])

    L = ["# CROSS_REPORT — E1′ (gate) và E4 (cross-method exposure)\n"]
    for stg in ("gate", "e4", "e4c"):
        mm = meta.get(stg)
        if mm:
            L.append(f"- **{stg}**: {len(mm['images'])} ảnh, {mm['minutes']:.1f} phút (ước tính {mm['estimate'][stg + '_min']:.1f}), "
                     f"GPU `{mm['gpu']}`" + (" — **DEBUG, không dùng để kết luận**" if mm.get("debug") else ""))
    sig = cfg["cross"]["mode_sigma"]
    L.append(f"""
## Định nghĩa

Exposure W_γ = Σ_i ∫|∂_iF||dx_i| và A^γ như MECHANISM_REPORT, trên cùng điểm cầu phương phân tầng.
**Mode gradient** (không dùng mask), tách chính xác g = cm_low + cm_high + cd_low + cd_high:
cm = trung bình theo kênh (cùng dấu giữa 3 kênh), cd = phần lệch giữa kênh; low = Gauss σ = {sig} px, high = phần còn lại.
W_mode = Σ_i ∫|g_mode,i||dx_i|. Các W_mode không cộng lại thành W (bất đẳng thức tam giác), chỉ so theo tỉ số
r_mode = W_mode / W_mode,ref.
**Tham chiếu**: IG (đoạn thẳng đen → x) cho SPI và Guided IG; đoạn thẳng blur_σmax(x) → x cho BlurIG (cùng hai đầu).
**Độ dài đường** Σ_i∫|dx_i|: bằng |Δx|₁ với đường đơn điệu từng toạ độ (IG, SPI, Guided IG); BlurIG có thể dài hơn.
**Chỉ số chọn lọc** (đặt trước khi xem số liệu):
S_freq = log(r_low / r_high) (> 0: tránh tần số cao nhiều hơn thấp; không cần mask),
S_cm_high = log(r_cm_low / r_cm_high), S_mask = log(r_in / r_out) (> 0: giữ exposure trong vật tương đối hơn; **dùng mask**, chỉ là chẩn đoán).
**Giả thuyết không cần mask (H_mode)**: phương pháp/ảnh có S_freq lớn hơn thì cải thiện chất lượng nhiều hơn.
Bootstrap theo ảnh; các đường của cùng một ảnh không coi là quan sát độc lập.
""")

    # ── gate ──────────────────────────────────────────────────────────────
    L.append("## E1′. Cổng số học bổ sung\n")
    if gate is not None and len(gate):
        bud = gate_budget(gate)
        L.append(f"### Phần between do nhiễu cầu phương ({gate.image.nunique()} ảnh, SPI K = "
                 f"{int(gate[(gate.level == 'image') & (gate.method == 'spi10')].K.max())} đường, M lồng nhau trên cùng đường)\n")
        L.append("Nếu between chỉ do đường khác nhau thì nó không đổi khi tăng M; phần giảm theo M là nhiễu cầu phương riêng "
                 "từng đường bị trung bình hoá. Cột `*_inf` là ngoại suy thô y(M) = a + b/√M.\n")
        if bud is not None:
            L.append(md(bud.groupby("M")[["budget_÷W_IG", "share_exposure", "share_within", "share_between", "between_frac",
                                         "within_frac_spi", "within_frac_ig"]].mean(), index=True))
            ex = extrapolate(bud, ["share_exposure", "share_within", "share_between", "between_frac", "within_frac_spi"])
            if len(ex):
                L.append("Ngoại suy M → ∞ (trung bình qua ảnh gate):\n")
                L.append(md(ex.mean(numeric_only=True).to_frame("M→∞ (thô)").T))
        gp = gate[(gate.level == "path")]
        if len(gp):
            Ms = sorted(gp.M.unique())
            lo_, hi_ = Ms[0], Ms[-1]
            a = gp[gp.M == lo_].set_index(["image", "p"])
            b = gp[gp.M == hi_].set_index(["image", "p"])
            j = a[["iou", "W", "L1"]].join(b[["iou", "W", "L1"]], lsuffix="_lo", rsuffix="_hi")
            rel = within_image_spearman(j.reset_index(), "iou_lo", "iou_hi")
            r1 = within_image_spearman(a.reset_index(), "W", "iou")
            r2 = within_image_spearman(b.reset_index(), "W", "iou")
            L.append(f"**Độ tin cậy metric từng đường.** Spearman trong ảnh giữa IoU(A^γ) ở M = {lo_} và M = {hi_} trên cùng đường: "
                     f"{fmt(rel[0])} [{fmt(rel[1])}, {fmt(rel[2])}]. Spearman(W_γ, IoU) trong ảnh: M = {lo_} → {fmt(r1[0])}, "
                     f"M = {hi_} → {fmt(r2[0])}. |IoU_lo − IoU_hi| median {fmt((j.iou_lo - j.iou_hi).abs().median())}.\n")
    if toy is not None and len(toy):
        L.append("### Toy: gộp bước nhảy DP có làm chệch E[A] không\n")
        L.append("So ghép cặp từng đường ở lưới thô với chính nó ở lưới mịn. `paired_z_rms` = rms qua feature của "
                 "mean(A_N − A_mịn)/SE: ≈ 1 nghĩa là không phát hiện chệch quần thể; ≫ 1 là có chệch. A chính xác "
                 "(Gauss–Legendre 3 điểm); W bằng lưới trung điểm mịn. Bậc hai: kỳ vọng bất biến theo lý thuyết (đối chứng).\n")
        cols = ["family", "N", "paired_bias_rel", "paired_z_rms", "paired_z_max", "perpath_A_dist_fine", "perpath_W_ratio_fine",
                "EW", "within", "between", "W_ig", "L1_ig"]
        L.append(md(toy.sort_values(["family", "N"])[cols]))

    # ── E4a ───────────────────────────────────────────────────────────────
    L.append("## E4a. Cùng thước đo cho bốn phương pháp\n")
    if mt is not None:
        L.append(f"{mt.image.nunique()} ảnh. Median qua ảnh. ΔX = phương pháp − tham chiếu cùng ảnh.\n")
        cols = ["exposure_ratio", "len_ratio", "within_frac", "between_frac", "across_frac", "r_in", "r_out",
                "iou", "insdel_blur", "insdel_black", "energy_mask", "d_iou", "d_insdel_blur", "d_insdel_black"]
        L.append(md(mt.groupby("method")[[c for c in cols if c in mt]].median(), index=True))
        rows = []
        for m in ("spi10", "guided", "blur"):
            g = mt[mt.method == m]
            for c in ("exposure_ratio", "d_iou", "d_insdel_blur", "d_insdel_black", "S_freq", "S_mask"):
                if c in g:
                    e, lo, hi, n = boot_ci(g[c].values)
                    rows.append({"method": m, "metric": c, "mean": e, "lo95": lo, "hi95": hi, "n": n})
        L.append("Trung bình qua ảnh, bootstrap 95%:\n")
        L.append(md(pd.DataFrame(rows)))
        L.append("**BlurIG** dùng hai đầu riêng (blur → x); `exposure_ratio` của nó so với đoạn thẳng cùng hai đầu, và chất lượng "
                 "so với `line_blur`. Completeness của BlurIG theo D_blur = f(x) − f(blur(x)).\n")
        L.append("### IDGI trên cùng đường cố định\n")
        L.append("IDGI = Σ_k g_k²/‖g_k‖²·ΔF_k với g_k, F tại các nút của chính đường đó (không dùng W làm thước đo cho IDGI).\n")
        rows = []
        for m in ("ig", "spi10", "guided", "blur", "line_blur"):
            g = mt[mt.method == m]
            for q in ("iou", "insdel_blur", "insdel_black"):
                c = f"idgi_minus_canon_{q}"
                if c in g:
                    e, lo, hi, n = boot_ci(g[c].values)
                    rows.append({"method": m, "metric": f"IDGI − canonical: {q}", "mean": e, "lo95": lo, "hi95": hi, "n": n})
        L.append(md(pd.DataFrame(rows)))

    # ── E4b ───────────────────────────────────────────────────────────────
    L.append("## E4b. Các phương pháp có tránh cùng chỗ không?\n")
    if mt is not None:
        L.append("### Theo mode gradient (không cần mask)\n")
        L.append("r_mode < 1: phương pháp tiếp xúc ít hơn tham chiếu với mode đó. Nếu các phương pháp tránh cùng mode, "
                 "hồ sơ r_mode của chúng giống nhau.\n")
        L.append(md(mt[mt.method.isin(["spi10", "guided", "blur"])].groupby("method")[
            [f"r_{x}" for x in MODES] + ["r_low", "r_high", "S_freq", "S_cm_high"]].median(), index=True))
        comp = mt[mt.method.isin(["ig", "line_blur"])].copy()
        tot = sum(comp[f"W_{x}"] for x in MODES)
        for x in MODES:
            comp[f"share_{x}"] = comp[f"W_{x}"] / tot
        L.append("Thành phần mode của exposure ở tham chiếu (W_mode / Σ W_mode):\n")
        L.append(md(comp.groupby("method")[[f"share_{x}" for x in MODES]].median(), index=True))
    if pr is not None and len(pr):
        L.append("### Theo feature (pixel): bản đồ exposure bị tránh ΔW = W_ref − W\n")
        rows = []
        for (a, b), g in pr.groupby(["a", "b"]):
            for c in ["spearman_avoided", "cos_avoided", "spearman_logratio"] + [f"spearman_avoided_{x}" for x in MODES]:
                e, lo, hi, n = boot_ci(g[c].values)
                rows.append({"cặp": f"{a} vs {b}", "metric": c, "mean": e, "lo95": lo, "hi95": hi, "n": n})
        L.append(md(pd.DataFrame(rows)))
    if st is not None and len(st):
        L.append("### Theo trạng thái\n")
        L.append("Phần exposure theo tiến độ τ (tỉ lệ L1 đã đi; median qua ảnh, SPI trung bình qua đường) và "
                 "khoảng cách tới dây cung nối hai đầu, trọng số exposure (`dist_mean_w`).\n")
        t = st[st.axis == "tau"].groupby(["method", "image", "bin"]).share.mean().reset_index()
        L.append(md(t.pivot_table(index="method", columns="bin", values="share", aggfunc="median"), index=True))
        d = st[st.axis == "dist_mean_w"].groupby(["method", "image"]).share.mean().reset_index()
        L.append(md(d.groupby("method").share.median().to_frame("dist_mean_w"), index=True))

    # ── chọn lọc và chất lượng ─────────────────────────────────────────────
    L.append("## Chọn lọc exposure có dự đoán chất lượng không?\n")
    if mt is not None:
        sub = mt[mt.method.isin(["spi10", "guided", "blur"])]
        rows = []
        for x in ("S_freq", "S_cm_high", "exposure_ratio", "S_mask"):
            for y in ("d_iou", "d_insdel_blur", "d_insdel_black"):
                if y in sub:
                    e, lo, hi, n = cluster_spearman(sub, x, y)
                    rows.append({"x": x, "y": y, "spearman (ảnh × phương pháp)": e, "lo95": lo, "hi95": hi, "n": n,
                                 "cần mask": x == "S_mask"})
        L.append("Gộp ảnh × {SPI, Guided, BlurIG}; bootstrap theo ảnh.\n")
        L.append(md(pd.DataFrame(rows)))
        if focus:
            f = mt[mt.image.isin(focus)]
            if len(f):
                L.append(f"**Ảnh đối chứng** {', '.join(focus)}:\n")
                L.append(md(f[["image", "method", "exposure_ratio", "r_low", "r_high", "S_freq", "S_mask", "r_in", "r_out",
                               "d_iou", "d_insdel_blur"]]))
    if at is not None and "S_freq" in at:
        L.append("### E4c. Quét mức lệch nhịp (SPI, α thay đổi)\n")
        L.append("`async` = khoảng cách trung bình (trọng số exposure) tới đường chéo / ‖Δx‖, đo trên đường thật. Tỉ số so với IG cùng ảnh (E4).\n")
        ap = a0[a0.level == "path"]
        asy = ap.groupby(["image", "alpha"])["async"].mean().reset_index().groupby("alpha")["async"].median()
        cols = [c for c in ["h", "exposure_ratio", "r_in", "r_out", "r_low", "r_high", "S_freq", "S_mask", "within_frac",
                            "between_frac", "iou", "insdel_blur", "insdel_black", "d_iou", "d_insdel_blur"] if c in at]
        t = at.groupby("alpha")[cols].median()
        t["async"] = asy
        if "iou" in ap:
            t["iou_K1_med"] = ap.groupby("alpha").iou.median()
        if "insdel_blur" in ap:
            t["insdel_blur_K1_med"] = ap.groupby("alpha").insdel_blur.median()
        L.append(md(t.sort_values("h"), index=True))
        rows = []
        for x in ("h", "S_freq", "S_mask", "exposure_ratio"):
            for y in ("iou", "insdel_blur", "insdel_black"):
                if y in at:
                    e, lo, hi, n = within_image_spearman(at, x, y)
                    rows.append({"x": x, "y": y, "mean Spearman theo α trong ảnh": e, "lo95": lo, "hi95": hi, "n_ảnh": n})
        L.append("Trong từng ảnh, giữa các mức α (rồi trung bình qua ảnh):\n")
        L.append(md(pd.DataFrame(rows)))
        if focus:
            f = at[at.image.isin(focus)]
            if len(f):
                L.append(f"Ảnh đối chứng theo α:\n")
                L.append(md(f[["image", "alpha", "exposure_ratio", "S_freq", "S_mask", "r_in", "r_out", "iou", "insdel_blur"]
                              ].sort_values(["image", "alpha"])))

    # ── tóm tắt ───────────────────────────────────────────────────────────
    L.append("## Tóm tắt số đo (mô tả, không ngưỡng)\n")
    if mt is not None:
        for m in ("spi10", "guided", "blur"):
            g = mt[mt.method == m]
            if not len(g):
                continue
            e = boot_ci(g.exposure_ratio.values)
            s = boot_ci(g.S_freq.values)
            L.append(f"- **{m}**: exposure/ref {fmt(e[0])} [{fmt(e[1])}, {fmt(e[2])}]; S_freq {fmt(s[0])} [{fmt(s[1])}, {fmt(s[2])}]; "
                     f"ΔIoU median {fmt(g.d_iou.median())}.")
    if pr is not None and len(pr):
        for (a, b), g in pr.groupby(["a", "b"]):
            e = boot_ci(g.spearman_avoided.values)
            L.append(f"- Chồng vùng tránh {a}–{b}: Spearman {fmt(e[0])} [{fmt(e[1])}, {fmt(e[2])}].")
    L.append("")
    try:
        files = plots(out, _plt(), gate, toy, mt, st, at, a0)
    except Exception as ex:
        print(f"[cross plots] lỗi: {type(ex).__name__}: {ex}")
        files = []
    L.append("## Tệp\n")
    L.append("`gate.csv`, `gate_toy.csv`, `e4_methods.csv`, `e4_pairs.csv`, `e4_state.csv`, `e4_spi_paths.csv`, "
             "`e4c_alpha.csv`, `meta.json`.\n")
    for f in files:
        L.append(f"- `{os.path.basename(f)}`")
    path = os.path.join(out, "CROSS_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[cross report] {path}")
    return path
