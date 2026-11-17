"""E5_REPORT.md + đồ thị (đọc CSV trong <out>/e5)."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .report import _read, boot_ci, fmt, md

FCOL = {"ig": "#2a78d6", "spi": "#eb6834", "spi_first": "#eb6834", "spi_perm_bin": "#e87ba4", "guided": "#1baf7a",
        "guided_perm_bin": "#008300", "guided_perm_pixel": "#7cc47c", "guided_perm_block": "#b5dfb5",
        "spi_lam": "#eda100", "blur": "#9a998f", "line_blur": "#c8c7bf"}
MUTED = "#6b6a63"
QUAL = ("iou", "insdel_blur", "insdel_black", "energy_mask")


def _plt():
    from .plots import _setup
    return _setup()


def fam_table(f):
    f = f.copy()
    ref = f.set_index(["image", "family"])
    f["exposure_ratio"] = [r.EW / ref.loc[(r.image, r.ref), "EW"] for r in f.itertuples()]
    f["within_frac"] = f.within / f.EW
    f["between_frac"] = f["between"] / f.EW
    for q in QUAL:
        if q in f:
            f[f"d_{q}"] = [getattr(r, q) - ref.loc[(r.image, r.ref), q] for r in f.itertuples()]
    return f


def paired(f, a, b, col):
    x = f[f.family == a].set_index("image")[col]
    y = f[f.family == b].set_index("image")[col]
    d = (x - y).dropna()
    return boot_ci(d.values)


def plots(out, plt, f, pr):
    files = []

    def save(fig, name):
        p = os.path.join(out, name)
        fig.tight_layout(); fig.savefig(p); plt.close(fig); files.append(p)

    if f is not None and len(f):
        fams = [x for x in ["spi", "guided", "guided_perm_bin", "guided_perm_pixel", "guided_perm_block", "spi_lam",
                            "spi_first", "spi_perm_bin", "blur"] if x in set(f.family)]
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        for a, col in zip(ax, ["exposure_ratio", "d_iou", "d_insdel_blur"]):
            vals = [boot_ci(f[f.family == x][col].values) for x in fams]
            a.bar(range(len(fams)), [v[0] for v in vals], 0.7, color=[FCOL.get(x, MUTED) for x in fams],
                  edgecolor="white", linewidth=1)
            a.errorbar(range(len(fams)), [v[0] for v in vals],
                       yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt="none", ecolor=MUTED, lw=1)
            a.axhline(1 if col == "exposure_ratio" else 0, color=MUTED, lw=0.8)
            a.set_xticks(range(len(fams)), fams, rotation=45, ha="right", fontsize=7)
            a.set_title(f"{col} (trung bình, 95% CI; so với ref)", fontsize=9)
        save(fig, "e5_families.png")
    if pr is not None and len(pr):
        cr = pr[pr.kind == "cross"]
        if len(cr):
            pairs = list(dict.fromkeys(zip(cr.a, cr.b)))
            cols = ["spearman_avoided_raw", "spearman_resid", "partial_spearman_resid", "spearman_logratio_top50"]
            colc = ["#9a998f", "#2a78d6", "#eb6834", "#1baf7a"]
            fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(pairs) + 2), 4))
            w = 0.8 / len(cols)
            for i, (c, cc) in enumerate(zip(cols, colc)):
                v = [cr[(cr.a == a) & (cr.b == b)][c].mean() for a, b in pairs]
                ax.bar(np.arange(len(pairs)) + (i - 1.5) * w, v, w * 0.92, color=cc, label=c)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xticks(range(len(pairs)), [f"{a}\nvs {b}" for a, b in pairs], fontsize=7)
            ax.legend(fontsize=7, ncol=2); ax.set_ylabel("trung bình qua ảnh")
            save(fig, "e5_pairs.png")
    p = os.path.join(out, "e5_heatmaps.pt")
    if os.path.exists(p):
        import torch
        from .data import MEAN, STD
        for im, h in torch.load(p, weights_only=False).items():
            ks = [k for k in h if k.startswith("R_")]
            fig, ax = plt.subplots(1, len(ks) + 1, figsize=(3 * (len(ks) + 1), 3))
            img = (h["x"][0].float() * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)).clamp(0, 1)
            ax[0].imshow(img.permute(1, 2, 0).numpy())
            ax[0].contour(h["mask"].numpy().astype(float), levels=[0.5], colors="white", linewidths=0.8)
            for a, k in zip(ax[1:], ks):
                v = h[k].numpy()
                lim = float(np.quantile(np.abs(v), 0.995)) + 1e-30
                a.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim)
                a.set_title(f"{k[2:]}: R = cW_ref − W (đỏ: tránh hơn mức đều)", fontsize=7)
            for a in ax:
                a.set_xticks([]); a.set_yticks([]); a.grid(False)
            fig.suptitle(im, fontsize=9)
            save(fig, f"e5_resid_{im}.png")
    return files


def write(out: str, cfg: dict) -> str:
    import pandas as pd
    warnings.filterwarnings("ignore")
    meta = json.load(open(os.path.join(out, "meta.json"))) if os.path.exists(os.path.join(out, "meta.json")) else {}
    f0, pr, lam, sg, ck = (_read(out, f"e5_{n}") for n in ("families", "pairs", "lam", "signal", "checks"))
    f = fam_table(f0) if f0 is not None and len(f0) else None
    L = ["# E5_REPORT — bất đồng bộ hay chọn feature theo gradient?\n"]
    if meta:
        L.append(f"- {len(meta['images'])} ảnh, {meta['minutes']:.1f} phút (ước tính {meta['estimate']['e5_min']:.1f}), "
                 f"GPU `{meta['gpu']}`" + (" — **DEBUG, không dùng để kết luận**" if meta.get("debug") else ""))
    c = cfg["e5"]
    L.append(f"""
## Thiết kế

Mỗi đường: hai bộ điểm cầu phương phân tầng độc lập (a, b), mỗi bộ M = {c['M']} điểm/đoạn; A, W = trung bình hai bộ.
Bản đồ ở mức pixel (cộng 3 kênh). Tham chiếu exposure: IG (đen → x) cho mọi họ, trừ BlurIG (đoạn thẳng blur → x).

**P0 — residual sau khi loại co giãn toàn cục**: R^m = c_m·W^ref − W^m, c_m = ΣW^m/ΣW^ref. Khi so m với n, R^m dùng W^ref
bộ a và R^n dùng W^ref bộ b, nên nhiễu của tham chiếu không tạo tương quan giả. Báo:
`spearman_avoided_raw` (ΔW = W^ref − W như E4, có confound), `spearman_resid`, `pearson_resid`,
`partial_spearman_resid` (khống chế hạng W^ref), `spearman_logratio_top50` (log(W^m/W^ref) trên pixel có W^ref trên trung vị).
**Trần** = split-half của chính họ đó (đường chẵn/bộ a vs đường lẻ/bộ b; họ một đường: bộ a vs bộ b).

**P1 — hoán vị lịch trình Guided IG** (lịch trình Z_k = (X_k − x′)/Δx lấy từ đường Guided đã sinh; giữ thời điểm nút):
`bin` trong (kênh × thập phân vị |Δx_i|), `pixel` hoán vị vị trí pixel (3 kênh đi cùng) trong thập phân vị Σ_c|Δx|,
`block` hoán vị khối {c['block']}×{c['block']} (giữ cấu trúc không gian trong khối). {c['perm_draws']} lần rút mỗi kiểu.
**Đối chứng âm**: cùng phép `bin` lên chính {c['spi_perm_paths']} đường SPI đầu (`spi_first` → `spi_perm_bin`); phân phối
đường SPI không đổi dưới hoán vị, nên chênh lệch phải nằm trong nhiễu.

**P2 — đường λ** Z^λ = (1−λ)t + λZ^SPI, N = {c['guided_steps']}, λ = dist_Guided/dist_SPI (khớp độ lệch hình học trung bình
tới dây cung, không nhìn gradient hay chất lượng). {c['lam_paths']} đường mỗi ảnh.

**P3 — tín hiệu cộng tính**: trên mọi đường không thích nghi, canonical giữ đúng 100% tín hiệu tuyến tính
(A(S) = aR⊙Δx, chính xác với cầu phương phân tầng) — đúng cho mọi họ ở đây trừ Guided IG gốc. Guided IG được chạy lại trên F+S.
""")
    if ck is not None and len(ck):
        L.append(f"Kiểm: Guided end_gap max {fmt(ck.guided_end_gap.max())}; sai số dựng lại đường Guided từ lịch trình max "
                 f"{fmt(ck.guided_sched_recon_err.max())}.\n")
    if lam is not None and len(lam):
        L.append(f"Khớp hình học P2: dist_Guided median {fmt(lam.dist_guided.median())}, dist_λ median {fmt(lam.dist_lam.median())}, "
                 f"λ median {fmt(lam['lambda'].median())} [q10 {fmt(lam['lambda'].quantile(0.1))}, q90 {fmt(lam['lambda'].quantile(0.9))}].\n")

    # ── P0 ────────────────────────────────────────────────────────────────
    L.append("## P0. Tương đồng vùng tránh sau khi loại co giãn toàn cục\n")
    rel = {}
    if pr is not None and len(pr):
        sh = pr[pr.kind == "split_half"]
        rows = []
        for fam, g in sh.groupby("a"):
            e = boot_ci(g.spearman_resid.values)
            full = 2 * e[0] / (1 + e[0]) if np.isfinite(e[0]) and e[0] > -1 else float("nan")
            rel[fam] = full
            rows.append({"họ": fam, "split_half_spearman_resid": e[0], "lo95": e[1], "hi95": e[2],
                         "ρ_full (Spearman–Brown)": full, "split_half_logratio": g.spearman_logratio_top50.mean(), "n": e[3]})
        L.append("**Trần độ tin cậy** (cùng họ, hai nửa độc lập):\n")
        L.append(md(pd.DataFrame(rows)))
        cr = pr[pr.kind == "cross"]
        rows = []
        for (a, b), g in cr.groupby(["a", "b"], sort=False):
            r = {"cặp": f"{a} vs {b}"}
            for col in ("spearman_avoided_raw", "spearman_resid", "partial_spearman_resid", "spearman_logratio_top50",
                        "pearson_resid"):
                e = boot_ci(g[col].values)
                r[col] = e[0]
                if col in ("spearman_resid", "partial_spearman_resid"):
                    r[f"{col}_CI"] = f"[{fmt(e[1])}, {fmt(e[2])}]"
            ceil = np.sqrt(rel.get(a, np.nan) * rel.get(b, np.nan))
            r["trần √(ρa·ρb)"] = ceil
            r["resid / trần"] = r["spearman_resid"] / ceil if ceil and np.isfinite(ceil) and ceil > 0 else float("nan")
            r["n"] = len(g)
            rows.append(r)
        L.append("**Giữa các họ** (trung bình qua ảnh, bootstrap 95%; `resid / trần` ≈ tương quan đã hiệu chỉnh độ tin cậy):\n")
        L.append(md(pd.DataFrame(rows)))
        L.append("`resid_vs_ref_*` (tương quan của R với W^ref, trung bình): " + ", ".join(
            f"{a}: {fmt(g.resid_vs_ref_m.mean())}" for a, g in cr.groupby("a")) + ".\n")

    # ── P1, P2 ────────────────────────────────────────────────────────────
    L.append("## P1–P2. Chất lượng và exposure theo họ đường\n")
    if f is not None:
        cols = ["K", "exposure_ratio", "dist_geom", "within_frac", "between_frac", "iou", "insdel_blur", "insdel_black",
                "energy_mask", "d_iou", "d_insdel_blur", "d_insdel_black"]
        L.append(md(f.groupby("family")[[x for x in cols if x in f]].median(), index=True))
        comps = [("guided", "ig"), ("spi", "ig"), ("spi", "guided")]
        comps += [(f"guided_perm_{k}", "guided") for k in c["perm_kinds"]]
        comps += [(f"guided_perm_{k}", "ig") for k in c["perm_kinds"]]
        comps += [("spi_lam", "guided"), ("spi_lam", "ig"), ("spi_perm_bin", "spi_first")]
        rows = []
        for a, b in comps:
            if a not in set(f.family) or b not in set(f.family):
                continue
            for col in ("exposure_ratio", "iou", "insdel_blur", "insdel_black", "energy_mask"):
                e = paired(f, a, b, col)
                rows.append({"so sánh": f"{a} − {b}", "metric": col, "mean": e[0], "lo95": e[1], "hi95": e[2], "n": e[3]})
        L.append("**Hiệu ghép cặp theo ảnh** (bootstrap 95%). `spi_perm_bin − spi_first` là đối chứng âm: CI phải chứa 0.\n")
        L.append(md(pd.DataFrame(rows)))
        rows = []
        for k in c["perm_kinds"]:
            name = f"guided_perm_{k}"
            if name not in set(f.family):
                continue
            for col in ("iou", "insdel_blur", "insdel_black"):
                g_ = f[f.family == "guided"][col].mean() - f[f.family == "ig"][col].mean()
                p_ = f[f.family == name][col].mean() - f[f.family == "ig"][col].mean()
                rows.append({"họ": name, "metric": col, "lợi ích Guided so với IG": g_, "lợi ích còn lại": p_,
                             "tỉ lệ giữ lại": p_ / g_ if abs(g_) > 1e-12 else float("nan")})
        L.append("**Phần lợi ích của Guided IG còn lại sau hoán vị** (tỉ số của trung bình; không ổn định khi lợi ích gốc nhỏ):\n")
        L.append(md(pd.DataFrame(rows)))

    # ── P3 ────────────────────────────────────────────────────────────────
    L.append("## P3. Tín hiệu cộng tính\n")
    if sg is not None and len(sg):
        s = sg[sg.skipped == False] if "skipped" in sg else sg  # noqa: E712
        if len(s):
            L.append("`retention_guided_adaptive` = Σ_R[A(F+S;γ′) − A(F;γ)]/ΣA*, γ′ = Guided chạy trên F+S. `path_part_R` = phần "
                     "do đường đổi (A(F;γ′) − A(F;γ) trên R). Mọi đường không thích nghi: retention = 1 chính xác.\n")
            L.append(md(s.groupby(["side", "loc"])[["retention_guided_adaptive", "path_part_R", "path_part_norm_rel",
                                                    "path_shift_rel"]].median(), index=True))

    # ── tiêu chí ──────────────────────────────────────────────────────────
    L.append("## Đối chiếu với tiêu chí (do ChatGPT đặt trước)\n")
    if pr is not None and len(pr) and f is not None:
        cr = pr[pr.kind == "cross"]
        g = cr[(cr.a == "spi") & (cr.b == "guided")]
        if len(g):
            e = boot_ci(g.spearman_resid.values)
            ep = boot_ci(g.partial_spearman_resid.values)
            L.append(f"- **Hội tụ (selective avoidance chung)**: SPI–Guided residual Spearman {fmt(e[0])} [{fmt(e[1])}, {fmt(e[2])}], "
                     f"partial {fmt(ep[0])} [{fmt(ep[1])}, {fmt(ep[2])}]; raw (E4-style) {fmt(g.spearman_avoided_raw.mean())}.")
        for k in c["perm_kinds"]:
            name = f"guided_perm_{k}"
            if name in set(f.family):
                e = paired(f, name, "guided", "iou")
                e2 = paired(f, name, "guided", "insdel_blur")
                L.append(f"- **Bất đồng bộ là cơ chế chính?** {name} − guided: ΔIoU {fmt(e[0])} [{fmt(e[1])}, {fmt(e[2])}], "
                         f"Δinsdel_blur {fmt(e2[0])} [{fmt(e2[1])}, {fmt(e2[2])}].")
        if "spi_lam" in set(f.family):
            e = paired(f, "spi_lam", "guided", "iou")
            L.append(f"- **Hình học khớp (P2)**: spi_lam − guided ΔIoU {fmt(e[0])} [{fmt(e[1])}, {fmt(e[2])}].")
        if "spi_perm_bin" in set(f.family):
            e = paired(f, "spi_perm_bin", "spi_first", "iou")
            L.append(f"- **Đối chứng âm**: spi_perm_bin − spi_first ΔIoU {fmt(e[0])} [{fmt(e[1])}, {fmt(e[2])}] (phải chứa 0).")
        L.append("- Luật đọc: \"hội tụ\" nếu residual SPI–Guided còn dương rõ (CI > 0) và đáng kể so với trần; \"bất đồng bộ là "
                 "chính\" nếu các họ hoán vị / λ giữ phần lớn lợi ích của Guided; \"hai cơ chế khác nhau\" nếu residual mất hoặc "
                 "hoán vị làm mất lợi ích dù độ lệch hình học được giữ.")
    L.append("")
    try:
        files = plots(out, _plt(), f, pr)
    except Exception as ex:
        print(f"[e5 plots] lỗi: {type(ex).__name__}: {ex}")
        files = []
    L.append("## Tệp\n")
    L.append("`e5_families.csv`, `e5_pairs.csv`, `e5_lam.csv`, `e5_signal.csv`, `e5_checks.csv`, `e5_images.csv`, `meta.json`.\n")
    for p in files:
        L.append(f"- `{os.path.basename(p)}`")
    path = os.path.join(out, "E5_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[e5 report] {path}")
    return path
