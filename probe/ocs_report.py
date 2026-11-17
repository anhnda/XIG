"""OCS_REPORT.md — Output Convergence Study (đọc CSV trong <out>/ocs_<split>)."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .report import _read, boot_ci, fmt, md

MAIN = ["ig", "spi", "guided", "blur", "line_blur"]
REFS = ["grad", "gxi", "edge", "center"]
MUTED = "#6b6a63"
CORR = ("spearman_abs", "spearman_signed_pix", "spearman_signed_elem", "cos_elem", "cos_pix_signed", "kendall_abs_56",
        "spearman_abs_56", "spearman_abs_14")


def _plt():
    from .plots import _setup
    return _setup()


def pair_vals(pr, kind, a, b, col):
    m = pr[(pr.kind == kind) & (((pr.a == a) & (pr.b == b)) | ((pr.a == b) & (pr.b == a)))]
    return m.set_index("image")[col] if col in m else None


def boot_ratio(num, den_a, den_b, B=2000, seed=0, corr=True):
    """Tỉ số cross / trần, trần = √(w_a·w_b) cho thước đo tương quan, min(w_a, w_b) cho thước đo tập; bootstrap theo ảnh."""
    d = np.column_stack([num.values, den_a.reindex(num.index).values, den_b.reindex(num.index).values])
    d = d[np.isfinite(d).all(1)]
    if len(d) < 3:
        return (float("nan"),) * 3

    def f(x):
        c = np.sqrt(max(x[:, 1].mean(), 1e-12) * max(x[:, 2].mean(), 1e-12)) if corr else min(x[:, 1].mean(), x[:, 2].mean())
        return x[:, 0].mean() / c
    rng = np.random.default_rng(seed)
    bs = [f(d[rng.integers(0, len(d), len(d))]) for _ in range(B)]
    return float(f(d)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def write(out: str, cfg: dict) -> str:
    import pandas as pd
    warnings.filterwarnings("ignore")
    c = cfg["ocs"]
    meta = json.load(open(os.path.join(out, "meta.json"))) if os.path.exists(os.path.join(out, "meta.json")) else {}
    pr, wi, kc, cs, im = (_read(out, f"ocs_{n}") for n in ("pairs", "within", "kcross", "consensus", "images"))
    L = ["# OCS_REPORT — các phương pháp có hội tụ về cùng một lời giải thích không?\n"]
    if meta:
        L.append(f"- split **{meta['split']}**, {len(meta['images'])} ảnh, {meta['minutes']:.1f} phút (ước tính {meta['estimate']['ocs_min']:.1f}), "
                 f"GPU `{meta['gpu']}`" + (" — **DEBUG**" if meta.get("debug") else "")
                 + (" — **exploratory** (ảnh đã dùng để hình thành giả thuyết)" if meta.get("split") == "pilot" else ""))
    dc = c["decision"]
    L.append(f"""
## Thiết kế

Phương pháp: `ig` (đen → x), `spi` (DP α, đen → x), `guided` (đen → x), `blur` (BlurIG, blur → x), `line_blur` (IG từ cùng điểm đầu
mờ → x: **đối chứng baseline** — khác biệt blur ↔ line_blur mới là hiệu ứng đường của BlurIG). Tham chiếu đơn giản: `grad` |∇F(x)|,
`gxi` ∇F(x)⊙Δx, `edge` (Sobel), `center` (Gauss tâm). Null: bản đồ thứ hai bị xáo khối {c['shuffle_block']}×{c['shuffle_block']}.
Trong-phương-pháp: hai ước lượng độc lập (đường tất định: hai bộ điểm cầu phương M = {c['M']}; SPI: hai tập {c['spi_paths_per_half']} đường rời nhau).
So chéo trên nửa 1; **trần** = √(w_a·w_b) (thước đo tương quan) hoặc min(w_a, w_b) (Jaccard, RBO, chồng lấn); tỉ số = cross/trần.

**Luật quyết định (đặt trước khi xem số liệu, trong config `ocs.decision`)** cho cặp chính {dc['key_pair']}:
- hội tụ **xếp hạng** nếu tỉ số Spearman |A| ≥ {dc['rank_ratio_min']};
- hội tụ **credit có dấu** nếu tỉ số cosine (phần tử) ≥ {dc['signed_ratio_min']} **và** trùng dấu trên top-10% ≥ {dc['sign_agree_min']};
- **B** = cả hai; **A** = chỉ xếp hạng; **C** = không đạt cả hai nhưng Jaccard top-10% vượt null rõ (CI > 0) **và** vùng consensus
  có hiệu ứng xoá lớn hơn mọi đối chứng khớp hạng (CI > 0, baseline {dc['consensus_base']}, {dc['consensus_metric']});
- còn lại: **không hội tụ**. Điểm dừng: nếu không B và consensus không hơn đối chứng khớp → dừng hướng shared attribution.
""")
    if pr is None or not len(pr):
        L.append("_(không có dữ liệu)_")
        path = os.path.join(out, "OCS_REPORT.md")
        open(path, "w").write("\n".join(L))
        return path

    # ── 1. ma trận theo tầng ───────────────────────────────────────────────
    L.append("## 1. Tương đồng giữa các phương pháp (bản đồ đầy đủ, trung bình qua ảnh)\n")
    full = pr[pr.kind == "cross_full"]
    tiers = {"ranking": ["spearman_abs", "spearman_signed_pix", "spearman_signed_elem", "kendall_abs_56", "spearman_abs_56",
                         "spearman_abs_14"],
             "top": [x for x in full.columns if x.startswith("jaccard_")] + ["rbo"],
             "signed": ["cos_elem", "cos_pix_signed", "l1_norm_dist", "sign_agree_top10", "sign_agree_w"],
             "spatial": [x for x in full.columns if x.startswith("overlap_r")] + ["sliced_w1"]}
    for tname, cols in tiers.items():
        cols = [x for x in cols if x in full]
        t = full.groupby(["a", "b"])[cols].mean()
        L.append(f"**{tname}**:\n")
        L.append(md(t, index=True))
    L.append("`l1_norm_dist` ∈ [0, 2] (0 = cùng phân bổ credit có dấu sau chuẩn hoá); `sliced_w1` tính bằng pixel (nhỏ = gần).\n")

    # ── 2. cross vs within ─────────────────────────────────────────────────
    L.append("## 2. Hội tụ vượt nhiễu lấy mẫu? (cross so với within)\n")
    if wi is not None and len(wi):
        w0 = wi[wi.kind.isin(["path_sets", "quadrature_sets"])]
        cols = [x for x in ("spearman_abs", "spearman_signed_pix", "cos_elem", "jaccard_10", "rbo", "sign_agree_top10",
                            "overlap_r4") if x in w0]
        L.append("**Within** (hai ước lượng độc lập của cùng phương pháp):\n")
        L.append(md(w0.groupby("method")[cols].mean(), index=True))
        rows = []
        for i, a in enumerate(MAIN):
            for b in MAIN[i + 1:]:
                for col in cols:
                    num = pair_vals(pr, "cross_half1", a, b, col)
                    if num is None or not len(num):
                        continue
                    wa = w0[w0.method == a].set_index("image")[col]
                    wb = w0[w0.method == b].set_index("image")[col]
                    r = boot_ratio(num, wa, wb, corr=col in CORR or col.startswith(("spearman", "cos", "kendall")))
                    nul = pair_vals(pr, "null_block", a, b, col)
                    fv = pair_vals(pr, "cross_full", a, b, col)
                    exc = boot_ci((fv - nul).dropna().values) if nul is not None else (np.nan,) * 4
                    rows.append({"cặp": f"{a}–{b}", "metric": col, "cross": float(num.mean()), "within_a": float(wa.mean()),
                                 "within_b": float(wb.mean()), "tỉ số": r[0], "lo95": r[1], "hi95": r[2],
                                 "vượt null (full − xáo khối)": exc[0], "lo95 ": exc[1], "hi95 ": exc[2]})
        L.append("**Cross / trần** (bootstrap theo ảnh) và phần vượt null:\n")
        L.append(md(pd.DataFrame(rows)))
        ratio_tab = pd.DataFrame(rows)
    else:
        ratio_tab = pd.DataFrame()

    # ── 3. đối chứng ───────────────────────────────────────────────────────
    L.append("## 3. Đối chứng: tham chiếu đơn giản và baseline của BlurIG\n")
    ref = pr[pr.kind == "ref"]
    if len(ref):
        cols = [x for x in ("spearman_abs", "jaccard_10", "overlap_r4", "sliced_w1") if x in ref]
        L.append("Tương đồng của từng phương pháp với bản đồ tham chiếu đơn giản (nếu cao ngang tương đồng giữa các phương pháp, "
                 "phần \"hội tụ\" có thể chỉ là cùng bám cạnh/vật/tâm ảnh):\n")
        L.append(md(ref.groupby(["a", "b"])[cols].mean(), index=True))
    rows = []
    for a, b in (("blur", "line_blur"), ("blur", "ig"), ("line_blur", "ig"), ("spi", "guided"), ("spi", "ig"), ("guided", "ig")):
        for col in ("spearman_abs", "cos_elem", "jaccard_10"):
            v = pair_vals(pr, "cross_full", a, b, col)
            if v is not None and len(v):
                e = boot_ci(v.values)
                rows.append({"cặp": f"{a}–{b}", "metric": col, "mean": e[0], "lo95": e[1], "hi95": e[2]})
    L.append("**Baseline BlurIG**: blur–line_blur = cùng hai đầu, khác đường; blur–ig gộp cả khác đường và khác điểm đầu.\n")
    L.append(md(pd.DataFrame(rows)))

    # ── 4. theo K ──────────────────────────────────────────────────────────
    if kc is not None and len(kc) and wi is not None:
        L.append("## 4. SPI theo K (kết luận có đổi vì Monte Carlo không?)\n")
        wk = wi[wi.kind == "path_sets_K"].groupby("K")[["spearman_abs", "cos_elem", "jaccard_10"]].mean()
        wk.columns = [f"within_{x}" for x in wk.columns]
        ck = kc.pivot_table(index="K", columns="ref", values="spearman_abs", aggfunc="mean")
        ck.columns = [f"spearman_abs vs {x}" for x in ck.columns]
        cc2 = kc.pivot_table(index="K", columns="ref", values="cos_elem", aggfunc="mean")
        cc2.columns = [f"cos_elem vs {x}" for x in cc2.columns]
        t = wk.join(ck).join(cc2)
        wg = wi[(wi.kind == "quadrature_sets") & (wi.method == "guided")][["spearman_abs", "cos_elem"]].mean()
        t["tỉ số spearman (guided)"] = t["spearman_abs vs guided"] / np.sqrt(t["within_spearman_abs"] * wg.spearman_abs)
        t["tỉ số cos (guided)"] = t["cos_elem vs guided"] / np.sqrt(t["within_cos_elem"].clip(lower=1e-12) * wg.cos_elem)
        L.append(md(t, index=True))

    # ── 5. consensus ───────────────────────────────────────────────────────
    L.append("## 5. Vùng consensus có mang tín hiệu faithful hơn không?\n")
    if im is not None and len(im):
        L.append(f"Top {int(100 * c['consensus']['top_frac'])}% của {', '.join(c['consensus']['methods'])}. |C| median "
                 f"{fmt(im.n_consensus.median())} pixel ({fmt(im.consensus_frac_of_top.median())} của top mỗi phương pháp); "
                 f"phần C nằm trong mask median {fmt(im.consensus_in_mask.median())}. Ảnh có |C| < {c['consensus']['min_size']} bị bỏ.\n")
    cons_ok = None
    if cs is not None and len(cs):
        g = cs.groupby(["image", "base", "set", "method"])[["del_logit", "del_prob", "ins_logit", "ins_prob", "compactness",
                                                            "in_mask"] + [x for x in cs.columns if x.startswith("absA_")]].mean().reset_index()
        L.append("**Đặc điểm các tập** (cùng kích thước |C|; median qua ảnh). `absA_m` = phần khối lượng |A| của phương pháp m nằm "
                 "trong tập — tập khớp hạng phải có absA của m gần với C:\n")
        L.append(md(g.groupby(["base", "set", "method"])[[x for x in g.columns if x.startswith(("absA_", "compactness", "in_mask"))]].median(),
                    index=True))
        rows = []
        cvals = g[g.set == "C"].set_index(["image", "base"])
        for (st, m), gg in g[g.set != "C"].groupby(["set", "method"]):
            gg = gg.set_index(["image", "base"])
            for bn in sorted(set(g.base)):
                for met in ("del_logit", "ins_logit", "del_prob", "ins_prob"):
                    try:
                        d = (cvals.xs(bn, level="base")[met] - gg.xs(bn, level="base")[met]).dropna()
                    except KeyError:
                        continue
                    e = boot_ci(d.values)
                    rows.append({"so với": f"{st}" + (f"[{m}]" if m != "-" else ""), "base": bn, "metric": met, "C − tập": e[0],
                                 "lo95": e[1], "hi95": e[2], "n_ảnh": e[3]})
        ct = pd.DataFrame(rows)
        L.append("**Hiệu ứng của C trừ hiệu ứng của tập đối chứng cùng kích thước** (dương = C quan trọng hơn với mô hình; ghép cặp theo ảnh):\n")
        L.append(md(ct))
        sel = ct[(ct.base == dc["consensus_base"]) & (ct.metric == dc["consensus_metric"]) & ct["so với"].str.startswith("matched")]
        cons_ok = bool(len(sel)) and bool((sel.lo95 > 0).all())

    # ── 6. quyết định ──────────────────────────────────────────────────────
    L.append("## 6. Quyết định theo luật đặt trước\n")
    a, b = dc["key_pair"]
    key = f"{a}–{b}" if f"{a}–{b}" in set(ratio_tab.get("cặp", [])) else f"{b}–{a}"
    rt = ratio_tab[ratio_tab["cặp"] == key].set_index("metric") if len(ratio_tab) else pd.DataFrame()

    def g_(col, field="tỉ số"):
        return float(rt.loc[col, field]) if len(rt) and col in rt.index else float("nan")
    rank_r, sign_r = g_("spearman_abs"), g_("cos_elem")
    sagree = pair_vals(pr, "cross_full", a, b, "sign_agree_top10")
    sagree = float(sagree.mean()) if sagree is not None and len(sagree) else float("nan")
    jx = g_("jaccard_10", "lo95 ")
    rank_ok = np.isfinite(rank_r) and rank_r >= dc["rank_ratio_min"]
    sign_ok = np.isfinite(sign_r) and sign_r >= dc["signed_ratio_min"] and sagree >= dc["sign_agree_min"]
    if rank_ok and sign_ok:
        case = "B — cả xếp hạng và credit có dấu đều hội tụ"
    elif rank_ok:
        case = "A — cùng xếp hạng, khác credit có dấu"
    elif np.isfinite(jx) and jx > 0 and cons_ok:
        case = "C — chỉ một phần trùng nhau, và phần chung mang tín hiệu faithful hơn đối chứng khớp"
    else:
        case = "Không hội tụ (theo luật)"
    L.append(f"- Cặp {key}: tỉ số Spearman |A| = **{fmt(rank_r)}** (ngưỡng {dc['rank_ratio_min']}); tỉ số cosine phần tử = **{fmt(sign_r)}** "
             f"(ngưỡng {dc['signed_ratio_min']}); trùng dấu top-10% = **{fmt(sagree)}** (ngưỡng {dc['sign_agree_min']}); "
             f"Jaccard top-10% vượt null: CI dưới = {fmt(jx)}; consensus hơn mọi đối chứng khớp: **{cons_ok}**.")
    L.append(f"- **Kết luận theo luật: {case}.**")
    if not case.startswith("B") and not cons_ok:
        L.append("- **Điểm dừng đạt**: không có hội tụ đầy đủ và consensus không hơn đối chứng khớp → dừng hướng shared attribution.")
    if meta.get("split") == "pilot":
        L.append("- Đây là split pilot (exploratory). Xác nhận: đặt `ocs.split: fresh` (ảnh chưa dùng ở bất kỳ lượt nào), lần chạy đầu ghi "
                 "`ocs_prereg.json`, không sửa mục `ocs` sau đó.")

    # ── đồ thị ─────────────────────────────────────────────────────────────
    files = []
    try:
        plt = _plt()
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for a_, col in zip(ax, ["spearman_abs", "cos_elem", "jaccard_10"]):
            mat = np.full((len(MAIN), len(MAIN)), np.nan)
            for i, p in enumerate(MAIN):
                for j, q in enumerate(MAIN):
                    if i == j:
                        if wi is not None:
                            w = wi[(wi.method == p) & wi.kind.isin(["path_sets", "quadrature_sets"])]
                            mat[i, j] = float(w[col].mean()) if col in w and len(w) else np.nan
                    else:
                        v = pair_vals(pr, "cross_full", p, q, col)
                        mat[i, j] = float(v.mean()) if v is not None and len(v) else np.nan
            a_.imshow(mat, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(MAIN)):
                for j in range(len(MAIN)):
                    if np.isfinite(mat[i, j]):
                        a_.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if mat[i, j] > 0.6 else "#1a1a19")
            a_.set_xticks(range(len(MAIN)), MAIN, rotation=40, ha="right", fontsize=7)
            a_.set_yticks(range(len(MAIN)), MAIN, fontsize=7)
            a_.set_title(f"{col} (đường chéo = within)", fontsize=9); a_.grid(False)
        fig.tight_layout(); f = os.path.join(out, "ocs_matrix.png"); fig.savefig(f); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[ocs plots] lỗi: {type(ex).__name__}: {ex}")
    L.append("\n## Tệp\n")
    L.append("`ocs_pairs.csv` (kind = cross_half1 / cross_full / null_block / ref), `ocs_within.csv`, `ocs_kcross.csv`, `ocs_consensus.csv`, "
             "`ocs_images.csv`, `meta.json`" + (", `ocs_prereg.json`" if meta.get("split") != "pilot" else "") + ".\n")
    for f in files:
        L.append(f"- `{os.path.basename(f)}`")
    path = os.path.join(out, "OCS_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[ocs report] {path}")
    return path
