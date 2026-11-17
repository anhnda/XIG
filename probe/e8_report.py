"""E8_REPORT.md — kết quả xác nhận theo đúng các phép so đã tiền đăng ký (đọc CSV trong <out>/e8_<split>)."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .report import _read, fmt, md

MUTED = "#6b6a63"


def _plt():
    from .plots import _setup
    return _setup()


def image_means(p, exp):
    d = p[p.exp == exp].copy()
    igW = d[d.family == "ig"].groupby("image").W.mean()
    d["exposure_ratio"] = d.W / d.image.map(igW)
    d["within_frac"] = d.within / d.W
    cols = [c for c in ("exposure_ratio", "within_frac", "dist_geom", "iou", "insdel_blur", "insdel_black", "energy_mask",
                        "signed_ratio_to_noise", "signed_l1_vs_ig", "A_rel_diff_main") if c in d]
    g = d.groupby(["image", "family"])
    m = g[cols].mean()
    m["n_real"] = g.size()
    return m.reset_index()


def paired(m, a, b, col):
    x = m[m.family == a].set_index("image")[col]
    y = m[m.family == b].set_index("image")[col]
    return (x - y).dropna()


def boot(d, B, seed):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    return np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def analyse(m, hyps, st):
    import pandas as pd
    B, a_, seed = int(st["bootstrap"]), float(st["alpha"]), int(st["seed"])
    rows = []
    for h in hyps:
        for met in h["metrics"]:
            if met not in m or h["a"] not in set(m.family) or h["b"] not in set(m.family):
                rows.append({"ID": h["id"], "so sánh": f"{h['a']} − {h['b']}", "metric": met, "test": h["test"],
                             "verdict": "THIẾU DỮ LIỆU"})
                continue
            d = paired(m, h["a"], h["b"], met)
            bs = boot(d.values, B, seed)
            lo95, hi95 = np.quantile(bs, [a_ / 2, 1 - a_ / 2])
            lo90, hi90 = np.quantile(bs, [a_, 1 - a_])
            p_le, p_ge = max((bs <= 0).mean(), 1 / B), max((bs >= 0).mean(), 1 / B)
            test = h["test"]
            p = {"two_sided": min(1.0, 2 * min(p_le, p_ge)), "less": p_ge, "greater": p_le}.get(test, float("nan"))
            rows.append({"ID": h["id"], "so sánh": f"{h['a']} − {h['b']}", "metric": met, "test": test,
                         "family": h.get("family", ""), "margin": h.get("margin", float("nan")), "n": len(d),
                         "mean": float(d.mean()), "median": float(d.median()), "lo95": lo95, "hi95": hi95,
                         "lo90": lo90, "hi90": hi90, "p_boot": p, "số ảnh âm": int((d < 0).sum()), "số ảnh dương": int((d > 0).sum())})
    t = pd.DataFrame(rows)
    prim = t[(t.get("family", "") == "primary") & t.p_boot.notna()] if "p_boot" in t else t.iloc[:0]
    t["p_holm"] = np.nan
    if len(prim):
        t.loc[prim.index, "p_holm"] = holm(prim.p_boot.values)
    verd = []
    for _, r in t.iterrows():
        if r.get("verdict") == "THIẾU DỮ LIỆU":
            verd.append(r["verdict"]); continue
        if r.test == "equivalence":
            mg = float(r.margin)
            if -mg < r.lo90 and r.hi90 < mg:
                verd.append("TƯƠNG ĐƯƠNG (trong ±biên)")
            elif (r.lo95 > 0 or r.hi95 < 0) and abs(r["mean"]) > mg:
                verd.append("KHÁC (vượt biên)")
            else:
                verd.append("CHƯA KẾT LUẬN")
            continue
        p = r.p_holm if np.isfinite(r.p_holm) else r.p_boot
        sig = p < a_
        if r.test == "two_sided":
            verd.append("XÁC NHẬN (≠ 0)" if sig else "KHÔNG XÁC NHẬN")
        elif r.test == "less":
            verd.append("XÁC NHẬN (< 0)" if sig else ("NGƯỢC DỰ ĐOÁN" if r.lo95 > 0 else "KHÔNG XÁC NHẬN"))
        elif r.test == "greater":
            verd.append("XÁC NHẬN (> 0)" if sig else ("NGƯỢC DỰ ĐOÁN" if r.hi95 < 0 else "KHÔNG XÁC NHẬN"))
        else:
            verd.append("?")
    t["verdict"] = verd
    return t


def write(out: str, cfg: dict) -> str:
    import pandas as pd
    warnings.filterwarnings("ignore")
    e = cfg["e8"]
    meta = json.load(open(os.path.join(out, "meta.json"))) if os.path.exists(os.path.join(out, "meta.json")) else {}
    pre = json.load(open(os.path.join(out, "e8_prereg.json"))) if os.path.exists(os.path.join(out, "e8_prereg.json")) else {}
    p = _read(out, "e8_paths")
    L = ["# E8_REPORT — xác nhận held-out các phép so đã tiền đăng ký\n"]
    if meta:
        L.append(f"- split **{meta['split']}**, {len(meta['images'])} ảnh, {meta['minutes']:.1f} phút (ước tính "
                 f"{meta['estimate']['e8_min']:.1f}), GPU `{meta['gpu']}`" + (" — **DEBUG, không dùng để kết luận**" if meta.get("debug") else ""))
    if pre:
        same = pre.get("digest") == meta.get("digest")
        L.append(f"- tiền đăng ký lúc {pre['time']}, hash `{pre['digest'][:16]}` — " +
                 ("**trùng** với lần chạy này" if same else "**KHÔNG TRÙNG** — kết quả không hợp lệ cho xác nhận"))
        L.append("- thiết kế đầy đủ: `E8_PREREG.md`, `e8_prereg.json`")
    st = e["stats"]
    L.append(f"""
## Quy tắc (đã khoá)

Hiệu ghép cặp theo ảnh d_i = (trung bình K = {e['K']} đường của a) − (của b). Bootstrap theo ảnh {st['bootstrap']} lần (seed {st['seed']}).
p_boot hai phía = 2·min(P*(d̄* ≤ 0), P*(d̄* ≥ 0)); một phía theo hướng dự đoán. Holm trên họ `primary`, α = {st['alpha']}.
**XÁC NHẬN** nếu p (Holm) < α theo đúng hướng; **NGƯỢC DỰ ĐOÁN** nếu CI 95% nằm hẳn phía ngược. Tương đương: **TƯƠNG ĐƯƠNG** nếu
CI {int(100 * (1 - 2 * st['alpha']))}% nằm trọn trong ±biên (TOST); **KHÁC** nếu CI 95% loại 0 và |trung bình| > biên; còn lại **CHƯA KẾT LUẬN**.
""")
    if p is None or not len(p):
        L.append("_(không có dữ liệu)_")
    else:
        m = image_means(p, "main")
        t = analyse(m, e["hypotheses"], st)
        L.append("## Kết quả các phép so đã đăng ký (báo tất cả)\n")
        cols = ["ID", "so sánh", "metric", "test", "family", "margin", "n", "mean", "lo95", "hi95", "lo90", "hi90", "p_boot", "p_holm",
                "số ảnh âm", "số ảnh dương", "verdict"]
        L.append(md(t[[c for c in cols if c in t]]))
        g = m[m.family == "global"]
        if len(g) and "signed_ratio_to_noise" in g:
            med = float(g.signed_ratio_to_noise.median())
            ok = med <= float(e["global_signed_ratio_max"])
            L.append(f"**global ↔ IG, attribution có dấu**: median_ảnh(‖A_global − A_IG,a‖₁ / ‖A_IG,b − A_IG,a‖₁) = **{fmt(med)}** "
                     f"(ngưỡng ≤ {e['global_signed_ratio_max']}) → **{'NHẤT QUÁN với IG' if ok else 'KHÔNG NHẤT QUÁN'}**; "
                     f"‖A_global − A_IG‖₁/‖A_IG‖₁ median {fmt(g.signed_l1_vs_ig.median())}.\n")

        L.append("## Mô tả theo kiểu nhóm (median qua ảnh của trung bình K = 1)\n")
        order = ["ig"] + list(e["kinds"])
        dcols = ["n_real", "exposure_ratio", "dist_geom", "within_frac", "iou", "insdel_blur", "insdel_black", "energy_mask"]
        dt = m.groupby("family")[[c for c in dcols if c in m]].median()
        L.append(md(dt.reindex([o for o in order if o in dt.index]), index=True))

        if "gate" in set(p.exp):
            mg = image_means(p, "gate")
            ims = set(mg.image)
            gk = e["gate"]["kinds"]
            L.append(f"## Cổng số học {gk[0]} − {gk[1]} ({len(ims)} ảnh, cùng realization; N = {e['N']}/M = {e['M']} so với "
                     f"N = {e['N_fine']}/M = {e['gate']['M']} với điểm cầu phương độc lập)\n")
            rows = []
            for met in ("exposure_ratio", "iou", "insdel_blur", "insdel_black"):
                for lab, mm in (("main", m[m.image.isin(ims)]), ("gate", mg)):
                    d = paired(mm, gk[0], gk[1], met)
                    bs = boot(d.values, int(st["bootstrap"]), int(st["seed"]))
                    lo, hi = np.quantile(bs, [0.025, 0.975])
                    rows.append({"metric": met, "mức": lab, "mean": float(d.mean()), "lo95": lo, "hi95": hi, "n": len(d)})
            L.append(md(pd.DataFrame(rows)))
            if "A_rel_diff_main" in mg:
                L.append("‖A(N_fine) − A(N)‖₁/‖A(N)‖₁ trên cùng realization (median): " + ", ".join(
                    f"{k} {fmt(v)}" for k, v in mg[mg.family.isin(gk)].groupby("family").A_rel_diff_main.median().items()) + "\n")

        if "sec" in set(p.exp):
            ms = image_means(p, "sec")
            L.append("## Phụ (không thuộc phép so xác nhận): Guided IG\n")
            rows = []
            for a, b in (("guided", "guided_xfer"), ("guided", "ig"), ("guided_xfer", "ig")):
                for met in ("iou", "insdel_blur", "insdel_black", "exposure_ratio"):
                    d = paired(ms, a, b, met)
                    if len(d):
                        bs = boot(d.values, int(st["bootstrap"]), int(st["seed"]))
                        lo, hi = np.quantile(bs, [0.025, 0.975])
                        rows.append({"so sánh": f"{a} − {b}", "metric": met, "mean": float(d.mean()), "lo95": lo, "hi95": hi,
                                     "n": len(d)})
            L.append(md(pd.DataFrame(rows)))

        files = []
        try:
            plt = _plt()
            tt = t[t.verdict != "THIẾU DỮ LIỆU"].reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(8, 0.42 * len(tt) + 1.2))
            y = np.arange(len(tt))[::-1]
            for yi, (_, r) in zip(y, tt.iterrows()):
                col = "#1baf7a" if r.verdict.startswith(("XÁC NHẬN", "TƯƠNG")) else ("#eb6834" if r.verdict.startswith(("NGƯỢC", "KHÁC"))
                                                                                     else MUTED)
                if r.test == "equivalence":
                    ax.add_patch(plt.Rectangle((-r.margin, yi - 0.35), 2 * r.margin, 0.7, color="#e4e3dc", zorder=0))
                ax.plot([r.lo95, r.hi95], [yi, yi], color=col, lw=2)
                ax.scatter([r["mean"]], [yi], color=col, s=30, zorder=3)
            ax.axvline(0, color=MUTED, lw=0.8)
            ax.set_yticks(y, [f"{r.ID} {r['so sánh']} · {r.metric}" for _, r in tt.iterrows()], fontsize=7)
            ax.set_xlabel("hiệu ghép cặp theo ảnh (95% CI; nền xám = biên tương đương)")
            fig.tight_layout(); f = os.path.join(out, "e8_forest.png"); fig.savefig(f); plt.close(fig); files.append(f)
        except Exception as ex:
            print(f"[e8 plots] lỗi: {type(ex).__name__}: {ex}")
        L.append("\n## Tệp\n")
        L.append("`e8_paths.csv` (từng realization; exp = main / gate / sec), `e8_images.csv`, `E8_PREREG.md`, `e8_prereg.json`, `meta.json`.\n")
        for f in files:
            L.append(f"- `{os.path.basename(f)}`")
    path = os.path.join(out, "E8_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[e8 report] {path}")
    return path
