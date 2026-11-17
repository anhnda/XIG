"""E7_REPORT.md + đồ thị (đọc CSV trong <out>/e7)."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .report import _read, boot_ci, fmt, md

MUTED = "#6b6a63"
METRICS = ("exposure_ratio", "iou", "insdel_blur", "insdel_black", "energy_mask")
KEY = [("block4", "indep"), ("rand16", "indep"), ("block4", "rand16"), ("block16", "rand256"), ("block16", "indep"),
       ("rand256", "indep"), ("block4", "rgb"), ("rand16", "rgb"), ("rgb", "indep"), ("global", "ig")]


def _plt():
    from .plots import _setup
    return _setup()


def image_means(p, exp):
    d = p[p.exp == exp].copy()
    igW = d[d.family == "ig"].set_index("image").W
    d["exposure_ratio"] = d.W / d.image.map(igW)
    d["within_frac"] = d.within / d.W
    d["in_frac_W"] = d.W_in / d.W
    cols = ["exposure_ratio", "within_frac", "in_frac_W", "dist_geom", "iou", "insdel_blur", "insdel_black", "energy_mask"]
    cols = [c for c in cols if c in d] + [c for c in ("groups", "group_size") if c in d]
    g = d.groupby(["image", "family"])
    m = g[cols].mean()
    m["n_real"] = g.size()
    return m.reset_index()


def diffs(m, a, b, col):
    x = m[m.family == a].set_index("image")[col]
    y = m[m.family == b].set_index("image")[col]
    return (x - y).dropna()


def cmp_table(m, pairs):
    import pandas as pd
    rows = []
    fams = set(m.family)
    for a, b in pairs:
        if a not in fams or b not in fams:
            continue
        for col in METRICS:
            if col not in m:
                continue
            d = diffs(m, a, b, col)
            e = boot_ci(d.values)
            rows.append({"so sánh": f"{a} − {b}", "metric": col, "mean": e[0], "lo95": e[1], "hi95": e[2],
                         "median": float(d.median()) if len(d) else float("nan"),
                         "số ảnh âm": int((d < 0).sum()), "n": len(d)})
    return pd.DataFrame(rows)


def write(out: str, cfg: dict) -> str:
    import pandas as pd
    warnings.filterwarnings("ignore")
    meta = json.load(open(os.path.join(out, "meta.json"))) if os.path.exists(os.path.join(out, "meta.json")) else {}
    p, en, ms = _read(out, "e7_paths"), _read(out, "e7_ens"), _read(out, "e7_mapsim")
    c = cfg["e7"]
    L = ["# E7_REPORT — nhóm đồng bộ cùng cỡ: bố trí không gian hay số/cỡ nhóm?\n"]
    if meta:
        L.append(f"- {len(meta['images'])} ảnh pilot (**exploratory**), {meta['minutes']:.1f} phút (ước tính "
                 f"{meta['estimate']['e7_min']:.1f}), GPU `{meta['gpu']}`" + (" — **DEBUG**" if meta.get("debug") else ""))
    s, nj, q = c["S"], c["Njump"], c["Q"]
    L.append(f"""
## Thiết kế

DP α = {cfg['paths']['spi10']['alpha']}, mọi phần tử cùng biên Beta(αt, α(1−t)); chỉ đổi việc **chia sẻ lịch trình**. Mỗi nhóm gồm
cả 3 kênh của các pixel được chọn. `blockB`: B×B pixel liền kề; `randG`: G pixel chọn ngẫu nhiên khắp ảnh (phân hoạch mới mỗi đường).
**block4 ↔ rand16** và **block16 ↔ rand256** cùng số nhóm, cùng cỡ nhóm, chỉ khác bố trí không gian. `indep` = mỗi phần tử
(pixel × kênh) một lịch trình; `rgb` = mỗi pixel một lịch trình (mốc cùng quy ước nhóm RGB); `global` = đường thẳng IG tham số hoá
lại (đối chứng, kỳ vọng ≈ IG).
Phép so chính ở mức K = 1 (metric từng đường, trung bình trong ảnh, hiệu ghép cặp theo ảnh, bootstrap theo ảnh).
S: N = {s['N']}, M = {s['M']}, {s['paths']} đường/kiểu. N: cùng so sánh ở N = {nj['N']} (M = {nj['M']}) trên {nj['n_images']} ảnh — ít gộp
bước nhảy hơn; quy tắc nội suy (gấp khúc qua nút lưới) giống nhau cho mọi kiểu nhóm. Q: Guided ở M = {q['M_hi']} và M = {q['M_lo']}
lồng nhau trên cùng đường, {q['n_images']} ảnh.

**Đọc kết quả (ChatGPT đặt trước):** rand16 ≈ indep còn block4 kém → tương tác giữa pixel gần nhau (spatial interaction topology);
rand16 kém tương tự block4 → số/cỡ cụm đồng bộ quan trọng hơn bố trí không gian.
""")
    if p is None or not len(p):
        L.append("_(không có dữ liệu)_")
    else:
        m = image_means(p, "S")
        order = ["ig"] + list(s["kinds"])
        L.append("## S. Cấu trúc nhóm (K = 1)\n")
        cols = ["groups", "group_size", "n_real", "exposure_ratio", "dist_geom", "within_frac", "in_frac_W", "iou",
                "insdel_blur", "insdel_black", "energy_mask"]
        t = m.groupby("family")[[x for x in cols if x in m]].median()
        L.append("Median qua ảnh của trung bình K = 1 trong ảnh:\n")
        L.append(md(t.reindex([o for o in order if o in t.index]), index=True))
        L.append("**Hiệu ghép cặp theo ảnh** (`số ảnh âm` = số ảnh có hiệu < 0):\n")
        L.append(md(cmp_table(m, KEY)))
        L.append("### Hiệu theo từng ảnh (có bị vài ảnh chi phối không?)\n")
        for a, b in [("block4", "indep"), ("rand16", "indep"), ("block4", "rand16"), ("block16", "rand256")]:
            if a not in set(m.family) or b not in set(m.family):
                continue
            per = pd.DataFrame({col: diffs(m, a, b, col) for col in ("iou", "insdel_blur", "insdel_black", "exposure_ratio")})
            loo = {}
            for col in ("iou", "insdel_blur", "insdel_black"):
                v = per[col].dropna()
                if len(v) > 2:
                    lo = [float(v.drop(i).mean()) for i in v.index]
                    loo[col] = f"[{fmt(min(lo))}, {fmt(max(lo))}]"
            L.append(f"**{a} − {b}** — khoảng trung bình khi bỏ từng ảnh (leave-one-out): "
                     + ", ".join(f"{k} {v}" for k, v in loo.items()) + "\n")
            L.append(md(per.reset_index().rename(columns={"index": "image"})))
        if ms is not None and len(ms):
            t = ms[(ms.exp == "S") & (ms.ref == "indep")].groupby("family").spearman_absmap.median()
            L.append("**Bản đồ ensemble giống `indep` đến đâu** (Spearman của Σ_c|Ā|, median qua ảnh): "
                     + ", ".join(f"{k} {fmt(v)}" for k, v in t.items()) + "\n")

        # ── N ──────────────────────────────────────────────────────────────
        if "N" in set(p.exp):
            mN = image_means(p, "N")
            ims = set(mN.image)
            L.append(f"## N. Độ bền theo độ phân giải lưới (N = {nj['N']} so với N = {s['N']}, cùng {len(ims)} ảnh)\n")
            pairs = [(a, b) for a, b in KEY if a in set(mN.family) and b in set(mN.family)]
            tN = cmp_table(mN, pairs)
            tS = cmp_table(m[m.image.isin(ims)], pairs)
            if len(tN) and len(tS):
                j = tN.merge(tS, on=["so sánh", "metric"], suffixes=(f"_N{nj['N']}", f"_N{s['N']}"))
                L.append(md(j[["so sánh", "metric", f"mean_N{nj['N']}", f"lo95_N{nj['N']}", f"hi95_N{nj['N']}",
                               f"mean_N{s['N']}", f"lo95_N{s['N']}", f"hi95_N{s['N']}"]]))

        # ── Q ──────────────────────────────────────────────────────────────
        qs = [e for e in (f"Q{q['M_lo']}", f"Q{q['M_hi']}") if e in set(p.exp)]
        if qs:
            L.append(f"## Q. Guided IG với cầu phương dày (M = {q['M_lo']} lồng trong M = {q['M_hi']}, cùng đường)\n")
            tabs = []
            for e in qs:
                mq = image_means(p, e)
                t = cmp_table(mq, [("guided", "guided_perm"), ("guided", "guided_xfer"), ("guided", "ig"),
                                   ("guided_perm", "ig"), ("guided_xfer", "ig")])
                t.insert(0, "M", e[1:])
                tabs.append(t)
            L.append(md(pd.concat(tabs)))
            L.append("Nếu hiệu `guided − guided_perm` đổi đáng kể giữa hai mức M thì kết luận E6-A về guidance nhạy với estimator.\n")

        # ── ensemble phụ ───────────────────────────────────────────────────
        if en is not None and len(en):
            e2 = en.copy()
            for k in ("within", "between", "across"):
                e2[f"{k}_frac"] = e2[k] / e2.EW
            L.append("## Phụ: ensemble\n")
            L.append(md(e2.groupby(["exp", "family"])[["K", "within_frac", "between_frac", "across_frac", "iou", "insdel_blur",
                                                      "insdel_black"]].median(), index=True))

        # ── đồ thị ─────────────────────────────────────────────────────────
        files = []
        try:
            plt = _plt()
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
            for axx, col in zip(ax, ["iou", "insdel_blur", "insdel_black"]):
                pairs = [(a, b) for a, b in KEY[:6] if a in set(m.family) and b in set(m.family)]
                vals = [boot_ci(diffs(m, a, b, col).values) for a, b in pairs]
                colors = ["#2a78d6" if a.startswith("block") else "#eb6834" for a, _ in pairs]
                axx.bar(range(len(pairs)), [v[0] for v in vals], 0.7, color=colors, edgecolor="white", linewidth=1)
                axx.errorbar(range(len(pairs)), [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                             fmt="none", ecolor=MUTED, lw=1)
                for k, (a, b) in enumerate(pairs):
                    dd = diffs(m, a, b, col)
                    axx.scatter(np.full(len(dd), k) + np.random.default_rng(0).uniform(-0.2, 0.2, len(dd)), dd.values, s=8,
                                color=MUTED, zorder=3)
                axx.axhline(0, color=MUTED, lw=0.8)
                axx.set_xticks(range(len(pairs)), [f"{a}\n− {b}" for a, b in pairs], fontsize=7)
                axx.set_title(f"Δ{col} (K=1; cột = trung bình, chấm = từng ảnh)", fontsize=9)
            fig.tight_layout(); f = os.path.join(out, "e7_structure.png"); fig.savefig(f); plt.close(fig); files.append(f)
        except Exception as ex:
            print(f"[e7 plots] lỗi: {type(ex).__name__}: {ex}")
        L.append("\n## Tệp\n")
        L.append("`e7_paths.csv` (từng realization; cột exp = S / N / Q<M>), `e7_ens.csv`, `e7_mapsim.csv`, `e7_images.csv`, `meta.json`.\n")
        for f in files:
            L.append(f"- `{os.path.basename(f)}`")
    path = os.path.join(out, "E7_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[e7 report] {path}")
    return path
