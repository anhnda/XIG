"""
Cửa 1 — lợi thế population của SPI so với IG (protocol 6.5, Cổng 1).

Biến chính: d_j = IoU(SPI_α tham chiếu) − IoU(IG tham chiếu), α ∈ gate1.spi.
Tham chiếu = mức cao nhất đã đủ (b = 'mean'); "mức trước" = mức ngay dưới, cho điều kiện (iii).
Điều kiện dương (ít nhất một α thỏa cả ba):
    (i)   trung bình d ≥ min_mean_d
    (ii)  cận dưới BCa (mức conf, Bonferroni) > 0
    (iii) d giữ dấu và |d_top − d_prev| < ½|d_top| (trung bình qua ảnh)
Wilcoxon một phía (Holm cho các α) là kiểm định phụ. insdelp_blur, ROAD báo cáo cùng cách, không quyết định.
Quyết định: POSITIVE → 1B;  không dương và power (từ SD_d thật) ≥ ngưỡng ở mọi α → NEGATIVE;  còn lại UNDERPOWERED.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .config import canon
from .reference import n_sub_of
from .setting import Setting
from .stats import Table, achieved_power, bca_ci, describe, holm, n_required, pct_ci, wilcoxon_greater


def _pair(tab: Table, a_spec, a_lvl, b_spec, b_lvl, metric, images):
    a = tab.values(a_spec, a_lvl, metric, images)
    b = tab.values(b_spec, b_lvl, metric, images)
    d = a - b
    ok = ~np.isnan(d)
    return d[ok], a[ok], b[ok]


def run_gate1(st: Setting, group: str = "stage1a") -> dict:
    cfg = st.cfg
    gc = cfg["gate1"]
    tab = Table(st.store.rows("metrics"))
    images = st.image_ids()
    n_img = len(images)
    ig = canon(gc["ig_ref"])
    spis = [canon(s) for s in gc["spi"]]
    conf, n_boot, seed = float(gc["conf"]), int(gc["n_boot"]), int(gc["boot_seed"])
    alpha2 = 1.0 - conf

    conv_path = st.store.p(f"convergence_{group}.json")
    conv = json.load(open(conv_path)) if os.path.exists(conv_path) else {}

    L_ig = tab.levels(ig, images)
    if len(L_ig) < 2:
        raise SystemExit(f"{ig}: cần ≥ 2 mức đã đánh giá cho mọi ảnh (có {L_ig}). Chạy attr/converge/eval trước.")
    ig_top, ig_prev = L_ig[-1], L_ig[-2]

    out = {"n_images": n_img, "ig": ig, "ig_levels": [ig_top, ig_prev], "alphas": {}}
    lines = [f"# Cửa 1 — {st.tag} — {n_img} ảnh\n",
             f"Tham chiếu IG: `{ig}` mức {ig_top} (mức trước {ig_prev}). Khoảng tin cậy BCa {conf:.1%}, "
             f"{n_boot} lần bootstrap. Ngưỡng (i): trung bình d ≥ {gc['min_mean_d']}.\n"]

    # cờ hội tụ
    flags = []
    for spec in [ig] + spis:
        c = conv.get(spec, {})
        if c.get("status") != "converged":
            flags.append(f"`{spec}`: {c.get('status', 'chưa kiểm hội tụ')}")
    if flags:
        lines.append("**Cảnh báo hội tụ:** " + "; ".join(flags) + "\n")

    # biến chính
    lines += ["## Biến chính: IoU\n",
              "| α | mức SPI top/prev | IoU SPI | IoU IG | mean d | BCa CI | d_prev | (i) | (ii) | (iii) | "
              "Wilcoxon p (Holm) | SD_d | power | n cần |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows, pvals = [], []
    for spec in spis:
        L = tab.levels(spec, images)
        if len(L) < 2:
            raise SystemExit(f"{spec}: cần ≥ 2 mức đã đánh giá cho mọi ảnh (có {L})")
        top, prev = L[-1], L[-2]
        d, a, b = _pair(tab, spec, top, ig, ig_top, "iou", images)
        dp, _, _ = _pair(tab, spec, prev, ig, ig_prev, "iou", images)
        m, mp = float(d.mean()), float(dp.mean())
        lo, hi = bca_ci(d, conf, n_boot, seed)
        sd = float(d.std(ddof=1))
        c1 = m >= float(gc["min_mean_d"])
        c2 = lo > 0
        c3 = (np.sign(m) == np.sign(mp)) and abs(m - mp) < 0.5 * abs(m)
        pw = achieved_power(sd, float(gc["delta"]), len(d), alpha2)
        n_need = n_required(sd, float(gc["delta"]), alpha2, float(gc["power"]))
        p = wilcoxon_greater(d)
        pvals.append(p)
        se_spi = np.nanmedian(tab.sub_se(spec, top, "iou", images, n_sub_of(cfg, spec)))
        se_ig = np.nanmedian(tab.sub_se(ig, ig_top, "iou", images, n_sub_of(cfg, ig)))
        rows.append(dict(spec=spec, top=top, prev=prev, iou_spi=float(a.mean()), iou_ig=float(b.mean()),
                         mean_d=m, ci=[lo, hi], mean_d_prev=mp, c1=bool(c1), c2=bool(c2), c3=bool(c3),
                         sd_d=sd, power=pw, n_required=n_need, wilcoxon_p=p, n=len(d),
                         median_sub_se_iou_spi=float(se_spi), median_sub_se_iou_ig=float(se_ig)))
    for r, ph in zip(rows, holm(pvals)):
        r["wilcoxon_p_holm"] = ph
        mark = lambda v: "✓" if v else "✗"
        lines.append(f"| {r['spec']} | {r['top']}/{r['prev']} | {r['iou_spi']:.4f} | {r['iou_ig']:.4f} | "
                     f"{r['mean_d']:+.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['mean_d_prev']:+.4f} | "
                     f"{mark(r['c1'])} | {mark(r['c2'])} | {mark(r['c3'])} | {r['wilcoxon_p']:.3g} ({ph:.3g}) | "
                     f"{r['sd_d']:.4f} | {r['power']:.2f} | {r['n_required']} |")
        out["alphas"][r["spec"]] = r
    lines.append("\nSai số chuẩn IoU của tham chiếu (trung vị qua ảnh, từ sub-estimate): " +
                 ", ".join(f"{r['spec']}: {r['median_sub_se_iou_spi']:.4f}" for r in rows) +
                 f"; IG: {rows[0]['median_sub_se_iou_ig']:.4f}\n")

    positive = any(r["c1"] and r["c2"] and r["c3"] for r in rows)
    powered = all(r["power"] >= float(gc["power"]) for r in rows)
    if positive:
        decision = "POSITIVE"
        msg = "Dương theo 6.5 → mở Giai đoạn 1B (data collapse)."
    elif powered:
        decision = "NEGATIVE"
        msg = ("Âm với power đủ → dừng nhánh 'SPI có lợi thế population' trên thiết lập này. "
               "Không lập kế hoạch paper kết quả âm.")
    else:
        need = max(r["n_required"] for r in rows)
        decision = "UNDERPOWERED"
        msg = (f"Thiếu power (power đạt < {gc['power']} ở ít nhất một α). Tăng n lên {need} "
               "(data.imagenet.n_images), khoá lại, chạy lại. KHÔNG đọc là 'SPI không có lợi thế'.")
    if flags and decision == "POSITIVE":
        msg += " (Có tham chiếu chưa hội tụ: xem cảnh báo ở trên trước khi tin kết quả.)"
    out["decision"], out["message"] = decision, msg
    lines.insert(1, f"## Quyết định: **{decision}**\n\n{msg}\n")

    # metric phụ
    for metric in cfg["gate1"]["secondary"]:
        lines += [f"\n## Metric phụ (không quyết định): {metric}\n",
                  "| α | mean SPI | mean IG | mean d | BCa CI | Wilcoxon p |", "|---|---|---|---|---|---|"]
        for r in rows:
            d, a, b = _pair(tab, r["spec"], r["top"], ig, ig_top, metric, images)
            if not len(d):
                lines.append(f"| {r['spec']} | – | – | – | – | – |")
                continue
            lo, hi = bca_ci(d, conf, n_boot, seed)
            p = wilcoxon_greater(d)
            out["alphas"][r["spec"]][f"{metric}_mean_d"] = float(d.mean())
            out["alphas"][r["spec"]][f"{metric}_ci"] = [lo, hi]
            lines.append(f"| {r['spec']} | {a.mean():.4f} | {b.mean():.4f} | {d.mean():+.4f} | "
                         f"[{lo:+.4f}, {hi:+.4f}] | {p:.3g} |")

    # cùng ngân sách
    ref = canon(gc["budget_ref"])
    lines += [f"\n## Cùng ngân sách (báo cáo riêng, không dùng kết luận population)\n",
              f"Δ so với `{ref}`, CI percentile 95% theo ảnh.\n",
              "| spec | backward | IoU | Δ IoU [CI] | insdelp_blur | Δ [CI] | ROAD | Δ [CI] | trung vị abs residual |",
              "|---|---|---|---|---|---|---|---|---|"]
    out["budget"] = {}
    for spec in [canon(s) for s in gc["budget"]]:
        d0 = tab.get(spec, 0)
        if not all(im in d0 for im in images):
            lines.append(f"| {spec} | – | chưa đủ | | | | | | |")
            continue
        bwd = int(np.median([d0[im]["n_backward"] for im in images]))
        cells, rec = [], {"n_backward": bwd}
        for metric in ("iou", "insdelp_blur", "road"):
            v = tab.values(spec, 0, metric, images)
            if np.all(np.isnan(v)):
                cells += ["–", "–"]
                continue
            if spec == ref:
                cells += [describe(v), "ref"]
            else:
                dd, _, _ = _pair(tab, spec, 0, ref, 0, metric, images)
                lo, hi = pct_ci(dd, 0.95, n_boot, seed)
                cells += [describe(v), f"{dd.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"]
                rec[f"{metric}_delta"] = float(dd.mean())
            rec[metric] = float(np.nanmean(v))
        res = np.nanmedian(np.abs(tab.values(spec, 0, "residual", images)))
        rec["median_abs_residual"] = float(res)
        out["budget"][spec] = rec
        lines.append(f"| {spec} | {bwd} | " + " | ".join(cells) + f" | {res:.3f} |")

    text = "\n".join(lines) + "\n"
    path = st.store.write_report("gate1", text, out)
    print(text)
    print(f"[gate1] -> {path}")
    return out
