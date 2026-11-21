"""
P4 — thống kê, đồ thị và ATTR_PROBE_SAMPLING_REPORT.md, dựng hoàn toàn từ CSV (+ sampling_heatmaps.pt, meta.json) trong
<out>/pilot/sampling; không chạy mô hình. Ghi thêm sampling_aggregate.csv (mọi phép so ghép cặp).

Thống kê: mỗi (ảnh, phương pháp, K) lấy trung bình qua seed TRƯỚC (tránh giả lặp), rồi hiệu ghép cặp theo ảnh và bootstrap
theo ảnh. p hai phía từ bootstrap (2·min(P*(≤0), P*(≥0))), Holm trên họ chính (InsDel blur/đen × S1–S4 × K, so với S0).
Phương sai: tỷ số theo ảnh, trung bình nhân và CI bootstrap trên log.
"""

from __future__ import annotations

import json
import math
import os
import warnings

import numpy as np

from .report import _read, fmt, md

# Thứ tự màu cố định theo phương pháp (bảng categorical mặc định của skill dataviz; không xoay vòng)
COLORS = {"S0": "#2a78d6", "S1": "#eb6834", "S2": "#1baf7a", "S3": "#eda100", "S4": "#e87ba4", "S5": "#008300",
          "S6": "#6250d6", "S7": "#e34948"}
MARKERS = {"S0": "o", "S1": "s", "S2": "^", "S3": "D", "S4": "v", "S5": "P", "S6": "X", "S7": "*"}
MUTED = "#6b6a63"
PRIMARY_FAMILY = ("S1", "S2", "S3", "S4")
LOWER_BETTER = {"del_blur", "del_black", "signed_del_blur", "signed_del_black", "abs_resid_rel", "rel_rmse",
                "rel_rmse_deb", "var_elem", "var_patch", "var_elem_rel"}


def base_of(m: str) -> str:
    return m.split("@")[0]


def variant_of(m: str) -> str:
    return m.split("@")[1] if "@" in m else ""


def same_op_control(m: str) -> str | None:
    b = base_of(m)
    if b in ("S2", "S3", "S4", "S5"):
        return "S6" + ("@" + variant_of(m) if variant_of(m) else "")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# thống kê
# ─────────────────────────────────────────────────────────────────────────────

def boot_stats(d, B: int, seed: int, lower_better: bool = False) -> dict:
    d = np.asarray([v for v in d if np.isfinite(v)], dtype=float)
    n = len(d)
    out = {"n_images": n, "mean": float(np.mean(d)) if n else np.nan, "median": float(np.median(d)) if n else np.nan}
    if n < 2:
        out.update({"lo95": np.nan, "hi95": np.nan, "p_boot": np.nan, "frac_improved": np.nan, "sign_p": np.nan, "dz": np.nan})
        return out
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, n, (B, n))].mean(1)
    out["lo95"], out["hi95"] = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
    out["p_boot"] = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    imp = (d < 0) if lower_better else (d > 0)
    out["frac_improved"] = float(imp.mean())
    nz = d[d != 0]
    k, m = int(((nz < 0) if lower_better else (nz > 0)).sum()), len(nz)
    if m:
        tail = sum(math.comb(m, i) for i in range(0, min(k, m - k) + 1)) / 2 ** m
        out["sign_p"] = float(min(1.0, 2 * tail))
    else:
        out["sign_p"] = 1.0
    sd = float(np.std(d, ddof=1))
    out["dz"] = float(np.mean(d)) / sd if sd > 0 else np.nan
    return out


def holm(ps) -> list:
    ps = [p if np.isfinite(p) else 1.0 for p in ps]
    order = np.argsort(ps)
    m = len(ps)
    adj = [1.0] * m
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j]))
        adj[j] = run
    return adj


def paired_diff(pi, metric: str, a: str, b: str, K: int):
    import pandas as pd
    if metric not in pi:
        return pd.Series(dtype=float)
    da = pi[(pi.method == a) & (pi.K == K)].set_index("image")[metric]
    db = pi[(pi.method == b) & (pi.K == K)].set_index("image")[metric]
    return (da - db).dropna()


def log_ratio(pi, metric: str, a: str, b: str, K: int):
    import pandas as pd
    if metric not in pi:
        return pd.Series(dtype=float)
    da = pi[(pi.method == a) & (pi.K == K)].set_index("image")[metric]
    db = pi[(pi.method == b) & (pi.K == K)].set_index("image")[metric]
    r = (da / db).replace([np.inf, -np.inf], np.nan)
    return np.log(r[r > 0]).dropna()


def aggregate(pi, st: dict) -> "pd.DataFrame":
    import pandas as pd
    B, seed, alpha = int(st.get("bootstrap", 10000)), int(st.get("seed", 4)), float(st.get("alpha", 0.05))
    methods = list(dict.fromkeys(pi.method))
    Ks = sorted(set(pi.K))
    nseeds = pi.groupby("method").n_seeds.max().to_dict() if "n_seeds" in pi else {}
    bgrad = pi.groupby(["method", "K"]).B_grad.mean().to_dict() if "B_grad" in pi else {}
    qual = [c for c in ("insdel_blur", "insdel_black", "ins_blur", "del_blur", "ins_black", "del_black", "iou", "energy_mask",
                        "top_in_mask", "stab_cos", "stab_spearman", "stab_topk", "stab_patch_cos", "stab_patch_spearman",
                        "abs_resid_rel", "signed_insdel_blur", "signed_insdel_black", "cos_ref", "spearman_ref") if c in pi]
    comps = []
    for m in methods:
        if m != "S0":
            comps.append(("vs_S0", m, "S0"))
        c = same_op_control(m)
        if c and c in methods:
            comps.append(("vs_same_operator_S6", m, c))
    for a, b in (("S4", "S2"), ("S3", "S2"), ("S3", "S1"), ("S2", "S6"), ("S4", "S6")):
        if a in methods and b in methods and ("vs_same_operator_S6", a, b) not in comps:
            comps.append(("contrast", a, b))
    rows = []
    for kind, a, b in comps:
        for K in Ks:
            for metric in qual:
                d = paired_diff(pi, metric, a, b, K)
                if not len(d):
                    continue
                s = boot_stats(d.values, B, seed, metric in LOWER_BETTER)
                rows.append({"comparison": kind, "a": a, "b": b, "metric": metric, "type": "diff", "K": K,
                             "B_grad": bgrad.get((a, K), np.nan), "n_seeds": nseeds.get(a, np.nan), **s,
                             "primary": kind == "vs_S0" and a in PRIMARY_FAMILY and metric in ("insdel_blur", "insdel_black")})
            for metric in ("var_elem", "var_patch", "rel_mse_deb"):
                if metric == "rel_mse_deb":
                    same_ref = (base_of(a) in ("S1",) and b == "S0") or (same_op_control(a) == b) or (a, b) in (("S4", "S2"), ("S3", "S2"))
                    if not same_ref:
                        continue
                lr = log_ratio(pi, metric, a, b, K)
                if not len(lr):
                    continue
                s = boot_stats(lr.values, B, seed, True)
                rows.append({"comparison": kind, "a": a, "b": b, "metric": metric, "type": "log_ratio", "K": K,
                             "B_grad": bgrad.get((a, K), np.nan), "n_seeds": nseeds.get(a, np.nan), **s,
                             "ratio_gmean": math.exp(s["mean"]) if np.isfinite(s["mean"]) else np.nan,
                             "ratio_lo95": math.exp(s["lo95"]) if np.isfinite(s["lo95"]) else np.nan,
                             "ratio_hi95": math.exp(s["hi95"]) if np.isfinite(s["hi95"]) else np.nan, "primary": False})
    ag = pd.DataFrame(rows)
    if len(ag):
        ag["p_holm"] = np.nan
        prim = ag.index[ag.primary]
        if len(prim):
            ag.loc[prim, "p_holm"] = holm(list(ag.loc[prim, "p_boot"]))
        ag["significant"] = ag.p_holm < alpha
    return ag


def cell_status(r, st: dict) -> str:
    me, wf, alpha = float(st.get("min_effect", 0.01)), float(st.get("win_frac", 0.5)), float(st.get("alpha", 0.05))
    if not np.isfinite(r.get("p_holm", np.nan)) or r["p_holm"] >= alpha:
        return "ns"
    if r["mean"] > me and r["frac_improved"] >= wf:
        return "win"
    if r["mean"] < -me:
        return "loss"
    return "sig_small"


def verdict(ag, st: dict, n_images: int, debug: bool, Ks: list) -> dict:
    out = {"wins": {}, "losses": {}, "category": "Inconclusive", "reasons": []}
    if ag is None or not len(ag):
        out["reasons"].append("không có dữ liệu")
        return out
    prim = ag[ag.primary].copy()
    if not len(prim):
        out["reasons"].append("không có phép so chính (S1–S4 vs S0 trên InsDel)")
        return out
    prim["status"] = [cell_status(r, st) for _, r in prim.iterrows()]
    for m, g in prim.groupby("a"):
        out["wins"][m] = {b: sorted(g[(g.metric == b) & (g.status == "win")].K.tolist()) for b in ("insdel_blur", "insdel_black")}
        out["losses"][m] = {b: sorted(g[(g.metric == b) & (g.status == "loss")].K.tolist()) for b in ("insdel_blur", "insdel_black")}
    need = math.ceil(len(Ks) / 2)
    positive = [m for m in out["wins"] if any(len(v) >= need for v in out["wins"][m].values())
                and not any(out["losses"][m].values())]
    any_win = any(any(v) for w in out["wins"].values() for v in w.values())
    me = float(st.get("min_effect", 0.01))
    frac_excl = float((prim.hi95 < me).mean())
    out["frac_cells_ci_below_min_effect"] = frac_excl
    out["positive_methods"] = positive
    if debug:
        out["category"] = "Inconclusive"
        out["reasons"].append("chạy DEBUG (ít ảnh, N nhỏ) — không dùng để kết luận")
    elif n_images < int(st.get("min_images", 16)):
        out["category"] = "Inconclusive"
        out["reasons"].append(f"chỉ {n_images} ảnh < min_images = {st.get('min_images', 16)}")
    elif positive:
        out["category"] = "Positive"
    elif any_win:
        out["category"] = "Conditional"
    elif frac_excl >= 0.75:
        out["category"] = "Negative"
    else:
        out["category"] = "Inconclusive"
        out["reasons"].append(f"không có thắng có ý nghĩa nhưng chỉ {frac_excl:.0%} ô có CI trên < min_effect (cần ≥ 75%) — "
                              "chưa loại trừ được hiệu ứng thực tế")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# đồ thị
# ─────────────────────────────────────────────────────────────────────────────

def _plt():
    from .plots import _setup
    return _setup()


def _style(m: str):
    b = base_of(m)
    return {"color": COLORS.get(b, MUTED), "marker": MARKERS.get(b, "o"), "linestyle": "-" if "@" not in m else "--"}


def _mean_ci(pi, m, metric, Ks):
    mu, lo, hi = [], [], []
    rng = np.random.default_rng(0)
    for K in Ks:
        v = pi[(pi.method == m) & (pi.K == K)][metric].dropna().values if metric in pi else np.array([])
        if len(v) == 0:
            mu.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        mu.append(v.mean())
        if len(v) > 1:
            bs = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
            lo.append(np.quantile(bs, 0.025)); hi.append(np.quantile(bs, 0.975))
        else:
            lo.append(np.nan); hi.append(np.nan)
    return np.array(mu), np.array(lo), np.array(hi)


def plots(out: str, pi, ps, ag, cov, cvg, loc, curves, heat, ref, bases, Kmax) -> list:
    files = []
    plt = _plt()
    methods = list(dict.fromkeys(pi.method))
    Ks = sorted(set(pi.K))

    def save(fig, name):
        f = os.path.join(out, name)
        fig.tight_layout()
        fig.savefig(f)
        plt.close(fig)
        files.append(f)

    def legend(ax):
        ax.legend(fontsize=7, ncol=2)

    for xkey, name, xl in (("K", "sampling_insdel_vs_K.png", "K (số đường hoàn chỉnh)"),
                           ("B_grad", "sampling_insdel_vs_budget.png", "B = tổng lượt forward+backward của ensemble")):
        try:
            fig, ax = plt.subplots(1, len(bases), figsize=(5.2 * len(bases), 3.8), squeeze=False)
            for i, b in enumerate(bases):
                a = ax[0, i]
                for m in methods:
                    mu, lo, hi = _mean_ci(pi, m, f"insdel_{b}", Ks)
                    xs = np.array(Ks, dtype=float) if xkey == "K" else np.array(
                        [pi[(pi.method == m) & (pi.K == K)].B_grad.mean() for K in Ks], dtype=float)
                    stl = _style(m)
                    a.plot(xs, mu, label=m, **stl)
                    a.fill_between(xs, lo, hi, color=stl["color"], alpha=0.12, linewidth=0)
                a.set_xscale("log", base=2)
                a.set_xlabel(xl)
                a.set_ylabel(f"InsDel ({b}) — trung bình ảnh, CI 95%")
                a.set_title(f"baseline mask: {b}", fontsize=9)
                legend(a)
            save(fig, name)
        except Exception as ex:
            print(f"[sampling plots] {name}: {type(ex).__name__}: {ex}")

    try:
        if "rel_rmse_deb" in pi and pi.rel_rmse_deb.notna().any():
            fig, ax = plt.subplots(figsize=(5.8, 4))
            for m in methods:
                mu, lo, hi = _mean_ci(pi, m, "rel_rmse_deb", Ks)
                if np.all(~np.isfinite(mu)):
                    continue
                stl = _style(m)
                op = pi[pi.method == m].operator.iloc[0]
                ax.plot(Ks, mu, label=f"{m} (ref {op})", **stl)
            ax.plot(Ks, [np.nanmax(_mean_ci(pi, "S0", "rel_rmse_deb", Ks)[0][:1]) * math.sqrt(Ks[0] / k) for k in Ks],
                    color=MUTED, lw=1, ls=":", label="∝ 1/√K (từ S0, K nhỏ nhất)")
            ax.set_xscale("log", base=2); ax.set_yscale("log")
            ax.set_xlabel("K"); ax.set_ylabel("‖Â_K − μ_ref‖ / ‖μ_ref‖ (khử nhiễu ref)")
            ax.set_title("Sai số ước lượng so với reference CÙNG operator", fontsize=9)
            legend(ax)
            save(fig, "sampling_rmse_vs_K.png")
    except Exception as ex:
        print(f"[sampling plots] rmse: {type(ex).__name__}: {ex}")

    try:
        if cov is not None and len(cov):
            fams = [f for f in ("S0_independent", "S1_antithetic", "S6_independent", "S3_antithetic") if f in set(cov.family)]
            c = cov[cov.block == "primary"].groupby(["image", "family"])[["rho_elem", "rho_patch"]].mean().reset_index()
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
            for i, col in enumerate(("rho_elem", "rho_patch")):
                data = [c[c.family == f][col].dropna().values for f in fams]
                ax[i].boxplot(data, widths=0.5, showfliers=False)
                for j, v in enumerate(data):
                    ax[i].scatter(np.full(len(v), j + 1) + np.random.default_rng(j).uniform(-0.12, 0.12, len(v)), v, s=14,
                                  color=COLORS["S1" if "S1" in fams[j] else "S3" if "S3" in fams[j] else "S0" if "S0" in fams[j] else "S6"],
                                  edgecolor="white", linewidth=0.8, zorder=3)
                ax[i].set_xticks(range(1, len(fams) + 1)); ax[i].set_xticklabels(fams, fontsize=7, rotation=15)
                ax[i].axhline(0, color=MUTED, lw=0.8)
                ax[i].set_ylabel("tương quan gộp của độ lệch quanh μ_ref" + (" (patch)" if i else " (phần tử)"))
            fig.suptitle("D1 — cặp antithetic so với cặp độc lập (mỗi điểm = một ảnh, trung bình seed)", fontsize=9)
            save(fig, "sampling_antithetic_corr.png")
    except Exception as ex:
        print(f"[sampling plots] antithetic: {type(ex).__name__}: {ex}")

    try:
        if cvg is not None and len(cvg):
            cols = [c for c in ("deficit_third", "nbr4_order_balance", "nbr4_overlap", "tau_spread") if c in cvg]
            fig, ax = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.5), squeeze=False)
            g = cvg.groupby(["method", "K"])[cols].mean().reset_index()
            for i, col in enumerate(cols):
                for m in dict.fromkeys(g.method):
                    s = g[g.method == m]
                    ax[0, i].plot(s.K, s[col], label=m, **_style(m))
                ax[0, i].set_xscale("log", base=2); ax[0, i].set_xlabel("K"); ax[0, i].set_title(col, fontsize=9)
            legend(ax[0, 0])
            fig.suptitle("D2 — coverage của thời điểm kích hoạt cấp patch (τ = lúc lịch trình trung bình patch ≥ ½)", fontsize=9)
            save(fig, "sampling_coverage.png")
    except Exception as ex:
        print(f"[sampling plots] coverage: {type(ex).__name__}: {ex}")

    try:
        if ag is not None and len(ag):
            s = ag[(ag.comparison == "vs_S0") & (ag.type == "diff") & ag.metric.isin([f"insdel_{b}" for b in bases])]
            fig, ax = plt.subplots(1, len(bases), figsize=(5.2 * len(bases), 3.6), squeeze=False)
            for i, b in enumerate(bases):
                for m in dict.fromkeys(s.a):
                    q = s[(s.a == m) & (s.metric == f"insdel_{b}")].sort_values("K")
                    ax[0, i].plot(q.K, q.frac_improved, label=m, **_style(m))
                ax[0, i].axhline(0.5, color=MUTED, lw=0.8)
                ax[0, i].set_ylim(-0.02, 1.02); ax[0, i].set_xscale("log", base=2)
                ax[0, i].set_xlabel("K"); ax[0, i].set_ylabel("tỷ lệ ảnh tốt hơn S0"); ax[0, i].set_title(f"InsDel {b}", fontsize=9)
                legend(ax[0, i])
            save(fig, "sampling_winrate_vs_S0.png")
    except Exception as ex:
        print(f"[sampling plots] winrate: {type(ex).__name__}: {ex}")

    try:
        if ag is not None and len(ag):
            s = ag[(ag.type == "log_ratio") & (ag.metric == "var_elem")]
            sel = s[((s.comparison == "vs_S0") & (s.a == "S1")) | (s.comparison == "vs_same_operator_S6")]
            if len(sel):
                fig, ax = plt.subplots(figsize=(6, 4))
                for (a, b), q in sel.groupby(["a", "b"]):
                    q = q.sort_values("K")
                    stl = _style(a)
                    ax.plot(q.K, q.ratio_gmean, label=f"{a} / {b}", **stl)
                    ax.fill_between(q.K.values.astype(float), q.ratio_lo95, q.ratio_hi95, color=stl["color"], alpha=0.12, linewidth=0)
                ax.axhline(1, color=MUTED, lw=0.8)
                ax.set_xscale("log", base=2); ax.set_yscale("log")
                ax.set_xlabel("K"); ax.set_ylabel("tỷ số phương sai giữa ensemble (trung bình nhân, CI 95%)")
                ax.set_title("Phương sai so với đối chứng độc lập CÙNG operator (< 1 = giảm)", fontsize=9)
                legend(ax)
                save(fig, "sampling_variance_ratio.png")
    except Exception as ex:
        print(f"[sampling plots] variance: {type(ex).__name__}: {ex}")

    try:
        if loc is not None and len(loc):
            fig, ax = plt.subplots(figsize=(6, 3.8))
            for blk, q in loc.groupby("block"):
                g = q.groupby("dist")[["abs_I_rel", "null_abs_I_rel"]].mean()
                ax.plot(g.index, g.abs_I_rel, marker="o", label=f"{blk}: quan sát", color=COLORS["S6"])
                ax.plot(g.index, g.null_abs_I_rel, marker="x", ls=":", label=f"{blk}: mức nền hoán vị", color=MUTED)
            ax.set_xlabel("khoảng cách Chebyshev giữa hai patch"); ax.set_ylabel("|tác động thứ tự| / |D|")
            ax.set_title("D3 — proxy: credit của g đổi bao nhiêu khi h kích hoạt trước g (đường S6)", fontsize=9)
            legend(ax)
            save(fig, "sampling_locality.png")
    except Exception as ex:
        print(f"[sampling plots] locality: {type(ex).__name__}: {ex}")
    return files


def heat_and_curves(out: str, pi, curves, heat, bases, Kmax, cases: list) -> list:
    """Bản đồ (seed 0, K_max) và đường cong insertion/deletion (prob, logit) cho các ảnh thắng / thua."""
    files = []
    if not cases:
        return files
    plt = _plt()
    try:
        if heat:
            show = [m for m in ("S0", "S1", "S2", "S4", "S6") if any(m in heat.get(im, {}) for im, _ in cases)]
            refs = sorted({k for im, _ in cases for k in heat.get(im, {}) if k.startswith("ref:")})
            cols = ["ảnh"] + show + refs
            fig, ax = plt.subplots(len(cases), len(cols), figsize=(2.1 * len(cols), 2.3 * len(cases)), squeeze=False)
            for i, (im, tag) in enumerate(cases):
                h = heat.get(im, {})
                for j, c in enumerate(cols):
                    a = ax[i, j]
                    a.set_xticks([]); a.set_yticks([]); a.grid(False)
                    if c == "ảnh":
                        if "_x" in h:
                            x = h["_x"].float()[0].permute(1, 2, 0).numpy()
                            x = x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                            a.imshow(np.clip(x, 0, 1))
                            if "_mask" in h:
                                a.contour(h["_mask"].numpy().astype(float), levels=[0.5], colors="white", linewidths=0.8)
                        a.set_title(f"{im} ({tag})", fontsize=7)
                    elif c in h:
                        v = h[c]["amap"].float().numpy()
                        a.imshow(v, cmap="Blues", vmin=0, vmax=np.quantile(v, 0.99))
                        a.set_title(c if not c.startswith("ref:") else c.replace("ref:", "μ_ref "), fontsize=7)
            fig.suptitle(f"Σ_c|A| — ensemble K = {Kmax}, seed 0 (màu = độ lớn, cùng thang trong từng ô)", fontsize=9)
            f = os.path.join(out, "sampling_heatmaps.png")
            fig.tight_layout(); fig.savefig(f); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[sampling plots] heatmaps: {type(ex).__name__}: {ex}")
    try:
        if curves is not None and len(curves):
            fig, ax = plt.subplots(len(cases), 2 * len(bases), figsize=(3.6 * 2 * len(bases), 2.8 * len(cases)), squeeze=False)
            for i, (im, tag) in enumerate(cases):
                c = curves[curves.image == im]
                for j, (b, meas) in enumerate([(b, ms) for b in bases for ms in ("prob", "logit")]):
                    a = ax[i, j]
                    for m in ("S0", "S2", "S4"):
                        for kind, ls in (("ins", "-"), ("del", "--")):
                            q = c[(c.method == m) & (c.base_kind == b) & (c.measure == meas) & (c.curve == kind)].sort_values("step")
                            if len(q):
                                a.plot(q.frac, q.value, ls=ls, color=COLORS[m], lw=1.5, label=f"{m} {kind}")
                    a.set_title(f"{im} ({tag}) — {b}, {meas}", fontsize=7)
                    a.set_xlabel("tỷ lệ pixel đã chèn / xoá", fontsize=7)
                    if i == 0 and j == 0:
                        a.legend(fontsize=6, ncol=2)
            fig.tight_layout()
            f = os.path.join(out, "sampling_insdel_curves.png")
            fig.savefig(f); plt.close(fig); files.append(f)
    except Exception as ex:
        print(f"[sampling plots] curves: {type(ex).__name__}: {ex}")
    return files


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo
# ─────────────────────────────────────────────────────────────────────────────

AUDIT = """
**Audit (trước khi implement).**

| Câu hỏi | Trong code P0–P3 |
|---|---|
| SPI sinh đường thế nào | `paths.spi_increments(α, N, shape)`: mỗi phần tử (pixel × kênh, 150 528 phần tử) có số gia Dirichlet(α/N, …, α/N) **độc lập** trên N ô (Gamma trong log-space + softmax); `cum_nodes` → Z tại N+1 nút; đường = đường gấp khúc x′ + Z⊙Δx. Pilot: α = 10, N = 64. |
| α, số nút, coupling | α = nồng độ DP (α → ∞: IG; α nhỏ: bước nhảy lớn). N = số ô **thời gian**. "Coupling giữa các lưới" = lưới thời gian (rút ở N_max, cộng gộp số gia cho N thô) — **không** phải lưới không gian. |
| Độc lập / đồng bộ | Hoàn toàn độc lập theo phần tử (cả 3 kênh RGB). Không có cấu trúc patch hay nhiều tầng. Nhóm không gian chỉ có ở E6–E8 (`e7.group_labels`), lưới g×g chỉ ở E9 (chẩn đoán). |
| canonical | `canon_rand` = Σ_k g(x_k + U_kΔx_k)⊙Δx_k, một U/đoạn dùng chung mọi phần tử — không chệch trên đường gấp khúc. `mechanism.aw_stream` với M = 1 ≡ canon_rand (P4 dùng hàm này). |
| Đếm eval | `quad.Counter.n` đếm mỗi ảnh qua forward+backward trong `fg`. Forward-only (InsDel, LIME) KHÔNG được đếm ở P0–P3 → P4 thêm `sampling.FwdCounter`. P1 tính thêm gradient tại nút (canon_left, IDGI): 129 eval/đường; P4 chỉ canon_rand: N·M = 64 eval/đường. |
| InsDel, IoU, energy | `metrics.insdel`: hạng pixel theo Σ_c abs(A) (tuyệt đối), 50 bước, xác suất softmax lớp đích, trả AUC_ins − AUC_del. IoU: top-k (k = diện tích mask) theo Σ_c abs(A), hoà phá bằng hoán vị cố định theo ảnh. Energy = Σ_mask A² / Σ A². |
| Cache, seed, resume | File .pt mỗi ảnh kèm hash config; `seed_of` = sha256 các phần; chạy lại bỏ qua phần trùng hash. |
| Tái sử dụng eval | P1–P3: `shift/unshift` áp thành phần cộng thêm. P4: đường dùng chung giữa phương pháp (S1 ⊃ nửa đầu S0, S3 ⊃ nửa đầu S2) cùng khoá cầu phương ⇒ không tính lại. |

**Tái sử dụng:** `spi_increments`, `cum_nodes`, `aw_stream`, `fg/Counter`, `Ctx` (dữ liệu, split, model, seed), `iou`, `energy_mask`,
`tie_perm`, `make_base`, `report.fmt/md`, `plots._setup`. **Bổ sung:** `probe/sampling.py` (sampler, chẩn đoán, LIME-grid, InsDel trả riêng
AUC + đường cong prob/logit — cùng thứ tự hạng với `metrics.insdel`, có test so khớp), `probe/sampling_benchmark.py` (runner, cache theo
khối × seed, ước tính, kế hoạch ngân sách), `probe/sampling_report.py`, `tests/test_sampling.py`, mục `sampling:` trong config (thêm vào
`UNHASHED` nên hash pilot P0–P3 **không đổi**), lệnh `sampling-*` trong `scripts/run.sh` và CLI. Không sửa logic P0–P3.

**Rủi ro so sánh không công bằng và cách xử lý.** (a) S2–S7 ở cấp patch là **operator khác** S0 (các phần tử trong patch chung lịch
trình) → S6 (patch, pha độc lập) là đối chứng cùng operator; tách "đổi operator" (S6 − S0) và "cân bằng" (S2 − S6); mỗi operator có
reference riêng. (b) Antithetic: hai đường phụ thuộc → phân tích theo ensemble, K tính cả hai. (c) Mọi đường tốn đúng N·M eval → K ↔ B
một-một; LIME báo riêng. (d) InsDel phi tuyến theo A → không đánh đồng giảm phương sai với tăng InsDel. (e) Hạng |A| giữ nguyên; có dấu
chỉ là phân tích phụ có nhãn. (f) Nhiễu cầu phương (M = 1) có thể che hiệu ứng thiết kế → đo tỷ lệ ở khối ref.
"""

DEFS = """
| | định nghĩa | operator (phân phối của MỘT đường) |
|---|---|---|
| S0 | SPI độc lập — đúng `spi_increments(α, N)` hiện tại | q_SPI |
| S1 | K/2 cặp (Z, Z^R), Z^R(t) = 1 − Z(1 − t), Z = đường S0 cùng chỉ số | q_SPI — Dirichlet(α/N,…) đối xứng ⇒ đảo số gia cùng phân phối ⇒ **không chệch**, cùng μ với S0 |
| S2 | lưới G×G; mỗi patch một DP(α, N) dùng chung cho mọi phần tử của patch; anchor (số gia lớn nhất) tại ⌊Nφ⌋; φ_{k,g} = frac(π_g + (z_g b(k) mod K_max + ε)/K_max), z_g lẻ ngẫu nhiên, b = đảo bit | q_patch |
| S3 | K/2 đường S2 đầu + đường đảo của chúng | q_patch (đường đảo cùng phân phối) |
| S4 | như S2 nhưng z_g = 1 + 2·parity(i+j) + 4r_g ⇒ cặp kề cạnh có độ lệch pha trải đều qua các đường; π_g, r_g, màu bàn cờ ngẫu nhiên (thành phần toàn cục) | q_patch |
| S5 | φ_{k,g} = π_g cố định cho cả ensemble (hình dạng vẫn mới mỗi đường) | q_patch |
| S6 | φ iid U(0,1) mới mỗi đường | q_patch (đối chứng độc lập) |
| S7 | φ = (hạng của q_g + τξ + U)/P, q_g = hạng hệ số LIME-grid (cao → muộn) | **khác** (phụ thuộc mô hình) |
| η > 0 | (1 − η)Z_patch + ηZ_S0 (đường S0 cùng chỉ số) | khác q_patch khi η ∉ {0} |

Bảo toàn phân phối: (i) xoay vòng số gia về anchor giữ đúng DP (dãy số gia trao đổi được ⇒ argmax đều và độc lập với hình dạng);
(ii) π_g iid U(0,1) ⇒ pha trong một đường iid U(0,1) ở S2/S4/S5/S6 ⇒ **cùng q_patch**; khác biệt S2/S4/S5 so với S6 là sai số ước lượng
thuần (ghép nối giữa các đường), không phải đổi operator. Kiểm bằng toán (ở trên) và thống kê (test T4: moment của Z, phân bố anchor,
tần suất pha) — chẩn đoán thống kê không phải chứng minh đầy đủ. S2/S4: với mọi tiền tố K = 2^j mỗi patch phủ đúng một lần mỗi cung 1/K
của pha (test T3). Lưu ý "cân bằng" là ở mức **pha anchor**; thời điểm kích hoạt thật (Z ≥ ½) còn phụ thuộc hình dạng — D2 đo trên τ thật.
"""


def _tests_text(out: str, cfg: dict) -> str:
    cands = [os.path.join(out, "tests_report.txt"),
             os.path.join((cfg.get("local", {}) or {}).get("out", "runs_probe"), "pilot", "sampling", "tests_report.txt")]
    for p in cands:
        if os.path.exists(p):
            lines = [l_.rstrip() for l_ in open(p, encoding="utf-8", errors="replace") if l_.strip()]
            summ = [l_ for l_ in lines if ("passed" in l_ or "failed" in l_ or "error" in l_.lower())]
            return f"`{p}`: " + (summ[-1] if summ else "(không thấy dòng tổng kết)") + "\n\n```\n" + "\n".join(lines[-12:]) + "\n```\n"
    return "_Chưa có tests_report.txt — chạy `bash scripts/run.sh sampling-test`._\n"


def write(out: str, cfg: dict) -> str:
    import pandas as pd
    import torch
    warnings.filterwarnings("ignore")
    from .sampling_benchmark import report_path, settings
    S = settings(cfg)
    st = S.get("stats", {}) or {}
    meta_p = os.path.join(out, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    rd = lambda n: _read(out, n)
    ps, pi, cvg, cov, cpair, bud, pth, loc, cur, ref, lime, imgs = (rd(f"sampling_{n}") for n in (
        "per_seed", "per_image", "coverage", "covariance", "covariance_pairs", "budget", "paths", "locality", "curves", "ref",
        "lime", "images"))
    hp = os.path.join(out, "sampling_heatmaps.pt")
    heat = torch.load(hp, weights_only=False) if os.path.exists(hp) else {}
    bases = list(S["insdel_bases"])
    Ks, Kmax = S["K"], S["Kmax"]
    rp = report_path(cfg)
    L = ["# ATTR_PROBE_SAMPLING_REPORT — P4: Real-CNN Stochastic Path Sampling Benchmark\n"]
    debug = bool(meta.get("debug", cfg.get("_debug")))
    if debug:
        L.append("> **DEBUG** — ít ảnh, N nhỏ, K nhỏ. Mọi con số chỉ để kiểm pipeline, KHÔNG dùng để kết luận.\n")
    if pi is None or not len(pi):
        L.append("_Chưa có kết quả: chưa chạy `bash scripts/run.sh sampling-pilot` (hoặc `sampling-debug`). Không có số liệu nào "
                 "trong báo cáo này là kết quả thật._\n")
        L.append("## Tests\n")
        L.append(_tests_text(out, cfg))
        with open(rp, "w") as fh:
            fh.write("\n".join(L) + "\n")
        print(f"[sampling report] {rp} (trống)")
        return rp

    ag = aggregate(pi, st)
    if len(ag):
        ag.to_csv(os.path.join(out, "sampling_aggregate.csv"), index=False)
    n_img = int(pi.image.nunique())
    seeds_used = int(pi[pi.method == "S0"].n_seeds.max()) if "n_seeds" in pi else 0
    V = verdict(ag, st, n_img, debug, Ks)
    files = plots(out, pi, ps, ag, cov, cvg, loc, cur, heat, ref, bases, Kmax)

    # ── loss / win cases (D4)
    cases = []
    lossrows = []
    pb = bases[0]
    for m in ("S2", "S4"):
        d = paired_diff(pi, f"insdel_{pb}", m, "S0", Kmax)
        if len(d):
            cases.append((d.idxmax(), f"{m} thắng nhất"))
            cases.append((d.idxmin(), f"{m} thua nhất"))
    seen, cases_u = set(), []
    for c in cases:
        if c[0] not in seen:
            seen.add(c[0]); cases_u.append(c)
    files += heat_and_curves(out, pi, cur, heat, bases, Kmax, cases_u[:4])

    # ── 1. tóm tắt
    gpu = (meta.get("provenance") or {}).get("gpu", "?")
    gb = bud.grad_evals.sum() if bud is not None and "grad_evals" in bud else float("nan")
    fb = bud.fwd_evals.sum() if bud is not None and "fwd_evals" in bud else float("nan")
    L.append("## 1. Executive summary\n")
    L.append(f"- {n_img} ảnh (tiền tố split pilot, như P1), {seeds_used} seed độc lập mỗi (ảnh, phương pháp, K); K ∈ {Ks}; "
             f"SPI α = {S['alpha']:g}, N = {S['N']}, M = {S['M']} ⇒ {S['N'] * S['M']} lượt forward+backward mỗi đường; "
             f"lưới chính {S['grid']}×{S['grid']}, η = {S['eta']:g}. GPU `{gpu}`; tổng {fmt(gb)} grad eval + {fmt(fb)} forward "
             f"(cache, gộp mọi lượt chạy).")
    L.append(f"- Khối đã chạy: {sorted(set(pi.block))}; khối phụ bị bỏ vì ngân sách: {(meta.get('plan') or {}).get('secondary_skipped') or '—'}.")
    L.append(f"- **Phán quyết: {V['category']}**" + (f" — {'; '.join(V['reasons'])}" if V["reasons"] else "") + ".")
    if V.get("positive_methods"):
        L.append(f"- Phương pháp đạt tiêu chí Positive: {V['positive_methods']}.")
    L.append("- Chi tiết theo câu hỏi Q1–Q5 ở mục 13. Mọi số trong báo cáo được tính từ CSV của lượt chạy này; không có số dự kiến.\n")

    # ── 2. audit
    L.append("## 2. Repository audit and implementation\n")
    L.append(AUDIT)
    L.append("### Tests\n")
    L.append(_tests_text(out, cfg))

    # ── 3. định nghĩa
    L.append("## 3. Sampling definitions\n")
    L.append(DEFS)

    # ── 4. giao thức & ngân sách
    L.append("## 4. Experimental protocol and budget\n")
    L.append(f"Target: logit lớp dự đoán trên ảnh gốc, cố định cho mọi đường/phương pháp. Baseline đen. Estimator: cầu phương phân tầng "
             f"M = {S['M']} điểm/đoạn trên đường gấp khúc (M = 1 ≡ `canon_rand`), cùng cho mọi phương pháp. Ensemble K = trung bình K đường "
             "(K đầu của ngân hàng đường cố định theo seed; thiết kế cân bằng lồng nhau nên mỗi tiền tố vẫn cân bằng). Cặp antithetic: cả hai "
             "đường tính vào K. Reference: μ_ref của từng operator từ K_ref đường độc lập với khoá ngẫu nhiên riêng.\n")
    L.append("**K và B.** Mọi đường tốn đúng N·M lượt forward+backward ⇒ B = K·N·M như nhau cho mọi phương pháp ở cùng K (bảng dưới). "
             "Đường dùng chung (S1 ⊃ nửa S0, S3 ⊃ nửa S2) chỉ tiết kiệm tính toán của benchmark, không giảm B của ensemble. "
             "S7 tốn thêm forward cho prior LIME (cột `B_prior_fwd`, không trộn vào B).\n")
    bt = pi.groupby(["method", "K"])[["B_grad"] + (["B_prior_fwd"] if "B_prior_fwd" in pi else [])].mean().reset_index()
    L.append(md(bt.pivot(index="method", columns="K", values="B_grad").reset_index()))
    if bud is not None and len(bud):
        L.append("Ngân sách thực tế (đếm trong code, mọi ảnh):\n")
        L.append(md(bud.groupby("block")[["grad_evals", "fwd_evals", "seconds_grad", "seconds_metrics"]].sum().reset_index()))
    if meta:
        pr = meta.get("provenance", {}) or {}
        L.append(f"- Reproducibility: git `{(pr.get('git') or {}).get('commit', 'không có git')}`"
                 f"{' (dirty)' if (pr.get('git') or {}).get('dirty') else ''}, source sha256 `{str(pr.get('source_sha256', ''))[:16]}`; "
                 f"torch {pr.get('torch')}, torchvision {pr.get('torchvision')}, CUDA {pr.get('cuda')}; {pr.get('model')}; TF32 {pr.get('allow_tf32')}.")
        L.append(f"- Hash khối: " + ", ".join(f"{k} `{v[:12]}`" for k, v in (meta.get("digests") or {}).items())
                 + f"; hash pilot P0–P3 `{str(meta.get('pilot_config_digest', ''))[:12]}`; seed toàn cục {meta.get('global_seed')}; "
                 f"seed ensemble {meta.get('seeds')}.")
        L.append(f"- Ảnh (chỉ số trong gtsegs_ijcv.mat): {meta.get('indices')}.")
        L.append(f"- Đo tốc độ: {fmt((meta.get('bench') or {}).get('per_eval', float('nan')) * 1e3)} ms/grad eval, "
                 f"{fmt((meta.get('bench') or {}).get('per_fwd', float('nan')) * 1e3)} ms/forward; lượt chạy cuối "
                 f"{fmt(meta.get('minutes_total_this_run'))} phút ({meta.get('minutes_by_block')}); kế hoạch: {meta.get('plan')}.\n")

    def table(metric, methods=None, agg="mean"):
        if metric not in pi:
            return "_(không có)_\n"
        q = pi[pi.method.isin(methods)] if methods else pi
        return md(q.pivot_table(index="method", columns="K", values=metric, aggfunc=agg).reset_index())

    def cmp_table(comparison, metrics, a_filter=None, typ="diff"):
        if not len(ag):
            return "_(không có)_\n"
        q = ag[(ag.comparison == comparison) & ag.metric.isin(metrics) & (ag.type == typ)]
        if a_filter:
            q = q[q.a.isin(a_filter)]
        if typ == "diff":
            cols = ["a", "b", "metric", "K", "B_grad", "n_images", "n_seeds", "mean", "median", "lo95", "hi95", "frac_improved",
                    "p_boot", "p_holm", "sign_p", "dz"]
        else:
            cols = ["a", "b", "metric", "K", "n_images", "ratio_gmean", "ratio_lo95", "ratio_hi95", "frac_improved"]
        return md(q.sort_values(["metric", "a", "K"]), [c for c in cols if c in q])

    # ── 5. ổn định / ước lượng
    L.append("## 5. Attribution stability / estimation accuracy\n")
    if ref is not None and len(ref):
        rr = ref[ref.kind == "reference"]
        if len(rr):
            L.append("**Reference** (μ_ref, K_ref đường độc lập; median qua ảnh). `ref_rel_se` = √tr Var(μ̂_ref)/‖μ̂_ref‖ — sai số của "
                     "chính reference; RMSE được khử chệch bằng cách trừ tr Var(μ̂_ref).\n")
            cols = [c for c in ("K_ref", "ref_rel_se", "split_half_cos", "split_half_rel_diff", "insdel_blur", "insdel_black", "iou",
                                "energy_mask", "abs_resid_rel") if c in rr]
            L.append(md(rr.groupby("operator")[cols].median().reset_index()))
        od = ref[ref.kind == "operator_diff"]
        if len(od):
            L.append("**Khác biệt operator** (μ_patch so với μ_spi, median qua ảnh) — đây là phần khác biệt KHÔNG phải sai số ước lượng:\n")
            L.append(md(od[["rel_l2", "noise_floor_rel", "cos", "spearman", "topk"]].median().to_frame("median").T))
        qs = ref[ref.kind == "quad_share"]
        if len(qs):
            L.append(f"**Nhiễu cầu phương** (M = {S['M']}): phương sai do U chiếm median {fmt(qs.quad_share.median())} "
                     f"(min {fmt(qs.quad_share.min())}, max {fmt(qs.quad_share.max())}) tổng phương sai một đường S0. Phần còn lại do "
                     "lựa chọn đường — là phần thiết kế sampler có thể tác động.\n")
    L.append("**RMSE tương đối đã khử nhiễu ref** (‖Â_K − μ_ref‖/‖μ_ref‖, chỉ so với reference CÙNG operator; S7/η/lưới phụ không có ref), "
             "trung bình ảnh:\n")
    L.append(table("rel_rmse_deb"))
    L.append("**Phương sai giữa các ensemble độc lập** (Σ_i Var_seed(Â_K,i), trung bình ảnh) — không cần reference:\n")
    L.append(table("var_elem"))
    L.append("Tỷ số phương sai (trung bình nhân theo ảnh, CI 95%) — S1 so với S0 (cùng operator spi) và S2–S5 so với S6 (cùng operator patch):\n")
    L.append(cmp_table("vs_S0", ["var_elem", "var_patch", "rel_mse_deb"], ["S1"], typ="log_ratio"))
    L.append(cmp_table("vs_same_operator_S6", ["var_elem", "var_patch", "rel_mse_deb"], typ="log_ratio"))
    L.append("Ổn định giữa các seed (trung bình cặp seed): cosine có dấu / Spearman Σ_c|A| / top-5% overlap:\n")
    for c in ("stab_cos", "stab_spearman", "stab_topk", "stab_patch_spearman"):
        if c in pi:
            L.append(f"`{c}`\n")
            L.append(table(c))
    L.append("**Completeness**: |ΣA − (F(x) − F(x′))| / |F(x) − F(x′)| của ensemble (trung bình ảnh):\n")
    L.append(table("abs_resid_rel"))
    if pth is not None and len(pth):
        L.append(f"Theo từng đường: median |resid_rel| = {fmt(pth.resid_rel.abs().median())}, q90 = {fmt(pth.resid_rel.abs().quantile(0.9))}; "
                 "theo loại đường:\n")
        L.append(md(pth.assign(abs_resid=pth.resid_rel.abs()).groupby(["kind", "design"])[["abs_resid", "d_hat"]].median().reset_index()))
        L.append("`d_hat` = ∫(Z − t)² (mức exploration của đường; DP lý thuyết trên lưới ≈ ∫t(1−t)/(α+1)).\n")

    # ── 6. InsDel
    L.append("## 6. Insertion–Deletion results (primary endpoint)\n")
    L.append("Hạng pixel theo Σ_c|A| (quy ước P0–P3, như nhau cho mọi phương pháp); xác suất softmax lớp đích; hai giao thức mask "
             "báo **riêng**. Trung bình theo ảnh (mỗi ảnh đã lấy trung bình seed):\n")
    for b in bases:
        for pre in ("insdel", "ins", "del"):
            L.append(f"`{pre}_{b}`\n")
            L.append(table(f"{pre}_{b}"))
    L.append("**So với S0 (ghép cặp theo ảnh, bootstrap theo ảnh; p_holm trên họ chính S1–S4 × K × {blur, black})**:\n")
    L.append(cmp_table("vs_S0", [f"insdel_{b}" for b in bases]))
    L.append("**So với S6 (cùng operator patch) — phần do cân bằng / antithetic / pha cố định, tách khỏi phần đổi operator**:\n")
    L.append(cmp_table("vs_same_operator_S6", [f"insdel_{b}" for b in bases]))
    if any(f"signed_insdel_{b}" in pi for b in bases):
        L.append("_Phân tích phụ (nhãn rõ): hạng theo Σ_c A **có dấu**, K = K_max:_\n")
        for b in bases:
            L.append(table(f"signed_insdel_{b}"))

    # ── 7. localization
    L.append("## 7. Localization results (không dùng để chọn đường)\n")
    for c in ("iou", "energy_mask", "top_in_mask"):
        L.append(f"`{c}`\n")
        L.append(table(c))
    L.append(cmp_table("vs_S0", ["iou", "energy_mask", "top_in_mask"], list(PRIMARY_FAMILY)))
    L.append("Localization và InsDel được báo riêng; chúng có thể không đồng thuận.\n")

    # ── 8. antithetic
    L.append("## 8. Antithetic covariance diagnostics (D1)\n")
    if cov is not None and len(cov):
        c = cov.groupby(["block", "family", "center"])[["rho_elem", "rho_patch", "pair_vs_half_single", "frac_raw_cos_gt_0.99",
                                                         "frac_centered_cos_gt_0.9"]].mean().reset_index()
        L.append("ρ = tương quan gộp (qua cặp và phần tử) của độ lệch quanh μ_ref của hai đường trong cặp; `pair_vs_half_single` = MSE của "
                 "trung bình cặp / (MSE một đường / 2) ≈ 1 + ρ — < 1 nghĩa là cặp tốt hơn hai đường độc lập. Cặp độc lập (S0_independent, "
                 "S6_independent) là mức nền thực nghiệm. Trung bình qua ảnh và seed:\n")
        L.append(md(c))
        if cpair is not None and len(cpair):
            L.append("Phân bố theo cặp (median, q10, q90):\n")
            L.append(md(cpair.groupby("family")[["raw_cos", "centered_cos", "patch_centered_cos", "rel_diff"]].quantile(0.5).reset_index()))
            same = cpair[(cpair.family.str.contains("antithetic")) & (cpair.centered_cos > 0.9)]
            L.append(f"Cặp antithetic gần như trùng nhau (cosine độ lệch > 0,9): {len(same)} / "
                     f"{int(cpair.family.str.contains('antithetic').sum())} cặp.\n")
    else:
        L.append("_(không có dữ liệu)_\n")

    # ── 9. global vs neighborhood
    L.append("## 9. Global vs neighborhood balancing (D2, D3)\n")
    L.append(cmp_table("contrast", [f"insdel_{b}" for b in bases] + ["var_elem"], ["S4", "S3"]))
    if len(ag):
        q = ag[(ag.comparison.isin(["contrast", "vs_same_operator_S6"])) & (ag.a == "S4") & (ag.b == "S2") & (ag.type == "log_ratio")]
        if len(q):
            L.append(md(q[["a", "b", "metric", "K", "ratio_gmean", "ratio_lo95", "ratio_hi95"]]))
    if cvg is not None and len(cvg):
        cols = [c for c in ("tau_spread", "deficit_third", "deficit_kbin", "nbr4_order_balance", "nbr4_overlap", "nbr4_tie",
                            "far_order_balance", "far_overlap", "nbr4_spatial_corr", "elem_tau_spread", "d_hat_mean") if c in cvg]
        L.append("Coverage (trung bình ảnh × seed). τ = thời điểm lịch trình trung bình của patch đạt ½; `deficit_third` = tỷ lệ vùng sớm/giữa/muộn "
                 "mà một patch không chạm tới qua K đường; `order_balance` = 1 − 2|P(g trước h) − ½| (1 = cân bằng); `overlap` = tỷ lệ đường có "
                 f"|τ_g − τ_h| < {S['overlap_tol']}; S0/S1 ở cấp patch gần như đồng bộ (τ ≈ ½ do trung bình hoá) — đó là đặc điểm thật của SPI.\n")
        L.append(md(cvg[cvg.K.isin([min(Ks), Kmax])].groupby(["method", "K"])[cols].mean().reset_index()))
    if loc is not None and len(loc):
        L.append("**D3 — proxy locality** từ các đường S6 (không tốn thêm truy vấn): C̃_g = phần dư của credit patch g sau hồi quy lên đa thức "
                 "bậc 3 của thời điểm kích hoạt của CHÍNH g (khử hiệu ứng 'kích hoạt muộn/sớm', vốn bị đọc nhầm thành tác động của h); "
                 "|tác động thứ tự| = |E[C̃_g | h trước g] − E[C̃_g | h sau g]| / |D| theo khoảng cách Chebyshev, so với mức nền (hoán vị τ_h giữa "
                 "các đường). Đây KHÔNG phải độ mạnh tương tác thật (còn trộn nhiễu và thứ tự của các patch khác):\n")
        L.append(md(loc.groupby(["block", "dist"])[["abs_I_rel", "null_abs_I_rel", "n_pairs"]].mean().reset_index()))

    # ── 10. pixel-level
    L.append("## 10. Pixel-level randomness sensitivity (H4)\n")
    sec = [m for m in pi.method.unique() if "@" in m]
    if sec:
        L.append("Biến thể η / lưới (đường nét đứt trên đồ thị). Mỗi biến thể so với S0 và với S6 cùng biến thể:\n")
        L.append(cmp_table("vs_S0", [f"insdel_{b}" for b in bases], sec))
        L.append(cmp_table("vs_same_operator_S6", [f"insdel_{b}" for b in bases] + ["var_elem"], sec))
        s6v = [m for m in sec if base_of(m) == "S6"]
        if s6v and len(ag):
            L.append("Tác động của η / lưới trên operator độc lập (S6 biến thể − S6 chính) — so trực tiếp trong per-image:\n")
            rows = []
            for m in s6v:
                for K in Ks:
                    for b in bases:
                        d = paired_diff(pi, f"insdel_{b}", m, "S6", K)
                        if len(d):
                            rows.append({"a": m, "b": "S6", "metric": f"insdel_{b}", "K": K,
                                         **boot_stats(d.values, int(st.get("bootstrap", 10000)), int(st.get("seed", 4)))})
            L.append(md(pd.DataFrame(rows), ["a", "b", "metric", "K", "n_images", "mean", "lo95", "hi95", "frac_improved"]) if rows else "")
    else:
        L.append("_Khối η = 0,5 / lưới 14×14 **chưa chạy** (bị bỏ vì ngân sách hoặc chưa cấu hình) — không có kết quả._\n")

    # ── 11. thống kê & giới hạn
    L.append("## 11. Statistical significance and limitations\n")
    L.append(f"- Đơn vị thống kê là **ảnh** ({n_img} ảnh); seed được lấy trung bình trong ảnh trước (tránh giả lặp). Bootstrap "
             f"{st.get('bootstrap', 10000)} lần theo ảnh; p hai phía từ phân phối bootstrap (xấp xỉ); Holm trên họ chính; sign test báo kèm. "
             f"Ngưỡng thực tế min_effect = {st.get('min_effect', 0.01)} (đơn vị AUC); `dz` = hiệu trung bình / SD hiệu theo ảnh.")
    L.append("- Ý nghĩa thống kê ≠ độ lớn hiệu ứng: một ô chỉ được gọi là **thắng** khi p_holm < α, hiệu trung bình > min_effect và tỷ lệ ảnh "
             f"cải thiện ≥ {st.get('win_frac', 0.5)}.")
    L.append("- Ảnh là tiền tố split pilot đã dùng ở P1 (exploratory, không phải held-out); cùng ảnh cho mọi phương pháp; không chọn ảnh theo kết quả.")
    L.append("- S2–S6 ở cấp patch là operator khác S0: hiệu với S0 gồm cả phần đổi operator (đo riêng ở mục 5 và bằng S6 − S0); giảm phương sai "
             "chỉ được khẳng định so với đối chứng CÙNG operator.")
    L.append("- InsDel phi tuyến theo A: giảm phương sai không kéo theo tăng InsDel; InsDel của trung bình K đường thay đổi theo K ngay cả khi "
             "không có chệch.")
    L.append("- Reference có sai số hữu hạn (ref_rel_se ở mục 5); RMSE đã trừ tr Var(μ̂_ref) nhưng vẫn là ước lượng.")
    L.append(f"- Các phép so phụ (vs S6, contrast, localization, η/lưới) không hiệu chỉnh đa so sánh — đọc như exploratory.\n")

    # ── 12. thua
    L.append("## 12. Cases where structured sampling loses (D4)\n")
    for m in ("S2", "S4", "S1", "S3"):
        rows = []
        for b in bases:
            d = paired_diff(pi, f"insdel_{b}", m, "S0", Kmax)
            for im, v in d[d < 0].items():
                r_ = {"method": m, "image": im, "base": b, "Δinsdel": v}
                if ref is not None and len(ref) and f"insdel_{b}" in ref:
                    rr = ref[(ref.kind == "reference") & (ref.image == im)].set_index("operator")[f"insdel_{b}"]
                    pop = [o for o in rr.index if o.startswith("patch")]
                    if "spi" in rr and pop and m in ("S2", "S4", "S3"):
                        r_["Δoperator (μ_patch − μ_spi)"] = rr[pop[0]] - rr["spi"]
                q = pi[(pi.image == im) & (pi.K == Kmax)].set_index("method")
                for c in ("spearman_ref", "abs_resid_rel", "stab_spearman"):
                    if c in q and m in q.index:
                        r_[c] = q.loc[m, c]
                if cvg is not None and len(cvg):
                    cc = cvg[(cvg.image == im) & (cvg.method == m) & (cvg.K == Kmax)]
                    if len(cc):
                        r_["deficit_third"] = cc.deficit_third.mean()
                        r_["nbr4_order_balance"] = cc.nbr4_order_balance.mean()
                h = heat.get(im, {})
                if m in h and "S0" in h:
                    from .sampling_benchmark import _rank_corr
                    r_["spearman_map_vs_S0"] = _rank_corr(h[m]["amap"].float(), h["S0"]["amap"].float())
                rows.append(r_)
        if rows:
            L.append(f"**{m}** thua S0 ở K = {Kmax}:\n")
            L.append(md(pd.DataFrame(rows).sort_values("Δinsdel")))
    L.append("`Δoperator` = InsDel(μ_patch) − InsDel(μ_spi) của hai reference trên chính ảnh đó: nếu ≈ Δinsdel thì thua do operator, không do "
             "lấy mẫu hữu hạn. Đường cong insertion/deletion (prob và logit) của ảnh thắng / thua nhất: `sampling_insdel_curves.png`; bản đồ: "
             "`sampling_heatmaps.png`. Không suy diễn cấu trúc tương tác từ các bảng này.\n")

    # ── 13. phán quyết
    L.append("## 13. Final research verdict\n")
    L.append(f"Luật: ô (phương pháp, K, giao thức mask) **thắng** nếu p_holm < {st.get('alpha', 0.05)}, hiệu > {st.get('min_effect', 0.01)} và "
             f"≥ {st.get('win_frac', 0.5)} ảnh cải thiện; **Positive** nếu có phương pháp thắng ở ≥ ⌈|K|/2⌉ giá trị K trên ít nhất một giao "
             "thức và không thua có ý nghĩa ở đâu; **Conditional** nếu có ô thắng nhưng chưa đủ; **Negative** nếu không ô nào thắng, đủ ảnh, và "
             "≥ 75% ô có CI trên < min_effect; còn lại **Inconclusive**.\n")
    L.append(f"- Ô thắng theo phương pháp: {V['wins']}")
    L.append(f"- Ô thua có ý nghĩa: {V['losses']}")
    L.append(f"- Tỷ lệ ô chính có CI trên < min_effect: {fmt(V.get('frac_cells_ci_below_min_effect'))}\n")

    def q_var(a, b):
        if not len(ag):
            return "không có dữ liệu"
        q = ag[(ag.a == a) & (ag.b == b) & (ag.metric == "var_elem") & (ag.type == "log_ratio")].sort_values("K")
        if not len(q):
            return "không có dữ liệu"
        parts = []
        for _, r in q.iterrows():
            tag = "giảm" if r.ratio_hi95 < 1 else ("TĂNG" if r.ratio_lo95 > 1 else "không rõ")
            parts.append(f"K={int(r.K)}: ×{fmt(r.ratio_gmean)} [{fmt(r.ratio_lo95)}, {fmt(r.ratio_hi95)}] {tag}")
        return "; ".join(parts)

    rho = ""
    if cov is not None and len(cov):
        g = cov[cov.block == "primary"].groupby("family").rho_elem.mean()
        rho = ", ".join(f"{k} ρ = {fmt(v)}" for k, v in g.items())
    cat = V["category"]
    win_any = lambda m: {b: V["wins"].get(m, {}).get(b, []) for b in ("insdel_blur", "insdel_black")}
    L.append(f"**Q1. Independent SPI có bị structured sampling đánh bại dưới cùng ngân sách gradient không?** — {cat}. "
             f"Ô thắng S1–S4: {V['wins']}. (Mọi so sánh ở cùng K ⇔ cùng B.)\n")
    L.append(f"**Q2. Antithetic có giảm phương sai attribution trên CNN thật?** — S1/S0: {q_var('S1', 'S0')}. S3/S6: {q_var('S3', 'S6')}. "
             f"Tương quan cặp: {rho}. InsDel S1 vs S0: thắng ở {win_any('S1')}.\n")
    L.append(f"**Q3. Global- / neighborhood-balanced có cải thiện InsDel? Ổn định giữa ảnh?** — S2: {win_any('S2')}, S4: {win_any('S4')} "
             "(xem cột `frac_improved` mục 6). Phương sai cùng operator: S2/S6: " + q_var("S2", "S6") + "; S4/S6: " + q_var("S4", "S6") + ".\n")
    s42 = ""
    if len(ag):
        q = ag[(ag.a == "S4") & (ag.b == "S2") & (ag.type == "diff") & ag.metric.isin([f"insdel_{b}" for b in bases])]
        s42 = "; ".join(f"{r.metric} K={int(r.K)}: {fmt(r['mean'])} [{fmt(r.lo95)}, {fmt(r.hi95)}]" for _, r in q.sort_values(["metric", "K"]).iterrows())
    L.append(f"**Q4. Prior locality có thêm lợi ích so với ngẫu nhiên toàn cục?** — S4 − S2 (InsDel, CI 95%, không hiệu chỉnh): {s42 or 'không có dữ liệu'}. "
             f"Phương sai S4/S2: {q_var('S4', 'S2')}.\n")
    rec = (f"Có — {V['positive_methods']} đạt tiêu chí Positive trên {n_img} ảnh pilot; vẫn cần xác nhận trên ảnh held-out trước khi đổi mặc định."
           if cat == "Positive" else "Không — chưa đủ bằng chứng để thay independent SPI làm sampler mặc định.")
    L.append(f"**Q5. Có phương pháp nào đủ bằng chứng để thay independent SPI làm mặc định?** — {rec}\n")
    L.append(f"**Kết luận: {cat}.**\n")

    L.append("## Tệp\n")
    for f in sorted(os.listdir(out)):
        if f.endswith(".csv") or f.endswith(".png") or f in ("meta.json", "tests_report.txt", "sampling_heatmaps.pt"):
            L.append(f"- `sampling/{f}`")
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[sampling report] {rp}; {len(files)} hình")
    return rp