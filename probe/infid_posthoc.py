"""
`infid-posthoc` — kiểm toán thống kê SAU kiểm định (không GPU, chỉ đọc CSV của sampling-infid). Không đổi protocol, không chọn lại
cấu hình. Đầu ra: <out>/pilot/infid/INFID_POSTHOC_REPORT.md + <out>/pilot/infid/posthoc/*.csv.

  A. H3b (so với IG): kiểm số ảnh có L̂(μ_IG) = L(Ā) − σ̂²/r ≤ 0 và nguyên nhân; KHÔNG kẹp rồi lấy log. Chênh lệch tuyệt đối là kết quả
     chính; log-tỷ số trên L̂ đã khử phương sai bị đánh dấu "không ước lượng đáng tin" nếu có giá trị không dương. Log-tỷ số mô tả trên
     loss CHƯA khử phương sai (L(Ē), L(Ā)) được báo riêng như một estimand khác.
  B. E2 theo cặp ảnh, 8 D: d_{i,D} = L_{D,i}(μ_chọn) − L_{D,i}(μ_baseline); bootstrap theo ẢNH, giữ nguyên mọi D trong mỗi lần (các D
     tương quan ⇒ không phải 8 lần tái lập độc lập); TB, CI, trung vị, số ảnh thắng, leave-one-out; P(cả 8 D có TB < 0).
  C. Metric ngoài theo cặp ảnh, đúng chiều từng metric (tốt hơn = lớn hơn, trừ road_morf); "không có ý nghĩa" ⇒ "chưa ghi nhận cải
     thiện", không phải "tương đương".
  D. J_α(K) = L_D(μ_α) + 2·V_e,α / K (ước lượng `rev`) tại K = 2…64, riêng dev và test (và E2 mô tả): chênh lệch theo cặp giữa α, xác
     suất bootstrap mỗi α thắng, điểm giao cắt K* = 2(V_e,a − V_e,b)/(L_b − L_a) và phân phối bootstrap của nó.
  E. (POST HOC) độ nhạy với ảnh có C_D nhỏ (ΔF bão hoà): tỷ số gộp ΣE_D/ΣC_D theo cấu hình, và so sánh theo cặp sau khi bỏ phần tư ảnh
     có C_D nhỏ nhất. Không phải giả thuyết xác nhận.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from . import config as C
from .infid_e0 import icfg
from .sampling_infid import STAGE_DEFAULT, cname, load_selection, pcfg, sroot

KS = (2, 4, 8, 16, 32, 64)
# chiều tốt: +1 ⇒ lớn hơn là tốt; −1 ⇒ nhỏ hơn là tốt
EXT_DIR = {"insdel_blur": +1, "road": +1, "road_morf": -1, "road_lerf": +1, "iou": +1, "pointing": +1, "auroc_mask": +1, "ap_mask": +1}


def _read(stage: str, name: str):
    from .report import _read as rd
    return rd(os.path.join(SROOT, stage), name)


def boot_idx(n: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (int(B), n))


def ci_of(bs: np.ndarray, al: float) -> tuple:
    bs = bs[np.isfinite(bs)]
    if not len(bs):
        return math.nan, math.nan
    return float(np.quantile(bs, al / 2)), float(np.quantile(bs, 1 - al / 2))


def fmt_ci(m, lo, hi, f="{:.4g}") -> str:
    return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]" if np.isfinite(lo) else (f.format(m) if np.isfinite(m) else "—")


def loo(v: np.ndarray) -> np.ndarray:
    n = len(v)
    return (v.sum() - v) / (n - 1) if n > 1 else np.array([math.nan])


def wide(p, set_name: str, dset: str, col: str = "L_mu"):
    """bảng ảnh × config của cột `col` (set, group all, dset)."""
    q = p[(p.set == set_name) & (p.group == "all") & (p.dset == dset)]
    return q.pivot_table(index="image", columns="config", values=col, aggfunc="first").sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# A. H3b
# ─────────────────────────────────────────────────────────────────────────────

def sec_A(cfg, chosen: str, base: str, B: int, seed: int, al: float) -> list:
    import pandas as pd
    from .report import md
    D = icfg(cfg)["D"]["primary"]
    p, ig = _read("e1b-test", "pairs"), _read("e1b-test", "ig")
    L = ["## A. H3b — so với IG (E1b test): giá trị không dương của L̂(μ_IG)\n"]
    if p is None or ig is None:
        return L + ["_(chưa có dữ liệu E1b test)_\n"]
    g = ig[(ig.set == "eval") & (ig.group == "all") & (ig.dset == D)].set_index("image").sort_index()
    q = p[(p.set == "eval") & (p.group == "all") & (p.dset == D) & (p.config == chosen)].set_index("image").reindex(g.index)
    r = g.r.iloc[0] if "r" in g else math.nan
    g = g.assign(L_Abar=g.L_mu + g.sigma2 / g.r, var_term=g.sigma2 / g.r)
    neg = g[g.L_mu <= 0]
    L.append(f"r = {r:g} lần rút U trên cùng đường thẳng; L̂(μ_IG) = L(Ā) − σ̂²/r. Không dương: **{len(neg)}/{len(g)} ảnh**.\n")
    if len(neg):
        L.append("Nguyên nhân: σ̂²/r (phương sai quadrature của IG ở N = 64 chia r) lớn hơn L(Ā) — ước lượng không chệch nhưng có thể âm "
                 "trong mẫu hữu hạn, trong khi loss thật ≥ 0:\n")
        L.append(md(neg.reset_index()[["image", "L_Abar", "var_term", "L_mu", "sigma2", "L_one"]]))
    # QUY ƯỚC DUY NHẤT cho mọi dòng: hiệu = IG − chọn; > 0 ⇔ cấu hình chọn TỐT HƠN IG (infid thấp hơn). Cùng chiều với cột
    # "giảm L_D(μ)" của bảng H3b trong INFID_SAMPLING_REPORT.md (= L_IG − L_chọn).
    d = (g.L_mu - q.L_mu).values
    idx = boot_idx(len(d), B, seed)
    bs = d[idx].mean(1)
    lo, hi = ci_of(bs, al)
    rows = [{"đại lượng": f"L̂(μ_IG) − L̂(μ_{chosen}) (KẾT QUẢ CHÍNH của H3b; > 0 ⇔ {chosen} tốt hơn)", "giá trị": fmt_ci(float(d.mean()), lo, hi),
             "ảnh chọn tốt hơn (hiệu > 0)": f"{int((d > 0).sum())}/{len(d)}"}]
    if len(neg) or (q.L_mu <= 0).any():
        rows.append({"đại lượng": "log-tỷ số trên L̂ đã khử phương sai", "giá trị": "KHÔNG ước lượng đáng tin (có L̂ ≤ 0; không kẹp)",
                     "ảnh chọn tốt hơn (hiệu > 0)": "—"})
    else:
        lr = np.log(g.L_mu.values / q.L_mu.values)
        lo2, hi2 = ci_of(lr[idx].mean(1), al)
        rows.append({"đại lượng": "log(L̂_IG / L̂_chọn) (> 0 ⇔ chọn tốt hơn)", "giá trị": fmt_ci(float(lr.mean()), lo2, hi2),
                     "ảnh chọn tốt hơn (hiệu > 0)": f"{int((lr > 0).sum())}/{len(lr)}"})
    lr3 = np.log(g.L_Abar.values / q.L_Ebar.values)
    lo3, hi3 = ci_of(lr3[idx].mean(1), al)
    rows.append({"đại lượng": f"log(L(Ā_IG, r={r:g}) / L(Ē_chọn, {int(q.m.iloc[0]) * 2} đường)) — ESTIMAND KHÁC (gồm phương sai hữu hạn; "
                               "> 0 ⇔ chọn tốt hơn)",
                 "giá trị": fmt_ci(float(lr3.mean()), lo3, hi3), "ảnh chọn tốt hơn (hiệu > 0)": f"{int((lr3 > 0).sum())}/{len(lr3)}"})
    L.append(md(pd.DataFrame(rows)))
    # kiểm chéo: cùng hiệu với bảng H3b của báo cáo chính (tính lại độc lập từ cùng CSV)
    chk = float(np.nanmean(g.L_mu.values - q.L_mu.values))
    L.append(f"Kiểm chéo dấu: TB(L̂_IG − L̂_chọn) = {chk:+.4g} (phải trùng dấu và giá trị với cột 'giảm L_D(μ)' của dòng IG trong bảng H3b "
             f"của INFID_SAMPLING_REPORT.md). Ảnh chọn kém hơn IG: {', '.join(g.index[d < 0]) or 'không có'}.\n")
    L.append("H3 chính (so với SPI baseline) là phép kiểm riêng, không phụ thuộc log-tỷ số của IG.\n")
    return L


# ─────────────────────────────────────────────────────────────────────────────
# B. E2 theo cặp, 8 D, bootstrap chung
# ─────────────────────────────────────────────────────────────────────────────

def sec_B(cfg, chosen: str, base: str, B: int, seed: int, al: float, out: str) -> list:
    import pandas as pd
    from .report import md
    p = _read("e2", "pairs")
    L = ["## B. E2 — so sánh theo cặp ảnh trên từng D (bootstrap chung theo ảnh)\n"]
    if p is None:
        return L + ["_(chưa có dữ liệu E2)_\n"]
    dsets = list(dict.fromkeys(p.dset))
    cols = {}
    for dn in dsets:
        w = wide(p, "eval", dn)
        if chosen in w and base in w:
            cols[dn] = (w[chosen] - w[base])
    M = pd.DataFrame(cols).dropna()                                                  # ảnh × D, cùng tập ảnh mọi D
    if not len(M):
        return L + ["_(thiếu cấu hình)_\n"]
    X = M.values
    idx = boot_idx(len(X), B, seed)
    bs = X[idx].mean(1)                                                              # (B, nD) — giữ nguyên các D trong mỗi lần
    rows = []
    for j, dn in enumerate(M.columns):
        v = X[:, j]
        lo, hi = ci_of(bs[:, j], al)
        lv = loo(v)
        imax = int(np.argmax(np.abs(v)))
        rows.append({"D": dn, "n ảnh": len(v), "TB d [CI]": fmt_ci(float(v.mean()), lo, hi), "trung vị d": float(np.median(v)),
                     "ảnh cải thiện (d<0)": f"{int((v < 0).sum())}/{len(v)}", "LOO [min, max]": f"[{lv.min():.3g}, {lv.max():.3g}]",
                     "dấu ổn định khi bỏ 1 ảnh": bool((lv < 0).all() or (lv > 0).all()), "CI < 0": bool(hi < 0),
                     "ảnh |d| lớn nhất": f"{M.index[imax]} ({v[imax]:+.3g})"})
    allneg = float((bs < 0).all(1).mean())
    n_ci = int(sum(r["CI < 0"] for r in rows))
    sim = simultaneous(X, bs, al)
    for r_, sm in zip(rows, sim):
        r_.update(sm)
    L.append(f"d_(i,D) = L_D,i(μ_{chosen}) − L_D,i(μ_{base}); {len(M)} ảnh có đủ {len(M.columns)} D. Mỗi lần bootstrap rút lại ẢNH và giữ "
             "nguyên cả hàng D ⇒ CI đồng thời phản ánh tương quan giữa các D; KHÔNG xem các D là các lần tái lập độc lập.\n")
    L.append(md(pd.DataFrame(rows)))
    nD = len(M.columns)
    n_sim = int(sum(r["CI đồng thời < 0"] for r in rows))
    n_holm = int(sum(r["Holm: có ý nghĩa"] for r in rows))
    L.append(f"- **Từng D riêng lẻ (CI 95% chưa điều chỉnh):** {n_ci}/{nD} D có CI nằm hoàn toàn dưới 0"
             + (" — MỌI D đều có bằng chứng riêng lẻ." if n_ci == nD else " — không phải mọi D đều có bằng chứng riêng lẻ.") + "")
    L.append(f"- **Đồng thời (mức gia đình 95%):** CI đồng thời kiểu max-|t| bootstrap: {n_sim}/{nD} D nằm dưới 0; Holm trên p bootstrap hai "
             f"phía: {n_holm}/{nD} D có ý nghĩa. Đây là bảo đảm cho các khẳng định TỪNG D trong cả gia đình {nD} D.")
    L.append(f"- **Khẳng định \"cải thiện trên MỌI D\"** là phép thử giao–hợp (intersection–union): bác bỏ H0 \"có ít nhất một D không cải thiện\" "
             f"ở mức α khi TỪNG D bị bác bỏ ở mức α (một phía) — không cần hiệu chỉnh đa kiểm định. Theo CI riêng lẻ: "
             f"{'ĐẠT' if n_ci == nD else 'KHÔNG đạt'}. P_bootstrap(TB d < 0 trên cả {nD} D cùng lúc) = {allneg:.3f} (mô tả).")
    L.append("- E2 là phân tích MÔ TẢ (ngoài mẫu, không có giả thuyết đăng ký trước cho từng D).\n")
    M.to_csv(os.path.join(out, "B_e2_paired_d.csv"))
    return L


def boot_p(bs: np.ndarray) -> float:
    """p bootstrap hai phía cho H0: TB = 0 (đảo dấu phân phối bootstrap): 2·min(P(bs ≤ 0), P(bs ≥ 0)), hiệu chỉnh +1."""
    bs = bs[np.isfinite(bs)]
    n = len(bs)
    if not n:
        return math.nan
    return float(min(1.0, 2 * min((1 + (bs <= 0).sum()) / (n + 1), (1 + (bs >= 0).sum()) / (n + 1))))


def holm(ps) -> list:
    """p hiệu chỉnh Holm (step-down, đơn điệu)."""
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj.tolist()


def simultaneous(X: np.ndarray, bs: np.ndarray, al: float) -> list:
    """CI đồng thời max-|t| (bootstrap): t*_b = max_j |bs_bj − x̄_j| / sd_j; c = q_{1−α}(t*); CI_j = x̄_j ± c·sd_j. Kèm Holm trên p bootstrap."""
    xbar = X.mean(0)
    sd = bs.std(0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)
    tmax = np.nanmax(np.abs(bs - xbar) / sd, axis=1)
    c = float(np.nanquantile(tmax, 1 - al))
    ps = [boot_p(bs[:, j]) for j in range(X.shape[1])]
    ph = holm(ps)
    out = []
    for j in range(X.shape[1]):
        lo, hi = xbar[j] - c * sd[j], xbar[j] + c * sd[j]
        out.append({"CI đồng thời (max-|t|)": fmt_ci(float(xbar[j]), float(lo), float(hi)), "CI đồng thời < 0": bool(hi < 0),
                    "p bootstrap": ps[j], "p Holm": ph[j], "Holm: có ý nghĩa": bool(ph[j] < al)})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# C. Metric ngoài theo cặp
# ─────────────────────────────────────────────────────────────────────────────

def sec_C(cfg, chosen: str, base: str, B: int, seed: int, al: float, out: str) -> list:
    import pandas as pd
    from .report import md
    ex = _read("e2", "ext")
    L = ["## C. Metric ngoài — chênh lệch theo cặp ảnh (E2, ước lượng thực tế K chính)\n"]
    if ex is None:
        return L + ["_(chưa có metric ngoài)_\n"]
    W = {c: g.set_index("image").sort_index() for c, g in ex.groupby("config")}
    comps = [(chosen, base)] + [(chosen, c) for c in W if c.startswith("ig_")] + [(base, c) for c in W if c.startswith("ig_")]
    rows = []
    for a, b in comps:
        if a not in W or b not in W:
            continue
        for met, sgn in EXT_DIR.items():
            if met not in W[a] or met not in W[b]:
                continue
            j = pd.concat([W[a][met].rename("a"), W[b][met].rename("b")], axis=1).dropna()
            if len(j) < 2:
                continue
            v = sgn * (j.a - j.b).values                                             # > 0 ⇒ a tốt hơn b theo đúng chiều của metric
            bsv = v[boot_idx(len(v), B, seed)].mean(1)
            lo, hi = ci_of(bsv, al)
            verdict = ("a tốt hơn" if lo > 0 else "a kém hơn" if hi < 0 else "chưa ghi nhận khác biệt")
            rows.append({"a": a, "b": b, "metric": met, "chiều tốt": "lớn" if sgn > 0 else "nhỏ", "n ảnh": len(v),
                         "TB a": float(j.a.mean()), "TB b": float(j.b.mean()), "(a − b) theo chiều tốt [CI chưa điều chỉnh]": fmt_ci(float(v.mean()), lo, hi),
                         "a thắng": f"{int((v > 0).sum())}/{len(v)}", "kết luận (chưa điều chỉnh)": verdict, "p bootstrap": boot_p(bsv),
                         "_sign": float(np.sign(v.mean()))})
    df = pd.DataFrame(rows)
    if len(df):
        adj = []
        for _, g in df.groupby(["a", "b"], sort=False):                                # gia đình = các metric của MỘT cặp so sánh
            adj += list(zip(g.index, holm(g["p bootstrap"].values)))
        df["p Holm (trong cặp)"] = df.index.map(dict(adj))
        df["kết luận (Holm)"] = [("a tốt hơn" if sg > 0 else "a kém hơn") if ph < al else "chưa ghi nhận khác biệt"
                                 for sg, ph in zip(df["_sign"], df["p Holm (trong cặp)"])]
        df = df.drop(columns="_sign")
    df.to_csv(os.path.join(out, "C_ext_paired.csv"), index=False)
    L.append("Dương = a tốt hơn b theo chiều tốt của metric (road_morf: nhỏ hơn là tốt). Báo cả CI 95% CHƯA điều chỉnh và p hiệu chỉnh Holm "
             "trong gia đình các metric của MỖI cặp so sánh (p bootstrap hai phía). \"Chưa ghi nhận khác biệt\" — KHÔNG phải \"tương đương\". "
             "Metric đo trên attribution hữu hạn K chính (rev K = 4 / IG 4 lần rút): chưa thể quy khác biệt cho attribution KỲ VỌNG — phương "
             "sai lấy mẫu (lớn hơn ở α nhỏ) có thể góp phần; tách hai phần cần tính lại trên Ē nhiều đường (GPU).\n")
    L.append(md(df))
    return L


# ─────────────────────────────────────────────────────────────────────────────
# D. J(K) và điểm giao cắt
# ─────────────────────────────────────────────────────────────────────────────

def jk_block(p, set_name: str, dset: str, configs: list, ref_best: str, B: int, seed: int, al: float, label: str) -> tuple:
    Lw, Vw = wide(p, set_name, dset, "L_mu"), wide(p, set_name, dset, "Ve")
    cfgs = [c for c in configs if c in Lw and c in Vw]
    Lw, Vw = Lw[cfgs].dropna(), Vw[cfgs].dropna()
    imgs = Lw.index.intersection(Vw.index)
    Lm, Vm = Lw.loc[imgs].values, Vw.loc[imgs].values                                 # ảnh × config
    n = len(imgs)
    if n < 2 or len(cfgs) < 2:
        return [], [], []
    idx = boot_idx(n, B, seed)
    Lb, Vb = Lm[idx].mean(1), Vm[idx].mean(1)                                         # (B, nc)
    rows, wins, kst = [], [], []
    for K in KS:
        J = Lm + 2 * Vm / K
        Jb = Lb + 2 * Vb / K
        win = np.bincount(Jb.argmin(1), minlength=len(cfgs)) / len(Jb)
        wins.append({"tập": label, "K": K, **{f"P(thắng) {c}": float(win[i]) for i, c in enumerate(cfgs)},
                     **{f"J {c}": float(J[:, i].mean()) for i, c in enumerate(cfgs)}})
        ib = cfgs.index(ref_best) if ref_best in cfgs else 0
        for i, c in enumerate(cfgs):
            if i == ib:
                continue
            dv = J[:, i] - J[:, ib]
            lo, hi = ci_of((Jb[:, i] - Jb[:, ib]), al)
            rows.append({"tập": label, "K": K, "so sánh": f"{c} − {cfgs[ib]}", "TB ΔJ [CI]": fmt_ci(float(dv.mean()), lo, hi),
                         f"ảnh {cfgs[ib]} thắng": f"{int((dv > 0).sum())}/{n}"})
    ib = cfgs.index(ref_best) if ref_best in cfgs else 0
    for i, c in enumerate(cfgs):
        if i == ib:
            continue
        # J_a(K) − J_b(K) = (L_a − L_b) + 2(V_e,a − V_e,b)/K = −ΔL + ΔV/K, ΔL = L_b − L_a, ΔV = 2(V_e,a − V_e,b).
        # Cắt nhau tại K* = ΔV/ΔL khi ΔV/ΔL > 0.
        num = 2 * (Vb[:, ib] - Vb[:, i])
        den = Lb[:, i] - Lb[:, ib]
        with np.errstate(divide="ignore", invalid="ignore"):
            ks = num / den
        ok = np.isfinite(ks) & (ks > 0)
        n0 = 2 * (Vm[:, ib].mean() - Vm[:, i].mean())
        d0 = Lm[:, i].mean() - Lm[:, ib].mean()
        k0 = n0 / d0 if d0 != 0 and n0 / d0 > 0 else math.nan
        if d0 > 0 and n0 > 0:
            read = f"{cfgs[ib]} thắng khi K > K*, {c} thắng khi K < K*"
        elif d0 > 0 and n0 <= 0:
            read = f"{cfgs[ib]} tốt hơn ở MỌI K"
        elif d0 < 0 and n0 < 0:
            read = f"{c} thắng khi K > K*, {cfgs[ib]} thắng khi K < K*"
        else:
            read = f"{c} tốt hơn (hoặc bằng) ở MỌI K"
        kst.append({"tập": label, "a": cfgs[ib], "b": c, "ΔL = L_b − L_a": float(d0), "ΔV = 2(V_e,a − V_e,b)": float(n0),
                    "K* điểm": k0, "K* bootstrap [q2,5; q50; q97,5] (chỉ khi cắt)":
                    (f"[{np.quantile(ks[ok], 0.025):.3g}; {np.quantile(ks[ok], 0.5):.3g}; {np.quantile(ks[ok], 0.975):.3g}]" if ok.any() else "—"),
                    "P_bootstrap(hai đường cắt nhau)": float(ok.mean()), "đọc (theo điểm ước lượng)": read})
    return rows, wins, kst


def sec_D(cfg, chosen: str, base: str, B: int, seed: int, al: float, out: str) -> list:
    import pandas as pd
    from .report import md
    D = icfg(cfg)["D"]["primary"]
    L = ["## D. J_α(K) = L_D(μ_α) + 2·V_e,α/K (ước lượng `rev`) — ngân sách hữu hạn\n",
         "Dev và test tính RIÊNG. Test chỉ có hai cấu hình đã khoá (chọn, baseline) ⇒ đối đầu α nhỏ khác (vd. 0,3) với α = 0,1 ở cùng ngân sách "
         "CHƯA được kiểm định trên ảnh mới. K* = 2(V_e,a − V_e,b)/(L_b − L_a) có thể rất bất ổn khi hai attribution kỳ vọng gần nhau.\n"]
    allr, allw, allk = [], [], []
    for stage, label in (("e1b-dev", "dev"), ("e1b-test", "test")):
        p = _read(stage, "pairs")
        if p is None:
            continue
        cf = list(dict.fromkeys(p[(p.set == "eval") & (p.dset == D)].config))
        r, w, k = jk_block(p, "eval", D, cf, chosen, B, seed, al, label)
        allr += r
        allw += w
        allk += k
    p = _read("e2", "pairs")
    if p is not None:
        for dn in list(dict.fromkeys(p.dset)):
            r, w, k = jk_block(p, "eval", dn, [chosen, base], chosen, B, seed, al, f"E2 {dn} (mô tả)")
            allw += w
            allk += k
    if allw:
        L.append(f"**Xác suất bootstrap mỗi cấu hình có J nhỏ nhất và J trung bình** (D chính `{D}` cho dev/test):\n")
        L.append(md(pd.DataFrame(allw)))
    if allr:
        L.append(f"**Chênh lệch theo cặp ảnh của J so với `{chosen}`** (dương ⇒ `{chosen}` tốt hơn):\n")
        L.append(md(pd.DataFrame(allr)))
    if allk:
        L.append("**Điểm giao cắt K\\***:\n")
        L.append(md(pd.DataFrame(allk)))
    pd.DataFrame(allw).to_csv(os.path.join(out, "D_JK_wins.csv"), index=False)
    pd.DataFrame(allk).to_csv(os.path.join(out, "D_Kstar.csv"), index=False)
    return L


# ─────────────────────────────────────────────────────────────────────────────
# E. POST HOC: ảnh có C_D nhỏ
# ─────────────────────────────────────────────────────────────────────────────

def sec_E(cfg, chosen: str, base: str, B: int, seed: int, al: float) -> list:
    import pandas as pd
    from .report import md
    L = ["## E. (POST HOC) Độ nhạy với ảnh có C_D nhỏ (ΔF bão hoà)\n",
         "**Phân tích sau khi đã xem kết quả test — KHÔNG phải giả thuyết xác nhận.** Estimand chính (TB theo ảnh của L chuẩn hoá từng ảnh) "
         "đặt trọng số lớn vào ảnh có C_D = E[ΔF_S²] rất nhỏ. Hai độ nhạy: (1) tỷ số gộp ΣE_D / ΣC_D (≡ TB L có trọng số C_D); "
         "(2) so sánh theo cặp sau khi bỏ phần tư ảnh có C_D nhỏ nhất (C_D giống nhau giữa cấu hình vì cùng ô).\n"]
    rows = []
    for stage, label in (("e1b-test", "E1b test"), ("e2", "E2")):
        p = _read(stage, "pairs")
        if p is None or "C_D" not in p:
            continue
        for dn in list(dict.fromkeys(p.dset)):
            Lw, Ew, Cw = wide(p, "eval", dn, "L_mu"), wide(p, "eval", dn, "E_D"), wide(p, "eval", dn, "C_D")
            if chosen not in Lw or base not in Lw:
                continue
            j = pd.DataFrame({"La": Lw[chosen], "Lb": Lw[base], "Ea": Ew[chosen], "Eb": Ew[base], "C": Cw[base]}).dropna()
            n = len(j)
            if n < 4:
                continue
            idx = boot_idx(n, B, seed)
            Ea, Eb, Cc = j.Ea.values, j.Eb.values, j.C.values
            pa, pb = Ea.sum() / Cc.sum(), Eb.sum() / Cc.sum()
            pbs = (Ea[idx].sum(1) - Eb[idx].sum(1)) / Cc[idx].sum(1)
            lo, hi = ci_of(pbs, al)
            q25 = np.quantile(Cc, 0.25)
            keep = Cc > q25
            dv = (j.La - j.Lb).values[keep]
            lo2, hi2 = ci_of(dv[boot_idx(len(dv), B, seed)].mean(1), al) if len(dv) >= 2 else (math.nan, math.nan)
            dall = (j.La - j.Lb).values
            lo3, hi3 = ci_of(dall[idx].mean(1), al)
            rows.append({"tập": label, "D": dn, "n": n, "TB d (tất cả) [CI]": fmt_ci(float(dall.mean()), lo3, hi3),
                         f"gộp ΣE/ΣC: {chosen}": float(pa), f"gộp ΣE/ΣC: {base}": float(pb), "Δ gộp [CI]": fmt_ci(float(pa - pb), lo, hi),
                         "TB d bỏ 25% C_D nhỏ [CI]": fmt_ci(float(dv.mean()), lo2, hi2) if len(dv) else "—", "n giữ": int(keep.sum()),
                         "cùng dấu với kết quả chính": bool(np.sign(pa - pb) == np.sign(dall.mean()) and (len(dv) == 0 or np.sign(dv.mean()) == np.sign(dall.mean())))})
    if rows:
        L.append(md(pd.DataFrame(rows)))
    else:
        L.append("_(thiếu cột E_D / C_D — cần CSV từ phiên bản ước lượng 2)_\n")
    return L


# ─────────────────────────────────────────────────────────────────────────────

SROOT = ""


def run(cfg: dict) -> str:
    global SROOT
    SROOT = sroot(cfg)
    pr = pcfg(cfg)
    B, seed, al = int(pr["bootstrap"]), int(pr["seed"]), float(pr["alpha"])
    out = os.path.join(C.out_dir(cfg, "pilot"), "infid", "posthoc")
    os.makedirs(out, exist_ok=True)
    b = C._merge(STAGE_DEFAULT["e1b"], icfg(cfg).get("e1b", {}) or {})
    base = cname(b["baseline"])
    try:
        sel = load_selection(cfg)
        chosen = sel["chosen_name"]
    except SystemExit:
        chosen = base
    L = ["# Kiểm toán thống kê sau kiểm định (post hoc, không GPU)\n",
         f"Cấu hình đã chọn (khoá): `{chosen}`; baseline: `{base}`. Bootstrap theo ảnh, B = {B}, α = {al}. Không đổi protocol, không chọn lại "
         "cấu hình; mọi phân tích ở đây là kiểm toán / mô tả bổ sung cho H1, H3 đã kiểm định.\n"]
    for f in (lambda: sec_A(cfg, chosen, base, B, seed, al), lambda: sec_B(cfg, chosen, base, B, seed, al, out),
              lambda: sec_C(cfg, chosen, base, B, seed, al, out), lambda: sec_D(cfg, chosen, base, B, seed, al, out),
              lambda: sec_E(cfg, chosen, base, B, seed, al)):
        L += f()
    rp = os.path.join(C.out_dir(cfg, "pilot"), "infid", "INFID_POSTHOC_REPORT.md")
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"chosen": chosen, "base": base, "B": B, "seed": seed, "alpha": al}, fh, indent=1)
    print(f"[infid-posthoc] {rp}")
    return rp


# ─────────────────────────────────────────────────────────────────────────────
# đóng băng
# ─────────────────────────────────────────────────────────────────────────────

def freeze(cfg: dict) -> str:
    """Ghi infid/FROZEN.json: sha256 của mọi CSV / JSON / báo cáo dùng cho bản kết quả, khoá protocol, lựa chọn, config và mã nguồn.
    Phân loại: đăng ký trước (H1, H3), mô tả (E2, metric ngoài, J(K)), hậu kiểm (độ nhạy C_D). Chạy SAU infid-posthoc."""
    import datetime as _dt
    import glob
    import hashlib
    root = os.path.join(C.out_dir(cfg, "pilot"), "infid")

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            h.update(fh.read())
        return h.hexdigest()
    files = sorted(set(glob.glob(os.path.join(root, "*.md")) + glob.glob(os.path.join(root, "*.json")) +
                       glob.glob(os.path.join(root, "sampling", "*", "*.csv")) + glob.glob(os.path.join(root, "sampling", "*", "meta.json")) +
                       glob.glob(os.path.join(root, "e0", "*.csv")) + glob.glob(os.path.join(root, "quad", "*.csv")) +
                       glob.glob(os.path.join(root, "toy", "*.csv")) + glob.glob(os.path.join(root, "posthoc", "*"))))
    files = [f for f in files if os.path.isfile(f) and not f.endswith("FROZEN.json")]
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hs = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(src, "probe", "*.py"))):
        hs.update(open(f, "rb").read())
    pre = os.path.join(root, "infid_prereg.json")
    rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "config": cfg["_path"], "config_sha256": sha(cfg["_path"]),
           "source_sha256": hs.hexdigest(), "prereg_digest": json.load(open(pre))["digest"] if os.path.exists(pre) else None,
           "evidence_levels": {"đăng ký trước (xác nhận)": ["H1 (E1a)", "H3 (E1b test, so với SPI baseline)"],
                               "đối chứng riêng": ["H3b (so với IG) — chênh tuyệt đối"],
                               "mô tả": ["E2 (8 D)", "metric ngoài", "J(K) và K*", "dev / pilot"],
                               "hậu kiểm (post hoc)": ["độ nhạy C_D (INFID_POSTHOC_REPORT.md mục E)"],
                               "hoãn": ["H2 (nbr)"]},
           "files": {os.path.relpath(f, root): sha(f) for f in files}}
    p = os.path.join(root, "FROZEN.json")
    if os.path.exists(p):
        rec["previous"] = json.load(open(p))
    with open(p, "w") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print(f"[infid-freeze] {p}: {len(files)} file")
    return p
