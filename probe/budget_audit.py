"""
budget-audit — kiểm toán vòng BUDGET 1 (chỉ đọc CSV / JSON, không GPU) → infid/INFID_BUDGET_AUDIT.md

(1) Phân phối xác suất chọn ĐẦY ĐỦ trên dev1 / dev2 (không chỉ top-3 theo TB J) + ảnh hưởng của từng ảnh (bỏ-một-ảnh, ảnh có mặt
    nhiều trong các bootstrap mà θ* thua) + kiểm tái lập P_select từ CSV.
(2) Hiệu chuẩn phép kiểm R(B) ≥ r₀: p đã đăng ký (tỷ lệ bootstrap percentile) so với các phép kiểm hiệu chuẩn khác —
    tuyến tính hoá chính xác  R > r₀ ⇔ E[D] > 0,  D_i = (b_i − t_i) − r₀ (b_i − f_i)  (khi E[b − f] > 0): t một phía,
    bootstrap-t trên D; bootstrap "basic"; CI Fieller và BCa cho R; bỏ-một-ảnh.
(3) Kiểm J thực nghiệm so với công thức: sai khác tuyệt đối, tương đối theo J và theo số hạng phương sai V_K, độ lệch (skew),
    Holm trong mỗi giai đoạn, lặp lại trên hai nửa ảnh rời nhau, gộp các dòng.
"""

from __future__ import annotations

import json
import math
import os
from statistics import NormalDist

import numpy as np

from . import budget_opt as BO
from . import config as C
from .infid_posthoc import boot_idx, boot_p, ci_of, fmt_ci, holm


def t_sf(t: float, df: int) -> float:
    """P(T_df ≥ t). scipy nếu có, ngược lại xấp xỉ chuẩn (df ≥ 30 sai khác < 0,01 ở mức p ~ 0,05)."""
    try:
        from scipy.stats import t as _t
        return float(_t.sf(t, df))
    except Exception:  # noqa: BLE001
        return float(1 - NormalDist().cdf(t))


def t_ppf(q: float, df: int) -> float:
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(q, df))
    except Exception:  # noqa: BLE001
        return float(NormalDist().inv_cdf(q))


# ─────────────────────────────────────────────────────────────────────────────
# hàm thuần (unit test)
# ─────────────────────────────────────────────────────────────────────────────

def ratio_R(t, b, f) -> float:
    return float((b.mean() - t.mean()) / (b.mean() - f.mean()))


def linear_test(t, b, f, r0: float) -> dict:
    """D_i = (b_i − t_i) − r₀ (b_i − f_i); H0: E[D] ≤ 0 ⇔ R ≤ r₀ (khi E[b − f] > 0). t một phía, df = n − 1."""
    D = (b - t) - r0 * (b - f)
    n = len(D)
    se = D.std(ddof=1) / math.sqrt(n)
    T = D.mean() / se if se > 0 else math.inf
    return {"D_mean": float(D.mean()), "D_se": float(se), "t": float(T), "p_t": t_sf(T, n - 1), "n": n}


def boot_t_p(D, n_boot: int, seed: int) -> float:
    """bootstrap-t một phía cho H0: E[D] ≤ 0: p = P*((D̄* − D̄)/se* ≥ D̄/se)."""
    n = len(D)
    se = D.std(ddof=1) / math.sqrt(n)
    T = D.mean() / se
    X = D[boot_idx(n, n_boot, seed)]
    Ts = (X.mean(1) - D.mean()) / (X.std(1, ddof=1) / math.sqrt(n))
    Ts = Ts[np.isfinite(Ts)]
    return float((1 + (Ts >= T).sum()) / (len(Ts) + 1))


def fieller(t, b, f, level: float = 0.95, one_sided: bool = False) -> tuple:
    """CI Fieller cho R = E[A]/E[C], A = b − t, C = b − f (trung bình theo ảnh; df = n − 1).
    Trả (lo, hi); hi = +inf nếu khoảng không bị chặn (mẫu số không có ý nghĩa)."""
    A, Cc = b - t, b - f
    n = len(A)
    a, c = A.mean(), Cc.mean()
    vaa, vcc = A.var(ddof=1) / n, Cc.var(ddof=1) / n
    vac = float(np.cov(A, Cc, ddof=1)[0, 1]) / n
    q = t_ppf(level if one_sided else (1 + level) / 2, n - 1)
    q2 = q * q
    # (a − R c)² ≤ q² (vaa − 2R vac + R² vcc)  ⇔  α R² − 2β R + γ ≤ 0
    al = c * c - q2 * vcc
    be = a * c - q2 * vac
    ga = a * a - q2 * vaa
    disc = be * be - al * ga
    if al <= 0 or disc < 0:
        return (-math.inf, math.inf)
    r = math.sqrt(disc)
    lo, hi = (be - r) / al, (be + r) / al
    return (lo, math.inf) if one_sided else (lo, hi)


def bca(stat, X: np.ndarray, n_boot: int, seed: int, al: float = 0.05) -> tuple:
    """CI BCa hai phía cho stat(X) (X: mảng theo ảnh, hàng = ảnh)."""
    n = X.shape[0]
    th = stat(X)
    idx = boot_idx(n, n_boot, seed)
    bs = np.array([stat(X[i]) for i in idx])
    bs = bs[np.isfinite(bs)]
    nd = NormalDist()
    p0 = np.clip((bs < th).mean(), 1e-6, 1 - 1e-6)
    z0 = nd.inv_cdf(float(p0))
    jk = np.array([stat(np.delete(X, i, axis=0)) for i in range(n)])
    d = jk.mean() - jk
    acc = float((d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5)) if (d ** 2).sum() > 0 else 0.0
    out = []
    for q in (al / 2, 1 - al / 2):
        z = nd.inv_cdf(q)
        a2 = nd.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
        out.append(float(np.quantile(bs, min(max(a2, 0.0), 1.0))))
    return tuple(out)


def skew(v) -> float:
    v = np.asarray(v, float)
    s = v.std()
    return float(((v - v.mean()) ** 3).mean() / s ** 3) if s > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# (1) xác suất chọn
# ─────────────────────────────────────────────────────────────────────────────

def _dev_jl(cfg: dict, stage: str):
    import pandas as pd
    pr = BO.bpr(cfg)
    p = BO._read(stage, "pairs", cfg)
    if p is None:
        return None
    if stage == "dev2":
        p1 = BO._read("dev1", "pairs", cfg)
        if p1 is not None:
            p = pd.concat([p1, p], ignore_index=True)
    return BO.j_long(p, "sel", BO.primary_D(cfg), pr["budgets"], pr["M"])


def selection_section(cfg: dict) -> list:
    import pandas as pd
    from .report import md
    pr = BO.bpr(cfg)
    nb, seed = int(pr["bootstrap"]), int(pr["seed"])
    L = ["## 1. Xác suất chọn đầy đủ trên dev (bộ ô CHỌN)\n",
         "Mọi ứng viên có P_select > 0 (bootstrap theo ảnh, cùng seed với lựa chọn). \"Tái lập\" = P tính lại từ CSV khớp JSON.\n"]
    for stage in ("dev1", "dev2"):
        try:
            sel = BO.load_sel(cfg, stage)
        except SystemExit:
            continue
        jl = _dev_jl(cfg, stage)
        for B, s in sel["final"].items():
            allowed = [x["cand"] for x in s["table"]]
            W = jl[jl.B == int(B)].pivot_table(index="image", columns="cand", values="J", aggfunc="first")[allowed].dropna()
            X = W.values
            idx = boot_idx(X.shape[0], nb, seed)
            wins = X[idx].mean(1).argmin(1)
            pw = np.bincount(wins, minlength=X.shape[1]) / len(idx)
            ok = all(abs(pw[allowed.index(x["cand"])] - x["p_select"]) < 1e-9 for x in s["table"])
            rows = [{"ứng viên": c, "TB J": float(X[:, k].mean()), "trung vị J": float(np.median(X[:, k])), "P_select": float(pw[k])}
                    for k, c in enumerate(allowed) if pw[k] > 0]
            rows.sort(key=lambda r: -r["P_select"])
            L.append(f"### {stage}, B = {B} — θ* = `{s['chosen']}` (tái lập: {'✓' if ok else '✗ — KHÁC JSON'})\n")
            L.append(md(pd.DataFrame(rows)))
            # ảnh hưởng: bootstrap θ* thua → ảnh nào bị rút nhiều hơn kỳ vọng
            k0 = allowed.index(s["chosen"])
            lose = wins != k0
            if lose.any():
                cnt = np.stack([np.bincount(r, minlength=X.shape[0]) for r in idx])
                exc = cnt[lose].mean(0) - cnt[~lose].mean(0)
                top = np.argsort(-exc)[:3]
                alt = int(np.bincount(wins[lose], minlength=X.shape[1]).argmax())
                rr = []
                for i in top:
                    rr.append({"ảnh": W.index[i], "số lần rút thêm khi θ* thua (TB)": float(exc[i]), "J θ*": float(X[i, k0]),
                               f"J {allowed[alt]}": float(X[i, alt]), "J θ* / TB J θ*": float(X[i, k0] / X[:, k0].mean())})
                L.append(f"Khi θ* thua ({lose.mean():.0%} số bootstrap), người thắng thường gặp nhất là `{allowed[alt]}`. "
                         "Ảnh bị rút nhiều hơn trong các bootstrap đó:\n")
                L.append(md(pd.DataFrame(rr)))
            # bỏ-một-ảnh
            ch = []
            for i in range(X.shape[0]):
                w = int(np.delete(X, i, 0).mean(0).argmin())
                if w != k0:
                    ch.append(f"{W.index[i]} → `{allowed[w]}`")
            L.append("Bỏ-một-ảnh: " + ("θ* không đổi với mọi ảnh bị bỏ." if not ch else "θ* đổi khi bỏ " + "; ".join(ch) + ".") + "\n")
    return L


# ─────────────────────────────────────────────────────────────────────────────
# (2) hiệu chuẩn phép kiểm R
# ─────────────────────────────────────────────────────────────────────────────

def test_arrays(cfg: dict, B: int):
    import pandas as pd
    pr = BO.bpr(cfg)
    b = BO.bcfg(cfg)
    p = BO._read("test", "pairs", cfg)
    sel = BO.load_sel(cfg, b["test"]["selection"])
    s = sel["final"].get(str(B))
    if p is None or not s:
        return None
    jl = BO.j_long(p, "eval", BO.primary_D(cfg), [B], pr["M"])
    W = jl.pivot_table(index="image", columns="cand", values="J", aggfunc="first")
    fl = BO._floor(cfg, "test", BO.primary_D(cfg))
    j = pd.concat([W[s["chosen"]].rename("t"), W[f"{BO.cname(b['test']['baseline'])}|indep"].rename("b"), fl.rename("f")], axis=1).dropna()
    return j.t.values, j.b.values, j.f.values, s["chosen"], list(j.index)


def r_section(cfg: dict) -> list:
    import pandas as pd
    from .report import md
    pr = BO.bpr(cfg)
    nb, seed, al, r0 = int(pr["bootstrap"]), int(pr["seed"]), float(pr["alpha"]), float(pr["min_excess_reduction"])
    L = ["## 2. Hiệu chuẩn phép kiểm R(B) ≥ r₀ (test, bộ ô ĐÁNH GIÁ)\n",
         f"r₀ = {r0}. **Tuyến tính hoá chính xác**: với E[b − f] > 0, R > r₀ ⇔ E[D] > 0, D_i = (b_i − t_i) − r₀ (b_i − f_i). "
         "Phép kiểm trên D là kiểm định trung bình theo cặp ảnh thông thường (không có tỷ số), nên hiệu chuẩn chuẩn mực: t một phía "
         "(df = n − 1) và bootstrap-t. p đã đăng ký = (1 + #{R* ≤ r₀}) / (B + 1) là phép đảo khoảng percentile một phía: "
         "p < 0,05 ⇔ phân vị 5 % của R* > r₀ ⇔ cận dưới CI percentile MỘT phía 95 % > r₀ — khác với cận dưới CI HAI phía 95 % "
         "(phân vị 2,5 %), nên \"p < 0,05 mà CI hai phía chứa 0,20\" không mâu thuẫn. Percentile chỉ hiệu chuẩn xấp xỉ "
         "(phụ thuộc phân phối R* đối xứng quanh R̂), vì vậy báo các phép kiểm hiệu chuẩn khác bên cạnh.\n"]
    rows, P = [], {}
    for B in pr["budgets"]:
        a = test_arrays(cfg, int(B))
        if a is None:
            continue
        t, b, f, ch, _ = a
        n = len(t)
        R = ratio_R(t, b, f)
        idx = boot_idx(n, nb, seed)
        Rb = (b[idx].mean(1) - t[idx].mean(1)) / (b[idx].mean(1) - f[idx].mean(1))
        p_pct = float((1 + (Rb <= r0).sum()) / (len(Rb) + 1))
        p_basic = float((1 + (Rb - R >= R - r0).sum()) / (len(Rb) + 1))
        lt = linear_test(t, b, f, r0)
        D = (b - t) - r0 * (b - f)
        p_bt = boot_t_p(D, nb, seed)
        D_b = D[idx].mean(1)
        p_Dpct = float((1 + (D_b <= 0).sum()) / (len(D_b) + 1))
        X = np.stack([t, b, f], 1)
        bl, bh = bca(lambda Z: ratio_R(Z[:, 0], Z[:, 1], Z[:, 2]), X, nb, seed, al)
        fl2 = fieller(t, b, f, 1 - al)
        fl1 = fieller(t, b, f, 1 - al, one_sided=True)
        loo = np.array([ratio_R(np.delete(t, i), np.delete(b, i), np.delete(f, i)) for i in range(n)])
        P[B] = {"đã đăng ký (percentile)": p_pct, "basic bootstrap": p_basic, "t trên D (tuyến tính hoá)": lt["p_t"],
                "bootstrap-t trên D": p_bt, "percentile trên D̄": p_Dpct}
        rows.append({"B": B, "θ*": ch, "n": n, "R̂": R, "CI percentile 2 phía": fmt_ci(R, *ci_of(Rb, al), "{:.3f}"),
                     "cận dưới percentile 1 phía 95 %": float(np.quantile(Rb, al)), "CI BCa 2 phía": fmt_ci(R, bl, bh, "{:.3f}"),
                     "CI Fieller 2 phía": fmt_ci(R, *fl2, "{:.3f}"), "cận dưới Fieller 1 phía": fl1[0],
                     "D̄ [SE]": f"{lt['D_mean']:.3f} [{lt['D_se']:.3f}]", "R bỏ-một-ảnh: min / max": f"{loo.min():.3f} / {loo.max():.3f}"})
    if not rows:
        return L
    L.append(md(pd.DataFrame(rows)))
    rows = []
    for meth in next(iter(P.values())):
        ps = [P[B][meth] for B in P]
        ph = holm(ps)
        for B, p_, h in zip(P, ps, ph):
            rows.append({"phương pháp": meth, "B": B, "p một phía": p_, "p Holm (2 B)": h, "xác nhận ở α = 0,05": "✓" if h < al else "✗"})
    L.append("**p một phía theo phương pháp** (Holm qua các B chính trong mỗi phương pháp):\n")
    L.append(md(pd.DataFrame(rows)))
    L.append("Đọc: quyết định đã đăng ký giữ nguyên nếu các phép kiểm hiệu chuẩn (t / bootstrap-t trên D) cùng kết luận; nếu chỉ "
             "percentile đạt mà t trên D không đạt ở một B, ghi B đó là \"xác nhận theo protocol, nhạy với phương pháp hiệu chuẩn\".\n")
    return L


# ─────────────────────────────────────────────────────────────────────────────
# (3) kiểm J thực nghiệm
# ─────────────────────────────────────────────────────────────────────────────

def emp_section(cfg: dict) -> list:
    import pandas as pd
    from .report import md
    pr = BO.bpr(cfg)
    nb, seed, al = int(pr["bootstrap"]), int(pr["seed"]), float(pr["alpha"])
    Dp = BO.primary_D(cfg)
    L = ["## 3. Kiểm J thực nghiệm so với công thức L̂ + V̂_K\n",
         "Cả hai ước lượng đều KHÔNG CHỆCH cho J(K) trên mỗi ảnh (đại số chính xác: với ng nhóm rời, mỗi nhóm h = K/2 cặp, "
         "TB_g infid(Ē_g) = L(Ē) + (1/ng)Σ_g‖Ē_g − Ē‖²/‖ΔF‖², và E[(1/ng)Σ_g‖Ē_g − Ē‖²] = (1 − 1/ng)·2V_e/K = 2V_e/K − V_e/m, "
         "đúng bằng phần bù của công thức L(Ē) − V̂_e/m + 2V̂_e/K). Hiệu theo ảnh = hiệu của hai ước lượng phương sai (ng − 1 bậc "
         "tự do so với m − 1) ⇒ phân phối lệch phải khi ng nhỏ ⇒ trung bình 32–48 ảnh với CI percentile có thể lệch. Bảng dưới "
         "tách: sai khác tuyệt đối, tương đối theo J và theo số hạng phương sai V_K (thang đo đúng của phép kiểm), skew, tỷ lệ ảnh "
         "dương, Holm trong mỗi giai đoạn, và lặp lại trên hai nửa ảnh rời nhau (ảnh chẵn / lẻ theo thứ tự).\n"]
    pooled = []
    for stage in ("dev1", "dev2", "test"):
        em, p = BO._read(stage, "emp", cfg), BO._read(stage, "pairs", cfg)
        if em is None or p is None:
            continue
        em = em[em.dset == Dp]
        Lm = p[(p.set == "eval") & (p.group == "all") & (p.dset == Dp)].set_index(["image", "config"]).L_mu
        rows = []
        for c, g in em.groupby("config"):
            g = g.set_index("image").sort_index()
            Lc = Lm.xs(c, level="config").reindex(g.index)
            for K in pr["check_K"]:
                for cp in BO.COUPLINGS:
                    a, bcol = f"Jemp_{cp}_{K}", f"Jf_{cp}_{K}"
                    if a not in g or bcol not in g or not g[a].notna().any():
                        continue
                    v = (g[a] - g[bcol]).values
                    VK = (g[bcol] - Lc).values
                    ok = np.isfinite(v) & np.isfinite(VK)
                    v, VK, Jf = v[ok], VK[ok], g[bcol].values[ok]
                    n = len(v)
                    bs = v[boot_idx(n, nb, seed)].mean(1)
                    lo, hi = ci_of(bs, al)
                    halves = []
                    for h in (slice(0, None, 2), slice(1, None, 2)):
                        vh = v[h]
                        lo_h, hi_h = ci_of(vh[boot_idx(len(vh), nb, seed)].mean(1), al)
                        halves.append((float(vh.mean()), lo_h > 0 or hi_h < 0))
                    rows.append({"config": c, "coupling": cp, "K": int(K), "n": n, "chênh TB [CI]": fmt_ci(float(v.mean()), lo, hi),
                                 "chênh / TB J": float(v.mean() / Jf.mean()), "chênh / TB V_K": float(v.mean() / VK.mean()),
                                 "skew": skew(v), "trung vị": float(np.median(v)), "tỷ lệ ảnh > 0": float((v > 0).mean()),
                                 "p": boot_p(bs), "nửa chẵn / lẻ: TB (CI loại 0?)": f"{halves[0][0]:.3g} ({'có' if halves[0][1] else 'không'}) / "
                                 f"{halves[1][0]:.3g} ({'có' if halves[1][1] else 'không'})",
                                 "_rel": v / np.where(VK > 0, VK.mean(), np.nan), "_miss": not (lo <= 0 <= hi)})
        if not rows:
            continue
        ph = holm([r["p"] for r in rows])
        for r, h in zip(rows, ph):
            r["p Holm (giai đoạn)"] = h
        nm = sum(r["_miss"] for r in rows)
        pooled.append({"giai đoạn": stage, "số dòng": len(rows), "CI loại 0 (chưa đ/c)": nm, "kỳ vọng nếu CI đúng 95 %": 0.05 * len(rows),
                       "Holm < 0,05": sum(h < al for h in ph),
                       "TB chênh / V_K gộp mọi dòng": float(np.nanmean(np.concatenate([r["_rel"] for r in rows])))})
        for r in rows:
            del r["_rel"], r["_miss"]
        L.append(f"### {stage}\n")
        L.append(md(pd.DataFrame(rows)))
    if pooled:
        L.append("**Tóm tắt** (các dòng cùng cấu hình dùng chung đường ⇒ không độc lập; số CI trượt chỉ là chỉ báo):\n")
        L.append(md(pd.DataFrame(pooled)))
    return L


def run(cfg: dict) -> str:
    rp = os.path.join(C.out_dir(cfg, "pilot"), "infid", "INFID_BUDGET_AUDIT.md")
    lk = BO.lock_path(cfg)
    L = ["# Kiểm toán vòng BUDGET 1 (không GPU, không chọn lại)\n",
         f"Khoá test: `{json.load(open(lk))['digest'][:12]}`." if os.path.exists(lk) else "Chưa có khoá.", ""]
    L += selection_section(cfg)
    L += r_section(cfg)
    L += emp_section(cfg)
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[budget-audit] {rp}")
    return rp
