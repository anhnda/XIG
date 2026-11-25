"""
PATHFAM — kiểm toán cuối trước khi đóng phần thực nghiệm (CPU; không GPU, không chọn lại, không đổi endpoint).

  1. Thang đo β (từ signdiag2.csv): β = yᵀd/yᵀy = (‖d‖/‖y‖)·cos(y, d) trên từng ảnh; so L ở β = 1 (đã khoá), β riêng từng ảnh (trong mẫu
     và cross-fit theo ô), β CHUNG cho mọi ảnh (trong mẫu) và β chung ngoài mẫu theo ẢNH (bỏ-một-ảnh, fit trên bộ ô chọn của các ảnh khác,
     đánh giá trên bộ ô đánh giá của ảnh bỏ ra). β riêng từng ảnh dùng ΔF_S của chính ảnh đó ⇒ KHÔNG phải ước lượng attribution triển
     khai độc lập được; chỉ β chung ngoài mẫu là triển khai được (một hằng số hiệu chỉnh mỗi cấu hình).
  2. Completeness: (a) mô phỏng CPU với cầu phương `rand` M = 1 trên đường lưới, cặp `rev` độc lập — kiểm luật E|ΣÂ − ΔF| ∝ K^{−1/2}
     (thiết kế lồng nhau như dữ liệu thật và thiết kế độc lập), CI bootstrap cho độ dốc và cho sai số có dấu; (b) cùng phân tích trên
     dữ liệu thật signdiag2 (CI theo ảnh).
  3. Khoá chẩn đoán: chỉ InsDel nền đen (có dấu / không dấu) và endpoint đã khoá (InsDel blur không dấu) được dùng; InsDel blur CÓ DẤU bị
     loại (lệch nền, docs/PATHFAM_METRIC_AUDIT.md); mọi so sánh cùng exploration là hậu kiểm mô tả. → pathfam/DIAG_LOCK.json
Lệnh: bash scripts/run.sh pathfam-final-audit → runs_probe/pilot/infid/pathfam/PATHFAM_FINAL_AUDIT.md
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os

import numpy as np
import torch

from . import path_families as PF
from . import pathfam as PFM

F64 = torch.float64
ADMISSIBLE = {"insdel_abs_blur": "endpoint ngoài đã khoá (không dấu, nền blur) — giữ, kèm ghi chú lệch nền",
              "insdel_abs_black": "chẩn đoán, nền nhất quán với attribution", "insdel_signed_black": "chẩn đoán, nền nhất quán với attribution",
              "iou_abs": "chẩn đoán (không dùng nền)", "iou_pos": "chẩn đoán (không dùng nền)", "energy_abs": "chẩn đoán (không dùng nền)",
              "energy_pos": "chẩn đoán (không dùng nền)"}
EXCLUDED = {"insdel_signed_blur": "LOẠI: xếp hạng có dấu theo nền đen nhưng đo với nền blur — không có diễn giải nhất quán",
            "signdiag_v1.H-SU.insdel": "LOẠI: kết luận H-SU vòng 1 về InsDel dùng nền blur"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. thang đo
# ─────────────────────────────────────────────────────────────────────────────

def beta_parts(yy, yd, dd):
    """β = r·cos với r = ‖d‖/‖y‖, cos = yᵀd/(‖y‖‖d‖); L(1) = 1/r² − 2cos/r + 1; L(β_opt) = 1 − cos²."""
    yy, yd, dd = (np.asarray(a, float) for a in (yy, yd, dd))
    r = np.sqrt(dd / yy)
    cos = yd / np.sqrt(yy * dd)
    return {"r": r, "cos": cos, "beta": r * cos, "L1": (yy - 2 * yd + dd) / dd, "Lopt": 1 - cos ** 2}


def L_beta(b, yy, yd, dd):
    return (b * b * yy - 2 * b * yd + dd) / dd


def pooled_beta(yy, yd, dd) -> float:
    """β chung tối thiểu Σ_ảnh L_i(β) (L chuẩn hoá theo ảnh): β* = Σ(yd/dd) / Σ(yy/dd)."""
    yy, yd, dd = (np.asarray(a, float) for a in (yy, yd, dd))
    return float((yd / dd).sum() / (yy / dd).sum())


def loio_beta(yy_s, yd_s, dd_s) -> np.ndarray:
    """β chung bỏ-một-ảnh (fit trên bộ ô chọn của các ảnh khác)."""
    a = np.asarray(yd_s, float) / np.asarray(dd_s, float)
    b = np.asarray(yy_s, float) / np.asarray(dd_s, float)
    return (a.sum() - a) / (b.sum() - b)


def beta_section(t, nb: int, seed: int, al: float) -> list:
    import pandas as pd
    from .infid_posthoc import boot_idx, ci_of, fmt_ci
    from .report import md
    rows, parts = [], []
    for (cfgn, est), g in t.groupby(["config", "estimator"]):
        N = int(g.N.iloc[0])
        if est != "mean_all" and est not in {f"rev_K{B // N}" for B in (256, 1024)}:
            continue
        g = g.sort_values("image")
        e = beta_parts(g.yy_eval, g.yd_eval, g.dd_eval)
        s = beta_parts(g.yy_sel, g.yd_sel, g.dd_sel)
        Ls = {"β = 1 (đã khoá)": e["L1"],
              "β ảnh, trong mẫu (ô đánh giá)": e["Lopt"],
              "β ảnh, cross-fit ô (chọn → đánh giá)": L_beta(s["beta"], g.yy_eval.values, g.yd_eval.values, g.dd_eval.values),
              "β chung, trong mẫu": L_beta(pooled_beta(g.yy_eval, g.yd_eval, g.dd_eval), g.yy_eval.values, g.yd_eval.values, g.dd_eval.values),
              "β chung, ngoài mẫu theo ảnh (triển khai được)": L_beta(loio_beta(g.yy_sel, g.yd_sel, g.dd_sel), g.yy_eval.values, g.yd_eval.values, g.dd_eval.values)}
        n = len(g)
        idx = boot_idx(n, nb, seed)
        r = {"config": cfgn, "bản đồ": est, "n": n, "β chung (trong mẫu)": pooled_beta(g.yy_eval, g.yd_eval, g.dd_eval)}
        for k, v in Ls.items():
            v = np.asarray(v, float)
            lo, hi = ci_of(v[idx].mean(1), al)
            r[f"L: {k}"] = fmt_ci(float(v.mean()), lo, hi, "{:.3f}")
        rows.append(r)
        q = lambda x: " / ".join(f"{z:.3g}" for z in np.quantile(x[np.isfinite(x)], [0.1, 0.5, 0.9]))
        lr, lc = np.log(e["r"]), np.log(np.clip(e["cos"], 1e-12, None))
        parts.append({"config": cfgn, "bản đồ": est, "‖d‖/‖y‖ (q10 / q50 / q90)": q(e["r"]), "cos(y, d) (q10 / q50 / q90)": q(e["cos"]),
                      "β ảnh (q10 / q50 / q90)": q(e["beta"]), "tỷ lệ ảnh β < 1": float((e["beta"] < 1).mean()),
                      "tỷ lệ ảnh cos ≤ 0": float((e["cos"] <= 0).mean()),
                      "TB log(‖d‖/‖y‖)": float(lr.mean()), "TB log cos (cos > 0)": float(lc[e["cos"] > 0].mean()),
                      "max |β − r·cos|": float(np.abs(e["beta"] - (g.yd_eval / g.yy_eval).values).max())})
    return ["## 1. Thang đo β = (‖d‖/‖y‖)·cos(y, d)\n",
            "Bộ ô đánh giá, D chính, bản đồ ở K của B = 256 / 1024 và Ē. log β = log(‖d‖/‖y‖) + log cos: phần **chuẩn** (bản đồ chiếu quá lớn / "
            "quá nhỏ) tách khỏi phần **hướng** (cos — không sửa được bằng thang đo). L(β_opt) = 1 − cos² là cận dưới của mọi hiệu chỉnh thang "
            "đo. **β riêng từng ảnh dùng ΔF_S của chính ảnh ⇒ không phải ước lượng attribution triển khai độc lập**; chỉ β chung ngoài mẫu theo "
            "ảnh là hiệu chỉnh triển khai được.\n",
            md(pd.DataFrame(parts)), "L dưới các lựa chọn β (TB theo ảnh [CI 95 %]):\n", md(pd.DataFrame(rows))]


# ─────────────────────────────────────────────────────────────────────────────
# 2. completeness
# ─────────────────────────────────────────────────────────────────────────────

def toy_model(d: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    b = torch.randn(d, generator=g, dtype=F64)
    Q = torch.randn(d, d, generator=g, dtype=F64) / math.sqrt(d)
    Q = (Q + Q.T) / 2
    w = torch.randn(d, generator=g, dtype=F64) * 1.5
    x = torch.rand(d, generator=g, dtype=F64) + 0.5

    def F(X):
        return X @ b + 0.5 * ((X @ Q) * X).sum(-1) + torch.tanh(X @ w)

    def grad(X):
        return b + X @ Q + (1 - torch.tanh(X @ w) ** 2).unsqueeze(-1) * w
    return F, grad, x


def rand_attr_err(grad, F, x, Z: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Sai số completeness ΣÂ − ΔF của cầu phương `rand` M = 1 (một điểm đều mỗi đoạn, chung cho mọi feature). Z (N+1, P, d)."""
    N = Z.shape[0] - 1
    X = Z * x
    dX = X[1:] - X[:-1]
    u = torch.rand((N, Z.shape[1], 1), generator=g, dtype=F64)
    A = (grad(X[:-1] + u * dX) * dX).sum(0)
    return A.sum(-1) - (F(x) - F(torch.zeros_like(x)))


def completeness_sim(reps: int = 4000, d: int = 16, N: int = 32, Ks=(2, 4, 8, 16, 32), alpha: float = 1.0, seed: int = 0,
                     nb: int = 2000) -> dict:
    """Mỗi lần lặp: m = max K/2 cặp rev độc lập. Lồng nhau: K đường = K/2 cặp đầu (như signdiag2). Độc lập: mỗi K dùng cặp mới."""
    from .infid_posthoc import boot_idx, ci_of
    F, grad, x = toy_model(d, 900 + seed)
    g = torch.Generator().manual_seed(seed)
    m = max(Ks) // 2
    out = {}
    for design in ("lồng nhau", "độc lập"):
        E = {K: [] for K in Ks}
        for a in range(0, reps, 250):
            R = min(250, reps - a)
            if design == "lồng nhau":
                Z = PF.schedule("dp", N, (R * m, d), g, {"alpha": alpha})
                e1 = rand_attr_err(grad, F, x, Z, g).view(R, m)
                e2 = rand_attr_err(grad, F, x, 1.0 - Z.flip(0), g).view(R, m)
                pm = (e1 + e2) / 2
                for K in Ks:
                    E[K].append(pm[:, : K // 2].mean(1))
            else:
                for K in Ks:
                    Z = PF.schedule("dp", N, (R * (K // 2), d), g, {"alpha": alpha})
                    e1 = rand_attr_err(grad, F, x, Z, g).view(R, K // 2)
                    e2 = rand_attr_err(grad, F, x, 1.0 - Z.flip(0), g).view(R, K // 2)
                    E[K].append(((e1 + e2) / 2).mean(1))
        M = np.stack([torch.cat(E[K]).numpy() for K in Ks], 1)                # (reps, |K|)
        lk = np.log(np.array(Ks, float))
        idx = boot_idx(M.shape[0], nb, seed)
        slope = float(np.polyfit(lk, np.log(np.abs(M).mean(0)), 1)[0])
        sb = np.array([np.polyfit(lk, np.log(np.abs(M[i]).mean(0)), 1)[0] for i in idx])
        rows = []
        for j, K in enumerate(Ks):
            v = M[:, j]
            lo, hi = ci_of(v[idx].mean(1), 0.05)
            alo, ahi = ci_of(np.abs(v)[idx].mean(1), 0.05)
            rows.append({"K": K, "TB sai số có dấu [CI]": f"{v.mean():.4g} [{lo:.3g}, {hi:.3g}]", "TB |sai số| [CI]": f"{np.abs(v).mean():.4g} [{alo:.3g}, {ahi:.3g}]",
                         "√(K/2)·TB|sai số| (hằng nếu ∝ K^{−1/2})": float(np.abs(v).mean() * math.sqrt(K / 2))})
        out[design] = {"slope": slope, "slope_ci": ci_of(sb, 0.05), "rows": rows}
    return out


def completeness_real(t, nb: int, seed: int, al: float) -> list:
    import pandas as pd
    from .infid_posthoc import boot_idx, ci_of, fmt_ci
    from .report import md
    tau = float(np.median(np.abs(t.groupby("image").dF.first().values)))
    t = t.assign(err=t.sumA - t.dF)
    rows, srows = [], []
    for cfgn, g in t.groupby("config"):
        Ks = sorted(int(k) for k in g[g.estimator.str.startswith("rev_K")].K.unique())
        if len(Ks) < 3:
            continue
        W = g[g.K.isin(Ks) & g.estimator.str.startswith("rev_K")].pivot_table(index="image", columns="K", values="err", aggfunc="first").dropna()
        dF = g.groupby("image").dF.first().reindex(W.index).values
        X = W.values
        lk = np.log(np.array(Ks, float))
        idx = boot_idx(X.shape[0], nb, seed)
        slope = float(np.polyfit(lk, np.log(np.abs(X).mean(0)), 1)[0])
        lo, hi = ci_of(np.array([np.polyfit(lk, np.log(np.abs(X[i]).mean(0)), 1)[0] for i in idx]), al)
        srows.append({"config": cfgn, "n ảnh": X.shape[0], "độ dốc log TB|err| theo log K [CI]": fmt_ci(slope, lo, hi, "{:.3f}"),
                      "chứa −0,5": bool(lo <= -0.5 <= hi)})
        for j, K in enumerate(Ks):
            v = X[:, j]
            l1, h1 = ci_of(v[idx].mean(1), al)
            rob = np.abs(v) / (np.abs(dF) + tau)
            rows.append({"config": cfgn, "K": int(K), "TB err có dấu [CI]": fmt_ci(float(v.mean()), l1, h1, "{:.3g}"),
                         "TB |err|": float(np.abs(v).mean()), "TB |err|/(|ΔF| + τ)": float(rob.mean())})
    return ["### 2b. Dữ liệu thật (signdiag2, bản đồ lồng nhau, CI theo ảnh)\n",
            f"τ = trung vị |ΔF| qua ảnh = {tau:.4g}. Sai số có dấu phải có CI chứa 0 (cầu phương không chệch); độ dốc ≈ −0,5.\n",
            md(pd.DataFrame(srows)), md(pd.DataFrame(rows))]


def blocks_from_nested(S: np.ndarray) -> tuple:
    """S (n_ảnh, 5): sai số trung bình của K = 2, 4, 8, 16, 32 đường LỒNG NHAU (= TB của 1, 2, 4, 8, 16 cặp đầu).
    Trả tổng sai số theo khối cặp RỜI NHAU (kích thước 1, 1, 2, 4, 8) và kích thước khối."""
    n = np.array([1, 1, 2, 4, 8], float)
    c = np.array([1, 2, 4, 8, 16], float)
    T = S * c                                                                # tổng của c cặp đầu
    B = np.concatenate([T[:, :1], np.diff(T, axis=1)], axis=1)
    return B, n


def within_var(B: np.ndarray, n: np.ndarray) -> tuple:
    """σ̂²_i (phương sai TB một cặp) không chệch và không phụ thuộc độ lệch riêng ảnh b_i: Σ_j n_j(B_j/n_j − m̂)²/(J − 1), m̂ = ΣB/Σn."""
    m = B.sum(1) / n.sum()
    s2 = (n * (B / n - m[:, None]) ** 2).sum(1) / (B.shape[1] - 1)
    return m, s2


KV = np.array([2, 4, 8, 16, 32], float)
_SLOPE_W = np.linalg.pinv(np.vstack([np.log(KV), np.ones(5)]).T)[0]         # độ dốc OLS = _SLOPE_W @ y


def slope_parts(S: np.ndarray) -> np.ndarray:
    """S (..., n, 5). Phân rã CHÍNH XÁC: log TB|e_K| = log RMS_K + log G_K, G_K = TB|e_K| / RMS_K (tỷ số Geary gộp qua ảnh).
    Trả (..., 3): độ dốc TB|·|, độ dốc RMS, độ dốc log G (cột 0 = cột 1 + cột 2)."""
    ma = np.abs(S).mean(-2)
    rm = np.sqrt((S ** 2).mean(-2))
    return np.stack([np.log(ma) @ _SLOPE_W, np.log(rm) @ _SLOPE_W, np.log(ma / rm) @ _SLOPE_W], -1)


def shape_draw(rng, shape: str, size: tuple) -> np.ndarray:
    """Sai số một cặp chuẩn hoá (TB 0, phương sai 1) theo hình dạng: normal, uniform (đuôi nhẹ), t3 (đuôi nặng),
    spike (mỗi đường: nhiễu nhỏ + 5 % xác suất một đoạn trội — mô hình 'một đoạn chứa phần lớn biến thiên gradient')."""
    if shape == "normal":
        return rng.standard_normal(size)
    if shape == "uniform":
        return (rng.random(size) - 0.5) * math.sqrt(12)
    if shape == "t3":
        return rng.standard_t(3, size) / math.sqrt(3)
    if shape == "spike":
        e = 0.3 * rng.standard_normal(size + (2,)) + (rng.random(size + (2,)) < 0.05) * rng.standard_normal(size + (2,)) * math.sqrt(0.91 / 0.05)
        return e.mean(-1) * math.sqrt(2)
    raise ValueError(shape)


def slope_shape_sim(n: int, s: float, shapes=("normal", "uniform", "t3", "spike"), tbs=(0.0, 0.2), R: int = 300, nb: int = 300,
                    seed: int = 0) -> list:
    """Phân phối lấy mẫu của các độ dốc ở ĐÚNG thiết kế dữ liệu thật: n ảnh, σ_i log-chuẩn độ lệch s, 16 cặp độc lập lồng nhau K = 2…32,
    bootstrap theo ảnh. tb: độ lệch riêng ảnh b_i ~ N(0, (tb·σ_i)²). Trả hàng: E và SD của từng độ dốc, độ phủ −½ của CI percentile."""
    rows = []
    c = np.array([1, 2, 4, 8, 16])
    for shape in shapes:
        for tb in tbs:
            if tb and shape != "normal":
                continue
            rng = np.random.default_rng([seed, hash(shape) % 2 ** 31, int(tb * 1000)])
            sig = np.exp(s * rng.standard_normal((R, n, 1)))
            pairs = tb * sig[..., 0:1] * rng.standard_normal((R, n, 1)) + sig * shape_draw(rng, shape, (R, n, 16))
            S = np.stack([pairs[..., :k].mean(-1) for k in c], -1)
            sp = slope_parts(S)
            cov = np.zeros(2)
            for r in range(R):
                bs = slope_parts(S[r][rng.integers(0, n, (nb, n))])
                for j in range(2):
                    cov[j] += np.quantile(bs[:, j], 0.025) <= -0.5 <= np.quantile(bs[:, j], 0.975)
            rows.append({"hình dạng sai số cặp": shape, "độ lệch riêng ảnh tb": tb,
                         "E độ dốc TB|err| (SD)": f"{sp[:, 0].mean():.3f} ({sp[:, 0].std():.3f})",
                         "E độ dốc RMS (SD)": f"{sp[:, 1].mean():.3f} ({sp[:, 1].std():.3f})",
                         "E độ dốc log G": f"{sp[:, 2].mean():.3f}",
                         "độ phủ −½: TB|err|": cov[0] / R, "độ phủ −½: RMS": cov[1] / R})
    return rows


def completeness_diag(t, nb: int, seed: int, al: float, R: int = 300) -> list:
    """Vì sao độ dốc log TB|err| theo log K trên CNN có thể khác −½ (dữ liệu signdiag2, không GPU).
    Sự kiện cấu trúc: với tích phân CHÍNH XÁC, Σ_i A_i(Z) = ΔF cho MỌI đường ⇒ err chỉ là nhiễu cầu phương `rand` (U độc lập giữa đường,
    giữa hai thành viên của cặp, giữa các cặp). Phân rã chính xác: độ dốc TB|·| = độ dốc RMS + độ dốc log G. Mô hình
    E e_K² = E b² + 2σ²/K (đúng chính xác, mọi hình dạng, cặp độc lập) ⇒ độ dốc RMS lệch −½ trong kỳ vọng CHỈ khi E b² > 0;
    độ dốc log G ≠ 0 ⇔ hình dạng phân phối thay đổi theo K (đuôi nặng ⇒ > 0, làm phẳng; đuôi nhẹ ⇒ < 0, làm dốc)."""
    import pandas as pd
    from .infid_posthoc import boot_idx, ci_of, fmt_ci
    from .report import md
    rows, svals = [], []
    t = t.assign(err=t.sumA - t.dF)
    for cfgn, g in t.groupby("config"):
        q = g[g.estimator.str.startswith("rev_K")]
        if sorted(int(k) for k in q.K.unique()) != [2, 4, 8, 16, 32]:
            continue
        S = q.pivot_table(index="image", columns="K", values="err", aggfunc="first").dropna()[[2, 4, 8, 16, 32]].values
        idx = boot_idx(S.shape[0], nb, seed)
        sp = slope_parts(S)
        bs = slope_parts(S[idx])
        B, n = blocks_from_nested(S)
        m, s2 = within_var(B, n)
        b2 = m ** 2 - s2 / 16.0                                              # ước lượng không chệch của b_i²
        b2lo, b2hi = ci_of(b2[idx].mean(1), al)
        mse32 = float((S[:, 4] ** 2).mean())
        G = np.abs(S).mean(0) / np.sqrt((S ** 2).mean(0))
        ok = s2 > 0
        # độ phân tán log σ_i qua ảnh (trừ phần do σ̂² chỉ có 4 bậc tự do: Var log(χ²₄/4) = ψ'(2) = π²/6 − 1)
        sl = float(np.sqrt(max(np.var(np.log(s2[ok])) - (math.pi ** 2 / 6 - 1), 0.0)) / 2) if ok.sum() > 2 else math.nan
        svals.append(sl)
        rows.append({"config": cfgn, "n ảnh": S.shape[0],
                     "độ dốc TB|err| [CI]": fmt_ci(float(sp[0]), *ci_of(bs[:, 0], al), "{:.3f}"),
                     "= độ dốc RMS [CI]": fmt_ci(float(sp[1]), *ci_of(bs[:, 1], al), "{:.3f}"),
                     "+ độ dốc log G [CI]": fmt_ci(float(sp[2]), *ci_of(bs[:, 2], al), "{:.3f}"),
                     "G ở K = 2 / 32": f"{G[0]:.3f} / {G[4]:.3f}",
                     "G tham chiếu chuẩn + không đồng nhất": float(math.sqrt(2 / math.pi) * math.exp(-sl ** 2 / 2)) if np.isfinite(sl) else math.nan,
                     "TB b̂² [CI]": fmt_ci(float(b2.mean()), b2lo, b2hi, "{:.3g}"),
                     "b̂² / MSE(K = 32)": float(b2.mean() / mse32) if mse32 > 0 else math.nan,
                     "SD log σ_i qua ảnh": sl,
                     "độ dốc RMS dự báo từ b̂², σ̂²": float(0.5 * np.log(np.clip(b2.mean() + s2.mean() * 2 / KV, 1e-300, None)) @ _SLOPE_W)})
    out = ["### 2c. Vì sao độ dốc trên CNN có thể khác −½ (từ dữ liệu signdiag2, không GPU)\n",
           "Bản đồ lồng nhau K = 2…32 là trung bình của 1, 2, 4, 8, 16 cặp đầu ⇒ khôi phục tổng sai số trên các khối cặp RỜI NHAU (1, 1, 2, 4, 8) "
           "cho từng ảnh ⇒ σ̂²_i (phương sai một cặp, không phụ thuộc b_i) và b̂²_i = m̂_i² − σ̂²_i/16 (không chệch cho b_i²). Phân rã CHÍNH XÁC: "
           "độ dốc TB|err| = độ dốc RMS + độ dốc log G với G_K = TB|e_K|/RMS_K. Vì E e_K² = E b² + 2σ²/K đúng với mọi hình dạng khi các cặp "
           "độc lập, độ dốc RMS chỉ lệch −½ trong kỳ vọng khi E b² > 0 (riêng ước lượng log của nó có chệch Jensen khi đuôi nặng, xem 2d); "
           "độ dốc log G ≠ 0 nghĩa là hình dạng phân phối hữu hạn K đang hội tụ về chuẩn (G = 0,798): đuôi nặng ⇒ G nhỏ ở K nhỏ ⇒ độ dốc TB|·| "
           "phẳng hơn −½; đuôi nhẹ ⇒ dốc hơn. G gộp qua ảnh có thang σ_i khác nhau thấp hơn 0,798 ngay cả khi sai số chuẩn: tham chiếu «chuẩn + không "
           "đồng nhất» = 0,798·exp(−s²/2) với s = SD log σ_i (giả định σ_i log-chuẩn) — mức này không đổi theo K nên không ảnh hưởng độ dốc. "
           "Cột «độ dốc RMS dự báo» dùng E e_K² = b̂² + 2σ̂²/K: so với độ dốc RMS quan sát để biết b̂² giải thích được bao nhiêu. Độ lệch riêng ảnh b_i (nếu có) chưa xác định nguồn — không gọi là lỗi cài đặt khi chưa có bằng chứng "
           "trực tiếp.\n", md(pd.DataFrame(rows))]
    if rows:
        s_use = float(np.nanmedian(svals)) if np.isfinite(np.nanmedian(svals)) else 1.0
        n_use = int(np.median([r["n ảnh"] for r in rows]))
        out += ["### 2d. Phân phối lấy mẫu của độ dốc ở đúng thiết kế (CPU, mô phỏng)\n",
                f"n = {n_use} ảnh, SD log σ_i = {s_use:.2f} (trung vị ước lượng ở 2c), 16 cặp độc lập lồng nhau, bootstrap theo ảnh, R = {R} lần lặp. "
                "Cho biết với cỡ mẫu và dải K này, độ dốc TB|err| dao động bao nhiêu và lệch hệ thống bao nhiêu chỉ do hình dạng phân phối "
                "hữu hạn K hoặc do b_i — KHÔNG phải ước lượng cho CNN.\n",
                md(pd.DataFrame(slope_shape_sim(n_use, s_use, R=R, nb=nb if nb < 300 else 300, seed=seed)))]
    return out


def offset_shared(t, nb: int, seed: int, al: float) -> list:
    """Độ lệch riêng ảnh có CHUNG giữa các cấu hình không. m̂_ic = sai số ở K = 32 (TB 16 cặp) của ảnh i, cấu hình c. Nhiễu cầu phương / đường
    của hai cấu hình độc lập (luồng ngẫu nhiên theo tên cấu hình) ⇒ E[m̂_ic m̂_ic′] = E[b_ic b_ic′] CHÍNH XÁC, không cần trừ phương sai.
    Nguồn theo ảnh không phụ thuộc họ đường (ΔF, số học ở đầu mút) ⇒ hiệp phương sai chéo ≈ b̂²; độ lệch riêng theo cấu hình ⇒ ≈ 0."""
    import pandas as pd
    from .infid_posthoc import boot_idx, ci_of, fmt_ci
    from .report import md
    t = t.assign(err=t.sumA - t.dF)
    q = t[t.estimator.str.startswith("rev_K")]
    q = q[q.K == 32]
    if q.empty or q.config.nunique() < 2:
        return []
    Mx = q.pivot_table(index="image", columns="config", values="err", aggfunc="first").dropna()
    dF = t.groupby("image").dF.first().reindex(Mx.index).values
    rows = []
    groups = {"mọi cấu hình": list(Mx.columns)}
    n32 = [c for c in Mx.columns if c.endswith("_N32")]
    if 1 < len(n32) < len(Mx.columns):
        groups["chỉ N = 32"] = n32
    for gname, cols in groups.items():
        X = Mx[cols].values
        n, C = X.shape
        idx = boot_idx(n, nb, seed)

        def cross(Y):
            s = Y.sum(-1)
            return ((s ** 2 - (Y ** 2).sum(-1)) / (C * (C - 1))).mean(-1)   # TB qua ảnh của TB m̂_c m̂_c′ (c ≠ c′)
        cr = float(cross(X))
        crb = cross(X[idx])
        ms2 = float((X ** 2).mean())                                         # MSE(K = 32) gộp
        sh = X.mean(1)
        rk = pd.Series(sh).rank().values
        rd = pd.Series(dF).rank().values
        rho = float(np.corrcoef(rk, rd)[0, 1])
        rhob = np.array([np.corrcoef(pd.Series(sh[i]).rank().values, pd.Series(dF[i]).rank().values)[0, 1] for i in idx])
        rows.append({"nhóm": gname, "n ảnh": n, "số cấu hình": C,
                     "TB m̂_c·m̂_c′ (c ≠ c′) = E b_c b_c′ [CI]": fmt_ci(cr, *ci_of(crb, al), "{:.3g}"),
                     "÷ MSE(K = 32)": cr / ms2 if ms2 > 0 else math.nan,
                     "Spearman(TB m̂ qua cấu hình, ΔF) [CI]": fmt_ci(rho, *ci_of(rhob, al), "{:.3f}")})
    return ["### 2e. Độ lệch riêng ảnh có chung giữa các cấu hình không\n",
            "m̂_ic = sai số completeness ở K = 32 của ảnh i với cấu hình c. Nhiễu của hai cấu hình độc lập ⇒ TB m̂_ic·m̂_ic′ (c ≠ c′) ước lượng "
            "KHÔNG CHỆCH E[b_ic b_ic′] mà không cần trừ phương sai (khác b̂² ở 2c). Đọc: (a) dương rõ, cỡ b̂² ⇒ độ lệch theo ảnh, chung cho mọi họ "
            "đường (nguồn khả dĩ: phép tính ΔF / số học ở đầu mút) — cần kiểm tra trực tiếp trước khi gọi tên nguồn; (b) ≈ 0 trong khi b̂² > 0 ⇒ "
            "độ lệch (nếu có) phụ thuộc cấu hình; (c) ≈ 0 và b̂² ≈ 0 ⇒ không có bằng chứng độ lệch, độ phẳng của độ dốc là do lấy mẫu / đuôi nặng.\n",
            md(pd.DataFrame(rows))]


# ─────────────────────────────────────────────────────────────────────────────
# 3. khoá chẩn đoán
# ─────────────────────────────────────────────────────────────────────────────

def lock_diag(cfg: dict) -> dict:
    root = PFM.root(cfg)
    files = {}
    for rel in ("signdiag2/signdiag2.csv", "signdiag2/meta.json", "signdiag/signdiag.csv", "PATHFAM_SIGNDIAG2_AUDIT.md", "PATHFAM_SIGNED_UNSIGNED.md"):
        p = os.path.join(root, rel)
        if os.path.exists(p):
            files[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "status": "chẩn đoán hậu kiểm — mô tả, không xác nhận",
           "admissible_metrics": ADMISSIBLE, "excluded": EXCLUDED,
           "same_exploration_comparisons": "hậu kiểm mô tả trên tập test đã dùng; không tạo tuyên bố xác nhận; khớp h không bảo đảm khớp J",
           "beta": "β riêng từng ảnh cần ΔF_S của chính ảnh ⇒ chỉ là chẩn đoán; hiệu chỉnh triển khai được duy nhất là β chung ngoài mẫu",
           "files_sha256": files}
    p = os.path.join(root, "DIAG_LOCK.json")
    with open(p, "w") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    return rec


def run(cfg: dict) -> str:
    import pandas as pd
    from .report import md
    P = PFM.pr(cfg)
    nb, seed, al = int(P["bootstrap"]), int(P["seed"]), float(P["alpha"])
    root = PFM.root(cfg)
    L = ["# PATHFAM — kiểm toán cuối trước khi đóng phần thực nghiệm (CPU, mô tả)\n",
         f"{_dt.datetime.now().isoformat(timespec='seconds')}. Không GPU, không chọn lại, không đổi endpoint.\n"]
    csv = os.path.join(root, "signdiag2", "signdiag2.csv")
    t = pd.read_csv(csv) if os.path.exists(csv) else None
    if t is not None:
        L += beta_section(t, nb, seed, al)
    else:
        L.append(f"## 1. Thang đo\n\nkhông thấy {csv}.\n")
    dbg = bool(cfg.get("_debug"))
    sim = completeness_sim(reps=400 if dbg else 4000, nb=200 if dbg else 2000, seed=seed)
    L.append("## 2. Completeness: luật K^{−1/2}\n")
    L.append("### 2a. Mô phỏng CPU\n")
    L.append("F = bᵀx + ½xᵀQx + tanh(wᵀx) (d = 16), đường dp α = 1 trên lưới N = 32, cầu phương `rand` M = 1 (một điểm đều mỗi đoạn, chung "
             "mọi feature — như pipeline), cặp `rev` độc lập. Sai số của trung bình K đường = trung bình K/2 cặp độc lập ⇒ lý thuyết E|err| ∝ "
             "K^{−1/2}. Hai thiết kế: lồng nhau (như signdiag2: K đường là K/2 cặp đầu của cùng một dãy) và độc lập.\n")
    for dsg, r in sim.items():
        L.append(f"**{dsg}**: độ dốc log TB|err| theo log K = {r['slope']:.3f} [CI 95 % {r['slope_ci'][0]:.3f}, {r['slope_ci'][1]:.3f}] (lý thuyết −0,5).\n")
        L.append(md(pd.DataFrame(r["rows"])))
    if t is not None:
        L += completeness_real(t, nb, seed, al)
        L += completeness_diag(t, nb, seed, al, R=40 if dbg else 300)
        L += offset_shared(t, nb, seed, al)
    rec = lock_diag(cfg)
    L += ["## 3. Khoá chẩn đoán (pathfam/DIAG_LOCK.json)\n",
          md(pd.DataFrame([{"metric": k, "trạng thái": v} for k, v in {**rec["admissible_metrics"], **rec["excluded"]}.items()])),
          f"So sánh cùng exploration: {rec['same_exploration_comparisons']}. β: {rec['beta']}.\n",
          "File đã băm: " + ", ".join(f"`{k}`" for k in rec["files_sha256"]) + ".\n"]
    rp = os.path.join(root, "PATHFAM_FINAL_AUDIT.md")
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[pathfam-final-audit] {rp}")
    return rp
