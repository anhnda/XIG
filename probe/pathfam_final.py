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
