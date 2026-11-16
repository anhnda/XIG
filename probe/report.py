"""
Sinh ATTR_PROBE_PILOT_REPORT.md từ các CSV của pilot. Mọi phán quyết ở mục G là chỉ báo theo luật cố định,
ghi rõ luật ngay cạnh kết quả; pilot nhỏ, nên CONFIRMED/REFUTED ở đây nghĩa là "rõ trong pilot", chưa phải xác nhận.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

BOOT = 2000


# ─────────────────────────────────────────────────────────────────────────────
# tiện ích
# ─────────────────────────────────────────────────────────────────────────────

def _read(out, name):
    import pandas as pd
    p = os.path.join(out, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "✓" if v else "✗"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf"
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e4 or a < 1e-3:
        return f"{v:.2e}"
    return f"{v:.3f}" if a < 10 else f"{v:.1f}"


def md(df, cols=None, index=False):
    if df is None or not len(df):
        return "_(không có dữ liệu)_\n"
    d = df.reset_index() if index else df
    cols = cols or list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in d[cols].iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def boot_ci(x, stat=np.mean, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) < 2:
        return (float(stat(x)) if len(x) else float("nan"), float("nan"), float("nan"), len(x))
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(BOOT)]
    return float(stat(x)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), len(x)


def spearman(a, b):
    import pandas as pd
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 3:
        return float("nan")
    return float(s.a.rank().corr(s.b.rank()))


def boot_spearman(a, b, seed=0):
    import pandas as pd
    s = pd.DataFrame({"a": list(a), "b": list(b)}).dropna().values
    if len(s) < 4:
        return float("nan"), float("nan"), float("nan"), len(s)
    rng = np.random.default_rng(seed)
    est = spearman(s[:, 0], s[:, 1])
    bs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(s), len(s))
        bs.append(spearman(s[i, 0], s[i, 1]))
    bs = [v for v in bs if np.isfinite(v)]
    return est, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), len(s)


def q10(x):
    return float(np.nanquantile(x, 0.1))


def q90(x):
    return float(np.nanquantile(x, 0.9))


# ─────────────────────────────────────────────────────────────────────────────
# các mục
# ─────────────────────────────────────────────────────────────────────────────

AUDIT = """
Kiểm tra code `attr_probe` của lượt debug trước, đối chiếu với task. Các điểm đã sửa:

| # | Vấn đề ở bản trước | Sửa trong bản này |
|---|---|---|
| 1 | Canonical chỉ có Riemann trái `Σ g(x_k)⊙Δx_k`. Với SPI, Δx_k chứa bước nhảy DP lớn nên residual lớn (lượt debug: residual canonical lớn hơn IDGI). | Thêm `canon_mid` (U_k=½) và `canon_rand` (U_k~U(0,1), một U cho mỗi đoạn, dùng chung mọi phần tử, không chệch với tích phân trên đoạn nội suy). Giữ `canon_left` để so. Quy tắc canonical chính của P1–P3 là `canon_rand`. |
| 2 | IDGI chia `‖g_k‖²` đã kẹp ở 1e-300; bước có gradient bằng 0 vẫn cho credit bất định. | Bước có ‖g_k‖² ≤ `idgi_zero_tol`·max_k‖g_k‖² bị **bỏ**. ΔF của các bước đó báo trong `dropped_dF_rel` và hiện ra trong residual. |
| 3 | IDGI chỉ có một biến thể. | Giữ nguyên `idgi` (gradient tại nút trái, không gian input đã chuẩn hoá, đúng bài Yang et al.; thứ tự: w=g², chia Σw, nhân ΔF_k, cộng theo k). Thêm `idgi_raw`: trọng số theo gradient pixel thô g/std_c. Thêm `idgi_rand`: gradient tại đúng điểm cầu phương của `canon_rand`. Cả ba dùng cùng nút, cùng ΔF_k, cùng đường. |
| 4 | Guided IG trong thí nghiệm bẫy/tín hiệu: ΔA trộn phần do thành phần cộng thêm và phần do đường thay đổi. | Phân rã: A(F+c;γ′), A(F;γ′), A(F;γ). Với IDGI (không tuyến tính) thêm: I(F+c;γ)−I(F;γ) (đường cố định) và I(F+c;γ′)−I(F+c;γ) (đổi đường). |
| 5 | BlurIG: ground truth của tín hiệu tính từ ảnh đen, `sign_ok` so với dấu hệ số thay vì dấu của tổng đúng. | Ground truth từ điểm đầu thật của từng đường: A* = a(x − x_start)⊙R; với BlurIG x_start = blur_σmax(x). `sign_ok` so với dấu của ΣA*. Nếu \\|ΣA*\\| < 5% \\|B\\| thì retention = nan và cờ `B_true_ok`=✗. |
| 6 | Tín hiệu S = w·Σ_R x: chuẩn hoá theo x thay vì x−x′, gây hiểu lầm về tổng đóng góp. | S(x) = a·Σ_R(x_i − x′_i), a = B/Σ_R(x_i − x′_i), B = ±B_frac·\\|F(x)−F(x′)\\|. Kiểm mẫu số (trung bình x−x′ trên R ≥ `min_denominator`); vùng không đạt thì rút lại. |
| 7 | Bẫy: κ chọn theo thay đổi IG (τ). | κ chọn theo η = rms‖∇H‖/rms‖∇F‖ trên nút IG. Báo η thực tế trên IG và trên từng đường. ∇H tính bằng autograd, có unit test so với giải tích. |
| 8 | Mỗi biến thể chạy lại mạng. | Với IG, SPI, BlurIG (đường không phụ thuộc mô hình), F+S và F+H áp giải tích lên **chính** các đánh giá của F: cùng nút, cùng U_k (common random numbers tuyệt đối). Chỉ Guided IG chạy lại cho từng biến thể, với cùng U_k. |
| 9 | Ngoài phạm vi pilot. | Hoãn: ROAD, sanity check theo tầng, bẫy dạng cong, split xác nhận, sweep lớn. |

Không train gì. Mô hình ResNet-50 `IMAGENET1K_V2`, logit lớp dự đoán, baseline đen, cùng tiền xử lý và tập ảnh như trước.
"""


def sec_header(out, meta):
    L = ["# ATTR_PROBE_PILOT_REPORT\n"]
    if not meta:
        L.append("_(thiếu meta.json)_\n")
        return L
    for st in ("conv", "main"):
        m = meta.get(st)
        if not m:
            continue
        L.append(f"- **{st}**: {len(m['images'])} ảnh, {m['minutes']:.1f} phút (ước tính trước khi chạy "
                 f"{m['estimate'][st + '_min']:.1f}), GPU `{m['gpu']}`, torch {m['torch']}, "
                 f"{m['bench']['per_eval'] * 1e3:.2f} ms/eval, config sha256 `{m['digest'][:12]}`"
                 + (" — **CHẾ ĐỘ DEBUG, số liệu không dùng để kết luận**" if m.get("debug") else ""))
    L.append("")
    return L


def sec_tests():
    for p in ("tests_report.txt",):
        if os.path.exists(p):
            with open(p) as fh:
                txt = fh.read().strip().splitlines()
            return ["**Unit test** (`tests_report.txt`, 30 dòng cuối):\n", "```", *txt[-30:], "```", ""]
    return ["**Unit test**: chưa có `tests_report.txt` (chạy `bash scripts/run.sh test`).\n"]


def sec_B(out, primary):
    L = ["## B. Tính đúng số học\n"]
    cv = _read(out, "conv")
    if cv is not None and len(cv):
        Nmax = cv.N.max()
        L.append(f"### B1. Hội tụ theo N (stage conv, {cv.image.nunique()} ảnh; SPI {cv.real.nunique()} "
                 f"realization/ảnh, các lưới lồng nhau trên cùng một đường DP)\n")
        L.append("`resid_rel` = (ΣA − D)/|D|. `dist_ref` = ‖A_N − A_ref‖/‖A_ref‖, ref = cùng đường ở N_max "
                 f"(= {Nmax}), canon_rand cho mọi canon_*, cùng quy tắc cho IDGI. Với `canon_rand`, dist_ref ở N nhỏ "
                 "gồm cả nhiễu cầu phương của chính ref.\n")
        g = cv.assign(a=cv.resid_rel.abs()).groupby(["path", "rule", "N"]).agg(
            med_abs_resid=("a", "median"), mean_resid=("resid_rel", "mean"),
            se_resid=("resid_rel", lambda s: s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")),
            med_dist_ref=("dist_ref", "median"), n=("a", "size"))
        L.append(md(g, index=True))
        spi = cv[cv.path.str.startswith("spi")]
        if len(spi):
            L.append("**Chệch của SPI do gộp bước nhảy DP.** Cùng một đường DP rút ở N_max, lưới thô là cộng gộp số gia. "
                     "Nếu `canon_rand` hội tụ không chệch thì mean_resid ≈ 0 ở mọi N (trong ±2·SE); "
                     "`canon_left` lệch có hệ thống, và lệch giảm khi N tăng.\n")
            t = spi[spi.rule.str.startswith("canon")].groupby(["rule", "N"]).agg(
                mean_resid=("resid_rel", "mean"),
                se=("resid_rel", lambda s: s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float("nan")),
                med_dist_ref=("dist_ref", "median"))
            t["|mean|/SE"] = (t.mean_resid.abs() / t.se)
            L.append(md(t, index=True))
    mp = _read(out, "maps")
    if mp is not None and len(mp):
        L.append("### B2. Residual ở stage main (cấu hình chính)\n")
        t = mp.groupby(["path", "rule"]).agg(med_abs_resid=("abs_resid_rel", "median"),
                                             mean_resid=("resid_rel", "mean"),
                                             max_abs_resid=("abs_resid_rel", "max"))
        if "n_dropped" in mp:
            t["mean_n_dropped"] = mp.groupby(["path", "rule"])["n_dropped"].mean()
            t["max_dropped_dF_rel"] = mp.groupby(["path", "rule"])["dropped_dF_rel"].apply(lambda s: s.abs().max())
        L.append(md(t, index=True))
        L.append("IDGI có completeness theo cấu tạo trên **chính các nút** (Σ_k ΔF_k = D). Residual ≈ 0 của IDGI "
                 "**không** chứng tỏ nó chính xác hơn canonical: nó chỉ phân phối lại D, còn canonical ước lượng một "
                 "tích phân đường.\n")
    sig, tr = _read(out, "signal"), _read(out, "trap")
    L.append("### B3. Kiểm code bằng đối chứng giải tích\n")
    if sig is not None and len(sig):
        c = sig[sig.rule.str.startswith("canon") & (sig.path != "guided")]
        t = c.groupby(["path", "rule"]).agg(max_error=("error", "max"), med_retention=("retention", "median"))
        L.append("Tín hiệu tuyến tính trên đường không phụ thuộc mô hình: canonical phải cho đúng A* "
                 "(gradient của S là hằng nên tổng các đoạn telescoping, mọi cầu phương đều chính xác). "
                 "Kỳ vọng max_error ~1e-6, retention = 1.\n")
        L.append(md(t, index=True))
    if tr is not None and "ig_formula_err" in tr:
        t = tr.dropna(subset=["ig_formula_err"]).groupby(["rule"]).agg(max_ig_formula_err=("ig_formula_err", "max"))
        L.append("Bẫy trên IG: ΔA = (κ/2)·u⊙Δx theo giải tích (Σ_k sin²(πk/N)/N = ½ chính xác với N ≥ 2 ở nút trái; "
                 "với cầu phương rand là xấp xỉ có nhiễu O(1/√N)).\n")
        L.append(md(t, index=True))
    gp = _read(out, "guided_parts")
    if gp is not None and len(gp):
        L.append(f"Guided IG: max end_gap (khoảng cách nút cuối tới x) = {fmt(gp.end_gap.max())}.\n")
    return L


def paired(mp, metric, a, b):
    """a, b: (path, rule). Hiệu ghép cặp theo ảnh."""
    x = mp[(mp.path == a[0]) & (mp.rule == a[1])].set_index("image")[metric]
    y = mp[(mp.path == b[0]) & (mp.rule == b[1])].set_index("image")[metric]
    d = (x - y).dropna()
    return boot_ci(d.values)


def sec_C(out, primary):
    import pandas as pd
    L = ["## C. Đường × quy tắc (P1)\n"]
    mp = _read(out, "maps")
    if mp is None or not len(mp):
        return L + ["_(không có maps.csv)_\n"], {}
    n = mp.image.nunique()
    metrics = [m for m in ["iou", "energy_mask", "insdel_blur", "insdel_black", "abs_resid_rel", "cancel_frac", "l2"]
               if m in mp]
    L.append(f"{n} ảnh. Trung bình theo ảnh (SPI: trung bình K đường; năng lượng và L2 khử nhiễu MC). "
             f"`cancel_frac` = 1 − |ΣA|/‖A‖₁ ∈ [0,1] (có chặn; với SPI ‖A‖₁ còn bị nhiễu MC thổi phồng).\n")
    L.append(md(mp.groupby(["path", "rule"])[metrics].mean(), index=True))
    L.append("**BlurIG — baseline và completeness riêng.** D_blur = f(x) − f(blur_σmax(x)), khác D_black = f(x) − f(đen).\n")
    im = _read(out, "images")
    if im is not None:
        L.append(md(im[["image", "cls", "D_black", "D_blur", "mask_area"]]))
    paths = list(dict.fromkeys(mp.path))
    rows = []
    for p in paths:
        for r in [x for x in ["idgi", "idgi_raw", "idgi_rand", "canon_left"] if x in set(mp.rule)]:
            for m in ["iou", "energy_mask", "insdel_blur"]:
                if m in mp:
                    est, lo, hi, k = paired(mp, m, (p, r), (p, primary))
                    rows.append({"path": p, "rule": f"{r} − {primary}", "metric": m, "mean_diff": est, "lo95": lo,
                                 "hi95": hi, "n": k})
    L.append(f"**Hiệu ứng quy tắc trong từng đường** (ghép cặp theo ảnh, bootstrap 95%, so với `{primary}`):\n")
    L.append(md(pd.DataFrame(rows)))
    rows = []
    for r in [primary, "idgi"]:
        for p in paths:
            if p == "ig":
                continue
            for m in ["iou", "energy_mask", "insdel_blur"]:
                if m in mp:
                    est, lo, hi, k = paired(mp, m, (p, r), ("ig", r))
                    rows.append({"rule": r, "path": f"{p} − ig", "metric": m, "mean_diff": est, "lo95": lo,
                                 "hi95": hi, "n": k})
    L.append("**Hiệu ứng đường trong từng quy tắc** (so với IG cùng quy tắc):\n")
    L.append(md(pd.DataFrame(rows)))
    pr = _read(out, "pairs")
    if pr is not None and len(pr):
        keep = pr.a_rule.isin([primary, "idgi"]) & pr.b_rule.isin([primary, "idgi"])
        t = pr[keep].groupby(["a_path", "a_rule", "b_path", "b_rule"])[["cos", "spearman", "top5"]].mean()
        L.append("**Tương đồng giữa các bản đồ** (cosine có dấu khử nhiễu, Spearman trên Σ_c|A|, chồng top-5%):\n")
        L.append(md(t, index=True))
    dg = _read(out, "idgi_diag")
    extra = {}
    if dg is not None and len(dg):
        agg = dg.groupby(["image", "path"]).agg(P_M=("P_M", "mean"), area=("area", "first"),
                                                entropy_w=("entropy_w", "mean"), top1_w=("top1_w", "mean"),
                                                q_sd_w=("q_sd_w", "mean"), dF_pos=("dF_pos", "mean"),
                                                dF_neg=("dF_neg", "mean")).reset_index()
        agg["P_M_minus_area"] = agg.P_M - agg.area
        L.append("### C2. Chẩn đoán IDGI: g² tập trung ở đâu\n")
        L.append("q_k(M) = Σ_{i∈M} g²/Σ g² tại nút k (M = mask vật). P_M = Σ_k |ΔF_k| q_k / Σ_k |ΔF_k|: phần ΔF mà IDGI "
                 "đổ vào mask. Nếu g² phân bố đều thì P_M = diện tích mask. `entropy_w` = entropy của g²/Σg² "
                 "chia log(số phần tử) (1 = đều), `top1_w` = khối lượng của 1% phần tử lớn nhất; cả hai lấy trung bình "
                 "theo trọng số |ΔF_k|. `q_sd_w` = độ lệch chuẩn của q_k theo trọng số |ΔF_k|. "
                 "`dF_pos`/`dF_neg` = tổng ΔF_k dương/âm (SPI: trung bình theo đường).\n")
        L.append(md(agg.groupby("path")[["P_M", "area", "P_M_minus_area", "entropy_w", "top1_w", "q_sd_w",
                                         "dF_pos", "dF_neg"]].mean(), index=True))
        piv = mp[mp.rule.isin([primary, "idgi"])].pivot_table(index=["image", "path"], columns="rule",
                                                              values="iou").reset_index()
        if primary in piv and "idgi" in piv:
            piv["dIoU"] = piv["idgi"] - piv[primary]
            m = agg.merge(piv, on=["image", "path"])
            est, lo, hi, k = boot_spearman(m.P_M_minus_area, m.dIoU)
            extra["conc"] = (est, lo, hi, k)
            est2, lo2, hi2, k2 = boot_spearman(m.top1_w, m.dIoU)
            extra["conc_top1"] = (est2, lo2, hi2, k2)
            L.append(f"Spearman(P_M − diện tích, ΔIoU idgi−{primary}) trên các cặp ảnh×đường: "
                     f"**{fmt(est)}** [{fmt(lo)}, {fmt(hi)}], n={k}. "
                     f"Spearman(top1_w, ΔIoU) = {fmt(est2)} [{fmt(lo2)}, {fmt(hi2)}].\n")
            L.append(md(m.groupby("path")[["P_M_minus_area", "top1_w", "dIoU"]].mean(), index=True))
    return L, extra


def sec_D(out, primary):
    import pandas as pd
    L = ["## D. Tín hiệu cộng tính đã biết (P2)\n"]
    sg = _read(out, "signal")
    if sg is None or not len(sg):
        return L + ["_(không có signal.csv)_\n"], {}
    L.append(f"{sg.image.nunique()} ảnh. S(x) = a·Σ_R(x_i − x′_i), tổng đóng góp đúng B = ±B_frac·|D|. "
             "retention = Σ_R ΔA / ΣA*, error = ‖ΔA − A*‖/‖A*‖, `leak_l1` = Σ_{∉R}|ΔA|/Σ|ΔA|, "
             "`leak_energy` = phần năng lượng (khử nhiễu) ngoài R, `credit_R_total` = Σ_R A(F+S). "
             "ΔA = A(F+S) − A(F), với SPI là cùng đường (common random numbers).\n")
    t = sg.groupby(["path", "rule", "shape", "n_elements", "loc", "sign"]).agg(
        retention=("retention", "median"), error=("error", "median"), leak_l1=("leak_l1", "median"),
        leak_energy=("leak_energy", "median"), sign_ok=("sign_ok", "mean"), n=("image", "size"))
    keep = t.reset_index()
    keep = keep[keep.rule.isin([primary, "idgi", "idgi_rand", "idgi_raw"])]
    L.append(md(keep))
    out_ = {}
    idg = sg[(sg.rule == "idgi") & sg.path.isin(["ig", "spi10"]) & (sg["shape"] == "square")]
    if "idgi_pred_toy" in idg and len(idg):
        tt = idg.groupby(["path", "n_elements"]).agg(retention=("retention", "median"),
                                                     toy_pred=("idgi_pred_toy", "median"))
        L.append("**IDGI so với xấp xỉ toy (Test D).** toy_pred = Σ_k a²n/(‖g_k‖²+a²n)·ΔF̃_k / ΣA* (bỏ tương tác "
                 "⟨g, R⟩ và khối lượng g² sẵn có trên R). n = số phần tử của R.\n")
        L.append(md(tt, index=True))
        out_["toy_rho"] = boot_spearman(idg.idgi_pred_toy, idg.retention)
        out_["toy_absdiff"] = float((idg.idgi_pred_toy - idg.retention).abs().median())
        L.append(f"Spearman(toy_pred, retention) = {fmt(out_['toy_rho'][0])} [{fmt(out_['toy_rho'][1])}, "
                 f"{fmt(out_['toy_rho'][2])}]; median |toy_pred − retention| = {fmt(out_['toy_absdiff'])}.\n")
    sizes = sorted(idg.n_elements.unique()) if len(idg) else []
    if sizes:
        med = idg[idg.sign > 0].groupby("n_elements").retention.median()
        out_["idgi_ret_by_size"] = med.to_dict()
    gp = _read(out, "guided_parts")
    if gp is not None and len(gp):
        g = gp[gp.kind == "signal"].copy()
        if len(g):
            g["path_share"] = g.pathF_norm / (g.pathF_norm + g.compnew_norm)
            g["fixed_share"] = g.fixed_norm / (g.fixed_norm + g.pathFc_norm)
            L.append("**Guided IG — phân rã.** canonical: ΔA = A(S;γ′) [`compnew`] + (A(F;γ′) − A(F;γ)) [`pathF`]. "
                     "IDGI: ΔA = (I(F+S;γ) − I(F;γ)) [`fixed`, đường cố định] + (I(F+S;γ′) − I(F+S;γ)) [`pathFc`, đổi đường]. "
                     "`path_share` = ‖pathF‖/(‖pathF‖+‖compnew‖). `path_shift_rel` = ‖X′ − X‖/‖X − x′‖ trên các nút.\n")
            t = g.groupby(["rule", "n_elements", "loc"]).agg(
                tot_R=("tot_R", "median"), compnew_R=("compnew_R", "median"), pathF_R=("pathF_R", "median"),
                compnew_norm=("compnew_norm", "median"), pathF_norm=("pathF_norm", "median"),
                path_share=("path_share", "median"), fixed_R=("fixed_R", "median"), pathFc_R=("pathFc_R", "median"),
                path_shift_rel=("path_shift_rel", "median"))
            L.append(md(t, index=True))
            out_["guided_path_share_signal"] = float(g[g.rule.str.startswith("canon")].path_share.median())
    sk = _read(out, "skipped")
    if sk is not None and len(sk):
        L.append(f"Biến thể bị bỏ (không tìm được vùng): {len(sk)}\n")
        L.append(md(sk))
    return L, out_


def sec_E(out, primary, cfg):
    L = ["## E. Bẫy gradient (P3)\n"]
    tr = _read(out, "trap")
    if tr is None or not len(tr):
        return L + ["_(không có trap.csv)_\n"], {}
    L.append(f"{tr.image.nunique()} ảnh. H(x) = κ sin²(πs)⟨u,w⟩exp(−‖w‖²/(2r²)), r = ρ‖Δx‖, u ⟂ Δx. Trên đường IG "
             "H = 0 nhưng ∇H = κ sin²(πs)u. κ chọn sao cho η = rms‖∇H‖/rms‖∇F‖ trên nút IG đạt giá trị mục tiêu. "
             f"`rel_ig` = ‖ΔA‖/‖ΔA_IG,{primary}‖ (cùng bẫy), `rel_A` = ‖ΔA‖/‖A(F)‖ (cùng đường và quy tắc; SPI khử "
             "nhiễu MC), `eta_path` = η đo trên nút của chính đường đó. `pred_att` = a(ρ) = 2∫sin²(πt)exp(−h t(1−t)/(2ρ²))dt, "
             "h = 1/(α+1): **xấp xỉ** bậc nhất cho SPI canonical, bỏ đạo hàm của thừa số Gauss.\n")
    cols = ["dA_norm", "rel_ig", "rel_A", "eta_path"] + [c for c in ["pred_att", "ratio_to_pred"] if c in tr]
    t = tr[tr.rule.isin([primary, "idgi", "idgi_rand", "canon_left"])].groupby(
        ["u", "eta", "rho", "path", "rule"])[cols].median()
    L.append(md(t, index=True))
    out_ = {}
    if "eta_ig" in tr:
        L.append("η thực tế trên nút IG (phải trùng mục tiêu): " +
                 ", ".join(f"η={e:g}: {fmt(v)}" for e, v in tr.groupby("eta").eta_ig.median().items()) + "\n")
    to = _read(out, "trap_output")
    if to is not None and len(to):
        L.append("**Bẫy có đổi hành vi mô hình không?** `H_at_x` = H tại chính ảnh (phải ≈ 0). Nhiễu x + N(0, σ²): "
                 "`H_noise_rel_D` = mean|H|/|D|, `H_noise_rel_sdF` = mean|H| / độ lệch chuẩn của logit đích dưới cùng nhiễu, "
                 "`flip_rate` = tỉ lệ đổi lớp top-1 khi cộng H vào logit đích. `H_other_max_abs` = max|H| trên ảnh khác. "
                 "`H_nodes_max_rel_<đường>` = max|H|/|D| trên các nút của đường đó (Guided: đường của F gốc chạy lại với F+H).\n")
        c = [x for x in to.columns if x.startswith("H_") or x == "flip_rate"]
        L.append(md(to.groupby(["u", "eta", "rho"])[c].median(), index=True))
    gp = _read(out, "guided_parts")
    if gp is not None and len(gp):
        g = gp[gp.kind == "trap"].copy()
        if len(g):
            g["path_share"] = g.pathF_norm / (g.pathF_norm + g.compnew_norm)
            g["fixed_share"] = g.fixed_norm / (g.fixed_norm + g.pathFc_norm)
            L.append("**Guided IG dưới bẫy — phân rã** (như mục D):\n")
            L.append(md(g.groupby(["rule", "u", "eta", "rho"])[["tot_norm", "compnew_norm", "pathF_norm", "path_share",
                                                                 "fixed_norm", "pathFc_norm", "path_shift_rel"]].median(),
                        index=True))
            out_["guided_path_share_trap"] = float(g[g.rule.str.startswith("canon")].path_share.median())
    eb = cfg["exp_b"]
    s = tr[(tr.path.str.startswith("spi")) & (tr.rule == primary) & (tr.u == "high") & np.isclose(tr.eta, float(eb["eta_ref"]))]
    if len(s) and "ratio_to_pred" in s:
        m = s.groupby("rho").agg(rel_ig=("rel_ig", "median"), pred=("pred_att", "median"))
        out_["spi_trap"] = m
    ig = tr[(tr.path == "ig") & (tr.rule == "idgi") & (tr.u == "high") & np.isclose(tr.eta, float(eb["eta_ref"]))]
    if len(ig):
        out_["idgi_ig_rel"] = ig.groupby("rho").rel_ig.median()
    return L, out_


def sec_F(out, primary):
    L = ["## F. SPI với K = 1 đường\n"]
    sk = _read(out, "spi_k")
    if sk is None or not len(sk):
        return L + ["_(không có spi_k.csv)_\n"], {}
    L.append("Không hiệu chỉnh MC cho K=1 (và K=8): metric trên bản đồ thô. Hàng `debiased=True` là K đầy đủ có khử nhiễu "
             "(năng lượng, L2). Phân phối theo mọi (ảnh, đường/nhóm).\n")
    t = sk.groupby(["rule", "K", "debiased"]).agg(
        iou_med=("iou", "median"), iou_q10=("iou", q10), iou_q90=("iou", q90),
        E_med=("energy_mask", "median"), E_q10=("energy_mask", q10), E_q90=("energy_mask", q90),
        l2_med=("l2", "median"), l2_q10=("l2", q10), l2_q90=("l2", q90),
        resid_med=("resid_rel", "median"), n=("iou", "size"))
    L.append(md(t, index=True))
    out_ = {}
    for r in [primary, "idgi"]:
        a = sk[(sk.rule == r) & (sk.K == 1)].iou.median()
        b = sk[(sk.rule == r) & (sk.K == sk.K.max()) & (~sk.debiased)].iou.median()
        out_[r] = (a, b)
    return L, out_


# ─────────────────────────────────────────────────────────────────────────────
# G — phán quyết theo luật
# ─────────────────────────────────────────────────────────────────────────────

def verdicts(out, primary, cfg, xC, xD, xE, xF):
    import pandas as pd
    V = []
    mp = _read(out, "maps")

    # G1 — SPI + IDGI tốt nhất?
    if mp is not None and len(mp):
        cand = mp[mp.rule.isin([primary, "idgi"])]
        mets = [m for m in ["iou", "energy_mask", "insdel_blur"] if m in mp]
        best_hits, worse_hits, det = 0, 0, []
        for m in mets:
            means = cand.groupby(["path", "rule"])[m].mean().sort_values(ascending=False)
            top = means.index[0]
            others = [k for k in means.index if k != ("spi10", "idgi")]
            if ("spi10", "idgi") not in means.index:
                continue
            rival = others[0]
            est, lo, hi, _ = paired(mp, m, ("spi10", "idgi"), rival)
            det.append(f"{m}: hạng 1 = {top[0]}+{top[1]}; SPI+IDGI − {rival[0]}+{rival[1]} = {fmt(est)} [{fmt(lo)}, {fmt(hi)}]")
            if top == ("spi10", "idgi") and lo > 0:
                best_hits += 1
            if hi < 0:
                worse_hits += 1
        v = "CONFIRMED" if best_hits >= 2 else ("REFUTED" if worse_hits >= 2 else "INCONCLUSIVE")
        V.append(("G1", f"SPI+IDGI tốt nhất sau khi sửa estimator (canonical = `{primary}`)", v,
                  "CONFIRMED nếu SPI+IDGI đứng đầu và CI ghép cặp với đối thủ gần nhất > 0 ở ≥2/3 metric "
                  "(IoU, E_mask, insdel_blur); REFUTED nếu CI < 0 (thua đối thủ) ở ≥2/3.", "; ".join(det)))
        # G1b — quy tắc IDGI cải thiện SPI nhưng không cải thiện IG
        e1 = paired(mp, "iou", ("spi10", "idgi"), ("spi10", primary))
        e2 = paired(mp, "iou", ("ig", "idgi"), ("ig", primary))
        v = "CONFIRMED" if (e1[1] > 0 and e2[2] < 0) else ("REFUTED" if (e1[2] < 0 or e2[1] > 0) else "INCONCLUSIVE")
        V.append(("G1b", "IDGI tăng IoU trên SPI nhưng giảm trên IG (tương tác đường × quy tắc)", v,
                  "CONFIRMED nếu ΔIoU(idgi−canon) trên SPI có CI > 0 và trên IG có CI < 0.",
                  f"SPI: {fmt(e1[0])} [{fmt(e1[1])}, {fmt(e1[2])}]; IG: {fmt(e2[0])} [{fmt(e2[1])}, {fmt(e2[2])}]"))
        # G8 — residual canonical
        a = mp[(mp.path == "spi10") & (mp.rule == "canon_left")].abs_resid_rel.median()
        b = mp[(mp.path == "spi10") & (mp.rule == primary)].abs_resid_rel.median()
        if primary != "canon_left":
            v = "CONFIRMED" if b < a / 3 else ("REFUTED" if b >= a else "INCONCLUSIVE")
            V.append(("G8", f"Residual lớn của SPI canonical là do Riemann trái; `{primary}` sửa được", v,
                      "CONFIRMED nếu median |resid| của canon_rand < 1/3 của canon_left trên SPI; REFUTED nếu không nhỏ hơn.",
                      f"canon_left {fmt(a)}, {primary} {fmt(b)}"))

    # G2/G3 — IDGI mất tín hiệu phân tán, khớp toy
    if "idgi_ret_by_size" in xD and xD["idgi_ret_by_size"]:
        d = xD["idgi_ret_by_size"]
        ks = sorted(d)
        small, large = d[ks[0]], d[ks[-1]]
        mono = all(d[ks[i]] >= d[ks[i + 1]] for i in range(len(ks) - 1))
        v = "CONFIRMED" if (large < 0.3 and large < 0.5 * small and mono) else ("REFUTED" if large >= 0.5 else "INCONCLUSIVE")
        V.append(("G2", "IDGI mất phần lớn tín hiệu cộng tính khi tín hiệu trải trên nhiều phần tử", v,
                  "Ô vuông dương, IG và SPI gộp. CONFIRMED nếu median retention giảm đơn điệu theo số phần tử, "
                  "ở cỡ lớn nhất < 0.3 và < ½ cỡ nhỏ nhất; REFUTED nếu cỡ lớn nhất ≥ 0.5.",
                  ", ".join(f"n={k}: {fmt(d[k])}" for k in ks)))
    if "toy_rho" in xD:
        r, lo, hi, n = xD["toy_rho"]
        ad = xD["toy_absdiff"]
        v = "CONFIRMED" if (lo > 0.7 and ad < 0.15) else ("REFUTED" if (hi < 0.3 or ad > 0.5) else "INCONCLUSIVE")
        V.append(("G3", "Mức mất tín hiệu của IDGI khớp xấp xỉ toy Test D", v,
                  "CONFIRMED nếu Spearman(toy, thực) có CI dưới > 0.7 và median |toy − thực| < 0.15; "
                  "REFUTED nếu CI trên < 0.3 hoặc median |hiệu| > 0.5.",
                  f"Spearman {fmt(r)} [{fmt(lo)}, {fmt(hi)}], n={n}; median |hiệu| {fmt(ad)}"))

    # G4 — phụ thuộc độ tập trung g²
    if "conc" in xC:
        r, lo, hi, n = xC["conc"]
        v = "CONFIRMED" if lo > 0 else ("REFUTED" if hi < 0 else "INCONCLUSIVE")
        V.append(("G4", "Lợi ích của IDGI (ΔIoU) tăng khi g² dồn vào vật (P_M − diện tích mask lớn)", v,
                  "Spearman trên các cặp ảnh×đường, bootstrap 95%. CONFIRMED nếu CI > 0; REFUTED nếu CI < 0.",
                  f"{fmt(r)} [{fmt(lo)}, {fmt(hi)}], n={n}"))

    # G5 — Guided: đổi đường
    shares = [xD.get("guided_path_share_signal"), xE.get("guided_path_share_trap")]
    shares = [s for s in shares if s is not None and np.isfinite(s)]
    if shares:
        lo, hi = min(shares), max(shares)
        v = "CONFIRMED" if lo > 0.5 else ("REFUTED" if hi < 0.2 else "INCONCLUSIVE")
        V.append(("G5", "Thay đổi của Guided IG (canonical) dưới S/H chủ yếu do đường đổi, không do thành phần", v,
                  "path_share = ‖A(F;γ′)−A(F;γ)‖/(‖·‖+‖A(c;γ′)‖), median. CONFIRMED nếu > 0.5 ở cả tín hiệu và bẫy; "
                  "REFUTED nếu < 0.2 ở cả hai.",
                  f"tín hiệu {fmt(xD.get('guided_path_share_signal'))}, bẫy {fmt(xE.get('guided_path_share_trap'))}"))

    # G6 — SPI theo dự đoán a(ρ)
    if "spi_trap" in xE:
        m = xE["spi_trap"]
        rat = (m.rel_ig / m.pred).replace([np.inf, -np.inf], np.nan).dropna()
        mono = all(m.rel_ig.values[i] <= m.rel_ig.values[i + 1] * 1.1 for i in range(len(m) - 1))
        dev = float(np.nanmax(np.abs(np.log(rat)))) if len(rat) else float("nan")
        v = "CONFIRMED" if (mono and dev < math.log(2)) else ("REFUTED" if not mono else "INCONCLUSIVE")
        V.append(("G6", "SPI tránh bẫy hẹp, độ nhạy tăng theo ρ đúng cỡ dự đoán a(ρ)", v,
                  "SPI canonical, u cao tần, η tham chiếu. CONFIRMED nếu rel_ig tăng theo ρ (dung sai 10%) và "
                  "|log(thực/dự đoán)| < log 2 ở mọi ρ; REFUTED nếu không tăng theo ρ.",
                  "; ".join(f"ρ={k:g}: {fmt(a)} (dự đoán {fmt(b)})" for k, a, b in zip(m.index, m.rel_ig, m.pred))))
    if "idgi_ig_rel" in xE:
        s = xE["idgi_ig_rel"]
        v = "CONFIRMED" if s.max() < 0.1 else ("REFUTED" if s.min() > 0.5 else "INCONCLUSIVE")
        V.append(("G7", "IDGI trên đường IG gần như miễn nhiễm bẫy", v,
                  "‖ΔA_IDGI‖/‖ΔA_IG,canon‖ (median theo ρ, η tham chiếu). CONFIRMED nếu < 0.1 ở mọi ρ; REFUTED nếu > 0.5 ở mọi ρ.",
                  ", ".join(f"ρ={k:g}: {fmt(v_)}" for k, v_ in s.items())))

    # G9 — K=1
    if primary in xF:
        a, b = xF[primary]
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            v = "CONFIRMED" if a >= 0.8 * b else ("REFUTED" if a < 0.5 * b else "INCONCLUSIVE")
            ai, bi = xF.get("idgi", (float("nan"), float("nan")))
            V.append(("G9", "Một đường SPI (K=1) đã cho bản đồ định vị gần bằng K đầy đủ", v,
                      "Median IoU, canonical. CONFIRMED nếu K=1 ≥ 0.8 × K đầy đủ; REFUTED nếu < 0.5 ×.",
                      f"canonical K=1 {fmt(a)} / K đầy đủ {fmt(b)}; IDGI K=1 {fmt(ai)} / K đầy đủ {fmt(bi)}"))
    return V


def answers(V):
    d = {k: v for k, _, v, _, _ in V}

    def g(k):
        return d.get(k, "—")
    A = []
    A.append(f"1. **SPI+IDGI còn tốt nhất sau khi sửa estimator?** G1 = {g('G1')}, G1b = {g('G1b')}, G8 = {g('G8')}.")
    A.append(f"2. **IDGI mất tín hiệu phân tán đến mức nào, có khớp toán không?** G2 = {g('G2')}, G3 = {g('G3')} "
             "(số liệu ở mục D).")
    A.append(f"3. **Tác dụng của IDGI phụ thuộc vị trí tập trung của g² ra sao?** G4 = {g('G4')} (mục C2).")
    A.append(f"4. **Sai khác của Guided IG do đổi quy tắc hay đổi đường?** G5 = {g('G5')}. Trong bảng P1, Guided−IG cùng "
             "quy tắc là hiệu ứng đường thuần; hiệu idgi−canon trong Guided là hiệu ứng quy tắc thuần (mục C).")
    if g("G1b") == "CONFIRMED" and g("G2") == "CONFIRMED":
        s = ("**Đánh đổi** (theo luật): IDGI giúp định vị khi đi cùng SPI nhưng làm mất tín hiệu cộng tính phân tán, "
             "nên lợi ích của nó gắn với việc dồn credit vào nơi g² tập trung, không phải một nguyên lý chung về độ trung thực.")
    elif g("G1") == "CONFIRMED" and g("G2") == "REFUTED":
        s = "**Ứng viên nguyên lý chung** (theo luật): SPI+IDGI tốt nhất mà không mất tín hiệu phân tán."
    else:
        s = "**Chưa phân biệt được** với pilot này (các chỉ báo then chốt chưa cùng chiều hoặc INCONCLUSIVE)."
    A.append(f"5. **Nguyên lý chung, đánh đổi, hay chưa phân biệt được?** {s} Luật: Đánh đổi nếu G1b và G2 đều "
             "CONFIRMED; nguyên lý chung nếu G1 CONFIRMED và G2 REFUTED; còn lại chưa phân biệt.")
    return A


def write(out: str, cfg: dict) -> str:
    meta_p = os.path.join(out, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    primary = "canon_rand" if "rand" in cfg["quadrature"] else "canon_left"
    from .plots import make_all
    plots = make_all(out, cfg)

    L = sec_header(out, meta)
    L += ["## A. Audit code và thay đổi\n", AUDIT]
    L += sec_tests()
    L += sec_B(out, primary)
    c, xC = sec_C(out, primary)
    L += c
    d, xD = sec_D(out, primary)
    L += d
    e, xE = sec_E(out, primary, cfg)
    L += e
    f, xF = sec_F(out, primary)
    L += f
    V = verdicts(out, primary, cfg, xC, xD, xE, xF)
    L.append("## G. Phán quyết theo luật cố định\n")
    L.append("Luật đặt trước khi xem số liệu, áp máy móc. Pilot nhỏ (ít ảnh), nên CONFIRMED/REFUTED nghĩa là "
             "\"rõ trong pilot\", cần bước xác nhận trên ảnh mới.\n")
    L.append("| ID | Giả thuyết | Phán quyết | Luật | Số liệu |\n|---|---|---|---|---|")
    for k, h, v, rule, det in V:
        L.append(f"| {k} | {h} | **{v}** | {rule} | {det} |")
    L.append("")
    L.append("### Trả lời 5 câu hỏi\n")
    L += answers(V)
    L.append("")
    L.append("## Tệp\n")
    L.append("CSV trong cùng thư mục: `conv.csv`, `maps.csv`, `pairs.csv`, `idgi_diag.csv`, `idgi_steps.csv`, "
             "`spi_k.csv`, `signal.csv`, `trap.csv`, `trap_output.csv`, `guided_parts.csv`, `images.csv`, `meta.json`.\n")
    for p in plots:
        L.append(f"- `{os.path.basename(p)}`")
    L.append("")
    if meta.get("config"):
        L.append("<details><summary>config đã dùng</summary>\n\n```json\n" + json.dumps(meta["config"], indent=1, default=str)
                 + "\n```\n</details>\n")
    path = os.path.join(out, "ATTR_PROBE_PILOT_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    print(f"[report] {path}")
    return path
