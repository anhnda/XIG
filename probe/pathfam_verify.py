"""
PATH_FAMILIES — đóng băng, kiểm toán độc lập và chẩn đoán J ↔ metric ngoài (CPU, không GPU, không chọn lại).

  pathfam-verify  Tính lại TỪ DỮ LIỆU THÔ (pairs.csv) bằng mã viết riêng, không dùng budget_opt.j_long / pathfam.tests_for / holm:
                  J_rev, J_indep theo ảnh; g; p bootstrap-t (NI, superiority, TOST) + p t giải tích + độ ổn định theo 5 seed khác;
                  Holm; quyết định. So với giá trị của runner (gọi runner riêng) ⇒ bảng sai khác. Kiểm khoá: digest, sha lựa chọn,
                  cấu hình test, n test, ảnh test có đủ và không trùng dev. → infid/pathfam/PATHFAM_VERIFY.md
  pathfam-freeze  sha256 mọi artifact (protocol, khoá, lựa chọn, cấu hình, CSV dev / test, meta, báo cáo, kiểm toán, mã nguồn) + seed
                  + σ đã hiệu chuẩn → infid/pathfam/FROZEN.json; sau đó pathfam-dev / lock / test bị chặn.
  pathfam-diag    J ↔ metric ngoài: đồng thuận thứ hạng theo ảnh (Kendall τ) giữa −J / −L(μ) và từng metric ngoài, ở mức ngân sách và
                  ở mức Ē (2m đường ≈ μ); xu hướng theo exploration h; trường hợp smooth (so incumbent và so SPI α = 10 cùng h).
                  → infid/pathfam/PATHFAM_J_VS_EXTERNAL.md
"""

from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import math
import os

import numpy as np

from . import pathfam as PFM
from .budget_audit import t_sf

ROOT_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# cài đặt độc lập (không import thống kê của runner)
# ─────────────────────────────────────────────────────────────────────────────

def j_from_raw(L: np.ndarray, Ve: np.ndarray, Vo: np.ndarray, K: int, coupling: str) -> np.ndarray:
    """J theo ảnh từ cột thô: rev L + 2V_e/K (K chẵn), indep L + (V_e + V_o)/K."""
    if coupling == "rev":
        if K % 2:
            raise ValueError("rev cần K chẵn")
        return L + 2.0 * Ve / K
    return L + (Ve + Vo) / K


def my_boot_t(D: np.ndarray, n_boot: int, seed: int) -> float:
    """bootstrap-t một phía H0: E D ≤ 0 (cài lại từ đầu: vòng lặp theo khối, cùng định nghĩa thống kê)."""
    rng = np.random.default_rng(seed)
    n = D.size
    m, s = D.mean(), D.std(ddof=1)
    T = m / (s / math.sqrt(n))
    idx = rng.integers(0, n, (int(n_boot), n))
    cnt, tot = 0, 0
    for a in range(0, int(n_boot), 2000):
        X = D[idx[a:a + 2000]]
        sb = X.std(axis=1, ddof=1)
        ok = sb > 0
        Ts = (X.mean(axis=1)[ok] - m) / (sb[ok] / math.sqrt(n))
        cnt += int((Ts >= T).sum())
        tot += int(ok.sum())
    return (1 + cnt) / (tot + 1)


def my_t(D: np.ndarray) -> float:
    n = D.size
    return t_sf(D.mean() / (D.std(ddof=1) / math.sqrt(n)), n - 1)


def my_holm(ps: list) -> list:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, cur = [0.0] * m, 0.0
    for r, i in enumerate(order):
        cur = max(cur, min(1.0, (m - r) * ps[i]))
        adj[i] = cur
    return adj


def three_tests(ji: np.ndarray, jf: np.ndarray, eps: float, n_boot: int, seed: int) -> dict:
    d = ji - jf
    out = {"g": float(d.mean() / ji.mean()), "n": int(ji.size), "wins": int((d > 0).sum())}
    for nm, D in (("ni", d + eps * ji), ("sup", d), ("up", eps * ji - d)):
        out[f"p_{nm}_bt"] = my_boot_t(D, n_boot, seed)
        out[f"p_{nm}_t"] = my_t(D)
        out[f"p_{nm}_bt_seeds"] = [my_boot_t(D, n_boot, seed + 1000 * k) for k in range(1, 6)]
    out["p_tost_bt"] = max(out["p_ni_bt"], out["p_up_bt"])
    out["p_tost_t"] = max(out["p_ni_t"], out["p_up_t"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ji.size, (int(n_boot), ji.size))
    gb = (ji[idx].mean(1) - jf[idx].mean(1)) / ji[idx].mean(1)
    out["g_lo"], out["g_hi"] = float(np.quantile(gb, 0.025)), float(np.quantile(gb, 0.975))
    return out


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# verify
# ─────────────────────────────────────────────────────────────────────────────

def _raw_wide(cfg: dict, stage: str, B: int, coupling: str, set_name: str = "eval", dset: str = None):
    import pandas as pd
    p = pd.read_csv(os.path.join(PFM.root(cfg), stage, "pairs.csv"))
    dset = dset or PFM.pc(cfg)["sets"][0]
    q = p[(p.set == set_name) & (p.group == "all") & (p.dset == dset)]
    out = {}
    for c, g in q.groupby("config"):
        N = int(g.N.iloc[0])
        if B % N:
            continue
        K = B // N
        if coupling == "rev" and K % 2:
            continue
        out[c] = pd.Series(j_from_raw(g.L_mu.values, g.Ve.values, g.Vo.values, K, coupling), index=g.image.values)
    return pd.DataFrame(out)


def verify(cfg: dict) -> str:
    import pandas as pd
    from . import infid_core as IC
    from .report import md
    b, P = PFM.pc(cfg), PFM.pr(cfg)
    root = PFM.root(cfg)
    eps, al, nb, seed = float(P["ni_margin"]), float(P["alpha"]), int(P["bootstrap"]), int(P["seed"])
    L = ["# PATHFAM — kiểm toán độc lập từ dữ liệu thô\n",
         f"Thời điểm: {_dt.datetime.now().isoformat(timespec='seconds')}. Mã tính lại không dùng `budget_opt.j_long`, `pathfam.tests_for`, "
         "`infid_posthoc.holm`; so với giá trị runner (gọi riêng).\n"]
    # 1. khoá
    lk_p = PFM.lock_path(cfg)
    rows = []
    if os.path.exists(lk_p):
        lk = json.load(open(lk_p))
        pl_now = PFM.lock_payload(cfg)
        rows.append({"kiểm": "digest khoá = digest payload tính lại", "kết quả": lk["digest"] == IC._h(pl_now),
                     "chi tiết": f"{lk['digest'][:12]} / {IC._h(pl_now)[:12]}"})
        rows.append({"kiểm": "sha256 selection_dev.json = giá trị trong khoá", "kết quả": lk["payload"]["selection_sha256"] == sha(PFM.sel_path(cfg)), "chi tiết": ""})
        n_lock = int(lk["payload"]["n_test"])
        meta_t = os.path.join(root, "test", "meta.json")
        if os.path.exists(meta_t):
            mt = json.load(open(meta_t))
            imgs_t = mt["images"]
            rows.append({"kiểm": "số ảnh test = n trong khoá", "kết quả": len(imgs_t) == n_lock, "chi tiết": f"{len(imgs_t)} / {n_lock}"})
            md_ = json.load(open(os.path.join(root, "dev", "meta.json")))
            rows.append({"kiểm": "ảnh test rời ảnh dev", "kết quả": not (set(imgs_t) & set(md_["images"])), "chi tiết": ""})
            cf_t = sorted(PFM.cname(c) for c in mt["sc"]["configs"])
            rows.append({"kiểm": "cấu hình test = cấu hình trong khoá", "kết quả": cf_t == sorted(lk["payload"]["test_configs"]),
                         "chi tiết": f"{len(cf_t)} cấu hình"})
            rows.append({"kiểm": "test chạy SAU khi khoá (thời điểm)", "kết quả": mt["time"] >= lk.get("time", ""), "chi tiết": f"khoá {lk.get('time', '?')} / test {mt['time']}"})
    else:
        rows.append({"kiểm": "có khoá", "kết quả": False, "chi tiết": lk_p})
    L += ["## 1. Khoá và tính toàn vẹn\n", md(pd.DataFrame(rows))]
    # 2. kiểm định chính, tính lại
    sel = PFM.load_sel(cfg)
    inc = PFM.cname(b["incumbent"])
    res = []
    for B in P["budgets"]:
        W = _raw_wide(cfg, "test", int(B), "rev")
        for fam in b["primary_families"] + ["smooth"]:
            s = sel["chosen"].get(str(B), {}).get(fam)
            if not s or s["config"] not in W or inc not in W:
                continue
            j = W[[inc, s["config"]]].dropna()
            r = three_tests(j[inc].values, j[s["config"]].values, eps, nb, seed)
            rr = PFM.tests_for(j[inc].values, j[s["config"]].values, eps, nb, seed, al)          # runner
            res.append({"B": int(B), "họ": fam, "config": s["config"], "chính": fam in b["primary_families"], **r,
                        "runner_g": rr["g"], "runner_p_ni": rr["p_ni"], "runner_p_sup": rr["p_sup"], "runner_p_tost": rr["p_tost"]})
    prim = [r for r in res if r["chính"]]
    for key in ("p_ni_bt", "p_sup_bt", "p_tost_bt", "p_ni_t", "p_sup_t", "p_tost_t"):
        for r, h in zip(prim, my_holm([r[key] for r in prim])):
            r[key + "_holm"] = h
    rows = []
    for r in res:
        o = {"B": r["B"], "họ": r["họ"], "config": r["config"], "n": r["n"], "g (độc lập)": r["g"], "|g − g runner|": abs(r["g"] - r["runner_g"]),
             "g CI 95 %": f"[{r['g_lo']:.3f}, {r['g_hi']:.3f}]"}
        for k in ("ni", "sup", "tost"):
            o[f"|p_{k} − runner|"] = abs(r[f"p_{k}_bt"] - r[f"runner_p_{k}"])
        if r["chính"]:
            o.update({"p NI bt (Holm)": r["p_ni_bt_holm"], "p NI t (Holm)": r["p_ni_t_holm"],
                      "p NI bt 5 seed khác, chưa Holm (min–max)": f"{min(r['p_ni_bt_seeds']):.4f}–{max(r['p_ni_bt_seeds']):.4f}",
                      "p sup bt (Holm)": r["p_sup_bt_holm"], "p TOST bt (Holm)": r["p_tost_bt_holm"], "p TOST t (Holm)": r["p_tost_t_holm"],
                      "NI (bt)": ("không kém hơn" if r["p_ni_bt_holm"] < al else "kém hơn thực tiễn" if r["g_hi"] < -eps else "chưa kết luận"),
                      "NI (t) cùng kết luận": (r["p_ni_bt_holm"] < al) == (r["p_ni_t_holm"] < al),
                      "TOST (bt)": "tương đương" if r["p_tost_bt_holm"] < al else "chưa"})
        rows.append(o)
    L += ["## 2. Kiểm định chính tính lại từ pairs.csv\n",
          f"ε = {eps}, α = {al}, bootstrap {nb}, seed {seed}. `|… − runner|` phải ≈ 0 (cùng định nghĩa, cùng seed). Cột t và 5 seed khác "
          "cho thấy kết luận có nhạy với phương pháp / nhiễu Monte Carlo hay không.\n", md(pd.DataFrame(rows))]
    # 3. đối chiếu với báo cáo đã sinh (nếu có)
    rp = PFM.report_path(cfg)
    L.append(f"## 3. Báo cáo runner\n\n`{rp}` sha256 `{sha(rp)[:16]}…` (đối chiếu bằng mắt các cột g / p với bảng trên).\n" if os.path.exists(rp)
             else "## 3. Báo cáo runner\n\nkhông thấy.\n")
    out = os.path.join(root, "PATHFAM_VERIFY.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[pathfam-verify] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# freeze
# ─────────────────────────────────────────────────────────────────────────────

def freeze(cfg: dict) -> str:
    root = PFM.root(cfg)
    inf = os.path.dirname(root)
    files = sorted(glob.glob(os.path.join(root, "*", "*.csv")) + glob.glob(os.path.join(root, "*", "meta.json")) +
                   glob.glob(os.path.join(root, "*.json")) + glob.glob(os.path.join(root, "*.md")) + glob.glob(os.path.join(root, "*.csv")) +
                   [p for p in (PFM.report_path(cfg),) if os.path.exists(p)])
    files = [f for f in files if not f.endswith("FROZEN.json")]
    docs = sorted(glob.glob(os.path.join(ROOT_SRC, "docs", "PATH_FAMILIES_*.md")) + glob.glob(os.path.join(ROOT_SRC, "docs", "PATHFAM_*.md")))
    src = sorted(glob.glob(os.path.join(ROOT_SRC, "probe", "*.py")) + [os.path.join(ROOT_SRC, "config.yaml")])
    hs = hashlib.sha256()
    for f in src:
        hs.update(open(f, "rb").read())
    meta = {}
    for st in ("dev", "test"):
        mp = os.path.join(root, st, "meta.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            meta[st] = {"time": m.get("time"), "images": len(m.get("images", [])), "sigma": m.get("sigma"), "grad_evals": m.get("grad_evals")}
    lk = PFM.lock_path(cfg)
    rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "status": "PATHFAM Phase 1 đóng băng sau test",
           "lock_digest": json.load(open(lk))["digest"] if os.path.exists(lk) else None,
           "seeds": {"protocol": PFM.pr(cfg)["seed"], "global": cfg["seed"], "ln_calibration": {"seed": 10, "d": 20000}},
           "stages": meta, "pathfam_config_sha256": hashlib.sha256(json.dumps(PFM.pc(cfg), sort_keys=True, default=str).encode()).hexdigest(),
           "source_sha256": hs.hexdigest(), "docs": {os.path.relpath(d, ROOT_SRC): sha(d) for d in docs},
           "files": {os.path.relpath(f, inf): sha(f) for f in files}}
    p = os.path.join(root, "FROZEN.json")
    if os.path.exists(p):
        rec["previous"] = json.load(open(p))
    with open(p, "w") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print(f"[pathfam-freeze] {p}: {len(files)} file kết quả, {len(docs)} tài liệu; pathfam-dev / lock / test bị chặn")
    return p


def check_frozen(cfg: dict) -> None:
    p = os.path.join(PFM.root(cfg), "FROZEN.json")
    if os.path.exists(p) and not cfg.get("_debug"):
        raise SystemExit(f"PATHFAM đã đóng băng ({p}). Muốn chạy lại phải có protocol mới và chuyển FROZEN.json có chủ đích.")


# ─────────────────────────────────────────────────────────────────────────────
# diag: J ↔ metric ngoài
# ─────────────────────────────────────────────────────────────────────────────

def kendall(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    s, c = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b_ = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if a != 0 and b_ != 0:
                s += a * b_
                c += 1
    return s / c if c else math.nan


def diag(cfg: dict) -> str:
    import pandas as pd
    from .infid_posthoc import EXT_DIR, boot_idx, ci_of, fmt_ci
    from .report import md
    b, P = PFM.pc(cfg), PFM.pr(cfg)
    root = PFM.root(cfg)
    nb, seed, al = int(P["bootstrap"]), int(P["seed"]), float(P["alpha"])
    pairs = pd.read_csv(os.path.join(root, "test", "pairs.csv"))
    ext = pd.read_csv(os.path.join(root, "test", "ext.csv"))
    Dp = b["sets"][0]
    q = pairs[(pairs.set == "eval") & (pairs.group == "all") & (pairs.dset == Dp)].copy()
    q["V1"] = q.Ve + q.Vo
    inc = PFM.cname(b["incumbent"])
    confs = sorted(q.config.unique())
    hmap = {}
    for c in confs:
        g = q[q.config == c].iloc[0]
        cf = {"family": g.family, "N": int(g.N)}
        if g.family == "smooth":
            cf["delta"] = float(g.delta)
        else:
            cf["alpha"] = float(g.alpha)
        hmap[c] = PFM.h_of(PFM.norm_conf(cf))
    L = ["# PATHFAM — J ↔ metric ngoài (chẩn đoán, không đổi objective, không chọn lại)\n",
         "## 0. Sự kiện từ mã (không cần dữ liệu)\n",
         "- **Infidelity** dùng attribution CÓ DẤU, cộng qua mọi kênh trong từng ô che (p_Sᵀ A), chuẩn hoá theo ‖ΔF_S‖² của CÙNG ảnh và "
         "cùng bộ ô cho mọi cấu hình ⇒ so sánh giữa họ đường không bị lệch chuẩn hoá.\n"
         "- **Metric ngoài** (InsDel, ROAD, IoU, pointing, AUROC/AP mask) xếp hạng pixel theo Σ_c |A_c| (`metrics.amap`, `insdel_full(signed=False)`): "
         "bỏ dấu, gộp trị tuyệt đối qua kênh, bất biến với phép nhân dương ⇒ **chỉ phụ thuộc thứ hạng độ lớn**, không phụ thuộc thang đo, còn "
         "infidelity phụ thuộc giá trị có dấu (cả triệt tiêu dương/âm).\n"
         "- **Completeness**: với tích phân đường chính xác, Σ_i A_i = F(x) − F(x′) cho MỌI đường, mọi họ (định lý gradient; kiểm ở "
         "`pathfam-plan`: sai số ≤ 2·10⁻¹⁵ với F tuyến tính). Quy tắc `rand` chỉ không chệch ⇒ tổng attribution bằng ΔF theo kỳ vọng, "
         "giống nhau giữa các họ. Thang đo tổng vì vậy không thể là nguồn khác biệt giữa họ; khác biệt nằm ở **phân bố khối lượng có dấu** "
         "giữa các pixel.\n"
         "- Hệ quả kiểm được: hai bản đồ có cùng p_Sᵀ A trên mọi ô (cùng infidelity) vẫn có thể có thứ hạng Σ|A| khác hẳn (triệt tiêu trong "
         "ô). Mâu thuẫn J ↔ metric ngoài vì vậy KHÔNG mâu thuẫn logic; nó đo hai thuộc tính khác nhau của cùng bản đồ.\n"]
    # 1. bảng tổng hợp theo cấu hình
    piv = {}
    for key in ("L_mu", "V1"):
        piv[key] = q.pivot_table(index="image", columns="config", values=key, aggfunc="first")
    ext = ext.assign(key=ext.config + "|" + ext.estimator)
    rows = []
    for c in confs:
        r = {"config": c, "h": hmap[c], "L(μ)": float(piv["L_mu"][c].mean()), "V₁": float(piv["V1"][c].mean())}
        for B in P["budgets"]:
            N = int(q[q.config == c].N.iloc[0])
            if B % N == 0 and (B // N) % 2 == 0:
                K = B // N
                r[f"J_rev B{B}"] = float((piv["L_mu"][c] + 2 * q[q.config == c].set_index("image").Ve / K).mean())
                e = ext[ext.key == f"{c}|rev_B{B}_K{K}"]
                for met in ("insdel_blur", "iou", "auroc_mask"):
                    if met in e:
                        r[f"{met} B{B}"] = float(e[met].mean())
        e = ext[ext.key == f"{c}|mean_all"]
        for met in ("insdel_blur", "iou", "auroc_mask", "road"):
            if met in e:
                r[f"{met} Ē"] = float(e[met].mean())
        rows.append(r)
    T = pd.DataFrame(rows).sort_values("h", ascending=False)
    L += ["## 1. Tổng hợp theo cấu hình (test, bộ ô đánh giá, D chính)\n", md(T)]
    # 2. đồng thuận thứ hạng theo ảnh
    L.append("## 2. Đồng thuận thứ hạng giữa các cấu hình, theo từng ảnh (Kendall τ, TB qua ảnh, CI bootstrap theo ảnh)\n")
    L.append("τ > 0: cấu hình có J (hay L(μ)) tốt hơn cũng có metric ngoài tốt hơn trên cùng ảnh; τ < 0: mâu thuẫn có hệ thống. Mức μ: "
             "−L(μ) so với metric trên Ē của 2m đường (cả hai là đại lượng của attribution kỳ vọng ⇒ loại phương sai K). Mô tả: CI chưa "
             "hiệu chỉnh đa kiểm định (tối đa 24 dòng).\n")
    rows = []
    for level, Jw, est in [(f"B{B}", None, None) for B in P["budgets"]] + [("μ (Ē)", "L_mu", "mean_all")]:
        for met, sgn in EXT_DIR.items():
            taus = []
            for im, gq in q.groupby("image"):
                xs, ys = [], []
                for c in confs:
                    if level.startswith("B"):
                        B = int(level[1:])
                        N = int(gq[gq.config == c].N.iloc[0]) if (gq.config == c).any() else None
                        if N is None or B % N or (B // N) % 2:
                            continue
                        K = B // N
                        rr = gq[gq.config == c].iloc[0]
                        jv = rr.L_mu + 2 * rr.Ve / K
                        ek = f"{c}|rev_B{B}_K{K}"
                    else:
                        if not (gq.config == c).any():
                            continue
                        jv = float(gq[gq.config == c].L_mu.iloc[0])
                        ek = f"{c}|mean_all"
                    e = ext[(ext.key == ek) & (ext.image == im)]
                    if not len(e) or met not in e or not np.isfinite(e[met].iloc[0]):
                        continue
                    xs.append(-jv)
                    ys.append(sgn * float(e[met].iloc[0]))
                if len(xs) >= 3:
                    taus.append(kendall(np.array(xs), np.array(ys)))
            taus = np.array([t for t in taus if np.isfinite(t)])
            if len(taus) < 3:
                continue
            lo, hi = ci_of(taus[boot_idx(len(taus), nb, seed)].mean(1), al)
            rows.append({"mức": level, "metric": met, "n ảnh": len(taus), "τ TB [CI]": fmt_ci(float(taus.mean()), lo, hi, "{:.3f}"),
                         "đọc": "mâu thuẫn" if hi < 0 else "đồng thuận" if lo > 0 else "không rõ"})
    L.append(md(pd.DataFrame(rows)))
    # 3. xu hướng theo h
    L.append("## 3. Xu hướng theo exploration h (qua TB cấu hình; mô tả)\n")
    rows = []
    for col in [c for c in T.columns if c not in ("config", "h")]:
        v = T[["h", col]].dropna()
        if len(v) >= 4:
            rows.append({"đại lượng": col, "Kendall τ với h (qua cấu hình)": kendall(v.h.values, v[col].values), "số cấu hình": len(v)})
    L.append(md(pd.DataFrame(rows)))
    # 4. smooth
    L.append("## 4. Trường hợp smooth warp\n")
    rows = []
    sm = [c for c in confs if c.startswith("smooth_")]
    refs = [inc] + [c for c in confs if c.startswith("spi_a10_")]
    for s_ in sm:
        for rf in refs:
            r = {"smooth": s_, "so với": rf, "h smooth / ref": f"{hmap[s_]:.3f} / {hmap[rf]:.3f}"}
            for lab, key in (("ΔL(μ)", "L_mu"), ("ΔV₁", "V1")):
                v = (piv[key][s_] - piv[key][rf]).dropna().values
                lo, hi = ci_of(v[boot_idx(len(v), nb, seed)].mean(1), al)
                r[f"{lab} smooth − ref [CI]"] = fmt_ci(float(v.mean()), lo, hi)
            for met, sgn in EXT_DIR.items():
                a = ext[ext.key == f"{s_}|mean_all"].set_index("image")
                c_ = ext[ext.key == f"{rf}|mean_all"].set_index("image")
                if met in a and met in c_:
                    j = pd.concat([a[met].rename("a"), c_[met].rename("b")], axis=1).dropna()
                    v = sgn * (j.a - j.b).values
                    lo, hi = ci_of(v[boot_idx(len(v), nb, seed)].mean(1), al)
                    r[f"{met} Ē (tốt +) [CI]"] = fmt_ci(float(v.mean()), lo, hi, "{:.3f}")
            rows.append(r)
    L.append("So ở mức μ (Ē của 2m đường) để loại phương sai K. smooth δ = 0,674 và SPI α = 10 có cùng D (h ≈ 0,091): khác biệt giữa hai "
             "cấu hình này là hiệu ứng **họ đường ở cùng exploration**; khác biệt smooth − incumbent trộn họ và exploration (h 0,2 so với 0,5).\n")
    L.append(md(pd.DataFrame(rows)) if rows else "_(không có smooth trên test)_\n")
    L.append("## 5. Chưa kiểm được từ dữ liệu hiện có (cần lưu thêm bản đồ ở lần chạy sau; không cần thêm gradient)\n"
             "- Tỷ lệ triệt tiêu ‖A‖₁/|ΣA| và khối lượng âm theo cấu hình; InsDel xếp hạng CÓ DẤU (`signed=True`); năng lượng trong mask "
             "có dấu và không dấu; sai số completeness thực nghiệm ΣÂ − ΔF. Các đại lượng này tính từ bản đồ đã có trong lúc chạy, nhưng runner "
             "hiện chỉ lưu metric tổng hợp.\n")
    out = os.path.join(root, "PATHFAM_J_VS_EXTERNAL.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[pathfam-diag] {out}")
    return out
