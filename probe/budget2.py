"""
BUDGET2 — vòng tối ưu ngân sách thứ hai (docs/PROTOCOL_BUDGET2.md). Ảnh mới, khoá riêng (infid/budget2/budget2_prereg.json).

Incumbent = nghiệm đã xác nhận ở BUDGET 1: SPI α = 1, N = 32, `rev`. Đối chứng phụ: SPI α = 0,1, N = 64, `rev` (nghiên cứu trước).
  Nhánh A (chính): ngân sách nhỏ B ∈ {256 (chính), 1024 (phụ)}; ứng viên SPI α ∈ {0,3; 1; 3} × N ∈ {8, 16, 32}, `rev`.
  Nhánh B (chỉ dev, có cổng): ngân sách lớn B ∈ {2048, 4096}; hier G = 14, α ∈ {0,1; 0,3}, η ∈ {0,25; 0,5; 0,75}, N = 64, `rev`;
           tham chiếu = SPI `rev` tốt nhất trên dev ở cùng B (gồm incumbent, α = 0,1 và lưới nhánh A) — chặt hơn chỉ so incumbent.
Cổng dev (đặt trước): ứng viên tốt nhất (theo TB J, bộ ô CHỌN) vào test ở B nếu giảm TB J ≥ gate_min_gain so với tham chiếu VÀ
P_boot(tốt hơn) ≥ gate_min_prob. Không ứng viên nào qua ⇒ incumbent giữ nguyên ở B đó, không cần test.
Cỡ mẫu test: tính lúc khoá từ SD của D_i trên dev (bộ ô ĐÁNH GIÁ), hiệu ứng giả định = trung điểm giữa ngưỡng δ và hiệu ứng dev
(co về ngưỡng để trừ lời nguyền người thắng), lực 0,8, một phía α; kẹp vào [n_min, n_max].
Phép kiểm test (mỗi B đã qua cổng và có trong test.budgets): H0: E[J_ref − J_new] ≤ δ·E[J_ref]  ⇔  E[D] ≤ 0,
D_i = (J_ref,i − J_new,i) − δ·J_ref,i;  p một phía bootstrap-t (chính), t (độ nhạy); Holm qua các B được kiểm.
xác nhận nếu p Holm < α; bác bỏ nếu cận trên CI 95 % của g = (J̄_ref − J̄_new)/J̄_ref < δ; còn lại chưa kết luận.

Lệnh: budget2-dev → budget2-lock → budget2-test; budget2-report dựng lại infid/INFID_BUDGET2_REPORT.md.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import time

import numpy as np

from . import budget_opt as BO
from . import config as C
from . import infid_core as IC
from . import infid_stats as ST
from .budget_audit import boot_t_p, fieller, linear_test
from .infid_posthoc import EXT_DIR, boot_idx, boot_p, ci_of, fmt_ci, holm
from .quad import Counter
from .sampling_infid import cname, norm_conf, stage_images

STAGES = ("dev", "test")
TAG = {"dev": "r2dev", "test": "r2test"}

DEFAULT2 = {
    "dir": "budget2",
    "protocol": {"rule": "rand", "M": 1, "budgets": [256, 1024, 2048, 4096], "budgets_secondary": [], "Ks": [2, 4, 8, 16, 32],
                 "check_K": [4, 16, 32], "alpha": 0.05, "bootstrap": 10000, "seed": 47, "floor_grids": [14],
                 "floor_lams": [0.0, 1.0e-3, 1.0e-2, 1.0e-1], "floor_folds": 4,
                 "branchA_budgets": [256, 1024], "primary_budget": 256, "branchB_budgets": [2048, 4096],
                 "gate_min_gain": 0.05, "gate_min_prob": 0.80, "delta_rel": 0.05, "power": 0.80, "n_min": 48, "n_max": 128},
    "sets": ["del_sq"],
    "coupling": "rev",
    "incumbent": {"family": "spi", "alpha": 1, "N": 32},
    "prior": {"family": "spi", "alpha": 0.1, "N": 64},
    "dev": {"split": "fresh", "offset": 900, "n_images": 32, "m": 32},
    "branchA": {"alphas": [0.3, 1, 3], "N": [8, 16, 32]},
    "branchB": {"enabled": True, "alphas": [0.1, 0.3], "G": [14], "eta": [0.25, 0.5, 0.75], "N": 64},
    "test": {"split": "fresh", "offset": 1000, "m": 32, "budgets": [256, 1024], "extra_configs": [],
             "sets": ["del_sq", "ins_sq", "sym_sq", "frac25", "frac50", "frac75", "del_sq_blur", "del_sp"], "external": True,
             "road": {"fracs": [0.1, 0.3, 0.5, 0.7, 0.9], "noise": 0.01, "iters": 1000, "tol": 1.0e-5},
             "insdel_steps": 50, "pointing_tol": 15, "top_frac": 0.05},
    "gpu_minutes": {"dev": None, "test": None},
}


def b2(cfg: dict) -> dict:
    return C._merge(DEFAULT2, cfg.get("budget2", {}) or {})


def view(cfg: dict) -> dict:
    """cfg nhìn như vòng 1 để dùng lại budget_opt (process, j_long, emp_table, …) với thư mục / protocol riêng."""
    v = dict(cfg)
    v["budget"] = b2(cfg)
    return v


def pr2(cfg: dict) -> dict:
    return b2(cfg)["protocol"]


def root(cfg: dict) -> str:
    return BO.broot(view(cfg))


def sel_path(cfg: dict) -> str:
    return os.path.join(root(cfg), "selection_dev.json")


def lock_path(cfg: dict) -> str:
    return os.path.join(root(cfg), "budget2_prereg.json")


def report_path(cfg: dict) -> str:
    return os.path.join(C.out_dir(cfg, "pilot"), "infid", "INFID_BUDGET2_REPORT.md")


def cand(conf: dict, coupling: str) -> str:
    return f"{cname(conf)}|{coupling}"


# ─────────────────────────────────────────────────────────────────────────────
# hàm thuần (unit test)
# ─────────────────────────────────────────────────────────────────────────────

def gate(j_ref: np.ndarray, j_ch: np.ndarray, min_gain: float, min_prob: float, n_boot: int, seed: int) -> dict:
    """Cổng dev: gain = (J̄_ref − J̄_ch)/J̄_ref; P_boot(J̄*_ch < J̄*_ref) theo cặp ảnh."""
    gain = float((j_ref.mean() - j_ch.mean()) / j_ref.mean())
    idx = boot_idx(len(j_ref), n_boot, seed)
    pb = float((j_ch[idx].mean(1) < j_ref[idx].mean(1)).mean())
    return {"gain_rel": gain, "p_better": pb, "pass": bool(gain >= min_gain and pb >= min_prob)}


def plan_n(j_ref: np.ndarray, j_ch: np.ndarray, delta: float, alpha: float, power: float) -> dict:
    """n cho phép kiểm trên D = (J_ref − J_ch) − δ·J_ref: hiệu ứng giả định g* = (g_dev + δ)/2 (co về ngưỡng)."""
    g_dev = float((j_ref.mean() - j_ch.mean()) / j_ref.mean())
    D = (j_ref - j_ch) - delta * j_ref
    sd = float(D.std(ddof=1))
    g_star = (g_dev + delta) / 2
    eff = (g_star - delta) * float(j_ref.mean())
    n = ST.n_required(sd, eff, alpha, power, one_sided=True) if eff > 0 else math.nan
    return {"g_dev": g_dev, "g_star": g_star, "sd_D": sd, "effect_D": eff, "n_required": n}


def decide_test(j_ref: np.ndarray, j_new: np.ndarray, delta: float, n_boot: int, seed: int, al: float) -> dict:
    D = (j_ref - j_new) - delta * j_ref
    n = len(D)
    idx = boot_idx(n, n_boot, seed)
    g = float((j_ref.mean() - j_new.mean()) / j_ref.mean())
    gb = (j_ref[idx].mean(1) - j_new[idx].mean(1)) / j_ref[idx].mean(1)
    lo, hi = ci_of(gb, al)
    # Fieller cho g = E[J_ref − J_new]/E[J_ref]: dùng fieller(t=j_new, b=j_ref, f=0)
    flo, fhi = fieller(j_new, j_ref, np.zeros_like(j_ref), 1 - al)
    dJ = j_ref - j_new
    dlo, dhi = ci_of(dJ[idx].mean(1), al)
    return {"n": n, "J_ref": float(j_ref.mean()), "J_new": float(j_new.mean()), "dJ": float(dJ.mean()), "dJ_lo": dlo, "dJ_hi": dhi,
            "g": g, "g_lo": lo, "g_hi": hi, "g_fieller_lo": flo, "g_fieller_hi": fhi, "D_mean": float(D.mean()),
            "p_boot_t": boot_t_p(D, n_boot, seed), "p_t": linear_test(j_new, j_ref, np.zeros_like(j_ref), delta)["p_t"],
            "wins": int((dJ > 0).sum())}


# ─────────────────────────────────────────────────────────────────────────────
# cấu hình
# ─────────────────────────────────────────────────────────────────────────────

def dev_configs(cfg: dict) -> list:
    b = b2(cfg)
    cf = [norm_conf({"family": "spi", "alpha": a, "N": n}) for a in b["branchA"]["alphas"] for n in b["branchA"]["N"]]
    cf += [norm_conf(b["incumbent"]), norm_conf(b["prior"])]
    bb = b["branchB"]
    if bb.get("enabled", True):
        cf += [norm_conf({"family": "hier", "alpha": a, "N": bb["N"], "G": G, "eta": e}) for a in bb["alphas"] for G in bb["G"]
               for e in bb["eta"]]
    seen, out = set(), []
    for c in cf:
        if cname(c) not in seen:
            seen.add(cname(c))
            out.append(c)
    return out


def load_sel(cfg: dict) -> dict:
    p = sel_path(cfg)
    if not os.path.exists(p):
        raise SystemExit(f"chưa có {p} — chạy bash scripts/run.sh budget2-dev trước")
    with open(p) as fh:
        return json.load(fh)


def tested_budgets(cfg: dict, sel: dict) -> list:
    return [int(B) for B in b2(cfg)["test"]["budgets"] if sel["budgets"].get(str(int(B)), {}).get("pass")]


def n_test(cfg: dict, sel: dict) -> dict:
    pr = pr2(cfg)
    tb = tested_budgets(cfg, sel)
    ns = [sel["budgets"][str(B)]["plan"]["n_required"] for B in tb]
    ns = [n for n in ns if n is not None and np.isfinite(n)]
    n_req = int(max(ns)) if ns else int(pr["n_min"])
    n = int(min(max(n_req, int(pr["n_min"])), int(pr["n_max"])))
    return {"n_test": n, "n_required": n_req, "underpowered": n_req > int(pr["n_max"])}


def stage_sc(cfg: dict, stage: str) -> dict:
    b = b2(cfg)
    if stage == "dev":
        return {**b["dev"], "sets": b["sets"], "configs": dev_configs(cfg), "external": False, "ig": False}
    sel = load_sel(cfg)
    tb = tested_budgets(cfg, sel)
    if not tb:
        raise SystemExit("không ngân sách nào trong budget2.test.budgets qua cổng dev — incumbent giữ nguyên, không cần test")
    cf = []
    for B in tb:
        s = sel["budgets"][str(B)]
        cf += [s["challenger_conf"], s["ref_conf"]]
    cf += [b["incumbent"], b["prior"]] + list(b["test"].get("extra_configs") or [])
    seen, out = set(), []
    for c in cf:
        c = norm_conf(c)
        if cname(c) not in seen:
            seen.add(cname(c))
            out.append(c)
    t = b["test"]
    return {**t, "n_images": n_test(cfg, sel)["n_test"], "configs": out, "ig": False, "map_budgets": tb}


def check_images(ctx, cfg: dict, stage: str, idxs: list) -> None:
    if cfg.get("_debug"):
        return
    b = b2(cfg)
    used = BO.used_infid_images(ctx, cfg)
    b1 = BO.bcfg(cfg)
    for sc in (b1["dev"], b1["test"]):
        try:
            used |= set(stage_images(ctx, sc))
        except (SystemExit, KeyError):
            pass
    other = b["test"] if stage == "dev" else b["dev"]
    try:
        o = dict(other)
        o.setdefault("n_images", int(pr2(cfg)["n_max"]))
        used |= set(stage_images(ctx, o))
    except (SystemExit, KeyError):
        pass
    inter = set(idxs) & used
    if inter:
        raise SystemExit(f"[budget2-{stage}] {len(inter)} ảnh trùng với ảnh đã dùng (infid / BUDGET 1 / giai đoạn kia) — đổi offset")


# ─────────────────────────────────────────────────────────────────────────────
# chạy
# ─────────────────────────────────────────────────────────────────────────────

def lock_payload(cfg: dict) -> dict:
    b = b2(cfg)
    sp = sel_path(cfg)
    sel = load_sel(cfg)
    D = d_cfg(cfg)
    return {"protocol": b["protocol"], "sets": b["sets"], "coupling": b["coupling"], "incumbent": b["incumbent"], "prior": b["prior"],
            "test": b["test"], "D": {k: D["sets"][k] for k in b["test"]["sets"]}, "n_sel": D.get("n_sel"), "n_eval": D.get("n_eval"),
            "selection_sha256": hashlib.sha256(open(sp, "rb").read()).hexdigest(), "tested_budgets": tested_budgets(cfg, sel),
            **n_test(cfg, sel), "model": cfg["model"], "data": cfg["data"], "seed": cfg["seed"], "v": 1}


def d_cfg(cfg: dict) -> dict:
    from .infid_e0 import icfg
    return icfg(cfg)["D"]


def check_frozen(cfg: dict) -> None:
    fz = os.path.join(root(cfg), "FROZEN.json")
    if os.path.exists(fz) and not cfg.get("_debug"):
        raise SystemExit(f"BUDGET2 đã đóng băng dev-only ({fz}); tập test được giữ lại. Muốn mở test phải có quyết định mới, ghi vào "
                         "docs/PROTOCOL_BUDGET2.md, rồi mới xoá / chuyển FROZEN.json")


def lock(cfg: dict) -> None:
    check_frozen(cfg)
    sel = load_sel(cfg)
    if not tested_budgets(cfg, sel):
        raise SystemExit("không ngân sách nào trong budget2.test.budgets qua cổng dev — incumbent giữ nguyên; không khoá / không test")
    for B in b2(cfg)["test"]["budgets"]:
        if int(B) in pr2(cfg)["branchB_budgets"] and sel["budgets"].get(str(int(B)), {}).get("pass"):
            print(f"[budget2-lock] CHÚ Ý: B = {B} (nhánh B) có trong test.budgets — chỉ để vậy nếu đã được duyệt mở test nhánh B")
    pl = lock_payload(cfg)
    if pl["underpowered"]:
        print(f"[budget2-lock] CHÚ Ý: n cần {pl['n_required']} > n_max {pr2(cfg)['n_max']} — test thiếu lực, khoá với n = {pl['n_test']}")
    ST.write_prereg(lock_path(cfg), IC._h(pl), {"payload": pl})
    print(f"[budget2-lock] ngân sách kiểm: {pl['tested_budgets']}, n test = {pl['n_test']} (cần {pl['n_required']})")


def run(ctx, stage: str) -> None:
    from .ctx import bench
    import pandas as pd
    cfg = ctx.cfg
    v = view(cfg)
    if stage not in STAGES:
        raise SystemExit(stage)
    check_frozen(cfg)
    if stage == "test":
        if not cfg["_debug"] and not os.path.exists(lock_path(cfg)):
            raise SystemExit(f"budget2-test cần khoá: bash scripts/run.sh budget2-lock (chưa thấy {lock_path(cfg)})")
        ST.require_prereg(lock_path(cfg), IC._h(lock_payload(cfg)), cfg["_debug"])
    sc = stage_sc(cfg, stage)
    out = os.path.join(root(cfg), stage)
    os.makedirs(out, exist_ok=True)
    idxs = stage_images(ctx, sc)
    check_images(ctx, cfg, stage, idxs)
    bm = bench(ctx)
    est = BO.estimate(v, stage, sc, bm["per_eval"])
    print(f"[budget2-{stage}] {len(idxs)} ảnh × {len(sc['configs'])} cấu hình: {est['grad_per_image']} grad/ảnh ≈ {est['minutes']:.1f} phút "
          "(nếu chưa có cache)", flush=True)
    lim = (b2(cfg).get("gpu_minutes") or {}).get(stage)
    if lim is not None and est["minutes"] > float(lim):
        raise SystemExit(f"[budget2-{stage}] ước tính {est['minutes']:.1f} phút > budget2.gpu_minutes.{stage} = {lim}")
    allr = {}
    t0, n0 = time.time(), Counter.n
    for j, idx in enumerate(idxs):
        print(f"  [budget2-{stage}] ảnh {j + 1}/{len(idxs)} seg{idx:05d}", flush=True)
        for k, val in BO.process(ctx, v, TAG[stage], sc, idx, out).items():
            allr.setdefault(k, []).extend(val)
    for k, val in allr.items():
        if val:
            pd.DataFrame(val).to_csv(os.path.join(out, f"{k}.csv"), index=False)
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"time": _dt.datetime.now().isoformat(timespec="seconds"), "stage": stage, "images": [f"seg{i:05d}" for i in idxs],
                   "sc": sc, "protocol": pr2(cfg), "estimate": est, "minutes": (time.time() - t0) / 60, "grad_evals": Counter.n - n0,
                   "config_digest": IC._h(b2(cfg)), "debug": cfg["_debug"]}, fh, indent=1, default=str)
    if stage == "dev":
        select(cfg)
    write_report(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# lựa chọn + cổng trên dev
# ─────────────────────────────────────────────────────────────────────────────

def _read(stage: str, name: str, cfg: dict):
    return BO._read(stage, name, view(cfg))


def _wide(cfg: dict, stage: str, set_name: str, dset: str = None, budgets=None):
    pr = pr2(cfg)
    p = _read(stage, "pairs", cfg)
    if p is None:
        return None, None
    jl = BO.j_long(p, set_name, dset or b2(cfg)["sets"][0], budgets or pr["budgets"], pr["M"])
    return jl, p


def select(cfg: dict) -> None:
    b, pr = b2(cfg), pr2(cfg)
    cp = b["coupling"]
    nb, seed = int(pr["bootstrap"]), int(pr["seed"])
    jl, p = _wide(cfg, "dev", "sel")
    je, _ = _wide(cfg, "dev", "eval")
    inc = cand(b["incumbent"], cp)
    rec = {}
    for B in pr["branchA_budgets"] + (pr["branchB_budgets"] if b["branchB"].get("enabled", True) else []):
        q = jl[(jl.B == int(B)) & (jl.coupling == cp)]
        W = q.pivot_table(index="image", columns="cand", values="J", aggfunc="first").dropna(axis=1, how="all").dropna()
        spi = [c for c in W.columns if c.startswith("spi_")]
        if int(B) in pr["branchA_budgets"]:
            branch, ref = "A", inc
            chal = [c for c in spi if c != inc]
        else:
            branch = "B"
            ref = min(spi, key=lambda c: W[c].mean())
            chal = [c for c in W.columns if c.startswith("hier_")]
        if ref not in W or not chal:
            continue
        best = min(chal, key=lambda c: W[c].mean())
        g = gate(W[ref].values, W[best].values, float(pr["gate_min_gain"]), float(pr["gate_min_prob"]), nb, seed)
        sl = BO.select_budget(q, int(B), nb, seed, allowed=list(W.columns))
        We = je[(je.B == int(B)) & (je.coupling == cp)].pivot_table(index="image", columns="cand", values="J", aggfunc="first")
        pe = We[[ref, best]].dropna()
        plan = plan_n(pe[ref].values, pe[best].values, float(pr["delta_rel"]), float(pr["alpha"]), float(pr["power"]))
        rec[str(int(B))] = {"B": int(B), "branch": branch, "ref": ref, "challenger": best, **g, "J_ref": float(W[ref].mean()),
                            "J_challenger": float(W[best].mean()), "K": int(q[q.cand == best].K.iloc[0]),
                            "ref_conf": BO.conf_of(p, ref.split("|")[0]), "challenger_conf": BO.conf_of(p, best.split("|")[0]),
                            "plan": plan, "p_select": sl.get("table", []), "n_images": int(W.shape[0])}
    out = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "set": "sel", "dset": b["sets"][0], "coupling": cp,
           "incumbent": inc, "rule": {k: pr[k] for k in ("gate_min_gain", "gate_min_prob", "delta_rel", "power")}, "budgets": rec}
    p_ = sel_path(cfg)
    if os.path.exists(p_):
        old = json.load(open(p_))
        if os.path.isdir(os.path.join(root(cfg), "test", "cache")) and not cfg["_debug"] and old.get("budgets") != rec:
            raise SystemExit(f"{p_} đã dùng cho test; không đổi lựa chọn sau khi mở tập test")
        out["previous"] = {k: val for k, val in old.items() if k != "previous"}
    with open(p_, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"[budget2-dev] lựa chọn + cổng → {p_}")


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo
# ─────────────────────────────────────────────────────────────────────────────

def decomp_table(cfg: dict, stage: str, set_name: str, budgets, configs=None) -> str:
    """L_D(μ), phương sai một đường V₁ = V_e + V_o, 2V_e (rev), ρ, J_rev / J_indep tại từng B — tách nguồn thắng."""
    import pandas as pd
    from .report import md
    pr = pr2(cfg)
    p = _read(stage, "pairs", cfg)
    if p is None:
        return ""
    q = p[(p.set == set_name) & (p.group == "all") & (p.dset == b2(cfg)["sets"][0])]
    if configs is not None:
        q = q[q.config.isin(configs)]
    rows = []
    for c, g in q.groupby("config"):
        r = {"config": c, "N": int(g.N.iloc[0]), "L_D(μ)": float(g.L_mu.mean()), "V₁ = V_e + V_o": float((g.Ve + g.Vo).mean()),
             "2V_e": float((2 * g.Ve).mean()), "ρ = 2V_e/V₁ (tỷ số TB)": float((2 * g.Ve).mean() / (g.Ve + g.Vo).mean())}
        for B in budgets:
            K = BO.k_of(B, int(g.N.iloc[0]), pr["M"])
            if K is None:
                continue
            for cp in BO.COUPLINGS:
                if BO.coupling_ok(cp, K):
                    r[f"J {cp} B{B} (K{K})"] = float(BO.j_val(g.L_mu, g.Ve, g.Vo, K, cp).mean())
        rows.append(r)
    t = pd.DataFrame(rows)
    sk = f"J {b2(cfg)['coupling']} B{budgets[0]}"
    col = [c_ for c_ in t.columns if c_.startswith(sk)]
    if col:
        t = t.sort_values(col[0])
    return md(t)


def dev_section(cfg: dict) -> list:
    import pandas as pd
    from .report import md
    pr = pr2(cfg)
    if _read("dev", "pairs", cfg) is None:
        return []
    L = [f"## Dev — chọn + cổng (D chính `{b2(cfg)['sets'][0]}`, coupling chính `{b2(cfg)['coupling']}`)\n",
         "**Phân rã nguồn J — bộ ô CHỌN** (L_D(μ) và phương sai không phụ thuộc B; J = L + V_K):\n",
         decomp_table(cfg, "dev", "sel", pr["budgets"]),
         "**Phân rã — bộ ô ĐÁNH GIÁ** (độc lập, mô tả):\n", decomp_table(cfg, "dev", "eval", pr["budgets"])]
    try:
        sel = load_sel(cfg)
    except SystemExit:
        return L
    rows, pl = [], []
    for B, s in sel["budgets"].items():
        rows.append({"B": B, "nhánh": s["branch"], "tham chiếu": s["ref"], "J tham chiếu": s["J_ref"], "ứng viên tốt nhất": s["challenger"],
                     "K": s["K"], "J ứng viên": s["J_challenger"], "giảm tương đối": s["gain_rel"], "P_boot(tốt hơn)": s["p_better"],
                     "qua cổng": s["pass"], "trong test.budgets": int(B) in [int(x) for x in b2(cfg)["test"]["budgets"]]})
        pl.append({"B": B, "g dev (eval)": s["plan"]["g_dev"], "g* giả định": s["plan"]["g_star"], "SD(D)": s["plan"]["sd_D"],
                   "hiệu ứng trên D": s["plan"]["effect_D"], "n cần": s["plan"]["n_required"]})
    L.append(f"**Cổng dev** (giảm ≥ {pr['gate_min_gain']:.0%} và P ≥ {pr['gate_min_prob']}; nhánh A so với incumbent `{sel['incumbent']}`, "
             "nhánh B so với SPI tốt nhất ở cùng B):\n")
    L.append(md(pd.DataFrame(rows)))
    L.append(f"**Kế hoạch cỡ mẫu** (δ = {pr['delta_rel']:.0%}, lực {pr['power']}, một phía α = {pr['alpha']}; n kẹp [{pr['n_min']}, {pr['n_max']}]):\n")
    L.append(md(pd.DataFrame(pl)))
    rows = []
    for B, s in sel["budgets"].items():
        for x in s["p_select"]:
            if x["p_select"] > 0:
                rows.append({"B": B, "ứng viên": x["cand"], "TB J": x["J_mean"], "P_select": x["p_select"]})
    if rows:
        L.append("**Phân phối xác suất chọn đầy đủ** (mọi ứng viên cùng coupling; P > 0):\n")
        L.append(md(pd.DataFrame(rows).sort_values(["B", "P_select"], ascending=[True, False])))
    em = _read("dev", "emp", cfg)
    if em is not None:
        L.append(BO.emp_table(view(cfg), em))
    return L


def test_section(cfg: dict) -> list:
    import pandas as pd
    from .report import md
    b, pr = b2(cfg), pr2(cfg)
    if _read("test", "pairs", cfg) is None:
        return []
    sel = load_sel(cfg)
    tb = tested_budgets(cfg, sel)
    nb, seed, al, dl = int(pr["bootstrap"]), int(pr["seed"]), float(pr["alpha"]), float(pr["delta_rel"])
    cp = b["coupling"]
    Dp = b["sets"][0]
    jl, p = _wide(cfg, "test", "eval", budgets=tb)
    L = [f"## Test — kiểm định trên ảnh mới (D chính `{Dp}`, bộ ô ĐÁNH GIÁ)\n",
         f"H0: E[J_ref − J_new] ≤ δ·E[J_ref], δ = {dl:.0%}; p một phía bootstrap-t trên D (chính), t (độ nhạy); Holm qua {tb}. "
         f"B chính = {pr['primary_budget']}.\n"]
    rows, ps = [], []
    for B in tb:
        s = sel["budgets"][str(B)]
        W = jl[(jl.B == B) & (jl.coupling == cp)].pivot_table(index="image", columns="cand", values="J", aggfunc="first")
        j = W[[s["ref"], s["challenger"]]].dropna()
        d = decide_test(j[s["ref"]].values, j[s["challenger"]].values, dl, nb, seed, al)
        ps.append(d["p_boot_t"])
        rows.append({"B": B, "nhánh": s["branch"], "tham chiếu": s["ref"], "mới": s["challenger"], "n": d["n"], "J tham chiếu": d["J_ref"],
                     "J mới": d["J_new"], "ΔJ [CI]": fmt_ci(d["dJ"], d["dJ_lo"], d["dJ_hi"]), "g [CI percentile]": fmt_ci(d["g"], d["g_lo"], d["g_hi"], "{:.3f}"),
                     "g [CI Fieller]": fmt_ci(d["g"], d["g_fieller_lo"], d["g_fieller_hi"], "{:.3f}"), "ảnh mới tốt hơn": f"{d['wins']}/{d['n']}",
                     "p bootstrap-t": d["p_boot_t"], "p t (độ nhạy)": d["p_t"], "_hi": d["g_hi"]})
    if rows:
        for r, h in zip(rows, holm(ps)):
            r["p Holm"] = h
            r["quyết định"] = ST.CONFIRM if h < al else ST.REJECT if r["_hi"] < dl else ST.OPEN
            del r["_hi"]
        L.append(md(pd.DataFrame(rows)))
    # coupling: J indep của cùng cấu hình
    rows = []
    for B in tb:
        s = sel["budgets"][str(B)]
        W = jl[jl.B == B].pivot_table(index="image", columns="cand", values="J", aggfunc="first")
        for c in (s["challenger"], s["ref"]):
            base = c.split("|")[0]
            if f"{base}|indep" in W and f"{base}|rev" in W:
                v = (W[f"{base}|indep"] - W[f"{base}|rev"]).dropna().values
                lo, hi = ci_of(v[boot_idx(len(v), nb, seed)].mean(1), al)
                rows.append({"B": B, "config": base, "J rev": float(W[f"{base}|rev"].mean()), "J indep": float(W[f"{base}|indep"].mean()),
                             "J indep − J rev [CI]": fmt_ci(float(v.mean()), lo, hi)})
    if rows:
        L.append("**Đóng góp của coupling** (cùng α, N, K; tách khỏi lợi ích của α / N):\n")
        L.append(md(pd.DataFrame(rows)))
    names = sorted({x for B in tb for x in (sel["budgets"][str(B)]["ref"].split("|")[0], sel["budgets"][str(B)]["challenger"].split("|")[0])}
                   | {cname(b["incumbent"]), cname(b["prior"])})
    L.append("**Phân rã nguồn J trên test** (bộ ô đánh giá):\n")
    L.append(decomp_table(cfg, "test", "eval", tb, names))
    # so với đối chứng phụ α = 0,1 + độ bền theo D
    rows = []
    prior = cand(b["prior"], cp)
    for B in tb:
        s = sel["budgets"][str(B)]
        W = jl[jl.B == B].pivot_table(index="image", columns="cand", values="J", aggfunc="first")
        for other in (prior, cand(b["incumbent"], cp)):
            if other in W and other not in (s["challenger"],) and s["challenger"] in W:
                v = (W[other] - W[s["challenger"]]).dropna().values
                lo, hi = ci_of(v[boot_idx(len(v), nb, seed)].mean(1), al)
                rows.append({"B": B, "so với": other, "ΔJ = J(so với) − J(mới) [CI]": fmt_ci(float(v.mean()), lo, hi),
                             "ảnh mới tốt hơn": f"{int((v > 0).sum())}/{len(v)}"})
    if rows:
        L.append("**So sánh phụ** (mô tả; dương ⇒ ứng viên mới tốt hơn):\n")
        L.append(md(pd.DataFrame(rows).drop_duplicates()))
    rows = []
    for B in tb:
        s = sel["budgets"][str(B)]
        cols, colsL = {}, {}
        for dn in list(dict.fromkeys(p.dset)):
            jd = BO.j_long(p, "eval", dn, [B], pr["M"])
            W = jd.pivot_table(index="image", columns="cand", values="J", aggfunc="first")
            WL = jd.pivot_table(index="image", columns="config", values="L_mu", aggfunc="first")
            if s["challenger"] in W and s["ref"] in W:
                cols[dn] = W[s["ref"]] - W[s["challenger"]]
                colsL[dn] = WL[s["ref"].split("|")[0]] - WL[s["challenger"].split("|")[0]]
        for lab, M_ in (("J ở B", pd.DataFrame(cols).dropna()), ("L_D(μ)", pd.DataFrame(colsL).dropna())):
            if not len(M_):
                continue
            X = M_.values
            bs = X[boot_idx(len(X), nb, seed)].mean(1)
            ph = holm([boot_p(bs[:, k]) for k in range(X.shape[1])])
            for k, dn in enumerate(M_.columns):
                lo, hi = ci_of(bs[:, k], al)
                rows.append({"B": B, "đại lượng": lab, "D": dn, "tham chiếu − mới [CI]": fmt_ci(float(X[:, k].mean()), lo, hi),
                             "ảnh mới tốt hơn": f"{int((X[:, k] > 0).sum())}/{len(X)}", "p Holm (qua D)": ph[k]})
    if rows:
        L.append("**Độ bền theo D** (mô tả; dương ⇒ mới tốt hơn tham chiếu):\n")
        L.append(md(pd.DataFrame(rows)))
    em = _read("test", "emp", cfg)
    if em is not None:
        L.append(BO.emp_table(view(cfg), em))
    L += ext_section(cfg, sel, tb)
    return L


def ext_section(cfg: dict, sel: dict, tb: list) -> list:
    import pandas as pd
    from .report import md
    ex = _read("test", "ext", cfg)
    if ex is None:
        return []
    b, pr = b2(cfg), pr2(cfg)
    nb, seed, al = int(pr["bootstrap"]), int(pr["seed"]), float(pr["alpha"])
    cp = b["coupling"]
    ex = ex.assign(key=ex.config + "|" + ex.estimator)
    Wd = {k: g.set_index("image").sort_index() for k, g in ex.groupby("key")}

    def key(conf_name, B):
        N = int(conf_name.split("_N")[1].split("_")[0])
        return f"{conf_name}|{cp}_B{B}_K{BO.k_of(B, N, pr['M'])}"
    comps = []
    for B in tb:
        s = sel["budgets"][str(B)]
        ch, rf = s["challenger"].split("|")[0], s["ref"].split("|")[0]
        comps += [(key(ch, B), key(rf, B), f"B = {B}"), (key(ch, B), key(cname(b["prior"]), B), f"B = {B}"),
                  (f"{ch}|mean_all", f"{rf}|mean_all", "Ē toàn bộ 2m đường"), (f"{ch}|mean_all", f"{cname(b['prior'])}|mean_all", "Ē toàn bộ 2m đường")]
    rows, seen = [], set()
    for a, bb, lab in comps:
        if (a, bb) in seen or a == bb or a not in Wd or bb not in Wd:
            continue
        seen.add((a, bb))
        grp = []
        for met, sgn in EXT_DIR.items():
            if met not in Wd[a] or met not in Wd[bb]:
                continue
            j = pd.concat([Wd[a][met].rename("a"), Wd[bb][met].rename("b")], axis=1).dropna()
            v = sgn * (j.a - j.b).values
            bs = v[boot_idx(len(v), nb, seed)].mean(1)
            lo, hi = ci_of(bs, al)
            grp.append({"mức": lab, "a": a, "b": bb, "metric": met, "TB a": float(j.a.mean()), "TB b": float(j.b.mean()),
                        "(a − b) theo chiều tốt [CI]": fmt_ci(float(v.mean()), lo, hi), "p": boot_p(bs), "_s": float(np.sign(v.mean()))})
        for g_, h in zip(grp, holm([g_["p"] for g_ in grp])):
            g_["p Holm (trong cặp)"] = h
            g_["kết luận"] = ("a tốt hơn" if g_["_s"] > 0 else "a kém hơn") if h < al else "chưa ghi nhận khác biệt"
            del g_["_s"]
        rows += grp
    if not rows:
        return []
    return ["## Metric ngoài (chẩn đoán bắt buộc; KHÔNG dùng để chọn lại)\n",
            "Ở K của ngân sách và trên Ē của 2m đường (tách phần do phương sai K hữu hạn).\n", md(pd.DataFrame(rows))]


def write_report(cfg: dict) -> str:
    rp = report_path(cfg)
    pr = pr2(cfg)
    lk = lock_path(cfg)
    L = ["# BUDGET2 — vượt incumbent ở cùng ngân sách gradient (nhánh A chính; nhánh B có cổng)\n"]
    if cfg.get("_debug"):
        L.append("> **DEBUG** — chỉ kiểm pipeline.\n")
    b = b2(cfg)
    L.append(f"- Incumbent `{cand(b['incumbent'], b['coupling'])}`; đối chứng phụ `{cand(b['prior'], b['coupling'])}`. Nhánh A B = "
             f"{pr['branchA_budgets']} (chính {pr['primary_budget']}), nhánh B B = {pr['branchB_budgets']}. Quy tắc `{pr['rule']}`, M = {pr['M']}. "
             "Khoá test: " + (f"`{json.load(open(lk))['digest'][:12]}`" if os.path.exists(lk) else "**chưa khoá**") + ".")
    L.append("- Đơn vị thống kê là ảnh; bootstrap theo ảnh. Dev chọn trên bộ ô CHỌN; test trên ảnh mới, bộ ô ĐÁNH GIÁ.\n")
    L += dev_section(cfg)
    L += test_section(cfg)
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[budget2] {rp}")
    return rp

