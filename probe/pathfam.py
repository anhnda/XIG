"""
PATH_FAMILIES Phase 1 — Dirichlet có cần thiết? So 4 họ đường theo J ở ngân sách gradient cố định (docs/PATH_FAMILIES_PROTOCOL.md).

Họ (N = 32, lưới khớp exploration theo α / h = 1/(α + 1)): spi (= DP lưới), ln (logistic-normal, σ hiệu chuẩn), cmix (COORD_MIX,
λ = h), smooth (δ ≤ 1, chỉ mô tả). Đối chứng cố định: incumbent SPI α = 1, N = 32, `rev`; SPI α = 0,1, N = 64; SPI α = 10, N = 64.
Chọn trên dev bằng J_rev (bộ ô CHỌN); so sánh với incumbent bằng J_rev. J_indep báo riêng, KHÔNG so chéo với J_rev.

Kiểm định chính (test, bộ ô ĐÁNH GIÁ, D chính): ln và cmix tại B ∈ {256, 1024} (4 phép so), g = (J̄_inc − J̄_F)/J̄_inc,
non-inferiority H0: g ≤ −ε (ε = 0,10) ⇔ E[D] ≤ 0 với D_i = J_inc,i − J_F,i + ε·J_inc,i; p một phía bootstrap-t; Holm qua 4.
Phụ (đặt trước, mỗi họ phép một Holm qua 4): superiority H0: g ≤ 0; equivalence TOST |g| ≥ ε. Smooth: mô tả.

Lệnh: pathfam-plan (CPU) → pathfam-dev → pathfam-lock → pathfam-test; pathfam-report dựng lại infid/INFID_PATHFAM_REPORT.md.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import time
from statistics import NormalDist

import numpy as np
import torch

from . import budget_opt as BO
from . import config as C
from . import infid_core as IC
from . import infid_stats as ST
from . import path_families as PF
from .budget_audit import boot_t_p, fieller
from .infid_posthoc import EXT_DIR, boot_idx, boot_p, ci_of, fmt_ci, holm
from .quad import Counter
from .sampling_infid import cname, norm_conf, stage_images

STAGES = ("dev", "test")
TAG = {"dev": "pfdev", "test": "pftest"}

DEFAULT = {
    "dir": "pathfam",
    "protocol": {"rule": "rand", "M": 1, "budgets": [256, 1024], "budgets_secondary": [128, 512, 2048, 4096], "Ks": [2, 4, 8, 16, 32],
                 "check_K": [4, 16], "alpha": 0.05, "bootstrap": 10000, "seed": 61, "floor_grids": [14],
                 "floor_lams": [0.0, 1.0e-3, 1.0e-2, 1.0e-1], "floor_folds": 4, "select_coupling": "rev", "ni_margin": 0.10,
                 "eq_margin": 0.10, "power": 0.80, "n_min": 48, "n_max": 128, "emp_ci_method": "auto"},
    "sets": ["del_sq"],
    "incumbent": {"family": "spi", "alpha": 1, "N": 32},
    "controls": [{"family": "spi", "alpha": 0.1, "N": 64}, {"family": "spi", "alpha": 10, "N": 64}],
    "grid": {"alphas": [0.1, 0.3, 1, 3, 10], "N": 32, "families": ["spi", "ln", "cmix"], "smooth_delta": [0.5, 0.674, 1.0]},
    "primary_families": ["ln", "cmix"],
    "descriptive_families": ["smooth", "spi"],
    "dev": {"split": "fresh", "offset": 1200, "n_images": 32, "m": 16},
    "test": {"split": "fresh", "offset": 1300, "m": 32,
             "sets": ["del_sq", "ins_sq", "sym_sq", "frac25", "frac50", "frac75", "del_sq_blur", "del_sp"], "external": True,
             "road": {"fracs": [0.1, 0.3, 0.5, 0.7, 0.9], "noise": 0.01, "iters": 1000, "tol": 1.0e-5},
             "insdel_steps": 50, "pointing_tol": 15, "top_frac": 0.05},
    "gpu_minutes": {"dev": None, "test": None},
    "per_eval_seconds": None,
}


def pc(cfg: dict) -> dict:
    return C._merge(DEFAULT, cfg.get("pathfam", {}) or {})


def pr(cfg: dict) -> dict:
    return pc(cfg)["protocol"]


def view(cfg: dict) -> dict:
    v = dict(cfg)
    v["budget"] = pc(cfg)
    return v


def root(cfg: dict) -> str:
    return BO.broot(view(cfg))


def sel_path(cfg: dict) -> str:
    return os.path.join(root(cfg), "selection_dev.json")


def lock_path(cfg: dict) -> str:
    return os.path.join(root(cfg), "pathfam_prereg.json")


def report_path(cfg: dict) -> str:
    return os.path.join(C.out_dir(cfg, "pilot"), "infid", "INFID_PATHFAM_REPORT.md")


def fam_of(name: str) -> str:
    return name.split("_")[0]


def cand(conf, cp: str) -> str:
    return f"{conf if isinstance(conf, str) else cname(conf)}|{cp}"


# ─────────────────────────────────────────────────────────────────────────────
# cấu hình
# ─────────────────────────────────────────────────────────────────────────────

def grid_configs(cfg: dict) -> list:
    g = pc(cfg)["grid"]
    N = int(g["N"])
    cf = [norm_conf({"family": f, "alpha": a, "N": N}) for f in g["families"] for a in g["alphas"]]
    cf += [norm_conf({"family": "smooth", "delta": d, "N": N}) for d in g["smooth_delta"]]
    return cf


def dev_configs(cfg: dict) -> list:
    b = pc(cfg)
    seen, out = set(), []
    for c in grid_configs(cfg) + [norm_conf(b["incumbent"])] + [norm_conf(x) for x in b["controls"]]:
        if cname(c) not in seen:
            seen.add(cname(c))
            out.append(c)
    return out


def family_members(cfg: dict, fam: str) -> list:
    return [cname(c) for c in grid_configs(cfg) if c["family"] == fam]


def h_of(conf: dict) -> float:
    if conf["family"] == "smooth":
        return conf["delta"] ** 2 / 5
    if conf["family"] == "ig":
        return 0.0
    return 1.0 / (conf["alpha"] + 1.0)


def load_sel(cfg: dict) -> dict:
    p = sel_path(cfg)
    if not os.path.exists(p):
        raise SystemExit(f"chưa có {p} — chạy bash scripts/run.sh pathfam-dev trước")
    with open(p) as fh:
        return json.load(fh)


def test_configs(cfg: dict, sel: dict) -> list:
    b = pc(cfg)
    names = []
    for B in pr(cfg)["budgets"]:
        for fam in b["primary_families"] + ["smooth"]:
            s = sel["chosen"].get(str(B), {}).get(fam)
            if s:
                names.append(s["conf"])
    cf = names + [b["incumbent"]] + list(b["controls"])
    seen, out = set(), []
    for c in cf:
        c = norm_conf(c)
        if cname(c) not in seen:
            seen.add(cname(c))
            out.append(c)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# hàm thuần (unit test)
# ─────────────────────────────────────────────────────────────────────────────

def tests_for(j_inc: np.ndarray, j_f: np.ndarray, eps: float, n_boot: int, seed: int, al: float) -> dict:
    """g, CI; p một phía bootstrap-t cho NI (H0: g ≤ −ε), superiority (H0: g ≤ 0), TOST (|g| ≥ ε)."""
    n = len(j_inc)
    idx = boot_idx(n, n_boot, seed)
    g = float((j_inc.mean() - j_f.mean()) / j_inc.mean())
    gb = (j_inc[idx].mean(1) - j_f[idx].mean(1)) / j_inc[idx].mean(1)
    lo, hi = ci_of(gb, al)
    flo, fhi = fieller(j_f, j_inc, np.zeros_like(j_inc), 1 - al)
    d = j_inc - j_f
    p_ni = boot_t_p(d + eps * j_inc, n_boot, seed)
    p_sup = boot_t_p(d, n_boot, seed)
    p_up = boot_t_p(eps * j_inc - d, n_boot, seed)
    lr = np.log(j_f / j_inc)
    lr = lr[np.isfinite(lr)]
    llo, lhi = ci_of(lr[boot_idx(len(lr), n_boot, seed)].mean(1), al) if len(lr) > 1 else (math.nan, math.nan)
    return {"n": n, "J_inc": float(j_inc.mean()), "J_F": float(j_f.mean()), "g": g, "g_lo": lo, "g_hi": hi, "g_flo": flo, "g_fhi": fhi,
            "p_ni": p_ni, "p_sup": p_sup, "p_tost": max(p_ni, p_up), "wins": int((d > 0).sum()),
            "logratio": float(lr.mean()) if len(lr) else math.nan, "logratio_lo": llo, "logratio_hi": lhi}


def decide(r: dict, eps: float, al: float) -> str:
    if r["p_ni_holm"] < al:
        s = "KHÔNG KÉM HƠN (NI xác nhận)"
    elif r["g_hi"] < -eps:
        s = "KÉM HƠN thực tiễn"
    else:
        s = "CHƯA KẾT LUẬN"
    return s


def plan_ni(j_inc: np.ndarray, j_f: np.ndarray, eps: float, al_adj: float, power: float) -> dict:
    """n cho NI với g thật = 0 (giả định đặt trước): hiệu ứng trên D = ε·J̄_inc; mức một phía α/4 (cận Bonferroni của Holm)."""
    D = (j_inc - j_f) + eps * j_inc
    sd = float(D.std(ddof=1))
    eff = eps * float(j_inc.mean())
    z = NormalDist().inv_cdf
    n = math.ceil(((z(1 - al_adj) + z(power)) * sd / eff) ** 2) if eff > 0 and sd > 0 else math.nan
    g_dev = float((j_inc.mean() - j_f.mean()) / j_inc.mean())
    eff_obs = (g_dev + eps) * float(j_inc.mean())
    n_obs = math.ceil(((z(1 - al_adj) + z(power)) * sd / eff_obs) ** 2) if eff_obs > 0 and sd > 0 else math.nan
    return {"sd_D": sd, "g_dev": g_dev, "n_req_g0": n, "n_req_gdev": n_obs}


# ─────────────────────────────────────────────────────────────────────────────
# chạy
# ─────────────────────────────────────────────────────────────────────────────

def stage_sc(cfg: dict, stage: str) -> dict:
    b = pc(cfg)
    if stage == "dev":
        return {**b["dev"], "sets": b["sets"], "configs": dev_configs(cfg), "external": False, "ig": False}
    sel = load_sel(cfg)
    lk = json.load(open(lock_path(cfg)))["payload"] if os.path.exists(lock_path(cfg)) else None
    n = lk["n_test"] if lk else int(pr(cfg)["n_min"])
    return {**b["test"], "n_images": int(n), "configs": test_configs(cfg, sel), "ig": False, "map_budgets": [int(x) for x in pr(cfg)["budgets"]]}


def used_images(ctx, cfg: dict) -> set:
    from . import budget2 as B2
    used = BO.used_infid_images(ctx, cfg)
    b1 = BO.bcfg(cfg)
    b2 = B2.b2(cfg)
    for sc in (b1["dev"], b1["test"], b2["dev"], {**b2["test"], "n_images": int(B2.pr2(cfg)["n_max"])}):
        try:
            used |= set(stage_images(ctx, sc))
        except (SystemExit, KeyError):
            pass
    return used


def check_images(ctx, cfg: dict, stage: str, idxs: list) -> None:
    if cfg.get("_debug"):
        return
    b = pc(cfg)
    used = used_images(ctx, cfg)
    other = b["test"] if stage == "dev" else b["dev"]
    try:
        used |= set(stage_images(ctx, {**other, "n_images": int(pr(cfg)["n_max"]) if stage == "dev" else int(b["dev"]["n_images"])}))
    except (SystemExit, KeyError):
        pass
    inter = set(idxs) & used
    if inter:
        raise SystemExit(f"[pathfam-{stage}] {len(inter)} ảnh trùng với ảnh đã dùng (infid / BUDGET / BUDGET2 / giai đoạn kia) — đổi offset")


def lock_payload(cfg: dict) -> dict:
    from .infid_e0 import icfg
    b = pc(cfg)
    sp = sel_path(cfg)
    sel = load_sel(cfg)
    D = icfg(cfg)["D"]
    pl = power_table(cfg, sel)
    return {"protocol": b["protocol"], "sets": b["sets"], "incumbent": b["incumbent"], "controls": b["controls"], "grid": b["grid"],
            "primary_families": b["primary_families"], "test": b["test"], "D": {k: D["sets"][k] for k in b["test"]["sets"]},
            "n_sel": D.get("n_sel"), "n_eval": D.get("n_eval"), "selection_sha256": hashlib.sha256(open(sp, "rb").read()).hexdigest(),
            "test_configs": [cname(c) for c in test_configs(cfg, sel)], "power": pl["rows"], "n_test": pl["n_test"],
            "underpowered": pl["underpowered"], "model": cfg["model"], "data": cfg["data"], "seed": cfg["seed"], "v": 1}


def lock(cfg: dict) -> None:
    from .pathfam_verify import check_frozen
    check_frozen(cfg)
    pl = lock_payload(cfg)
    ST.write_prereg(lock_path(cfg), IC._h(pl), {"payload": pl})
    print(f"[pathfam-lock] n test = {pl['n_test']}; thiếu lực: {pl['underpowered'] or 'không'}; cấu hình test: {pl['test_configs']}")


def run(ctx, stage: str) -> None:
    from .ctx import bench
    import pandas as pd
    cfg = ctx.cfg
    v = view(cfg)
    if stage not in STAGES:
        raise SystemExit(stage)
    from .pathfam_verify import check_frozen
    check_frozen(cfg)
    if stage == "test":
        if not cfg["_debug"] and not os.path.exists(lock_path(cfg)):
            raise SystemExit(f"pathfam-test cần khoá: bash scripts/run.sh pathfam-lock (chưa thấy {lock_path(cfg)})")
        ST.require_prereg(lock_path(cfg), IC._h(lock_payload(cfg)), cfg["_debug"])
    sc = stage_sc(cfg, stage)
    out = os.path.join(root(cfg), stage)
    os.makedirs(out, exist_ok=True)
    idxs = stage_images(ctx, sc)
    check_images(ctx, cfg, stage, idxs)
    for c in sc["configs"]:                                                    # hiệu chuẩn σ trước (CPU, tất định)
        if c["family"] == "ln":
            PF.family_params(c)
    bm = bench(ctx)
    est = BO.estimate(v, stage, sc, bm["per_eval"])
    print(f"[pathfam-{stage}] {len(idxs)} ảnh × {len(sc['configs'])} cấu hình: {est['grad_per_image']} grad/ảnh ≈ {est['minutes']:.1f} phút "
          "(nếu chưa có cache)", flush=True)
    lim = (pc(cfg).get("gpu_minutes") or {}).get(stage)
    if lim is not None and est["minutes"] > float(lim):
        raise SystemExit(f"[pathfam-{stage}] ước tính {est['minutes']:.1f} phút > pathfam.gpu_minutes.{stage} = {lim}")
    allr = {}
    t0, n0 = time.time(), Counter.n
    for j, idx in enumerate(idxs):
        print(f"  [pathfam-{stage}] ảnh {j + 1}/{len(idxs)} seg{idx:05d}", flush=True)
        for k, val in BO.process(ctx, v, TAG[stage], sc, idx, out).items():
            allr.setdefault(k, []).extend(val)
    for k, val in allr.items():
        if val:
            pd.DataFrame(val).to_csv(os.path.join(out, f"{k}.csv"), index=False)
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"time": _dt.datetime.now().isoformat(timespec="seconds"), "stage": stage, "images": [f"seg{i:05d}" for i in idxs],
                   "sc": sc, "protocol": pr(cfg), "estimate": est, "bench": bm, "minutes": (time.time() - t0) / 60,
                   "grad_evals": Counter.n - n0, "sigma": {cname(c): PF.family_params(c)["sigma"] for c in sc["configs"] if c["family"] == "ln"},
                   "config_digest": IC._h(pc(cfg)), "debug": cfg["_debug"]}, fh, indent=1, default=str)
    if stage == "dev":
        select(cfg)
    write_report(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# lựa chọn + lực trên dev
# ─────────────────────────────────────────────────────────────────────────────

def _read(stage: str, name: str, cfg: dict):
    return BO._read(stage, name, view(cfg))


def jl_of(cfg: dict, stage: str, set_name: str, dset: str = None, budgets=None):
    p = _read(stage, "pairs", cfg)
    if p is None:
        return None, None
    P = pr(cfg)
    return BO.j_long(p, set_name, dset or pc(cfg)["sets"][0], budgets or (P["budgets"] + P["budgets_secondary"]), P["M"]), p


def wide(jl, B: int, cp: str):
    return jl[(jl.B == int(B)) & (jl.coupling == cp)].pivot_table(index="image", columns="config", values="J", aggfunc="first")


def select(cfg: dict) -> None:
    b, P = pc(cfg), pr(cfg)
    cp = P["select_coupling"]
    nb, seed = int(P["bootstrap"]), int(P["seed"])
    jl, p = jl_of(cfg, "dev", "sel")
    confs = {cname(c): c for c in dev_configs(cfg)}
    chosen = {}
    for B in P["budgets"]:
        W = wide(jl, B, cp).dropna()
        chosen[str(B)] = {}
        for fam in b["primary_families"] + b["descriptive_families"]:
            mem = [m for m in family_members(cfg, fam) if m in W]
            if not mem:
                continue
            X = W[mem].values
            win = np.bincount(X[boot_idx(X.shape[0], nb, seed)].mean(1).argmin(1), minlength=len(mem)) / nb
            best = mem[int(np.argmin(X.mean(0)))]
            chosen[str(B)][fam] = {"config": best, "conf": confs[best], "J_mean": float(W[best].mean()),
                                   "p_select": {m: float(w) for m, w in zip(mem, win)},
                                   "J_by_member": {m: float(W[m].mean()) for m in mem}}
    rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "set": "sel", "coupling": cp, "dset": b["sets"][0],
           "incumbent": cname(b["incumbent"]), "chosen": chosen}
    p_ = sel_path(cfg)
    if os.path.exists(p_):
        old = json.load(open(p_))
        if os.path.isdir(os.path.join(root(cfg), "test", "cache")) and not cfg["_debug"] and old.get("chosen") != chosen:
            raise SystemExit(f"{p_} đã dùng cho test; không đổi lựa chọn sau khi mở tập test")
        rec["previous"] = {k: val for k, val in old.items() if k != "previous"}
    with open(p_, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(f"[pathfam-dev] lựa chọn → {p_}")


def power_table(cfg: dict, sel: dict) -> dict:
    b, P = pc(cfg), pr(cfg)
    cp = P["select_coupling"]
    je, _ = jl_of(cfg, "dev", "eval")
    inc = cname(b["incumbent"])
    al_adj = float(P["alpha"]) / (len(P["budgets"]) * len(b["primary_families"]))
    rows, ns, under = [], [], []
    for B in P["budgets"]:
        W = wide(je, B, cp)
        for fam in b["primary_families"]:
            s = sel["chosen"].get(str(B), {}).get(fam)
            if not s or s["config"] not in W or inc not in W:
                continue
            j = W[[inc, s["config"]]].dropna()
            pl = plan_ni(j[inc].values, j[s["config"]].values, float(P["ni_margin"]), al_adj, float(P["power"]))
            rows.append({"B": int(B), "family": fam, "config": s["config"], **pl})
            if np.isfinite(pl["n_req_g0"]):
                ns.append(pl["n_req_g0"])
                if pl["n_req_g0"] > int(P["n_max"]):
                    under.append(f"{fam}@B{B} (cần {pl['n_req_g0']})")
    n_req = int(max(ns)) if ns else int(P["n_min"])
    return {"rows": rows, "n_required": n_req, "n_test": int(min(max(n_req, int(P["n_min"])), int(P["n_max"]))), "underpowered": under}


# ─────────────────────────────────────────────────────────────────────────────
# kế hoạch + kiểm toán chi phí gradient (CPU, không mô hình)
# ─────────────────────────────────────────────────────────────────────────────

def grads_per_path(conf: dict, M: int) -> dict:
    """Chạy đúng pipeline schedule → nodes → integrate(rand) với hàm giả trên tensor nhỏ, đếm lượt forward+backward."""
    from .pathquad import integrate
    shape = (3, 6, 6)
    g = torch.Generator().manual_seed(0)
    fam = "dp" if conf["family"] == "spi" else conf["family"]
    Z = PF.schedule(fam, conf["N"], shape, g, PF.family_params(conf))
    dx = torch.rand(shape, generator=g, dtype=torch.float64) + 0.5
    X = Z * dx                                                                  # x′ = 0
    w = torch.randn(shape, generator=g, dtype=torch.float64)
    n0 = Counter.n
    A_lin = integrate(lambda xb: (xb * w).flatten(1).sum(1), X, M, "rand", gen=torch.Generator().manual_seed(1), batch=64)
    n_used = Counter.n - n0
    return {"grads_per_path": n_used, "expected": int(conf["N"]) * int(M),
            "completeness_err_linear": float((A_lin - w * dx).abs().max())}


def plan(cfg: dict) -> str:
    import pandas as pd
    from .report import md
    b, P = pc(cfg), pr(cfg)
    M = int(P["M"])
    rows = []
    for c in dev_configs(cfg):
        gp = grads_per_path(c, M)
        r = {"config": cname(c), "họ": c["family"], "N": c["N"], "h": h_of(c)}
        if c["family"] == "ln":
            r["σ"] = PF.family_params(c)["sigma"]
        if c["family"] == "cmix":
            r["λ"] = PF.family_params(c)["lam"]
        for B in P["budgets"]:
            K = BO.k_of(B, c["N"], M)
            r[f"K rev @B{B}"] = K if BO.coupling_ok("rev", K) else "—"
        r.update({"grad/đường (đo)": gp["grads_per_path"], "N·M": gp["expected"], "khớp": gp["grads_per_path"] == gp["expected"],
                  "|A − w⊙Δx| (F tuyến tính)": gp["completeness_err_linear"], "đường dev": 2 * int(b["dev"]["m"]),
                  "grad dev / ảnh": 2 * int(b["dev"]["m"]) * gp["grads_per_path"]})
        rows.append(r)
    t = pd.DataFrame(rows)
    per_img = int(t["grad dev / ảnh"].sum())
    n_dev = int(b["dev"]["n_images"])
    m_t = int(b["test"]["m"])
    # test: tối đa 3 họ × 2 B nghiệm (N = 32) + incumbent + đối chứng
    test_cfgs = 3 * len(P["budgets"]) * int(b["grid"]["N"]) + int(b["incumbent"]["N"]) + sum(int(x["N"]) for x in b["controls"])
    test_img = 2 * m_t * test_cfgs * M
    pe = pc(cfg).get("per_eval_seconds")
    tm = (lambda g: f"{g * float(pe) / 60:.0f} phút") if pe else (lambda g: "— (đặt pathfam.per_eval_seconds để quy ra phút)")
    L = ["# PATH_FAMILIES — kế hoạch thí nghiệm cuối cùng + kiểm toán chi phí gradient (CPU, không mô hình)\n",
         "Chi phí đo bằng chính pipeline `schedule → nodes → integrate(rand)` với hàm giả (đếm lượt forward + backward). "
         "`|A − w⊙Δx|` = sai số completeness với F tuyến tính (phải ≈ 0 với mọi họ: đường hợp lệ, cầu phương đúng).\n",
         md(t),
         f"**Dev**: {len(rows)} cấu hình, m = {b['dev']['m']} cặp, {n_dev} ảnh (fresh {b['dev']['offset']}–{b['dev']['offset'] + n_dev - 1}) ⇒ "
         f"{per_img:,} grad / ảnh, tổng {per_img * n_dev:,} grad ({tm(per_img * n_dev)}); so với BUDGET G1 524 288: {per_img * n_dev / 524288:.2f}×.\n",
         f"**Test** (cận trên): ≤ {3 * len(P['budgets'])} nghiệm theo họ (N = {b['grid']['N']}) + incumbent + {len(b['controls'])} đối chứng, "
         f"m = {m_t} ⇒ ≤ {test_img:,} grad / ảnh; n ∈ [{P['n_min']}, {P['n_max']}] (tính lúc khoá) ⇒ "
         f"{test_img * int(P['n_min']):,}–{test_img * int(P['n_max']):,} grad ({tm(test_img * int(P['n_min']))} – {tm(test_img * int(P['n_max']))}), "
         "cộng metric ngoài (chỉ forward).\n",
         f"Trần GPU: dev {pc(cfg)['gpu_minutes'].get('dev')} phút, test {pc(cfg)['gpu_minutes'].get('test')} phút.\n"]
    out = root(cfg)
    t.to_csv(os.path.join(out, "plan.csv"), index=False)
    rp = os.path.join(out, "PATHFAM_PLAN.md")
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[pathfam-plan] {rp}")
    return rp


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo
# ─────────────────────────────────────────────────────────────────────────────

def decomp(cfg: dict, stage: str, set_name: str, budgets, configs=None) -> str:
    import pandas as pd
    from .report import md
    P = pr(cfg)
    p = _read(stage, "pairs", cfg)
    if p is None:
        return ""
    q = p[(p.set == set_name) & (p.group == "all") & (p.dset == pc(cfg)["sets"][0])]
    if configs is not None:
        q = q[q.config.isin(configs)]
    rows = []
    for c, g in q.groupby("config"):
        N = int(g.N.iloc[0])
        conf = norm_conf({k: g[k].iloc[0] for k in ("family", "N", "alpha", "delta") if k in g and g[k].notna().any()})
        r = {"config": c, "họ": fam_of(c), "h": h_of(conf), "L_D(μ)": float(g.L_mu.mean()), "V₁": float((g.Ve + g.Vo).mean()),
             "2V_e": float((2 * g.Ve).mean()), "ρ": float((2 * g.Ve).mean() / (g.Ve + g.Vo).mean())}
        for B in budgets:
            K = BO.k_of(B, N, P["M"])
            if K is None:
                continue
            for cp in ("rev", "indep"):
                if BO.coupling_ok(cp, K):
                    r[f"J_{cp} B{B}"] = float(BO.j_val(g.L_mu, g.Ve, g.Vo, K, cp).mean())
        rows.append(r)
    t = pd.DataFrame(rows).sort_values(["h", "họ"], ascending=[False, True])
    return md(t)


def same_h(cfg: dict, stage: str, set_name: str) -> str:
    """Cùng α (cùng h): hiệu L(μ) và V₁ của ln / cmix so với spi, CI theo ảnh (mô tả). Khớp h KHÔNG bảo đảm khớp J."""
    import pandas as pd
    from .report import md
    P = pr(cfg)
    p = _read(stage, "pairs", cfg)
    if p is None:
        return ""
    nb, seed, al = int(P["bootstrap"]), int(P["seed"]), float(P["alpha"])
    q = p[(p.set == set_name) & (p.group == "all") & (p.dset == pc(cfg)["sets"][0])].assign(V1=lambda x: x.Ve + x.Vo)
    rows = []
    for a in pc(cfg)["grid"]["alphas"]:
        ref = f"spi_a{float(a):g}_N{int(pc(cfg)['grid']['N'])}"
        for fam in ("ln", "cmix"):
            other = f"{fam}_a{float(a):g}_N{int(pc(cfg)['grid']['N'])}"
            r = {"α (h)": f"{a} ({1 / (float(a) + 1):.3f})", "so sánh": f"{fam} − spi"}
            ok = False
            for lab, col in (("ΔL(μ)", "L_mu"), ("ΔV₁", "V1")):
                W = q.pivot_table(index="image", columns="config", values=col, aggfunc="first")
                if ref in W and other in W:
                    v = (W[other] - W[ref]).dropna().values
                    lo, hi = ci_of(v[boot_idx(len(v), nb, seed)].mean(1), al)
                    r[f"{lab} [CI]"] = fmt_ci(float(v.mean()), lo, hi)
                    r[f"{lab} / TB spi"] = float(v.mean() / W[ref].mean())
                    ok = True
            if ok:
                rows.append(r)
    return md(pd.DataFrame(rows)) if rows else ""


def dev_section(cfg: dict) -> list:
    import pandas as pd
    from .report import md
    P = pr(cfg)
    if _read("dev", "pairs", cfg) is None:
        return []
    B_all = P["budgets"] + P["budgets_secondary"]
    L = ["## Dev — lựa chọn theo J_rev, phân rã, cùng h, lực\n",
         "**Phân rã — bộ ô CHỌN** (J_rev và J_indep là hai đại lượng khác nhau; chỉ so J_rev với J_rev):\n", decomp(cfg, "dev", "sel", B_all),
         "**Phân rã — bộ ô ĐÁNH GIÁ** (độc lập):\n", decomp(cfg, "dev", "eval", B_all),
         "**Cùng exploration h** (bộ ô đánh giá; mô tả — dự đoán Mệnh đề: ở cùng D, khác biệt chủ yếu ở V hơn L(μ); khớp h không bảo đảm khớp J):\n",
         same_h(cfg, "dev", "eval")]
    try:
        sel = load_sel(cfg)
    except SystemExit:
        return L
    rows = []
    for B, d in sel["chosen"].items():
        for fam, s in d.items():
            ps = sorted(s["p_select"].items(), key=lambda x: -x[1])
            rows.append({"B": B, "họ": fam, "θ*_F (J_rev)": s["config"], "TB J_rev": s["J_mean"],
                         "P_select (đầy đủ)": "; ".join(f"{k} {v:.2f}" for k, v in ps if v > 0)})
    L.append(f"**Lựa chọn trong từng họ** (argmin TB J_rev, bộ ô CHỌN; incumbent `{sel['incumbent']}|rev`):\n")
    L.append(md(pd.DataFrame(rows)))
    pt = power_table(cfg, sel)
    L.append(f"**Lực cho NI** (bộ ô đánh giá dev; ε = {P['ni_margin']:.0%}; một phía α/4 = {float(P['alpha']) / 4:.4f}; lực {P['power']}; "
             f"giả định g thật = 0 — cột `n_req_gdev` dùng g quan sát, chỉ tham khảo). n test = {pt['n_test']} (cần {pt['n_required']}); "
             f"thiếu lực: {pt['underpowered'] or 'không'}. Thiếu lực ⇒ báo cáo trung thực, KHÔNG đổi ε / endpoint.\n")
    L.append(md(pd.DataFrame(pt["rows"])))
    return L


def emp_ci_method(cfg: dict) -> tuple:
    from . import budget2 as B2
    from . import jcheck_calib as JC
    m = pr(cfg)["emp_ci_method"]
    miss = {}
    if m == "auto":
        f = os.path.join(B2.root(cfg), "audit_sim_coverage.csv")
        if os.path.exists(f):
            import pandas as pd
            t = pd.read_csv(f)
            m = JC.choose_method(t.to_dict("records"))["chosen"]
            dat = t[(t["phương pháp"] == m) & t["kịch bản"].str.startswith("dữ liệu")]
            miss = {(c_, int(k_)): 1 - float(g_["độ phủ"].mean()) for (c_, k_), g_ in dat.groupby(["coupling", "K"])}
        else:
            m = "bootstrap-t"
    return m, miss


def test_section(cfg: dict) -> list:
    import pandas as pd
    from . import jcheck_calib as JC
    from .report import md
    b, P = pc(cfg), pr(cfg)
    if _read("test", "pairs", cfg) is None:
        return []
    sel = load_sel(cfg)
    nb, seed, al = int(P["bootstrap"]), int(P["seed"]), float(P["alpha"])
    eps = float(P["ni_margin"])
    inc = cname(b["incumbent"])
    jl, p = jl_of(cfg, "test", "eval")
    L = [f"## Test — kiểm định (bộ ô ĐÁNH GIÁ, D chính `{b['sets'][0]}`, J_rev so với incumbent `{inc}|rev`)\n",
         f"NI chính: H0 g ≤ −{eps:.0%}, bootstrap-t một phía, Holm qua 4 (ln, cmix × B {P['budgets']}). Superiority (H0 g ≤ 0) và TOST "
         f"(|g| ≥ {float(P['eq_margin']):.0%}) mỗi loại một Holm qua cùng 4 phép. Log-tỷ số theo ảnh chỉ là chẩn đoán.\n"]
    rows = []
    for B in P["budgets"]:
        W = wide(jl, B, "rev")
        for fam in b["primary_families"] + ["smooth"]:
            s = sel["chosen"].get(str(B), {}).get(fam)
            if not s or s["config"] not in W or inc not in W:
                continue
            j = W[[inc, s["config"]]].dropna()
            r = tests_for(j[inc].values, j[s["config"]].values, eps, nb, seed, al)
            rows.append({"B": int(B), "họ": fam, "config": s["config"], "chính": fam in b["primary_families"], **r})
    if rows:
        prim = [r for r in rows if r["chính"]]
        for key in ("p_ni", "p_sup", "p_tost"):
            for r, h in zip(prim, holm([r[key] for r in prim])):
                r[key + "_holm"] = h
        out = []
        for r in rows:
            o = {"B": r["B"], "họ": r["họ"], "config": r["config"], "n": r["n"], "J_rev inc": r["J_inc"], "J_rev F": r["J_F"],
                 "g [CI percentile]": fmt_ci(r["g"], r["g_lo"], r["g_hi"], "{:.3f}"), "g [CI Fieller]": fmt_ci(r["g"], r["g_flo"], r["g_fhi"], "{:.3f}"),
                 "ảnh F tốt hơn": f"{r['wins']}/{r['n']}"}
            if r["chính"]:
                o.update({"p NI (Holm)": r["p_ni_holm"], "p sup (Holm)": r["p_sup_holm"], "p TOST (Holm)": r["p_tost_holm"],
                          "NI": decide(r, eps, al), "tốt hơn": "có" if r["p_sup_holm"] < al else "chưa",
                          f"tương đương ±{float(P['eq_margin']):.0%}": "có" if r["p_tost_holm"] < al else "chưa"})
            else:
                o["NI"] = "mô tả (smooth không trong kiểm định chính)"
            o["log(J_F/J_inc) TB [CI] (chẩn đoán)"] = fmt_ci(r["logratio"], r["logratio_lo"], r["logratio_hi"], "{:.3f}")
            out.append(o)
        L.append(md(pd.DataFrame(out)))
    # J_indep riêng (không coupling)
    rows = []
    for B in P["budgets"]:
        W = wide(jl, B, "indep")
        names = [inc] + [sel["chosen"].get(str(B), {}).get(f, {}).get("config") for f in b["primary_families"] + ["smooth"]]
        for c in [n for n in names if n and n in W]:
            v = W[c].dropna()
            rows.append({"B": int(B), "config": c, "TB J_indep": float(v.mean())})
        for c in [n for n in names[1:] if n and n in W and inc in W]:
            v = (W[inc] - W[c]).dropna().values
            lo, hi = ci_of(v[boot_idx(len(v), nb, seed)].mean(1), al)
            rows.append({"B": int(B), "config": f"{c} so với {inc} (cả hai indep)", "J_indep inc − J_indep F [CI]": fmt_ci(float(v.mean()), lo, hi)})
    if rows:
        L.append("**J_indep (không coupling; mô tả, KHÔNG so với J_rev)**:\n")
        L.append(md(pd.DataFrame(rows)))
    L.append("**Phân rã trên test** và **cùng h** (mô tả):\n")
    L.append(decomp(cfg, "test", "eval", P["budgets"] + P["budgets_secondary"]))
    L.append(same_h(cfg, "test", "eval"))
    # đối chứng nối mạch
    rows = []
    for B in P["budgets"]:
        W = wide(jl, B, "rev")
        for c in [cname(x) for x in b["controls"]]:
            if c in W and inc in W:
                v = (W[c] - W[inc]).dropna().values
                lo, hi = ci_of(v[boot_idx(len(v), nb, seed)].mean(1), al)
                rows.append({"B": int(B), "đối chứng": c, "J_rev(đối chứng) − J_rev(inc) [CI]": fmt_ci(float(v.mean()), lo, hi)})
    if rows:
        L.append("**Nối mạch với BUDGET / E10** (mô tả; dương ⇒ incumbent tốt hơn):\n")
        L.append(md(pd.DataFrame(rows)))
    # độ bền theo D
    rows = []
    for B in P["budgets"]:
        for fam in b["primary_families"]:
            s = sel["chosen"].get(str(B), {}).get(fam)
            if not s:
                continue
            cols = {}
            for dn in list(dict.fromkeys(p.dset)):
                jd = BO.j_long(p, "eval", dn, [B], P["M"])
                W = jd[jd.coupling == "rev"].pivot_table(index="image", columns="config", values="J", aggfunc="first")
                if s["config"] in W and inc in W:
                    cols[dn] = (W[inc] - W[s["config"]]) / W[inc].mean()
            M_ = pd.DataFrame(cols).dropna()
            if not len(M_):
                continue
            X = M_.values
            bs = X[boot_idx(len(X), nb, seed)].mean(1)
            ph = holm([boot_p(bs[:, k]) for k in range(X.shape[1])])
            for k, dn in enumerate(M_.columns):
                lo, hi = ci_of(bs[:, k], al)
                rows.append({"B": int(B), "họ": fam, "D": dn, "g theo D [CI]": fmt_ci(float(X[:, k].mean()), lo, hi, "{:.3f}"), "p Holm (qua D)": ph[k]})
    if rows:
        L.append("**Độ bền theo D** (g = (J_inc − J_F)/J̄_inc theo từng D; mô tả):\n")
        L.append(md(pd.DataFrame(rows)))
    meth, miss = emp_ci_method(cfg)
    L.append(f"**Kiểm J thực nghiệm** (CI `{meth}` — phương pháp đã hiệu chuẩn ở budget2-audit, nếu có):\n")
    L += JC.recheck(view(cfg), ("dev", "test"), nb, seed, al, meth, miss)
    L += ext_section(cfg, sel)
    L += cost_section(cfg)
    return L


def ext_section(cfg: dict, sel: dict) -> list:
    import pandas as pd
    from .report import md
    ex = _read("test", "ext", cfg)
    if ex is None:
        return []
    b, P = pc(cfg), pr(cfg)
    nb, seed, al = int(P["bootstrap"]), int(P["seed"]), float(P["alpha"])
    inc = cname(b["incumbent"])
    ex = ex.assign(key=ex.config + "|" + ex.estimator)
    Wd = {k: g.set_index("image").sort_index() for k, g in ex.groupby("key")}

    def key(c, B):
        N = int(c.split("_N")[1].split("_")[0])
        return f"{c}|rev_B{B}_K{BO.k_of(B, N, P['M'])}"
    comps = []
    for B in P["budgets"]:
        for fam in b["primary_families"] + ["smooth"]:
            s = sel["chosen"].get(str(B), {}).get(fam)
            if s:
                comps += [(key(s["config"], B), key(inc, B), f"B = {B}"), (f"{s['config']}|mean_all", f"{inc}|mean_all", "Ē 2m đường")]
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
    return ["## Metric ngoài (kết quả phụ, mô tả; KHÔNG dùng để chọn)\n", md(pd.DataFrame(rows))]


def cost_section(cfg: dict) -> list:
    import pandas as pd
    from .report import md
    rows = []
    for st in STAGES:
        c = _read(st, "cost", cfg)
        if c is None or "config" not in c:
            continue
        c = c[c.config.notna() & c.N.notna()] if "N" in c else c
        g = c.groupby("config").agg(grads=("grads", "mean"), paths=("paths", "mean"), seconds=("seconds", "mean"), N=("N", "first")).reset_index()
        g["grad/đường (đo)"] = g.grads / g.paths
        g["N·M"] = g.N * int(pr(cfg)["M"])
        g["giây/đường"] = g.seconds / g.paths
        g.insert(0, "giai đoạn", st)
        rows.append(g[["giai đoạn", "config", "grad/đường (đo)", "N·M", "giây/đường"]])
    if not rows:
        return []
    return ["## Kiểm toán chi phí thực tế (cost.csv)\n", "grad/đường phải bằng N·M cho mọi họ; giây/đường cho thấy chi phí sinh lịch trình.\n",
            md(pd.concat(rows, ignore_index=True))]


def write_report(cfg: dict) -> str:
    rp = report_path(cfg)
    b, P = pc(cfg), pr(cfg)
    lk = lock_path(cfg)
    L = ["# PATH_FAMILIES Phase 1 — Dirichlet có cần thiết? (J ở ngân sách gradient cố định)\n"]
    if cfg.get("_debug"):
        L.append("> **DEBUG** — chỉ kiểm pipeline.\n")
    L.append(f"- Họ chính {b['primary_families']}, mô tả {b['descriptive_families']}; lưới N = {b['grid']['N']}, α khớp {b['grid']['alphas']}, "
             f"smooth δ {b['grid']['smooth_delta']}. Incumbent `{cname(b['incumbent'])}|rev`; đối chứng {[cname(x) for x in b['controls']]}. "
             f"B chính {P['budgets']}, mô tả {P['budgets_secondary']}. Khoá test: "
             + (f"`{json.load(open(lk))['digest'][:12]}`" if os.path.exists(lk) else "**chưa khoá**") + ".")
    L.append("- Đơn vị thống kê là ảnh; bootstrap theo ảnh. Chọn bằng J_rev trên bộ ô CHỌN (dev); kiểm định trên ảnh mới, bộ ô ĐÁNH GIÁ.\n")
    L += dev_section(cfg)
    L += test_section(cfg)
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[pathfam] {rp}")
    return rp

