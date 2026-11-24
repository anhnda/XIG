"""
BUDGET — tối ưu chiến lược lấy mẫu SPI theo objective J ở ngân sách gradient cố định (nghiên cứu tiếp theo, khoá riêng).

    θ*(B, D) = argmin_{θ : C(θ) ≤ B} J_D(θ),   J_D(θ, K) = L_D(μ_θ) + V_K(θ),   B = K · N · M
    V_K = (V_e + V_o)/K  (indep)      V_K = 2·V_e/K  (rev, K chẵn, cặp độc lập)

Giai đoạn (bash scripts/run.sh <lệnh>):
  budget-dev1    G1: α × coupling × K trên ảnh dev (họ spi, N = 64). Chọn θ*(B) cho từng B chính theo TB J trên bộ ô CHỌN;
                 xác suất chọn bootstrap; ghi selection_dev1.json (gồm top-k α cho G2).
  budget-dev2    G2: từng chiều một — N ∈ {32, 64, 128} (spi), họ patch / hier với G — so ở CÙNG B (K = B/(N·M)). Quy tắc dừng
                 đặt trước: một chiều chỉ vào tập ứng viên cuối nếu TB J giảm ≥ dim_min_gain so với nghiệm G1 và P_boot ≥ dim_min_prob.
                 Ghi selection_dev2.json = θ*(B) cuối.
  budget-lock    khoá tiêu chí + lựa chọn cuối (sha của selection) → budget_prereg.json. Bắt buộc trước test.
  budget-test    G3 (+ G4): ảnh test mới; θ*(B), SPI baseline α = 10 (indep & rev), α = 0,1 đã xác nhận, IG cùng ngân sách;
                 8 D; kiểm J thực nghiệm (ensemble thật K đường) so với công thức; metric ngoài trên bản đồ ở K của ngân sách và
                 trên Ē của toàn bộ 2m đường (tách phần phương sai hữu hạn K).
  budget-report  dựng lại INFID_BUDGET_REPORT.md.
Tiêu chí chính (đăng ký trước, mới — không hồi tố cho nghiên cứu cũ): với mỗi B chính,
    R(B) = [J(base, B) − J(θ*, B)] / [J(base, B) − sàn]   (tỷ số của các trung bình theo ảnh; sàn = L*_comp patch out-of-sample)
xác nhận nếu p một phía (H0: R ≤ r₀) hiệu chỉnh Holm qua các B chính < α; bác bỏ nếu cận trên CI 95% < r₀; còn lại chưa kết luận.
base = SPI α = 10, N = 64, `indep` ở cùng B. Sàn patch ≥ sàn pixel ⇒ mẫu số bị ước lượng thấp ⇒ R hơi lạc quan (ghi rõ).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import time

import numpy as np
import torch

from . import config as C
from . import infid_core as IC
from . import infid_stats as ST
from . import sampling as SM
from .infid_e0 import icfg
from .infid_posthoc import EXT_DIR, boot_idx, boot_p, ci_of, fmt_ci, holm
from .paths import straight_nodes
from .quad import Counter
from .sampling_infid import PathGen, attr_of, build_dsets, cname, ext_metrics, flat_rows, norm_conf, prep_ext, stage_cfg, stage_images

F64 = torch.float64
BUD_VERSION = 1
STAGES = ("dev1", "dev2", "test")
COUPLINGS = ("indep", "rev")

DEFAULT = {
    "protocol": {"rule": "rand", "M": 1, "budgets": [256, 1024], "budgets_secondary": [128, 512, 2048], "Ks": [1, 2, 4, 8, 16, 32],
                 "check_K": [4, 16], "min_excess_reduction": 0.20, "dim_min_gain": 0.05, "dim_min_prob": 0.80,
                 "alpha": 0.05, "bootstrap": 10000, "seed": 31, "floor_grids": [14], "floor_lams": [0.0, 1.0e-3, 1.0e-2, 1.0e-1],
                 "floor_folds": 4, "n_flip": 0},
    "sets": ["del_sq"],
    "dev": {"split": "fresh", "offset": 700, "n_images": 32},
    "stage1": {"m": 32, "configs": [{"family": "spi", "alpha": a, "N": 64} for a in (0.1, 0.3, 1, 10)]},
    "stage2": {"m": 16, "alphas": "auto", "top_k": 3, "N": [32, 64, 128], "N_family": 64,
               "families": {"patch": {"G": [7, 14]}, "hier": {"G": [7, 14], "eta": [0.5]}}},
    "test": {"split": "fresh", "offset": 800, "n_images": 48, "m": 32, "selection": "dev2",
             "baseline": {"family": "spi", "alpha": 10, "N": 64}, "prior": {"family": "spi", "alpha": 0.1, "N": 64},
             "sets": ["del_sq", "ins_sq", "sym_sq", "frac25", "frac50", "frac75", "del_sq_blur", "del_sp"],
             "ig": True, "ig_N": 64, "ig_repeats": 32, "external": True,
             "road": {"fracs": [0.1, 0.3, 0.5, 0.7, 0.9], "noise": 0.01, "iters": 1000, "tol": 1.0e-5},
             "insdel_steps": 50, "pointing_tol": 15, "top_frac": 0.05},
    "gpu_minutes": {"dev1": None, "dev2": None, "test": None},
}


def bcfg(cfg: dict) -> dict:
    return C._merge(DEFAULT, cfg.get("budget", {}) or {})


def bpr(cfg: dict) -> dict:
    return bcfg(cfg)["protocol"]


def broot(cfg: dict) -> str:
    d = os.path.join(C.out_dir(cfg, "pilot"), "infid", "budget")
    os.makedirs(d, exist_ok=True)
    return d


def sel_path(cfg: dict, stage: str) -> str:
    return os.path.join(broot(cfg), f"selection_{stage}.json")


def lock_path(cfg: dict) -> str:
    return os.path.join(broot(cfg), "budget_prereg.json")


def primary_D(cfg: dict) -> str:
    return bcfg(cfg)["sets"][0]


# ─────────────────────────────────────────────────────────────────────────────
# hàm thuần (unit test)
# ─────────────────────────────────────────────────────────────────────────────

def k_of(B: int, N: int, M: int):
    """K = B / (N·M) nếu nguyên ≥ 1, ngược lại None."""
    q, r = divmod(int(B), int(N) * int(M))
    return q if r == 0 and q >= 1 else None


def coupling_ok(coupling: str, K) -> bool:
    return K is not None and (coupling == "indep" or (K % 2 == 0 and K >= 2))


def j_val(L, Ve, Vo, K, coupling: str):
    """J = L(μ) + V_K: indep (V_e + V_o)/K; rev 2·V_e/K."""
    return L + (Ve + Vo) / K if coupling == "indep" else L + 2.0 * Ve / K


def empirical_J(y1: torch.Tensor, y2: torch.Tensor, d: torch.Tensor, Ks) -> dict:
    """infid TRUNG BÌNH của ensemble thật K đường dựng từ các đường đã lấy (nhóm rời nhau). rev: K/2 cặp; indep: K thành viên đầu
    (hoặc K thành viên sau) của K cặp khác nhau — mỗi nhóm là ước lượng không chệch của J. Không tốn gradient."""
    m = int(y1.shape[0])
    out = {}
    for K in Ks:
        K = int(K)
        if K % 2 == 0 and m // (K // 2) >= 1:
            h = K // 2
            ng = m // h
            G = ((y1[:ng * h] + y2[:ng * h]) / 2).view(ng, h, -1).mean(1)
            out[f"Jemp_rev_{K}"], out[f"Jemp_rev_{K}_ng"] = float(IC.infid(G, d).mean()), ng
        ng = m // K
        if ng >= 1:
            G1 = y1[:ng * K].view(ng, K, -1).mean(1)
            G2 = y2[:ng * K].view(ng, K, -1).mean(1)
            out[f"Jemp_indep_{K}"], out[f"Jemp_indep_{K}_ng"] = float(torch.cat([IC.infid(G1, d), IC.infid(G2, d)]).mean()), 2 * ng
    return out


def j_long(p, set_name: str, dset: str, budgets, M: int):
    """Bảng dài: ảnh × cấu hình × coupling × B → J. Chỉ các (coupling, K) khả thi."""
    import pandas as pd
    q = p[(p.set == set_name) & (p.group == "all") & (p.dset == dset)]
    rows = []
    for _, r in q.iterrows():
        for B in budgets:
            K = k_of(B, int(r.N), M)
            for cp in COUPLINGS:
                if coupling_ok(cp, K):
                    rows.append({"image": r.image, "config": r.config, "family": r.family, "alpha": r.get("alpha"), "N": int(r.N),
                                 "G": r.get("G"), "eta": r.get("eta"), "coupling": cp, "B": int(B), "K": int(K),
                                 "cand": f"{r.config}|{cp}", "J": j_val(r.L_mu, r.Ve, r.Vo, K, cp), "L_mu": r.L_mu})
    return pd.DataFrame(rows)


def select_budget(jl, B: int, n_boot: int, seed: int, allowed=None) -> dict:
    """argmin TB J theo ảnh trên các ứng viên ở B; P_boot(mỗi ứng viên được chọn)."""
    q = jl[jl.B == int(B)]
    if allowed is not None:
        q = q[q.cand.isin(allowed)]
    W = q.pivot_table(index="image", columns="cand", values="J", aggfunc="first").dropna(axis=0)
    if W.shape[1] == 0 or W.shape[0] == 0:
        return {}
    X = W.values
    means = X.mean(0)
    idx = boot_idx(X.shape[0], n_boot, seed)
    win = np.bincount(X[idx].mean(1).argmin(1), minlength=X.shape[1]) / len(idx)
    order = np.argsort(means)
    best = W.columns[order[0]]
    meta = q[q.cand == best].iloc[0]
    return {"B": int(B), "chosen": best, "K": int(meta.K), "coupling": meta.coupling, "config": meta.config, "n_images": int(X.shape[0]),
            "table": [{"cand": W.columns[i], "J_mean": float(means[i]), "p_select": float(win[i])} for i in order]}


def conf_of(p, config_name: str) -> dict:
    r = p[p.config == config_name].iloc[0]
    c = {"family": r.family, "N": int(r.N)}
    if r.family != "ig":
        c["alpha"] = float(r.alpha)
    if r.family in ("patch", "hier"):
        c["G"] = int(r.G)
    if r.family == "hier":
        c["eta"] = float(r.eta)
    return norm_conf(c)


# ─────────────────────────────────────────────────────────────────────────────
# lấy mẫu một cấu hình (giữ chiếu + bản đồ cho metric ngoài)
# ─────────────────────────────────────────────────────────────────────────────

def pair_maps(E, conf: dict, m: int, pr: dict, dsets: dict, tag: str, keep) -> dict:
    """m cặp (Z, RZ) độc lập; chiếu lên mọi D. keep: [(nhãn, kiểu, K)] với kiểu ∈ rev (K/2 cặp đầu, cả hai thành viên),
    indep (thành viên đầu của K cặp đầu), mean (toàn bộ 2m đường)."""
    pg = PathGen(E, conf, tag)
    t0, n0 = time.time(), Counter.n
    Y = {(nm, s): ([], []) for nm, d in dsets.items() for s in d}
    acc = {lab: torch.zeros(E.shape, dtype=F64, device=E.x.device) for lab, _, _ in keep}
    sums = []
    for j in range(int(m)):
        Z = pg.Z(j)
        for w, Zx in ((0, Z), (1, SM.reverse_nodes(Z))):
            A = attr_of(E, SM.nodes(E.x0, E.dx, Zx), pr, (tag, pg.name, E.image, j, w))
            sums.append(float(A.sum()))
            for (nm, s), (y1, y2) in Y.items():
                (y1 if w == 0 else y2).append(dsets[nm][s].project(A)[0])
            for lab, kind, K in keep:
                if kind == "rev" and j < K // 2:
                    acc[lab] += A / K
                elif kind == "indep" and w == 0 and j < K:
                    acc[lab] += A / K
                elif kind == "mean":
                    acc[lab] += A / (2 * int(m))
            del A
        del Z
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"Y": {k: (torch.stack(a), torch.stack(b)) for k, (a, b) in Y.items()}, "maps": acc, "sums": sums,
            "grads": Counter.n - n0, "seconds": time.time() - t0, "name": pg.name}


# ─────────────────────────────────────────────────────────────────────────────
# cấu hình giai đoạn
# ─────────────────────────────────────────────────────────────────────────────

def load_sel(cfg: dict, stage: str) -> dict:
    p = sel_path(cfg, stage)
    if not os.path.exists(p):
        raise SystemExit(f"chưa có {p} — chạy bash scripts/run.sh budget-{stage} trước")
    with open(p) as fh:
        return json.load(fh)


def stage2_alphas(cfg: dict) -> list:
    s2 = bcfg(cfg)["stage2"]
    if s2["alphas"] != "auto":
        return [float(a) for a in s2["alphas"]]
    return [float(a) for a in load_sel(cfg, "dev1")["top_alphas"]][: int(s2["top_k"])]


def stage_sc(cfg: dict, stage: str) -> dict:
    b = bcfg(cfg)
    if stage == "dev1":
        sc = {**b["dev"], "m": int(b["stage1"]["m"]), "sets": b["sets"], "configs": [norm_conf(c) for c in b["stage1"]["configs"]],
              "external": False, "ig": False}
    elif stage == "dev2":
        s2 = b["stage2"]
        al = stage2_alphas(cfg)
        cf = [norm_conf({"family": "spi", "alpha": a, "N": n}) for a in al for n in s2["N"] if int(n) != n_ref(cfg)]
        fam = s2.get("families") or {}
        for a in al:
            for G in (fam.get("patch") or {}).get("G", []):
                cf.append(norm_conf({"family": "patch", "alpha": a, "N": s2["N_family"], "G": G}))
            for G in (fam.get("hier") or {}).get("G", []):
                for e in (fam.get("hier") or {}).get("eta", [0.5]):
                    cf.append(norm_conf({"family": "hier", "alpha": a, "N": s2["N_family"], "G": G, "eta": e}))
        sc = {**b["dev"], "m": int(s2["m"]), "sets": b["sets"], "configs": cf, "external": False, "ig": False, "alphas": al}
    elif stage == "test":
        t = b["test"]
        sel = load_sel(cfg, t["selection"])
        cf, seen = [], set()
        for c in [v["conf"] for v in sel["final"].values()] + [t["baseline"], t["prior"]]:
            c = norm_conf(c)
            if cname(c) not in seen:
                seen.add(cname(c))
                cf.append(c)
        sc = {**t, "configs": cf}
    else:
        raise SystemExit(stage)
    return sc


def used_infid_images(ctx, cfg: dict) -> set:
    used = set()
    for st in ("pilot", "e1a", "e1b-dev", "e1b-test", "e2"):
        try:
            used |= set(stage_images(ctx, stage_cfg(cfg, st)))
        except (SystemExit, KeyError):
            pass
    return used


def check_images(ctx, cfg: dict, stage: str, idxs: list) -> None:
    b = bcfg(cfg)
    other = stage_images(ctx, b["test"] if stage != "test" else b["dev"])
    if set(idxs) & set(other):
        raise SystemExit(f"[budget-{stage}] ảnh dev và test giao nhau — đổi offset")
    if not cfg.get("_debug"):
        inter = set(idxs) & used_infid_images(ctx, cfg)
        if inter:
            raise SystemExit(f"[budget-{stage}] {len(inter)} ảnh trùng với nghiên cứu infid trước — đổi offset")


# ─────────────────────────────────────────────────────────────────────────────
# một ảnh
# ─────────────────────────────────────────────────────────────────────────────

def digest(cfg: dict, stage: str, sc: dict, extra) -> str:
    pr = bpr(cfg)
    keys = ("rule", "M", "Ks", "check_K", "floor_grids", "floor_lams", "floor_folds", "budgets", "budgets_secondary")
    return IC._h({"v": BUD_VERSION, "stage": stage, "sc": {k: v for k, v in sc.items() if k not in ("configs", "split", "offset", "n_images", "alphas")},
                  "pr": {k: pr.get(k) for k in keys}, "D": icfg(cfg)["D"], "seed": cfg["seed"], "model": cfg["model"],
                  "data": cfg["data"], "extra": extra})


def maps_spec(conf: dict, cfg: dict) -> list:
    pr = bpr(cfg)
    out = []
    for B in pr["budgets"]:
        K = k_of(B, conf["N"], pr["M"])
        for cp in COUPLINGS:
            if coupling_ok(cp, K):
                out.append((f"{cp}_B{B}_K{K}", cp, K))
    out.append(("mean_all", "mean", 0))
    return out


def process(ctx, cfg: dict, stage: str, sc: dict, idx: int, out: str) -> dict:
    pr = bpr(cfg)
    image = f"seg{idx:05d}"
    cdir = os.path.join(out, "cache", image)
    os.makedirs(cdir, exist_ok=True)
    rows = {k: [] for k in ("pairs", "emp", "floors", "ig", "ext", "cost")}
    st = {"E": None, "ds": None}

    def env():
        if st["E"] is None:
            st["E"] = IC.IEnv(ctx, idx)
        if st["ds"] is None:
            st["ds"] = build_dsets(st["E"], cfg, sc, "budget")
        return st["E"], st["ds"]
    Ks = [int(k) for k in pr["Ks"]]
    for conf in sc["configs"]:
        nm_c = cname(conf)
        cp = os.path.join(cdir, f"{nm_c}.pt")
        dg = digest(cfg, stage, sc, nm_c)
        if os.path.exists(cp) and torch.load(cp, weights_only=False).get("digest") == dg:
            for k, v in torch.load(cp, weights_only=False)["rows"].items():
                rows[k] += v
            continue
        E, ds = env()
        keep = maps_spec(conf, cfg) if (stage == "test" and sc.get("external")) else []
        res = pair_maps(E, conf, int(sc["m"]), pr, ds, f"BUD-{stage}", keep)
        base = {"stage": stage, "image": image, "config": nm_c, **norm_conf(conf), "rule": pr["rule"], "M": int(pr["M"])}
        rr = {k: [] for k in rows}
        for (nm, s_), (y1, y2) in res["Y"].items():
            d = ds[nm][s_].dF.to(y1.device)
            sts = IC.pair_stats(y1, y2, d, Ks, groups=ds[nm][s_].group.to(y1.device))
            rr["pairs"] += flat_rows({**base, "dset": nm, "set": s_}, sts)
            if s_ == "eval":
                a = sts["all"]
                emp = empirical_J(y1, y2, d, pr["check_K"])
                frm = {f"Jf_{c_}_{K}": j_val(a["L_mu"], a["Ve"], a["Vo"], K, c_) for K in pr["check_K"] for c_ in COUPLINGS
                       if coupling_ok(c_, K)}
                rr["emp"].append({**base, "dset": nm, "m": int(sc["m"]), **emp, **frm})
        if keep:
            P = prep_ext(E, sc)
            for lab, A in res["maps"].items():
                rr["ext"].append({**base, "estimator": lab, **ext_metrics(E, A, sc, P)})
        rr["cost"].append({**base, "grads": res["grads"], "seconds": res["seconds"], "paths": 2 * int(sc["m"])})
        torch.save({"digest": dg, "rows": rr}, cp)
        for k in rows:
            rows[k] += rr[k]
        print(f"    [budget-{stage}] {image} {nm_c}: {res['grads']} grad, {res['seconds']:.0f}s", flush=True)
        del res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # IG cùng ngân sách + sàn (một lần mỗi ảnh)
    cp = os.path.join(cdir, "_ig_floors.pt")
    dg = digest(cfg, stage, sc, "igfl")
    if os.path.exists(cp) and torch.load(cp, weights_only=False).get("digest") == dg:
        for k, v in torch.load(cp, weights_only=False)["rows"].items():
            rows[k] += v
        return rows
    E, ds = env()
    rr = {k: [] for k in rows}
    base = {"stage": stage, "image": image, "rule": pr["rule"], "M": int(pr["M"])}
    if sc.get("ig"):
        N = int(sc["ig_N"])
        r_ig = int(sc["ig_repeats"])
        t0, n0 = time.time(), Counter.n
        Xig = straight_nodes(E.x, E.x0, N)
        Aigs = [attr_of(E, Xig, pr, (f"BUD-{stage}", "ig", image, q)) for q in range(r_ig)]
        del Xig
        for nm, d in ds.items():
            for s_, dd in d.items():
                Yg = torch.cat([dd.project(A) for A in Aigs])
                stt = IC.indep_core(Yg, dd.dF.to(Yg.device), [1], c2_sizes=())
                row = {**base, "config": f"ig_N{N}", "family": "ig", "N": N, "dset": nm, "set": s_, "group": "all", "r": r_ig,
                       "L_mu": stt["L_mu"], "sigma2": stt["sigma2"]}
                for B in pr["budgets"] + pr["budgets_secondary"]:
                    K = k_of(B, N, pr["M"])
                    if K is not None:
                        row[f"J_B{B}"] = stt["L_mu"] + stt["sigma2"] / K
                rr["ig"].append(row)
        if sc.get("external"):
            P = prep_ext(E, sc)
            for B in pr["budgets"]:
                K = k_of(B, N, pr["M"])
                if K is not None and K <= r_ig:
                    rr["ext"].append({**base, "config": f"ig_N{N}", "family": "ig", "estimator": f"ig_B{B}_K{K}",
                                      **ext_metrics(E, sum(Aigs[:K]) / K, sc, P)})
        rr["cost"].append({**base, "config": f"ig_N{N}", "grads": Counter.n - n0, "seconds": time.time() - t0, "paths": r_ig})
        del Aigs
    for nm, d in ds.items():
        for G in pr["floor_grids"]:
            Xf, Xe = IC.patch_design(E, d["sel"], int(G)), IC.patch_design(E, d["eval"], int(G))
            fl = IC.floor_fit(Xf, d["sel"].dF.to(Xf.device), Xe, d["eval"].dF.to(Xf.device), E.D, [float(x) for x in pr["floor_lams"]],
                              int(pr["floor_folds"]), E.cgen("BUD-floor", stage, nm, image, G))
            rr["floors"].append({**base, "dset": nm, "G": int(G), **{k: v for k, v in fl.items() if not k.startswith("_")}})
    torch.save({"digest": dg, "rows": rr}, cp)
    for k in rows:
        rows[k] += rr[k]
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# chạy một giai đoạn
# ─────────────────────────────────────────────────────────────────────────────

def lock_payload(cfg: dict) -> dict:
    b = bcfg(cfg)
    sp = sel_path(cfg, b["test"]["selection"])
    sel_sha = hashlib.sha256(open(sp, "rb").read()).hexdigest() if os.path.exists(sp) else None
    D = icfg(cfg)["D"]
    return {"protocol": b["protocol"], "sets": b["sets"], "test": b["test"], "D": {k: D["sets"][k] for k in b["test"]["sets"]},
            "n_sel": D.get("n_sel"), "n_eval": D.get("n_eval"), "selection_sha256": sel_sha, "model": cfg["model"], "data": cfg["data"],
            "seed": cfg["seed"], "v": BUD_VERSION}


def lock(cfg: dict) -> None:
    b = bcfg(cfg)
    if not os.path.exists(sel_path(cfg, b["test"]["selection"])):
        raise SystemExit(f"chưa có selection_{b['test']['selection']}.json — chạy budget-{b['test']['selection']} trước khi khoá")
    ST.write_prereg(lock_path(cfg), IC._h(lock_payload(cfg)), {"payload": lock_payload(cfg)})


def estimate(cfg: dict, stage: str, sc: dict, per_eval: float) -> dict:
    pr = bpr(cfg)
    g = sum(2 * int(sc["m"]) * int(c["N"]) * int(pr["M"]) for c in sc["configs"])
    if sc.get("ig"):
        g += int(sc["ig_N"]) * int(pr["M"]) * int(sc["ig_repeats"])
    return {"grad_per_image": g, "minutes": int(sc["n_images"]) * g * per_eval / 60}


def run(ctx, stage: str) -> None:
    from .ctx import bench
    import pandas as pd
    cfg = ctx.cfg
    if stage not in STAGES:
        raise SystemExit(stage)
    if stage == "test":
        if not cfg["_debug"] and not os.path.exists(lock_path(cfg)):
            raise SystemExit(f"budget-test cần khoá: bash scripts/run.sh budget-lock (chưa thấy {lock_path(cfg)})")
        ST.require_prereg(lock_path(cfg), IC._h(lock_payload(cfg)), cfg["_debug"])
    sc = stage_sc(cfg, stage)
    out = os.path.join(broot(cfg), stage)
    os.makedirs(out, exist_ok=True)
    idxs = stage_images(ctx, sc)
    check_images(ctx, cfg, stage, idxs)
    bm = bench(ctx)
    est = estimate(cfg, stage, sc, bm["per_eval"])
    print(f"[budget-{stage}] {len(idxs)} ảnh × {len(sc['configs'])} cấu hình: {est['grad_per_image']} grad/ảnh ≈ {est['minutes']:.1f} phút "
          "(nếu chưa có cache)", flush=True)
    lim = (bcfg(cfg).get("gpu_minutes") or {}).get(stage)
    if lim is not None and est["minutes"] > float(lim):
        raise SystemExit(f"[budget-{stage}] ước tính {est['minutes']:.1f} phút > budget.gpu_minutes.{stage} = {lim}")
    allr = {}
    t0, n0 = time.time(), Counter.n
    for j, idx in enumerate(idxs):
        print(f"  [budget-{stage}] ảnh {j + 1}/{len(idxs)} seg{idx:05d}", flush=True)
        for k, v in process(ctx, cfg, stage, sc, idx, out).items():
            allr.setdefault(k, []).extend(v)
    for k, v in allr.items():
        if v:
            pd.DataFrame(v).to_csv(os.path.join(out, f"{k}.csv"), index=False)
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"time": _dt.datetime.now().isoformat(timespec="seconds"), "stage": stage, "images": [f"seg{i:05d}" for i in idxs],
                   "sc": sc, "protocol": bpr(cfg), "estimate": est, "minutes": (time.time() - t0) / 60, "grad_evals": Counter.n - n0,
                   "config_digest": IC._h(bcfg(cfg)), "debug": cfg["_debug"]}, fh, indent=1, default=str)
    if stage == "dev1":
        select_dev1(cfg)
    elif stage == "dev2":
        select_dev2(cfg)
    write_report(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# lựa chọn trên dev
# ─────────────────────────────────────────────────────────────────────────────

def _read(stage: str, name: str, cfg: dict):
    from .report import _read as rd
    return rd(os.path.join(broot(cfg), stage), name)


def _write_sel(cfg: dict, stage: str, rec: dict) -> None:
    p = sel_path(cfg, stage)
    if os.path.exists(p):
        old = json.load(open(p))
        if os.path.isdir(os.path.join(broot(cfg), "test", "cache")) and not cfg["_debug"] and stage == bcfg(cfg)["test"]["selection"]:
            if old.get("final") != rec.get("final"):
                raise SystemExit(f"{p} đã dùng cho test; không đổi lựa chọn sau khi đã mở tập test")
        rec["previous"] = {k: v for k, v in old.items() if k != "previous"}
    with open(p, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(f"[budget-{stage}] lựa chọn → {p}")


def select_dev1(cfg: dict) -> None:
    pr = bpr(cfg)
    p = _read("dev1", "pairs", cfg)
    jl = j_long(p, "sel", primary_D(cfg), pr["budgets"] + pr["budgets_secondary"], pr["M"])
    final, ranks = {}, {}
    for B in pr["budgets"]:
        s = select_budget(jl, B, int(pr["bootstrap"]), int(pr["seed"]))
        if s:
            s["conf"] = conf_of(p, s["config"])
            final[str(B)] = s
            # hạng α: J tốt nhất (qua coupling) của mỗi α ở B
            q = jl[jl.B == B].groupby(["alpha", "cand"]).J.mean().groupby("alpha").min().rank()
            for a, rk in q.items():
                ranks.setdefault(float(a), []).append(float(rk))
    top = sorted(ranks, key=lambda a: np.mean(ranks[a]))
    _write_sel(cfg, "dev1", {"stage": "dev1", "final": final, "top_alphas": top, "alpha_rank": {str(a): np.mean(v) for a, v in ranks.items()},
                             "time": _dt.datetime.now().isoformat(timespec="seconds"), "set": "sel", "dset": primary_D(cfg)})


def n_ref(cfg: dict) -> int:
    """N của G1 (mọi cấu hình stage1 phải cùng N)."""
    ns = {int(c.get("N", 64)) for c in bcfg(cfg)["stage1"]["configs"]}
    if len(ns) != 1:
        raise SystemExit(f"budget.stage1.configs phải cùng N (đang có {sorted(ns)})")
    return ns.pop()


def dim_of(row, N1: int) -> str:
    if row.family == "spi" and int(row.N) == N1:
        return f"G1 (spi, N = {N1})"
    if row.family == "spi":
        return "N"
    return f"họ {row.family}"


def select_dev2(cfg: dict) -> None:
    """Quy tắc dừng theo chiều (đặt trước trong budget.protocol): chiều d vào tập ứng viên cuối ở B nếu nghiệm tốt nhất của d
    giảm TB J ≥ dim_min_gain (tương đối) so với nghiệm G1 VÀ P_boot(J_d < J_G1) ≥ dim_min_prob (bootstrap theo ảnh, theo cặp)."""
    import pandas as pd
    pr = bpr(cfg)
    p1, p2 = _read("dev1", "pairs", cfg), _read("dev2", "pairs", cfg)
    al = [float(a) for a in stage_sc(cfg, "dev2")["alphas"]]
    p1 = p1[p1.alpha.astype(float).isin(al)] if p1 is not None else None
    p = pd.concat([x for x in (p1, p2) if x is not None], ignore_index=True)
    jl = j_long(p, "sel", primary_D(cfg), pr["budgets"] + pr["budgets_secondary"], pr["M"])
    jl["dim"] = [dim_of(r, n_ref(cfg)) for r in jl.itertuples()]
    sel1 = load_sel(cfg, "dev1")["final"]
    final, dims = {}, {}
    B_ = int(pr["bootstrap"])
    for B in pr["budgets"]:
        q = jl[jl.B == B]
        ref = sel1.get(str(B), {}).get("chosen")
        if ref not in set(q.cand):                                          # nghiệm G1 nằm ngoài top-α ⇒ dùng tốt nhất G1 trong top-α
            g1 = q[q.dim.str.startswith("G1")]
            ref = g1.groupby("cand").J.mean().idxmin()
        W = q.pivot_table(index="image", columns="cand", values="J", aggfunc="first").dropna()
        allowed = [c for c in q[q.dim.str.startswith("G1")].cand.unique()]
        dB = {}
        for dname, g in q[~q.dim.str.startswith("G1")].groupby("dim"):
            bc = g.groupby("cand").J.mean().idxmin()
            dv = (W[bc] - W[ref]).values
            idx = boot_idx(len(dv), B_, int(pr["seed"]))
            pb = float((dv[idx].mean(1) < 0).mean())
            gain = float((W[ref].mean() - W[bc].mean()) / W[ref].mean())
            ok = gain >= float(pr["dim_min_gain"]) and pb >= float(pr["dim_min_prob"])
            dB[dname] = {"best": bc, "gain_rel": gain, "p_better": pb, "pass": ok, "J_best": float(W[bc].mean()), "J_ref": float(W[ref].mean())}
            if ok:
                allowed += list(g.cand.unique())
        s = select_budget(jl, B, B_, int(pr["seed"]), allowed=allowed)
        s["conf"] = conf_of(p, s["config"])
        s["ref_G1"] = ref
        final[str(B)] = s
        dims[str(B)] = dB
    _write_sel(cfg, "dev2", {"stage": "dev2", "final": final, "dims": dims, "alphas": al,
                             "rule": {"dim_min_gain": pr["dim_min_gain"], "dim_min_prob": pr["dim_min_prob"]},
                             "time": _dt.datetime.now().isoformat(timespec="seconds"), "set": "sel", "dset": primary_D(cfg)})


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo
# ─────────────────────────────────────────────────────────────────────────────

def _floor(cfg, stage, dset):
    fl = _read(stage, "floors", cfg)
    if fl is None:
        return None
    G = min(bpr(cfg)["floor_grids"])
    return fl[(fl.dset == dset) & (fl.G == G)].set_index("image").L_comp_oos


def dev_section(cfg: dict, stage: str) -> list:
    import pandas as pd
    from .report import md
    pr = bpr(cfg)
    p = _read(stage, "pairs", cfg)
    if p is None:
        return []
    if stage == "dev2":
        p1 = _read("dev1", "pairs", cfg)
        al = [float(a) for a in stage_sc(cfg, "dev2")["alphas"]]
        if p1 is not None:
            p = pd.concat([p1[p1.alpha.astype(float).isin(al)], p], ignore_index=True)
    Dp = primary_D(cfg)
    L = [f"## {'G1 — α × coupling × K' if stage == 'dev1' else 'G2 — N, họ đường, G ở cùng ngân sách B'} (dev, D chính `{Dp}`)\n"]
    budgets = pr["budgets"] + pr["budgets_secondary"]
    for set_name, title in (("sel", "bộ ô CHỌN (dùng để chọn)"), ("eval", "bộ ô ĐÁNH GIÁ (độc lập, mô tả)")):
        jl = j_long(p, set_name, Dp, budgets, pr["M"])
        t = jl.groupby(["B", "cand", "N", "K", "coupling"]).agg(J=("J", "mean"), L_mu=("L_mu", "mean"), n=("image", "nunique")).reset_index()
        t = t.sort_values(["B", "J"])
        L.append(f"**TB J theo ngân sách B = K·N·M — {title}** (B chính: {pr['budgets']}):\n")
        L.append(md(t))
    sel = None
    try:
        sel = load_sel(cfg, stage)
    except SystemExit:
        pass
    if sel:
        L.append("**Lựa chọn θ*(B)** (argmin TB J trên bộ ô chọn; P_select = xác suất bootstrap theo ảnh):\n")
        rows = []
        for B, s in sel["final"].items():
            top3 = "; ".join(f"{x['cand']} ({x['J_mean']:.3g}, P {x['p_select']:.2f})" for x in s["table"][:3])
            rows.append({"B": B, "θ*(B)": s["chosen"], "K": s["K"], "P_select(θ*)": s["table"][0]["p_select"], "3 ứng viên đầu": top3})
        L.append(md(pd.DataFrame(rows)))
        if stage == "dev1":
            L.append(f"Top α cho G2 (hạng TB qua các B chính): {sel.get('top_alphas')}.\n")
        if stage == "dev2":
            rows = []
            for B, dd in sel["dims"].items():
                for dn, v in dd.items():
                    rows.append({"B": B, "chiều": dn, "tốt nhất": v["best"], "J tốt nhất": v["J_best"], "J nghiệm G1": v["J_ref"],
                                 "giảm tương đối": v["gain_rel"], "P_boot(tốt hơn)": v["p_better"], "vào tập cuối": v["pass"]})
            L.append(f"**Quy tắc dừng theo chiều** (giảm ≥ {pr['dim_min_gain']:.0%} và P ≥ {pr['dim_min_prob']}):\n")
            L.append(md(pd.DataFrame(rows)))
    em = _read(stage, "emp", cfg)
    if em is not None:
        L.append(emp_table(cfg, em))
    return L


def emp_table(cfg: dict, em) -> str:
    """Kiểm suy luận phương sai: J thực nghiệm (ensemble thật, nhóm rời) − J công thức; CI theo ảnh phải chứa 0."""
    import pandas as pd
    from .report import md
    pr = bpr(cfg)
    B_, seed, al = int(pr["bootstrap"]), int(pr["seed"]), float(pr["alpha"])
    q = em[em.dset == primary_D(cfg)]
    rows = []
    for c, g in q.groupby("config"):
        for K in pr["check_K"]:
            for cp in COUPLINGS:
                a, b = f"Jemp_{cp}_{K}", f"Jf_{cp}_{K}"
                if a in g and b in g and g[a].notna().any():
                    v = (g[a] - g[b]).dropna().values
                    lo, hi = ci_of(v[boot_idx(len(v), B_, seed)].mean(1), al)
                    rows.append({"config": c, "coupling": cp, "K": int(K), "TB J thực nghiệm": float(g[a].mean()), "TB J công thức": float(g[b].mean()),
                                 "chênh [CI]": fmt_ci(float(v.mean()), lo, hi), "CI chứa 0": bool(lo <= 0 <= hi),
                                 "số nhóm / ảnh": int(g.get(f"{a}_ng", pd.Series([0])).iloc[0])})
    return ("**Kiểm J thực nghiệm (ensemble thật K đường dựng từ các đường đã lấy, nhóm rời nhau) so với công thức L + V_K** — "
            "chênh theo ảnh phải có CI chứa 0:\n\n" + md(pd.DataFrame(rows)))


def test_section(cfg: dict) -> list:
    import pandas as pd
    from .report import md
    pr = bpr(cfg)
    b = bcfg(cfg)
    B_, seed, al = int(pr["bootstrap"]), int(pr["seed"]), float(pr["alpha"])
    p, ig = _read("test", "pairs", cfg), _read("test", "ig", cfg)
    if p is None:
        return []
    sel = load_sel(cfg, b["test"]["selection"])
    base, prior = cname(b["test"]["baseline"]), cname(b["test"]["prior"])
    Dp = primary_D(cfg)
    r0 = float(pr["min_excess_reduction"])
    L = [f"## G3 — kiểm định trên ảnh test mới (D chính `{Dp}`, bộ ô ĐÁNH GIÁ)\n",
         f"Tiêu chí đăng ký trước: R(B) = [J(base) − J(θ*)] / [J(base) − sàn] ≥ {r0:.0%}, base = `{base}` `indep` ở cùng B; Holm qua các B "
         f"chính {pr['budgets']} (p một phía, H0: R ≤ {r0:.0%}). Sàn = L*_comp patch {min(pr['floor_grids'])}×{min(pr['floor_grids'])} "
         "out-of-sample (≥ sàn pixel ⇒ R hơi lạc quan).\n"]
    budgets = pr["budgets"]
    jl = j_long(p, "eval", Dp, budgets + pr["budgets_secondary"], pr["M"])
    fl = _floor(cfg, "test", Dp)
    if fl is None:
        L.append("> thiếu floors.csv — không tính được R(B).\n")
        fl = pd.Series(dtype=float, name="f")
    rows, ps = [], []
    for B in budgets:
        s = sel["final"].get(str(B))
        if not s:
            continue
        q = jl[jl.B == B]
        W = q.pivot_table(index="image", columns="cand", values="J", aggfunc="first")
        bcand = f"{base}|indep"
        if s["chosen"] not in W or bcand not in W:
            continue
        j = pd.concat([W[s["chosen"]].rename("t"), W[bcand].rename("b"), fl.rename("f")], axis=1).dropna()
        n = len(j)
        idx = boot_idx(n, B_, seed)
        t, bb, f = j.t.values, j.b.values, j.f.values
        R = (bb.mean() - t.mean()) / (bb.mean() - f.mean())
        Rb = (bb[idx].mean(1) - t[idx].mean(1)) / (bb[idx].mean(1) - f[idx].mean(1))
        lo, hi = ci_of(Rb, al)
        p1 = float((1 + (Rb <= r0).sum()) / (len(Rb) + 1))
        ps.append(p1)
        dJ = bb - t
        lo2, hi2 = ci_of(dJ[idx].mean(1), al)
        rp = (bb - t) / np.where(bb - f > 0, bb - f, np.nan)
        rows.append({"B": B, "θ*(B)": s["chosen"], "K": s["K"], "n ảnh": n, "J base (indep)": float(bb.mean()), "J θ*": float(t.mean()),
                     "sàn TB": float(f.mean()), "R (tỷ số TB) [CI 95%]": fmt_ci(float(R), lo, hi, "{:.3f}"),
                     "R TB theo ảnh (độ nhạy)": float(np.nanmean(rp)), "ΔJ = J base − J θ* [CI]": fmt_ci(float(dJ.mean()), lo2, hi2),
                     "ảnh θ* tốt hơn": f"{int((dJ > 0).sum())}/{n}", "p một phía (R ≤ r₀)": p1, "_hi": hi})
    if rows:
        ph = holm(ps)
        for r_, h in zip(rows, ph):
            r_["p Holm"] = h
            r_["quyết định"] = (ST.CONFIRM if h < al else ST.REJECT if r_["_hi"] < r0 else ST.OPEN)
            del r_["_hi"]
        L.append("**Tiêu chí chính**:\n")
        L.append(md(pd.DataFrame(rows)))
    # phụ: θ* so với α = 0,1 đã xác nhận, IG, rev baseline
    rows = []
    for B in budgets:
        s = sel["final"].get(str(B))
        if not s:
            continue
        q = jl[jl.B == B]
        W = q.pivot_table(index="image", columns="cand", values="J", aggfunc="first")
        comps = [(f"{base}|rev", "base rev"), (f"{prior}|rev", "α = 0,1 rev (đã xác nhận)"), (f"{prior}|indep", "α = 0,1 indep")]
        for cand, lab in comps:
            if cand in W and s["chosen"] in W and cand != s["chosen"]:
                v = (W[cand] - W[s["chosen"]]).dropna().values
                lo, hi = ci_of(v[boot_idx(len(v), B_, seed)].mean(1), al)
                rows.append({"B": B, "so với": lab, "ΔJ = J(so với) − J(θ*) [CI]": fmt_ci(float(v.mean()), lo, hi),
                             "ảnh θ* tốt hơn": f"{int((v > 0).sum())}/{len(v)}"})
        if ig is not None:
            gi = ig[(ig.set == "eval") & (ig.dset == Dp)].set_index("image")
            if f"J_B{B}" in gi and s["chosen"] in W:
                v = (gi[f"J_B{B}"] - W[s["chosen"]]).dropna().values
                lo, hi = ci_of(v[boot_idx(len(v), B_, seed)].mean(1), al)
                rows.append({"B": B, "so với": f"IG cùng ngân sách ({gi.config.iloc[0]})", "ΔJ = J(so với) − J(θ*) [CI]": fmt_ci(float(v.mean()), lo, hi),
                             "ảnh θ* tốt hơn": f"{int((v > 0).sum())}/{len(v)}"})
    if rows:
        L.append("**So sánh phụ ở cùng ngân sách** (mô tả; dương ⇒ θ* tốt hơn):\n")
        L.append(md(pd.DataFrame(rows)))
    # độ bền theo D: J(θ*) và L(μ) so với base
    rows = []
    for B in budgets:
        s = sel["final"].get(str(B))
        if not s:
            continue
        cols, colsL = {}, {}
        for dn in list(dict.fromkeys(p.dset)):
            jd = j_long(p, "eval", dn, [B], pr["M"])
            W = jd.pivot_table(index="image", columns="cand", values="J", aggfunc="first")
            WL = jd.pivot_table(index="image", columns="config", values="L_mu", aggfunc="first")
            if s["chosen"] in W and f"{base}|indep" in W:
                cols[dn] = W[f"{base}|indep"] - W[s["chosen"]]
                colsL[dn] = WL[base] - WL[s["config"]]
        for lab, M_ in (("J ở B", pd.DataFrame(cols).dropna()), ("L_D(μ)", pd.DataFrame(colsL).dropna())):
            if not len(M_):
                continue
            X = M_.values
            idx = boot_idx(len(X), B_, seed)
            bs = X[idx].mean(1)
            pv = [boot_p(bs[:, k]) for k in range(X.shape[1])]
            ph = holm(pv)
            for k, dn in enumerate(M_.columns):
                lo, hi = ci_of(bs[:, k], al)
                rows.append({"B": B, "đại lượng": lab, "D": dn, "base − θ* [CI]": fmt_ci(float(X[:, k].mean()), lo, hi),
                             "ảnh θ* tốt hơn": f"{int((X[:, k] > 0).sum())}/{len(X)}", "p Holm (qua D)": ph[k]})
    if rows:
        L.append("**Độ bền theo D** (mô tả; dương ⇒ θ* tốt hơn base; tách J ở K hữu hạn khỏi L_D(μ) kỳ vọng):\n")
        L.append(md(pd.DataFrame(rows)))
    t = jl.groupby(["B", "cand", "N", "K", "coupling"]).agg(J=("J", "mean"), L_mu=("L_mu", "mean"), n=("image", "nunique")).reset_index()
    L.append(f"**TB J mọi ứng viên trên test theo B** (mô tả, gồm B phụ {pr['budgets_secondary']}; KHÔNG dùng để chọn lại):\n")
    L.append(md(t.sort_values(["B", "J"])))
    em = _read("test", "emp", cfg)
    if em is not None:
        L.append(emp_table(cfg, em))
    L += ext_section(cfg, sel, base, prior)
    return L


def ext_section(cfg: dict, sel: dict, base: str, prior: str) -> list:
    """G4: metric ngoài theo cặp ảnh, Holm trong mỗi cặp so sánh; ở K của ngân sách VÀ trên Ē của 2m đường (tách phương sai K)."""
    import pandas as pd
    from .report import md
    ex = _read("test", "ext", cfg)
    if ex is None:
        return []
    pr = bpr(cfg)
    B_, seed, al = int(pr["bootstrap"]), int(pr["seed"]), float(pr["alpha"])
    ex = ex.assign(key=ex.config + "|" + ex.estimator)
    Wd = {k: g.set_index("image").sort_index() for k, g in ex.groupby("key")}
    t = bcfg(cfg)["test"]
    Nb, Np, Ng = int(t["baseline"]["N"]), int(t["prior"]["N"]), int(t["ig_N"])
    comps = []
    for B in pr["budgets"]:
        s = sel["final"].get(str(B))
        if not s:
            continue
        a = f"{s['config']}|{s['coupling']}_B{B}_K{s['K']}"
        comps += [(a, f"{base}|indep_B{B}_K{k_of(B, Nb, pr['M'])}", f"B = {B}"), (a, f"{prior}|rev_B{B}_K{k_of(B, Np, pr['M'])}", f"B = {B}"),
                  (a, f"ig_N{Ng}|ig_B{B}_K{k_of(B, Ng, pr['M'])}", f"B = {B}")]
        comps += [(f"{s['config']}|mean_all", f"{base}|mean_all", "Ē toàn bộ 2m đường"), (f"{s['config']}|mean_all", f"{prior}|mean_all", "Ē toàn bộ 2m đường")]
    rows, seen = [], set()
    for a, b, lab in comps:
        if (a, b) in seen or a not in Wd or b not in Wd:
            continue
        seen.add((a, b))
        grp = []
        for met, sgn in EXT_DIR.items():
            if met not in Wd[a] or met not in Wd[b]:
                continue
            j = pd.concat([Wd[a][met].rename("a"), Wd[b][met].rename("b")], axis=1).dropna()
            v = sgn * (j.a - j.b).values
            bs = v[boot_idx(len(v), B_, seed)].mean(1)
            lo, hi = ci_of(bs, al)
            grp.append({"mức": lab, "a": a, "b": b, "metric": met, "chiều tốt": "lớn" if sgn > 0 else "nhỏ", "TB a": float(j.a.mean()),
                        "TB b": float(j.b.mean()), "(a − b) theo chiều tốt [CI chưa đ/c]": fmt_ci(float(v.mean()), lo, hi), "p": boot_p(bs),
                        "_s": float(np.sign(v.mean()))})
        ph = holm([g_["p"] for g_ in grp])
        for g_, h in zip(grp, ph):
            g_["p Holm (trong cặp)"] = h
            g_["kết luận (Holm)"] = ("a tốt hơn" if g_["_s"] > 0 else "a kém hơn") if h < al else "chưa ghi nhận khác biệt"
            del g_["_s"]
        rows += grp
    if not rows:
        return []
    return ["## G4 — metric ngoài (mô tả; KHÔNG dùng để chọn lại θ)\n",
            "So ở K của ngân sách (bản đồ thực tế) và trên Ē của toàn bộ 2m đường (gần attribution kỳ vọng): khác biệt chỉ ở mức ngân sách mà "
            "biến mất ở Ē ⇒ chủ yếu do phương sai hữu hạn K. \"Chưa ghi nhận khác biệt\" — không phải \"tương đương\".\n",
            md(pd.DataFrame(rows))]


def write_report(cfg: dict) -> str:
    rp = os.path.join(C.out_dir(cfg, "pilot"), "infid", "INFID_BUDGET_REPORT.md")
    pr = bpr(cfg)
    lk = lock_path(cfg)
    L = ["# BUDGET — tối ưu chiến lược lấy mẫu SPI theo J ở ngân sách gradient cố định\n"]
    if cfg.get("_debug"):
        L.append("> **DEBUG** — chỉ kiểm pipeline.\n")
    L.append(f"- J(θ, K) = L_D(μ_θ) + V_K; B = K·N·M; B chính {pr['budgets']}, phụ {pr['budgets_secondary']}; quy tắc `{pr['rule']}`, M = {pr['M']}. "
             f"Khoá test: " + (f"`{json.load(open(lk))['digest'][:12]}`" if os.path.exists(lk) else "**chưa khoá**") + ".")
    L.append("- Đơn vị thống kê là ảnh; bootstrap theo ảnh. Dev chọn trên bộ ô CHỌN; test trên ảnh mới, bộ ô ĐÁNH GIÁ. K khác nhau suy từ "
             "cùng các đường (công thức L + V_K), kèm kiểm J thực nghiệm bằng ensemble thật.\n")
    for st in ("dev1", "dev2"):
        L += dev_section(cfg, st)
    L += test_section(cfg)
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[budget] {rp}")
    return rp
