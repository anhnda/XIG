"""
`sampling-infid` — E1a (RQ1: ghép nối inter-path ở q cố định), E1b (RQ2: thiết kế phân phối đường), E2 (kiểm định ngoài mẫu).

Giai đoạn (bash scripts/run.sh <lệnh>):
  sampling-infid-pilot     E1a pilot: m cặp `rev` + nhánh `indep` nhỏ. Kiểm phương sai suy từ cặp (V̂_e + V̂_o) khớp phương sai đo
                           trực tiếp, C1c (sign-flip), C2 (K·V(K)), và chọn số cặp m theo độ rộng CI. Không cần khoá.
  infid-lock               khoá protocol (K chính, δ₁–δ₃, quy tắc tích phân, D, cấu hình các giai đoạn) → infid_prereg.json.
  sampling-infid-e1a       E1a chính — CHỈ cặp `rev`: V̂_e, V̂_o, L̂(μ) ⇒ J_indep(K), J_rev(K), ΔV_D(K) cho mọi K. Tuỳ chọn
                           `global` / `nbr` (thiết kế ghép giữ phân phối biên: `pix` cho họ spi, `pat` cho họ patch) chạy riêng
                           từng K với R bản sao ⇒ H2. Endpoint chính H1: ΔV_D(K chính).
  sampling-infid-e1b-dev   E1b trên ảnh dev + bộ ô CHỌN: L_D(μ_q) = L(Ē) − V̂_e/m cho lưới cấu hình (họ, α, N, G, η) + IG ⇒ chọn
                           cấu hình, ghi e1b_selection.json (một lần; không ghi đè).
  sampling-infid-e1b-test  E1b trên ảnh test + bộ ô ĐÁNH GIÁ độc lập: H3 (cấu hình đã chọn so với SPI baseline VÀ IG).
  sampling-infid-e2        E2: ảnh và ô mới, nhiều D (xoá / chèn / đối xứng / phân số / nền blur / superpixel), metric ngoài
                           (ROAD MoRF/LeRF, InsDel blur, định vị) trên attribution đã lấy mẫu; tuỳ chọn thêm kiến trúc (ViT).
  sampling-infid-report    dựng lại INFID_SAMPLING_REPORT.md từ CSV.
Đơn vị thống kê là ảnh. Mọi phương sai / infid chuẩn hoá theo Σ ΔF² của chính tập ô của ảnh (estimand chính = trung bình theo ảnh).
"""

from __future__ import annotations

import copy
import datetime as _dt
import itertools
import json
import math
import os
import time

import numpy as np
import torch

from . import config as C
from . import infid_core as IC
from . import infid_stats as ST
from . import pathscore_metrics as PM
from . import sampling as SM
from .infid_e0 import icfg, out_dir, protocol
from .pathquad import integrate
from .paths import straight_nodes
from .quad import Counter

F64 = torch.float64
SI_VERSION = 1
STAGES = ("pilot", "e1a", "e1b-dev", "e1b-test", "e2")
PROTO_DEFAULT = {"rule": "mid", "M": 1, "K_main": None, "Ks": [1, 2, 4, 8, 16], "delta0": None, "delta1": None, "delta2": None,
                 "delta3": None, "rho_min_vindep": 1.0e-4, "alpha": 0.05, "bootstrap": 10000, "seed": 23, "n_flip": 2000,
                 "floor_grids": [14], "floor_lams": [0.0, 1.0e-3, 1.0e-2, 1.0e-1], "floor_folds": 4, "quad_probe": 2,
                 "quad_probe_factor": 4}


def pcfg(cfg: dict) -> dict:
    return C._merge(PROTO_DEFAULT, protocol(cfg))


def sroot(cfg: dict) -> str:
    return out_dir(cfg, "sampling")


def prereg_path(cfg: dict) -> str:
    return os.path.join(C.out_dir(cfg, "pilot"), "infid", "infid_prereg.json")


def selection_path(cfg: dict) -> str:
    return os.path.join(C.out_dir(cfg, "pilot"), "infid", "e1b_selection.json")


# ─────────────────────────────────────────────────────────────────────────────
# cấu hình đường
# ─────────────────────────────────────────────────────────────────────────────

def norm_conf(c: dict) -> dict:
    f = c.get("family", "spi")
    if f not in ("spi", "patch", "hier", "ig"):
        raise SystemExit(f"họ đường không biết: {f}")
    out = {"family": f, "N": int(c.get("N", 64))}
    if f != "ig":
        out["alpha"] = float(c["alpha"])
    if f in ("patch", "hier"):
        out["G"] = int(c.get("G", 7))
    if f == "hier":
        out["eta"] = float(c.get("eta", 0.5))
    return out


def cname(c: dict) -> str:
    c = norm_conf(c)
    if c["family"] == "ig":
        return f"ig_N{c['N']}"
    s = f"{c['family']}_a{c['alpha']:g}_N{c['N']}"
    if c["family"] in ("patch", "hier"):
        s += f"_G{c['G']}"
    if c["family"] == "hier":
        s += f"_e{c['eta']:g}"
    return s


def expand_grid(g: dict) -> list:
    out = []
    for fam in g.get("family", ["spi"]):
        for a, N in itertools.product(g.get("alpha", [10]), g.get("N", [64])):
            if fam == "spi":
                out.append(norm_conf({"family": "spi", "alpha": a, "N": N}))
            elif fam == "patch":
                out += [norm_conf({"family": "patch", "alpha": a, "N": N, "G": G}) for G in g.get("G", [7])]
            elif fam == "hier":
                out += [norm_conf({"family": "hier", "alpha": a, "N": N, "G": G, "eta": e})
                        for G in g.get("G", [7]) for e in g.get("eta", [0.5]) if 0 < float(e) < 1]
    seen, uniq = set(), []
    for c in out:
        if cname(c) not in seen:
            seen.add(cname(c))
            uniq.append(c)
    return uniq


class PathGen:
    """Lịch trình Z (N+1, C,H,W) float64 cho đường thứ j (rút độc lập theo j). rev hợp lệ cho cả ba họ (P0_AUDIT §1)."""

    def __init__(self, E: IC.IEnv, conf: dict, tag: str):
        self.E, self.c, self.tag = E, norm_conf(conf), tag
        self.name = cname(conf)
        if self.c["family"] in ("patch", "hier"):
            self.lab = SM.grid_elem_labels(E.shape, self.c["G"]).to(E.x.device)

    def Z(self, j: int) -> torch.Tensor:
        E, c = self.E, self.c
        key = (self.tag, self.name, E.image, int(j))
        if c["family"] == "spi":
            return SM.spi_schedule(c["alpha"], c["N"], E.shape, E.gen("INFspi", *key))
        P = c["G"] * c["G"]
        phi = torch.rand(P, generator=E.cgen("INFphi", *key), dtype=F64)              # thiết kế indep: pha iid U(0,1)
        Zp = SM.expand_patch(SM.patch_schedule(c["alpha"], c["N"], phi, E.gen("INFpat", *key)), self.lab, E.shape)
        if c["family"] == "patch":
            return Zp
        return SM.mix(Zp, SM.spi_schedule(c["alpha"], c["N"], E.shape, E.gen("INFspi", *key)), c["eta"])


def attr_of(E: IC.IEnv, X: torch.Tensor, pr: dict, gkey) -> torch.Tensor:
    gen = E.gen("INFq", *gkey) if pr["rule"] == "rand" else None
    return integrate(E.fn, X, int(pr["M"]), pr["rule"], gen=gen, batch=E.ctx.batch)


# ─────────────────────────────────────────────────────────────────────────────
# cấu hình giai đoạn
# ─────────────────────────────────────────────────────────────────────────────

STAGE_DEFAULT = {
    "pilot": {"split": "pilot", "offset": 0, "n_images": 8, "m": 16, "m_indep": 32, "configs": [{"family": "spi", "alpha": 10, "N": 64}],
              "sets": None, "ig": False},
    "e1a": {"split": "pilot", "offset": 0, "n_images": 16, "m": 32, "configs": [{"family": "spi", "alpha": 10, "N": 64}], "sets": None,
            "ig": False, "coupled": {"enabled": False, "designs": ["global", "nbr"], "Ks": [2, 4, 8, 16], "replicates": 8}},
    "e1b": {"dev": {"split": "pilot", "offset": 0, "n_images": 16}, "test": {"split": "fresh", "offset": 400, "n_images": 48},
            "m": 16, "grid": {"family": ["spi", "patch"], "alpha": [1, 3, 10, 30, 100, 1000], "N": [64], "G": [7], "eta": [0.5]},
            "baseline": {"family": "spi", "alpha": 10, "N": 64}, "ig_N": 64, "sets": None, "select_set": "sel"},
    "e2": {"split": "fresh", "offset": 500, "n_images": 32, "m": 16, "sets": None, "external": True, "floors": True,
           "extra_arch": [], "road": {"fracs": [0.1, 0.3, 0.5, 0.7, 0.9], "noise": 0.01, "iters": 1000, "tol": 1.0e-5},
           "insdel_steps": 50, "pointing_tol": 15, "top_frac": 0.05},
}


def stage_cfg(cfg: dict, stage: str) -> dict:
    ic = icfg(cfg)
    if stage in ("e1b-dev", "e1b-test"):
        b = C._merge(STAGE_DEFAULT["e1b"], ic.get("e1b", {}) or {})
        sc = {**b, **b["dev" if stage == "e1b-dev" else "test"]}
        if stage == "e1b-dev":
            sc["configs"] = expand_grid(b["grid"])
            if cname(b["baseline"]) not in [cname(c) for c in sc["configs"]]:
                sc["configs"].append(norm_conf(b["baseline"]))
        else:
            sel = load_selection(cfg)
            sc["configs"] = [norm_conf(sel["chosen"])] + ([norm_conf(b["baseline"])] if cname(sel["chosen"]) != cname(b["baseline"]) else [])
        sc["ig"] = True
        return sc
    sc = C._merge(STAGE_DEFAULT[stage], ic.get(stage, {}) or {})
    if stage == "e2":
        b = C._merge(STAGE_DEFAULT["e1b"], ic.get("e1b", {}) or {})
        try:
            ch = norm_conf(load_selection(cfg)["chosen"])
        except SystemExit:
            if not cfg.get("_debug"):
                raise
            ch = norm_conf(b["baseline"])
        sc["configs"] = [ch] + ([norm_conf(b["baseline"])] if cname(ch) != cname(b["baseline"]) else [])
        sc["ig"] = True
        sc["ig_N"] = int(b["ig_N"])
    sc["configs"] = [norm_conf(c) for c in sc["configs"]]
    return sc


def dsets_spec(cfg: dict, sc: dict) -> dict:
    D = icfg(cfg)["D"]
    names = sc.get("sets") or [D["primary"]]
    bad = [n for n in names if n not in D["sets"]]
    if bad:
        raise SystemExit(f"infid.D.sets thiếu: {bad}")
    return {n: D["sets"][n] for n in names}


def load_selection(cfg: dict) -> dict:
    p = selection_path(cfg)
    if not os.path.exists(p):
        raise SystemExit(f"chưa có {p} — chạy bash scripts/run.sh sampling-infid-e1b-dev trước")
    with open(p) as fh:
        return json.load(fh)


def lock_payload(cfg: dict) -> dict:
    ic = icfg(cfg)
    return {"protocol": pcfg(cfg), "D": ic["D"], "e1a": ic.get("e1a"), "e1b": ic.get("e1b"), "e2": ic.get("e2"),
            "model": cfg["model"], "data": cfg["data"], "seed": cfg["seed"], "v": SI_VERSION}


def lock_digest(cfg: dict) -> str:
    return IC._h(lock_payload(cfg))


def lock(cfg: dict) -> None:
    pr = pcfg(cfg)
    miss = [k for k in ("K_main", "delta1", "delta2", "delta3") if pr.get(k) is None]
    if miss:
        raise SystemExit(f"không khoá được: infid.protocol còn null: {miss} (chọn từ pilot trước khi mở tập test)")
    if int(pr["K_main"]) not in [int(k) for k in pr["Ks"]]:
        raise SystemExit("protocol.K_main phải thuộc protocol.Ks")
    if int(pr["K_main"]) < 2 or int(pr["K_main"]) % 2:
        raise SystemExit("protocol.K_main phải chẵn và ≥ 2 (ensemble rev gồm K/2 cặp)")
    os.makedirs(os.path.dirname(prereg_path(cfg)), exist_ok=True)
    ST.write_prereg(prereg_path(cfg), lock_digest(cfg), {"payload": lock_payload(cfg)})


# ─────────────────────────────────────────────────────────────────────────────
# khối tính trong một ảnh
# ─────────────────────────────────────────────────────────────────────────────

def flat_rows(meta: dict, stats_by_group: dict) -> list:
    return [{**meta, "group": g, **st} for g, st in stats_by_group.items()]


def pair_block(E: IC.IEnv, conf: dict, m: int, pr: dict, dsets: dict, tag: str, keep_K: int = 0) -> dict:
    """m cặp (Z, RZ) ⇒ chiếu lên mọi D (sel / eval / fit). keep_K > 0: giữ bản đồ của ước lượng rev K đường (K/2 cặp đầu)."""
    pg = PathGen(E, conf, tag)
    t0, n0 = time.time(), Counter.n
    Y = {(nm, s): ([], []) for nm, d in dsets.items() for s in d}
    sums = []
    Esum = torch.zeros(E.shape, dtype=F64, device=E.x.device)
    Akeep = torch.zeros_like(Esum)
    qerr = []
    for j in range(int(m)):
        Z = pg.Z(j)
        for w, Zx in ((0, Z), (1, SM.reverse_nodes(Z))):
            X = SM.nodes(E.x0, E.dx, Zx)
            A = attr_of(E, X, pr, (tag, pg.name, E.image, j, w))
            sums.append(float(A.sum()))
            Esum += A / 2
            if keep_K and j < keep_K // 2:
                Akeep += A / keep_K
            for (nm, s), (y1, y2) in Y.items():
                (y1 if w == 0 else y2).append(dsets[nm][s].project(A)[0])
            if j * 2 + w < int(pr.get("quad_probe", 0)) and pr["rule"] != "rand":
                Af = integrate(E.fn, X, int(pr["M"]) * int(pr["quad_probe_factor"]), "mid", batch=E.ctx.batch)
                nm0 = next(iter(dsets))
                ds0 = dsets[nm0]["eval"]
                pa, pf = ds0.project(A)[0], ds0.project(Af)[0]
                qerr.append(float((pa - pf).norm() / pf.norm().clamp_min(1e-300)))
            del X, A
        del Z
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"Y": {k: (torch.stack(a), torch.stack(b)) for k, (a, b) in Y.items()}, "sums": sums, "Ebar": Esum / int(m),
            "Akeep": Akeep if keep_K else None, "grads": Counter.n - n0, "seconds": time.time() - t0,
            "quad_rel_proj": float(np.mean(qerr)) if qerr else math.nan, "name": pg.name}


def indep_block(E: IC.IEnv, conf: dict, n: int, pr: dict, dsets: dict, tag: str) -> dict:
    pg = PathGen(E, conf, tag + "-indep")
    t0, n0 = time.time(), Counter.n
    Y = {(nm, s): [] for nm, d in dsets.items() for s in d}
    for j in range(int(n)):
        X = SM.nodes(E.x0, E.dx, pg.Z(j))
        A = attr_of(E, X, pr, (tag + "-indep", pg.name, E.image, j))
        for (nm, s), y in Y.items():
            y.append(dsets[nm][s].project(A)[0])
        del X, A
    return {"Y": {k: torch.stack(v) for k, v in Y.items()}, "grads": Counter.n - n0, "seconds": time.time() - t0}


def coupled_block(E: IC.IEnv, conf: dict, design: str, K: int, R: int, pr: dict, dsets: dict, tag: str) -> dict:
    """R bản sao độc lập của ensemble K đường theo thiết kế ghép (global / nbr). Họ spi ⇒ pha theo phần tử (pix, giữ q_SPI);
    họ patch ⇒ pha theo patch (pat, giữ q_patch). Mỗi bản sao rút bảng pha mới."""
    c = norm_conf(conf)
    if c["family"] not in ("spi", "patch"):
        raise SystemExit("thiết kế ghép chỉ cho họ spi / patch")
    if not SM.is_pow2(int(K)) or int(K) < 2:
        raise SystemExit("K của thiết kế ghép phải là luỹ thừa của 2, ≥ 2")
    name = cname(c)
    t0, n0 = time.time(), Counter.n
    Yr = {(nm, s): [] for nm, d in dsets.items() for s in d}
    tot = torch.zeros(E.shape, dtype=F64, device=E.x.device)
    for r in range(int(R)):
        key = (tag, design, name, E.image, int(K), r)
        if c["family"] == "spi":
            n_el = E.shape[0] * E.shape[1] * E.shape[2]
            common = SM.common_phase_noise(n_el, int(K), E.cgen("INFcp-common", *key))
            phi, _ = SM.pixel_phase_design(design, E.shape, int(K), E.cgen("INFcp-phase", *key), common=common)
        else:
            P = c["G"] * c["G"]
            common = SM.common_phase_noise(P, int(K), E.cgen("INFcp-common", *key))
            phi, _ = SM.phase_design(design, P, int(K), E.cgen("INFcp-phase", *key), g=c["G"], common=common)
            lab = SM.grid_elem_labels(E.shape, c["G"]).to(E.x.device)
        Am = torch.zeros(E.shape, dtype=F64, device=E.x.device)
        for k in range(int(K)):
            g = E.gen("INFcp-inc", *key, k)
            if c["family"] == "spi":
                Z = SM.pixel_schedule(c["alpha"], c["N"], E.shape, phi[k], g)
            else:
                Z = SM.expand_patch(SM.patch_schedule(c["alpha"], c["N"], phi[k], g), lab, E.shape)
            A = attr_of(E, SM.nodes(E.x0, E.dx, Z), pr, (*key, k))
            Am += A / int(K)
            del Z, A
        tot += Am / int(R)
        for (nm, s), y in Yr.items():
            y.append(dsets[nm][s].project(Am)[0])
    return {"Y": {k: torch.stack(v) for k, v in Yr.items()}, "mean": tot, "grads": Counter.n - n0, "seconds": time.time() - t0}


def ext_metrics(E: IC.IEnv, A: torch.Tensor, sc: dict, P: dict) -> dict:
    """Metric ngoài trên attribution đã lấy mẫu: ROAD (MoRF / LeRF), InsDel nền blur, định vị theo mask GT."""
    out = {}
    q = SM.insdel_full(E.ctx.bb, E.cls, E.x, E.bg("blur"), A.float(), int(sc["insdel_steps"]), E.ins_batch)
    out["insdel_blur"] = q["diff"]
    rc = sc["road"]
    out.update(PM.road(P["prob"], E.x, A, E.perm, P["fracs"], P["road_z"], P["road_sd"], E.ins_batch, int(rc["iters"]), float(rc["tol"])))
    out.update(PM.localization(A, E.mask, E.perm, float(sc["top_frac"]), int(sc["pointing_tol"])))
    return out


def prep_ext(E: IC.IEnv, sc: dict) -> dict:
    rc = sc["road"]
    fr = [float(f) for f in rc["fracs"]]
    bb, cls = E.ctx.bb, E.cls

    def prob(t):
        SM.FwdCounter.n += int(t.shape[0])
        return torch.softmax(bb(t), dim=-1)[:, cls]
    return {"fracs": fr, "prob": prob, "road_sd": PM.road_noise_sd(float(rc["noise"]), E.x.device),
            "road_z": torch.randn((2 * len(fr), E.x.shape[1], E.H, E.W), generator=E.cgen("infid-road", E.image),
                                  dtype=F64).to(E.x.device)}


# ─────────────────────────────────────────────────────────────────────────────
# một ảnh × một giai đoạn
# ─────────────────────────────────────────────────────────────────────────────

def stage_digest(cfg: dict, stage: str, sc: dict, extra=None) -> str:
    return IC._h({"v": SI_VERSION, "stage": stage, "sc": sc, "proto": pcfg(cfg), "D": icfg(cfg)["D"], "seed": cfg["seed"],
                  "model": cfg["model"], "data": cfg["data"], "extra": extra})


def build_dsets(E: IC.IEnv, cfg: dict, sc: dict, stage: str) -> dict:
    D = icfg(cfg)["D"]
    dcache = os.path.join(C.out_dir(cfg, "pilot"), "infid", "dcache", E.image)
    out = {}
    for nm, spec in dsets_spec(cfg, sc).items():
        if stage == "e2":
            st = {"eval": ("e2-eval", int(D.get("n_eval", 512)))}
            if sc.get("floors", True):
                st["fit"] = ("e2-fit", int(D.get("n_sel", 512)))
        else:
            st = {"sel": ("e1-sel", int(D.get("n_sel", 512))), "eval": ("e1-eval", int(D.get("n_eval", 512)))}
        out[nm] = {s: IC.dset_cached(E, dcache, nm, spec, n, stream, E.ins_batch) for s, (stream, n) in st.items()}
    return out


def process_image(ctx, cfg: dict, stage: str, sc: dict, idx: int, out: str) -> dict:
    pr = pcfg(cfg)
    Ks = [int(k) for k in pr["Ks"]]
    image = f"seg{idx:05d}"
    cdir = os.path.join(out, "cache", image)
    os.makedirs(cdir, exist_ok=True)
    rows = {"pairs": [], "indep": [], "ig": [], "floors": [], "coupled": [], "ext": [], "cost": []}
    E = None

    def env():
        nonlocal E
        if E is None:
            E = IC.IEnv(ctx, idx)
        return E
    dsets = None
    arch = cfg["model"]["arch"]
    keep_K = min(max(2, 2 * (int(pr["K_main"] or 4) // 2)), 2 * int(sc["m"])) if stage == "e2" else 0
    fit_key = "fit" if stage == "e2" else "sel"
    for conf in sc["configs"]:
        nm_c = cname(conf)
        cp = os.path.join(cdir, f"{nm_c}.pt")
        dg = stage_digest(cfg, stage, {k: v for k, v in sc.items() if k not in ("configs", "split", "offset", "n_images")}, nm_c)
        if os.path.exists(cp) and torch.load(cp, weights_only=False).get("digest") == dg:
            r = torch.load(cp, weights_only=False)["rows"]
            for k in rows:
                rows[k] += r.get(k, [])
            print(f"    [{stage}] {image} {nm_c}: đã có")
            continue
        E = env()
        if dsets is None:
            dsets = build_dsets(E, cfg, sc, stage)
        res = pair_block(E, conf, int(sc["m"]), pr, dsets, stage, keep_K=keep_K)
        rr = {k: [] for k in rows}
        base = {"stage": stage, "arch": arch, "image": image, "config": nm_c, **norm_conf(conf), "rule": pr["rule"], "M": int(pr["M"])}
        gsum = torch.tensor(res["sums"], dtype=F64)
        compl = float(((gsum - E.D) / abs(E.D)).abs().mean()) if E.D else math.nan
        for (nm, s), (y1, y2) in res["Y"].items():
            ds = dsets[nm][s]
            st = IC.pair_stats(y1, y2, ds.dF.to(y1.device), Ks, groups=ds.group.to(y1.device),
                               n_flip=int(pr["n_flip"]) if s == "eval" else 0, gen=E.cgen("infid-flip", stage, nm_c, nm, image),
                               jackknife=(s == "eval"))
            rr["pairs"] += flat_rows({**base, "dset": nm, "set": s, "compl_gap_abs": compl, "quad_rel_proj": res["quad_rel_proj"]}, st)
        rr["cost"].append({**base, "grads": res["grads"], "seconds": res["seconds"], "paths": 2 * int(sc["m"])})
        if stage == "pilot":
            ind = indep_block(E, conf, int(sc["m_indep"]), pr, dsets, stage)
            for (nm, s), Y in ind["Y"].items():
                ds = dsets[nm][s]
                st = IC.indep_core(Y, ds.dF.to(Y.device), Ks)
                rr["indep"].append({**base, "dset": nm, "set": s, **st})
            rr["cost"].append({**base, "config": nm_c + "+indep", "grads": ind["grads"], "seconds": ind["seconds"],
                               "paths": int(sc["m_indep"])})
        if stage == "e1a" and (sc.get("coupled") or {}).get("enabled") and conf["family"] in ("spi", "patch"):
            cc = sc["coupled"]
            for design in cc["designs"]:
                for K in cc["Ks"]:
                    cb = coupled_block(E, conf, design, int(K), int(cc["replicates"]), pr, dsets, stage)
                    for (nm, s), Yr in cb["Y"].items():
                        if s != "eval":
                            continue
                        ds = dsets[nm][s]
                        d = ds.dF.to(Yr.device)
                        den = float((d * d).sum())
                        Vd = float(((Yr - Yr.mean(0)) ** 2).sum()) / (Yr.shape[0] - 1) / den
                        y1, y2 = res["Y"][(nm, s)]
                        pst = IC.pair_core(y1, y2, d, [int(K)])
                        diff = ds.project(cb["mean"])[0] - (y1 + y2).mean(0) / 2
                        Tm = float((diff ** 2).sum()) / den
                        exp0 = Vd / Yr.shape[0] + pst["Ve"] / y1.shape[0]
                        rr["coupled"].append({**base, "dset": nm, "set": s, "design": design, "K": int(K), "R": int(Yr.shape[0]),
                                              "V_design": Vd, "V_rev": 2 * pst["Ve"] / int(K), "V_indep": pst["Vpath"] / int(K),
                                              "rev_minus_design": 2 * pst["Ve"] / int(K) - Vd,
                                              "indep_minus_design": pst["Vpath"] / int(K) - Vd,
                                              "mean_eq_ratio": Tm / exp0 if exp0 > 0 else math.nan,
                                              "grads": cb["grads"], "seconds": cb["seconds"]})
        if stage == "e2" and sc.get("external", True):
            P = prep_ext(E, sc)
            em = ext_metrics(E, res["Akeep"], sc, P)
            rr["ext"].append({**base, "estimator": f"rev_K{keep_K}", **em})
        torch.save({"digest": dg, "rows": rr}, cp)
        for k in rows:
            rows[k] += rr[k]
        print(f"    [{stage}] {image} {nm_c}: {res['grads']} grad, {res['seconds']:.0f}s", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # IG + sàn (một lần mỗi ảnh)
    if sc.get("ig") or stage != "pilot":
        cp = os.path.join(cdir, "_ig_floors.pt")
        dg = stage_digest(cfg, stage, {k: v for k, v in sc.items() if k not in ("configs", "split", "offset", "n_images")}, "igfl")
        if os.path.exists(cp) and torch.load(cp, weights_only=False).get("digest") == dg:
            r = torch.load(cp, weights_only=False)["rows"]
            for k in rows:
                rows[k] += r.get(k, [])
        else:
            E = env()
            if dsets is None:
                dsets = build_dsets(E, cfg, sc, stage)
            rr = {k: [] for k in rows}
            base = {"stage": stage, "arch": arch, "image": image, "rule": pr["rule"], "M": int(pr["M"])}
            if sc.get("ig"):
                N = int(sc.get("ig_N", 64))
                n0, t0 = Counter.n, time.time()
                Aig = attr_of(E, straight_nodes(E.x, E.x0, N), pr, (stage, "ig", image))
                for nm, d in dsets.items():
                    for s, ds in d.items():
                        y = ds.project(Aig)[0]
                        dF = ds.dF.to(y.device)
                        grp = ds.group.to(y.device)
                        rr["ig"].append({**base, "config": f"ig_N{N}", "dset": nm, "set": s, "group": "all",
                                         "L": float(IC.infid(y, dF)), "den": float((dF * dF).sum()),
                                         "compl_gap_abs": abs(float(Aig.sum()) - E.D) / abs(E.D) if E.D else math.nan})
                        for gv in sorted(set(grp.tolist())):
                            ix = torch.nonzero(grp == gv, as_tuple=True)[0]
                            rr["ig"].append({**base, "config": f"ig_N{N}", "dset": nm, "set": s, "group": f"g{gv}",
                                             "L": float(IC.infid(y[ix], dF[ix])), "den": float((dF[ix] ** 2).sum())})
                rr["cost"].append({**base, "config": f"ig_N{N}", "grads": Counter.n - n0, "seconds": time.time() - t0, "paths": 1})
                if stage == "e2" and sc.get("external", True):
                    rr["ext"].append({**base, "config": f"ig_N{N}", "family": "ig", "estimator": "ig",
                                      **ext_metrics(E, Aig, sc, prep_ext(E, sc))})
            for nm, d in dsets.items():
                if fit_key not in d:
                    continue
                for G in pr["floor_grids"]:
                    Xf = IC.patch_design(E, d[fit_key], int(G))
                    Xe = IC.patch_design(E, d["eval"], int(G))
                    fl = IC.floor_fit(Xf, d[fit_key].dF.to(Xf.device), Xe, d["eval"].dF.to(Xf.device), E.D,
                                      [float(x) for x in pr["floor_lams"]], int(pr["floor_folds"]), E.cgen("infid-floor", stage, nm, image, G))
                    rr["floors"].append({**base, "dset": nm, "G": int(G), **fl})
            torch.save({"digest": dg, "rows": rr}, cp)
            for k in rows:
                rows[k] += rr[k]
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# chạy một giai đoạn
# ─────────────────────────────────────────────────────────────────────────────

def stage_images(ctx, sc: dict) -> list:
    idxs = ctx.sp[sc["split"]][int(sc.get("offset", 0)):int(sc.get("offset", 0)) + int(sc["n_images"])]
    if len(idxs) < int(sc["n_images"]):
        raise SystemExit(f"split {sc['split']} không đủ {sc['n_images']} ảnh từ vị trí {sc.get('offset', 0)}")
    return idxs


def check_disjoint(ctx, cfg: dict, stage: str, idxs: list) -> None:
    """Ảnh test (E1b-test, E2) không được giao với ảnh dev / pilot đã dùng để chọn."""
    if stage not in ("e1b-test", "e2") or cfg.get("_debug"):
        return
    used = set()
    for st in ("pilot", "e1a", "e1b-dev"):
        try:
            used |= set(stage_images(ctx, stage_cfg(cfg, st)))
        except SystemExit:
            pass
    if stage == "e2":
        try:
            used |= set(stage_images(ctx, stage_cfg(cfg, "e1b-test")))
        except SystemExit:
            pass
    inter = used & set(idxs)
    if inter:
        raise SystemExit(f"[{stage}] {len(inter)} ảnh trùng với ảnh đã dùng ở giai đoạn trước — đổi split/offset")


def estimate(cfg: dict, stage: str, sc: dict, per_eval: float) -> dict:
    pr = pcfg(cfg)
    NM = lambda c: int(c["N"]) * int(pr["M"])
    g = sum(2 * int(sc["m"]) * NM(c) for c in sc["configs"])
    if stage == "pilot":
        g += sum(int(sc["m_indep"]) * NM(c) for c in sc["configs"])
    if stage == "e1a" and (sc.get("coupled") or {}).get("enabled"):
        cc = sc["coupled"]
        g += sum(len(cc["designs"]) * int(cc["replicates"]) * sum(int(k) for k in cc["Ks"]) * NM(c) for c in sc["configs"]
                 if c["family"] in ("spi", "patch"))
    if sc.get("ig"):
        g += int(sc.get("ig_N", 64)) * int(pr["M"])
    return {"grad_per_image": g, "minutes": int(sc["n_images"]) * g * per_eval / 60}


def run(ctx, stage: str) -> None:
    from .ctx import bench
    cfg = ctx.cfg
    if stage not in STAGES:
        raise SystemExit(f"giai đoạn không biết: {stage}")
    if stage != "pilot":
        ST.require_prereg(prereg_path(cfg), lock_digest(cfg), cfg["_debug"])
    sc = stage_cfg(cfg, stage)
    pr = pcfg(cfg)
    if stage == "e2" and pr["K_main"] is None and not cfg["_debug"]:
        raise SystemExit("E2 cần protocol.K_main")
    out = os.path.join(sroot(cfg), stage)
    os.makedirs(out, exist_ok=True)
    archs = [None] + list(sc.get("extra_arch") or []) if stage == "e2" else [None]
    for arch in archs:
        c2 = ctx
        if arch is not None:
            from .ctx import Ctx
            cfg2 = copy.deepcopy(cfg)
            cfg2["model"] = {**cfg2["model"], **arch}
            c2 = Ctx(cfg2, str(ctx.dev))
        _run_one(c2, stage, sc, out if arch is None else os.path.join(out, f"arch_{arch['arch']}"), bench)
    write_report(cfg)


def _run_one(ctx, stage: str, sc: dict, out: str, bench) -> None:
    cfg = ctx.cfg
    os.makedirs(out, exist_ok=True)
    idxs = stage_images(ctx, sc)
    check_disjoint(ctx, cfg, stage, idxs)
    b = bench(ctx)
    est = estimate(cfg, stage, sc, b["per_eval"])
    print(f"[{stage}] {cfg['model']['arch']}: {len(idxs)} ảnh × {len(sc['configs'])} cấu hình, m = {sc['m']}: "
          f"{est['grad_per_image']} grad/ảnh ≈ {est['minutes']:.1f} phút (nếu chưa có cache)", flush=True)
    lim = (icfg(cfg).get("budget") or {}).get("minutes")
    if lim is not None and est["minutes"] > float(lim) and (icfg(cfg).get("budget") or {}).get("enforce", True):
        raise SystemExit(f"[{stage}] ước tính {est['minutes']:.1f} phút > infid.budget.minutes = {lim}")
    allr = {}
    t0, n0, f0 = time.time(), Counter.n, SM.FwdCounter.n
    for j, idx in enumerate(idxs):
        print(f"  [{stage}] ảnh {j + 1}/{len(idxs)} seg{idx:05d}", flush=True)
        r = process_image(ctx, cfg, stage, sc, idx, out)
        for k, v in r.items():
            allr.setdefault(k, []).extend(v)
    import pandas as pd
    for k, v in allr.items():
        if v:
            pd.DataFrame(v).to_csv(os.path.join(out, f"{k}.csv"), index=False)
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"time": _dt.datetime.now().isoformat(timespec="seconds"), "stage": stage, "images": [f"seg{i:05d}" for i in idxs],
                   "sc": sc, "protocol": pcfg(cfg), "estimate": est, "bench": b, "minutes": (time.time() - t0) / 60,
                   "grad_evals": Counter.n - n0, "fwd_evals": SM.FwdCounter.n - f0, "debug": cfg["_debug"],
                   "prereg_digest": lock_digest(cfg), "arch": cfg["model"]["arch"]}, fh, indent=1, default=str)
    if stage == "e1b-dev":
        select(cfg, out, sc)


def select(cfg: dict, out: str, sc: dict) -> None:
    """Chọn cấu hình trên ảnh dev + bộ ô CHỌN (min trung bình theo ảnh của L_D(μ_q)); ghi một lần, không ghi đè."""
    import pandas as pd
    D = icfg(cfg)["D"]
    p = pd.read_csv(os.path.join(out, "pairs.csv"))
    q = p[(p.set == sc.get("select_set", "sel")) & (p.group == "all") & (p.dset == D["primary"])]
    tab = q.groupby("config").L_mu.mean().sort_values()
    chosen = tab.index[0]
    conf = next(c for c in sc["configs"] if cname(c) == chosen)
    rec = {"chosen": conf, "chosen_name": chosen, "table": tab.to_dict(), "select_set": sc.get("select_set", "sel"),
           "dset": D["primary"], "n_images": int(q.image.nunique()), "protocol_digest": lock_digest(cfg),
           "time": _dt.datetime.now().isoformat(timespec="seconds")}
    sp = selection_path(cfg)
    if os.path.exists(sp) and not cfg["_debug"]:
        old = json.load(open(sp))
        if old["chosen_name"] != chosen:
            raise SystemExit(f"{sp} đã tồn tại với cấu hình {old['chosen_name']} ≠ {chosen}; không ghi đè lựa chọn sau khi đã chọn")
        return
    with open(sp, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(f"[e1b-dev] chọn {chosen} (L_D(μ) dev = {tab.iloc[0]:.4g}) → {sp}")


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo
# ─────────────────────────────────────────────────────────────────────────────

def _r(out, stage, name):
    from .report import _read
    return _read(os.path.join(out, stage), name)


def _ci_str(t):
    m, lo, hi = t
    return f"{m:.4g} [{lo:.4g}, {hi:.4g}]" if np.isfinite(lo) else (f"{m:.4g}" if np.isfinite(m) else "—")


def write_report(cfg: dict) -> str:
    import pandas as pd
    from .report import md
    pr = pcfg(cfg)
    D = icfg(cfg)["D"]
    out = sroot(cfg)
    B, seed, al = int(pr["bootstrap"]), int(pr["seed"]), float(pr["alpha"])
    Km = pr["K_main"]
    ci = lambda v: ST.boot_ci(v, B, seed, al)
    rp = os.path.join(C.out_dir(cfg, "pilot"), "infid", "INFID_SAMPLING_REPORT.md")
    L = ["# Lấy mẫu đường SPI theo infidelity — E1a / E1b / E2\n"]
    if cfg.get("_debug"):
        L.append("> **DEBUG** — chỉ kiểm pipeline, không dùng để kết luận.\n")
    lockp = prereg_path(cfg)
    L.append(f"- Protocol: quy tắc `{pr['rule']}`, M = {pr['M']}, K chính = {Km}, δ₁ = {pr['delta1']}, δ₂ = {pr['delta2']}, "
             f"δ₃ = {pr['delta3']}; D chính `{D['primary']}`. Khoá: "
             + (f"`{json.load(open(lockp))['digest'][:12]}`" if os.path.exists(lockp) else "**chưa khoá**") + ".")
    L.append("- Đơn vị thống kê là ảnh; CI 95% bootstrap theo ảnh. Estimand chính = trung bình theo ảnh của infid chuẩn hoá từng ảnh; "
             "tỷ số gộp chỉ là độ nhạy. 'CI chứa 0' = chưa kết luận, không phải bác bỏ.\n")

    # ── pilot
    pp, pi = _r(out, "pilot", "pairs"), _r(out, "pilot", "indep")
    if pp is not None:
        L.append("## E1a pilot\n")
        ev = pp[(pp.set == "eval") & (pp.group == "all")]
        if pi is not None:
            iv = pi[pi.set == "eval"]
            mg = ev.merge(iv[["image", "config", "dset", "sigma2"] + [c for c in iv.columns if c.startswith("KV_")]],
                          on=["image", "config", "dset"])
            mg["ratio"] = mg.Vpath / mg.sigma2
            L.append("**Phương sai suy từ cặp (V̂_e + V̂_o) so với đo trực tiếp từ nhánh indep** (cùng đích nếu C1 đúng; tỷ số ≈ 1):\n")
            tab = []
            for (c, dn), g in mg.groupby(["config", "dset"]):
                tab.append({"config": c, "D": dn, "n ảnh": len(g), "V̂_e+V̂_o": _ci_str(ci(g.Vpath)), "σ̂² trực tiếp": _ci_str(ci(g.sigma2)),
                            "chênh (cặp − trực tiếp)": _ci_str(ci(g.Vpath - g.sigma2)), "tỷ số trung vị": float(g.ratio.median())})
            L.append(md(pd.DataFrame(tab)))
            kv = [c for c in mg.columns if c.startswith("KV_") and not c.endswith("ngroups")]
            if kv:
                L.append("**C2 — K·V(K) từ nhóm rời nhau của nhánh indep** (ổn định theo K trong sai số; khớp 1/K không chứng minh C1):\n")
                L.append(md(mg.groupby("config")[kv + ["sigma2"]].mean().reset_index()))
        if "signflip_p" in ev:
            from statistics import NormalDist
            nd = NormalDist()
            tab = []
            for (c, dn), g in ev.groupby(["config", "dset"]):
                ps = g.signflip_p.clip(1e-12, 1 - 1e-12)
                z = sum(nd.inv_cdf(1 - p) for p in ps) / math.sqrt(len(ps))
                tab.append({"config": c, "D": dn, "n ảnh": len(g), "p trung vị": float(ps.median()),
                            f"tỷ lệ p ≤ {al}": float((ps <= al).mean()), "p gộp (Stouffer)": 1 - nd.cdf(z)})
            L.append(f"**C1c — sign-flip trên Ō (thống kê ŌᵀWŌ, {pr['n_flip']} lần lật)**: điều kiện cần, không đủ.\n")
            L.append(md(pd.DataFrame(tab)))
        if "jk_se_VoVe" in ev:
            Kp = int(Km or 4)
            plan = []
            for (c, dn), g in ev.groupby(["config", "dset"]):
                for r in ST.plan_m(g[f"dV_{Kp}"], g.jk_se_VoVe / Kp, int(g.m.iloc[0])):
                    plan.append({"config": c, "D": dn, **r})
            if plan:
                L.append(f"**Chọn m theo độ rộng CI của ΔV_D({Kp})** (s_b², s_w² từ jackknife theo cặp trong pilot):\n")
                pl = pd.DataFrame(plan)
                L.append(md(pl.pivot_table(index=["config", "D", "m"], columns="n_images", values="half_width").reset_index()))
        L.append(md(ev.groupby(["config", "dset"])[["Ve", "Vo", "L_mu", "rho"] + [f"dV_{k}" for k in pr["Ks"] if f"dV_{k}" in ev]]
                    .mean().reset_index()))

    # ── E1a chính
    ep = _r(out, "e1a", "pairs")
    if ep is not None:
        L.append("## E1a chính — RQ1: `rev` so với `indep` cùng ngân sách (chỉ dùng cặp)\n")
        ev = ep[(ep.set == "eval") & (ep.group == "all")]
        try:
            c_prim = cname(stage_cfg(cfg, "e1a")["configs"][0])
        except SystemExit:
            c_prim = None
        L.append(f"Endpoint chính H1 = ΔV_D(K chính) cho cấu hình đầu tiên `{c_prim}` trên D chính `{D['primary']}`; các dòng khác là phụ "
                 "(mô tả, không đạt/trượt).\n")
        tab = []
        for (c, dn), g in ev.groupby(["config", "dset"]):
            r = {"config": c, "D": dn, "n ảnh": len(g)}
            if Km is not None and f"dV_{Km}" in g:
                m_, lo, hi = ci(g[f"dV_{Km}"])
                r[f"ΔV_D({Km})"] = _ci_str((m_, lo, hi))
                r["H1"] = ST.decide(lo, hi, pr["delta1"]) if (c == c_prim and dn == D["primary"]) else "phụ: " + ST.decide(lo, hi, pr["delta1"])
            vind = g.Vpath
            ok = vind > float(pr["rho_min_vindep"])
            if ok.sum() >= 2:
                r["ρ̂_D (tỷ số gộp)"] = _ci_str(ST.boot_ratio(2 * g.Ve[ok], vind[ok], B, seed, al))
            r["p sign-flip ≤ α"] = float((g.signflip_p <= al).mean()) if "signflip_p" in g else np.nan
            tab.append(r)
        L.append(md(pd.DataFrame(tab)))
        L.append("**Bảng báo cáo bắt buộc** (trung bình theo ảnh, tập ô đánh giá):\n")
        L.append(mandatory_table(cfg, out, "e1a"))
        gr = ep[(ep.set == "eval") & (ep.group != "all")]
        if len(gr):
            L.append("Tách theo cỡ ô:\n")
            cols = ["Ve", "Vo", "L_mu"] + ([f"dV_{Km}"] if Km is not None and f"dV_{Km}" in gr else [])
            L.append(md(gr.groupby(["config", "dset", "group"])[cols].mean().reset_index()))
        L.append("**H4 — độ nhạy theo D** (mô tả, không đạt/trượt): ΔV_D và L_D(μ) cho từng D xem bảng trên; thứ hạng phần lệch "
                 "có thể đảo hợp lệ vì đích đổi theo D.\n")
    cpd = _r(out, "e1a", "coupled")
    if cpd is not None:
        L.append("## E1a — thiết kế ghép (`global`, `nbr`) — H2\n")
        tab = []
        for (c, dn, des, K), g in cpd.groupby(["config", "dset", "design", "K"]):
            m_, lo, hi = ci(g.rev_minus_design)
            # H2: nbr giảm thêm so với rev ⇔ V_rev − V_nbr ≥ δ₂ (global chỉ mô tả)
            tab.append({"config": c, "D": dn, "thiết kế": des, "K": int(K), "n ảnh": len(g), "V_rev − V_design": _ci_str((m_, lo, hi)),
                        "quyết định": (f"H2: {ST.decide(lo, hi, pr['delta2'])}" if des == "nbr" and Km is not None and int(K) == int(Km)
                                       else "mô tả"),
                        "V_indep − V_design": _ci_str(ci(g.indep_minus_design)),
                        "tỷ số kiểm kỳ vọng (≈1 nếu giữ biên)": float(g.mean_eq_ratio.median())})
        L.append(md(pd.DataFrame(tab)))
        L.append("H2 chỉ có nghĩa khi thiết kế giữ phân phối biên: tỷ số kiểm kỳ vọng ≫ 1 ⇒ trộn lệch với phương sai, không kết luận.\n")

    # ── E1b
    dp = _r(out, "e1b-dev", "pairs")
    if dp is not None:
        L.append("## E1b dev — RQ2: L_D(μ_q) theo cấu hình (bộ ô CHỌN, ảnh dev)\n")
        q = dp[(dp.set == "sel") & (dp.group == "all") & (dp.dset == D["primary"])]
        tab = q.groupby("config").agg(L_mu=("L_mu", "mean"), Ve=("Ve", "mean"), Vo=("Vo", "mean"), n=("image", "nunique")).reset_index()
        ig = _r(out, "e1b-dev", "ig")
        if ig is not None:
            g = ig[(ig.set == "sel") & (ig.group == "all") & (ig.dset == D["primary"])]
            tab = pd.concat([tab, pd.DataFrame([{"config": g.config.iloc[0], "L_mu": g.L.mean(), "Ve": 0.0, "Vo": 0.0,
                                                 "n": g.image.nunique()}])]) if len(g) else tab
        L.append(md(tab.sort_values("L_mu")))
        if os.path.exists(selection_path(cfg)):
            sel = json.load(open(selection_path(cfg)))
            L.append(f"Đã chọn (khoá): **{sel['chosen_name']}** ({sel['time']}).\n")
    tp = _r(out, "e1b-test", "pairs")
    if tp is not None:
        L.append("## E1b test — H3 (ảnh test, bộ ô ĐÁNH GIÁ độc lập)\n")
        L.append(h3_table(cfg, out, ci))
        L.append(mandatory_table(cfg, out, "e1b-test"))

    # ── E2
    e2p = _r(out, "e2", "pairs")
    if e2p is not None:
        L.append("## E2 — kiểm định ngoài mẫu (ảnh và ô mới)\n")
        L.append(e2_tables(cfg, out, ci))
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[sampling-infid] {rp}")
    return rp


def mandatory_table(cfg: dict, out: str, stage: str) -> str:
    from .report import md
    pr = pcfg(cfg)
    Km = int(pr["K_main"] or 4)
    p = _r(out, stage, "pairs")
    if p is None:
        return "_(không có dữ liệu)_\n"
    ev = p[(p.set == "eval") & (p.group == "all")]
    agg = {"L_mu": "mean", "Ve": "mean", "Vo": "mean", f"dV_{Km}": "mean", f"J_rev_{Km}": "mean", f"J_indep_{Km}": "mean",
           "quad_rel_proj": "mean", "compl_gap_abs": "mean"}
    agg = {k: v for k, v in agg.items() if k in ev}
    t = ev.groupby(["config", "dset"]).agg(agg).reset_index()
    rho = ev.groupby(["config", "dset"]).apply(lambda g: 2 * g.Ve.sum() / g.Vpath.sum() if g.Vpath.sum() > 0 else np.nan)
    t = t.merge(rho.rename("rho_pooled").reset_index(), on=["config", "dset"])
    if "signflip_p" in ev:
        t = t.merge(ev.groupby(["config", "dset"]).signflip_p.median().rename("p_signflip_med").reset_index(), on=["config", "dset"])
    fl = _r(out, stage, "floors")
    if fl is not None:
        f = fl.groupby(["dset", "G"])[["L_free_oos", "L_comp_oos"]].mean().reset_index()
        f = f[f.G == f.G.min()].drop(columns="G")
        t = t.merge(f, on="dset", how="left")
    cost = _r(out, stage, "cost")
    if cost is not None:
        t = t.merge(cost.groupby("config")[["grads", "seconds"]].mean().reset_index(), on="config", how="left")
    return md(t)


def h3_table(cfg: dict, out: str, ci) -> str:
    import pandas as pd
    from .report import md
    pr = pcfg(cfg)
    D = icfg(cfg)["D"]
    sel = load_selection(cfg)
    b = C._merge(STAGE_DEFAULT["e1b"], icfg(cfg).get("e1b", {}) or {})
    p = _r(out, "e1b-test", "pairs")
    ig = _r(out, "e1b-test", "ig")
    q = p[(p.set == "eval") & (p.group == "all") & (p.dset == D["primary"])].set_index("image")
    ch = q[q.config == sel["chosen_name"]].L_mu
    rows = []
    comps = [("SPI baseline", q[q.config == cname(b["baseline"])].L_mu)]
    if ig is not None:
        g = ig[(ig.set == "eval") & (ig.group == "all") & (ig.dset == D["primary"])].set_index("image").L
        comps.append(("IG", g))
    decisions = []
    for nm, other in comps:
        j = pd.concat([ch.rename("ch"), other.rename("ot")], axis=1).dropna()
        if not len(j):
            continue
        m_, lo, hi = ci(j.ot - j.ch)
        dec = ST.decide(lo, hi, pr["delta3"]) if nm != "SPI baseline" or sel["chosen_name"] != cname(b["baseline"]) else "trùng baseline"
        decisions.append(dec)
        lr = np.log(j.ot.clip(lower=1e-12) / j.ch.clip(lower=1e-12))
        rows.append({"so với": nm, "n ảnh": len(j), f"giảm L_D(μ) ({sel['chosen_name']})": f"{m_:.4g} [{lo:.4g}, {hi:.4g}]",
                     "log-infid theo cặp": _ci_str(ci(lr)), "quyết định": dec})
    allc = all(d == ST.CONFIRM for d in decisions) and len(decisions) == 2
    s = md(pd.DataFrame(rows))
    s += (f"\nH3 (phép thử giao–hợp: phải đạt CẢ hai so sánh, không cần hiệu chỉnh multiplicity): **"
          f"{ST.CONFIRM if allc else (ST.REJECT if any(d == ST.REJECT for d in decisions) else ST.OPEN)}**. Cấu hình chọn trên ảnh dev + "
          "bộ ô chọn; đánh giá trên ảnh test + bộ ô đánh giá độc lập.\n")
    return s


def e2_tables(cfg: dict, out: str, ci) -> str:
    import pandas as pd
    from .report import md
    pr = pcfg(cfg)
    Km = int(pr["K_main"] or 4)
    p = _r(out, "e2", "pairs")
    ig = _r(out, "e2", "ig")
    ev = p[(p.set == "eval") & (p.group == "all")]
    cols = [c for c in ("L_mu", "Ve", "Vo", f"dV_{Km}", f"J_rev_{Km}", f"J_indep_{Km}") if c in ev]
    s = "**L_D(μ), V̂_e, V̂_o, ΔV_D, J theo từng D** (trung bình theo ảnh):\n\n" + md(ev.groupby(["dset", "config"])[cols].mean().reset_index())
    if ig is not None:
        s += "\nIG:\n\n" + md(ig[(ig.set == "eval") & (ig.group == "all")].groupby(["dset", "config"]).L.mean().reset_index())
    tab = []
    for dn, g in ev.groupby("dset"):
        if f"dV_{Km}" in g:
            for c, h in g.groupby("config"):
                tab.append({"D": dn, "config": c, f"ΔV_D({Km})": _ci_str(ci(h[f"dV_{Km}"])), "L_D(μ)": _ci_str(ci(h.L_mu))})
    s += "\nCI theo ảnh:\n\n" + md(pd.DataFrame(tab))
    ex = _r(out, "e2", "ext")
    if ex is not None:
        mets = [c for c in ("insdel_blur", "road", "road_morf", "road_lerf", "iou", "pointing", "auroc_mask", "ap_mask") if c in ex]
        s += "\n**Metric ngoài** (ước lượng thực tế: rev với K chính; IG):\n\n" + md(ex.groupby(["config", "estimator"])[mets].mean().reset_index())
    e2d = os.path.join(out, "e2")
    for sub in (sorted(os.listdir(e2d)) if os.path.isdir(e2d) else []):
        if sub.startswith("arch_"):
            from .report import _read
            pa = _read(os.path.join(out, "e2", sub), "pairs")
            if pa is not None:
                q = pa[(pa.set == "eval") & (pa.group == "all")]
                s += f"\n**Kiến trúc {sub[5:]}**:\n\n" + md(q.groupby(["dset", "config"])[cols].mean().reset_index())
    s += ("\nD nền blur dùng dự đoán chỉ thị p_S = c·1_S (nền không nằm trên đoạn x′–x) — so sánh giữa cấu hình vẫn hợp lệ nhưng "
          "không phải infid đúng nghĩa Yeh et al.\n")
    return s
