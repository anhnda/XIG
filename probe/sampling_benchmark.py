"""
P4 — Real-CNN Stochastic Path Sampling Benchmark (ResNet-50 IMAGENET1K_V2, logit lớp dự đoán trên ảnh gốc, baseline đen,
cùng tiền xử lý / tập ảnh pilot / quy ước xếp hạng với P0–P3). Không train, không sửa mô hình.

Câu hỏi: với ngân sách gradient giới hạn, sampler có cấu trúc có vượt SPI độc lập (S0) một cách đáng tin cậy không?

Mọi đường có cùng N nút và M điểm cầu phương phân tầng mỗi đoạn (M = 1 ≡ canon_rand) ⇒ cùng N·M lượt forward+backward
mỗi đường. K = số đường hoàn chỉnh của ensemble (cặp antithetic tính cả hai), B = K·N·M = ngân sách gradient của ensemble.
  S0  SPI độc lập — đúng sampler hiện tại (paths.spi10)                       operator spi
  S1  antithetic: K/2 cặp (Z, Z^R), Z = đường S0 cùng chỉ số                     operator spi    (mỗi đường đúng q_SPI)
  S2  patch, pha cân bằng toàn cục (Latin hypercube giữa các đường)              operator patch
  S3  antithetic + S2: K/2 đường S2 đầu và đường đảo của chúng                    operator patch
  S4  patch, pha cân bằng lân cận (cặp kề cạnh) + ngẫu nhiên toàn cục            operator patch
  S5  patch, pha cố định cho cả ensemble (đối chứng thiên lệch thứ tự)            operator patch
  S6  patch, pha độc lập mỗi đường (đối chứng cùng operator với S2–S5)           operator patch
  S7  patch, thứ tự mềm theo LIME-grid (hệ số cao → muộn); prior tốn forward     operator lime (KHÁC)
Định nghĩa lịch trình: probe/sampling.py. Common random numbers: đường k của mọi phương pháp patch dùng cùng bộ sinh hình
dạng (chỉ khác pha); thành phần pixel (η > 0) là đường S0 cùng chỉ số; cùng khoá cầu phương cho cùng đường ⇒ đường dùng
chung (S1 ⊃ nửa đầu S0, S3 ⊃ nửa đầu S2) không tính lại — tiết kiệm tính toán, KHÔNG phải ngân sách (B vẫn = K·N·M).

Khối (hash riêng mỗi khối; cache <out>/cache/<ảnh>/; chạy lại bỏ qua phần đã xong):
  ref      μ_ref của operator spi và patch (lưới, η chính) từ K_ref đường độc lập (khoá riêng, không rò vào ensemble),
           tr Var(μ̂_ref) để khử chệch RMSE, cổng nhiễu cầu phương (cùng đường S0, hai bộ U độc lập).
  primary  sampling.primary (mặc định S0–S4 + S6), K ∈ sampling.K, `seeds` ensemble độc lập.
  phụ      sampling.secondary theo thứ tự ưu tiên — chỉ chạy khi ước tính còn nằm trong ngân sách phụ.
"""

from __future__ import annotations

import copy
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import defaultdict

import torch

from . import config as C
from . import sampling as SM
from .mechanism import aw_stream
from .metrics import _rank_top, energy_mask, iou, make_base, spearman, tie_perm, topk_overlap
from .model import TargetFn
from .pilot import Ctx, sync
from .quad import Counter

F64 = torch.float64
P4_VERSION = 1
PATCH_DESIGN = {"S2": "global", "S3": "global", "S4": "nbr", "S5": "fixed", "S6": "indep", "S7": "lime"}
ALL_METHODS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
DESCR = {
    "S0": "SPI độc lập (sampler hiện tại)", "S1": "antithetic SPI", "S2": "patch, cân bằng toàn cục",
    "S3": "antithetic + cân bằng toàn cục", "S4": "patch, cân bằng lân cận", "S5": "patch, pha cố định (đối chứng)",
    "S6": "patch, pha độc lập (đối chứng cùng operator)", "S7": "patch, thứ tự mềm LIME (cao → muộn)"}


# ─────────────────────────────────────────────────────────────────────────────
# cấu hình
# ─────────────────────────────────────────────────────────────────────────────

def settings(cfg: dict) -> dict:
    if "sampling" not in cfg:
        raise SystemExit("config thiếu mục sampling:")
    s = copy.deepcopy(cfg["sampling"])
    src = s.get("spi_from", "spi10")
    pc = cfg["paths"][src]
    if pc.get("kind") != "spi":
        raise SystemExit(f"sampling.spi_from = {src} không phải đường spi")
    s["alpha"], s["N"] = float(pc["alpha"]), int(pc["steps"])      # S0 = đúng tham số SPI hiện tại, không cho ghi đè
    s["M"] = int(s.get("M", 1))
    Ks = sorted({int(k) for k in s["K"]})
    bad = [k for k in Ks if k < 2 or not SM.is_pow2(k)]
    if bad:
        raise SystemExit(f"sampling.K phải là luỹ thừa của 2 và ≥ 2 (thiết kế lồng nhau, cặp antithetic): {bad}")
    s["K"], s["Kmax"] = Ks, Ks[-1]
    s["grid"], s["eta"], s["seeds"] = int(s["grid"]), float(s.get("eta", 0.0)), int(s["seeds"])
    if s["seeds"] < 1:
        raise SystemExit("sampling.seeds ≥ 1")
    for m in s["primary"]:
        if m not in ALL_METHODS:
            raise SystemExit(f"phương pháp không biết: {m}")
    if "S0" not in s["primary"]:
        raise SystemExit("S0 (SPI độc lập) phải có trong sampling.primary — là baseline chính")
    for b in s.get("secondary", []) or []:
        for m in b["methods"]:
            if m not in ALL_METHODS:
                raise SystemExit(f"phương pháp không biết: {m}")
            if m in ("S0", "S1"):
                raise SystemExit("S0/S1 không phụ thuộc lưới/η — chỉ đặt trong primary")
    return s


def blocks(S: dict) -> list:
    out = [{"name": "primary", "methods": list(S["primary"]), "seeds": S["seeds"], "grid": S["grid"], "eta": S["eta"]}]
    for b in S.get("secondary", []) or []:
        out.append({"name": str(b["name"]), "methods": list(b["methods"]), "seeds": int(b.get("seeds", S["seeds"])),
                    "grid": int(b.get("grid", S["grid"])), "eta": float(b.get("eta", S["eta"]))})
    names = [b["name"] for b in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"tên khối trùng: {names}")
    return out


def patch_op(grid: int, eta: float) -> str:
    return f"patch@g{grid}e{eta:g}"


def operator(m: str, grid: int, eta: float) -> str:
    if m in ("S0", "S1"):
        return "spi"
    if m == "S7":
        return f"lime@g{grid}e{eta:g}"
    return patch_op(grid, eta)


def label(m: str, blk: dict, S: dict) -> str:
    if m in ("S0", "S1") or (blk["grid"] == S["grid"] and blk["eta"] == S["eta"]):
        return m
    return f"{m}@g{blk['grid']}e{blk['eta']:g}"


def method_paths(m: str, grid: int, eta: float, Kmax: int) -> list:
    """Khoá đường theo thứ tự ensemble; ensemble K = K khoá đầu (S1/S3: K/2 cặp đầu)."""
    h = Kmax // 2
    if m == "S0":
        return [("spi", k) for k in range(Kmax)]
    if m == "S1":
        return [x for j in range(h) for x in (("spi", j), ("spiR", j))]
    if m == "S3":
        return [x for j in range(h) for x in (("pat", "global", grid, eta, j), ("patR", "global", grid, eta, j))]
    return [("pat", PATCH_DESIGN[m], grid, eta, k) for k in range(Kmax)]


def kstr(key) -> str:
    return ":".join(f"{v:g}" if isinstance(v, float) else str(v) for v in key)


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def base_spec(cfg: dict, S: dict) -> dict:
    keys = ("alpha", "N", "M", "K", "grid", "eta", "insdel_bases", "insdel_steps", "signed_ranking_secondary",
            "overlap_tol", "lime", "locality", "spi_from")
    return {"version": P4_VERSION, "seed": cfg["seed"], "model": cfg["model"], "data": cfg["data"],
            "insdel": cfg["metrics"]["insdel"], "topk_frac": cfg["metrics"]["topk_frac"], **{k: S.get(k) for k in keys}}


def digests(cfg: dict, S: dict) -> dict:
    base = base_spec(cfg, S)
    d = {"ref": _h({"base": base, "ref": S["reference"]}), "lime": _h({"base": base, "what": "lime"})}
    for b in blocks(S):
        d[b["name"]] = _h({"base": base, "block": b, "ref": S["reference"]})     # dòng RMSE phụ thuộc ref
    return d


def sampling_out(cfg: dict) -> str:
    d = os.path.join(C.out_dir(cfg, "pilot"), "sampling")
    os.makedirs(d, exist_ok=True)
    return d


def report_path(cfg: dict) -> str:
    return os.path.join(C.out_dir(cfg, "pilot"), "ATTR_PROBE_SAMPLING_REPORT.md")


# ─────────────────────────────────────────────────────────────────────────────
# một ảnh
# ─────────────────────────────────────────────────────────────────────────────

class Env:
    def __init__(self, ctx: Ctx, S: dict, idx: int):
        self.ctx, self.S = ctx, S
        it, x, mask, cls = ctx.load(idx)                      # một forward để lấy lớp dự đoán
        SM.FwdCounter.n += 1
        self.it, self.x, self.mask, self.cls, self.image = it, x, mask, cls, it.image
        self.fn = TargetFn(ctx.bb, cls)
        self.x0 = make_base(x, "black")
        self.dx = x - self.x0
        self.shape = tuple(self.dx.shape[1:])
        self.H, self.W = self.shape[1], self.shape[2]
        self.perm = tie_perm(it.image, self.H * self.W, ctx.seed, ctx.dev)
        ic = ctx.cfg["metrics"]["insdel"]
        self.bases = {b: make_base(x, b, ic["blur_kernel"], ic["blur_sigma"]) for b in S["insdel_bases"]}
        self.ins_batch = int(ic["batch"])
        self.topk = float(ctx.cfg["metrics"]["topk_frac"])
        with torch.no_grad():
            f = self.fn(torch.cat([self.x0, x])).double()
        SM.FwdCounter.n += 2
        self.F0, self.F1 = float(f[0]), float(f[1])
        self.D = self.F1 - self.F0
        self._lab, self._plab = {}, {}
        self.ref = {}
        self.lime = {}

    def lab(self, g: int) -> torch.Tensor:
        if g not in self._lab:
            self._lab[g] = SM.grid_elem_labels(self.shape, g).to(self.ctx.dev)
        return self._lab[g]

    def plab(self, g: int) -> torch.Tensor:
        if g not in self._plab:
            self._plab[g] = SM.grid_pixel_labels(self.H, self.W, g).to(self.ctx.dev)
        return self._plab[g]

    def gen(self, *parts):
        return self.ctx.gen(*parts)

    def cgen(self, *parts) -> torch.Generator:
        return torch.Generator().manual_seed(C.seed_of(self.ctx.seed, *parts))

    def info(self) -> dict:
        return {"image": self.image, "index": self.it.index, "cls": self.cls, "F_x": self.F1, "F_base": self.F0, "D": self.D,
                "mask_area": float(self.mask.float().mean())}


def quality(E: Env, A: torch.Tensor, ref=None, curves: bool = False, signed: bool = False):
    """Metric của một bản đồ attribution (C,H,W). Hạng InsDel/IoU theo Σ_c|A| (quy ước P0–P3)."""
    S = E.S
    A64 = A.double()
    A32 = A.float()
    s = float(A64.sum())
    D = E.D
    q = {"sumA": s, "resid_rel": (s - D) / abs(D) if D else float("nan"),
         "abs_resid_rel": abs(s - D) / abs(D) if D else float("nan"),
         "l1": float(A64.abs().sum()), "l2": float(A64.norm()),
         "iou": iou(A32, E.mask, E.perm), "energy_mask": energy_mask(A32, None, E.mask)}
    k = max(1, int(E.topk * E.H * E.W))
    top = _rank_top(A32.abs().sum(0), k, E.perm)
    q["top_in_mask"] = float((top & E.mask.flatten().to(top.device)).sum()) / k
    cv = {}
    for b, base in E.bases.items():
        r = SM.insdel_full(E.ctx.bb, E.cls, E.x, base, A32, int(S["insdel_steps"]), E.ins_batch, curves=curves)
        q[f"ins_{b}"], q[f"del_{b}"], q[f"insdel_{b}"] = r["ins"], r["del"], r["diff"]
        if curves:
            cv[b] = r["curves"]
        if signed:
            r2 = SM.insdel_full(E.ctx.bb, E.cls, E.x, base, A32, int(S["insdel_steps"]), E.ins_batch, signed=True)
            q[f"signed_ins_{b}"], q[f"signed_del_{b}"], q[f"signed_insdel_{b}"] = r2["ins"], r2["del"], r2["diff"]
    if ref is not None:
        mu = ref["mu"].to(A64.device).double()
        e = A64 - mu
        mse, nm = float((e * e).sum()), float(mu.norm())
        q["rel_rmse"] = math.sqrt(mse) / nm
        q["rel_rmse_deb"] = math.sqrt(max(mse - ref["trvar"], 0.0)) / nm
        q["rel_mse_deb"] = (mse - ref["trvar"]) / (nm * nm)
        q["cos_ref"] = float((A64 * mu).sum()) / max(float(A64.norm()) * nm, 1e-300)
        q["spearman_ref"] = spearman(A32, mu.float())
        q["topk_ref"] = topk_overlap(A32, mu.float(), E.topk, E.perm)
    return q, cv


class Bank:
    """Sinh lịch trình theo khoá đường cho một (ảnh, seed, tag). tag 'ens' cho ensemble, 'ref' cho reference (khoá riêng)."""

    def __init__(self, E: Env, r: int, tag: str):
        self.E, self.r, self.tag = E, int(r), tag
        self.S = E.S
        self._phase = {}

    def phase(self, design: str, grid: int) -> torch.Tensor:
        key = (design, grid)
        if key not in self._phase:
            E, S = self.E, self.S
            P = grid * grid
            common = SM.common_phase_noise(P, S["Kmax"], E.cgen("P4phase-common", self.tag, self.r, E.image, grid))
            lr = E.lime[grid]["rank"] if design == "lime" else None
            phi, _ = SM.phase_design(design, P, S["Kmax"], E.cgen("P4phase", design, self.tag, self.r, E.image, grid),
                                     g=grid, common=common, lime_rank=lr, tau=float(S["lime"]["tau"]))
            self._phase[key] = phi
        return self._phase[key]

    def Z(self, key) -> torch.Tensor:
        E, S = self.E, self.S
        kind = key[0]
        if kind == "spi":
            return SM.spi_schedule(S["alpha"], S["N"], E.shape, E.gen("P4spi", self.tag, self.r, E.image, key[1]))
        if kind == "spiR":
            return SM.reverse_nodes(self.Z(("spi", key[1])))
        if kind in ("pat", "patR"):
            _, design, grid, eta, k = key
            phi = self.phase(design, grid)[k]
            Zp = SM.patch_schedule(S["alpha"], S["N"], phi, E.gen("P4pat", self.tag, self.r, E.image, grid, k))
            Zf = SM.expand_patch(Zp, E.lab(grid), E.shape)
            if eta > 0:
                Zf = SM.mix(Zf, self.Z(("spi", k)), eta)
            return SM.reverse_nodes(Zf) if kind == "patR" else Zf
        raise ValueError(key)

    def eval_path(self, key, light: bool = False, qtag: str = "") -> dict:
        E, S = self.E, self.S
        Z = self.Z(key)
        X = SM.nodes(E.x0, E.dx, Z)
        A, _ = aw_stream(E.fn, X, S["M"], E.gen("P4q", self.tag, self.r, E.image, qtag, *key), E.ctx.batch)
        out = {"A": A}
        if not light:
            g = key[2] if key[0] in ("pat", "patR") else S["grid"]
            P = g * g
            out["grid"] = g
            out["tau"] = (SM.midpoint_index(SM.patch_mean_schedule(Z, E.lab(g), P)).double() / S["N"]).cpu()
            out["tau_elem"] = SM.midpoint_index(Z).reshape(-1).to(torch.uint8)
            out["Cp"] = SM.patch_sum(A, E.lab(g), P).cpu()
            out["d_hat"] = SM.d_hat_cols(Z)
        del Z, X
        return out


def get_lime(E: Env, grid: int, cdir: str, dg: str) -> dict:
    if grid in E.lime:
        return E.lime[grid]
    p = os.path.join(cdir, f"lime_g{grid}.pt")
    if os.path.exists(p):
        obj = torch.load(p, weights_only=False)
        if obj.get("digest") == dg:
            E.lime[grid] = obj
            return obj
    lc = E.S["lime"]
    t0, f0 = time.time(), SM.FwdCounter.n
    r = SM.lime_grid(E.fn, E.x, E.x0, E.plab(grid), grid * grid, int(lc["n_samples"]),
                     E.cgen("P4lime", E.image, grid), float(lc["kernel_width"]), float(lc["ridge"]), E.ctx.batch)
    r.update({"digest": dg, "grid": grid, "seconds": time.time() - t0, "fwd_evals": SM.FwdCounter.n - f0})
    torch.save(r, p)
    E.lime[grid] = r
    return r


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a * b).sum()) / max(float(a.norm() * b.norm()), 1e-300)


def _rank_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    def rk(v):
        v = v.double().flatten()
        r = torch.empty_like(v)
        r[torch.argsort(v)] = torch.arange(v.numel(), device=v.device, dtype=v.dtype)
        return r - r.mean()
    x, y = rk(a), rk(b)
    return float((x * y).sum() / max(float(x.norm() * y.norm()), 1e-300))


def antithetic_rows(E: Env, blk: dict, r: int, paths: dict) -> tuple[list, list]:
    """D1: cặp antithetic so với cặp độc lập cùng operator, trên cùng seed. Độ lệch tính quanh μ_ref của operator (nếu có),
    nếu không thì quanh trung bình các đường độc lập cùng operator của seed (ghi rõ trong cột `center`)."""
    S = E.S
    h = S["Kmax"] // 2
    grid, eta = blk["grid"], blk["eta"]
    pop = patch_op(grid, eta)
    fams = [("S1_antithetic", "spi", [(("spi", j), ("spiR", j)) for j in range(h)]),
            ("S0_independent", "spi", [(("spi", j), ("spi", j + h)) for j in range(h)]),
            ("S3_antithetic", pop, [(("pat", "global", grid, eta, j), ("patR", "global", grid, eta, j)) for j in range(h)]),
            ("S6_independent", pop, [(("pat", "indep", grid, eta, j), ("pat", "indep", grid, eta, j + h)) for j in range(h)])]
    indep_keys = {"spi": [("spi", k) for k in range(S["Kmax"])],
                  pop: [("pat", "indep", grid, eta, k) for k in range(S["Kmax"])]}
    lab, P = E.lab(grid), grid * grid
    pair_rows, pooled = [], []
    for fam, op, prs in fams:
        if not all(a in paths and b in paths for a, b in prs):
            continue
        if op in E.ref:
            center, ctype = E.ref[op]["mu"].to(E.ctx.dev).double(), "ref"
        elif all(k in paths for k in indep_keys[op]):
            center, ctype = torch.stack([paths[k]["A"] for k in indep_keys[op]]).mean(0), "seed_mean"
        else:
            continue
        sab = saa = sbb = 0.0
        pab = paa = pbb = 0.0
        pm, sm = [], []
        for j, (ka, kb) in enumerate(prs):
            A, B = paths[ka]["A"], paths[kb]["A"]
            ea, eb = A - center, B - center
            qa, qb = SM.patch_sum(ea, lab, P), SM.patch_sum(eb, lab, P)
            ab, aa, bb = float((ea * eb).sum()), float((ea * ea).sum()), float((eb * eb).sum())
            sab, saa, sbb = sab + ab, saa + aa, sbb + bb
            pab, paa, pbb = pab + float((qa * qb).sum()), paa + float((qa * qa).sum()), pbb + float((qb * qb).sum())
            pair_mse = float((((ea + eb) / 2) ** 2).sum())
            pm.append(pair_mse)
            sm.append((aa + bb) / 2)
            pair_rows.append({"image": E.image, "block": blk["name"], "seed": r, "family": fam, "operator": op, "center": ctype,
                              "pair": j, "raw_cos": _cos(A, B), "centered_cos": ab / max(math.sqrt(aa * bb), 1e-300),
                              "patch_centered_cos": _cos(qa, qb),
                              "rel_diff": float((A - B).norm()) / max(0.5 * float(A.norm() + B.norm()), 1e-300),
                              "pair_mse": pair_mse, "single_mse_mean": (aa + bb) / 2})
        n = len(prs)
        mean_single = sum(sm) / n
        pooled.append({"image": E.image, "block": blk["name"], "seed": r, "family": fam, "operator": op, "center": ctype,
                       "n_pairs": n, "rho_elem": sab / max(math.sqrt(saa * sbb), 1e-300),
                       "rho_patch": pab / max(math.sqrt(paa * pbb), 1e-300),
                       "pair_mse_mean": sum(pm) / n, "single_mse_mean": mean_single,
                       "pair_vs_half_single": (sum(pm) / n) / max(mean_single / 2, 1e-300),   # = 1 + ρ khi hai phương sai bằng nhau
                       "frac_raw_cos_gt_0.99": sum(1 for x in pair_rows[-n:] if x["raw_cos"] > 0.99) / n,
                       "frac_centered_cos_gt_0.9": sum(1 for x in pair_rows[-n:] if x["centered_cos"] > 0.9) / n})
    return pair_rows, pooled


def block_seed(E: Env, blk: dict, r: int, cdir: str, dgs: dict) -> dict:
    S = E.S
    Kmax, Ks, N, M = S["Kmax"], S["K"], S["N"], S["M"]
    grid, eta = blk["grid"], blk["eta"]
    if "S7" in blk["methods"]:
        get_lime(E, grid, cdir, dgs["lime"])                    # chi phí prior tính riêng (budget: lime)
    bank = Bank(E, r, "ens")
    mk = {m: method_paths(m, grid, eta, Kmax) for m in blk["methods"]}
    uniq = list(dict.fromkeys(k for ks in mk.values() for k in ks))
    t0, n0 = time.time(), Counter.n
    paths = {key: bank.eval_path(key) for key in uniq}
    sync()
    t_grad, n_grad = time.time() - t0, Counter.n - n0
    t1, f1 = time.time(), SM.FwdCounter.n
    rows = defaultdict(list)
    maps, heat = {}, {}
    signed = bool(S.get("signed_ranking_secondary", False))
    for m, ks in mk.items():
        lm, op = label(m, blk, S), operator(m, grid, eta)
        g_m = grid if m not in ("S0", "S1") else S["grid"]
        for K in Ks:
            sel = ks[:K]
            A = torch.stack([paths[k]["A"] for k in sel]).mean(0)
            last = K == Kmax
            q, cv = quality(E, A, E.ref.get(op), curves=(last and r == 0), signed=(last and signed))
            rows["per_seed"].append({"image": E.image, "block": blk["name"], "method": lm, "base": m, "grid": g_m,
                                     "eta": 0.0 if m in ("S0", "S1") else eta, "operator": op, "K": K, "seed": r,
                                     "B_grad": K * N * M, "B_prior_fwd": int(E.lime[grid]["n_queries"]) if m == "S7" else 0,
                                     "n_unique_paths": len(set(sel)), **q})
            maps[(lm, K)] = A.float().cpu()
            for b, c in cv.items():
                for kind in ("ins", "del"):
                    for meas in ("prob", "logit"):
                        v = c[f"{kind}_{meas}"]
                        for st in range(len(v)):
                            rows["curves"].append({"image": E.image, "block": blk["name"], "method": lm, "K": K, "seed": r,
                                                   "base_kind": b, "curve": kind, "measure": meas, "step": st,
                                                   "frac": float(c["frac"][st]), "value": float(v[st])})
            tau = torch.stack([paths[k]["tau"] for k in sel])
            te = torch.stack([paths[k]["tau_elem"] for k in sel]).float() / N
            cov = SM.coverage_stats(tau, g_m, float(S["overlap_tol"]))
            cov["elem_tau_spread"] = float(te.std(0).mean()) if K > 1 else 0.0
            cov["d_hat_mean"] = sum(paths[k]["d_hat"] for k in sel) / K
            rows["coverage"].append({"image": E.image, "block": blk["name"], "method": lm, "base": m, "operator": op,
                                     "grid": g_m, "K": K, "seed": r, **cov})
            if last and r == 0:
                heat[lm] = {"amap": A.abs().sum(0).half().cpu(), "signed": A.sum(0).half().cpu()}
    for key, pr in paths.items():
        s = float(pr["A"].sum())
        rows["paths"].append({"image": E.image, "block": blk["name"], "seed": r, "path": kstr(key), "kind": key[0],
                              "design": key[1] if key[0] in ("pat", "patR") else "spi", "sumA": s, "D": E.D,
                              "resid_rel": (s - E.D) / abs(E.D) if E.D else float("nan"), "d_hat": pr["d_hat"],
                              "tau_patch_sd": float(pr["tau"].std())})
    pr_rows, pooled = antithetic_rows(E, blk, r, paths)
    rows["cov_pairs"] += pr_rows
    rows["covariance"] += pooled
    loc = None
    s6 = mk.get("S6")
    if s6:
        loc = {"Cp": torch.stack([paths[k]["Cp"] for k in s6]), "tau": torch.stack([paths[k]["tau"] for k in s6]), "grid": grid}
    sync()
    rows["budget"].append({"image": E.image, "block": blk["name"], "seed": r, "part": "ensemble",
                           "grad_evals": n_grad, "fwd_evals": SM.FwdCounter.n - f1, "seconds_grad": t_grad,
                           "seconds_metrics": time.time() - t1, "unique_paths": len(uniq),
                           "nominal_paths": sum(len(v) for v in mk.values()), "evals_per_path": N * M})
    del paths
    return {"rows": dict(rows), "maps": maps, "heat": heat, "loc": loc}


def block_summary(E: Env, blk: dict, seed_objs: list) -> dict:
    """Theo ảnh: trung bình metric qua seed (tránh giả lặp) + ổn định giữa các ensemble độc lập + D3."""
    S = E.S
    R = len(seed_objs)
    gp = S["grid"]
    lab, P = E.lab(gp), gp * gp
    by = defaultdict(list)
    for o in seed_objs:
        for row in o["rows"].get("per_seed", []):
            by[(row["method"], row["K"])].append(row)
    rows = defaultdict(list)
    for (m, K), rs in by.items():
        num = [k for k, v in rs[0].items() if isinstance(v, (int, float)) and k not in ("K", "seed", "grid")]
        mean = {k: sum(r_[k] for r_ in rs) / len(rs) for k in num}
        row = {"image": E.image, "block": blk["name"], "method": m, "base": rs[0]["base"], "operator": rs[0]["operator"],
               "grid": rs[0]["grid"], "K": K, "n_seeds": len(rs), **mean}
        As = [o["maps"][(m, K)].to(E.ctx.dev).double() for o in seed_objs if (m, K) in o.get("maps", {})]
        if len(As) >= 2:
            st = torch.stack(As)
            row["var_elem"] = float(st.var(0, unbiased=True).sum())
            Pp = torch.stack([SM.patch_sum(a, lab, P) for a in As])
            row["var_patch"] = float(Pp.var(0, unbiased=True).sum())
            mu = st.mean(0)
            row["var_elem_rel"] = row["var_elem"] / max(float((mu * mu).sum()), 1e-300)
            cs, sp, tk, pc, ps = [], [], [], [], []
            for a in range(len(As)):
                for b in range(a + 1, len(As)):
                    cs.append(_cos(As[a], As[b]))
                    sp.append(spearman(As[a].float(), As[b].float()))
                    tk.append(topk_overlap(As[a].float(), As[b].float(), E.topk, E.perm))
                    pc.append(_cos(Pp[a], Pp[b]))
                    ps.append(_rank_corr(Pp[a], Pp[b]))
            for k_, v in (("stab_cos", cs), ("stab_spearman", sp), ("stab_topk", tk), ("stab_patch_cos", pc),
                          ("stab_patch_spearman", ps)):
                row[k_] = sum(v) / len(v)
        rows["per_image"].append(row)
    locs = [o["loc"] for o in seed_objs if o.get("loc") is not None]
    if locs:
        g = locs[0]["grid"]
        Cp = torch.cat([l_["Cp"] for l_ in locs])
        tau = torch.cat([l_["tau"] for l_ in locs])
        for rr in SM.order_effect_by_distance(Cp, tau, g, E.D, int(S["locality"]["n_null"]), E.cgen("P4loc", E.image, blk["name"])):
            rows["locality"].append({"image": E.image, "block": blk["name"], "grid": g, "n_paths": int(tau.shape[0]), **rr})
    heat = seed_objs[0].get("heat", {}) if seed_objs else {}
    return {"rows": dict(rows), "heat": heat, "n_seeds": R}


def run_reference(E: Env) -> dict:
    S = E.S
    rc = S["reference"]
    Kref = int(rc["K"])
    bank = Bank(E, 0, "ref")
    t0, n0, f0 = time.time(), Counter.n, SM.FwdCounter.n
    refs, rows, heat = {}, [], {}
    for op_kind in rc["operators"]:
        if op_kind == "spi":
            keys, op = [("spi", k) for k in range(Kref)], "spi"
        elif op_kind == "patch":
            keys, op = [("pat", "indep", S["grid"], S["eta"], k) for k in range(Kref)], patch_op(S["grid"], S["eta"])
        else:
            raise SystemExit(f"reference.operators không biết: {op_kind}")
        s1 = s2 = half = None
        h = Kref // 2
        for i, key in enumerate(keys):
            A = bank.eval_path(key, light=True)["A"]
            if s1 is None:
                s1, s2, half = torch.zeros_like(A), torch.zeros_like(A), torch.zeros_like(A)
            s1 += A
            s2 += A * A
            if i < h:
                half += A
            if op_kind == "spi" and i < int(rc.get("quad_gate_paths", 0)):
                Ab = bank.eval_path(key, light=True, qtag="alt")["A"]
                rows.append({"image": E.image, "kind": "quad_gate", "operator": op, "path": i,
                             "quad_var_path": float(((A - Ab) ** 2).sum()) / 2})
        mu = s1 / Kref
        var = ((s2 - Kref * mu * mu) / max(Kref - 1, 1)).clamp_min(0)
        trvar = float(var.sum()) / Kref
        h1, h2 = half / h, (s1 - half) / (Kref - h)
        q, _ = quality(E, mu, None)
        refs[op] = {"mu": mu.float().cpu(), "trvar": trvar, "var_path": float(var.sum()), "K": Kref}
        rows.append({"image": E.image, "kind": "reference", "operator": op, "K_ref": Kref, "var_path": float(var.sum()),
                     "trvar_mean": trvar, "ref_rel_se": math.sqrt(trvar) / max(float(mu.norm()), 1e-300),
                     "split_half_cos": _cos(h1, h2), "split_half_rel_diff": float((h1 - h2).norm()) / max(float(mu.norm()), 1e-300),
                     "d_D": E.D, **q})
        heat[f"ref:{op}"] = {"amap": mu.abs().sum(0).half().cpu(), "signed": mu.sum(0).half().cpu()}
        del s1, s2, half
    gates = [r_ for r_ in rows if r_["kind"] == "quad_gate"]
    if gates and "spi" in refs:
        qv = sum(r_["quad_var_path"] for r_ in gates) / len(gates)
        rows.append({"image": E.image, "kind": "quad_share", "operator": "spi", "quad_var_path": qv,
                     "var_path": refs["spi"]["var_path"], "quad_share": qv / max(refs["spi"]["var_path"], 1e-300)})
    ops = list(refs)
    if len(ops) >= 2:
        a, b = refs[ops[0]], refs[ops[1]]
        ma, mb = a["mu"].double(), b["mu"].double()
        rows.append({"image": E.image, "kind": "operator_diff", "operator": f"{ops[1]} − {ops[0]}",
                     "rel_l2": float((mb - ma).norm()) / max(float(ma.norm()), 1e-300),
                     "noise_floor_rel": math.sqrt(a["trvar"] + b["trvar"]) / max(float(ma.norm()), 1e-300),
                     "cos": _cos(ma, mb), "spearman": spearman(ma.float(), mb.float()),
                     "topk": topk_overlap(ma.float(), mb.float(), E.topk, E.perm.cpu())})
    sync()
    budget = {"image": E.image, "block": "ref", "seed": -1, "part": "reference", "grad_evals": Counter.n - n0,
              "fwd_evals": SM.FwdCounter.n - f0, "seconds_grad": time.time() - t0, "seconds_metrics": 0.0,
              "unique_paths": Kref * len(ops), "nominal_paths": Kref * len(ops), "evals_per_path": S["N"] * S["M"]}
    return {"refs": refs, "rows": {"ref": rows, "budget": [budget]}, "heat": heat}


# ─────────────────────────────────────────────────────────────────────────────
# ước tính, kế hoạch
# ─────────────────────────────────────────────────────────────────────────────

def bench_fwd(ctx: Ctx) -> float:
    xb = torch.randn(ctx.batch, 3, 224, 224, device=ctx.dev)
    with torch.no_grad():
        ctx.bb(xb)
        sync()
        t = time.time()
        for _ in range(3):
            ctx.bb(xb)
        sync()
    return (time.time() - t) / (3 * ctx.batch)


def estimate(cfg: dict, S: dict, per_eval: float, per_fwd: float) -> dict:
    NM = S["N"] * S["M"]
    ins_one = len(S["insdel_bases"]) * 2 * (int(S["insdel_steps"]) + 1)
    n = int(S["n_images"])
    out = {"per_eval_ms": per_eval * 1e3, "per_fwd_ms": per_fwd * 1e3, "blocks": {}}
    for blk in blocks(S):
        uniq = len({k for m in blk["methods"] for k in method_paths(m, blk["grid"], blk["eta"], S["Kmax"])})
        R = blk["seeds"]
        grad = uniq * NM * R
        fwd = len(blk["methods"]) * len(S["K"]) * R * ins_one
        if S.get("signed_ranking_secondary"):
            fwd += len(blk["methods"]) * R * ins_one
        if "S7" in blk["methods"]:
            fwd += int(S["lime"]["n_samples"])
        out["blocks"][blk["name"]] = {"grad_per_image": grad, "fwd_per_image": fwd, "unique_paths_per_seed": uniq,
                                      "minutes": n * (grad * per_eval + fwd * per_fwd) / 60}
    rc = S["reference"]
    if rc.get("enabled", True):
        grad = (int(rc["K"]) * len(rc["operators"]) + int(rc.get("quad_gate_paths", 0))) * NM
        fwd = len(rc["operators"]) * ins_one
        out["blocks"]["ref"] = {"grad_per_image": grad, "fwd_per_image": fwd, "unique_paths_per_seed": int(rc["K"]) * len(rc["operators"]),
                                "minutes": n * (grad * per_eval + fwd * per_fwd) / 60}
    out["env_minutes"] = n * 3 * per_fwd / 60
    return out


def plan(S: dict, est: dict) -> dict:
    bud = S.get("budget", {}) or {}
    slack = float(bud.get("slack", 1.25))
    prim = est["blocks"]["primary"]["minutes"] + est["blocks"].get("ref", {}).get("minutes", 0.0) + est["env_minutes"]
    lim = float(bud.get("minutes", 30)) * slack
    ok = prim <= lim
    run_sec, skip_sec, acc = [], [], 0.0
    for blk in blocks(S)[1:]:
        m = est["blocks"][blk["name"]]["minutes"]
        if acc + m <= float(bud.get("secondary_minutes", 0)) * slack:
            run_sec.append(blk["name"])
            acc += m
        else:
            skip_sec.append(blk["name"])
    return {"primary_minutes": prim, "primary_limit": lim, "primary_ok": ok, "secondary_run": run_sec,
            "secondary_skipped": skip_sec, "secondary_minutes": acc, "enforce": bool(bud.get("enforce", True))}


def print_estimate(S: dict, est: dict, pl: dict, n_images: int):
    print(f"[sampling] {est['per_eval_ms']:.2f} ms/eval (forward+backward, lô), {est['per_fwd_ms']:.2f} ms/forward")
    print(f"[sampling] SPI α = {S['alpha']:g}, N = {S['N']}, M = {S['M']} → {S['N'] * S['M']} eval/đường; K = {S['K']}; "
          f"lưới {S['grid']}×{S['grid']}, η = {S['eta']:g}; {n_images} ảnh")
    for name, b in est["blocks"].items():
        print(f"  {name:10s} {b['unique_paths_per_seed']:4d} đường duy nhất/seed, {b['grad_per_image']:8d} grad + "
              f"{b['fwd_per_image']:7d} fwd mỗi ảnh ≈ {b['minutes']:.1f} phút")
    print(f"[sampling] chính (ref + primary) ≈ {pl['primary_minutes']:.1f} phút (giới hạn {pl['primary_limit']:.1f}); "
          f"phụ chạy: {pl['secondary_run'] or '—'} ≈ {pl['secondary_minutes']:.1f} phút; bỏ: {pl['secondary_skipped'] or '—'}")
    per_img = pl["primary_minutes"] / max(n_images, 1)
    if per_img > 0:
        print(f"[sampling] gợi ý: với ngân sách chính hiện tại tối đa ≈ {int(pl['primary_limit'] / per_img)} ảnh ở cấu hình này")


# ─────────────────────────────────────────────────────────────────────────────
# chạy
# ─────────────────────────────────────────────────────────────────────────────

def _load(p: str, dg: str):
    if os.path.exists(p):
        obj = torch.load(p, weights_only=False)
        if obj.get("digest") == dg:
            return obj
    return None


def process_image(ctx: Ctx, S: dict, out: str, idx: int, blks: list, dgs: dict, do_ref: bool) -> None:
    image = f"seg{idx:05d}"
    cdir = os.path.join(out, "cache", image)
    os.makedirs(cdir, exist_ok=True)
    state = {"E": None}
    ref = None
    rp = os.path.join(cdir, "ref.pt")

    def env() -> Env:
        if state["E"] is None:
            E = Env(ctx, S, idx)
            ip = os.path.join(cdir, "image.pt")
            if not os.path.exists(ip):
                torch.save({"info": E.info(), "x": E.it.x.half(), "mask": E.mask.cpu()}, ip)
            state["E"] = E
        if ref is not None:
            state["E"].ref = ref["refs"]
        return state["E"]

    if S["reference"].get("enabled", True):
        ref = _load(rp, dgs["ref"])
        if ref is None and do_ref:
            t0 = time.time()
            ref = run_reference(env())
            ref["digest"] = dgs["ref"]
            torch.save(ref, rp)
            print(f"    [ref] {image}: {time.time() - t0:.0f}s", flush=True)
            env()
    for blk in blks:
        name, dg = blk["name"], dgs[blk["name"]]
        sp = os.path.join(cdir, f"{name}_summary.pt")
        if _load(sp, dg) is not None:
            print(f"    [{name}] {image}: đã có, bỏ qua")
            continue
        objs = []
        for r in range(blk["seeds"]):
            p = os.path.join(cdir, f"{name}_s{r}.pt")
            obj = _load(p, dg)
            if obj is None or obj.get("maps") is None:
                t0, n0 = time.time(), Counter.n
                obj = block_seed(env(), blk, r, cdir, dgs)
                obj["digest"] = dg
                torch.save(obj, p)
                print(f"    [{name}] {image} seed {r}: {Counter.n - n0} grad eval, {time.time() - t0:.0f}s", flush=True)
            objs.append(obj)
        summ = block_summary(env(), blk, objs)
        summ["digest"] = dg
        torch.save(summ, sp)
        if not S.get("keep_seed_maps", False):
            for r, obj in enumerate(objs):
                obj["maps"] = None
                obj["loc"] = None
                torch.save(obj, os.path.join(cdir, f"{name}_s{r}.pt"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def provenance(cfg: dict) -> dict:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git = {}
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        if h.returncode == 0:
            git["commit"] = h.stdout.strip()
            d = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10)
            git["dirty"] = bool(d.stdout.strip())
    except Exception:
        pass
    hs = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(root, "probe", "*.py"))) + [cfg["_path"]]:
        with open(f, "rb") as fh:
            hs.update(os.path.basename(f).encode())
            hs.update(fh.read())
    import torchvision
    info = {"git": git or None, "source_sha256": hs.hexdigest(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "python": platform.python_version(), "platform": platform.platform(),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "model": f"torchvision.models.{cfg['model']['arch']} weights={cfg['model']['weights']}",
            "allow_tf32": cfg["model"]["allow_tf32"]}
    return info


def write_csvs(out: str, names: list, S: dict) -> None:
    import pandas as pd
    allrows = defaultdict(list)
    heats = {}
    for im in names:
        cdir = os.path.join(out, "cache", im)
        ip = os.path.join(cdir, "image.pt")
        if os.path.exists(ip):
            io = torch.load(ip, weights_only=False)
            allrows["images"].append(io["info"])
            heats.setdefault(im, {})["_x"] = io["x"]
            heats[im]["_mask"] = io["mask"]
        rp = os.path.join(cdir, "ref.pt")
        if os.path.exists(rp):
            ro = torch.load(rp, weights_only=False)
            for k, rs in ro["rows"].items():
                allrows[k] += rs
            heats.setdefault(im, {}).update(ro.get("heat", {}))
        for lp in sorted(glob.glob(os.path.join(cdir, "lime_g*.pt"))):
            lo = torch.load(lp, weights_only=False)
            allrows["lime"].append({"image": im, "grid": lo["grid"], "n_queries": lo["n_queries"], "r2_weighted": lo["r2_weighted"],
                                    "intercept": lo["intercept"], "beta_l1": float(lo["beta"].abs().sum()),
                                    "seconds": lo.get("seconds", float("nan"))})
            allrows["budget"].append({"image": im, "block": f"lime_g{lo['grid']}", "seed": -1, "part": "lime_prior",
                                      "grad_evals": 0, "fwd_evals": lo["n_queries"], "seconds_grad": 0.0,
                                      "seconds_metrics": lo.get("seconds", float("nan"))})
        for blk in blocks(S):
            sp = os.path.join(cdir, f"{blk['name']}_summary.pt")
            if not os.path.exists(sp):
                continue
            so = torch.load(sp, weights_only=False)
            for k, rs in so["rows"].items():
                allrows[k] += rs
            for m, hm in so.get("heat", {}).items():
                heats.setdefault(im, {})[m] = hm
            for r in range(blk["seeds"]):
                p = os.path.join(cdir, f"{blk['name']}_s{r}.pt")
                if os.path.exists(p):
                    o = torch.load(p, weights_only=False)
                    for k, rs in o["rows"].items():
                        allrows[k] += rs
    rename = {"per_seed": "sampling_per_seed", "per_image": "sampling_per_image", "coverage": "sampling_coverage",
              "covariance": "sampling_covariance", "cov_pairs": "sampling_covariance_pairs", "budget": "sampling_budget",
              "paths": "sampling_paths", "locality": "sampling_locality", "curves": "sampling_curves", "ref": "sampling_ref",
              "lime": "sampling_lime", "images": "sampling_images"}
    for k, rs in allrows.items():
        if rs:
            pd.DataFrame(rs).to_csv(os.path.join(out, f"{rename.get(k, 'sampling_' + k)}.csv"), index=False)
            print(f"[sampling] {rename.get(k, k)}.csv: {len(rs)} dòng")
    if heats:
        torch.save(heats, os.path.join(out, "sampling_heatmaps.pt"))


def image_indices(ctx: Ctx, S: dict) -> list:
    n = int(S["n_images"])
    idxs = ctx.sp["pilot"][:n]
    if len(idxs) < n:
        raise SystemExit(f"split pilot chỉ có {len(idxs)} ảnh < sampling.n_images = {n}")
    return idxs


def run(ctx: Ctx, estimate_only: bool = False) -> None:
    from .pilot import bench
    cfg = ctx.cfg
    S = settings(cfg)
    out = sampling_out(cfg)
    dgs = digests(cfg, S)
    idxs = image_indices(ctx, S)
    names = [f"seg{i:05d}" for i in idxs]
    b = bench(ctx)
    pf = bench_fwd(ctx)
    est = estimate(cfg, S, b["per_eval"], pf)
    pl = plan(S, est)
    print_estimate(S, est, pl, len(idxs))
    if estimate_only:
        return
    if not pl["primary_ok"]:
        msg = (f"[sampling] ước tính phần chính {pl['primary_minutes']:.1f} phút > giới hạn {pl['primary_limit']:.1f} phút "
               "(sampling.budget.minutes × slack). Giảm n_images / seeds / reference.K, hoặc nâng budget.minutes.")
        if pl["enforce"]:
            raise SystemExit(msg)
        print(msg + " (budget.enforce = false → vẫn chạy)")
    all_blk = {b_["name"]: b_ for b_ in blocks(S)}
    t_all = time.time()
    n_start, f_start = Counter.n, SM.FwdCounter.n
    timings = {}
    t0 = time.time()
    for j, idx in enumerate(idxs):
        print(f"  [sampling/primary] ảnh {j + 1}/{len(idxs)} seg{idx:05d}", flush=True)
        process_image(ctx, S, out, idx, [all_blk["primary"]], dgs, do_ref=True)
    timings["primary+ref"] = (time.time() - t0) / 60
    for name in pl["secondary_run"]:
        t0 = time.time()
        for j, idx in enumerate(idxs):
            print(f"  [sampling/{name}] ảnh {j + 1}/{len(idxs)} seg{idx:05d}", flush=True)
            process_image(ctx, S, out, idx, [all_blk[name]], dgs, do_ref=False)
        timings[name] = (time.time() - t0) / 60
    write_csvs(out, names, S)
    meta_p = os.path.join(out, "meta.json")
    meta = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "images": names, "indices": [int(i) for i in idxs],
            "split": "pilot", "debug": cfg["_debug"], "device": str(ctx.dev), "bench": {**b, "per_fwd": pf},
            "estimate": est, "plan": pl, "minutes_total_this_run": (time.time() - t_all) / 60, "minutes_by_block": timings,
            "grad_evals_this_run": Counter.n - n_start, "fwd_evals_this_run": SM.FwdCounter.n - f_start,
            "digests": dgs, "pilot_config_digest": C.digest(cfg), "global_seed": cfg["seed"],
            "seeds": {b_["name"]: list(range(b_["seeds"])) for b_ in blocks(S)},
            "settings": {k: v for k, v in S.items()}, "provenance": provenance(cfg),
            "blocks": blocks(S)}
    with open(meta_p, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    print(f"[sampling] xong: {(time.time() - t_all) / 60:.1f} phút, {Counter.n - n_start} grad eval, "
          f"{SM.FwdCounter.n - f_start} forward", flush=True)
