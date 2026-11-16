"""
Pilot cơ chế attribution (P0–P3). Hai stage:

conv   P0 — completeness và hội tụ theo N ∈ convergence.N cho mọi họ đường × mọi estimator
       (canon_left/mid/rand, idgi, idgi_raw, idgi_rand). Lưới N lồng nhau: IG/BlurIG đánh giá nút ở N_max rồi lấy
       con; SPI rút số gia ở N_max và cộng gộp cho lưới thô (cùng một đường DP); Guided IG chạy lại theo từng N.
main   P1 (exp_a ảnh) + P2 tín hiệu (exp_c ảnh, tập con) + P3 bẫy (exp_b ảnh, tập con) trong MỘT lượt mỗi ảnh:
       IG/SPI/BlurIG đánh giá mạng một lần; F+S, F+H áp giải tích lên chính các đánh giá đó (common random
       numbers tuyệt đối: cùng nút, cùng U_k). Guided IG chạy lại cho từng biến thể và được phân rã.

Kết quả từng ảnh lưu ở <out>/cache/<stage>/<image>.pt (kèm hash config) → chạy lại sẽ bỏ qua ảnh đã xong.
CSV ghép ở cuối mỗi stage.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from . import config as C
from .components import (Signal, Trap, direction, eta_actual, kappa_for_eta, predicted_attenuation,
                         scatter_region, square_region)
from .data import STD, ImageNetSeg, splits
from .metrics import cos_deb, energy_mask, energy_total, insdel, iou, make_base, spearman, tie_perm, topk_overlap
from .model import ProbFn, TargetFn, load_backbone
from .paths import blur_nodes, guided_nodes, spi_increments, spi_nodes_from, straight_nodes
from .quad import Counter, Evals, evaluate, fg, idgi_diagnostics, rules, shift, unshift


# ─────────────────────────────────────────────────────────────────────────────
# hạ tầng
# ─────────────────────────────────────────────────────────────────────────────

class Acc:
    """Trung bình theo đường (SPI) và phương sai của trung bình, theo phần tử, float64."""

    def __init__(self):
        self.n, self.s, self.q = 0, {}, {}

    def add(self, maps: dict):
        for k, A in maps.items():
            A = A.double()
            if k not in self.s:
                self.s[k], self.q[k] = torch.zeros_like(A), torch.zeros_like(A)
            self.s[k] += A
            self.q[k] += A * A
        self.n += 1

    def keys(self):
        return list(self.s)

    def mean(self, k):
        return (self.s[k] / self.n).float()

    def var(self, k):
        if self.n < 2:
            return None
        v = (self.q[k] - self.s[k] ** 2 / self.n) / (self.n - 1)
        return (v.clamp_min(0) / self.n).float()


def deb_norm(A, V=None) -> float:
    n2 = float((A.double() ** 2).sum()) - (float(V.double().sum()) if V is not None else 0.0)
    return math.sqrt(max(n2, 0.0))


def nrm(A) -> float:
    return float(A.double().norm())


@dataclass
class Variant:
    name: str
    kind: str          # signal | trap
    comp: object
    meta: dict


@dataclass
class PathResult:
    acc: Acc
    deltas: dict
    ext: dict
    D: float
    drops: dict
    x_start: torch.Tensor
    info: dict = field(default_factory=dict)


class Ctx:
    def __init__(self, cfg: dict, device=None):
        self.cfg = cfg
        dev = device or (cfg.get("local", {}) or {}).get("device", "cuda")
        if str(dev).startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("không thấy CUDA; nếu thật sự muốn chạy CPU: --device cpu")
        self.dev = torch.device(dev)
        self.out = C.out_dir(cfg, "pilot")
        os.makedirs(self.out, exist_ok=True)
        self.ds = ImageNetSeg(cfg["local"]["imagenet_mat"], cfg["data"]["size"])
        self.sp = splits(cfg, self.ds.n_total)
        self.bb = load_backbone(cfg, self.dev)
        self.batch = int(cfg["model"]["batch"])
        self.std = torch.tensor(STD, device=self.dev, dtype=torch.float64).view(3, 1, 1)
        self.tol = float(cfg["idgi_zero_tol"])
        self.seed = int(cfg["seed"])
        self.quads = [q for q in cfg["quadrature"] if q != "left"]
        self.primary = "canon_rand" if "rand" in self.quads else "canon_left"
        self.digest = C.digest(cfg)

    def load(self, idx: int):
        it = self.ds.get(idx)
        x, mask = it.x.to(self.dev), it.mask.to(self.dev)
        with torch.no_grad():
            cls = int(self.bb(x)[0].argmax())
        return it, x, mask, cls

    def gen(self, *parts):
        return C.generator(C.seed_of(self.seed, *parts), self.dev)

    def cache(self, stage: str, image: str) -> str:
        d = os.path.join(self.out, "cache", stage)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{image}.pt")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def det_eval(ctx: Ctx, name: str, pc: dict, fn, x, x0, key, N=None, quads=None):
    """IG / BlurIG / Guided IG. U_k chỉ phụ thuộc (ảnh, đường, N) → mọi biến thể dùng chung."""
    N = int(N or pc["steps"])
    quads = ctx.quads if quads is None else quads
    gen = ctx.gen("quad", name, N, *key)
    k = pc["kind"]
    if k == "straight":
        return evaluate(fn, straight_nodes(x, x0, N), ctx.batch, quads, gen), {}
    if k == "blur":
        return evaluate(fn, blur_nodes(x, N, float(pc["max_sigma"])), ctx.batch, quads, gen), {}
    if k == "guided":
        X, f, g, gap = guided_nodes(fn, x, x0, N, float(pc["fraction"]), float(pc["max_dist"]), ctx.batch)
        return evaluate(fn, X, ctx.batch, quads, gen, f_nodes=f, g_nodes=g), {"end_gap": gap}
    raise ValueError(k)


def spi_increments_for(ctx: Ctx, pc: dict, shape, N: int, *key):
    return spi_increments(float(pc["alpha"]), N, shape, ctx.gen("spi", pc["alpha"], N, *key))


def family(rule: str) -> str:
    return "canon" if rule.startswith("canon") else rule


# ─────────────────────────────────────────────────────────────────────────────
# P0 — hội tụ
# ─────────────────────────────────────────────────────────────────────────────

def conv_image(ctx: Ctx, idx: int) -> dict:
    cfg, cc = ctx.cfg, ctx.cfg["convergence"]
    it, x, mask, cls = ctx.load(idx)
    fn = TargetFn(ctx.bb, cls)
    x0 = make_base(x, "black")
    dx = x - x0
    Ns = sorted(int(n) for n in cc["N"])
    Nmax = Ns[-1]
    quads = [q for q in cc["quadrature"] if q != "left"]
    rows = []

    def emit(name, real, evs: dict):
        maps = {N: rules(ev, ctx.tol, ctx.std) for N, ev in evs.items()}
        # tham chiếu: N_max, canon_rand cho mọi canon_*; cùng quy tắc cho các biến thể IDGI
        ref_map = {"canon": maps[Nmax][0].get("canon_rand", maps[Nmax][0]["canon_left"])}
        for r in ("idgi", "idgi_raw", "idgi_rand"):
            if r in maps[Nmax][0]:
                ref_map[r] = maps[Nmax][0][r]
        for N, (m, info) in maps.items():
            D = info["D"]
            for r, A in m.items():
                ref = ref_map[family(r)]
                fam = info.get(r, {}) if r.startswith("idgi") else {}
                rows.append({"image": it.image, "path": name, "real": real, "N": N, "rule": r, "D": D,
                             "sumA": float(A.double().sum()), "resid": float(A.double().sum()) - D,
                             "resid_rel": (float(A.double().sum()) - D) / abs(D) if D else float("nan"),
                             "dist_ref": nrm(A - ref) / max(nrm(ref), 1e-30), "l2": nrm(A),
                             "is_ref": N == Nmax and (r == "canon_rand" or r.startswith("idgi")),
                             "n_dropped": fam.get("n_dropped", 0),
                             "dropped_dF_rel": fam.get("dropped_dF", 0.0) / abs(D) if D else float("nan")})

    for name, pc in cfg["paths"].items():
        t0, n0 = time.time(), Counter.n
        kind = pc["kind"]
        if kind in ("straight", "blur", "spi"):
            reals = int(cc["spi_realizations"]) if kind == "spi" else 1
            for real in range(reals):
                if kind == "straight":
                    X = straight_nodes(x, x0, Nmax)
                elif kind == "blur":
                    X = blur_nodes(x, Nmax, float(pc["max_sigma"]))
                else:
                    X = spi_nodes_from(spi_increments_for(ctx, pc, dx.shape[1:], Nmax, "conv", it.image, real), x0, dx)
                f, g = fg(fn, X, ctx.batch)
                evs = {}
                for N in Ns:
                    fac = Nmax // N
                    gen = ctx.gen("conv-quad", name, N, it.image, real)
                    evs[N] = evaluate(fn, X[::fac], ctx.batch, quads, gen, f_nodes=f[::fac], g_nodes=g[::fac][:N])
                emit(name, real, evs)
                del X, f, g, evs
        elif kind == "guided":
            evs = {N: det_eval(ctx, name, pc, fn, x, x0, ("conv", it.image), N=N, quads=quads)[0] for N in Ns}
            emit(name, 0, evs)
            del evs
        sync()
        print(f"    [conv] {it.image} {name}: {Counter.n - n0} eval, {time.time() - t0:.1f}s", flush=True)
    return {"conv": rows}


# ─────────────────────────────────────────────────────────────────────────────
# biến thể: tín hiệu S và bẫy H
# ─────────────────────────────────────────────────────────────────────────────

def make_signals(ctx: Ctx, it, x, x0, mask, D: float) -> tuple[list, list]:
    c = ctx.cfg["exp_c"]
    gen = torch.Generator().manual_seed(C.seed_of(ctx.seed, "signal", it.image))
    B0 = float(c["B_frac"]) * abs(D)
    specs = []
    for side in c["sizes"]:
        for loc in c["locations"]:
            specs.append(("square", int(side), loc, 1))
            if int(side) in [int(s) for s in c.get("neg_sign_for", [])]:
                specs.append(("square", int(side), loc, -1))
    for s in c.get("scatter_signs", []):
        specs.append(("scatter", int(c["scatter_pixels"]), "any", int(s)))
    mask_cpu, dx = mask.cpu(), x - x0
    out, skipped = [], []
    for kind, size, loc, sgn in specs:
        R = meta = None
        for _ in range(int(c.get("region_retries", 20))):
            if kind == "square":
                thr = float(c["inside_min"]) if loc == "inside" else float(c["outside_max"])
                R, meta = square_region(mask_cpu, size, loc == "inside", thr, gen, int(c["tries"]))
            else:
                R, meta = scatter_region(mask_cpu, size, gen)
            if R is None:
                break
            R = R.to(ctx.dev)
            den = float((dx * R).sum())
            if den / float(R.sum()) >= float(c["min_denominator"]):
                break
            R = None
        name = f"S_{kind}{size}_{loc}_{'pos' if sgn > 0 else 'neg'}"
        if R is None:
            skipped.append({"image": it.image, "variant": name, "reason": "không tìm được vùng thoả mask/mẫu số"})
            continue
        a = sgn * B0 / den
        n_el = int(R.sum())
        out.append(Variant(name, "signal", Signal(R, x0, a), {
            "shape": kind, "size": size, "loc": loc, "sign": sgn, "n_elements": n_el, "n_pixels": n_el // 3,
            "mask_frac": meta["mask_frac"], "a": a, "B": sgn * B0, "denominator": den}))
    return out, skipped


def make_traps(ctx: Ctx, it, ev_ig: Evals, x0, dx) -> list:
    b = ctx.cfg["exp_b"]
    N = ev_ig.N
    t = torch.arange(N, device=ctx.dev, dtype=torch.float64) / N
    specs = [(float(r), float(b["eta_ref"]), "high") for r in b["rho"]]
    specs += [(float(b["rho_ref"]), float(e), "high") for e in b["eta"]]
    if b.get("u_low_at_ref", False):
        specs.append((float(b["rho_ref"]), float(b["eta_ref"]), "low"))
    seen, out, us = set(), [], {}
    for rho, eta, uk in specs:
        if (rho, eta, uk) in seen:
            continue
        seen.add((rho, eta, uk))
        if uk not in us:
            us[uk] = direction(uk, dx, ctx.gen("u", uk, it.image), float(b["low_sigma"]))
        kap = kappa_for_eta(eta, ev_ig.g, t)
        tr = Trap(x0, dx, us[uk], kap, rho)
        out.append(Variant(f"H_rho{rho:g}_eta{eta:g}_{uk}", "trap", tr, {
            "rho": rho, "eta": eta, "u": uk, "kappa": kap, "eta_ig": eta_actual(tr, ev_ig.X, ev_ig.g)}))
    return out


def idgi_pred_toy(ev: Evals, sig: Signal) -> float:
    """Xấp xỉ kiểu Test D (bỏ tương tác ⟨g, R⟩ và khối lượng g² sẵn có trên R):
    Σ_R ΔA_idgi ≈ Σ_k a²n/(‖g_k‖² + a²n) · ΔF̃_k."""
    s = ev.g.double().flatten(1).pow(2).sum(1)
    a2n = sig.a ** 2 * sig.n_elements
    v = sig.value(ev.X).double()
    dFt = (ev.f[1:] - ev.f[:-1]) + (v[1:] - v[:-1])
    return float((a2n / (s + a2n) * dFt).sum())


def variant_extras(ev: Evals, evc: Evals, v: Variant, D: float) -> dict:
    if v.kind == "signal":
        return {"idgi_pred": idgi_pred_toy(ev, v.comp)}
    gh = (evc.g - ev.g).double().flatten(1).norm(dim=1)
    gf = ev.g.double().flatten(1).norm(dim=1)
    return {"eta_path": float(gh.pow(2).mean().sqrt() / gf.pow(2).mean().sqrt()),
            "H_nodes_max_rel": float((evc.f - ev.f).abs().max()) / abs(D) if D else float("nan")}


# ─────────────────────────────────────────────────────────────────────────────
# P1–P3 — một lượt mỗi ảnh
# ─────────────────────────────────────────────────────────────────────────────

class ImageState:
    def __init__(self, ctx, it, x, mask, cls):
        self.ctx, self.it, self.x, self.mask, self.cls = ctx, it, x, mask, cls
        self.fn = TargetFn(ctx.bb, cls)
        self.prob = ProbFn(ctx.bb, cls)
        self.x0 = make_base(x, "black")
        self.dx = x - self.x0
        self.perm = tie_perm(it.image, x.shape[-1] * x.shape[-2], ctx.seed, ctx.dev)
        ic = ctx.cfg["metrics"]["insdel"]
        self.ins_bases = {b: make_base(x, b, ic["blur_kernel"], ic["blur_sigma"]) for b in ic["bases"]}
        self.rows = defaultdict(list)
        self.maps = {}
        self.ig_ref = {}
        self.D = None


def diag_rows(S: ImageState, name: str, p: int, ev: Evals, with_steps: bool):
    d = idgi_diagnostics(ev, S.mask)
    a = d["dF"].abs()
    w = a / a.sum() if float(a.sum()) > 0 else torch.full_like(a, 1.0 / len(a))
    S.rows["idgi_diag"].append({
        "image": S.it.image, "path": name, "p": p, "P_M": d["P_M"], "area": float(S.mask.float().mean()),
        "q_sd_w": d["q_sd_w"], "entropy_w": float((w * d["entropy"]).sum()) / d["max_entropy"],
        "top1_w": float((w * d["top1"]).sum()), "dF_pos": d["dF_pos"], "dF_neg": d["dF_neg"]})
    if with_steps:
        N = len(d["q"])
        for k in range(N):
            S.rows["idgi_steps"].append({"image": S.it.image, "path": name, "k": k, "t": k / N, "q": float(d["q"][k]),
                                         "entropy": float(d["entropy"][k]) / d["max_entropy"],
                                         "top1": float(d["top1"][k]), "dF": float(d["dF"][k])})


def nonadaptive(S: ImageState, name: str, pc: dict, ev_iter, variants: list, steps_diag: bool) -> PathResult:
    ctx = S.ctx
    acc = Acc()
    deltas = {v.name: Acc() for v in variants}
    ext = defaultdict(lambda: defaultdict(list))
    Ds, drops, x_start = [], defaultdict(list), None
    is_spi = pc["kind"] == "spi"
    kg = int(pc.get("k_group", 8)) if is_spi else 0
    kr = [ctx.primary, "idgi"]
    group = {r: None for r in kr}
    for p, ev in enumerate(ev_iter):
        if x_start is None:
            x_start = ev.X[:1].clone()
        m, info = rules(ev, ctx.tol, ctx.std)
        acc.add(m)
        Ds.append(info["D"])
        for r in m:
            if r.startswith("idgi"):
                drops[r].append((info[r]["n_dropped"], info[r]["dropped_dF"]))
        diag_rows(S, name, p, ev, steps_diag and p == 0)
        if is_spi:
            for r in kr:
                A = m[r]
                S.rows["spi_k"].append(k_row(S, r, 1, p, A, None, info["D"]))
                group[r] = A.double() if group[r] is None else group[r] + A.double()
                if (p + 1) % kg == 0:
                    S.rows["spi_k"].append(k_row(S, r, kg, p // kg, (group[r] / kg).float(), None, info["D"]))
                    group[r] = None
        for v in variants:
            evc = shift(ev, v.comp)
            mc, _ = rules(evc, ctx.tol, ctx.std)
            deltas[v.name].add({r: mc[r] - m[r] for r in m})
            for k, val in variant_extras(ev, evc, v, info["D"]).items():
                ext[v.name][k].append(val)
            del evc, mc
        del ev, m
    ext = {vn: {k: sum(vals) / len(vals) for k, vals in d.items()} for vn, d in ext.items()}
    dr = {r: (sum(a for a, _ in l) / len(l), sum(b for _, b in l) / len(l)) for r, l in drops.items()}
    return PathResult(acc, deltas, ext, sum(Ds) / len(Ds), dr, x_start)


def k_row(S, rule, K, idx, A, V, D):
    return {"image": S.it.image, "rule": rule, "K": K, "idx": idx, "debiased": V is not None,
            "iou": iou(A, S.mask, S.perm), "energy_mask": energy_mask(A, V, S.mask),
            "l2": deb_norm(A, V), "l1": float(A.double().abs().sum()),
            "resid_rel": (float(A.double().sum()) - D) / abs(D) if D else float("nan")}


def guided(S: ImageState, name: str, pc: dict, variants: list, steps_diag: bool) -> PathResult:
    ctx = S.ctx
    key = (S.it.image,)
    ev, ex = det_eval(ctx, name, pc, S.fn, S.x, S.x0, key)
    base, info = rules(ev, ctx.tol, ctx.std)
    acc = Acc()
    acc.add(base)
    diag_rows(S, name, 0, ev, steps_diag)
    deltas, ext = {}, {}
    for v in variants:
        t0 = time.time()
        ev2, ex2 = det_eval(ctx, name, pc, S.fn.with_extra(v.comp), S.x, S.x0, key)
        new, _ = rules(ev2, ctx.tol, ctx.std)                       # A(F+c; γ′)
        evF = unshift(ev2, v.comp)
        newF, _ = rules(evF, ctx.tol, ctx.std)                      # A(F; γ′)
        fixed, _ = rules(shift(ev, v.comp), ctx.tol, ctx.std)       # A(F+c; γ)
        d = Acc()
        d.add({r: new[r] - base[r] for r in base})
        deltas[v.name] = d
        e = variant_extras(evF, ev2, v, info["D"])
        e["end_gap"] = ex2.get("end_gap", float("nan"))
        e["path_shift_rel"] = nrm(ev2.X - ev.X) / max(nrm(ev.X - S.x0), 1e-30)
        ext[v.name] = e
        R = v.comp.R[0].bool() if v.kind == "signal" else None
        for r in base:
            parts = {"tot": new[r] - base[r], "compnew": new[r] - newF[r], "pathF": newF[r] - base[r],
                     "fixed": fixed[r] - base[r], "pathFc": new[r] - fixed[r]}
            row = {"image": S.it.image, "variant": v.name, "kind": v.kind, "rule": r, **v.meta,
                   "end_gap": e["end_gap"], "path_shift_rel": e["path_shift_rel"], "seconds": time.time() - t0}
            for k, A in parts.items():
                row[f"{k}_norm"] = nrm(A)
                row[f"{k}_R"] = float(A.double()[R].sum()) if R is not None else float("nan")
            S.rows["guided_parts"].append(row)
        del ev2, evF, new, newF, fixed
    return PathResult(acc, deltas, ext, info["D"], {r: (info[r]["n_dropped"], info[r]["dropped_dF"])
                                                    for r in base if r.startswith("idgi")}, ev.X[:1].clone(), ex)


def emit(S: ImageState, name: str, pc: dict, res: PathResult, variants: list, do_insdel: bool):
    ctx = S.ctx
    is_spi = pc["kind"] == "spi"
    D = res.D
    for r in res.acc.keys():
        A = res.acc.mean(r)
        V = res.acc.var(r) if is_spi else None
        S.maps[(name, r)] = (A, V)
        sA = float(A.double().sum())
        l1 = float(A.double().abs().sum())
        row = {"image": S.it.image, "path": name, "rule": r, "D": D, "D_black": S.D, "sumA": sA,
               "resid": sA - D, "resid_rel": (sA - D) / abs(D) if D else float("nan"),
               "abs_resid_rel": abs(sA - D) / abs(D) if D else float("nan"),
               "iou": iou(A, S.mask, S.perm), "energy_mask": energy_mask(A, V, S.mask),
               "energy_total": energy_total(A, V), "l1": l1, "l2": deb_norm(A, V),
               "cancel_frac": 1.0 - abs(sA) / l1 if l1 > 0 else float("nan"),
               "n_paths": res.acc.n}
        if r in res.drops:
            nd, ddf = res.drops[r]
            row["n_dropped"], row["dropped_dF_rel"] = nd, ddf / abs(D) if D else float("nan")
        if do_insdel:
            ic = ctx.cfg["metrics"]["insdel"]
            for b, base in S.ins_bases.items():
                row[f"insdel_{b}"] = insdel(S.prob, S.x, base, A, ic["n_steps"], ic["batch"])
        S.rows["maps"].append(row)
        if is_spi and r in (ctx.primary, "idgi"):
            S.rows["spi_k"].append(k_row(S, r, res.acc.n, 0, A, None, D))     # thô, cùng định nghĩa với K=1, K=8
            S.rows["spi_k"].append(k_row(S, r, res.acc.n, 0, A, V, D))        # khử nhiễu MC

    for v in variants:
        dacc = res.deltas[v.name]
        e = res.ext.get(v.name, {})
        if v.kind == "trap" and name == "ig":
            S.ig_ref[v.name] = deb_norm(dacc.mean(ctx.primary), dacc.var(ctx.primary))
        for r in dacc.keys():
            dA = dacc.mean(r)
            Vd = dacc.var(r)
            A0 = res.acc.mean(r)
            V0 = res.acc.var(r) if is_spi else None
            base = {"image": S.it.image, "variant": v.name, "path": name, "rule": r, **v.meta}
            if v.kind == "signal":
                R = v.comp.R[0].bool()
                Astar = v.comp.expected(S.x, res.x_start)
                tot = float(Astar.double().sum())
                credit = float(dA.double()[R].sum())
                l1 = float(dA.double().abs().sum())
                en = dA.double() ** 2 - (Vd.double() if Vd is not None else 0.0)
                ok = abs(tot) >= 0.05 * abs(v.meta["B"])
                row = {**base, "B_true": tot, "B_true_ok": ok, "credit_R": credit,
                       "retention": credit / tot if ok else float("nan"),
                       "error": nrm(dA - Astar) / max(nrm(Astar), 1e-30),
                       "leak_l1": float(dA.double().abs()[~R].sum()) / l1 if l1 > 0 else float("nan"),
                       "leak_energy": float(en[~R].sum() / en.sum()) if float(en.sum()) > 0 else float("nan"),
                       "credit_R_total": float((A0 + dA).double()[R].sum()),
                       "sign_ok": (credit > 0) == (tot > 0)}
                if r == "idgi" and "idgi_pred" in e:
                    row["idgi_pred_toy"] = e["idgi_pred"] / tot if ok else float("nan")
                S.rows["signal"].append(row)
            else:
                dn = deb_norm(dA, Vd)
                an = deb_norm(A0, V0)
                ref = S.ig_ref.get(v.name, float("nan"))
                row = {**base, "dA_norm": dn, "rel_ig": dn / ref if ref and ref > 0 else float("nan"),
                       "rel_A": dn / an if an > 0 else float("nan"),
                       "eta_path": e.get("eta_path", float("nan")),
                       "H_nodes_max_rel": e.get("H_nodes_max_rel", float("nan"))}
                if pc["kind"] == "spi" and r.startswith("canon"):
                    pa = predicted_attenuation(1.0 / (float(pc["alpha"]) + 1.0), v.meta["rho"])
                    row["pred_att"] = pa
                    row["ratio_to_pred"] = row["rel_ig"] / pa if pa > 0 else float("nan")
                if pc["kind"] == "straight" and r.startswith("canon"):
                    tr = v.comp
                    expect = (tr.kappa / 2) * (tr.u.view_as(S.dx) * S.dx)[0]
                    row["ig_formula_err"] = nrm(dA - expect) / max(nrm(expect), 1e-30)
                S.rows["trap"].append(row)


def trap_output_rows(S: ImageState, traps: list, others: list, results: dict):
    b = S.ctx.cfg["exp_b"]
    n = int(b["noise_samples"])
    gen = S.ctx.gen("noise", S.it.image)
    eps = torch.randn((n,) + tuple(S.x.shape[1:]), generator=gen, device=S.ctx.dev, dtype=S.x.dtype)
    xs = S.x + float(b["noise_sigma"]) * eps
    with torch.no_grad():
        logits = S.ctx.bb(xs)
        lt = logits[:, S.cls].double()
        top = logits.argmax(1)
        for v in traps:
            H = v.comp.value(xs).double()
            lg = logits.double().clone()
            lg[:, S.cls] += H
            row = {"image": S.it.image, "variant": v.name, **v.meta,
                   "H_at_x": float(v.comp.value(S.x)[0]), "H_noise_mean_abs": float(H.abs().mean()),
                   "H_noise_max_abs": float(H.abs().max()), "H_noise_rel_D": float(H.abs().mean()) / abs(S.D),
                   "H_noise_rel_sdF": float(H.abs().mean()) / max(float(lt.std()), 1e-30),
                   "flip_rate": float((lg.argmax(1) != top).float().mean()),
                   "H_other_max_abs": max([abs(float(v.comp.value(xo)[0])) for xo in others], default=float("nan"))}
            for name, ext in results.items():
                row[f"H_nodes_max_rel_{name}"] = ext.get(v.name, {}).get("H_nodes_max_rel", float("nan"))
            S.rows["trap_output"].append(row)


def main_image(ctx: Ctx, j: int, idx: int, others: list) -> dict:
    cfg = ctx.cfg
    it, x, mask, cls = ctx.load(idx)
    S = ImageState(ctx, it, x, mask, cls)
    do_c = j < int(cfg["exp_c"]["n_images"])
    do_b = j < int(cfg["exp_b"]["n_images"])
    steps_diag = j < int(cfg["exp_a"].get("diag_step_images", 3))
    P = cfg["paths"]
    key = (it.image,)
    timing = {}

    t0, n0 = time.time(), Counter.n
    ev_ig, _ = det_eval(ctx, "ig", P["ig"], S.fn, x, S.x0, key)
    S.D = float(ev_ig.f[-1] - ev_ig.f[0])
    variants = []
    if do_c:
        sigs, skipped = make_signals(ctx, it, x, S.x0, mask, S.D)
        variants += sigs
        S.rows["skipped"] += skipped
    traps = make_traps(ctx, it, ev_ig, S.x0, S.dx) if do_b else []
    variants += traps

    results = {}
    order = ["ig"] + [n for n in P if n != "ig"]
    for name in order:
        pc = P[name]
        if name != "ig":
            t0, n0 = time.time(), Counter.n
        kind = pc["kind"]
        if kind == "straight":
            res = nonadaptive(S, name, pc, iter([ev_ig]), variants, steps_diag)
        elif kind == "blur":
            res = nonadaptive(S, name, pc, iter([det_eval(ctx, name, pc, S.fn, x, S.x0, key)[0]]), variants, steps_diag)
        elif kind == "spi":
            N, K = int(pc["steps"]), int(pc["paths"])

            def gen_paths():
                for p in range(K):
                    X = spi_nodes_from(spi_increments_for(ctx, pc, S.dx.shape[1:], N, it.image, p), S.x0, S.dx)
                    yield evaluate(S.fn, X, ctx.batch, ctx.quads, ctx.gen("spi-quad", N, it.image, p))
            res = nonadaptive(S, name, pc, gen_paths(), variants, steps_diag)
        elif kind == "guided":
            res = guided(S, name, pc, variants, steps_diag)
        else:
            raise ValueError(kind)
        emit(S, name, pc, res, variants, do_insdel=True)
        if name == "ig":
            del ev_ig
        results[name] = res.ext
        del res
        sync()
        timing[name] = {"seconds": time.time() - t0, "evals": Counter.n - n0}
        print(f"    {it.image} {name}: {timing[name]['evals']} eval, {timing[name]['seconds']:.1f}s", flush=True)

    if traps:
        trap_output_rows(S, traps, others, results)

    names = list(S.maps)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            (A, VA), (B, VB) = S.maps[names[a]], S.maps[names[b]]
            S.rows["pairs"].append({"image": it.image, "a_path": names[a][0], "a_rule": names[a][1],
                                    "b_path": names[b][0], "b_rule": names[b][1], "cos": cos_deb(A, VA, B, VB),
                                    "spearman": spearman(A, B),
                                    "top5": topk_overlap(A, B, cfg["metrics"]["topk_frac"], S.perm)})
    blur_D = next((r["D"] for r in S.rows["maps"] if r["path"] == "blur"), float("nan"))
    S.rows["images"].append({"image": it.image, "index": it.index, "cls": cls, "D_black": S.D, "D_blur": blur_D,
                             "mask_area": float(mask.float().mean()), "do_signal": do_c, "do_trap": do_b,
                             "timing": json.dumps(timing)})
    heat = None
    if j < int(cfg["exp_a"].get("heatmap_images", 2)):
        heat = {"x": it.x.half(), "mask": mask.cpu(),
                "maps": {f"{p}|{r}": A.abs().sum(0).half().cpu() for (p, r), (A, V) in S.maps.items()}}
    return {"rows": dict(S.rows), "heat": heat}


# ─────────────────────────────────────────────────────────────────────────────
# ước tính chi phí
# ─────────────────────────────────────────────────────────────────────────────

def bench(ctx: Ctx) -> dict:
    fn = TargetFn(ctx.bb, 0)
    xb = torch.randn(ctx.batch, 3, 224, 224, device=ctx.dev)
    fg(fn, xb, ctx.batch); sync()
    t = time.time()
    for _ in range(3):
        fg(fn, xb, ctx.batch)
    sync()
    per = (time.time() - t) / (3 * ctx.batch)
    x1 = xb[:1]
    fg(fn, x1, 1); sync()
    t = time.time()
    for _ in range(5):
        fg(fn, x1, 1)
    sync()
    t1 = (time.time() - t) / 5
    v = x1.abs().double().flatten()
    t = time.time()
    for _ in range(5):
        torch.quantile(v, 0.25, interpolation="lower")
    sync()
    tq = (time.time() - t) / 5
    Counter.n = 0
    return {"per_eval": per, "per_seq": t1 + 3 * tq}        # giả định ~3 vòng phân vị mỗi bước Guided


def estimate(cfg: dict, b: dict) -> dict:
    nq = len([q for q in cfg["quadrature"] if q != "left"])
    P = cfg["paths"]
    cc = cfg["convergence"]
    Ns = sorted(int(n) for n in cc["N"])
    nqc = len([q for q in cc["quadrature"] if q != "left"])
    conv_b, conv_s = 0, 0
    for pc in P.values():
        if pc["kind"] in ("straight", "blur"):
            conv_b += Ns[-1] + 1 + nqc * sum(Ns)
        elif pc["kind"] == "spi":
            conv_b += int(cc["spi_realizations"]) * (Ns[-1] + 1 + nqc * sum(Ns))
        else:
            conv_s += sum(Ns) + len(Ns)
            conv_b += nqc * sum(Ns)
    conv = cfg["convergence"]["n_images"] * (conv_b * b["per_eval"] + conv_s * b["per_seq"])

    def path_cost(pc):
        N = int(pc["steps"])
        if pc["kind"] == "spi":
            return int(pc["paths"]) * (N + 1 + nq * N), 0
        if pc["kind"] == "guided":
            return nq * N, N + 1
        return N + 1 + nq * N, 0

    per_b = sum(path_cost(pc)[0] for pc in P.values())
    per_s = sum(path_cost(pc)[1] for pc in P.values())
    ic = cfg["metrics"]["insdel"]
    n_rules = 3 + nq + (1 if "rand" in cfg["quadrature"] else 0)
    ins = len(P) * n_rules * len(ic["bases"]) * 2 * (ic["n_steps"] + 1) / 3.0
    gpc = next(pc for pc in P.values() if pc["kind"] == "guided")
    c = cfg["exp_c"]
    n_sig = len(c["sizes"]) * len(c["locations"]) + len(c.get("neg_sign_for", [])) * len(c["locations"]) \
        + len(c.get("scatter_signs", []))
    eb = cfg["exp_b"]
    n_trap = len(set([(r, eb["eta_ref"], "h") for r in eb["rho"]] + [(eb["rho_ref"], e, "h") for e in eb["eta"]])) \
        + (1 if eb.get("u_low_at_ref") else 0)
    gN = int(gpc["steps"])
    main = cfg["exp_a"]["n_images"] * ((per_b + ins) * b["per_eval"] + per_s * b["per_seq"])
    main += cfg["exp_c"]["n_images"] * n_sig * (nq * gN * b["per_eval"] + (gN + 1) * b["per_seq"])
    main += cfg["exp_b"]["n_images"] * n_trap * (nq * gN * b["per_eval"] + (gN + 1) * b["per_seq"])
    return {"conv_min": conv / 60, "main_min": main / 60, "n_signal_variants": n_sig, "n_trap_variants": n_trap,
            "evals_per_image_A": per_b, "seq_per_image_A": per_s}


# ─────────────────────────────────────────────────────────────────────────────
# chạy stage
# ─────────────────────────────────────────────────────────────────────────────

def _write_csvs(ctx: Ctx, stage: str, images: list):
    import pandas as pd
    allrows = defaultdict(list)
    heats = {}
    for im in images:
        obj = torch.load(ctx.cache(stage, im), weights_only=False)
        for k, rs in obj["rows"].items():
            allrows[k] += rs
        if obj.get("heat") is not None:
            heats[im] = obj["heat"]
    for k, rs in allrows.items():
        if not rs:                              # không ghi CSV rỗng
            continue
        pd.DataFrame(rs).to_csv(os.path.join(ctx.out, f"{k}.csv"), index=False)
        print(f"[{stage}] {k}.csv: {len(rs)} dòng")
    if heats:
        torch.save(heats, os.path.join(ctx.out, "heatmaps.pt"))


def run_stage(ctx: Ctx, stage: str):
    cfg = ctx.cfg
    b = bench(ctx)
    est = estimate(cfg, b)
    n = int(cfg["convergence" if stage == "conv" else "exp_a"]["n_images"])
    idxs = ctx.sp["pilot"][:n]
    print(f"[{stage}] {n} ảnh; đo được {b['per_eval'] * 1e3:.2f} ms/eval (lô), {b['per_seq'] * 1e3:.1f} ms/bước Guided")
    print(f"[{stage}] ước tính: conv ≈ {est['conv_min']:.1f} phút, main ≈ {est['main_min']:.1f} phút "
          f"(chưa tính chi phí phụ; Guided phụ thuộc số vòng phân vị)", flush=True)
    others = []
    if stage == "main" and int(cfg["exp_b"]["n_images"]) > 0:
        others = [ctx.ds.get(i).x.to(ctx.dev) for i in ctx.sp["other"][:int(cfg["exp_b"].get("n_other", 0))]]
    names, t_all = [], time.time()
    for j, idx in enumerate(idxs):
        image = f"seg{idx:05d}"
        names.append(image)
        path = ctx.cache(stage, image)
        if os.path.exists(path):
            obj = torch.load(path, weights_only=False)
            if obj.get("digest") == ctx.digest:
                print(f"  [{stage}] {image}: đã có trong cache, bỏ qua")
                continue
        t0, n0 = time.time(), Counter.n
        print(f"  [{stage}] ảnh {j + 1}/{n} {image}", flush=True)
        obj = conv_image(ctx, idx) if stage == "conv" else main_image(ctx, j, idx, others)
        if stage == "conv":
            obj = {"rows": obj}
        obj["digest"] = ctx.digest
        torch.save(obj, path)
        el = time.time() - t_all
        print(f"  [{stage}] {image}: {Counter.n - n0} eval, {time.time() - t0:.0f}s; đã chạy {el / 60:.1f} phút",
              flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_csvs(ctx, stage, names)
    meta_p = os.path.join(ctx.out, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta[stage] = {"images": names, "bench": b, "estimate": est, "minutes": (time.time() - t_all) / 60,
                   "digest": ctx.digest, "debug": cfg["_debug"], "device": str(ctx.dev),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "torch": torch.__version__}
    meta["config"] = {k: v for k, v in cfg.items() if not k.startswith("_") and k not in ("local", "debug")}
    with open(meta_p, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)