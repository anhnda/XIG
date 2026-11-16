"""
Thí nghiệm A (đường đi × quy tắc), B (bẫy gradient), C (giữ tín hiệu), sanity (khởi tạo lại theo tầng).

Mọi biến thể mô hình (F, F + H, F + S, mô hình khởi tạo lại) dùng CÙNG seed đường SPI (common random
numbers), nên hiệu A(F̃) − A(F) gần như không mang nhiễu MC; phần còn lại được khử bằng phương sai
của hiệu các sub-estimate ghép cặp.

Ghi chú quan trọng khi đọc: các phương pháp canonical (IG, SPI, BlurIG) TUYẾN TÍNH theo hàm, nên
A(F + H) − A(F) = A(H) không phụ thuộc ResNet, và với tín hiệu tuyến tính S thì A(S) đúng bằng
w·(x − x_start) trên R (độ giữ tín hiệu = 1 theo cấu tạo, sai khác chỉ do rời rạc hoá). Phần thông tin
thật của B và C nằm ở các quy tắc/đường phụ thuộc mô hình: Guided IG (cả hai quy tắc) và mọi biến thể IDGI.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time

import torch

from .config import digest, generator, out_dir, seed_of
from .data import ImageNetSeg, splits
from .integrate import RULES, Counter, fg
from .metrics import (cos_deb, energy_mask, energy_total, iou, make_base, quality, spearman, tie_perm,
                      topk_overlap)
from .model import ProbFn, TargetFn, load_backbone, randomized
from .paths import run_method
from .traps import Signal, Trap, direction, kappa_for, pick_patch, predicted_attenuation


# ─────────────────────────────────────────────────────────────────────────────
# ngữ cảnh chung
# ─────────────────────────────────────────────────────────────────────────────

class Ctx:
    def __init__(self, cfg: dict, split: str, device=None):
        self.cfg, self.split = cfg, split
        loc = cfg.get("local", {}) or {}
        self.dev = torch.device(device or loc.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.out = out_dir(cfg, split)
        os.makedirs(os.path.join(self.out, "cache"), exist_ok=True)
        self.ds = ImageNetSeg(loc.get("imagenet_mat"), cfg["data"]["size"])
        self.sp = splits(cfg, self.ds.n_total)
        self.bb = load_backbone(cfg, self.dev)
        self.batch = int(cfg["model"]["batch"])
        self.mkey = hashlib.sha256(json.dumps(cfg["methods"], sort_keys=True).encode()).hexdigest()[:10]
        with open(os.path.join(self.out, "run_info.json"), "w") as fh:
            json.dump({"config_sha256": digest(cfg), "debug": cfg["_debug"], "split": split,
                       "n_excluded": self.sp["n_excluded"], "images": self.sp[split]}, fh, indent=1)

    def images(self, n: int) -> list[int]:
        ids = self.sp[self.split]
        if n > len(ids):
            print(f"[ctx] chỉ có {len(ids)} ảnh trong split {self.split}, cần {n}")
        return ids[:n]

    def load(self, i: int) -> dict:
        it = self.ds.get(i)
        x = it.x.to(self.dev)
        x0 = make_base(x, self.cfg["baseline"])
        with torch.no_grad():
            cls = int(self.bb(x).argmax(1))
        return {"x": x, "x0": x0, "mask": it.mask.to(self.dev), "image": it.image, "index": it.index,
                "cls": cls, "fn": TargetFn(self.bb, cls), "prob": ProbFn(self.bb, cls),
                "perm": tie_perm(it.image, it.mask.numel(), self.cfg["seed"], self.dev)}

    # bảng kết quả --------------------------------------------------------------
    def append(self, table: str, row: dict) -> None:
        with open(os.path.join(self.out, f"{table}.jsonl"), "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")

    def done(self, table: str, key: str) -> set:
        p = os.path.join(self.out, f"{table}.jsonl")
        if not os.path.exists(p):
            return set()
        out = set()
        with open(p) as fh:
            for line in fh:
                try:
                    out.add(json.loads(line)[key])
                except Exception:
                    pass
        return out


# ─────────────────────────────────────────────────────────────────────────────
# tính mọi phương pháp × quy tắc cho một hàm mục tiêu
# ─────────────────────────────────────────────────────────────────────────────

def compute_all(ctx: Ctx, fn, s: dict) -> dict:
    """-> {"<method>/<rule>": {"A", "V", "subs"}, "_info": {...}}"""
    res, info = {}, {}
    for name, mc in ctx.cfg["methods"].items():
        o, meta = run_method(name, mc, fn, s["x"], s["x0"], ctx.batch, seed_key=(ctx.cfg["seed"], s["image"]))
        info[name] = {"f0": o["_f0"], "fN": o["_fN"], "start": meta["start"], "end_gap": o.get("_end_gap", 0.0)}
        for r in RULES:
            res[f"{name}/{r}"] = {"A": o[r], "V": o.get(r + "_var"), "subs": o.get(r + "_subs")}
    res["_info"] = info
    return res


def base_maps(ctx: Ctx, s: dict) -> dict:
    p = os.path.join(ctx.out, "cache", f"{s['image']}_{ctx.mkey}.pt")
    if os.path.exists(p):
        return torch.load(p, map_location=ctx.dev)
    m = compute_all(ctx, s["fn"], s)
    torch.save(m, p)
    return m


def delta(a: dict, b: dict):
    """Hiệu b − a và phương sai MC của hiệu (sub-estimate ghép cặp)."""
    d = (b["A"] - a["A"]).double()
    if a["subs"] is not None and b["subs"] is not None:
        S = torch.stack([(y.double() - x.double()) for x, y in zip(a["subs"], b["subs"])])
        return d, S.var(0, unbiased=True) / S.shape[0]
    return d, torch.zeros_like(d)


def norm_deb(d, V) -> float:
    return math.sqrt(max(0.0, float((d * d).sum() - V.sum())))


# ─────────────────────────────────────────────────────────────────────────────
# ước tính thời gian
# ─────────────────────────────────────────────────────────────────────────────

def evals_per_variant(cfg) -> tuple[int, int]:
    batched, seq = 0, 0
    for mc in cfg["methods"].values():
        if mc["kind"] == "spi":
            batched += int(mc["paths"]) * (int(mc["steps"]) + 1)
        elif mc["kind"] == "guided":
            seq += int(mc["steps"]) + 1
        else:
            batched += int(mc["steps"]) + 1
    return batched, seq


def eta(ctx: Ctx, s: dict, n_images: int, n_variants: int, label: str) -> None:
    x = s["x"]
    X = x.expand(ctx.batch, -1, -1, -1).contiguous()
    fg(s["fn"], X, ctx.batch)                         # khởi động
    sync = torch.cuda.synchronize if ctx.dev.type == "cuda" else (lambda: None)
    sync(); t0 = time.time(); fg(s["fn"], X, ctx.batch); sync()
    t_b = (time.time() - t0) / ctx.batch
    t0 = time.time()
    for _ in range(5):
        fg(s["fn"], x, 1)
    sync()
    t_s = (time.time() - t0) / 5
    b, q = evals_per_variant(ctx.cfg)
    sec = n_images * n_variants * (b * t_b + q * t_s)
    print(f"[{label}] {n_images} ảnh × {n_variants} biến thể × ({b} eval theo lô + {q} eval tuần tự) "
          f"≈ {sec / 60:.1f} phút attribution (đo: {1 / t_b:.0f} eval/s theo lô, {1 / t_s:.0f} eval/s tuần tự), "
          "chưa tính metric")


# ─────────────────────────────────────────────────────────────────────────────
# A: đường đi × quy tắc
# ─────────────────────────────────────────────────────────────────────────────

def run_a(ctx: Ctx) -> None:
    cfg = ctx.cfg
    ids = ctx.images(int(cfg["exp_a"]["n_images"]))
    done = ctx.done("a_quality", "image")
    first = True
    for n, i in enumerate(ids):
        s = ctx.load(i)
        if s["image"] in done:
            continue
        if first:
            eta(ctx, s, len(ids) - len(done), 1, "A")
            first = False
        t0 = time.time()
        m = base_maps(ctx, s)
        keys = [k for k in m if not k.startswith("_")]
        for k in keys:
            meth = k.split("/")[0]
            inf = m["_info"][meth]
            row = {"image": s["image"], "method": k, "D": inf["fN"] - inf["f0"], "end_gap": inf["end_gap"]}
            row.update(quality(m[k]["A"], m[k]["V"], s["mask"], s["perm"], s["prob"], s["x"], cfg["metrics"],
                               seed=7919 + s["index"], full=True))
            row["residual"] = (row["D"] - row["sumA"]) / row["D"] if row["D"] else float("nan")
            ctx.append("a_quality", row)
        for k1, k2 in itertools.combinations(keys, 2):
            a, b = m[k1], m[k2]
            ctx.append("a_pairs", {"image": s["image"], "a": k1, "b": k2,
                                   "cos_deb": cos_deb(a["A"], a["V"], b["A"], b["V"]),
                                   "spearman": spearman(a["A"], b["A"]),
                                   "topk": topk_overlap(a["A"], b["A"], cfg["metrics"]["topk_frac"], s["perm"])})
        print(f"  [A {n + 1}/{len(ids)}] {s['image']} ({time.time() - t0:.0f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# B: bẫy gradient
# ─────────────────────────────────────────────────────────────────────────────

def _others(ctx: Ctx):
    xs = [ctx.ds.get(i).x for i in ctx.sp["other"]]
    return torch.cat(xs).to(ctx.dev) if xs else None


def run_b(ctx: Ctx) -> None:
    cfg, bc = ctx.cfg, ctx.cfg["exp_b"]
    ids = ctx.images(int(bc["n_images"]))
    grid = list(itertools.product(bc["shapes"], bc["u"], [float(r) for r in bc["rho"]], [float(t) for t in bc["tau"]]))
    done = ctx.done("b_traps", "key")
    others = _others(ctx)
    first = True
    for n, i in enumerate(ids):
        s = ctx.load(i)
        if first:
            eta(ctx, s, len(ids), len(grid) + 1, "B")
            first = False
        base = base_maps(ctx, s)
        x, x0 = s["x"], s["x0"]
        dx = x - x0
        A_ig = base["ig/canonical"]["A"]
        D = base["_info"]["ig"]["fN"] - base["_info"]["ig"]["f0"]
        t0 = time.time()
        for shape, uk, rho, tau in grid:
            key = f"{s['image']}|{shape}|{uk}|{rho}|{tau}"
            if key in done:
                continue
            gen = generator(seed_of(cfg["seed"], s["image"], "trap", shape, uk), ctx.dev)
            q = direction("low", x, [dx], gen, bc["low_sigma"]) if shape == "bent" else None
            u = direction(uk, x, [dx] + ([q] if q is not None else []), gen, bc["low_sigma"])
            kap = kappa_for(tau, float(A_ig.norm()), u, dx)
            trap = Trap(x0, dx, u, kap, rho, bc["bend"] if shape == "bent" else 0.0, q)
            with torch.no_grad():
                g2 = generator(seed_of(cfg["seed"], s["image"], "noise"), ctx.dev)
                noisy = x + bc["noise_sigma"] * torch.randn((5,) + tuple(x.shape[1:]), generator=g2, device=ctx.dev)
                h_x = float(trap(x).abs())
                h_noise = float(trap(noisy).abs().max()) / abs(D)
                h_other = float(trap(others).abs().max()) / abs(D) if others is not None else float("nan")
            mT = compute_all(ctx, s["fn"].with_extra(trap), s)
            expect_ig = (kap / 2) * u[0] * dx[0]
            for k in [k for k in mT if not k.startswith("_")]:
                meth, rule = k.split("/")
                d, V = delta(base[k], mT[k])
                ref = math.sqrt(max(energy_total(base[k]["A"], base[k]["V"]), 1e-30))
                row = {"key": key, "image": s["image"], "shape": shape, "u": uk, "rho": rho, "tau": tau,
                       "method": k, "kappa": kap, "R": norm_deb(d, V) / ref,
                       "R_vs_ig_change": norm_deb(d, V) / float(expect_ig.norm()),
                       "dE": energy_mask(mT[k]["A"], mT[k]["V"], s["mask"]) - energy_mask(base[k]["A"], base[k]["V"], s["mask"]),
                       "dIoU": iou(mT[k]["A"], s["mask"], s["perm"]) - iou(base[k]["A"], s["mask"], s["perm"]),
                       "H_at_x": h_x, "H_noise_rel": h_noise, "H_other_rel": h_other}
                mc = cfg["methods"][meth]
                if mc["kind"] == "spi" and shape == "line":
                    row["pred_attenuation"] = predicted_attenuation(1.0 / (float(mc["alpha"]) + 1.0), rho)
                if k == "ig/canonical" and shape == "line":
                    row["ig_formula_err"] = float((d.float() - expect_ig).norm() / expect_ig.norm())
                ctx.append("b_traps", row)
        print(f"  [B {n + 1}/{len(ids)}] {s['image']} ({time.time() - t0:.0f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# C: giữ tín hiệu đã biết
# ─────────────────────────────────────────────────────────────────────────────

def run_c(ctx: Ctx) -> None:
    cfg, cc = ctx.cfg, ctx.cfg["exp_c"]
    ids = ctx.images(int(cc["n_images"]))
    done = ctx.done("c_signal", "key")
    first = True
    for n, i in enumerate(ids):
        s = ctx.load(i)
        if first:
            eta(ctx, s, len(ids), 5, "C")
            first = False
        base = base_maps(ctx, s)
        x, x0 = s["x"], s["x0"]
        D = base["_info"]["ig"]["fN"] - base["_info"]["ig"]["f0"]
        gen = torch.Generator().manual_seed(seed_of(cfg["seed"], s["image"], "patch"))
        t0 = time.time()
        for loc in ("inside", "outside"):
            thr = cc["inside_min"] if loc == "inside" else cc["outside_max"]
            R, where = pick_patch(s["mask"].cpu(), int(cc["patch"]), loc == "inside", thr, gen, int(cc["tries"]))
            if R is None:
                print(f"    {s['image']}: không tìm được ô {loc}")
                continue
            R = R.to(ctx.dev)
            sdx = float(((x - x0) * R).sum())
            if abs(sdx) < 1e-6:
                continue
            for sign in (+1, -1):
                key = f"{s['image']}|{loc}|{sign}"
                if key in done:
                    continue
                sig = Signal(R, sign * float(cc["gamma"]) * abs(D) / sdx)
                mS = compute_all(ctx, s["fn"].with_extra(sig), s)
                Rm = R[0].expand_as(x[0]).bool()
                for k in [k for k in mS if not k.startswith("_")]:
                    meth = k.split("/")[0]
                    d, V = delta(base[k], mS[k])
                    star = sig.expected(x, base["_info"][meth]["start"]).double()
                    tot = norm_deb(d, V)
                    out_d, out_V = d * (~Rm), V * (~Rm)
                    ctx.append("c_signal", {
                        "key": key, "image": s["image"], "loc": loc, "sign": sign, "patch": where, "method": k,
                        "retention": float((d * star).sum() / (star * star).sum()),
                        "err": norm_deb(d - star, V) / float(star.norm()),
                        "leak": norm_deb(out_d, out_V) / tot if tot > 0 else float("nan"),
                        "sign_ok": bool((float(d[Rm].sum()) > 0) == (sign > 0))})
        print(f"  [C {n + 1}/{len(ids)}] {s['image']} ({time.time() - t0:.0f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# sanity: khởi tạo lại theo tầng
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity(ctx: Ctx) -> None:
    cfg, sc = ctx.cfg, ctx.cfg["sanity"]
    ids = ctx.images(int(sc["n_images"]))
    done = ctx.done("sanity", "key")
    cascade = list(sc["cascade"])
    first = True
    for n, i in enumerate(ids):
        s = ctx.load(i)
        if first:
            eta(ctx, s, len(ids), len(cascade) + 1, "sanity")
            first = False
        base = base_maps(ctx, s)
        t0 = time.time()
        for L in range(1, len(cascade) + 1):
            key = f"{s['image']}|{L}"
            if key in done:
                continue
            bbR = randomized(ctx.bb, cascade[:L], seed_of(cfg["seed"], "rand", L))
            mR = compute_all(ctx, TargetFn(bbR, s["cls"]), s)
            for k in [k for k in mR if not k.startswith("_")]:
                a, b = base[k], mR[k]
                ctx.append("sanity", {"key": key, "image": s["image"], "layers": "+".join(cascade[:L]),
                                      "depth": L, "method": k, "spearman": spearman(a["A"], b["A"]),
                                      "cos_deb": cos_deb(a["A"], a["V"], b["A"], b["V"]),
                                      "topk": topk_overlap(a["A"], b["A"], cfg["metrics"]["topk_frac"], s["perm"])})
            del bbR
        print(f"  [sanity {n + 1}/{len(ids)}] {s['image']} ({time.time() - t0:.0f}s)")
