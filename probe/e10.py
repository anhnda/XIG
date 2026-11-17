"""
E10 — Random Path Family Ablation: hiệu quả của SPI đến từ MỨC exploration hay từ CẤU TRÚC xác suất của Dirichlet Process?

Mọi họ: γ_i(t) = x′_i + Z_i(t)(x_i − x′_i), lịch trình đơn điệu, Z(0) = 0, Z(1) = 1, E Z_i(t) = t, độc lập giữa phần tử.
  ig         Z_i(t) = t
  dp         SPI: số gia Dirichlet(α/N, …) trên lưới N (DP với base đều, α)
  smooth     Z_i(t) = t + δ s_i t(1−t), s_i = ±1 đều; δ = √(5/(α+1)) khớp D với DP (liên tục); đơn điệu khi δ ≤ 1
  lognormal  số gia w_ik = softmax_k(σ ε_ik), ε ~ N(0,1); σ hiệu chuẩn (Monte Carlo, không cần ảnh) sao cho D trên lưới = D của DP
Mức exploration D(P) = E ∫₀¹ (1/d) Σ_i (Z_i(t) − t)² dt; DP: 1/(6(α+1)); smooth: δ²/30. Trên lưới N nút, D_hat dùng quy tắc
hình thang trên các nút (DP: Var Z(t_k) = t_k(1−t_k)/(α+1) đúng tại nút; lognormal: hiệu chuẩn theo đúng D_hat này).

Cùng baseline (đen), lớp đích, canonical attribution, N, M, số đường K (hai tập rời nhau 2×K_half), số eval, cùng ảnh.
Đo: exploration (D_hat thực, nhảy lớn nhất, độ tập trung tiến độ, độ lệch hình học), attribution (IoU, insdel, ổn định giữa hai
tập đường), chi phí (exposure W, thời gian, sai số cầu phương M so với M_hi trên vài ảnh).
Xác nhận trên ảnh fresh chưa dùng (bỏ qua `fresh_offset` ảnh đầu dành cho OCS), tiền đăng ký ở lần chạy đầu.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import time
from collections import defaultdict

import torch

from . import config as C
from .e5 import geom_dist
from .mechanism import aw_stream, summ
from .metrics import energy_mask, insdel, iou, make_base, tie_perm
from .model import ProbFn, TargetFn
from .ocs import Map, sim
from .paths import cum_nodes, spi_increments
from .pilot import Ctx, sync
from .quad import Counter, fg

F64 = torch.float64
FAMS = ("ig", "dp", "smooth", "lognormal")


# ─────────────────────────────────────────────────────────────────────────────
# lịch trình
# ─────────────────────────────────────────────────────────────────────────────

def sched(family: str, N: int, shape, gen, alpha: float, delta: float, sigma: float) -> torch.Tensor:
    """-> Z (N+1, *shape) float64 trên gen.device."""
    dev = gen.device
    t = torch.linspace(0, 1, N + 1, device=dev, dtype=F64).view((-1,) + (1,) * len(shape))
    if family == "ig":
        return t.expand((N + 1,) + tuple(shape)).clone()
    if family == "dp":
        return cum_nodes(spi_increments(alpha, N, shape, gen))
    if family == "smooth":
        s = 1.0 - 2.0 * (torch.rand(tuple(shape), generator=gen, device=dev, dtype=F64) < 0.5).double()
        return (t + delta * s * t * (1 - t)).clamp(0, 1)
    if family == "lognormal":
        eps = torch.randn((N,) + tuple(shape), generator=gen, device=dev, dtype=F64)
        return cum_nodes(torch.softmax(sigma * eps, dim=0))
    raise ValueError(family)


def d_hat(Z: torch.Tensor) -> float:
    """Trung bình theo phần tử của ∫(Z − t)² dt, hình thang trên các nút."""
    N = Z.shape[0] - 1
    t = torch.linspace(0, 1, N + 1, device=Z.device, dtype=F64).view((-1,) + (1,) * (Z.dim() - 1))
    e = ((Z.double() - t) ** 2).flatten(1).mean(1)
    return float(torch.trapezoid(e, dx=1.0 / N))


def d_dp_grid(alpha: float, N: int) -> float:
    t = torch.linspace(0, 1, N + 1, dtype=F64)
    return float(torch.trapezoid(t * (1 - t) / (alpha + 1), dx=1.0 / N))


def explore_stats(Z: torch.Tensor) -> dict:
    dZ = (Z[1:] - Z[:-1]).double().flatten(1)                        # (N, d)
    N = Z.shape[0] - 1
    mid = Z[N // 2].double().flatten()
    return {"D_hat": d_hat(Z), "max_jump": float(dZ.max(0).values.mean()), "conc": float((dZ ** 2).sum(0).mean()),
            "eff_steps": float((1.0 / (dZ ** 2).sum(0).clamp_min(1e-300)).mean()),
            "z_mid_q10": float(torch.quantile(mid[:200000].float(), 0.1)), "z_mid_q90": float(torch.quantile(mid[:200000].float(), 0.9))}


def calibrate(cfg: dict) -> dict:
    """δ theo công thức; σ bằng chia đôi trên log σ với Monte Carlo (CPU, seed cố định, không dùng ảnh)."""
    c = cfg["e10"]
    alpha, N = float(c["alpha"]), int(c["N"])
    target = d_dp_grid(alpha, N)
    delta = math.sqrt(5.0 / (alpha + 1.0))
    d = int(c["calib"]["d"])
    g = torch.Generator().manual_seed(int(c["calib"]["seed"]))
    eps = torch.randn((N, d), generator=g, dtype=F64)

    def D_ln(sig):
        return d_hat(cum_nodes(torch.softmax(sig * eps, dim=0)))
    lo, hi = math.log(1e-3), math.log(50.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        if D_ln(math.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
    sigma = math.exp((lo + hi) / 2)
    gs = torch.Generator().manual_seed(int(c["calib"]["seed"]) + 1)
    s = 1.0 - 2.0 * (torch.rand(d, generator=gs, dtype=F64) < 0.5).double()
    t = torch.linspace(0, 1, N + 1, dtype=F64).view(-1, 1)
    Zs = t + delta * s * t * (1 - t)
    gd = torch.Generator().manual_seed(int(c["calib"]["seed"]) + 2)
    Zd = cum_nodes(spi_increments(alpha, N, (d,), gd))
    return {"alpha": alpha, "N": N, "D_target_dp_grid": target, "D_dp_continuous": 1 / (6 * (alpha + 1)), "delta": delta,
            "D_smooth_grid_mc": d_hat(Zs), "D_smooth_continuous": delta ** 2 / 30, "sigma": sigma,
            "D_lognormal_grid_mc": D_ln(sigma), "D_dp_grid_mc": d_hat(Zd), "calib_d": d}


# ─────────────────────────────────────────────────────────────────────────────
# một ảnh
# ─────────────────────────────────────────────────────────────────────────────

def e10_image(ctx: Ctx, j: int, idx: int, cal: dict) -> dict:
    cfg = ctx.cfg
    c = cfg["e10"]
    it, x, mask, cls = ctx.load(idx)
    fn, prob = TargetFn(ctx.bb, cls), ProbFn(ctx.bb, cls)
    x0 = make_base(x, "black")
    dx = x - x0
    m3 = mask.unsqueeze(0).expand_as(x[0])
    H, W = x.shape[-2:]
    perm = tie_perm(it.image, H * W, ctx.seed, ctx.dev)
    ic = cfg["metrics"]["insdel"]
    bases = {b: make_base(x, b, ic["blur_kernel"], ic["blur_sigma"]) for b in ic["bases"]}
    N, M, Kh = int(c["N"]), int(c["M"]), int(c["K_half"])
    shape = tuple(dx.shape[1:])
    rows = defaultdict(list)
    gate = j < int(c["gate"]["n_images"])

    def quality(A, full=True):
        A = A.float()
        q = {"iou": iou(A, mask, perm), "energy_mask": energy_mask(A, None, mask)}
        if full:
            for bn, base in bases.items():
                q[f"insdel_{bn}"] = insdel(prob, x, base, A, ic["n_steps"], ic["batch"])
        return q

    for fam in FAMS:
        t0, n0 = time.time(), Counter.n
        sums = [None, None]
        Ws = []
        n_paths = 1 if fam == "ig" else 2 * Kh
        for p in range(n_paths):
            Z = sched(fam, N, shape, ctx.gen("e10Z", fam, N, it.image, p), cal["alpha"], cal["delta"], cal["sigma"])
            X = x0 + Z.to(x.dtype) * dx
            ex = explore_stats(Z)
            if fam == "ig":                                            # tất định: hai bộ điểm cầu phương làm hai "tập"
                Aa, Wa = aw_stream(fn, X, M, ctx.gen("e10q", fam, "a", N, M, it.image), ctx.batch)
                Ab, _ = aw_stream(fn, X, M, ctx.gen("e10q", fam, "b", N, M, it.image), ctx.batch)
                sums = [Aa, Ab]
                A, Wm = Aa, Wa
                halfK = 1
            else:
                A, Wm = aw_stream(fn, X, M, ctx.gen("e10q", fam, N, M, it.image, p), ctx.batch)
                h = 0 if p < Kh else 1
                sums[h] = A.clone() if sums[h] is None else sums[h] + A
                halfK = Kh
            s = summ(A, Wm, m3)
            rows["paths"].append({"image": it.image, "family": fam, "p": p, **s, "within": s["W"] - s["L1"],
                                  "dist_geom": geom_dist(X), **ex, **quality(A, full=bool(c["per_path_insdel"]))})
            Ws.append(s["W"])
            if gate and p < int(c["gate"]["paths"]):
                Ah, _ = aw_stream(fn, X, int(c["gate"]["M_hi"]), ctx.gen("e10g", fam, N, it.image, p), ctx.batch)
                rows["gate"].append({"image": it.image, "family": fam, "p": p, "M": M, "M_hi": int(c["gate"]["M_hi"]),
                                     "rel_l1_M_vs_Mhi": float((A - Ah).abs().sum() / Ah.abs().sum()),
                                     "iou_M": iou(A.float(), mask, perm), "iou_Mhi": iou(Ah.float(), mask, perm),
                                     "max_jump": ex["max_jump"]})
                del Ah
            del Z, X
        sync()
        A1, A2 = sums[0] / halfK, sums[1] / halfK
        Af = (sums[0] + sums[1]) / (2 * halfK)
        st = sim(Map(A1), Map(A2), c["metrics"], perm, ctx.gen("e10sw", fam, it.image))
        q1, qf = quality(A1), quality(Af)
        rows["fam"].append({"image": it.image, "family": fam, "K_total": 2 * halfK if fam != "ig" else 1,
                            **{f"{k}": v for k, v in qf.items()}, **{f"half1_{k}": v for k, v in q1.items()},
                            **{f"stab_{k}": v for k, v in st.items()}, "W_mean": sum(Ws) / len(Ws),
                            "seconds": time.time() - t0, "evals": Counter.n - n0})
    rows["images"].append({"image": it.image, "index": it.index, "cls": cls, "mask_area": float(mask.float().mean())})
    return dict(rows)


# ─────────────────────────────────────────────────────────────────────────────
# chạy
# ─────────────────────────────────────────────────────────────────────────────

def e10_digest(cfg: dict) -> str:
    keys = ("seed", "model", "data", "paths", "metrics", "e10")
    return hashlib.sha256(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def e10_out(cfg: dict) -> str:
    d = C.out_dir(cfg, "e10_" + cfg["e10"]["split"])
    os.makedirs(d, exist_ok=True)
    return d


def images_for(ctx: Ctx, cfg: dict) -> list:
    c = cfg["e10"]
    if c["split"] == "fresh":
        off = int(c["fresh_offset"])
        return ctx.sp["fresh"][off:off + int(c["n_images"])]
    return ctx.sp[c["split"]][:int(c["n_images"])]


def estimate(cfg: dict, per_eval: float) -> dict:
    c = cfg["e10"]
    N, M, Kh = int(c["N"]), int(c["M"]), int(c["K_half"])
    ic = cfg["metrics"]["insdel"]
    ins = len(ic["bases"]) * 2 * (ic["n_steps"] + 1) / 3
    per_img = 2 * N * M + 3 * 2 * Kh * N * M + 4 * 2 * ins + (4 * 2 * Kh * ins if c["per_path_insdel"] else 0)
    gate = (1 + 3 * min(int(c["gate"]["paths"]), 2 * Kh)) * N * int(c["gate"]["M_hi"])
    t = int(c["n_images"]) * per_img + int(c["gate"]["n_images"]) * gate
    return {"e10_min": t * per_eval / 60, "evals_per_image": per_img}


def run(ctx: Ctx):
    import pandas as pd
    from .pilot import bench
    cfg = ctx.cfg
    c = cfg["e10"]
    out = e10_out(cfg)
    dg = e10_digest(cfg)
    cal = calibrate(cfg)
    print(f"[e10] hiệu chuẩn: D_DP(lưới) = {cal['D_target_dp_grid']:.5f}; δ = {cal['delta']:.4f} (D_smooth lưới {cal['D_smooth_grid_mc']:.5f}); "
          f"σ = {cal['sigma']:.4f} (D_lognormal lưới {cal['D_lognormal_grid_mc']:.5f})", flush=True)
    idxs = images_for(ctx, cfg)
    if len(idxs) < int(c["n_images"]):
        raise SystemExit(f"split {c['split']} không đủ ảnh")
    if c["split"] == "fresh":
        used = set(ctx.sp["pilot"]) | set(ctx.sp["confirm"]) | set(ctx.sp["other"])
        if used & set(idxs):
            raise SystemExit("ảnh fresh giao với ảnh đã dùng")
    names = [f"seg{i:05d}" for i in idxs]
    if c["split"] != "pilot":
        pj = os.path.join(out, "e10_prereg.json")
        if os.path.exists(pj) and not cfg["_debug"]:
            if json.load(open(pj))["digest"] != dg:
                raise SystemExit(f"thiết kế E10 đã đổi sau khi tiền đăng ký ({pj})")
        else:
            with open(pj, "w") as fh:
                json.dump({"digest": dg, "time": _dt.datetime.now().isoformat(timespec="seconds"), "images": names,
                           "calibration": cal, "e10": c}, fh, indent=1, default=str)
            print(f"[e10] đã tiền đăng ký: {pj}")
    b = bench(ctx)
    est = estimate(cfg, b["per_eval"])
    print(f"[e10] split {c['split']}, {len(idxs)} ảnh; {b['per_eval'] * 1e3:.2f} ms/eval; ước tính ≈ {est['e10_min']:.1f} phút", flush=True)
    t_all = time.time()
    cd = os.path.join(out, "cache")
    os.makedirs(cd, exist_ok=True)
    for j, idx in enumerate(idxs):
        image = f"seg{idx:05d}"
        cp = os.path.join(cd, f"{image}.pt")
        if os.path.exists(cp) and torch.load(cp, weights_only=False).get("digest") == dg:
            print(f"  [e10] {image}: đã có trong cache, bỏ qua")
            continue
        t0, n0 = time.time(), Counter.n
        print(f"  [e10] ảnh {j + 1}/{len(idxs)} {image}", flush=True)
        rows = e10_image(ctx, j, idx, cal)
        torch.save({"rows": rows, "digest": dg}, cp)
        print(f"  [e10] {image}: {Counter.n - n0} eval, {time.time() - t0:.0f}s; tổng {(time.time() - t_all) / 60:.1f} phút", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    allrows = defaultdict(list)
    for im in names:
        for k, rs in torch.load(os.path.join(cd, f"{im}.pt"), weights_only=False)["rows"].items():
            allrows[k] += rs
    for k, rs in allrows.items():
        if rs:
            pd.DataFrame(rs).to_csv(os.path.join(out, f"e10_{k}.csv"), index=False)
            print(f"[e10] e10_{k}.csv: {len(rs)} dòng")
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"images": names, "split": c["split"], "calibration": cal, "bench": b, "estimate": est,
                   "minutes": (time.time() - t_all) / 60, "digest": dg, "debug": cfg["_debug"],
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}, fh, indent=1, default=str)
