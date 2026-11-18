"""
E12 — chỉ số đo được từ hình học đường và phản ứng mô hình (không cần ground truth) có dự đoán khi nào exploration quá mạnh
bắt đầu gây hại không?

Họ đường: DP (SPI) α ∈ alphas, K đường mỗi α, canonical (μ đều, cầu phương phân tầng M điểm/đoạn), KHÔNG LIG.
Mỗi realization được rút ở lưới N_fine = 2N rồi cộng gộp xuống N (đường dùng để giải thích); lưới mịn chỉ dùng để kiểm độ nhạy
của R_nl theo độ chia (trên các ảnh gate).

Trên từng đường:
  S        độ bất đồng bộ: dist_geom (khoảng cách trung bình của nút tới đường chéo / ‖Δx‖), D_hat = ∫(Z − t)², max_jump
  R_nl     Σ_k |F(m_k) − (F(z_k) + F(z_{k+1}))/2| / (Σ_k |F(z_{k+1}) − F(z_k)| + ε), m_k trung điểm đoạn — bằng 0 với mọi F affine,
           dù bước tiến lớn đến đâu (tách "bước lớn" khỏi "mô hình phi tuyến trong bước")
  jump     max_k |ΔF_k| / Σ_k |ΔF_k| (để đối chiếu: output nhảy lớn KHÔNG đồng nghĩa phi tuyến)
  chất lượng K = 1 (IoU, energy, insdel blur/đen) và W
Ensemble K đường mỗi α: chất lượng. Cổng số học (n_gate ảnh, K_gate đường mỗi α): cùng đường, M lồng nhau {4,8,16,32}
(chọn ngẫu nhiên một điểm trong mỗi nhóm tầng của M_hi): sai khác L1 và chất lượng theo M; R_nl ở lưới N_fine.
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
from .e10 import explore_stats
from .e5 import geom_dist
from .mechanism import aw_stream, nested_pick, summ
from .metrics import energy_mask, insdel, iou, make_base, tie_perm
from .model import ProbFn, TargetFn
from .paths import coarsen, cum_nodes, spi_increments
from .pilot import Ctx, sync
from .quad import Counter, fg

F64 = torch.float64


@torch.no_grad()
def forward_only(fn, X: torch.Tensor, batch: int) -> torch.Tensor:
    out = []
    for i in range(0, X.shape[0], batch):
        out.append(fn(X[i:i + batch]).double())
    return torch.cat(out)


def nonlinearity(fn, X: torch.Tensor, batch: int, eps: float = 1e-12) -> dict:
    """R_nl và các đại lượng đối chiếu trên đường gấp khúc X (N+1 nút)."""
    Fn = forward_only(fn, X, batch)
    Fm = forward_only(fn, (X[1:] + X[:-1]) / 2, batch)
    r = (Fm - (Fn[1:] + Fn[:-1]) / 2).abs()
    dF = (Fn[1:] - Fn[:-1]).abs()
    D = float(Fn[-1] - Fn[0])
    return {"R_nl": float(r.sum() / (dF.sum() + eps)), "R_nl_max": float(r.max()) / (abs(D) + eps),
            "jump": float(dF.max() / (dF.sum() + eps)), "tv_over_D": float(dF.sum()) / (abs(D) + eps), "D_nodes": D}


def nested_stream(fn, X: torch.Tensor, Mhi: int, Ms, gen, gen_pick, batch: int, max_points: int = 512) -> dict:
    """Cầu phương phân tầng ở M_hi; các M nhỏ hơn lồng bên trong (một điểm ngẫu nhiên trong mỗi nhóm tầng). -> {M: A}."""
    N = X.shape[0] - 1
    U = torch.rand(N, Mhi, generator=gen, device=gen.device, dtype=F64).to(X.device)
    s = (torch.arange(Mhi, device=X.device, dtype=F64).view(1, Mhi) + U) / Mhi
    picks = {M: nested_pick(N, Mhi, M, gen_pick, X.device) for M in Ms if M != Mhi}
    dX = X[1:] - X[:-1]
    shp = (-1, Mhi) + (1,) * (X.dim() - 1)
    A = {M: torch.zeros(tuple(X.shape[1:]), dtype=F64, device=X.device) for M in Ms}
    ck = max(1, max_points // Mhi)
    for k0 in range(0, N, ck):
        k1 = min(N, k0 + ck)
        P = X[k0:k1].unsqueeze(1) + s[k0:k1].view(shp).to(X.dtype) * dX[k0:k1].unsqueeze(1)
        _, G = fg(fn, P.reshape((-1,) + tuple(X.shape[1:])), batch)
        G = G.view((k1 - k0, Mhi) + tuple(X.shape[1:])).double()
        d = dX[k0:k1].double()
        for M in Ms:
            if M == Mhi:
                A[M] += (G.mean(1) * d).sum(0)
            else:
                ix = picks[M][k0:k1]
                g = torch.stack([G[j, ix[j]] for j in range(k1 - k0)])
                A[M] += (g.mean(1) * d).sum(0)
        del P, G
    return A


def e12_image(ctx: Ctx, j: int, idx: int, sigmas: dict) -> dict:
    cfg = ctx.cfg
    c = cfg["e12"]
    it, x, mask, cls = ctx.load(idx)
    fn, prob = TargetFn(ctx.bb, cls), ProbFn(ctx.bb, cls)
    x0 = make_base(x, "black")
    dx = x - x0
    m3 = mask.unsqueeze(0).expand_as(x[0])
    H, W = x.shape[-2:]
    perm = tie_perm(it.image, H * W, ctx.seed, ctx.dev)
    ic = cfg["metrics"]["insdel"]
    bases = {b: make_base(x, b, ic["blur_kernel"], ic["blur_sigma"]) for b in c["insdel_bases"]}
    N, M, K = int(c["N"]), int(c["M"]), int(c["K"])
    gate = j < int(c["gate"]["n_images"])
    Kg, Mhi, Ms = int(c["gate"]["K"]), int(c["gate"]["M_hi"]), [int(m) for m in c["gate"]["M"]]
    shape = tuple(dx.shape[1:])
    rows = defaultdict(list)

    def quality(A, full=True):
        A = A.float()
        q = {"iou": iou(A, mask, perm), "energy_mask": energy_mask(A, None, mask)}
        if full:
            for bn, base in bases.items():
                q[f"insdel_{bn}"] = insdel(prob, x, base, A, int(c["insdel_steps"]), ic["batch"])
        return q

    for fam in c["families"]:
        for lev in c["levels"][fam]:
            t0, n0 = time.time(), Counter.n
            lev = float(lev)
            Asum, gsum = None, {m: None for m in Ms}
            for p in range(K):
                g = ctx.gen("e12Z", fam, lev, 2 * N, it.image, p)
                if fam == "dp":
                    delta = spi_increments(lev, 2 * N, shape, g)
                else:                                                  # lognormal: lev = α tương đương (cùng D), σ hiệu chuẩn sẵn
                    sig = sigmas[lev]
                    delta = torch.softmax(sig * torch.randn((2 * N,) + shape, generator=g, device=g.device, dtype=F64), 0)
                Z = cum_nodes(coarsen(delta, 2))
                X = x0 + Z.to(x.dtype) * dx
                A, Wm = aw_stream(fn, X, M, ctx.gen("e12q", fam, lev, N, M, it.image, p), ctx.batch)
                s = summ(A, Wm, m3)
                ex = explore_stats(Z)
                nl = nonlinearity(fn, X, ctx.batch)
                row = {"image": it.image, "family": fam, "level": lev, "p": p, **s, "dist_geom": geom_dist(X),
                       "D_hat": ex["D_hat"], "max_jump": ex["max_jump"], **nl, **quality(A, full=bool(c["per_path_insdel"]))}
                if gate and p < Kg:
                    Zf = cum_nodes(delta)
                    nlf = nonlinearity(fn, x0 + Zf.to(x.dtype) * dx, ctx.batch)
                    row.update({"R_nl_fine": nlf["R_nl"], "jump_fine": nlf["jump"]})
                    An = nested_stream(fn, X, Mhi, Ms, ctx.gen("e12g", fam, lev, N, Mhi, it.image, p),
                                       ctx.gen("e12gp", fam, lev, it.image, p), ctx.batch)
                    for m_, Am in An.items():
                        rows["gate"].append({"image": it.image, "family": fam, "level": lev, "p": p, "M": m_,
                                             "rel_l1_vs_Mhi": float((Am - An[Mhi]).abs().sum() / An[Mhi].abs().sum()),
                                             **quality(Am, full=False)})
                        gsum[m_] = Am.clone() if gsum[m_] is None else gsum[m_] + Am
                    del Zf, An
                rows["paths"].append(row)
                Asum = A.clone() if Asum is None else Asum + A
                del delta, Z, X
            sync()
            rows["ens"].append({"image": it.image, "family": fam, "level": lev, "K": K, "M": M, **quality(Asum / K),
                                "seconds": time.time() - t0, "evals": Counter.n - n0})
            if gate:
                for m_, S_ in gsum.items():
                    rows["gate_ens"].append({"image": it.image, "family": fam, "level": lev, "K": Kg, "M": m_, **quality(S_ / Kg)})
    rows["images"].append({"image": it.image, "index": it.index, "cls": cls, "mask_area": float(mask.float().mean()), "gate": gate})
    return dict(rows)


# ─────────────────────────────────────────────────────────────────────────────
# chạy
# ─────────────────────────────────────────────────────────────────────────────

def e12_digest(cfg: dict) -> str:
    keys = ("seed", "model", "data", "paths", "metrics", "e12")
    return hashlib.sha256(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def e12_out(cfg: dict) -> str:
    d = C.out_dir(cfg, "e12_" + cfg["e12"]["split"])
    os.makedirs(d, exist_ok=True)
    return d


def estimate(cfg: dict, per_eval: float) -> dict:
    c = cfg["e12"]
    N, M, K = int(c["N"]), int(c["M"]), int(c["K"])
    n_lev = sum(len(v) for f, v in c["levels"].items() if f in c["families"])
    ic = cfg["metrics"]["insdel"]
    ins = len(c["insdel_bases"]) * 2 * (int(c["insdel_steps"]) + 1) / 3
    per = n_lev * (K * (N * M + (2 * N + 1) / 3 + (ins if c["per_path_insdel"] else 0)) + ins)
    g = c["gate"]
    gate = n_lev * int(g["K"]) * (N * int(g["M_hi"]) + (4 * N + 1) / 3) + n_lev * len(g["M"]) * ins
    t = int(c["n_images"]) * per + int(g["n_images"]) * gate
    return {"e12_min": t * per_eval / 60, "evals_per_image": per, "evals_gate_image": gate}


def run(ctx: Ctx):
    import pandas as pd
    from .pilot import bench
    cfg = ctx.cfg
    c = cfg["e12"]
    out = e12_out(cfg)
    dg = e12_digest(cfg)
    if c["split"] == "fresh":
        off = int(c["fresh_offset"])
        idxs = ctx.sp["fresh"][off:off + int(c["n_images"])]
        used = set(ctx.sp["pilot"]) | set(ctx.sp["confirm"]) | set(ctx.sp["other"])
        if used & set(idxs):
            raise SystemExit("ảnh fresh giao với ảnh đã dùng")
    else:
        idxs = ctx.sp[c["split"]][:int(c["n_images"])]
    names = [f"seg{i:05d}" for i in idxs]
    if c["split"] != "pilot":
        pj = os.path.join(out, "e12_prereg.json")
        if os.path.exists(pj) and not cfg["_debug"]:
            if json.load(open(pj))["digest"] != dg:
                raise SystemExit(f"thiết kế E12 đã đổi sau khi tiền đăng ký ({pj})")
        else:
            with open(pj, "w") as fh:
                json.dump({"digest": dg, "time": _dt.datetime.now().isoformat(timespec="seconds"), "images": names, "e12": c},
                          fh, indent=1, default=str)
            print(f"[e12] đã tiền đăng ký: {pj}")
    sigmas = {}
    if "lognormal" in c["families"]:
        from .e10 import calibrate
        for a in c["levels"]["lognormal"]:
            sigmas[float(a)] = calibrate({"e10": {"alpha": float(a), "N": 2 * int(c["N"]), "calib": c["calib"]}})["sigma"]
        print(f"[e12] σ lognormal khớp D của DP (lưới 2N): {sigmas}", flush=True)
    b = bench(ctx)
    est = estimate(cfg, b["per_eval"])
    print(f"[e12] split {c['split']}, {len(idxs)} ảnh; {b['per_eval'] * 1e3:.2f} ms/eval; ước tính ≈ {est['e12_min']:.1f} phút", flush=True)
    t_all = time.time()
    cd = os.path.join(out, "cache")
    os.makedirs(cd, exist_ok=True)
    for j, idx in enumerate(idxs):
        image = f"seg{idx:05d}"
        cp = os.path.join(cd, f"{image}.pt")
        if os.path.exists(cp) and torch.load(cp, weights_only=False).get("digest") == dg:
            print(f"  [e12] {image}: đã có trong cache, bỏ qua")
            continue
        t0, n0 = time.time(), Counter.n
        print(f"  [e12] ảnh {j + 1}/{len(idxs)} {image}", flush=True)
        rows = e12_image(ctx, j, idx, sigmas)
        torch.save({"rows": rows, "digest": dg}, cp)
        print(f"  [e12] {image}: {Counter.n - n0} eval, {time.time() - t0:.0f}s; tổng {(time.time() - t_all) / 60:.1f} phút", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    allrows = defaultdict(list)
    for im in names:
        for k, rs in torch.load(os.path.join(cd, f"{im}.pt"), weights_only=False)["rows"].items():
            allrows[k] += rs
    for k, rs in allrows.items():
        if rs:
            pd.DataFrame(rs).to_csv(os.path.join(out, f"e12_{k}.csv"), index=False)
            print(f"[e12] e12_{k}.csv: {len(rs)} dòng")
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"images": names, "split": c["split"], "bench": b, "estimate": est, "minutes": (time.time() - t_all) / 60,
                   "digest": dg, "debug": cfg["_debug"], "lognormal_sigmas": sigmas, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
                  fh, indent=1, default=str)
