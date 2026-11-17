"""
E7 — đối chứng cấu trúc nhóm cùng cỡ (exploratory, 16 ảnh pilot).

S  (chính) DP α, K = 1 mỗi đường, cùng N, M; mọi nhóm gồm cả 3 kênh của các pixel được chọn:
     indep    mỗi phần tử (pixel × kênh) một lịch trình            (như E6)
     rgb      mỗi pixel một lịch trình (3 kênh chung)              (mốc cùng quy ước "nhóm gồm RGB")
     blockB   khối B×B pixel liền kề                               (B² pixel/nhóm)
     randG    G pixel chọn ngẫu nhiên khắp ảnh (phân hoạch ngẫu nhiên mới cho mỗi đường)
     global   một lịch trình cho cả ảnh = đường thẳng IG tham số hoá lại (đối chứng)
   block4 ↔ rand16 và block16 ↔ rand256: cùng số nhóm, cùng cỡ nhóm, cùng biên Beta, chỉ khác bố trí không gian.
N  (độ bền) cùng so sánh ở N lớn hơn (ít gộp bước nhảy hơn) trên vài ảnh: các phần tử khác nhóm rơi vào cùng ô lưới
   thì cũng đi chéo cùng nhau — quy tắc nội suy (gấp khúc qua các nút lưới) giống hệt cho mọi kiểu nhóm.
Q  (cầu phương dày) Guided / Guided hoán vị / Guided chuyển lịch trình / IG ở M lớn, kèm M nhỏ lồng nhau trên CÙNG
   đường: hiệu ΔIoU ≈ 0,011 của E6-A có nhạy với estimator không.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import torch

from . import config as C
from .e5 import geom_dist, perm_index, permute, schedule
from .e6 import transfer
from .mechanism import aw, decomp, nested_pick, strat_points, summ
from .metrics import energy_mask, insdel, iou, make_base, tie_perm
from .model import ProbFn, TargetFn
from .paths import cum_nodes, guided_nodes, spi_increments, straight_nodes
from .pilot import Ctx, sync
from .quad import Counter, fg

F64 = torch.float64


def e7_digest(cfg: dict) -> str:
    keys = ("seed", "model", "data", "paths", "metrics", "e7")
    return __import__("hashlib").sha256(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def e7_out(cfg: dict) -> str:
    d = C.out_dir(cfg, "e7")
    os.makedirs(d, exist_ok=True)
    return d


def group_labels(kind: str, shape, gen: torch.Generator) -> tuple[torch.Tensor, int]:
    """Nhãn nhóm (C·H·W,) và số nhóm. randG dùng `gen` (CPU) cho phân hoạch ngẫu nhiên."""
    Cc, H, W = shape
    pix = None
    if kind == "indep":
        return torch.arange(Cc * H * W), Cc * H * W
    if kind == "rgb":
        pix, n = torch.arange(H * W), H * W
    elif kind == "global":
        return torch.zeros(Cc * H * W, dtype=torch.long), 1
    elif kind.startswith("block"):
        b = int(kind[5:])
        nw = (W + b - 1) // b
        h = torch.arange(H).view(H, 1).expand(H, W)
        w = torch.arange(W).view(1, W).expand(H, W)
        pix, n = ((h // b) * nw + (w // b)).flatten(), ((H + b - 1) // b) * nw
    elif kind.startswith("rand"):
        g = int(kind[4:])
        order = torch.randperm(H * W, generator=gen)
        pix = torch.empty(H * W, dtype=torch.long)
        pix[order] = torch.arange(H * W) // g
        n = (H * W + g - 1) // g
    else:
        raise ValueError(kind)
    return pix.repeat(Cc), n


def grouped_schedule(kind: str, alpha: float, N: int, shape, gen_dev, gen_cpu) -> tuple[torch.Tensor, int]:
    lab, n = group_labels(kind, shape, gen_cpu)
    Zg = cum_nodes(spi_increments(alpha, N, (n,), gen_dev))
    return Zg[:, lab.to(Zg.device)].view((N + 1,) + tuple(shape)), n


def e7_image(ctx: Ctx, j: int, idx: int, donors: list) -> dict:
    cfg = ctx.cfg
    c = cfg["e7"]
    it, x, mask, cls = ctx.load(idx)
    fn, prob = TargetFn(ctx.bb, cls), ProbFn(ctx.bb, cls)
    x0 = make_base(x, "black")
    dx = x - x0
    m3 = mask.unsqueeze(0).expand_as(x[0])
    perm_t = tie_perm(it.image, x.shape[-1] * x.shape[-2], ctx.seed, ctx.dev)
    ic = cfg["metrics"]["insdel"]
    bases = {b: make_base(x, b, ic["blur_kernel"], ic["blur_sigma"]) for b in ic["bases"]}
    fe, _ = fg(fn, torch.cat([x0, x]), ctx.batch)
    D = float(fe[1] - fe[0])
    alpha = float(cfg["paths"]["spi10"]["alpha"])
    rows = defaultdict(list)
    ens = {}
    timing = {}

    def quality(A):
        A = A.float()
        q = {"iou": iou(A, mask, perm_t), "energy_mask": energy_mask(A, None, mask)}
        for bn, base in bases.items():
            q[f"insdel_{bn}"] = insdel(prob, x, base, A, ic["n_steps"], ic["batch"])
        return q

    def record(exp, family, r, A, W, X, extra=None):
        s = summ(A, W, m3)
        rows["paths"].append({"image": it.image, "exp": exp, "family": family, "r": r, "D": D, **s,
                              "within": s["W"] - s["L1"], "dist_geom": geom_dist(X), **quality(A), **(extra or {})})
        e = ens.setdefault((exp, family), {"A": None, "rows": []})
        e["A"] = A.clone() if e["A"] is None else e["A"] + A
        e["rows"].append(s)

    def one(exp, family, r, X, M, extra=None):
        N = X.shape[0] - 1
        P, _ = strat_points(X, M, ctx.gen("e7q", exp, family, r, N, M, it.image))
        _, G = fg(fn, P, ctx.batch)
        A, W = aw(G, X[1:] - X[:-1], M)
        del P, G
        record(exp, family, r, A, W, X, extra)

    def structure(exp, N, M, kinds, paths):
        one(exp, "ig", 0, straight_nodes(x, x0, N), M, {"groups": 0, "group_size": 0})
        for kind in kinds:
            for p in range(paths):
                gc = torch.Generator().manual_seed(C.seed_of(ctx.seed, "e7part", kind, N, it.image, p))
                Z, n = grouped_schedule(kind, alpha, N, tuple(dx.shape[1:]), ctx.gen("e7S", kind, N, it.image, p), gc)
                one(exp, kind, p, x0 + Z.to(x.dtype) * dx, M, {"groups": n, "group_size": dx[0].numel() / n})
                del Z

    t0 = time.time()
    s = c["S"]
    structure("S", int(s["N"]), int(s["M"]), list(s["kinds"]), int(s["paths"]))
    timing["S"] = time.time() - t0
    if j < int(c["Njump"]["n_images"]):
        t0 = time.time()
        nj = c["Njump"]
        structure("N", int(nj["N"]), int(nj["M"]), list(nj["kinds"]), int(nj["paths"]))
        timing["N"] = time.time() - t0

    if j < int(c["Q"]["n_images"]):
        t0 = time.time()
        q = c["Q"]
        Ng, Mhi, Mlo = int(q["N"]), int(q["M_hi"]), int(q["M_lo"])
        gp = cfg["paths"]["guided"]

        def nested(family, r, X, extra=None):
            N = X.shape[0] - 1
            P, _ = strat_points(X, Mhi, ctx.gen("e7Q", family, r, N, Mhi, it.image))
            _, G = fg(fn, P, ctx.batch)
            del P
            G = G.view((N, Mhi) + tuple(X.shape[1:]))
            dX = X[1:] - X[:-1]
            A, W = aw(G.reshape((N * Mhi,) + tuple(X.shape[1:])), dX, Mhi)
            record(f"Q{Mhi}", family, r, A, W, X, extra)
            ix = nested_pick(N, Mhi, Mlo, ctx.gen("e7Qpick", family, r, it.image), x.device)
            Gl = torch.stack([G[k, ix[k]] for k in range(N)])
            A, W = aw(Gl.reshape((N * Mlo,) + tuple(X.shape[1:])), dX, Mlo)
            record(f"Q{Mlo}", family, r, A, W, X, extra)
            del G, Gl

        nested("ig", 0, straight_nodes(x, x0, Ng))
        Xg, _, _, gap = guided_nodes(fn, x, x0, Ng, float(gp["fraction"]), float(gp["max_dist"]), ctx.batch)
        nested("guided", 0, Xg, {"end_gap": gap})
        Zg = schedule(Xg, x0, dx)
        for r in range(int(q["perm_seeds"])):
            g = torch.Generator().manual_seed(C.seed_of(ctx.seed, "e6perm", r, it.image))     # cùng seed với E6-A
            nested("guided_perm", r, x0 + permute(Zg, perm_index("bin", dx, g, 10, 16)) * dx)
        at = dx[0].abs()
        for r, (dname, Zd, ad) in enumerate(donors):
            nested("guided_xfer", r, x0 + transfer(Zd, ad, at) * dx, {"donor": dname})
        del Xg, Zg
        timing["Q"] = time.time() - t0
    sync()

    # ensemble + tương đồng bản đồ trung bình với indep (cùng exp)
    maps = {}
    for (exp, fam), e in ens.items():
        K = len(e["rows"])
        Ab = e["A"] / K
        maps[(exp, fam)] = Ab.abs().sum(0).flatten().double()
        rows["ens"].append({"image": it.image, "exp": exp, "family": fam, "K": K, **decomp(e["rows"], Ab, D, m3), **quality(Ab)})

    def rank(v):
        r = torch.empty_like(v)
        r[v.argsort()] = torch.arange(v.numel(), device=v.device, dtype=v.dtype)
        return r - r.mean()

    for (exp, fam), v in maps.items():
        for refn in ("indep", "ig"):
            if (exp, refn) in maps and fam != refn:
                a, b = rank(v), rank(maps[(exp, refn)])
                rows["mapsim"].append({"image": it.image, "exp": exp, "family": fam, "ref": refn,
                                       "spearman_absmap": float((a * b).sum() / (a.norm() * b.norm()))})
    rows["images"].append({"image": it.image, "index": it.index, "cls": cls, "D": D, "mask_area": float(mask.float().mean()),
                           "timing": json.dumps(timing)})
    return {"rows": dict(rows)}


def make_donors(ctx: Ctx) -> list:
    cfg = ctx.cfg
    q = cfg["e7"]["Q"]
    gp = cfg["paths"]["guided"]
    out = []
    for idx in ctx.sp["other"][:int(q["donors"])]:
        it, xd, _, cls = ctx.load(idx)
        x0d = make_base(xd, "black")
        Xd, _, _, _ = guided_nodes(TargetFn(ctx.bb, cls), xd, x0d, int(q["N"]), float(gp["fraction"]), float(gp["max_dist"]),
                                   ctx.batch)
        out.append((it.image, schedule(Xd, x0d, xd - x0d), (xd - x0d)[0].abs()))
        del Xd
        print(f"  [e7] ảnh cho lịch trình: {it.image}", flush=True)
    return out


def estimate(cfg: dict, per_eval: float, per_seq: float) -> dict:
    c = cfg["e7"]
    ic = cfg["metrics"]["insdel"]
    ins = len(ic["bases"]) * 2 * (ic["n_steps"] + 1) / 3
    s, nj, q = c["S"], c["Njump"], c["Q"]
    eS = (1 + len(s["kinds"]) * int(s["paths"])) * (int(s["N"]) * int(s["M"]) + ins)
    eN = (1 + len(nj["kinds"]) * int(nj["paths"])) * (int(nj["N"]) * int(nj["M"]) + ins)
    nq = 2 + int(q["perm_seeds"]) + int(q["donors"])
    eQ = nq * (int(q["N"]) * int(q["M_hi"]) + 2 * ins)
    t = c["n_images"] * eS * per_eval + nj["n_images"] * eN * per_eval \
        + q["n_images"] * (eQ * per_eval + (int(q["N"]) + 1) * per_seq) + int(q["donors"]) * (int(q["N"]) + 1) * per_seq
    return {"e7_min": t / 60, "evals_S": eS, "evals_N": eN, "evals_Q": eQ}


def run(ctx: Ctx):
    import pandas as pd
    from .pilot import bench
    cfg = ctx.cfg
    c = cfg["e7"]
    out = e7_out(cfg)
    dg = e7_digest(cfg)
    b = bench(ctx)
    est = estimate(cfg, b["per_eval"], b["per_seq"])
    print(f"[e7] {b['per_eval'] * 1e3:.2f} ms/eval; ước tính ≈ {est['e7_min']:.1f} phút", flush=True)
    t_all = time.time()
    idxs = ctx.sp["pilot"][:int(c["n_images"])]
    names = [f"seg{i:05d}" for i in idxs]
    cd = os.path.join(out, "cache")
    os.makedirs(cd, exist_ok=True)

    def done(i):
        p = os.path.join(cd, f"seg{i:05d}.pt")
        return os.path.exists(p) and torch.load(p, weights_only=False).get("digest") == dg

    todo = [(j, i) for j, i in enumerate(idxs) if not done(i)]
    donors = make_donors(ctx) if any(j < int(c["Q"]["n_images"]) for j, _ in todo) else []
    for j, idx in todo:
        image = f"seg{idx:05d}"
        t0, n0 = time.time(), Counter.n
        print(f"  [e7] ảnh {j + 1}/{len(idxs)} {image}", flush=True)
        obj = e7_image(ctx, j, idx, donors)
        obj["digest"] = dg
        torch.save(obj, os.path.join(cd, f"{image}.pt"))
        print(f"  [e7] {image}: {Counter.n - n0} eval, {time.time() - t0:.0f}s; tổng {(time.time() - t_all) / 60:.1f} phút",
              flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    allrows = defaultdict(list)
    for im in names:
        obj = torch.load(os.path.join(cd, f"{im}.pt"), weights_only=False)
        for k, rs in obj["rows"].items():
            allrows[k] += rs
    for k, rs in allrows.items():
        if rs:
            pd.DataFrame(rs).to_csv(os.path.join(out, f"e7_{k}.csv"), index=False)
            print(f"[e7] e7_{k}.csv: {len(rs)} dòng")
    meta = {"images": names, "bench": b, "estimate": est, "minutes": (time.time() - t_all) / 60, "digest": dg,
            "debug": cfg["_debug"], "donors": [f"seg{i:05d}" for i in ctx.sp["other"][:int(c["Q"]["donors"])]],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "e7_config": c}
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
