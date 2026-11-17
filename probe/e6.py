"""
E6 — cô lập gradient guidance (A) và can thiệp vào cấu trúc đồng bộ (B).

Mọi đường đi từ x′ (đen) tới x, đánh giá bằng cùng cầu phương phân tầng M điểm/đoạn. Phép so CHÍNH ở mức K = 1:
metric tính trên TỪNG realization (một đường / một lần hoán vị), rồi lấy trung bình metric theo seed trong ảnh.
Bản đồ trung bình nhiều đường (ensemble) chỉ báo riêng, phụ.

E6-A  (N = guided_steps cho mọi họ)
  ig            đường thẳng
  guided        Guided IG gốc (chọn lịch trình theo gradient của chính ảnh)
  guided_perm   lịch trình Guided hoán vị "bin" (kênh × thập phân vị |Δx|), R seed
  guided_xfer   lịch trình Guided của ẢNH KHÁC (split other), gán theo quy tắc khớp biên độ đặt trước: trong từng kênh,
                phần tử có hạng |Δx| thứ k của ảnh đích nhận lịch trình của phần tử hạng k ở ảnh cho
  spi           DP α độc lập, K đường
  spi_perm      chính các đường spi, hoán vị "bin" (đối chứng âm, ghép cặp theo đường)
  spi_lam       Z^λ = (1−λ)t + λZ^SPI, λ khớp độ lệch hình học của Guided (chỉ hình học)
  Chi phí dựng đường (eval gradient chỉ để chọn đường) báo riêng: Guided = N, các họ khác = 0.

E6-B  DP α, cùng phân phối biên Beta(αt, α(1−t)) cho mọi phần tử; chỉ thay việc CHIA SẺ lịch trình:
  indep (mỗi phần tử), rgb (3 kênh một pixel), block_b (mỗi khối b×b, cả 3 kênh), global (một lịch trình cho cả ảnh;
  về hình học là đường thẳng IG với tham số hoá khác → canonical chính xác = IG: đối chứng).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import torch

from . import config as C
from .e5 import geom_dist, perm_index, permute, schedule
from .mechanism import aw, decomp, strat_points, summ
from .metrics import energy_mask, insdel, iou, make_base, tie_perm
from .model import ProbFn, TargetFn
from .paths import cum_nodes, guided_nodes, spi_increments, straight_nodes
from .pilot import Ctx, sync
from .quad import Counter, fg

F64 = torch.float64


def e6_digest(cfg: dict) -> str:
    keys = ("seed", "model", "data", "paths", "metrics", "e6")
    return __import__("hashlib").sha256(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def e6_out(cfg: dict) -> str:
    d = C.out_dir(cfg, "e6_" + cfg["e6"]["split"])
    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# lịch trình dùng chung, chuyển lịch trình
# ─────────────────────────────────────────────────────────────────────────────

def group_index(kind: str, shape) -> tuple[torch.Tensor, int]:
    """Nhãn nhóm (C·H·W,) cho kiểu chia sẻ; trả (nhãn, số nhóm)."""
    Cc, H, W = shape
    c = torch.arange(Cc).view(Cc, 1, 1).expand(Cc, H, W)
    h = torch.arange(H).view(1, H, 1).expand(Cc, H, W)
    w = torch.arange(W).view(1, 1, W).expand(Cc, H, W)
    if kind == "indep":
        lab = (c * H + h) * W + w
        n = Cc * H * W
    elif kind == "rgb":
        lab = h * W + w
        n = H * W
    elif kind.startswith("block"):
        b = int(kind[5:])
        nw = (W + b - 1) // b
        lab = (h // b) * nw + (w // b)
        n = ((H + b - 1) // b) * nw
    elif kind == "global":
        lab = torch.zeros_like(c)
        n = 1
    else:
        raise ValueError(kind)
    return lab.flatten(), n


def shared_schedule(kind: str, alpha: float, N: int, shape, gen) -> tuple[torch.Tensor, int]:
    lab, n = group_index(kind, shape)
    Zg = cum_nodes(spi_increments(alpha, N, (n,), gen))                       # (N+1, n)
    return Zg[:, lab.to(Zg.device)].view((N + 1,) + tuple(shape)), n


def transfer(Zd: torch.Tensor, ad: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
    """Khớp biên độ trong từng kênh: phần tử hạng k theo |Δx| ở ảnh đích nhận lịch trình của hạng k ở ảnh cho."""
    Zt = torch.empty_like(Zd)
    for c in range(Zd.shape[1]):
        od = torch.argsort(ad[c].flatten(), stable=True)
        ot = torch.argsort(at[c].flatten(), stable=True)
        src = Zd[:, c].flatten(1)
        dst = torch.empty_like(src)
        dst[:, ot] = src[:, od]
        Zt[:, c] = dst.view_as(Zd[:, c])
    return Zt


# ─────────────────────────────────────────────────────────────────────────────
# một ảnh
# ─────────────────────────────────────────────────────────────────────────────

def e6_image(ctx: Ctx, j: int, idx: int, donors: list) -> dict:
    cfg = ctx.cfg
    c = cfg["e6"]
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
    rows = defaultdict(list)
    ens = {}
    alpha = float(cfg["paths"]["spi10"]["alpha"])
    timing = {}

    def quality(A):
        A = A.float()
        q = {"iou": iou(A, mask, perm_t), "energy_mask": energy_mask(A, None, mask)}
        for bn, base in bases.items():
            q[f"insdel_{bn}"] = insdel(prob, x, base, A, ic["n_steps"], ic["batch"])
        return q

    def one(exp, family, r, X, M, extra=None):
        N = X.shape[0] - 1
        P, _ = strat_points(X, M, ctx.gen("e6q", exp, family, r, N, M, it.image))
        _, G = fg(fn, P, ctx.batch)
        A, W = aw(G, X[1:] - X[:-1], M)
        del P, G
        s = summ(A, W, m3)
        row = {"image": it.image, "exp": exp, "family": family, "r": r, "level": "path", "D": D, **s,
               "within": s["W"] - s["L1"], "dist_geom": geom_dist(X), **quality(A), **(extra or {})}
        rows["paths"].append(row)
        e = ens.setdefault((exp, family), {"A": None, "rows": []})
        e["A"] = A.clone() if e["A"] is None else e["A"] + A
        e["rows"].append(s)
        return A

    # ── E6-A ───────────────────────────────────────────────────────────────
    a = c["A"]
    Ng, M = int(a["N"]), int(a["M"])
    t0 = time.time()
    one("A", "ig", 0, straight_nodes(x, x0, Ng), M, {"path_cost": 0})
    gp = cfg["paths"]["guided"]
    n0 = Counter.n
    Xg, _, _, gap = guided_nodes(fn, x, x0, Ng, float(gp["fraction"]), float(gp["max_dist"]), ctx.batch)
    cost_g = Counter.n - n0
    one("A", "guided", 0, Xg, M, {"path_cost": cost_g, "end_gap": gap})
    Zg = schedule(Xg, x0, dx)
    dg = geom_dist(Xg)
    timing["guided"] = time.time() - t0
    t0 = time.time()
    for r in range(int(a["perm_seeds"])):
        g = torch.Generator().manual_seed(C.seed_of(ctx.seed, "e6perm", r, it.image))
        ix = perm_index("bin", dx, g, int(a["perm_bins"]), 16)
        one("A", "guided_perm", r, x0 + permute(Zg, ix) * dx, M, {"path_cost": cost_g})
    at = dx[0].abs()
    for r, (dname, Zd, ad) in enumerate(donors):
        one("A", "guided_xfer", r, x0 + transfer(Zd, ad, at) * dx, M, {"path_cost": 0, "donor": dname})
    tlin = torch.linspace(0, 1, Ng + 1, device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
    for p in range(int(a["spi_paths"])):
        Z = cum_nodes(spi_increments(alpha, Ng, dx.shape[1:], ctx.gen("e6spi", Ng, it.image, p))).to(x.dtype)
        one("A", "spi", p, x0 + Z * dx, M, {"path_cost": 0})
        g = torch.Generator().manual_seed(C.seed_of(ctx.seed, "e6spiperm", p, it.image))
        one("A", "spi_perm", p, x0 + permute(Z, perm_index("bin", dx, g, int(a["perm_bins"]), 16)) * dx, M, {"path_cost": 0})
        ds = geom_dist(x0 + Z * dx)
        lam = min(1.0, dg / ds) if ds > 0 else 0.0
        one("A", "spi_lam", p, x0 + ((1 - lam) * tlin + lam * Z) * dx, M, {"path_cost": 0, "lambda": lam})
        del Z
    timing["A"] = time.time() - t0

    # ── E6-B ───────────────────────────────────────────────────────────────
    b = c["B"]
    Nb, Mb = int(b["N"]), int(b["M"])
    t0 = time.time()
    one("B", "ig", 0, straight_nodes(x, x0, Nb), Mb, {"groups": 0})
    for kind in b["kinds"]:
        for p in range(int(b["paths"])):
            Z, n = shared_schedule(kind, alpha, Nb, tuple(dx.shape[1:]), ctx.gen("e6B", kind, Nb, it.image, p))
            one("B", kind, p, x0 + Z.to(x.dtype) * dx, Mb, {"groups": n})
            del Z
    timing["B"] = time.time() - t0
    sync()

    # ensemble (phụ)
    for (exp, fam), e in ens.items():
        K = len(e["rows"])
        Ab = e["A"] / K
        rows["ens"].append({"image": it.image, "exp": exp, "family": fam, "level": "ens", "K": K,
                            **decomp(e["rows"], Ab, D, m3), **quality(Ab)})
    rows["images"].append({"image": it.image, "index": it.index, "cls": cls, "D": D, "mask_area": float(mask.float().mean()),
                           "dist_guided": dg, "guided_cost": cost_g, "timing": json.dumps(timing)})
    return {"rows": dict(rows)}


def make_donors(ctx: Ctx) -> list:
    """Lịch trình Guided IG của các ảnh split `other` (mô hình, lớp dự đoán và baseline của chính ảnh đó)."""
    cfg = ctx.cfg
    a = cfg["e6"]["A"]
    gp = cfg["paths"]["guided"]
    out = []
    for idx in ctx.sp["other"][:int(a["donors"])]:
        it, xd, _, cls = ctx.load(idx)
        x0d = make_base(xd, "black")
        dxd = xd - x0d
        Xd, _, _, _ = guided_nodes(TargetFn(ctx.bb, cls), xd, x0d, int(a["N"]), float(gp["fraction"]), float(gp["max_dist"]),
                                   ctx.batch)
        out.append((it.image, schedule(Xd, x0d, dxd), dxd[0].abs()))
        del Xd
        print(f"  [e6] ảnh cho lịch trình: {it.image}", flush=True)
    return out


def estimate(cfg: dict, per_eval: float, per_seq: float) -> dict:
    c = cfg["e6"]
    a, b = c["A"], c["B"]
    ic = cfg["metrics"]["insdel"]
    ins = len(ic["bases"]) * 2 * (ic["n_steps"] + 1) / 3
    nA = 2 + int(a["perm_seeds"]) + int(a["donors"]) + 3 * int(a["spi_paths"])
    nB = 1 + len(b["kinds"]) * int(b["paths"])
    ev = nA * (int(a["N"]) * int(a["M"]) + ins) + nB * (int(b["N"]) * int(b["M"]) + ins)
    n_fam = 7 + 1 + len(b["kinds"])
    ev += n_fam * ins
    t = c["n_images"] * (ev * per_eval + (int(a["N"]) + 1) * per_seq) + int(a["donors"]) * (int(a["N"]) + 1) * per_seq
    return {"e6_min": t / 60, "evals_per_image": ev}


def run(ctx: Ctx):
    import pandas as pd
    from .pilot import bench
    cfg = ctx.cfg
    c = cfg["e6"]
    out = e6_out(cfg)
    dg = e6_digest(cfg)
    if c["split"] == "confirm":                       # tiền đăng ký: lần chạy đầu ghi hash, các lần sau phải trùng
        pp = os.path.join(out, "e6_prereg.json")
        if os.path.exists(pp):
            if json.load(open(pp))["digest"] != dg:
                raise SystemExit(f"config e6 đã đổi sau khi tiền đăng ký ({pp}); split confirm phải giữ nguyên config")
        else:
            import datetime as _dt
            with open(pp, "w") as fh:
                json.dump({"digest": dg, "time": _dt.datetime.now().isoformat(timespec="seconds"), "e6": c}, fh,
                          indent=1, default=str)
    b = bench(ctx)
    est = estimate(cfg, b["per_eval"], b["per_seq"])
    print(f"[e6] split {c['split']}; {b['per_eval'] * 1e3:.2f} ms/eval; ước tính ≈ {est['e6_min']:.1f} phút", flush=True)
    t_all = time.time()
    idxs = ctx.sp[c["split"]][:int(c["n_images"])]
    names = [f"seg{i:05d}" for i in idxs]
    cd = os.path.join(out, "cache")
    os.makedirs(cd, exist_ok=True)
    todo = [(j, i) for j, i in enumerate(idxs)
            if not (os.path.exists(os.path.join(cd, f"seg{i:05d}.pt"))
                    and torch.load(os.path.join(cd, f"seg{i:05d}.pt"), weights_only=False).get("digest") == dg)]
    donors = make_donors(ctx) if todo else []
    for j, idx in todo:
        image = f"seg{idx:05d}"
        t0, n0 = time.time(), Counter.n
        print(f"  [e6] ảnh {j + 1}/{len(idxs)} {image}", flush=True)
        obj = e6_image(ctx, j, idx, donors)
        obj["digest"] = dg
        torch.save(obj, os.path.join(cd, f"{image}.pt"))
        print(f"  [e6] {image}: {Counter.n - n0} eval, {time.time() - t0:.0f}s; tổng {(time.time() - t_all) / 60:.1f} phút",
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
            pd.DataFrame(rs).to_csv(os.path.join(out, f"e6_{k}.csv"), index=False)
            print(f"[e6] e6_{k}.csv: {len(rs)} dòng")
    meta = {"images": names, "split": c["split"], "bench": b, "estimate": est, "minutes": (time.time() - t_all) / 60,
            "digest": dg, "debug": cfg["_debug"],
            "donors": [f"seg{i:05d}" for i in ctx.sp["other"][:int(c["A"]["donors"])]],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "e6_config": c}
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
