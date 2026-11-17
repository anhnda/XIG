"""
E1′ (gate) và E4 (cross-method): cùng một thước đo exposure cho IG, SPI, Guided IG, BlurIG.

gate  (a) SPI K đường + IG, cùng đường, M ∈ {2,4,8} lồng nhau: phần "between" do nhiễu cầu phương
          (nhiễu riêng từng đường thổi phồng E‖A^γ‖₁, trung bình nhiều đường xoá nó) và độ tin cậy của metric
          từng đường (IoU ở M nhỏ so với M lớn trên cùng đường).
      (b) toy bậc hai và bậc năm: gộp bước nhảy DP có làm chệch E[A] (chệch quần thể) không — so ghép cặp
          từng đường lưới thô với chính nó ở lưới mịn. A chính xác (Gauss–Legendre 3 điểm, gradient bậc ≤ 4),
          W bằng lưới trung điểm rất mịn.
e4    E4a+b — mỗi ảnh, mỗi phương pháp: A, W, W theo 4 mode gradient (không cần mask):
          cm_low / cm_high : phần trung bình theo kênh (cùng dấu giữa kênh), tần số thấp / cao (Gauss σ px)
          cd_low / cd_high : phần lệch giữa kênh, tần số thấp / cao        (g = tổng 4 mode, chính xác)
      phân bố exposure theo tiến độ τ (tỉ lệ L1 đã đi) và theo khoảng cách tới đoạn thẳng nối hai đầu của chính đường,
      độ dài đường Σ∫|dx_i| (BlurIG không đơn điệu theo từng pixel), IDGI trên cùng đường cố định.
      Tham chiếu exposure: IG cho SPI/Guided (cùng đầu đen → x); đoạn thẳng blur(x) → x cho BlurIG.
e4c   Quét mức lệch nhịp: SPI với α ∈ alphas (α lớn ≈ đường thẳng), K đường, M điểm/đoạn.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict

import torch

from . import config as C
from .mechanism import aw, decomp, nested_pick, strat_points, summ
from .metrics import energy_mask, insdel, iou, make_base, tie_perm
from .model import ProbFn, TargetFn
from .paths import (blur_nodes, cum_nodes, gaussian_blur, guided_nodes, spi_increments, spi_nodes_from,
                    straight_nodes)
from .pilot import Ctx, spi_increments_for, sync
from .quad import Counter, fg, idgi

F64 = torch.float64
MODES = ("cm_low", "cm_high", "cd_low", "cd_high")


def cross_digest(cfg: dict, stage: str) -> str:
    keys = ("seed", "model", "data", "paths", "metrics")
    part = {k: cfg.get(k) for k in keys}
    part["cross"] = {"stage": cfg["cross"][stage], "mode_sigma": cfg["cross"]["mode_sigma"]}
    return __import__("hashlib").sha256(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()


def cross_out(cfg: dict) -> str:
    d = C.out_dir(cfg, "cross")
    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# exposure theo mode, theo trạng thái
# ─────────────────────────────────────────────────────────────────────────────

def mode_split(G: torch.Tensor, sigma: float) -> dict:
    gm = G.mean(1, keepdim=True)
    gd = G - gm
    lm, ld = gaussian_blur(gm, sigma), gaussian_blur(gd, sigma)
    return {"cm_low": lm.expand_as(G), "cm_high": (gm - lm).expand_as(G), "cd_low": ld, "cd_high": gd - ld}


def exposure(G: torch.Tensor, X: torch.Tensor, M: int, sigma: float, chunk: int = 16) -> dict:
    """G (N·M, C,H,W) tại các điểm phân tầng của đường X (N+1 nút). -> A, W, W theo mode (bản đồ float64),
    W từng đoạn, τ và khoảng cách tới dây cung tại trung điểm đoạn, độ dài đường."""
    N = X.shape[0] - 1
    dX = X[1:] - X[:-1]
    A, W = aw(G, dX, M)
    Gs = G.view((N, M) + tuple(G.shape[1:]))
    Wm = {m: torch.zeros_like(W) for m in MODES}
    segW = torch.zeros(N, dtype=F64, device=G.device)
    for k0 in range(0, N, chunk):
        k1 = min(N, k0 + chunk)
        g = Gs[k0:k1].reshape(((k1 - k0) * M,) + tuple(G.shape[1:]))
        ms = mode_split(g, sigma)
        d = dX[k0:k1].double().abs()
        for m in MODES:
            Wm[m] += (ms[m].double().abs().reshape((k1 - k0, M) + tuple(G.shape[1:])).sum(1) * d).sum(0) / M
        segW[k0:k1] = (Gs[k0:k1].double().abs().sum(1) * d).flatten(1).sum(1) / M
    l1 = dX.double().abs().flatten(1).sum(1)
    cum = torch.cumsum(l1, 0)
    tau = (cum - l1 / 2) / cum[-1]
    chord = (X[-1] - X[0]).double().flatten()
    mid = ((X[1:] + X[:-1]) / 2 - X[0]).double().flatten(1)
    s = (mid @ chord) / (chord @ chord)
    w = mid - s[:, None] * chord[None]
    dist = w.norm(dim=1) / chord.norm()
    return {"A": A, "W": W, "Wm": Wm, "segW": segW, "tau": tau, "dist": dist, "len": float(l1.sum()),
            "chord_l1": float(chord.abs().sum())}


def state_rows(prefix: dict, e: dict, tau_bins: int, dist_edges: list) -> list:
    rows = []
    tot = float(e["segW"].sum())
    tb = torch.clamp((e["tau"] * tau_bins).long(), 0, tau_bins - 1)
    for b in range(tau_bins):
        rows.append({**prefix, "axis": "tau", "bin": b, "lo": b / tau_bins, "hi": (b + 1) / tau_bins,
                     "share": float(e["segW"][tb == b].sum()) / tot})
    edges = torch.tensor(dist_edges, dtype=F64, device=e["dist"].device)
    db = torch.clamp(torch.bucketize(e["dist"], edges, right=True) - 1, 0, len(dist_edges) - 2)
    for b in range(len(dist_edges) - 1):
        rows.append({**prefix, "axis": "dist", "bin": b, "lo": dist_edges[b], "hi": dist_edges[b + 1],
                     "share": float(e["segW"][db == b].sum()) / tot})
    rows.append({**prefix, "axis": "dist_mean_w", "bin": -1, "lo": float("nan"), "hi": float("nan"),
                 "share": float((e["segW"] * e["dist"]).sum()) / tot})
    return rows


def mode_summ(e: dict, m3) -> dict:
    out = {}
    for m in MODES:
        out[f"W_{m}"] = float(e["Wm"][m].sum())
        out[f"W_{m}_in"] = float(e["Wm"][m][m3].sum())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# gate (E1′)
# ─────────────────────────────────────────────────────────────────────────────

def gate_image(ctx: Ctx, idx: int) -> dict:
    cfg = ctx.cfg
    g = cfg["cross"]["gate"]
    it, x, mask, cls = ctx.load(idx)
    fn = TargetFn(ctx.bb, cls)
    x0 = make_base(x, "black")
    dx = x - x0
    m3 = mask.unsqueeze(0).expand_as(x[0])
    perm = tie_perm(it.image, x.shape[-1] * x.shape[-2], ctx.seed, ctx.dev)
    fe, _ = fg(fn, torch.cat([x0, x]), ctx.batch)
    D = float(fe[1] - fe[0])
    Ms = sorted(int(m) for m in g["M"])
    Mmax = Ms[-1]
    rows = defaultdict(list)

    def per_M(fam, p, X):
        N = X.shape[0] - 1
        P, _ = strat_points(X, Mmax, ctx.gen("gate-q", fam, p, it.image))
        _, G = fg(fn, P, ctx.batch)
        G = G.view((N, Mmax) + tuple(X.shape[1:]))
        dX = X[1:] - X[:-1]
        out = {}
        for M in Ms:
            if M == Mmax:
                Gm = G
            else:
                ix = nested_pick(N, Mmax, M, ctx.gen("gate-pick", fam, p, M, it.image), x.device)
                Gm = torch.stack([G[k, ix[k]] for k in range(N)])
            out[M] = aw(Gm.reshape((N * M,) + tuple(X.shape[1:])), dX, M)
        return out

    sums = {M: None for M in Ms}
    prs = {M: [] for M in Ms}
    res = per_M("ig", 0, straight_nodes(x, x0, int(g["ig_steps"])))
    for M, (A, W) in res.items():
        s = summ(A, W, m3)
        rows["gate"].append({"image": it.image, "level": "image", "method": "ig", "M": M,
                             **decomp([s], A, D, m3), "iou": iou(A.float(), mask, perm)})
    pc = cfg["paths"]["spi10"]
    K = int(g["spi_paths"])
    for p in range(K):
        X = spi_nodes_from(spi_increments_for(ctx, pc, dx.shape[1:], int(pc["steps"]), it.image, p), x0, dx)
        res = per_M("spi10", p, X)
        for M, (A, W) in res.items():
            s = summ(A, W, m3)
            prs[M].append(s)
            sums[M] = A.clone() if sums[M] is None else sums[M] + A
            rows["gate"].append({"image": it.image, "level": "path", "method": "spi10", "M": M, "p": p, "D": D, **s,
                                 "within": s["W"] - s["L1"], "iou": iou(A.float(), mask, perm)})
        del X
    for M in Ms:
        Ab = sums[M] / K
        rows["gate"].append({"image": it.image, "level": "image", "method": "spi10", "M": M,
                             **decomp(prs[M], Ab, D, m3), "iou": iou(Ab.float(), mask, perm)})
    return dict(rows)


def toy_run(cfg: dict) -> list:
    t = cfg["cross"]["gate"]["toy"]
    gen = torch.Generator().manual_seed(int(t["seed"]))
    d, alpha = int(t["d"]), float(t["alpha"])
    b = torch.randn(d, generator=gen, dtype=F64) * float(t["b_scale"])
    Cm = torch.randn(d, d, generator=gen, dtype=F64) * float(t["c_scale"]) / math.sqrt(d)
    Cm = (Cm + Cm.T) / 2
    Cm.fill_diagonal_(0)
    V = torch.randn(int(t["q_dirs"]), d, generator=gen, dtype=F64) / math.sqrt(d) * 2.0
    x = 0.2 + 0.8 * torch.rand(d, generator=gen, dtype=F64)
    fams = {
        "quadratic": (lambda z: b + z @ Cm.T, lambda z: z @ b + 0.5 * ((z @ Cm) * z).sum(-1)),
        "quintic": (lambda z: b + z @ Cm.T + float(t["q_scale"]) * ((z @ V.T) ** 4) @ V,
                    lambda z: z @ b + 0.5 * ((z @ Cm) * z).sum(-1) + float(t["q_scale"]) * ((z @ V.T) ** 5).sum(-1) / 5),
    }
    Nf = int(t["N_fine"])
    Ns = sorted(set(int(n) for n in t["N"]) | {Nf})
    P = int(t["paths"])
    Z = torch.stack([cum_nodes(spi_increments(alpha, Nf, (d,), torch.Generator().manual_seed(C.seed_of(t["seed"], "toy5", p))))
                     for p in range(P)]) * x
    gl_t = torch.tensor([0.5 - 0.5 * math.sqrt(0.6), 0.5, 0.5 + 0.5 * math.sqrt(0.6)], dtype=F64)
    gl_w = torch.tensor([5 / 18, 8 / 18, 5 / 18], dtype=F64)
    rows = []
    for fam, (grad, Fz) in fams.items():
        D = float(Fz(x) - Fz(torch.zeros(d, dtype=F64)))
        lev = {}
        for N in Ns:
            Zn = Z[:, ::Nf // N]
            dZ = Zn[:, 1:] - Zn[:, :-1]
            T = max(4, int(t["w_points"]) // N)
            tm = (torch.arange(T, dtype=F64) + 0.5) / T
            As, Ws = [], []
            for c0 in range(0, P, 25):
                z0, dz = Zn[c0:c0 + 25, :-1], dZ[c0:c0 + 25]
                gA = sum(w * grad(z0 + tt * dz) for tt, w in zip(gl_t, gl_w))
                As.append((gA * dz).sum(1))
                Wc = torch.zeros(z0.shape[0], dtype=F64)
                for tt in tm:
                    Wc += (grad(z0 + tt * dz).abs() * dz.abs()).sum((1, 2)) / T
                Ws.append(Wc)
            lev[N] = (torch.cat(As), torch.cat(Ws))
        Af, Wf = lev[Nf]
        Ti = 4096
        tm = (torch.arange(Ti, dtype=F64) + 0.5) / Ti
        gi = torch.stack([grad(tt * x) for tt in tm])
        A_ig, W_ig = (gi.mean(0) * x), float((gi.abs().mean(0) * x).sum())
        for N, (A, W) in lev.items():
            L1 = A.abs().sum(1)
            Ab = A.mean(0)
            dd = A - Af
            md_ = dd.mean(0)
            se = dd.std(0) / math.sqrt(P)
            z = torch.where(se > 0, md_ / se.clamp_min(1e-300), torch.zeros_like(md_))
            rows.append({"family": fam, "N": N, "is_fine": N == Nf, "paths": P, "d": d, "D": D,
                         "EW": float(W.mean()), "within": float((W - L1).mean()), "between": float(L1.mean() - Ab.abs().sum()),
                         "across": float(Ab.abs().sum() - Ab.sum().abs()), "resid": float(Ab.sum().abs() - abs(D)),
                         "perpath_A_dist_fine": float((dd.abs().sum(1) / Af.abs().sum(1)).mean()),
                         "perpath_W_ratio_fine": float((W / Wf).mean()),
                         "paired_bias_rel": float(md_.abs().sum() / Af.mean(0).abs().sum()),
                         "paired_z_rms": float(z.pow(2).mean().sqrt()),
                         "paired_z_max": float(z.abs().max()),
                         "W_ig": W_ig, "L1_ig": float(A_ig.abs().sum())})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# E4a + E4b
# ─────────────────────────────────────────────────────────────────────────────

class MethodAcc:
    def __init__(self):
        self.rows, self.A, self.W, self.Wm, self.I = [], None, None, None, None
        self.n = 0
        self.len, self.chord = [], []

    def add(self, e, m3, A_idgi=None):
        s = summ(e["A"], e["W"], m3)
        s.update(mode_summ(e, m3))
        self.rows.append(s)
        self.A = e["A"].clone() if self.A is None else self.A + e["A"]
        self.W = e["W"].clone() if self.W is None else self.W + e["W"]
        self.Wm = {m: v.clone() for m, v in e["Wm"].items()} if self.Wm is None else \
            {m: self.Wm[m] + e["Wm"][m] for m in MODES}
        if A_idgi is not None:
            self.I = A_idgi.double().clone() if self.I is None else self.I + A_idgi.double()
        self.len.append(e["len"]); self.chord.append(e["chord_l1"])
        self.n += 1
        return s


def e4_image(ctx: Ctx, j: int, idx: int) -> dict:
    cfg = ctx.cfg
    c = cfg["cross"]["e4"]
    sigma = float(cfg["cross"]["mode_sigma"])
    it, x, mask, cls = ctx.load(idx)
    fn, prob = TargetFn(ctx.bb, cls), ProbFn(ctx.bb, cls)
    x0 = make_base(x, "black")
    dx = x - x0
    m3 = mask.unsqueeze(0).expand_as(x[0])
    perm = tie_perm(it.image, x.shape[-1] * x.shape[-2], ctx.seed, ctx.dev)
    ic = cfg["metrics"]["insdel"]
    bases = {b: make_base(x, b, ic["blur_kernel"], ic["blur_sigma"]) for b in ic["bases"]}
    M = int(c["M"])
    smax = float(cfg["paths"]["blur"]["max_sigma"])
    xb = gaussian_blur(x, smax)
    rows = defaultdict(list)
    accs = {}
    timing = {}

    def quality(A):
        A = A.float()
        q = {"iou": iou(A, mask, perm), "energy_mask": energy_mask(A, None, mask)}
        for bn, base in bases.items():
            q[f"insdel_{bn}"] = insdel(prob, x, base, A, ic["n_steps"], ic["batch"])
        return q

    def run_path(name, X, key, f_nodes=None, g_nodes=None):
        N = X.shape[0] - 1
        if f_nodes is None:
            f_nodes, g_all = fg(fn, X, ctx.batch)
            g_nodes = g_all[:N]
        P, _ = strat_points(X, M, ctx.gen("e4q", name, N, M, it.image, *key))
        _, G = fg(fn, P, ctx.batch)
        e = exposure(G, X, M, sigma)
        AI, info = idgi(g_nodes, f_nodes[1:] - f_nodes[:-1], ctx.tol)
        acc = accs.setdefault(name, MethodAcc())
        s = acc.add(e, m3, AI)
        acc.D = float(f_nodes[-1] - f_nodes[0])
        rows["state"] += state_rows({"image": it.image, "method": name, "p": key[0] if key else 0}, e,
                                    int(c["tau_bins"]), list(c["dist_bins"]))
        del P, G
        return e, s, info

    pc = cfg["paths"]
    t0, n0 = time.time(), Counter.n
    run_path("ig", straight_nodes(x, x0, int(c["ig_steps"])), (0,))
    run_path("line_blur", straight_nodes(x, xb, int(c["ig_steps"])), (0,))
    run_path("blur", blur_nodes(x, int(c["blur_steps"]), smax), (0,))
    timing["ig_blur"] = time.time() - t0
    t0 = time.time()
    gp = pc["guided"]
    Xg, fgd, ggd, gap = guided_nodes(fn, x, x0, int(c["guided_steps"]), float(gp["fraction"]), float(gp["max_dist"]), ctx.batch)
    run_path("guided", Xg, (0,), fgd, ggd)
    timing["guided"] = time.time() - t0
    t0 = time.time()
    sp = pc["spi10"]
    for p in range(int(c["spi_paths"])):
        X = spi_nodes_from(spi_increments_for(ctx, sp, dx.shape[1:], int(sp["steps"]), it.image, p), x0, dx)
        e, s, _ = run_path("spi10", X, (p,))
        rows["spi_paths"].append({"image": it.image, "p": p, **s, "within": s["W"] - s["L1"], "len": e["len"],
                                  "iou": iou(e["A"].float(), mask, perm), "energy_mask": energy_mask(e["A"].float(), None, mask)})
        del X
    sync()
    timing["spi10"] = time.time() - t0

    ref = {"ig": "ig", "spi10": "ig", "guided": "ig", "blur": "line_blur", "line_blur": "line_blur"}
    avoided = {}
    for name, acc in accs.items():
        K = acc.n
        Ab, Ib = acc.A / K, acc.I / K
        dd = decomp(acc.rows, Ab, acc.D, m3)
        row = {"image": it.image, "method": name, "ref": ref[name], "K": K, **dd,
               "path_len": sum(acc.len) / K, "chord_l1": sum(acc.chord) / K}
        for m in MODES:
            row[f"W_{m}"] = sum(r[f"W_{m}"] for r in acc.rows) / K
            row[f"W_{m}_in"] = sum(r[f"W_{m}_in"] for r in acc.rows) / K
        row.update({f"{k}": v for k, v in quality(Ab).items()})
        row.update({f"idgi_{k}": v for k, v in quality(Ib).items()})
        row["idgi_sum"] = float(Ib.sum())
        if name == "guided":
            row["end_gap"] = gap
        rows["methods"].append(row)
        Wpix = (acc.W / K).sum(0)
        avoided[name] = Wpix
        avoided[name + "|modes"] = {m: (acc.Wm[m] / K).sum(0) for m in MODES}

    def spear(a, b):
        ra = torch.empty_like(a.flatten()); ra[a.flatten().argsort()] = torch.arange(a.numel(), device=a.device, dtype=a.dtype)
        rb = torch.empty_like(b.flatten()); rb[b.flatten().argsort()] = torch.arange(b.numel(), device=b.device, dtype=b.dtype)
        ra, rb = ra - ra.mean(), rb - rb.mean()
        return float((ra * rb).sum() / (ra.norm() * rb.norm()))

    def cos(a, b):
        a, b = a.flatten(), b.flatten()
        return float((a @ b) / (a.norm() * b.norm()))

    meths = ["spi10", "guided", "blur"]
    dW = {m: avoided[ref[m]] - avoided[m] for m in meths}
    lr = {m: torch.log((avoided[m] + 1e-12) / (avoided[ref[m]] + 1e-12)) for m in meths}
    for a in range(len(meths)):
        for b in range(a + 1, len(meths)):
            ma, mb = meths[a], meths[b]
            row = {"image": it.image, "a": ma, "b": mb, "spearman_avoided": spear(dW[ma], dW[mb]),
                   "cos_avoided": cos(dW[ma], dW[mb]), "spearman_logratio": spear(lr[ma], lr[mb])}
            for m in MODES:
                da = avoided[ref[ma] + "|modes"][m] - avoided[ma + "|modes"][m]
                db = avoided[ref[mb] + "|modes"][m] - avoided[mb + "|modes"][m]
                row[f"spearman_avoided_{m}"] = spear(da, db)
            rows["pairs"].append(row)
    rows["images"].append({"image": it.image, "index": it.index, "cls": cls, "mask_area": float(mask.float().mean()),
                           "timing": json.dumps(timing)})
    heat = None
    if j < int(c.get("heatmap_images", 2)):
        heat = {"x": it.x.half(), "mask": mask.cpu(), **{f"W_{m}": avoided[m].float().cpu() for m in accs},
                **{f"dW_{m}": dW[m].float().cpu() for m in meths}}
    return {"rows": dict(rows), "heat": heat}


# ─────────────────────────────────────────────────────────────────────────────
# E4c — quét mức lệch nhịp
# ─────────────────────────────────────────────────────────────────────────────

def e4c_image(ctx: Ctx, j: int, idx: int) -> dict:
    cfg = ctx.cfg
    c = cfg["cross"]["e4c"]
    sigma = float(cfg["cross"]["mode_sigma"])
    it, x, mask, cls = ctx.load(idx)
    fn, prob = TargetFn(ctx.bb, cls), ProbFn(ctx.bb, cls)
    x0 = make_base(x, "black")
    dx = x - x0
    m3 = mask.unsqueeze(0).expand_as(x[0])
    perm = tie_perm(it.image, x.shape[-1] * x.shape[-2], ctx.seed, ctx.dev)
    ic = cfg["metrics"]["insdel"]
    bases = {b: make_base(x, b, ic["blur_kernel"], ic["blur_sigma"]) for b in ic["bases"]}
    fe, _ = fg(fn, torch.cat([x0, x]), ctx.batch)
    D = float(fe[1] - fe[0])
    M, K, N = int(c["M"]), int(c["paths"]), int(cfg["paths"]["spi10"]["steps"])
    rows = defaultdict(list)
    for alpha in c["alphas"]:
        alpha = float(alpha)
        Asum, prs, Wm_sum = None, [], {m: 0.0 for m in MODES}
        for p in range(K):
            delta = spi_increments(alpha, N, dx.shape[1:], ctx.gen("e4c", alpha, N, it.image, p))
            X = x0 + cum_nodes(delta).to(x.dtype) * dx
            del delta
            P, _ = strat_points(X, M, ctx.gen("e4c-q", alpha, N, M, it.image, p))
            _, G = fg(fn, P, ctx.batch)
            e = exposure(G, X, M, sigma)
            s = summ(e["A"], e["W"], m3)
            s.update(mode_summ(e, m3))
            prs.append(s)
            Asum = e["A"].clone() if Asum is None else Asum + e["A"]
            row = {"image": it.image, "level": "path", "alpha": alpha, "p": p, "D": D, **s,
                   "within": s["W"] - s["L1"], "async": float((e["segW"] * e["dist"]).sum() / e["segW"].sum()),
                   "async_unweighted": float(e["dist"].mean()),
                   "iou": iou(e["A"].float(), mask, perm), "energy_mask": energy_mask(e["A"].float(), None, mask)}
            if c.get("per_path_insdel", False) and "blur" in bases:
                row["insdel_blur"] = insdel(prob, x, bases["blur"], e["A"].float(), ic["n_steps"], ic["batch"])
            rows["alpha"].append(row)
            del X, P, G, e
        Ab = Asum / K
        row = {"image": it.image, "level": "image", "alpha": alpha, "h": 1.0 / (alpha + 1.0), **decomp(prs, Ab, D, m3)}
        for m in MODES:
            row[f"W_{m}"] = sum(r[f"W_{m}"] for r in prs) / K
            row[f"W_{m}_in"] = sum(r[f"W_{m}_in"] for r in prs) / K
        A = Ab.float()
        row.update({"iou": iou(A, mask, perm), "energy_mask": energy_mask(A, None, mask)})
        for bn, base in bases.items():
            row[f"insdel_{bn}"] = insdel(prob, x, base, A, ic["n_steps"], ic["batch"])
        rows["alpha"].append(row)
        sync()
        print(f"    [e4c] {it.image} α={alpha:g}", flush=True)
    return {"rows": dict(rows)}


# ─────────────────────────────────────────────────────────────────────────────
# chạy
# ─────────────────────────────────────────────────────────────────────────────

def estimate(cfg: dict, per_eval: float, per_seq: float) -> dict:
    cc = cfg["cross"]
    g, e4, e4c = cc["gate"], cc["e4"], cc["e4c"]
    N = int(cfg["paths"]["spi10"]["steps"])
    Mg = max(int(m) for m in g["M"])
    n_gate = int(g["ig_steps"]) * Mg + int(g["spi_paths"]) * N * Mg
    M = int(e4["M"])
    ig = int(e4["ig_steps"])
    n_e4 = 2 * (ig + 1 + ig * M) + (int(e4["blur_steps"]) + 1) * (1 + M) + int(e4["guided_steps"]) * M \
        + int(e4["spi_paths"]) * (N + 1 + N * M)
    ic = cfg["metrics"]["insdel"]
    ins4 = 5 * 2 * len(ic["bases"]) * 2 * (ic["n_steps"] + 1) / 3
    n_e4c = len(e4c["alphas"]) * int(e4c["paths"]) * N * int(e4c["M"])
    ins4c = len(e4c["alphas"]) * (len(ic["bases"]) + (int(e4c["paths"]) if e4c.get("per_path_insdel") else 0)) \
        * 2 * (ic["n_steps"] + 1) / 3
    return {"gate_min": g["n_images"] * n_gate * per_eval / 60,
            "e4_min": e4["n_images"] * ((n_e4 + ins4) * per_eval + (int(e4["guided_steps"]) + 1) * per_seq) / 60,
            "e4c_min": e4c["n_images"] * (n_e4c + ins4c) * per_eval / 60}


def run(ctx: Ctx, stage: str):
    import pandas as pd
    from .pilot import bench
    cfg = ctx.cfg
    out = cross_out(cfg)
    dg = cross_digest(cfg, stage)
    b = bench(ctx)
    est = estimate(cfg, b["per_eval"], b["per_seq"])
    print(f"[cross] {b['per_eval'] * 1e3:.2f} ms/eval; ước tính gate ≈ {est['gate_min']:.1f}, E4 ≈ {est['e4_min']:.1f}, "
          f"E4c ≈ {est['e4c_min']:.1f} phút", flush=True)
    t_all = time.time()
    if stage == "gate":
        pd.DataFrame(toy_run(cfg)).to_csv(os.path.join(out, "gate_toy.csv"), index=False)
        print("[cross] gate_toy.csv", flush=True)
    n = int(cfg["cross"][stage]["n_images"])
    fnc = {"gate": lambda j, i: gate_image(ctx, i), "e4": lambda j, i: e4_image(ctx, j, i),
           "e4c": lambda j, i: e4c_image(ctx, j, i)}[stage]
    names = []
    for j, idx in enumerate(ctx.sp["pilot"][:n]):
        image = f"seg{idx:05d}"
        names.append(image)
        cd = os.path.join(out, "cache", stage)
        os.makedirs(cd, exist_ok=True)
        cp = os.path.join(cd, f"{image}.pt")
        if os.path.exists(cp) and torch.load(cp, weights_only=False).get("digest") == dg:
            print(f"  [{stage}] {image}: đã có trong cache, bỏ qua")
            continue
        t0, n0 = time.time(), Counter.n
        print(f"  [{stage}] ảnh {j + 1}/{n} {image}", flush=True)
        obj = fnc(j, idx)
        if "rows" not in obj:
            obj = {"rows": obj}
        obj["digest"] = dg
        torch.save(obj, cp)
        print(f"  [{stage}] {image}: {Counter.n - n0} eval, {time.time() - t0:.0f}s; tổng {(time.time() - t_all) / 60:.1f} phút",
              flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    allrows, heats = defaultdict(list), {}
    for im in names:
        obj = torch.load(os.path.join(out, "cache", stage, f"{im}.pt"), weights_only=False)
        for k, rs in obj["rows"].items():
            allrows[k] += rs
        if obj.get("heat") is not None:
            heats[im] = obj["heat"]
    fname = {"gate": "gate", "methods": "e4_methods", "pairs": "e4_pairs", "state": "e4_state",
             "spi_paths": "e4_spi_paths", "images": "e4_images", "alpha": "e4c_alpha"}
    for k, rs in allrows.items():
        if rs:
            pd.DataFrame(rs).to_csv(os.path.join(out, f"{fname.get(k, k)}.csv"), index=False)
            print(f"[cross] {fname.get(k, k)}.csv: {len(rs)} dòng")
    if heats:
        torch.save(heats, os.path.join(out, "e4_heatmaps.pt"))
    mp = os.path.join(out, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    meta[stage] = {"images": names, "bench": b, "estimate": est, "minutes": (time.time() - t_all) / 60, "digest": dg,
                   "debug": cfg["_debug"], "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    meta["cross_config"] = cfg["cross"]
    with open(mp, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
