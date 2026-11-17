"""
E1–E3: phân rã cơ chế của SPI.

Với một đường γ (gấp khúc qua các nút), trên CÙNG các điểm cầu phương:
    A^γ_i = ∫_γ ∂_iF dx_i,           W_γ = Σ_i ∫_γ |∂_iF| |dx_i|        (exposure)
Với Ā = E_γ A^γ (ước lượng bằng trung bình K đường), D = F(x) − F(x′):
    E W − |D| = E[W − ‖A^γ‖₁]           within   (triệt tiêu trong từng feature dọc một đường)  ≥ 0
              + (E‖A^γ‖₁ − ‖Ā‖₁)        between  (triệt tiêu giữa các đường)                    ≥ 0
              + (‖Ā‖₁ − |ΣĀ|)           across   (triệt tiêu giữa các feature)                  ≥ 0
              + (|ΣĀ| − |D|)            resid    (sai số completeness của ước lượng; ≈ 0)
Cả ba số hạng đầu không âm CHÍNH XÁC với mọi cầu phương (bất đẳng thức tam giác trên cùng điểm).
Ngân sách so với IG (một đường, between = 0):
    ‖A_IG‖₁ − ‖Ā_SPI‖₁ = (W_IG − E W_SPI) + (within_SPI − within_IG) + between_SPI
                          [exposure]         [within thêm]                [between]
W bất biến khi đổi thang từng toạ độ (|∂F/∂x_i||dx_i|), nên không phụ thuộc chuẩn hoá input.

Cầu phương: mỗi đoạn k chia M tầng, một điểm U(0,1) mỗi tầng (dùng chung mọi phần tử), trọng số Δx_k/M —
không chệch cho cả ∫∂F dx và ∫|∂F||dx| trên đoạn. ‖·‖₁ của ước lượng thì chệch lên (Jensen) → E1 đo mức này.

e1   (a) đường cố định, tăng M (lồng nhau: M nhỏ = chọn ngẫu nhiên một điểm trong mỗi nhóm tầng của M_max);
     (b) gộp bước nhảy DP: cùng một đường DP ở N = 64/128/256 (cộng gộp số gia), cùng tổng số điểm;
     (c) toy F bậc hai có tương tác: W, A tính CHÍNH XÁC trên từng đường gấp khúc, so lưới thô với lưới mịn.
e23  E2 (gradient ResNet thật, K đường SPI, cùng seed với pilot → 32 đường đầu trùng đường pilot) và
     E3 (bẫy H trên các ảnh đầu, phân rã H riêng và F+H) trong một lượt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict

import torch

from . import config as C
from .components import Trap, direction, kappa_for_eta, predicted_attenuation
from .metrics import energy_mask, insdel, iou, make_base, tie_perm
from .model import ProbFn, TargetFn
from .paths import coarsen, cum_nodes, spi_increments, spi_nodes_from, straight_nodes
from .pilot import Ctx, spi_increments_for, sync
from .quad import Counter, fg

F64 = torch.float64


def mech_digest(cfg: dict) -> str:
    keys = ("seed", "model", "data", "paths", "metrics", "mechanism")
    return hashlib.sha256(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def mech_out(cfg: dict) -> str:
    d = C.out_dir(cfg, "mechanism")
    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# cầu phương phân tầng và A, W
# ─────────────────────────────────────────────────────────────────────────────

def strat_points(X: torch.Tensor, M: int, gen) -> tuple[torch.Tensor, torch.Tensor]:
    """-> P (N·M, C,H,W) theo thứ tự đoạn-chính, s (N, M) vị trí trong đoạn."""
    N = X.shape[0] - 1
    U = torch.rand(N, M, generator=gen, device=gen.device, dtype=F64).to(X.device)
    s = (torch.arange(M, device=X.device, dtype=F64).view(1, M) + U) / M
    dX = X[1:] - X[:-1]
    shp = (N, M) + (1,) * (X.dim() - 1)
    P = X[:-1].unsqueeze(1) + s.view(shp).to(X.dtype) * dX.unsqueeze(1)
    return P.reshape((N * M,) + tuple(X.shape[1:])), s


def grads(fn, P, batch):
    _, g = fg(fn, P, batch)
    return g


def aw(G: torch.Tensor, dX: torch.Tensor, M: int):
    """G (N·M, …), dX (N, …) -> A, W (bản đồ theo phần tử, float64)."""
    N = dX.shape[0]
    Gs = G.view((N, M) + tuple(G.shape[1:])).double()
    d = dX.double()
    A = (Gs.sum(1) * d).sum(0) / M
    W = (Gs.abs().sum(1) * d.abs()).sum(0) / M
    return A, W


def nested_pick(N: int, Mmax: int, M: int, gen, device) -> torch.Tensor:
    """Chỉ số (N, M) vào các tầng của M_max: mỗi tầng thô (gồm Mmax/M tầng mịn) chọn đều một tầng mịn
    → điểm được chọn phân phối đều trong tầng thô (cầu phương phân tầng hợp lệ ở mức M)."""
    r = Mmax // M
    off = torch.randint(0, r, (N, M), generator=gen, device=gen.device).to(device)
    return torch.arange(M, device=device).view(1, M) * r + off


def summ(A, W, m3) -> dict:
    a = A.abs()
    return {"W": float(W.sum()), "W_in": float(W[m3].sum()), "L1": float(a.sum()), "L1_in": float(a[m3].sum()),
            "sumA": float(A.sum())}


# ─────────────────────────────────────────────────────────────────────────────
# E1
# ─────────────────────────────────────────────────────────────────────────────

def e1_image(ctx: Ctx, idx: int) -> dict:
    cfg = ctx.cfg
    e = cfg["mechanism"]["e1"]
    it, x, mask, cls = ctx.load(idx)
    fn = TargetFn(ctx.bb, cls)
    x0 = make_base(x, "black")
    dx = x - x0
    m3 = mask.unsqueeze(0).expand_as(x[0])
    fe, _ = fg(fn, torch.cat([x0, x]), ctx.batch)
    D = float(fe[1] - fe[0])
    Ms = sorted(int(m) for m in e["M"])
    Mmax = Ms[-1]
    rows_q, rows_dp = [], []

    def fixed_path(kind, p, X):
        N = X.shape[0] - 1
        P, _ = strat_points(X, Mmax, ctx.gen("e1q", kind, p, it.image))
        G = grads(fn, P, ctx.batch).view((N, Mmax) + tuple(X.shape[1:]))
        dX = X[1:] - X[:-1]
        _, gl = fg(fn, X[:-1], ctx.batch)                         # nút trái
        res = {}
        Al = (gl.double() * dX.double()).sum(0)
        Wl = (gl.double().abs() * dX.double().abs()).sum(0)
        res["left"] = (Al, Wl)
        for M in Ms:
            if M == Mmax:
                Gm = G
            else:
                ix = nested_pick(N, Mmax, M, ctx.gen("e1pick", kind, p, M, it.image), x.device)
                Gm = torch.stack([G[k, ix[k]] for k in range(N)])
            res[M] = aw(Gm.reshape((N * M,) + tuple(X.shape[1:])), dX, M)
        Aref, Wref = res[Mmax]
        for q, (A, W) in res.items():
            s = summ(A, W, m3)
            rows_q.append({"image": it.image, "path": kind, "p": p, "N": N, "quad": str(q),
                           "points": N * (1 if q == "left" else int(q)), "D": D, **s,
                           "within": s["W"] - s["L1"],
                           "W_relerr": abs(s["W"] - float(Wref.sum())) / float(Wref.sum()),
                           "L1_relerr": abs(s["L1"] - float(Aref.abs().sum())) / float(Aref.abs().sum()),
                           "A_dist_l1": float((A - Aref).abs().sum()) / float(Aref.abs().sum()),
                           "A_dist_l2": float((A - Aref).norm()) / float(Aref.norm()),
                           "resid_rel": (s["sumA"] - D) / abs(D)})
        del G

    t0 = time.time()
    fixed_path("ig", 0, straight_nodes(x, x0, int(e["ig_steps"])))
    pc = cfg["paths"]["spi10"]
    for p in e["spi_paths"]:
        X = spi_nodes_from(spi_increments_for(ctx, pc, dx.shape[1:], int(pc["steps"]), it.image, int(p)), x0, dx)
        fixed_path("spi10", int(p), X)
    print(f"    [e1] {it.image} đường cố định: {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    Ns = sorted(int(n) for n in e["dp_N"])
    tot = int(e["dp_points"])
    for real in range(int(e["dp_realizations"])):
        delta = spi_increments(float(pc["alpha"]), Ns[-1], dx.shape[1:], ctx.gen("e1dp", it.image, real))
        out = {}
        for N in Ns:
            X = x0 + cum_nodes(coarsen(delta, Ns[-1] // N)).to(x.dtype) * dx
            M = max(1, tot // N)
            P, _ = strat_points(X, M, ctx.gen("e1dpq", N, it.image, real))
            out[N] = aw(grads(fn, P, ctx.batch), X[1:] - X[:-1], M) + (M,)
            del X, P
        Aref, Wref, _ = out[Ns[-1]]
        for N, (A, W, M) in out.items():
            s = summ(A, W, m3)
            rows_dp.append({"image": it.image, "real": real, "N": N, "M": M, "D": D, **s,
                            "within": s["W"] - s["L1"], "W_ratio_fine": s["W"] / float(Wref.sum()),
                            "within_ratio_fine": (s["W"] - s["L1"]) / max(float(Wref.sum() - Aref.abs().sum()), 1e-30),
                            "A_dist_l1_fine": float((A - Aref).abs().sum()) / float(Aref.abs().sum()),
                            "resid_rel": (s["sumA"] - D) / abs(D)})
        del delta
    print(f"    [e1] {it.image} gộp bước nhảy DP: {time.time() - t0:.0f}s", flush=True)
    return {"e1_quad": rows_q, "e1_dp": rows_dp}


def _abs_lin_int(a0, a1):
    """∫₀¹ |a0 + a1 t| dt, theo phần tử."""
    b = a0 + a1
    same = (a0 * b) >= 0
    whole = (a0 + b).abs() / 2
    split = (a0 * a0 + b * b) / (2 * a1.abs().clamp_min(1e-300))
    return torch.where(same, whole, split)


def toy_exact(Z: torch.Tensor, bvec, Cm):
    """F(z) = b·z + ½ zᵀCz. Z (P, N+1, d) nút. ∂F = b + Cz tuyến tính trên mỗi đoạn → A, W chính xác."""
    dZ = Z[:, 1:] - Z[:, :-1]
    g0 = bvec + Z[:, :-1] @ Cm.T
    g1 = bvec + Z[:, 1:] @ Cm.T
    A = ((g0 + g1) / 2 * dZ).sum(1)
    W = (_abs_lin_int(g0, g1 - g0) * dZ.abs()).sum((1, 2))      # (P,)
    return A, W


def e1_toy(cfg: dict) -> list:
    t = cfg["mechanism"]["e1"]["toy"]
    g = torch.Generator().manual_seed(int(t["seed"]))
    d, alpha = int(t["d"]), float(t["alpha"])
    bvec = torch.randn(d, generator=g, dtype=F64) * float(t["b_scale"])
    Cm = torch.randn(d, d, generator=g, dtype=F64) * float(t["c_scale"]) / math.sqrt(d)
    Cm = (Cm + Cm.T) / 2
    Cm.fill_diagonal_(0)
    x = 0.2 + 0.8 * torch.rand(d, generator=g, dtype=F64)
    Nf = int(t["N_fine"])
    Ns = sorted(int(n) for n in t["N"])
    P = int(t["paths"])
    Zf = []
    for p in range(P):
        gp = torch.Generator().manual_seed(C.seed_of(t["seed"], "toy", p))
        Zf.append(cum_nodes(spi_increments(alpha, Nf, (d,), gp)))
    Zf = torch.stack(Zf) * x                               # (P, Nf+1, d), x′ = 0
    D = float(bvec @ x + 0.5 * x @ Cm @ x)
    rows = []
    ref = {}
    levels = Ns + ([Nf] if Nf not in Ns else [])
    for N in sorted(levels, reverse=True):
        A, W = toy_exact(Zf[:, ::Nf // N], bvec, Cm)
        ref[N] = (A, W)
    Af, Wf = ref[Nf]
    Al = straight_nodes(x.view(1, d, 1, 1), torch.zeros(1, d, 1, 1, dtype=F64), 2).view(1, 3, d)
    Aig, Wig = toy_exact(Al, bvec, Cm)
    for N in sorted(ref):
        A, W = ref[N]
        L1 = A.abs().sum(1)
        Abar = A.mean(0)
        rows.append({"N": N, "is_fine": N == Nf, "paths": P, "d": d, "D": D,
                     "EW": float(W.mean()), "EL1": float(L1.mean()), "L1bar": float(Abar.abs().sum()),
                     "within": float((W - L1).mean()), "between": float(L1.mean() - Abar.abs().sum()),
                     "across": float(Abar.abs().sum() - Abar.sum().abs()), "resid": float(Abar.sum().abs() - abs(D)),
                     "perpath_A_dist_fine": float(((A - Af).abs().sum(1) / Af.abs().sum(1)).mean()),
                     "perpath_W_ratio_fine": float((W / Wf).mean()),
                     "mean_A_dist_fine": float((Abar - Af.mean(0)).abs().sum() / Af.mean(0).abs().sum()),
                     "mc_se_mean_A": float((Af.std(0) / math.sqrt(P)).abs().sum() / Af.mean(0).abs().sum()),
                     "W_ig": float(Wig[0]), "L1_ig": float(Aig[0].abs().sum())})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# E2 + E3
# ─────────────────────────────────────────────────────────────────────────────

class Comp:
    """Tích luỹ theo đường cho một thành phần (F, H, F+H) của một họ đường."""

    def __init__(self):
        self.Asum = None
        self.Wsum = None
        self.rows = []

    def add(self, A, W, m3):
        self.Asum = A.clone() if self.Asum is None else self.Asum + A
        self.Wsum = W.clone() if self.Wsum is None else self.Wsum + W
        s = summ(A, W, m3)
        self.rows.append(s)
        return s


def decomp(rs: list, Abar, D: float, m3) -> dict:
    K = len(rs)
    EW = sum(r["W"] for r in rs) / K
    EL1 = sum(r["L1"] for r in rs) / K
    EW_in = sum(r["W_in"] for r in rs) / K
    EL1_in = sum(r["L1_in"] for r in rs) / K
    a = Abar.abs()
    L1bar, L1bar_in = float(a.sum()), float(a[m3].sum())
    S = abs(float(Abar.sum()))
    return {"K": K, "D": D, "EW": EW, "EL1": EL1, "L1bar": L1bar, "absSum": S,
            "within": EW - EL1, "between": EL1 - L1bar, "across": L1bar - S, "resid": S - abs(D),
            "EW_in": EW_in, "EL1_in": EL1_in, "L1bar_in": L1bar_in,
            "within_in": EW_in - EL1_in, "between_in": EL1_in - L1bar_in,
            "W_sd": float(torch.tensor([r["W"] for r in rs], dtype=F64).std()) if K > 1 else 0.0}


def e23_image(ctx: Ctx, j: int, idx: int) -> dict:
    cfg = ctx.cfg
    mc = cfg["mechanism"]
    e2, e3 = mc["e2"], mc["e3"]
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
    M = int(e2["M"])
    do_trap = j < int(e3["n_images"])
    rows = defaultdict(list)
    timing = {}

    def quality(A, full=True, blur_only=False):
        A = A.float()
        q = {"iou": iou(A, mask, perm), "energy_mask": energy_mask(A, None, mask)}
        if full:
            for b, base in bases.items():
                if blur_only and b != "blur":
                    continue
                q[f"insdel_{b}"] = insdel(prob, x, base, A, ic["n_steps"], ic["batch"])
        return q

    # ── IG (một đường) ─────────────────────────────────────────────────────
    t0, n0 = time.time(), Counter.n
    Nig = int(e2["ig_steps"])
    Xig = straight_nodes(x, x0, Nig)
    Pig, _ = strat_points(Xig, M, ctx.gen("e2q", "ig", Nig, M, it.image))
    Gig = grads(fn, Pig, ctx.batch)
    dXig = Xig[1:] - Xig[:-1]
    traps = []
    if do_trap:
        _, gn = fg(fn, Xig[:-1], ctx.batch)                      # nút trái của IG: κ theo cùng định nghĩa với pilot
        tt = torch.arange(Nig, device=ctx.dev, dtype=F64) / Nig
        us = {}
        specs = [(float(r), float(e3["eta"]), "high") for r in e3["rho"]]
        if e3.get("u_low_at") is not None:
            specs.append((float(e3["u_low_at"]), float(e3["eta"]), "low"))
        for rho, eta, uk in specs:
            if uk not in us:
                us[uk] = direction(uk, dx, ctx.gen("u", uk, it.image), float(cfg["exp_b"]["low_sigma"]))
            kap = kappa_for_eta(eta, gn, tt)
            traps.append((f"H_rho{rho:g}_eta{eta:g}_{uk}", Trap(x0, dx, us[uk], kap, rho),
                          {"rho": rho, "eta": eta, "u": uk, "kappa": kap}))
        del gn
    comps = {"ig": defaultdict(Comp), "spi10": defaultdict(Comp)}

    def add_path(fam, p, G, P, dX, extra_row):
        A, W = aw(G, dX, M)
        s = comps[fam]["F"].add(A, W, m3)
        row = {"image": it.image, "level": "path", "method": fam, "component": "F", "p": p, "D": D, **s,
               "within": s["W"] - s["L1"], **extra_row}
        if fam == "spi10":
            row.update(quality(A, full=bool(e2["per_path_insdel"]), blur_only=True))
        rows["mech"].append(row)
        for name, tr, meta in traps:
            GH = tr.grad(P)
            AH, WH = aw(GH, dX, M)
            sH = comps[fam][f"H|{name}"].add(AH, WH, m3)
            AF, WF = aw(G + GH, dX, M)
            sF = comps[fam][f"F+H|{name}"].add(AF, WF, m3)
            for cn, ss in (("H", sH), ("F+H", sF)):
                rows["mech"].append({"image": it.image, "level": "path", "method": fam, "component": cn,
                                     "trap": name, **meta, "p": p, "D": D, **ss, "within": ss["W"] - ss["L1"]})
            del GH
        return A, W

    A_ig, W_ig = add_path("ig", 0, Gig, Pig, dXig, {})
    q_ig = quality(A_ig)
    del Gig, Pig
    sync()
    timing["ig"] = {"seconds": time.time() - t0, "evals": Counter.n - n0}

    # ── SPI ────────────────────────────────────────────────────────────────
    t0, n0 = time.time(), Counter.n
    pc = cfg["paths"]["spi10"]
    N, K = int(pc["steps"]), int(e2["paths"])
    ks = sorted(set(int(k) for k in e2["K_checkpoints"] if int(k) <= K) | {K})
    kq = {}
    Asum = None
    for p in range(K):
        X = spi_nodes_from(spi_increments_for(ctx, pc, dx.shape[1:], N, it.image, p), x0, dx)
        P, _ = strat_points(X, M, ctx.gen("e2q", "spi", N, M, it.image, p))
        G = grads(fn, P, ctx.batch)
        A, W = add_path("spi10", p, G, P, X[1:] - X[:-1], {})
        Asum = A.clone() if Asum is None else Asum + A
        if (p + 1) in ks:
            kq[p + 1] = (Asum / (p + 1)).clone()
        del X, P, G
    sync()
    timing["spi10"] = {"seconds": time.time() - t0, "evals": Counter.n - n0}

    # ── hàng mức ảnh ───────────────────────────────────────────────────────
    def img_rows(fam, kset, abars):
        for cname, c in comps[fam].items():
            for k in kset:
                if cname == "F":
                    Abar = abars[k]
                else:
                    Abar = None
                if Abar is None:
                    if k != len(c.rows):
                        continue
                    Abar = c.Asum / len(c.rows)
                dd = decomp(c.rows[:k], Abar, D, m3)
                row = {"image": it.image, "level": "image", "method": fam, "component": cname.split("|")[0], **dd}
                if "|" in cname:
                    nm = cname.split("|")[1]
                    meta = next(m for n, _, m in traps if n == nm)
                    row.update({"trap": nm, **meta})
                    if cname.startswith("H|") and fam == "spi10":
                        row["pred_att"] = predicted_attenuation(1.0 / (float(pc["alpha"]) + 1.0), meta["rho"])
                if cname == "F":
                    row.update(quality(Abar, full=True))
                rows["mech"].append(row)

    img_rows("ig", [1], {1: A_ig.clone()})
    img_rows("spi10", ks, kq)
    rows["images"].append({"image": it.image, "index": it.index, "cls": cls, "D": D, "mask_area": float(mask.float().mean()),
                           "do_trap": do_trap, "timing": json.dumps(timing), **{f"ig_{k}": v for k, v in q_ig.items()}})
    heat = None
    if j < int(e2.get("heatmap_images", 2)):
        cs = comps["spi10"]["F"]
        heat = {"x": it.x.half(), "mask": mask.cpu(),
                "W_ig": W_ig.sum(0).float().cpu(), "A_ig": A_ig.abs().sum(0).float().cpu(),
                "EW_spi": (cs.Wsum / len(cs.rows)).sum(0).float().cpu(),
                "Abar_spi": (cs.Asum / len(cs.rows)).abs().sum(0).float().cpu()}
    return {"rows": dict(rows), "heat": heat}


# ─────────────────────────────────────────────────────────────────────────────
# chạy
# ─────────────────────────────────────────────────────────────────────────────

def estimate(cfg: dict, per_eval: float) -> dict:
    mc = cfg["mechanism"]
    e1, e2 = mc["e1"], mc["e2"]
    Mmax = max(int(m) for m in e1["M"])
    pc = cfg["paths"]["spi10"]
    n1 = int(e1["ig_steps"]) * (Mmax + 1) + len(e1["spi_paths"]) * int(pc["steps"]) * (Mmax + 1) \
        + int(e1["dp_realizations"]) * len(e1["dp_N"]) * int(e1["dp_points"])
    n2 = int(e2["ig_steps"]) * (int(e2["M"]) + 1) + int(e2["paths"]) * int(pc["steps"]) * int(e2["M"])
    ic = cfg["metrics"]["insdel"]
    ins = (len(ic["bases"]) * (len(e2["K_checkpoints"]) + 1) + (int(e2["paths"]) if e2["per_path_insdel"] else 0)) \
        * 2 * (ic["n_steps"] + 1) / 3
    return {"e1_min": e1["n_images"] * n1 * per_eval / 60, "e2_min": e2["n_images"] * (n2 + ins) * per_eval / 60,
            "e1_evals_per_image": n1, "e2_evals_per_image": n2}


def run(ctx: Ctx, stage: str):
    import pandas as pd
    from .pilot import bench
    cfg = ctx.cfg
    out = mech_out(cfg)
    dg = mech_digest(cfg)
    b = bench(ctx)
    est = estimate(cfg, b["per_eval"])
    print(f"[mech] {b['per_eval'] * 1e3:.2f} ms/eval; ước tính E1 ≈ {est['e1_min']:.1f} phút, "
          f"E2+E3 ≈ {est['e2_min']:.1f} phút", flush=True)
    t_all = time.time()
    if stage == "e1":
        pd.DataFrame(e1_toy(cfg)).to_csv(os.path.join(out, "e1_toy.csv"), index=False)
        print("[mech] e1_toy.csv", flush=True)
    n = int(cfg["mechanism"][stage if stage == "e1" else "e2"]["n_images"])
    idxs = ctx.sp["pilot"][:n]
    names = []
    for j, idx in enumerate(idxs):
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
        obj = {"rows": e1_image(ctx, idx)} if stage == "e1" else e23_image(ctx, j, idx)
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
    fname = {"mech": "mechanism_decomposition", "images": "e2_images"}
    for k, rs in allrows.items():
        if rs:
            pd.DataFrame(rs).to_csv(os.path.join(out, f"{fname.get(k, k)}.csv"), index=False)
            print(f"[mech] {fname.get(k, k)}.csv: {len(rs)} dòng")
    if heats:
        torch.save(heats, os.path.join(out, "e2_heatmaps.pt"))
    mp = os.path.join(out, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    meta[stage] = {"images": names, "bench": b, "estimate": est, "minutes": (time.time() - t_all) / 60,
                   "digest": dg, "debug": cfg["_debug"],
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    meta["mechanism_config"] = cfg["mechanism"]
    with open(mp, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
