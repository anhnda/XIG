"""
E9 — từ hiện tượng sang cơ chế.

A1  Bất biến global-shared ≡ IG. Đường global: X(t) = x′ + Z(t)Δx, Z là MỘT lịch trình DP cho mọi phần tử → cùng tập điểm
    hình học với IG, chỉ khác tham số hoá. Tích phân canonical theo toạ độ tiến độ s = Z(t):
        A = ∫₀¹ ∇F(x′ + sΔx) ds ⊙ Δx         (không phụ thuộc Z)
    Code chứng minh: (i) điểm cầu phương phân tầng đều theo s, ánh xạ ngược s → t qua lịch trình gấp khúc, cho ĐÚNG các điểm
    và trọng số của IG (sai khác chỉ do làm tròn); (ii) ước lượng kiểu E8 (phân tầng theo đoạn của lưới Z, M điểm/đoạn) hội tụ
    về cùng tham chiếu IG dày khi M tăng — sai lệch ở E8 là do đoạn dài (bước nhảy DP) chỉ có M điểm.
    Dùng lại đúng các đường global và seed của E8 trên vài ảnh confirm.
A2  H1 (rgb − indep) trên vài ảnh confirm, cùng realization và seed của E8, ở lưới mịn N_fine (gấp 4 số eval) so với N của E8.
B   RGB coupling q: mỗi pixel, với xác suất q, ba kênh dùng chung một lịch trình DP; còn lại ba lịch trình độc lập.
    Mọi phần tử giữ biên DP. Common random numbers qua q: với mỗi đường rút sẵn Z_ind (mỗi phần tử), Z_pix (mỗi pixel),
    U_pix ~ U(0,1); ở mức q, pixel dùng Z_pix nếu U_pix < q (ghép đơn điệu), cùng điểm cầu phương.
    Locality: nếu không có tương tác giữa các pixel KHÁC nhau, thì trên từng đường A(q)_pixel = A(1)_pixel nếu pixel chia sẻ,
    = A(0)_pixel nếu không. Phần dư R(q) = ‖A(q) − [s⊙A(1) + (1−s)⊙A(0)]‖₁ / ‖A(1) − A(0)‖₁ đo ảnh hưởng của việc ghép kênh
    ở pixel khác lên credit của pixel này (so với mức nhiễu cầu phương đo bằng một bộ điểm độc lập).
    Tương tác hữu hạn sai phân trên lưới patch: I(u,v; z) = F(z+u+v) − F(z+u) − F(z+v) + F(z), z = x′ + tΔx,
    u, v = s·Δx giới hạn trên (kênh, patch): cặp kênh trong cùng patch (RGB), cùng kênh patch kề, cùng kênh patch xa.
    Liên hệ với |Ā(1) − Ā(0)| theo patch (Spearman trong ảnh, riêng phần khi khống chế độ lớn bậc một và |Ā(0)|).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import torch

from . import config as C
from .e5 import geom_dist
from .e7 import group_labels
from .mechanism import aw, aw_stream, decomp, strat_points, summ
from .metrics import energy_mask, insdel, iou, make_base, tie_perm
from .model import ProbFn, TargetFn
from .paths import cum_nodes, spi_increments, straight_nodes
from .pilot import Ctx, sync
from .quad import Counter, fg

F64 = torch.float64


def e9_digest(cfg: dict) -> str:
    keys = ("seed", "model", "data", "paths", "metrics", "e8", "e9")
    return __import__("hashlib").sha256(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def e9_out(cfg: dict) -> str:
    d = C.out_dir(cfg, "e9")
    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# A1: tích phân theo toạ độ tiến độ
# ─────────────────────────────────────────────────────────────────────────────

def s_uniform_points(Zs: torch.Tensor, x0, dx, T: int, gen) -> tuple[torch.Tensor, torch.Tensor]:
    """Zs (N+1,) lịch trình global (đơn điệu, 0 → 1). Điểm phân tầng đều theo s (T tầng), ánh xạ ngược s → t trên lịch trình
    gấp khúc rồi dựng X(t) = x′ + Z(t)Δx. Trả P (T, …) và s_đạt (T,) (= Z(t(s)), phải trùng s)."""
    U = torch.rand(T, generator=gen, device=gen.device, dtype=F64).to(Zs.device)
    s = (torch.arange(T, device=Zs.device, dtype=F64) + U) / T
    Z = Zs.double()
    k = torch.clamp(torch.searchsorted(Z, s, right=True) - 1, 0, len(Z) - 2)
    w = torch.where(Z[k + 1] > Z[k], (s - Z[k]) / (Z[k + 1] - Z[k]), torch.zeros_like(s))
    t_of_s = (k.double() + w) / (len(Z) - 1)                       # t trên lưới nút đều
    s_hat = Z[k] + w * (Z[k + 1] - Z[k])                           # Z(t(s)) bằng nội suy gấp khúc
    P = x0 + s_hat.view(-1, 1, 1, 1).to(dx.dtype) * dx
    return P, s_hat, s, t_of_s


def ig_uniform_points(x0, dx, T: int, gen):
    U = torch.rand(T, generator=gen, device=gen.device, dtype=F64).to(dx.device)
    s = (torch.arange(T, device=dx.device, dtype=F64) + U) / T
    return x0 + s.view(-1, 1, 1, 1).to(dx.dtype) * dx, s


# ─────────────────────────────────────────────────────────────────────────────
# B: lịch trình ghép kênh theo q, tương tác
# ─────────────────────────────────────────────────────────────────────────────

def q_schedule(Zi: torch.Tensor, Zp: torch.Tensor, U: torch.Tensor, q: float):
    """Zi (N+1, C,H,W), Zp (N+1, 1,H,W), U (H,W). Pixel chia sẻ (U < q) dùng Zp cho cả ba kênh."""
    s = (U < q).view(1, 1, *U.shape)
    return torch.where(s, Zp.expand_as(Zi), Zi), (U < q)


def patch_masks(H: int, W: int, g: int, device):
    ph, pw = H // g, W // g
    out = []
    for i in range(g):
        for j in range(g):
            m = torch.zeros(H, W, dtype=torch.bool, device=device)
            m[i * ph:(i + 1) * ph if i < g - 1 else H, j * pw:(j + 1) * pw if j < g - 1 else W] = True
            out.append(m)
    return out


@torch.no_grad()
def interactions(fn, x0, dx, g: int, ts, step: float, batch: int) -> list:
    """Theo patch: |I| trung bình qua t cho cặp kênh trong patch (rgb), cùng kênh patch kề phải (adj), cùng kênh patch đối
    xứng xa (far); và độ lớn bậc một |F(z+u) − F(z)| trung bình qua kênh."""
    _, Cc, H, W = dx.shape
    masks = patch_masks(H, W, g, dx.device)
    P = len(masks)
    pairs_rgb = [(0, 1), (0, 2), (1, 2)]
    res = [defaultdict(list) for _ in range(P)]

    def F(X):
        out = []
        for i in range(0, X.shape[0], batch):
            out.append(fn(X[i:i + batch]).double())
        return torch.cat(out)

    def u_of(c, m):
        u = torch.zeros_like(dx)
        u[:, c][:, m] = step * dx[:, c][:, m]
        return u

    for t in ts:
        z = x0 + float(t) * dx
        singles = {(c, p): u_of(c, masks[p]) for p in range(P) for c in range(Cc)}
        keys = list(singles)
        F0 = float(F(z))
        Fs = F(torch.cat([z + singles[k] for k in keys]))
        fs = {k: float(v) - F0 for k, v in zip(keys, Fs)}
        jobs = []
        for p in range(P):
            i, j = divmod(p, g)
            for c, c2 in pairs_rgb:
                jobs.append(("rgb", p, (c, p), (c2, p)))
            q_adj = i * g + (j + 1) if j + 1 < g else None
            q_far = ((i + g // 2) % g) * g + (j + g // 2) % g
            for c in range(Cc):
                if q_adj is not None:
                    jobs.append(("adj", p, (c, p), (c, q_adj)))
                jobs.append(("far", p, (c, p), (c, q_far)))
        Fj = F(torch.cat([z + singles[a] + singles[b] for _, _, a, b in jobs]))
        for (kind, p, a, b), v in zip(jobs, Fj):
            I = float(v) - fs[a] - fs[b] - F0
            res[p][kind].append(abs(I))
        for p in range(P):
            res[p]["first"].append(sum(abs(fs[(c, p)]) for c in range(Cc)) / Cc)
    return [{k: sum(v) / len(v) for k, v in r.items()} for r in res], masks


# ─────────────────────────────────────────────────────────────────────────────
# một ảnh
# ─────────────────────────────────────────────────────────────────────────────

class Ev:
    def __init__(self, ctx, x, mask, cls, image):
        self.ctx, self.x, self.mask, self.image = ctx, x, mask, image
        self.fn, self.prob = TargetFn(ctx.bb, cls), ProbFn(ctx.bb, cls)
        self.x0 = make_base(x, "black")
        self.dx = x - self.x0
        self.m3 = mask.unsqueeze(0).expand_as(x[0])
        self.perm = tie_perm(image, x.shape[-1] * x.shape[-2], ctx.seed, ctx.dev)
        ic = ctx.cfg["metrics"]["insdel"]
        self.ic = ic
        self.bases = {b: make_base(x, b, ic["blur_kernel"], ic["blur_sigma"]) for b in ic["bases"]}
        fe, _ = fg(self.fn, torch.cat([self.x0, x]), ctx.batch)
        self.D = float(fe[1] - fe[0])

    def quality(self, A, full=True):
        A = A.float()
        q = {"iou": iou(A, self.mask, self.perm), "energy_mask": energy_mask(A, None, self.mask)}
        if full:
            for bn, base in self.bases.items():
                q[f"insdel_{bn}"] = insdel(self.prob, self.x, base, A, self.ic["n_steps"], self.ic["batch"])
        return q

    def strat(self, X, M, *key):
        """Cùng số ngẫu nhiên và cùng điểm với strat_points (tái tạo đúng seed của E8), tính theo khối để vừa VRAM."""
        return aw_stream(self.fn, X, M, self.ctx.gen(*key), self.ctx.batch)


def a_image(ctx: Ctx, idx: int, do_a1: bool, do_h1: bool) -> dict:
    cfg = ctx.cfg
    e8, a = cfg["e8"], cfg["e9"]["A"]
    it, x, mask, cls = ctx.load(idx)
    E = Ev(ctx, x, mask, cls, it.image)
    N, Nf, M8, alpha = int(e8["N"]), int(e8["N_fine"]), int(e8["M"]), float(e8["alpha"])
    rows = defaultdict(list)
    Cs = tuple(E.dx.shape[1:])

    def e8_path(kind, p, n_nodes):
        gc = torch.Generator().manual_seed(C.seed_of(ctx.seed, "e8part", kind, it.image, p))
        lab, n = group_labels(kind, Cs, gc)
        Zf = cum_nodes(spi_increments(alpha, Nf, (n,), ctx.gen("e8S", kind, Nf, it.image, p)))
        Zc = Zf[::Nf // n_nodes]
        return Zc, lab, n

    if not do_a1:
        return h1_part(ctx, it, x, E, rows, Cs) if do_h1 else dict(rows)
    # tham chiếu IG dày
    Xr = straight_nodes(x, E.x0, int(a["ref_N"]))
    Aref, _ = E.strat(Xr, int(a["ref_M"]), "e9Aref", it.image)
    nref = float(Aref.abs().sum())
    qref = E.quality(Aref)

    def row(est, fam, p, A, points, extra=None, q=True):
        r = {"image": it.image, "part": "A1", "estimator": est, "family": fam, "p": p, "points": points, "D": E.D,
             "sumA": float(A.sum()), "dist_ref_l1": float((A - Aref).abs().sum()) / nref,
             "dist_ref_l2": float((A - Aref).norm() / Aref.norm()), **(extra or {})}
        if q:
            qq = E.quality(A)
            r.update(qq)
            r.update({f"d_{k}": v - qref[k] for k, v in qq.items()})
        rows["A"].append(r)

    row("ig_ref", "ig", 0, Aref, int(a["ref_N"]) * int(a["ref_M"]))
    Xig = straight_nodes(x, E.x0, N)
    for s_ in ("a", "b"):                                            # đúng IG của E8
        A, _ = E.strat(Xig, M8, "e8q", N, M8, it.image, "ig", s_)
        row(f"e8_ig_{s_}", "ig", 0, A, N * M8)
    for M in a["M"]:
        A, _ = E.strat(Xig, int(M), "e9Aig", N, int(M), it.image)
        row(f"node_M{M}", "ig", 0, A, N * int(M))

    for p in range(int(a["K"])):
        Zc, lab, _ = e8_path("global", p, N)
        Zs = Zc[:, 0]                                                # một lịch trình
        Xg = E.x0 + Zc[:, lab.to(Zc.device)].view((N + 1,) + Cs).to(x.dtype) * E.dx
        A, _ = E.strat(Xg, M8, "e8q", N, M8, it.image, "global", p)      # đúng ước lượng E8
        lens = (Zs[1:] - Zs[:-1]).double()
        row("e8_global", "global", p, A, N * M8, {"max_seg_len": float(lens.max()), "seg_len_cv": float(lens.std() / lens.mean())})
        for M in a["M"]:
            A, _ = E.strat(Xg, int(M), "e9Ag", N, int(M), it.image, p)
            row(f"node_M{M}", "global", p, A, N * int(M))
        # tích phân theo s: cùng seed với một IG đều theo s → phải cho đúng cùng điểm
        T = N * M8
        Pg, s_hat, s, _ = s_uniform_points(Zs, E.x0, E.dx, T, ctx.gen("e9As", it.image, p))
        Pi, s_i = ig_uniform_points(E.x0, E.dx, T, ctx.gen("e9As", it.image, p))
        _, Gg = fg(E.fn, Pg, ctx.batch)
        _, Gi = fg(E.fn, Pi, ctx.batch)
        Ag = (Gg.double().mean(0) * E.dx[0].double())
        Ai = (Gi.double().mean(0) * E.dx[0].double())
        row("s_uniform", "global", p, Ag, T, {"max_abs_s_err": float((s_hat - s).abs().max()),
                                             "max_abs_point_diff": float((Pg - Pi).abs().max()),
                                             "A_rel_diff_vs_ig_same_points": float((Ag - Ai).abs().sum() / Ai.abs().sum())})
        row("s_uniform", "ig", p, Ai, T, q=False)
        del Xg, Pg, Pi, Gg, Gi

    if do_h1:
        h1_part(ctx, it, x, E, rows, Cs)
    sync()
    return dict(rows)


def h1_part(ctx, it, x, E, rows, Cs):
    cfg = ctx.cfg
    e8 = cfg["e8"]
    N, Nf, M8, alpha = int(e8["N"]), int(e8["N_fine"]), int(e8["M"]), float(e8["alpha"])
    if True:
        h = cfg["e9"]["H1"]
        for kind in ("indep", "rgb"):
            for p in range(int(e8["K"])):
                gc = torch.Generator().manual_seed(C.seed_of(ctx.seed, "e8part", kind, it.image, p))
                lab, n = group_labels(kind, Cs, gc)
                Zf = cum_nodes(spi_increments(alpha, Nf, (n,), ctx.gen("e8S", kind, Nf, it.image, p)))
                lab = lab.to(Zf.device)
                Xc = E.x0 + Zf[::Nf // N][:, lab].view((N + 1,) + Cs).to(x.dtype) * E.dx
                A, W = E.strat(Xc, M8, "e8q", N, M8, it.image, kind, p)            # đúng E8
                rows["H1"].append({"image": it.image, "level": f"N{N}_M{M8}", "family": kind, "p": p, "evals": N * M8,
                                   **summ(A, W, E.m3), **E.quality(A)})
                del Xc
                Xf = E.x0 + Zf[:, lab].view((Nf + 1,) + Cs).to(x.dtype) * E.dx
                A, W = E.strat(Xf, int(h["M"]), "e9H1", Nf, int(h["M"]), it.image, kind, p)
                rows["H1"].append({"image": it.image, "level": f"N{Nf}_M{h['M']}", "family": kind, "p": p,
                                   "evals": Nf * int(h["M"]), **summ(A, W, E.m3), **E.quality(A)})
                del Xf, Zf
        Xi = straight_nodes(x, E.x0, N)
        A, W = E.strat(Xi, M8, "e8q", N, M8, it.image, "ig", "a")
        rows["H1"].append({"image": it.image, "level": f"N{N}_M{M8}", "family": "ig", "p": 0, "evals": N * M8,
                           **summ(A, W, E.m3), **E.quality(A)})
        Xi = straight_nodes(x, E.x0, Nf)
        A, W = E.strat(Xi, int(h["M"]), "e9H1", Nf, int(h["M"]), it.image, "ig")
        rows["H1"].append({"image": it.image, "level": f"N{Nf}_M{h['M']}", "family": "ig", "p": 0, "evals": Nf * int(h["M"]),
                           **summ(A, W, E.m3), **E.quality(A)})
    return dict(rows)


def b_image(ctx: Ctx, idx: int) -> dict:
    cfg = ctx.cfg
    b = cfg["e9"]["B"]
    it, x, mask, cls = ctx.load(idx)
    E = Ev(ctx, x, mask, cls, it.image)
    N, M, K, alpha = int(b["N"]), int(b["M"]), int(b["K"]), float(b["alpha"])
    qs = [float(q) for q in b["q"]]
    _, Cc, H, W = E.dx.shape
    rows = defaultdict(list)
    t0 = time.time()
    Xig = straight_nodes(x, E.x0, N)
    A_ig, W_ig = E.strat(Xig, M, "e9Bq", "ig", N, M, it.image)
    s_ig = summ(A_ig, W_ig, E.m3)
    rows["q_paths"].append({"image": it.image, "q": -1.0, "family": "ig", "p": 0, "D": E.D, **s_ig,
                            "within": s_ig["W"] - s_ig["L1"], **E.quality(A_ig)})
    acc = {q: {"A": None, "rows": []} for q in qs}
    half = {q: [None, None] for q in qs}
    dA_half = [None, None]
    for p in range(K):
        Zi = cum_nodes(spi_increments(alpha, N, (Cc, H, W), ctx.gen("e9Bi", N, it.image, p)))
        Zp = cum_nodes(spi_increments(alpha, N, (1, H, W), ctx.gen("e9Bp", N, it.image, p)))
        U = torch.rand(H, W, generator=ctx.gen("e9BU", it.image, p), device=ctx.dev, dtype=F64)
        Ap = {}
        shared_of = {}
        for q in qs:
            Z, sh = q_schedule(Zi, Zp, U, q)
            X = E.x0 + Z.to(x.dtype) * E.dx
            A, Wm = E.strat(X, M, "e9Bq", N, M, it.image, p)                       # cùng điểm cầu phương qua q (CRN)
            s = summ(A, Wm, E.m3)
            rows["q_paths"].append({"image": it.image, "q": q, "family": "rgbq", "p": p, "D": E.D, **s,
                                    "within": s["W"] - s["L1"], "shared_frac": float(sh.float().mean()),
                                    "dist_geom": geom_dist(X), **E.quality(A, full=bool(b["per_path_insdel"]))})
            acc[q]["A"] = A.clone() if acc[q]["A"] is None else acc[q]["A"] + A
            acc[q]["rows"].append(s)
            k = p % 2
            half[q][k] = A.clone() if half[q][k] is None else half[q][k] + A
            Ap[q], shared_of[q] = A, sh
            del X
        # nhiễu cầu phương: q = qs[0] với bộ điểm độc lập
        Z0, _ = q_schedule(Zi, Zp, U, qs[0])
        A0b, _ = E.strat(E.x0 + Z0.to(x.dtype) * E.dx, M, "e9Bq2", N, M, it.image, p)
        q0, q1 = qs[0], qs[-1]
        span = float((Ap[q1] - Ap[q0]).abs().sum())
        rec = {"image": it.image, "p": p, "span_l1": span, "quad_noise_ratio": float((A0b - Ap[q0]).abs().sum()) / max(span, 1e-30)}
        for q in qs[1:-1]:
            sm = shared_of[q].view(1, H, W).to(Ap[q1].dtype)
            pred = sm * Ap[q1] + (1 - sm) * Ap[q0]
            rec[f"locality_resid_q{q:g}"] = float((Ap[q] - pred).abs().sum()) / max(span, 1e-30)
        rows["locality"].append(rec)
        d = Ap[q1] - Ap[q0]
        dA_half[p % 2] = d.clone() if dA_half[p % 2] is None else dA_half[p % 2] + d
        del Zi, Zp, Z0
    t_paths = time.time() - t0

    # ensemble theo q, tuyến tính theo q của bản đồ trung bình
    Abar = {q: acc[q]["A"] / K for q in qs}
    span = float((Abar[qs[-1]] - Abar[qs[0]]).abs().sum())
    for q in qs:
        r = {"image": it.image, "q": q, "K": K, **decomp(acc[q]["rows"], Abar[q], E.D, E.m3), **E.quality(Abar[q])}
        lin = (1 - q) * Abar[qs[0]] + q * Abar[qs[-1]]
        r["linearity_resid"] = float((Abar[q] - lin).abs().sum()) / max(span, 1e-30)
        r["W_ig"] = s_ig["W"]
        rows["q_ens"].append(r)

    # tương tác theo patch và thay đổi credit theo patch
    t0 = time.time()
    it_ = b["interaction"]
    inter, masks = interactions(E.fn, E.x0, E.dx, int(it_["grid"]), it_["t"], float(it_["step"]), ctx.batch)
    dmap = (Abar[qs[-1]] - Abar[qs[0]]).abs().sum(0)
    h1 = (dA_half[0] / max(1, (K + 1) // 2)).abs().sum(0)
    h2 = (dA_half[1] / max(1, K // 2)).abs().sum(0) if dA_half[1] is not None else None
    a0 = Abar[qs[0]].abs().sum(0)
    for pi, (m, rec) in enumerate(zip(masks, inter)):
        rows["patches"].append({"image": it.image, "patch": pi, "dA": float(dmap[m].sum()), "dA_h1": float(h1[m].sum()),
                                "dA_h2": float(h2[m].sum()) if h2 is not None else float("nan"),
                                "absA0": float(a0[m].sum()), "mask_frac": float(mask[m].float().mean()),
                                "I_rgb": rec.get("rgb", float("nan")), "I_adj": rec.get("adj", float("nan")),
                                "I_far": rec.get("far", float("nan")), "first": rec.get("first", float("nan"))})
    rows["images"].append({"image": it.image, "index": it.index, "cls": cls, "D": E.D, "mask_area": float(mask.float().mean()),
                           "sec_paths": t_paths, "sec_interaction": time.time() - t0})
    sync()
    return dict(rows)


def estimate(cfg: dict, per_eval: float) -> dict:
    e8, c = cfg["e8"], cfg["e9"]
    a, h, b = c["A"], c["H1"], c["B"]
    ic = cfg["metrics"]["insdel"]
    ins = len(ic["bases"]) * 2 * (ic["n_steps"] + 1) / 3
    N, M8 = int(e8["N"]), int(e8["M"])
    nA = int(a["ref_N"]) * int(a["ref_M"]) + 2 * N * M8 + sum(N * int(m) for m in a["M"]) \
        + int(a["K"]) * (N * M8 + sum(N * int(m) for m in a["M"]) + 2 * N * M8) + (4 + len(a["M"]) * (1 + int(a["K"])) + 2 * int(a["K"])) * ins
    nH = 2 * int(e8["K"]) * (N * M8 + int(e8["N_fine"]) * int(h["M"]) + 2 * ins) + N * M8 + int(e8["N_fine"]) * int(h["M"])
    g = int(b["interaction"]["grid"]) ** 2
    nB = int(b["N"]) * int(b["M"]) * (1 + int(b["K"]) * (len(b["q"]) + 1)) \
        + (len(b["q"]) + 1 + (int(b["K"]) * len(b["q"]) if b["per_path_insdel"] else 0)) * ins \
        + len(b["interaction"]["t"]) * (1 + 3 * g + 3 * g + 6 * g) / 3
    t = (a["n_images"] * nA + h["n_images"] * nH + b["n_images"] * nB) * per_eval
    return {"e9_min": t / 60, "evals_A": nA, "evals_H1": nH, "evals_B": nB}


def run(ctx: Ctx):
    import pandas as pd
    from .pilot import bench
    cfg = ctx.cfg
    c = cfg["e9"]
    out = e9_out(cfg)
    dg = e9_digest(cfg)
    bnc = bench(ctx)
    est = estimate(cfg, bnc["per_eval"])
    print(f"[e9] {bnc['per_eval'] * 1e3:.2f} ms/eval; ước tính ≈ {est['e9_min']:.1f} phút", flush=True)
    t_all = time.time()
    cd = os.path.join(out, "cache")
    os.makedirs(cd, exist_ok=True)
    jobs = []
    nA, nH = int(c["A"]["n_images"]), int(c["H1"]["n_images"])
    for j, idx in enumerate(ctx.sp[c["A"]["split"]][:max(nA, nH)]):
        jobs.append(("A", idx, j < nA, j < nH))
    for idx in ctx.sp[c["B"]["split"]][:int(c["B"]["n_images"])]:
        jobs.append(("B", idx, False, True))
    names = []
    for part, idx, doA, h1 in jobs:
        image = f"seg{idx:05d}"
        cp = os.path.join(cd, f"{part}_{image}.pt")
        names.append(cp)
        if os.path.exists(cp) and torch.load(cp, weights_only=False).get("digest") == dg:
            print(f"  [e9] {part} {image}: đã có trong cache, bỏ qua")
            continue
        t0, n0 = time.time(), Counter.n
        print(f"  [e9] {part} {image}", flush=True)
        if part == "A":
            rows = a_image(ctx, idx, doA, h1)
        else:
            rows = b_image(ctx, idx)
        torch.save({"rows": rows, "digest": dg}, cp)
        print(f"  [e9] {part} {image}: {Counter.n - n0} eval, {time.time() - t0:.0f}s; tổng {(time.time() - t_all) / 60:.1f} phút",
              flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    allrows = defaultdict(list)
    for cp in names:
        for k, rs in torch.load(cp, weights_only=False)["rows"].items():
            allrows[k] += rs
    for k, rs in allrows.items():
        if rs:
            pd.DataFrame(rs).to_csv(os.path.join(out, f"e9_{k}.csv"), index=False)
            print(f"[e9] e9_{k}.csv: {len(rs)} dòng")
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"bench": bnc, "estimate": est, "minutes": (time.time() - t_all) / 60, "digest": dg, "debug": cfg["_debug"],
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "e9_config": c}, fh,
                  indent=1, default=str)