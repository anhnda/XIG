"""
E5 — tách "bất đồng bộ" khỏi "chọn feature theo gradient".

P0  Avoided-exposure đã loại co giãn toàn cục:  R^m_i = c_m W^ref_i − W^m_i,  c_m = ΣW^m / ΣW^ref.
    (ΔW = W^ref − W^m của E4 có tương quan = 1 khi hai phương pháp chỉ giảm đồng đều — confound.)
    Mọi bản đồ ở mức pixel (cộng 3 kênh). Mỗi đường đánh giá bằng HAI bộ điểm cầu phương độc lập (a, b):
      - so hai phương pháp m, n: R^m dùng W^ref bộ a, R^n dùng W^ref bộ b → nhiễu của tham chiếu không tạo tương quan giả;
      - trần độ tin cậy: cùng phương pháp, nửa 1 (đường chẵn / bộ a) so với nửa 2 (đường lẻ / bộ b);
      - tương quan riêng phần (partial Spearman) khi khống chế hạng của W^ref;
      - log-ratio log(W^m/W^ref) trên các pixel có W^ref trên trung vị.
P1  Hoán vị lịch trình của Guided IG: Z_k = (X_k − x′)/Δx từ đường Guided đã sinh; Z̃_i = Z_{π(i)}; X̃ = x′ + Z̃⊙Δx.
      bin   : hoán vị trong từng (kênh × thập phân vị |Δx_i|)
      pixel : hoán vị vị trí pixel (giữ 3 kênh đi cùng nhau) trong thập phân vị của Σ_c|Δx|
      block : hoán vị các khối block×block (giữ cấu trúc không gian trong khối; không khống chế |Δx|)
    Đối chứng âm: cùng phép "bin" lên đường SPI (độc lập, đồng phân phối → phân phối đường không đổi), ghép cặp
    với chính các đường SPI đó trước khi hoán vị.
P2  Đường ngẫu nhiên khớp hình học với Guided: Z^λ = (1−λ)t + λ Z^SPI (N = số bước Guided). Độ lệch khỏi dây cung
    tỉ lệ đúng với λ, nên λ = dist_Guided / dist_SPI (chỉ dùng hình học, không nhìn gradient hay IoU).
P3  Tín hiệu cộng tính tuyến tính S: trên mọi đường KHÔNG thích nghi (hoán vị, λ, SPI, IG), canonical giữ đúng 100%
    (A(S) = aR⊙Δx, chính xác với cầu phương phân tầng). Chỉ Guided IG thích nghi có thể mất/lệch credit: chạy lại
    Guided trên F+S và tách phần đổi đường.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import torch

from . import config as C
from .components import Signal, square_region
from .mechanism import aw, decomp, strat_points, summ
from .metrics import energy_mask, insdel, iou, make_base, tie_perm
from .model import ProbFn, TargetFn
from .paths import blur_nodes, cum_nodes, gaussian_blur, guided_nodes, spi_increments, straight_nodes
from .pilot import Ctx, spi_increments_for, sync
from .quad import Counter, fg

F64 = torch.float64


def e5_digest(cfg: dict) -> str:
    keys = ("seed", "model", "data", "paths", "metrics", "e5")
    return __import__("hashlib").sha256(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def e5_out(cfg: dict) -> str:
    d = C.out_dir(cfg, "e5")
    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# lịch trình, hoán vị, hình học
# ─────────────────────────────────────────────────────────────────────────────

def schedule(X: torch.Tensor, x0: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """Z (N+1, C,H,W) đơn điệu, Z_0 = 0, Z_N = 1. Phần tử Δx = 0 nhận lịch tuyến tính (không ảnh hưởng đường gốc,
    nhưng tránh làm hỏng điểm đầu nếu bị hoán vị sang phần tử khác)."""
    N = X.shape[0] - 1
    nz = dx[0] != 0
    Z = torch.where(nz, (X - x0) / torch.where(nz, dx[0], torch.ones_like(dx[0])), torch.zeros_like(X))
    t = torch.linspace(0, 1, N + 1, device=X.device, dtype=X.dtype).view(-1, 1, 1, 1)
    Z = torch.where(nz, Z, t.expand_as(Z)).clamp(0, 1)
    Z = torch.cummax(Z, dim=0).values
    Z[0] = 0
    Z[-1] = 1
    return Z


def _groups(v: torch.Tensor, nbins: int) -> torch.Tensor:
    """Nhãn nhóm: 0 cho v == 0, 1..nbins theo phân vị của các giá trị khác 0."""
    lab = torch.zeros_like(v, dtype=torch.long)
    nz = v > 0
    if int(nz.sum()) > 0:
        qs = torch.quantile(v[nz].double().cpu(), torch.linspace(0, 1, nbins + 1, dtype=F64)[1:-1]).to(v.device)
        lab[nz] = 1 + torch.bucketize(v[nz].double(), qs)
    return lab


def perm_index(kind: str, dx: torch.Tensor, gen: torch.Generator, nbins: int = 10, block: int = 16) -> torch.Tensor:
    """Chỉ số nguồn (C·H·W,): phần tử i nhận lịch của phần tử out[i]."""
    _, Cc, H, W = dx.shape
    n = Cc * H * W
    a = dx[0].abs()
    out = torch.arange(n, device=dx.device)

    def shuffle(members):
        if members.numel() > 1:
            out[members] = members[torch.randperm(members.numel(), generator=gen, device="cpu").to(members.device)]

    if kind == "bin":
        for c in range(Cc):
            lab = _groups(a[c].flatten(), nbins)
            for g in lab.unique():
                shuffle(c * H * W + torch.nonzero(lab == g).flatten())
    elif kind == "pixel":
        lab = _groups(a.sum(0).flatten(), nbins)
        src = torch.arange(H * W, device=dx.device)
        for g in lab.unique():
            m = torch.nonzero(lab == g).flatten()
            if m.numel() > 1:
                src[m] = m[torch.randperm(m.numel(), generator=gen, device="cpu").to(m.device)]
        out = torch.cat([c * H * W + src for c in range(Cc)])
    elif kind == "block":
        nb_h, nb_w = H // block, W // block
        bp = torch.randperm(nb_h * nb_w, generator=gen, device="cpu").to(dx.device)
        ii, jj = torch.meshgrid(torch.arange(H, device=dx.device), torch.arange(W, device=dx.device), indexing="ij")
        bi, bj = ii // block, jj // block
        inside = (bi < nb_h) & (bj < nb_w)
        b = (bi * nb_w + bj).clamp(max=nb_h * nb_w - 1)
        tb = bp[b]
        si = torch.where(inside, (tb // nb_w) * block + ii % block, ii)
        sj = torch.where(inside, (tb % nb_w) * block + jj % block, jj)
        src = (si * W + sj).flatten()
        out = torch.cat([c * H * W + src for c in range(Cc)])
    else:
        raise ValueError(kind)
    return out


def permute(Z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return Z.flatten(1)[:, idx].view_as(Z)


def geom_dist(X: torch.Tensor) -> float:
    """Trung bình theo nút của khoảng cách tới dây cung nối hai đầu / ‖dây cung‖ (chỉ hình học)."""
    c = (X[-1] - X[0]).double().flatten()
    Y = (X - X[0]).double().flatten(1)
    s = (Y @ c) / (c @ c)
    w = Y - s[:, None] * c[None]
    return float((w.norm(dim=1) / c.norm()).mean())


# ─────────────────────────────────────────────────────────────────────────────
# so sánh bản đồ
# ─────────────────────────────────────────────────────────────────────────────

def _rank(v):
    r = torch.empty_like(v, dtype=F64)
    r[v.argsort()] = torch.arange(v.numel(), device=v.device, dtype=F64)
    return r


def _pear(a, b):
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if float(d) > 0 else float("nan")


def sim(Wm, Wn, Ra, Rb, Rfull) -> dict:
    """Wm, Wn: bản đồ pixel của hai phương pháp (hoặc hai nửa); Ra, Rb: tham chiếu độc lập cho từng bên."""
    Wm, Wn, Ra, Rb, Rf = (t.flatten().double() for t in (Wm, Wn, Ra, Rb, Rfull))
    cm, cn = float(Wm.sum() / Ra.sum()), float(Wn.sum() / Rb.sum())
    Rm, Rn = cm * Ra - Wm, cn * Rb - Wn
    rm, rn, r0 = _rank(Rm), _rank(Rn), _rank(Rf)
    s12, s10, s20 = _pear(rm, rn), _pear(rm, r0), _pear(rn, r0)
    den = ((1 - s10 ** 2) * (1 - s20 ** 2)) ** 0.5
    top = Rf > Rf.median()
    lm, ln = torch.log((Wm + 1e-30) / (Ra + 1e-30)), torch.log((Wn + 1e-30) / (Rb + 1e-30))
    return {"c_m": cm, "c_n": cn,
            "spearman_avoided_raw": _pear(_rank(Ra - Wm), _rank(Rb - Wn)),
            "spearman_resid": s12, "pearson_resid": _pear(Rm, Rn),
            "partial_spearman_resid": (s12 - s10 * s20) / den if den > 0 else float("nan"),
            "spearman_logratio_top50": _pear(_rank(lm[top]), _rank(ln[top])),
            "resid_vs_ref_m": s10, "resid_vs_ref_n": s20}


# ─────────────────────────────────────────────────────────────────────────────
# một ảnh
# ─────────────────────────────────────────────────────────────────────────────

class Fam:
    """Tích luỹ một họ đường: A, W (theo phần tử), hai nửa W độc lập (bản đồ pixel), hàng tóm tắt từng đường."""

    def __init__(self, multi: bool):
        self.multi, self.A, self.W, self.h = multi, None, None, [None, None]
        self.rows, self.dist, self.n, self.D = [], [], 0, None

    def add(self, A, W, Wa, Wb, m3, p, dist):
        acc = lambda old, new: new.clone() if old is None else old + new
        self.A, self.W = acc(self.A, A), acc(self.W, W)
        if self.multi:
            k = p % 2
            self.h[k] = acc(self.h[k], (Wa if k == 0 else Wb).sum(0))
        else:
            self.h[0], self.h[1] = acc(self.h[0], Wa.sum(0)), acc(self.h[1], Wb.sum(0))
        self.rows.append(summ(A, W, m3))
        self.dist.append(dist)
        self.n += 1

    def pix(self):
        return (self.W / self.n).sum(0)

    def halves(self):
        """Hai nửa độc lập; None nếu họ nhiều đường mà chỉ có 1 đường."""
        if self.h[1] is None:
            return None
        n0 = (self.n + 1) // 2 if self.multi else self.n
        n1 = self.n // 2 if self.multi else self.n
        return self.h[0] / n0, self.h[1] / n1


def e5_image(ctx: Ctx, j: int, idx: int) -> dict:
    cfg = ctx.cfg
    c = cfg["e5"]
    it, x, mask, cls = ctx.load(idx)
    fn, prob = TargetFn(ctx.bb, cls), ProbFn(ctx.bb, cls)
    x0 = make_base(x, "black")
    dx = x - x0
    m3 = mask.unsqueeze(0).expand_as(x[0])
    perm_t = tie_perm(it.image, x.shape[-1] * x.shape[-2], ctx.seed, ctx.dev)
    ic = cfg["metrics"]["insdel"]
    bases = {b: make_base(x, b, ic["blur_kernel"], ic["blur_sigma"]) for b in ic["bases"]}
    M = int(c["M"])
    smax = float(cfg["paths"]["blur"]["max_sigma"])
    xb = gaussian_blur(x, smax)
    fe, _ = fg(fn, torch.cat([x0, x, xb]), ctx.batch)
    D_black, D_blur = float(fe[1] - fe[0]), float(fe[1] - fe[2])
    fams = {}
    rows = defaultdict(list)
    timing = {}

    def two_sets(X, key):
        N = X.shape[0] - 1
        dX = X[1:] - X[:-1]
        res = []
        for s in ("a", "b"):
            P, _ = strat_points(X, M, ctx.gen("e5q", s, N, M, it.image, *key))
            _, G = fg(fn, P, ctx.batch)
            res.append(aw(G, dX, M))
            del P, G
        (Aa, Wa), (Ab, Wb) = res
        return (Aa + Ab) / 2, (Wa + Wb) / 2, Wa, Wb

    def push(name, res, X, p=0, multi=False, D=None):
        A, W, Wa, Wb = res
        f = fams.setdefault(name, Fam(multi))
        f.add(A, W, Wa, Wb, m3, p, geom_dist(X))
        f.D = D_black if D is None else D

    def add(name, X, key, p=0, multi=False, D=None):
        push(name, two_sets(X, (name,) + tuple(key)), X, p, multi, D)

    t0 = time.time()
    add("ig", straight_nodes(x, x0, int(c["ig_steps"])), (0,))
    add("line_blur", straight_nodes(x, xb, int(c["ig_steps"])), (0,), D=D_blur)
    add("blur", blur_nodes(x, int(c["blur_steps"]), smax), (0,), D=D_blur)
    timing["ig_blur"] = time.time() - t0

    # Guided IG gốc
    t0 = time.time()
    gp = cfg["paths"]["guided"]
    Ng = int(c["guided_steps"])
    Xg, fgd, _, gap = guided_nodes(fn, x, x0, Ng, float(gp["fraction"]), float(gp["max_dist"]), ctx.batch)
    add("guided", Xg, (0,), D=float(fgd[-1] - fgd[0]))
    Zg = schedule(Xg, x0, dx)
    rows["checks"].append({"image": it.image, "guided_end_gap": gap,
                           "guided_sched_recon_err": float((x0 + Zg * dx - Xg).abs().max())})
    timing["guided"] = time.time() - t0

    # P1 — hoán vị lịch trình Guided
    t0 = time.time()
    for kind in c["perm_kinds"]:
        for r in range(int(c["perm_draws"])):
            g = torch.Generator().manual_seed(C.seed_of(ctx.seed, "e5perm", kind, r, it.image))
            ix = perm_index(kind, dx, g, int(c["perm_bins"]), int(c["block"]))
            X = x0 + permute(Zg, ix) * dx
            add(f"guided_perm_{kind}", X, (kind, r), p=r, multi=True)
            del X
    timing["guided_perm"] = time.time() - t0

    # SPI (đường pilot) + đối chứng âm: cùng đường, hoán vị "bin"
    t0 = time.time()
    sp = cfg["paths"]["spi10"]
    N = int(sp["steps"])
    for p in range(int(c["spi_paths"])):
        Z = cum_nodes(spi_increments_for(ctx, sp, dx.shape[1:], N, it.image, p)).to(x.dtype)
        X = x0 + Z * dx
        res = two_sets(X, ("spi", p))
        push("spi", res, X, p=p, multi=True)
        if p < int(c["spi_perm_paths"]):
            push("spi_first", res, X, p=p, multi=True)                       # chính các đường sẽ bị hoán vị
            g = torch.Generator().manual_seed(C.seed_of(ctx.seed, "e5spiperm", p, it.image))
            ix = perm_index("bin", dx, g, int(c["perm_bins"]), int(c["block"]))
            add("spi_perm_bin", x0 + permute(Z, ix) * dx, ("perm", p), p=p, multi=True)
        del Z
    timing["spi"] = time.time() - t0

    # P2 — đường λ khớp độ lệch hình học của Guided
    t0 = time.time()
    dg = geom_dist(Xg)
    tlin = torch.linspace(0, 1, Ng + 1, device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
    for p in range(int(c["lam_paths"])):
        Zs = cum_nodes(spi_increments(float(sp["alpha"]), Ng, dx.shape[1:], ctx.gen("e5lam", Ng, it.image, p))).to(x.dtype)
        ds = geom_dist(x0 + Zs * dx)
        lam = min(1.0, dg / ds) if ds > 0 else 0.0
        X = x0 + ((1 - lam) * tlin + lam * Zs) * dx
        add("spi_lam", X, (p,), p=p, multi=True)
        rows["lam"].append({"image": it.image, "p": p, "lambda": lam, "dist_guided": dg, "dist_spi": ds,
                            "dist_lam": geom_dist(X)})
        del Zs, X
    timing["spi_lam"] = time.time() - t0
    sync()

    # tóm tắt theo họ
    ref = {k: ("line_blur" if k in ("blur", "line_blur") else "ig") for k in fams}
    for name, f in fams.items():
        Ab = f.A / f.n
        row = {"image": it.image, "family": name, "ref": ref[name], "K": f.n, **decomp(f.rows, Ab, f.D, m3),
               "dist_geom": sum(f.dist) / f.n}
        A = Ab.float()
        row.update({"iou": iou(A, mask, perm_t), "energy_mask": energy_mask(A, None, mask)})
        for bn, base in bases.items():
            row[f"insdel_{bn}"] = insdel(prob, x, base, A, ic["n_steps"], ic["batch"])
        rows["families"].append(row)

    # P0 — so bản đồ
    def refs(name):
        r = fams[ref[name]]
        ha, hb = r.halves()
        return ha, hb, r.pix()

    pairs = [tuple(p_) for p_ in c["pairs"] if p_[0] in fams and p_[1] in fams]
    for a_, b_ in pairs:
        ra, _, rfa = refs(a_)
        _, rb, rfb = refs(b_)
        s = sim(fams[a_].pix(), fams[b_].pix(), ra, rb, (rfa + rfb) / 2)
        rows["pairs"].append({"image": it.image, "a": a_, "b": b_, "kind": "cross", **s})
    for name, f in fams.items():
        if name in ("ig", "line_blur"):
            continue
        if f.halves() is None:
            continue
        h0, h1 = f.halves()
        ra, rb, rf = refs(name)
        s = sim(h0, h1, ra, rb, rf)
        rows["pairs"].append({"image": it.image, "a": name, "b": name, "kind": "split_half", **s})

    # P3 — tín hiệu trên Guided thích nghi
    if j < int(c["signal"]["n_images"]):
        t0 = time.time()
        sc = c["signal"]
        gsig = torch.Generator().manual_seed(C.seed_of(ctx.seed, "e5sig", it.image))
        Ag = fams["guided"].A / fams["guided"].n
        for side in sc["sizes"]:
            for loc in sc["locations"]:
                thr = float(sc["inside_min"]) if loc == "inside" else float(sc["outside_max"])
                R = None
                for _ in range(20):
                    R, meta = square_region(mask.cpu(), int(side), loc == "inside", thr, gsig, int(sc["tries"]))
                    if R is None:
                        break
                    R = R.to(ctx.dev)
                    den = float((dx * R).sum())
                    if den / float(R.sum()) >= float(sc["min_denominator"]):
                        break
                    R = None
                if R is None:
                    rows["signal"].append({"image": it.image, "side": side, "loc": loc, "skipped": True})
                    continue
                a = float(sc["B_frac"]) * abs(D_black) / den
                sig = Signal(R, x0, a)
                Xs, _, _, gap_s = guided_nodes(fn.with_extra(sig), x, x0, Ng, float(gp["fraction"]), float(gp["max_dist"]),
                                               ctx.batch)
                AF_new, _, _, _ = two_sets(Xs, ("sig", side, loc))           # A(F; γ′)
                Astar = sig.expected(x, x0)
                AFS_new = AF_new + Astar.double()                              # A(F+S; γ′) = A(F; γ′) + A(S), chính xác
                Rb = R[0].bool()
                tot = float(Astar.double().sum())
                rows["signal"].append({
                    "image": it.image, "side": side, "loc": loc, "skipped": False, "B": tot, "end_gap": gap_s,
                    "retention_guided_adaptive": float((AFS_new - Ag)[Rb].sum()) / tot,
                    "path_part_R": float((AF_new - Ag)[Rb].sum()) / tot,
                    "path_part_norm_rel": float((AF_new - Ag).norm() / Astar.double().norm()),
                    "path_shift_rel": float((Xs - Xg).norm() / (Xg - x0).norm()),
                    "retention_nonadaptive": 1.0})
                del Xs
        timing["signal"] = time.time() - t0

    heat = None
    if j < int(c.get("heatmap_images", 3)):
        heat = {"x": it.x.half(), "mask": mask.cpu()}
        for name in ("spi", "guided", "guided_perm_bin", "spi_lam"):
            if name in fams:
                r = fams[ref[name]].pix()
                w = fams[name].pix()
                heat[f"R_{name}"] = (float(w.sum() / r.sum()) * r - w).float().cpu()
    rows["images"].append({"image": it.image, "index": it.index, "cls": cls, "D_black": D_black, "D_blur": D_blur,
                           "mask_area": float(mask.float().mean()), "timing": json.dumps(timing)})
    return {"rows": dict(rows), "heat": heat}


def estimate(cfg: dict, per_eval: float, per_seq: float) -> dict:
    c = cfg["e5"]
    M2 = 2 * int(c["M"])
    Ng, Ni, Nb = int(c["guided_steps"]), int(c["ig_steps"]), int(c["blur_steps"])
    N = int(cfg["paths"]["spi10"]["steps"])
    n = 2 * Ni * M2 + Nb * M2 + Ng * M2 + len(c["perm_kinds"]) * int(c["perm_draws"]) * Ng * M2 \
        + int(c["spi_paths"]) * N * M2 + int(c["spi_perm_paths"]) * N * M2 + int(c["lam_paths"]) * Ng * M2
    ic = cfg["metrics"]["insdel"]
    n_fam = 7 + len(c["perm_kinds"])
    ins = n_fam * len(ic["bases"]) * 2 * (ic["n_steps"] + 1) / 3
    sig = len(c["signal"]["sizes"]) * len(c["signal"]["locations"])
    main = c["n_images"] * ((n + ins) * per_eval + (Ng + 1) * per_seq)
    main += c["signal"]["n_images"] * sig * ((Ng + 1) * per_seq + Ng * M2 * per_eval)
    return {"e5_min": main / 60, "evals_per_image": n}


def run(ctx: Ctx):
    import pandas as pd
    from .pilot import bench
    cfg = ctx.cfg
    out = e5_out(cfg)
    dg = e5_digest(cfg)
    b = bench(ctx)
    est = estimate(cfg, b["per_eval"], b["per_seq"])
    print(f"[e5] {b['per_eval'] * 1e3:.2f} ms/eval; ước tính ≈ {est['e5_min']:.1f} phút "
          f"({est['evals_per_image']} eval theo lô mỗi ảnh)", flush=True)
    t_all = time.time()
    n = int(cfg["e5"]["n_images"])
    names = []
    for j, idx in enumerate(ctx.sp["pilot"][:n]):
        image = f"seg{idx:05d}"
        names.append(image)
        cd = os.path.join(out, "cache")
        os.makedirs(cd, exist_ok=True)
        cp = os.path.join(cd, f"{image}.pt")
        if os.path.exists(cp) and torch.load(cp, weights_only=False).get("digest") == dg:
            print(f"  [e5] {image}: đã có trong cache, bỏ qua")
            continue
        t0, n0 = time.time(), Counter.n
        print(f"  [e5] ảnh {j + 1}/{n} {image}", flush=True)
        obj = e5_image(ctx, j, idx)
        obj["digest"] = dg
        torch.save(obj, cp)
        print(f"  [e5] {image}: {Counter.n - n0} eval, {time.time() - t0:.0f}s; tổng {(time.time() - t_all) / 60:.1f} phút",
              flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    allrows, heats = defaultdict(list), {}
    for im in names:
        obj = torch.load(os.path.join(out, "cache", f"{im}.pt"), weights_only=False)
        for k, rs in obj["rows"].items():
            allrows[k] += rs
        if obj.get("heat") is not None:
            heats[im] = obj["heat"]
    for k, rs in allrows.items():
        if rs:
            pd.DataFrame(rs).to_csv(os.path.join(out, f"e5_{k}.csv"), index=False)
            print(f"[e5] e5_{k}.csv: {len(rs)} dòng")
    if heats:
        torch.save(heats, os.path.join(out, "e5_heatmaps.pt"))
    mp = os.path.join(out, "meta.json")
    meta = {"images": names, "bench": b, "estimate": est, "minutes": (time.time() - t_all) / 60, "digest": dg,
            "debug": cfg["_debug"], "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "e5_config": cfg["e5"]}
    with open(mp, "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
