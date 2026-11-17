"""
OCS — Output Convergence Study: IG, SPI, Guided IG, BlurIG có cho cùng một lời giải thích không, và phần chung có giá trị không.

Bản đồ: A (C,H,W) có dấu; |A| theo pixel = Σ_c|A|; có dấu theo pixel = Σ_c A.
Phương pháp: ig (đen → x), spi (đen → x, DP α), guided (đen → x), blur (BlurIG: blur → x) và line_blur (IG từ CÙNG điểm đầu mờ
→ x: đối chứng baseline, để tách hiệu ứng đường của BlurIG khỏi hiệu ứng điểm đầu). Bản đồ tham chiếu đơn giản: grad (|∇F(x)|),
gxi (∇F(x)⊙Δx), edge (Sobel của ảnh xám), center (Gauss tâm ảnh).

Bốn tầng tương đồng (hàm `sim`):
  ranking   Spearman |A| và có dấu (pixel và phần tử), Kendall τ-b ở 56×56, Spearman ở 56 và 14
  top       Jaccard top-k% (1, 5, 10, 20) theo |A| pixel; RBO (rank-biased overlap, mở rộng ngoại suy)
  signed    cosine (phần tử, pixel có dấu), ‖Ã − B̃‖₁ với Ã = A/‖A‖₁, tỉ lệ trùng dấu trên hợp top-10% phần tử, trùng dấu có trọng số
  spatial   chồng lấn top-10% trong bán kính r (đối xứng), sliced Wasserstein-1 giữa phân phối khối lượng |A| (56×56, đơn vị pixel 224)

Trong-phương-pháp: hai ước lượng độc lập — đường tất định: hai bộ điểm cầu phương; SPI: hai tập đường rời nhau, ở nhiều K.
So chéo dùng nửa 1 của mỗi phương pháp; trần = √(within_a · within_b); tỉ số = cross / trần.
Null: so a với b bị xáo khối 16×16 (giữ phân phối giá trị và cấu trúc cục bộ, phá vị trí).

Consensus: T_m = top q% pixel của m ∈ consensus_methods (bản đồ đầy đủ). C = ∩ T_m. So hiệu ứng xoá/chèn (logit và xác suất lớp
đích, baseline blur và đen) của C với tập đối chứng CÙNG kích thước: mỗi phương pháp m, chọn từ T_m \\ C theo phân tầng thập phân vị
hạng trong T_m khớp với C (khớp mức attribution theo m), nhiều lần rút; thêm tập riêng U_m và tập ngẫu nhiên. Báo độ co cụm không
gian (tỉ lệ láng giềng 4 cùng thuộc tập) của mọi tập.
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
import torch.nn.functional as Fnn

from . import config as C
from .cross import _corr, _rank, kendall_tau_b
from .mechanism import aw_stream
from .metrics import make_base, tie_perm
from .model import TargetFn
from .paths import blur_nodes, gaussian_blur, guided_nodes, spi_increments, spi_nodes_from, straight_nodes
from .pilot import Ctx, spi_increments_for, sync
from .quad import Counter, fg

F64 = torch.float64
MAIN = ("ig", "spi", "guided", "blur", "line_blur")
REFS = ("grad", "gxi", "edge", "center")


def ocs_digest(cfg: dict) -> str:
    keys = ("seed", "model", "data", "paths", "metrics", "ocs")
    return hashlib.sha256(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def ocs_out(cfg: dict) -> str:
    d = C.out_dir(cfg, "ocs_" + cfg["ocs"]["split"])
    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# bản đồ và các thước đo tương đồng
# ─────────────────────────────────────────────────────────────────────────────

class Map:
    def __init__(self, A: torch.Tensor | None = None, pix_abs=None, signed=True):
        if A is not None:
            self.A = A.double()
            self.abs = self.A.abs().sum(0)
            self.sgn = self.A.sum(0)
        else:
            self.A, self.abs, self.sgn = None, pix_abs.double(), None
        self.signed = signed and A is not None


def top_mask(v: torch.Tensor, frac: float, perm: torch.Tensor) -> torch.Tensor:
    v = v.flatten()
    k = max(1, int(round(frac * v.numel())))
    order = perm[torch.argsort(v[perm], descending=True, stable=True)]
    m = torch.zeros(v.numel(), dtype=torch.bool, device=v.device)
    m[order[:k]] = True
    return m


def rbo_ext(a: torch.Tensor, b: torch.Tensor, p: float, perm) -> float:
    """RBO mở rộng (Webber et al. 2010), xếp hạng giảm dần của a và b trên cùng tập phần tử (độ sâu = n)."""
    a, b = a.flatten(), b.flatten()
    n = a.numel()
    oa = perm[torch.argsort(a[perm], descending=True, stable=True)]
    ob = perm[torch.argsort(b[perm], descending=True, stable=True)]
    pa = torch.empty(n, dtype=torch.long, device=a.device); pa[oa] = torch.arange(n, device=a.device)
    pb = torch.empty(n, dtype=torch.long, device=a.device); pb[ob] = torch.arange(n, device=a.device)
    depth_both = torch.maximum(pa, pb)                                   # phần tử có mặt ở cả hai danh sách từ độ sâu này
    X = torch.cumsum(torch.bincount(depth_both, minlength=n).double(), 0)   # X_d, d = 1..n
    d = torch.arange(1, n + 1, device=a.device, dtype=F64)
    w = (1 - p) * p ** (d - 1)
    return float((w * X / d).sum() + p ** n * X[-1] / n)


def sliced_w1(ma: torch.Tensor, mb: torch.Tensor, n_proj: int, pool: int, gen) -> float:
    a = Fnn.avg_pool2d(ma[None, None], pool)[0, 0]
    b = Fnn.avg_pool2d(mb[None, None], pool)[0, 0]
    a, b = a.flatten() / a.sum(), b.flatten() / b.sum()
    h, w = ma.shape[-2] // pool, ma.shape[-1] // pool
    yy, xx = torch.meshgrid(torch.arange(h, device=a.device, dtype=F64), torch.arange(w, device=a.device, dtype=F64), indexing="ij")
    pts = torch.stack([yy.flatten(), xx.flatten()], 1) * pool + (pool - 1) / 2
    th = torch.rand(n_proj, generator=gen, device=gen.device, dtype=F64).to(a.device) * math.pi
    dirs = torch.stack([torch.cos(th), torch.sin(th)], 1)
    tot = 0.0
    for dv in dirs:
        proj = pts @ dv
        o = torch.argsort(proj)
        ps = proj[o]
        ca, cb = torch.cumsum(a[o], 0), torch.cumsum(b[o], 0)
        tot += float(((ca - cb)[:-1].abs() * (ps[1:] - ps[:-1])).sum())
    return tot / n_proj


def spatial_overlap(ta: torch.Tensor, tb: torch.Tensor, r: int, shape) -> float:
    H, W = shape
    A, B = ta.view(1, 1, H, W).double(), tb.view(1, 1, H, W).double()
    k = 2 * r + 1
    dA = Fnn.max_pool2d(A, k, stride=1, padding=r) > 0
    dB = Fnn.max_pool2d(B, k, stride=1, padding=r) > 0
    a_in = float((A.bool() & dB).sum()) / max(float(A.sum()), 1)
    b_in = float((B.bool() & dA).sum()) / max(float(B.sum()), 1)
    return (a_in + b_in) / 2


def sim(ma: Map, mb: Map, cfgm: dict, perm, gen) -> dict:
    out = {}
    a, b = ma.abs, mb.abs
    H, W = a.shape
    out["spearman_abs"] = _corr(_rank(a), _rank(b))
    for pool in cfgm["pools"]:
        pa, pb = Fnn.avg_pool2d(a[None, None], pool)[0, 0], Fnn.avg_pool2d(b[None, None], pool)[0, 0]
        out[f"spearman_abs_{H // pool}"] = _corr(_rank(pa), _rank(pb))
        if pa.numel() <= 4096:
            out[f"kendall_abs_{H // pool}"] = kendall_tau_b(pa, pb)
    for k in cfgm["topk"]:
        ta, tb = top_mask(a, k, perm), top_mask(b, k, perm)
        out[f"jaccard_{int(round(100 * k))}"] = float((ta & tb).sum()) / max(float((ta | tb).sum()), 1)
    out["rbo"] = rbo_ext(a, b, float(cfgm["rbo_p"]), perm)
    t10a, t10b = top_mask(a, 0.1, perm), top_mask(b, 0.1, perm)
    for r in cfgm["radii"]:
        out[f"overlap_r{r}"] = spatial_overlap(t10a, t10b, int(r), (H, W))
    out["sliced_w1"] = sliced_w1(a, b, int(cfgm["sw_proj"]), int(cfgm["sw_pool"]), gen)
    if ma.signed and mb.signed:
        A, B = ma.A.flatten(), mb.A.flatten()
        out["spearman_signed_pix"] = _corr(_rank(ma.sgn), _rank(mb.sgn))
        out["spearman_signed_elem"] = _corr(_rank(A), _rank(B))
        out["spearman_abs_elem"] = _corr(_rank(A.abs()), _rank(B.abs()))
        out["cos_elem"] = float((A @ B) / (A.norm() * B.norm()))
        out["cos_pix_signed"] = float((ma.sgn.flatten() @ mb.sgn.flatten()) / (ma.sgn.norm() * mb.sgn.norm()))
        out["l1_norm_dist"] = float((A / A.abs().sum() - B / B.abs().sum()).abs().sum())
        k = max(1, int(0.1 * A.numel()))
        ua = torch.zeros_like(A, dtype=torch.bool); ua[A.abs().topk(k).indices] = True
        ub = torch.zeros_like(ua); ub[B.abs().topk(k).indices] = True
        u = ua | ub
        out["sign_agree_top10"] = float((torch.sign(A[u]) == torch.sign(B[u])).double().mean())
        wgt = A.abs() / A.abs().sum() + B.abs() / B.abs().sum()
        out["sign_agree_w"] = float((wgt * (torch.sign(A) == torch.sign(B)).double()).sum() / wgt.sum())
    return out


def block_shuffle(m: Map, block: int, gen) -> Map:
    """Xáo vị trí các khối block×block (cùng một hoán vị cho mọi kênh)."""
    H, W = m.abs.shape
    nb_h, nb_w = H // block, W // block
    bp = torch.randperm(nb_h * nb_w, generator=gen).to(m.abs.device)

    def sh(t):                                                   # t (..., H, W)
        lead = t.shape[:-2]
        x = t[..., :nb_h * block, :nb_w * block].reshape(*lead, nb_h, block, nb_w, block).movedim(-3, -2)
        x = x.reshape(*lead, nb_h * nb_w, block, block)[..., bp, :, :]
        x = x.reshape(*lead, nb_h, nb_w, block, block).movedim(-2, -3).reshape(*lead, nb_h * block, nb_w * block)
        out = t.clone()
        out[..., :nb_h * block, :nb_w * block] = x
        return out
    if m.A is not None:
        return Map(sh(m.A), signed=m.signed)
    return Map(pix_abs=sh(m.abs))


def reference_maps(fn, x, x0, batch) -> dict:
    from .data import MEAN, STD
    _, g = fg(fn, x, batch)
    g = g[0].double()
    img = (x[0].double() * torch.tensor(STD, device=x.device, dtype=F64).view(3, 1, 1)
           + torch.tensor(MEAN, device=x.device, dtype=F64).view(3, 1, 1))
    gray = img.mean(0)[None, None]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=F64, device=x.device).view(1, 1, 3, 3)
    ex = Fnn.conv2d(Fnn.pad(gray, (1, 1, 1, 1), mode="reflect"), kx)
    ey = Fnn.conv2d(Fnn.pad(gray, (1, 1, 1, 1), mode="reflect"), kx.transpose(-1, -2))
    H, W = gray.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=F64), torch.arange(W, device=x.device, dtype=F64), indexing="ij")
    cen = torch.exp(-(((yy - (H - 1) / 2) ** 2 + (xx - (W - 1) / 2) ** 2) / (2 * (H / 4) ** 2)))
    return {"grad": Map(g), "gxi": Map(g * (x - x0)[0].double()), "edge": Map(pix_abs=(ex ** 2 + ey ** 2).sqrt()[0, 0]),
            "center": Map(pix_abs=cen)}


# ─────────────────────────────────────────────────────────────────────────────
# consensus
# ─────────────────────────────────────────────────────────────────────────────

def compactness(mask: torch.Tensor) -> float:
    m = mask.double()[None, None]
    k = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=F64, device=mask.device).view(1, 1, 3, 3)
    nb = Fnn.conv2d(Fnn.pad(m, (1, 1, 1, 1)), k)
    return float((nb * m).sum() / (4 * m.sum())) if float(m.sum()) > 0 else float("nan")


def matched_control(rank_m: torch.Tensor, T_m: torch.Tensor, Cset: torch.Tensor, gen, n_bins: int = 10) -> torch.Tensor:
    """Từ T_m \\ C, rút |C| pixel sao cho phân bố thập phân vị hạng (theo m, trong T_m) khớp với C."""
    n_top = int(T_m.sum())
    q = torch.clamp((rank_m.double() / max(n_top, 1) * n_bins).long(), 0, n_bins - 1)
    cand = T_m & ~Cset
    out = torch.zeros_like(Cset)
    short = 0
    for b in range(n_bins):
        need = int((Cset & (q == b)).sum()) + short
        pool = torch.nonzero(cand & (q == b)).flatten()
        take = min(need, pool.numel())
        if take:
            pick = pool[torch.randperm(pool.numel(), generator=gen)[:take].to(pool.device)]
            out[pick] = True
        short = need - take
    if short > 0:                                                  # bù từ phần còn lại của T_m \ C
        pool = torch.nonzero(cand & ~out).flatten()
        pick = pool[torch.randperm(pool.numel(), generator=gen)[:short].to(pool.device)]
        out[pick] = True
    return out


@torch.no_grad()
def set_effects(bb, cls, x, bases: dict, sets: dict, batch: int) -> dict:
    """Xoá (thay bằng baseline) và chèn (đặt vào baseline) từng tập pixel; logit và xác suất lớp đích."""
    H, W = x.shape[-2:]
    names = list(sets)
    res = {}
    full = bb(x)
    lx, px = float(full[0, cls]), float(torch.softmax(full, -1)[0, cls])
    for bn, base in bases.items():
        fb = bb(base)
        lb, pb = float(fb[0, cls]), float(torch.softmax(fb, -1)[0, cls])
        X = []
        for nme in names:
            m = sets[nme].view(1, 1, H, W).expand_as(x)
            X += [torch.where(m, base, x), torch.where(m, x, base)]
        X = torch.cat(X)
        outs = []
        for i in range(0, X.shape[0], batch):
            outs.append(bb(X[i:i + batch]))
        o = torch.cat(outs)
        lg, pr = o[:, cls].double(), torch.softmax(o, -1)[:, cls].double()
        for k, nme in enumerate(names):
            res[(nme, bn)] = {"del_logit": lx - float(lg[2 * k]), "del_prob": px - float(pr[2 * k]),
                              "ins_logit": float(lg[2 * k + 1]) - lb, "ins_prob": float(pr[2 * k + 1]) - pb}
    return res


# ─────────────────────────────────────────────────────────────────────────────
# một ảnh
# ─────────────────────────────────────────────────────────────────────────────

def ocs_image(ctx: Ctx, j: int, idx: int) -> dict:
    cfg = ctx.cfg
    c = cfg["ocs"]
    cm = c["metrics"]
    it, x, mask, cls = ctx.load(idx)
    fn = TargetFn(ctx.bb, cls)
    x0 = make_base(x, "black")
    dx = x - x0
    H, W = x.shape[-2:]
    perm = tie_perm(it.image, H * W, ctx.seed, ctx.dev)
    gsim = lambda *k: ctx.gen("ocs-sw", it.image, *k)
    M = int(c["M"])
    smax = float(cfg["paths"]["blur"]["max_sigma"])
    xb = gaussian_blur(x, smax)
    rows = defaultdict(list)
    halves = {}
    t0 = time.time()

    def det(name, X, key):
        N = X.shape[0] - 1
        Aa, _ = aw_stream(fn, X, M, ctx.gen("ocs", "a", name, N, M, it.image, *key), ctx.batch)
        Ab, _ = aw_stream(fn, X, M, ctx.gen("ocs", "b", name, N, M, it.image, *key), ctx.batch)
        halves[name] = (Map(Aa), Map(Ab), Map((Aa + Ab) / 2))

    det("ig", straight_nodes(x, x0, int(c["ig_steps"])), (0,))
    det("line_blur", straight_nodes(x, xb, int(c["ig_steps"])), (0,))
    det("blur", blur_nodes(x, int(c["blur_steps"]), smax), (0,))
    gp = cfg["paths"]["guided"]
    Xg, _, _, gap = guided_nodes(fn, x, x0, int(c["guided_steps"]), float(gp["fraction"]), float(gp["max_dist"]), ctx.batch)
    det("guided", Xg, (0,))
    del Xg
    sp = cfg["paths"]["spi10"]
    Kh = int(c["spi_paths_per_half"])
    ks = sorted(set(int(k) for k in c["k_list"] if int(k) <= Kh) | {Kh})
    sums, kmaps = [None, None], {0: {}, 1: {}}
    for h in range(2):
        for p in range(Kh):
            pid = h * Kh + p                                          # đường pilot 0..2Kh−1 (cùng seed với các lượt trước)
            X = spi_nodes_from(spi_increments_for(ctx, sp, dx.shape[1:], int(sp["steps"]), it.image, pid), x0, dx)
            A, _ = aw_stream(fn, X, M, ctx.gen("ocs", "spi", int(sp["steps"]), M, it.image, pid), ctx.batch)
            sums[h] = A.clone() if sums[h] is None else sums[h] + A
            if p + 1 in ks:
                kmaps[h][p + 1] = Map(sums[h] / (p + 1))
            del X
    halves["spi"] = (kmaps[0][Kh], kmaps[1][Kh], Map((sums[0] + sums[1]) / (2 * Kh)))
    refs = reference_maps(fn, x, x0, ctx.batch)
    sync()
    t_paths = time.time() - t0

    # 1) cross / within / null / tham chiếu
    for name in MAIN:
        rows["within"].append({"image": it.image, "method": name, "K": Kh if name == "spi" else 0,
                               "kind": "path_sets" if name == "spi" else "quadrature_sets",
                               **sim(halves[name][0], halves[name][1], cm, perm, gsim(name, "w"))})
    for k in ks:
        rows["within"].append({"image": it.image, "method": "spi", "K": k, "kind": "path_sets_K",
                               **sim(kmaps[0][k], kmaps[1][k], cm, perm, gsim("spiK", k))})
        for ref in ("guided", "ig", "blur"):
            rows["kcross"].append({"image": it.image, "K": k, "ref": ref,
                                   **sim(kmaps[0][k], halves[ref][0], cm, perm, gsim("kc", k, ref))})
    gshuf = torch.Generator().manual_seed(C.seed_of(ctx.seed, "ocs-shuf", it.image))
    for i, a in enumerate(MAIN):
        for b in MAIN[i + 1:]:
            rows["pairs"].append({"image": it.image, "a": a, "b": b, "kind": "cross_half1",
                                  **sim(halves[a][0], halves[b][0], cm, perm, gsim(a, b))})
            rows["pairs"].append({"image": it.image, "a": a, "b": b, "kind": "cross_full",
                                  **sim(halves[a][2], halves[b][2], cm, perm, gsim(a, b, "f"))})
            rows["pairs"].append({"image": it.image, "a": a, "b": b, "kind": "null_block",
                                  **sim(halves[a][2], block_shuffle(halves[b][2], int(c["shuffle_block"]), gshuf), cm, perm,
                                        gsim(a, b, "n"))})
        for r in REFS:
            rows["pairs"].append({"image": it.image, "a": a, "b": r, "kind": "ref",
                                  **sim(halves[a][2], refs[r], cm, perm, gsim(a, r))})

    # 2) consensus
    cc = c["consensus"]
    fm = {m: halves[m][2] for m in cc["methods"]}
    q = float(cc["top_frac"])
    T = {m: top_mask(v.abs, q, perm) for m, v in fm.items()}
    ranks = {}
    for m, v in fm.items():
        o = perm[torch.argsort(v.abs.flatten()[perm], descending=True, stable=True)]
        r = torch.empty_like(o); r[o] = torch.arange(o.numel(), device=o.device)
        ranks[m] = r
    Cset = torch.stack(list(T.values())).all(0)
    union = torch.stack(list(T.values())).any(0)
    nC = int(Cset.sum())
    info = {"image": it.image, "n_consensus": nC, "n_top": int(T[cc["methods"][0]].sum()),
            "consensus_frac_of_top": nC / max(int(T[cc["methods"][0]].sum()), 1),
            "consensus_in_mask": float(mask.flatten()[Cset].double().mean()) if nC else float("nan"),
            "seconds_paths": t_paths, "guided_end_gap": gap}
    if nC >= int(cc["min_size"]):
        g = torch.Generator().manual_seed(C.seed_of(ctx.seed, "ocs-cons", it.image))
        sets = {("C", "-", 0): Cset}
        for m in cc["methods"]:
            others = torch.stack([T[o] for o in cc["methods"] if o != m]).any(0)
            U = T[m] & ~others
            if int(U.sum()) >= nC:
                Ur = torch.nonzero(U).flatten()
                pick = torch.zeros_like(U)
                pick[Ur[torch.randperm(Ur.numel(), generator=g)[:nC].to(Ur.device)]] = True
                sets[("unique", m, 0)] = pick
            for d in range(int(cc["draws"])):
                sets[("matched", m, d)] = matched_control(ranks[m], T[m], Cset, g)
        for d in range(int(cc["draws"])):
            allp = torch.randperm(H * W, generator=g)[:nC].to(Cset.device)
            rm = torch.zeros_like(Cset); rm[allp] = True
            sets[("random", "-", d)] = rm
            up = torch.nonzero(union & ~Cset).flatten()
            if up.numel() >= nC:
                ru = torch.zeros_like(Cset); ru[up[torch.randperm(up.numel(), generator=g)[:nC].to(up.device)]] = True
                sets[("random_union", "-", d)] = ru
        ic = cfg["metrics"]["insdel"]
        bases = {b: make_base(x, b, ic["blur_kernel"], ic["blur_sigma"]) for b in cc["bases"]}
        eff = set_effects(ctx.bb, cls, x, bases, {k: v.view(H, W) for k, v in sets.items()}, ctx.batch)
        for (kind, m, d), sm in sets.items():
            att = {f"absA_{mm}": float(fm[mm].abs.flatten()[sm].sum() / fm[mm].abs.sum()) for mm in cc["methods"]}
            for bn in bases:
                rows["consensus"].append({"image": it.image, "set": kind, "method": m, "draw": d, "base": bn,
                                          "size": int(sm.sum()), "compactness": compactness(sm.view(H, W)),
                                          "in_mask": float(mask.flatten()[sm].double().mean()), **att, **eff[((kind, m, d), bn)]})
    rows["images"].append(info)
    heat = None
    if j < int(c.get("heatmap_images", 3)):
        heat = {"x": it.x.half(), "mask": mask.cpu(), "consensus": Cset.view(H, W).cpu(),
                **{m: halves[m][2].abs.float().cpu() for m in MAIN}}
    return {"rows": dict(rows), "heat": heat}


# ─────────────────────────────────────────────────────────────────────────────
# chạy
# ─────────────────────────────────────────────────────────────────────────────

def estimate(cfg: dict, per_eval: float, per_seq: float) -> dict:
    c = cfg["ocs"]
    M, ig = int(c["M"]), int(c["ig_steps"])
    N = int(cfg["paths"]["spi10"]["steps"])
    n = 2 * (2 * ig * M) + 2 * int(c["blur_steps"]) * M + 2 * int(c["guided_steps"]) * M + 2 * int(c["spi_paths_per_half"]) * N * M
    t = int(c["n_images"]) * (n * per_eval + (int(c["guided_steps"]) + 1) * per_seq)
    return {"ocs_min": t / 60, "evals_per_image": n}


def prereg(cfg: dict, out: str, names: list):
    dg = ocs_digest(cfg)
    pj = os.path.join(out, "ocs_prereg.json")
    if os.path.exists(pj) and not cfg["_debug"]:
        if json.load(open(pj))["digest"] != dg:
            raise SystemExit(f"thiết kế OCS đã đổi sau khi tiền đăng ký ({pj}); split xác nhận phải giữ nguyên")
        return
    with open(pj, "w") as fh:
        json.dump({"digest": dg, "time": _dt.datetime.now().isoformat(timespec="seconds"), "images": names, "ocs": cfg["ocs"]},
                  fh, indent=1, default=str)
    print(f"[ocs] đã tiền đăng ký: {pj}")


def run(ctx: Ctx):
    import pandas as pd
    from .pilot import bench
    cfg = ctx.cfg
    c = cfg["ocs"]
    out = ocs_out(cfg)
    dg = ocs_digest(cfg)
    split = c["split"]
    idxs = ctx.sp[split][:int(c["n_images"])]
    if split == "fresh":
        used = set(ctx.sp["pilot"]) | set(ctx.sp["confirm"]) | set(ctx.sp["other"])
        if used & set(idxs):
            raise SystemExit("ảnh fresh giao với ảnh đã dùng")
    names = [f"seg{i:05d}" for i in idxs]
    if split != "pilot":
        prereg(cfg, out, names)
    b = bench(ctx)
    est = estimate(cfg, b["per_eval"], b["per_seq"])
    print(f"[ocs] split {split}, {len(idxs)} ảnh; {b['per_eval'] * 1e3:.2f} ms/eval; ước tính ≈ {est['ocs_min']:.1f} phút", flush=True)
    t_all = time.time()
    cd = os.path.join(out, "cache")
    os.makedirs(cd, exist_ok=True)
    for j, idx in enumerate(idxs):
        image = f"seg{idx:05d}"
        cp = os.path.join(cd, f"{image}.pt")
        if os.path.exists(cp) and torch.load(cp, weights_only=False).get("digest") == dg:
            print(f"  [ocs] {image}: đã có trong cache, bỏ qua")
            continue
        t0, n0 = time.time(), Counter.n
        print(f"  [ocs] ảnh {j + 1}/{len(idxs)} {image}", flush=True)
        obj = ocs_image(ctx, j, idx)
        obj["digest"] = dg
        torch.save(obj, cp)
        print(f"  [ocs] {image}: {Counter.n - n0} eval, {time.time() - t0:.0f}s; tổng {(time.time() - t_all) / 60:.1f} phút", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    allrows, heats = defaultdict(list), {}
    for im in names:
        obj = torch.load(os.path.join(cd, f"{im}.pt"), weights_only=False)
        for k, rs in obj["rows"].items():
            allrows[k] += rs
        if obj.get("heat") is not None:
            heats[im] = obj["heat"]
    for k, rs in allrows.items():
        if rs:
            pd.DataFrame(rs).to_csv(os.path.join(out, f"ocs_{k}.csv"), index=False)
            print(f"[ocs] ocs_{k}.csv: {len(rs)} dòng")
    if heats:
        torch.save(heats, os.path.join(out, "ocs_heatmaps.pt"))
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"images": names, "split": split, "bench": b, "estimate": est, "minutes": (time.time() - t_all) / 60,
                   "digest": dg, "debug": cfg["_debug"], "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "ocs_config": c}, fh, indent=1, default=str)
