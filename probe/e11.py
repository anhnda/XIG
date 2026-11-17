"""
E11 — LIG chỉ Phase 1 (tối ưu độ đo μ trên các bước), đường CHO TRƯỚC là một mẫu SPI hoặc biến thể; không tối ưu đường (Phase 2).
Câu hỏi: cho một đường cố định, tối ưu μ kiểu LIG có luôn cải thiện attribution không?

LIG Phase 1 (chép đúng từ `utility.optimize_mu_signal_harvesting` của repo LIG):
    min_{μ ∈ simplex}  Var_ν(φ) − λ Σ_k μ_k |d_k| + (τ/2)‖μ‖²,   φ_k = d_k/Δf_k,  ν_k ∝ μ_k Δf_k²
    μ = softmax(logits), Adam lr 0.05, n_iter bước, logits khởi tạo 0
    A = rescale( Σ_k μ_k g_k ⊙ Δγ_k  →  tổng = f(x) − f(x′) )
với g_k = ∇f(γ_k) tại nút trái, d_k = g_k·Δγ_k, Δf_k = f(γ_{k+1}) − f(γ_k) (căn đúng; `lig_legacy` dùng Δf lệch một bước như
code gốc của bài: Δf_0 = 0, Δf_k = f(γ_k) − f(γ_{k−1})).

Ước lượng trên CÙNG một đường và cùng gradient tại nút (chỉ khác trọng số bước):
    uniform           μ = 1/N, không rescale  (= canonical Riemann trái, đúng như SPI dùng)
    uniform_rescale   μ = 1/N, rescale về D   (tách hiệu ứng của riêng bước rescale)
    lig               Phase 1 LIG (λ, τ mặc định của bài), rescale
    lig_norescale     Phase 1 LIG, không rescale
    lig_legacy        Phase 1 LIG với Δf lệch (như bài đã chạy), rescale
    lig_tau           Phase 1 LIG với τ lớn hơn (ít dồn μ), rescale
    mu_df             μ_k ∝ |Δf_k| (dạng μ của IDGI theo bài), rescale
    mu_proj           μ = Proj_simplex((λ/τ)|d|) (nghiệm khi bỏ số hạng Var), rescale
Chẩn đoán μ: max, số bước có μ > 1e-3, entropy chuẩn hoá, có sụp về một bước không (max > 0.9), điều kiện định lý
(khoảng cách hai |d| lớn nhất ≥ τ/λ ⇒ e_k* là nghiệm toàn cục), và **độ phủ**: phần phần tử có Σ_k μ_k|Δγ_k| > 0 — với đường
SPI mỗi bước chỉ di chuyển một phần nhỏ phần tử, nên μ dồn vào ít bước thì attribution chỉ phủ ít phần tử.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict

import torch

from . import config as C
from .e10 import calibrate, sched
from .e7 import group_labels
from .mechanism import summ
from .metrics import energy_mask, insdel, iou, make_base, tie_perm
from .model import ProbFn, TargetFn
from .paths import cum_nodes, spi_increments
from .pilot import Ctx, sync
from .quad import Counter, fg

F64 = torch.float64


# ─────────────────────────────────────────────────────────────────────────────
# LIG Phase 1 (chép từ utility.py của repo LIG)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_mu_signal_harvesting(d, delta_f, lam=1.0, tau=0.01, n_iter=300, lr=0.05):
    device = d.device
    N = d.shape[0]
    valid = delta_f.abs() > 1e-12
    safe_df = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe_df, torch.ones_like(d)).detach()
    df2 = (delta_f ** 2).detach()
    abs_d = d.abs().detach()
    logits = torch.zeros(N, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    with torch.enable_grad():
        for _ in range(n_iter):
            opt.zero_grad()
            mu = torch.softmax(logits, dim=0)
            nu = mu * df2
            nu_sum = nu.sum()
            if nu_sum < 1e-15:
                break
            w = nu / nu_sum
            mean_phi = (w * phi).sum()
            var_phi = (w * (phi - mean_phi) ** 2).sum()
            loss = var_phi - lam * (mu * abs_d).sum() + (tau / 2.0) * (mu ** 2).sum()
            loss.backward()
            opt.step()
    with torch.no_grad():
        mu = torch.softmax(logits, dim=0)
    return mu.detach()


def all_metrics(d, delta_f, mu):
    """(Var_ν, CV², Q) như `utility.compute_all_metrics`."""
    valid = delta_f.abs() > 1e-12
    safe_df = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe_df, torch.ones_like(d))
    nu = mu * delta_f ** 2
    if float(nu.sum()) < 1e-15:
        return 0.0, 0.0, 1.0
    nu = nu / nu.sum()
    pb = (nu * phi).sum()
    var = float((nu * (phi - pb) ** 2).sum())
    if abs(float(pb)) < 1e-12:
        return var, (float("inf") if var > 1e-15 else 0.0), (0.0 if var > 1e-15 else 1.0)
    cv2 = var / float(pb) ** 2
    return var, cv2, 1.0 / (1.0 + cv2)


def objective(d, delta_f, mu, lam, tau):
    var, _, _ = all_metrics(d, delta_f, mu)
    return var - lam * float((mu * d.abs()).sum()) + tau / 2 * float((mu ** 2).sum())


def project_simplex(v):
    u, _ = torch.sort(v, descending=True)
    css = u.cumsum(0) - 1.0
    k = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    rho = int((u - css / k > 0).nonzero().max())
    return (v - css[rho] / (rho + 1)).clamp(min=0.0)


def legacy_df(f):
    df = f[1:] - f[:-1]
    return torch.cat([df.new_zeros(1), df[:-1]])


def rescale(A, target):
    s = float(A.sum())
    return A * (target / s) if abs(s) > 1e-12 else A


# ─────────────────────────────────────────────────────────────────────────────
# đường
# ─────────────────────────────────────────────────────────────────────────────

def make_schedule(spec: str, N: int, shape, gen_dev, gen_cpu, cal: dict) -> torch.Tensor:
    """spec: ig | dp:<α> | smooth | lognormal | rgb:<α> | block<b>:<α>"""
    name, _, arg = spec.partition(":")
    if name in ("ig", "smooth", "lognormal"):
        return sched(name, N, shape, gen_dev, cal["alpha"], cal["delta"], cal["sigma"])
    alpha = float(arg) if arg else cal["alpha"]
    if name == "dp":
        return sched("dp", N, shape, gen_dev, alpha, 0.0, 0.0)
    if name == "rgb" or name.startswith("block"):
        lab, n = group_labels(name, tuple(shape), gen_cpu)
        Zg = cum_nodes(spi_increments(alpha, N, (n,), gen_dev))
        return Zg[:, lab.to(Zg.device)].view((N + 1,) + tuple(shape))
    raise ValueError(spec)


# ─────────────────────────────────────────────────────────────────────────────
# một ảnh
# ─────────────────────────────────────────────────────────────────────────────

def e11_image(ctx: Ctx, idx: int, cal: dict) -> dict:
    cfg = ctx.cfg
    c = cfg["e11"]
    it, x, mask, cls = ctx.load(idx)
    fn, prob = TargetFn(ctx.bb, cls), ProbFn(ctx.bb, cls)
    x0 = make_base(x, "black")
    dx = x - x0
    m3 = mask.unsqueeze(0).expand_as(x[0])
    H, W = x.shape[-2:]
    perm = tie_perm(it.image, H * W, ctx.seed, ctx.dev)
    ic = cfg["metrics"]["insdel"]
    bases = {b: make_base(x, b, ic["blur_kernel"], ic["blur_sigma"]) for b in c["insdel_bases"]}
    N, K = int(c["N"]), int(c["K"])
    lam, tau, tau2, n_iter = float(c["lam"]), float(c["tau"]), float(c["tau_alt"]), int(c["mu_iter"])
    shape = tuple(dx.shape[1:])
    rows = defaultdict(list)

    def quality(A, full=True):
        A = A.float()
        q = {"iou": iou(A, mask, perm), "energy_mask": energy_mask(A, None, mask)}
        if full:
            for bn, base in bases.items():
                q[f"insdel_{bn}"] = insdel(prob, x, base, A, int(c["insdel_steps"]), ic["batch"])
        return q

    for spec in c["paths"]:
        t0, n0 = time.time(), Counter.n
        ens = {}
        n_paths = 1 if spec == "ig" else K
        for p in range(n_paths):
            gc = torch.Generator().manual_seed(C.seed_of(ctx.seed, "e11part", spec, it.image, p))
            Z = make_schedule(spec, N, shape, ctx.gen("e11Z", spec, N, it.image, p), gc, cal)
            X = x0 + Z.to(x.dtype) * dx
            f, g_all = fg(fn, X, ctx.batch)                           # f tại N+1 nút, g tại nút 0..N−1 (trái)
            g = g_all[:N].double()
            h = (X[1:] - X[:-1]).double()
            d = (g * h).flatten(1).sum(1)
            df = f[1:] - f[:-1]
            D = float(f[-1] - f[0])
            terms = g * h                                             # (N, C, H, W): đóng góp từng bước
            a_abs = d.abs()
            top2 = torch.topk(a_abs, 2).values
            dc, dfc, fc = d.cpu(), df.cpu(), f.cpu()                     # tối ưu μ (N chiều) trên CPU cho nhanh
            mus = {"uniform": torch.full((N,), 1.0 / N, dtype=F64)}
            mus["lig"] = optimize_mu_signal_harvesting(dc.float(), dfc.float(), lam, tau, n_iter).double()
            mus["lig_legacy"] = optimize_mu_signal_harvesting(dc.float(), legacy_df(fc).float(), lam, tau, n_iter).double()
            mus["lig_tau"] = optimize_mu_signal_harvesting(dc.float(), dfc.float(), lam, tau2, n_iter).double()
            mus["mu_df"] = dfc.abs() / dfc.abs().sum() if float(dfc.abs().sum()) > 0 else mus["uniform"]
            mus["mu_proj"] = project_simplex((lam / tau) * dc.abs())
            est = {"uniform": ("uniform", False), "uniform_rescale": ("uniform", True), "lig": ("lig", True),
                   "lig_norescale": ("lig", False), "lig_legacy": ("lig_legacy", True), "lig_tau": ("lig_tau", True),
                   "mu_df": ("mu_df", True), "mu_proj": ("mu_proj", True)}
            move = (h.abs() > 0)                                      # phần tử di chuyển ở từng bước
            for ename in c["estimators"]:
                mname, rs = est[ename]
                mu = mus[mname]
                A = (mu.to(terms.device).view(-1, 1, 1, 1) * terms).sum(0)      # Σ_k μ_k g_k⊙Δγ_k
                raw_sum = float(A.sum())
                if rs:
                    A = rescale(A, D)
                var, cv2, Q = all_metrics(dc, dfc, mu)
                ent = float(-(mu.clamp_min(1e-30) * mu.clamp_min(1e-30).log()).sum() / math.log(N))
                mud = mu.to(h.device)
                cov = float(((mud.view(-1, 1, 1, 1) * h.abs()).sum(0) > 1e-12).double().mean())
                cov_steps = float(move[mud > 1e-3].any(0).double().mean()) if bool((mud > 1e-3).any()) else 0.0
                row = {"image": it.image, "path": spec, "p": p, "estimator": ename, "D": D, "raw_sum": raw_sum,
                       "resid_raw_rel": (raw_sum - D) / abs(D) if D else float("nan"),
                       "mu_max": float(mu.max()), "mu_support": int((mu > 1e-3).sum()), "mu_entropy": ent,
                       "mu_argmax": int(mu.argmax()), "collapsed": bool(float(mu.max()) > 0.9),
                       "coverage_support": cov_steps, "coverage_weighted": cov,
                       "Q": Q, "Var_nu": var, "J": objective(dc, dfc, mu, lam, tau),
                       "theorem_applies": bool(float(top2[0] - top2[1]) >= tau / lam and abs(float(df[int(a_abs.argmax())])) > 1e-12),
                       **quality(A, full=bool(c["per_path_insdel"]))}
                rows["paths"].append(row)
                e = ens.setdefault(ename, None)
                ens[ename] = A.clone() if e is None else e + A
            del X, Z, g_all, g, h, terms
        sync()
        for ename, S in ens.items():
            rows["ens"].append({"image": it.image, "path": spec, "estimator": ename, "K": n_paths, **quality(S / n_paths)})
        rows["timing"].append({"image": it.image, "path": spec, "seconds": time.time() - t0, "evals": Counter.n - n0})
    rows["images"].append({"image": it.image, "index": it.index, "cls": cls, "mask_area": float(mask.float().mean())})
    return dict(rows)


# ─────────────────────────────────────────────────────────────────────────────
# chạy
# ─────────────────────────────────────────────────────────────────────────────

def e11_digest(cfg: dict) -> str:
    keys = ("seed", "model", "data", "paths", "metrics", "e11")
    return __import__("hashlib").sha256(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def e11_out(cfg: dict) -> str:
    d = C.out_dir(cfg, "e11_" + cfg["e11"]["split"])
    os.makedirs(d, exist_ok=True)
    return d


def run(ctx: Ctx):
    import pandas as pd
    from .pilot import bench
    cfg = ctx.cfg
    c = cfg["e11"]
    out = e11_out(cfg)
    dg = e11_digest(cfg)
    cal = calibrate({"e10": {"alpha": float(c["alpha"]), "N": int(c["N"]), "calib": c["calib"]}})
    if c["split"] == "fresh":
        off = int(c["fresh_offset"])
        idxs = ctx.sp["fresh"][off:off + int(c["n_images"])]
    else:
        idxs = ctx.sp[c["split"]][:int(c["n_images"])]
    names = [f"seg{i:05d}" for i in idxs]
    b = bench(ctx)
    n_paths = sum(1 if s == "ig" else int(c["K"]) for s in c["paths"])
    ins = (len(c["insdel_bases"]) * 2 * (int(c["insdel_steps"]) + 1)) / 3
    per_img = n_paths * ((int(c["N"]) + 1) + len(c["estimators"]) * ins * (1 if c["per_path_insdel"] else 0)) \
        + len(c["paths"]) * len(c["estimators"]) * ins
    print(f"[e11] split {c['split']}, {len(idxs)} ảnh, {n_paths} đường/ảnh; {b['per_eval'] * 1e3:.2f} ms/eval; ước tính ≈ "
          f"{len(idxs) * per_img * b['per_eval'] / 60:.1f} phút", flush=True)
    t_all = time.time()
    cd = os.path.join(out, "cache")
    os.makedirs(cd, exist_ok=True)
    for j, idx in enumerate(idxs):
        image = f"seg{idx:05d}"
        cp = os.path.join(cd, f"{image}.pt")
        if os.path.exists(cp) and torch.load(cp, weights_only=False).get("digest") == dg:
            print(f"  [e11] {image}: đã có trong cache, bỏ qua")
            continue
        t0, n0 = time.time(), Counter.n
        print(f"  [e11] ảnh {j + 1}/{len(idxs)} {image}", flush=True)
        rows = e11_image(ctx, idx, cal)
        torch.save({"rows": rows, "digest": dg}, cp)
        print(f"  [e11] {image}: {Counter.n - n0} eval, {time.time() - t0:.0f}s; tổng {(time.time() - t_all) / 60:.1f} phút", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    allrows = defaultdict(list)
    for im in names:
        for k, rs in torch.load(os.path.join(cd, f"{im}.pt"), weights_only=False)["rows"].items():
            allrows[k] += rs
    for k, rs in allrows.items():
        if rs:
            pd.DataFrame(rs).to_csv(os.path.join(out, f"e11_{k}.csv"), index=False)
            print(f"[e11] e11_{k}.csv: {len(rs)} dòng")
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"images": names, "split": c["split"], "calibration": cal, "bench": b, "minutes": (time.time() - t_all) / 60,
                   "digest": dg, "debug": cfg["_debug"], "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
                  fh, indent=1, default=str)
