"""
E8 — xác nhận held-out (split confirm, 50 ảnh chưa dùng), các phép so đã ĐĂNG KÝ TRƯỚC trong config `e8.hypotheses`.

Lần chạy đầu ghi <out>/e8_prereg.json (hash của toàn bộ thiết kế: e8, paths, metrics, model, data, seed) và E8_PREREG.md;
mọi lần chạy sau phải trùng hash, nếu không thì dừng. Báo cáo liệt kê MỌI phép so đã đăng ký, kể cả kết quả không thuận lợi.

main   DP α, K đường mỗi kiểu nhóm, mỗi đường K = 1 metric riêng (trung bình trong ảnh); lịch trình rút ở N_fine rồi
       cộng gộp xuống N (cùng một realization cho cổng số học). Cầu phương phân tầng M điểm/đoạn.
       IG: cùng N, M, thêm một bộ điểm độc lập thứ hai để đo nhiễu cầu phương (dùng cho phép so global ↔ IG).
gate   block16 / rand256 trên cùng các realization, ở N_fine với bộ điểm cầu phương độc lập.
sec    (phụ) Guided IG, Guided chuyển lịch trình từ ảnh khác, IG ở N, M riêng.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import time
from collections import defaultdict

import torch

from . import config as C
from .e5 import geom_dist, schedule
from .e6 import transfer
from .e7 import group_labels
from .mechanism import aw, strat_points, summ
from .metrics import energy_mask, insdel, iou, make_base, tie_perm
from .model import ProbFn, TargetFn
from .paths import cum_nodes, guided_nodes, spi_increments, straight_nodes
from .pilot import Ctx, sync
from .quad import Counter, fg

F64 = torch.float64


def e8_digest(cfg: dict) -> str:
    keys = ("seed", "model", "data", "paths", "metrics", "e8")
    return hashlib.sha256(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def e8_out(cfg: dict) -> str:
    d = C.out_dir(cfg, "e8_" + cfg["e8"]["split"])
    os.makedirs(d, exist_ok=True)
    return d


def prereg(cfg: dict, out: str, images: list) -> None:
    dg = e8_digest(cfg)
    pj = os.path.join(out, "e8_prereg.json")
    if os.path.exists(pj) and not cfg["_debug"]:                 # debug: ghi đè tự do; bản thật: khoá cứng
        old = json.load(open(pj))
        if old["digest"] != dg:
            raise SystemExit(f"thiết kế E8 đã đổi sau khi tiền đăng ký ({pj}, hash {old['digest'][:12]} ≠ {dg[:12]}). "
                             "Không được sửa sau khi đã chạy; nếu thật sự cần, dùng thư mục out mới và ghi rõ lý do.")
        return
    e = cfg["e8"]
    rec = {"digest": dg, "time": _dt.datetime.now().isoformat(timespec="seconds"), "images": images,
           "design": {k: cfg.get(k) for k in ("seed", "model", "data", "paths", "metrics", "e8")}}
    with open(pj, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    L = [f"# E8_PREREG — ghi lúc {rec['time']}, hash `{dg[:16]}`\n",
         f"- split **{e['split']}**, {len(images)} ảnh (không giao với pilot/ảnh cũ theo `data.splits`)",
         f"- DP α = {e['alpha']}, K = {e['K']} đường/kiểu, N = {e['N']} (gộp từ N_fine = {e['N_fine']}), M = {e['M']} điểm/đoạn",
         f"- kiểu nhóm: {', '.join(e['kinds'])}",
         f"- thống kê: hiệu ghép cặp theo ảnh (trung bình K đường trong ảnh), bootstrap theo ảnh {e['stats']['bootstrap']} lần "
         f"(seed {e['stats']['seed']}), α = {e['stats']['alpha']}; Holm trên họ `primary`; tương đương = CI {int(100 * (1 - 2 * e['stats']['alpha']))}% "
         "nằm trọn trong ±biên (TOST)",
         f"- global ↔ IG (attribution có dấu): median_ảnh(‖A_global − A_IG,a‖₁ / ‖A_IG,b − A_IG,a‖₁) ≤ {e['global_signed_ratio_max']}",
         f"- cổng số học: {', '.join(e['gate']['kinds'])} trên {e['gate']['n_images']} ảnh đầu, cùng realization, N = {e['N_fine']}, "
         f"M = {e['gate']['M']}, bộ điểm độc lập",
         "", "| ID | so sánh | metric | kiểm định | biên | họ |", "|---|---|---|---|---|---|"]
    for h in e["hypotheses"]:
        L.append(f"| {h['id']} | {h['a']} − {h['b']} | {', '.join(h['metrics'])} | {h['test']} | {h.get('margin', '—')} | "
                 f"{h.get('family', '—')} |")
    with open(os.path.join(out, "E8_PREREG.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[e8] đã tiền đăng ký: {pj}")


def e8_image(ctx: Ctx, j: int, idx: int, donors: list) -> dict:
    cfg = ctx.cfg
    e = cfg["e8"]
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
    alpha, N, Nf, M, K = float(e["alpha"]), int(e["N"]), int(e["N_fine"]), int(e["M"]), int(e["K"])
    rows = defaultdict(list)
    timing = {}

    def quality(A):
        A = A.float()
        q = {"iou": iou(A, mask, perm_t), "energy_mask": energy_mask(A, None, mask)}
        for bn, base in bases.items():
            q[f"insdel_{bn}"] = insdel(prob, x, base, A, ic["n_steps"], ic["batch"])
        return q

    def evalp(X, M_, key):
        n = X.shape[0] - 1
        P, _ = strat_points(X, M_, ctx.gen("e8q", n, M_, it.image, *key))
        _, G = fg(fn, P, ctx.batch)
        A, W = aw(G, X[1:] - X[:-1], M_)
        del P, G
        return A, W

    def record(exp, family, r, A, W, X, extra=None, q=True):
        s = summ(A, W, m3)
        row = {"image": it.image, "exp": exp, "family": family, "r": r, "D": D, **s, "within": s["W"] - s["L1"],
               "dist_geom": geom_dist(X), **(extra or {})}
        if q:
            row.update(quality(A))
        rows["paths"].append(row)

    # IG (bộ a cho metric, bộ b chỉ để đo nhiễu cầu phương)
    t0 = time.time()
    Xig = straight_nodes(x, x0, N)
    A_ig, W_ig = evalp(Xig, M, ("ig", "a"))
    A_igb, _ = evalp(Xig, M, ("ig", "b"))
    noise = float((A_igb - A_ig).abs().sum())
    record("main", "ig", 0, A_ig, W_ig, Xig, {"groups": 0})
    gate = j < int(e["gate"]["n_images"])
    for kind in e["kinds"]:
        for p in range(K):
            gc = torch.Generator().manual_seed(C.seed_of(ctx.seed, "e8part", kind, it.image, p))
            lab, n = group_labels(kind, tuple(dx.shape[1:]), gc)
            Zf = cum_nodes(spi_increments(alpha, Nf, (n,), ctx.gen("e8S", kind, Nf, it.image, p)))
            Xc = x0 + Zf[::Nf // N][:, lab.to(Zf.device)].view((N + 1,) + tuple(dx.shape[1:])).to(x.dtype) * dx
            A, W = evalp(Xc, M, (kind, p))
            extra = {"groups": n}
            if kind == "global":
                extra["signed_l1_vs_ig"] = float((A - A_ig).abs().sum()) / float(A_ig.abs().sum())
                extra["signed_ratio_to_noise"] = float((A - A_ig).abs().sum()) / max(noise, 1e-30)
            record("main", kind, p, A, W, Xc, extra)
            del Xc
            if gate and kind in e["gate"]["kinds"]:
                Xf = x0 + Zf[:, lab.to(Zf.device)].view((Nf + 1,) + tuple(dx.shape[1:])).to(x.dtype) * dx
                A2, W2 = evalp(Xf, int(e["gate"]["M"]), ("gate", kind, p))
                record("gate", kind, p, A2, W2, Xf, {"groups": n, "A_rel_diff_main": float((A2 - A).abs().sum() / A.abs().sum())})
                if p == 0 and kind == e["gate"]["kinds"][0]:
                    Xi = straight_nodes(x, x0, Nf)
                    Ai, Wi = evalp(Xi, int(e["gate"]["M"]), ("gate", "ig"))
                    record("gate", "ig", 0, Ai, Wi, Xi, {"groups": 0})
                del Xf
            del Zf
    timing["main"] = time.time() - t0

    # phụ: Guided IG
    sc = e["secondary_guided"]
    if sc.get("enabled", False):
        t0 = time.time()
        Ng, Mg = int(sc["N"]), int(sc["M"])
        X = straight_nodes(x, x0, Ng)
        record("sec", "ig", 0, *evalp(X, Mg, ("sec", "ig")), X)
        gp = cfg["paths"]["guided"]
        Xg, _, _, gap = guided_nodes(fn, x, x0, Ng, float(gp["fraction"]), float(gp["max_dist"]), ctx.batch)
        record("sec", "guided", 0, *evalp(Xg, Mg, ("sec", "guided")), Xg, {"end_gap": gap})
        at = dx[0].abs()
        for r, (dname, Zd, ad) in enumerate(donors):
            X = x0 + transfer(Zd, ad, at) * dx
            record("sec", "guided_xfer", r, *evalp(X, Mg, ("sec", "xfer", r)), X, {"donor": dname})
        del Xg
        timing["sec"] = time.time() - t0
    sync()
    rows["images"].append({"image": it.image, "index": it.index, "cls": cls, "D": D, "mask_area": float(mask.float().mean()),
                           "ig_quad_noise_l1_rel": noise / float(A_ig.abs().sum()), "timing": json.dumps(timing)})
    return {"rows": dict(rows)}


def make_donors(ctx: Ctx) -> list:
    sc = ctx.cfg["e8"]["secondary_guided"]
    if not sc.get("enabled", False):
        return []
    gp = ctx.cfg["paths"]["guided"]
    out = []
    for idx in ctx.sp["other"][:int(sc["donors"])]:
        it, xd, _, cls = ctx.load(idx)
        x0d = make_base(xd, "black")
        Xd, _, _, _ = guided_nodes(TargetFn(ctx.bb, cls), xd, x0d, int(sc["N"]), float(gp["fraction"]), float(gp["max_dist"]),
                                   ctx.batch)
        out.append((it.image, schedule(Xd, x0d, xd - x0d), (xd - x0d)[0].abs()))
        print(f"  [e8] ảnh cho lịch trình: {it.image}", flush=True)
    return out


def estimate(cfg: dict, per_eval: float, per_seq: float) -> dict:
    e = cfg["e8"]
    ic = cfg["metrics"]["insdel"]
    ins = len(ic["bases"]) * 2 * (ic["n_steps"] + 1) / 3
    N, M, K = int(e["N"]), int(e["M"]), int(e["K"])
    main = (2 * N * M + ins) + len(e["kinds"]) * K * (N * M + ins)
    g = e["gate"]
    gate = (len(g["kinds"]) * K + 1) * (int(e["N_fine"]) * int(g["M"]) + ins)
    sc = e["secondary_guided"]
    sec = (2 + int(sc["donors"])) * (int(sc["N"]) * int(sc["M"]) + ins) if sc.get("enabled") else 0
    seq = (int(sc["N"]) + 1) * per_seq if sc.get("enabled") else 0
    n = int(e["n_images"])
    t = n * ((main + sec) * per_eval + seq) + int(g["n_images"]) * gate * per_eval + int(sc.get("donors", 0)) * seq
    return {"e8_min": t / 60, "evals_main": main, "evals_gate": gate, "evals_sec": sec}


def run(ctx: Ctx):
    import pandas as pd
    from .pilot import bench
    cfg = ctx.cfg
    e = cfg["e8"]
    out = e8_out(cfg)
    dg = e8_digest(cfg)
    idxs = ctx.sp[e["split"]][:int(e["n_images"])]
    if len(idxs) < int(e["n_images"]):
        raise SystemExit(f"split {e['split']} chỉ có {len(idxs)} ảnh < {e['n_images']}")
    used = set(ctx.sp["pilot"])
    if e["split"] != "pilot" and used & set(idxs):
        raise SystemExit("ảnh xác nhận giao với ảnh pilot")
    names = [f"seg{i:05d}" for i in idxs]
    prereg(cfg, out, names)
    b = bench(ctx)
    est = estimate(cfg, b["per_eval"], b["per_seq"])
    print(f"[e8] split {e['split']}, {len(idxs)} ảnh; {b['per_eval'] * 1e3:.2f} ms/eval; ước tính ≈ {est['e8_min']:.1f} phút",
          flush=True)
    t_all = time.time()
    cd = os.path.join(out, "cache")
    os.makedirs(cd, exist_ok=True)

    def done(i):
        p = os.path.join(cd, f"seg{i:05d}.pt")
        return os.path.exists(p) and torch.load(p, weights_only=False).get("digest") == dg

    todo = [(j, i) for j, i in enumerate(idxs) if not done(i)]
    donors = make_donors(ctx) if todo else []
    for j, idx in todo:
        image = f"seg{idx:05d}"
        t0, n0 = time.time(), Counter.n
        print(f"  [e8] ảnh {j + 1}/{len(idxs)} {image}", flush=True)
        obj = e8_image(ctx, j, idx, donors)
        obj["digest"] = dg
        torch.save(obj, os.path.join(cd, f"{image}.pt"))
        print(f"  [e8] {image}: {Counter.n - n0} eval, {time.time() - t0:.0f}s; tổng {(time.time() - t_all) / 60:.1f} phút",
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
            pd.DataFrame(rs).to_csv(os.path.join(out, f"e8_{k}.csv"), index=False)
            print(f"[e8] e8_{k}.csv: {len(rs)} dòng")
    meta = {"images": names, "split": e["split"], "bench": b, "estimate": est, "minutes": (time.time() - t_all) / 60,
            "digest": dg, "debug": cfg["_debug"], "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "donors": [f"seg{i:05d}" for i in ctx.sp["other"][:int(e["secondary_guided"].get("donors", 0))]]}
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
