"""
E-quadrature — `quad-check` (C3: tích phân đã hội tụ). Trên một tập đường nhỏ của ResNet-50, lấy gradient trên lưới lồng nhau
N → 2N → 4N → … (cùng polyline, 2^L điểm/đoạn), tinh dần cho tới khi sai số CHIẾU giữa hai mức liên tiếp của điểm giữa dưới
`tol_ref` (không mặc định 4N là đủ). Đo trực tiếp
    ‖P_D(T_M − T_ref)‖ / ‖P_D T_ref‖
cho left(M), mid(M) (M = 2^L) và rand(M) (quy tắc cũ của P4 / PATHSCORE), kèm sai số L2 theo phần tử và khoảng hở
completeness (khoảng hở = 0 không chứng tỏ đã hội tụ). Lưới 2N cho cùng lúc left(N), mid(N), left(2N) nhưng ba giá trị này
KHÔNG độc lập (mid = một bước Richardson của left).
Khuyến nghị: M nhỏ nhất mà q95 (qua ảnh × đường) của sai số chiếu ≤ `tol` — người dùng khoá vào infid.protocol.{rule, M}
TRƯỚC E1 (bash scripts/run.sh infid-lock).
Đầu ra: <out>/pilot/infid/quad/{quad_errors.csv, quad_levels.csv, INFID_QUAD_REPORT.md}.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import time

import numpy as np
import torch

from . import config as C
from . import infid_core as IC
from . import sampling as SM
from .infid_e0 import icfg, out_dir, primary_spec, protocol
from .pathquad import NestedLattice, integrate
from .paths import straight_nodes
from .quad import Counter

F64 = torch.float64
Q_DEFAULT = {"split": "pilot", "offset": 0, "n_images": 4, "paths_per_alpha": 2, "alphas": [1, 10, 100], "N": 64, "ig": True,
             "ig_N": 64, "L_max": 4, "tol_ref": 1.0e-3, "tol": 1.0e-2, "rand_M": [1], "n_sets": 256}


def qcfg(cfg: dict) -> dict:
    return C._merge(Q_DEFAULT, icfg(cfg).get("quad", {}) or {})


def images(ctx, split: str, offset: int, n: int) -> list:
    idxs = ctx.sp[split][int(offset):int(offset) + int(n)]
    if len(idxs) < int(n):
        raise SystemExit(f"split {split} không đủ {n} ảnh từ vị trí {offset}")
    return idxs


def one_path(E, X: torch.Tensor, ds: IC.DSet, qc: dict, tag: dict, gen_rand) -> tuple[list, list]:
    nl = NestedLattice(E.fn, X, E.ctx.batch)
    nl.refine()
    nl.refine()
    proj = lambda A: ds.project(A)[0]
    conv, hist = False, []
    while True:
        L = nl.level                                   # có mid(0..L−1)
        if L >= 2:
            a, b = proj(nl.mid(L - 1)), proj(nl.mid(L - 2))
            e = float((a - b).norm() / a.norm().clamp_min(1e-300))
            hist.append({**tag, "level": L - 1, "M": 1 << (L - 1), "rel_proj_mid_step": e, "grads": nl.grads})
            if e < float(qc["tol_ref"]):
                conv = True
                break
        if L >= int(qc["L_max"]):
            break
        nl.refine()
    Lf = nl.level
    ref = nl.mid(Lf - 1)
    pref = proj(ref)
    nref = float(pref.norm())
    rows = []

    def add(rule, M, A, grads):
        pa = proj(A)
        rows.append({**tag, "rule": rule, "M": M, "grads_path": grads, "rel_proj": float((pa - pref).norm()) / max(nref, 1e-300),
                     "rel_l2": float((A - ref).norm() / ref.norm().clamp_min(1e-300)),
                     "compl_gap": (float(A.sum()) - E.D) / abs(E.D) if E.D else math.nan, "ref_level": Lf - 1,
                     "ref_M": 1 << (Lf - 1), "ref_converged": conv})
    N = X.shape[0] - 1
    for L in range(Lf + 1):
        add("left", 1 << L, nl.left(L), N << L)
    for L in range(Lf - 1):
        add("mid", 1 << L, nl.mid(L), N << L)
    for M in qc["rand_M"]:
        add("rand", int(M), integrate(E.fn, X, int(M), "rand", gen=gen_rand(int(M)), batch=E.ctx.batch), N * int(M))
    rows.append({**tag, "rule": "ref", "M": 1 << (Lf - 1), "grads_path": nl.grads, "rel_proj": 0.0, "rel_l2": 0.0,
                 "compl_gap": (float(ref.sum()) - E.D) / abs(E.D) if E.D else math.nan, "ref_level": Lf - 1,
                 "ref_M": 1 << (Lf - 1), "ref_converged": conv})
    return rows, hist


def run(ctx) -> None:
    cfg = ctx.cfg
    qc = qcfg(cfg)
    out = out_dir(cfg, "quad")
    name, spec = primary_spec(cfg)
    idxs = images(ctx, qc["split"], qc["offset"], qc["n_images"])
    dg = IC._h({"quad": qc, "D": spec, "seed": cfg["seed"], "model": cfg["model"], "data": cfg["data"], "v": 1})
    allr, allh = [], []
    t0, n0 = time.time(), Counter.n
    for j, idx in enumerate(idxs):
        image = f"seg{idx:05d}"
        cp = os.path.join(out, "cache", f"{image}.pt")
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        if os.path.exists(cp) and torch.load(cp, weights_only=False).get("digest") == dg:
            obj = torch.load(cp, weights_only=False)
            allr += obj["rows"]
            allh += obj["hist"]
            print(f"  [quad] {image}: đã có, bỏ qua")
            continue
        print(f"  [quad] ảnh {j + 1}/{len(idxs)} {image}", flush=True)
        E = IC.IEnv(ctx, idx)
        ds = IC.DSet(name, IC.make_parts(E, spec, int(qc["n_sets"]), "quad"))      # chỉ cần ô (không cần ΔF)
        rows, hist = [], []
        for alpha in qc["alphas"]:
            for p in range(int(qc["paths_per_alpha"])):
                Z = SM.spi_schedule(float(alpha), int(qc["N"]), E.shape, E.gen("INFQ", "spi", alpha, qc["N"], image, p))
                X = SM.nodes(E.x0, E.dx, Z)
                tag = {"image": image, "path_kind": "spi", "alpha": float(alpha), "N": int(qc["N"]), "path": p}
                r, h = one_path(E, X, ds, qc, tag, lambda M, a=alpha, p=p: E.gen("INFQ-rand", a, image, p, M))
                rows += r
                hist += h
                del Z, X
        if qc.get("ig"):
            X = straight_nodes(E.x, E.x0, int(qc["ig_N"]))
            tag = {"image": image, "path_kind": "ig", "alpha": math.inf, "N": int(qc["ig_N"]), "path": 0}
            r, h = one_path(E, X, ds, qc, tag, lambda M: E.gen("INFQ-rand", "ig", image, M))
            rows += r
            hist += h
        torch.save({"digest": dg, "rows": rows, "hist": hist}, cp)
        allr += rows
        allh += hist
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    import pandas as pd
    pd.DataFrame(allr).to_csv(os.path.join(out, "quad_errors.csv"), index=False)
    pd.DataFrame(allh).to_csv(os.path.join(out, "quad_levels.csv"), index=False)
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"time": _dt.datetime.now().isoformat(timespec="seconds"), "images": [f"seg{i:05d}" for i in idxs], "quad": qc,
                   "minutes": (time.time() - t0) / 60, "grad_evals": Counter.n - n0, "digest": dg, "debug": cfg["_debug"]},
                  fh, indent=1, default=str)


def recommend(er, tol: float) -> list:
    """Theo (path_kind, α, rule): M nhỏ nhất có q95(rel_proj) ≤ tol."""
    out = []
    e = er[er.rule.isin(["left", "mid", "rand"])]
    for (pk, a, rule), g in e.groupby(["path_kind", "alpha", "rule"]):
        q = g.groupby("M").rel_proj.quantile(0.95).sort_index()
        ok = q[q <= tol]
        out.append({"path_kind": pk, "alpha": a, "rule": rule, "M_min": int(ok.index[0]) if len(ok) else None,
                    "q95_at_M1": float(q.get(1, np.nan)), "q95_at_M_min": float(ok.iloc[0]) if len(ok) else np.nan,
                    "q95_best": float(q.min())})
    return out


def write_report(cfg: dict) -> str:
    import pandas as pd
    from .report import _read, md
    out = out_dir(cfg, "quad")
    qc = qcfg(cfg)
    pr = protocol(cfg)
    rp = os.path.join(out, "INFID_QUAD_REPORT.md")
    er = _read(out, "quad_errors")
    L = ["# E-quadrature — C3: tích phân đã hội tụ?\n"]
    if cfg.get("_debug"):
        L.append("> **DEBUG** — chỉ kiểm pipeline.\n")
    if er is None:
        L.append("_Chưa có kết quả — `bash scripts/run.sh quad-check`._\n")
        open(rp, "w").write("\n".join(L) + "\n")
        return rp
    ref = er[er.rule == "ref"]
    L.append(f"- {er.image.nunique()} ảnh; α ∈ {qc['alphas']}, N = {qc['N']}, {qc['paths_per_alpha']} đường / α" +
             (f"; IG N = {qc['ig_N']}" if qc.get("ig") else "") + ".")
    L.append(f"- Tham chiếu = điểm giữa ở mức mịn nhất; tinh dần tới khi sai số chiếu giữa hai mức liên tiếp < {qc['tol_ref']} "
             f"hoặc tới L_max = {qc['L_max']}. Hội tụ: {int(ref.ref_converged.astype(bool).sum())}/{len(ref)} đường; M tham chiếu "
             f"{sorted(set(int(v) for v in ref.ref_M))}.")
    if (~ref.ref_converged.astype(bool)).any():
        L.append("- **Có đường chưa hội tụ ở L_max** — sai số dưới đây là so với tham chiếu chưa hội tụ (cận dưới của sai số thật); "
                 "tăng `infid.quad.L_max` trước khi khoá quy tắc.")
    L.append("")
    L.append("## Sai số chiếu ‖P_D(T_M − T_ref)‖ / ‖P_D T_ref‖ (trung vị / q95 qua ảnh × đường)\n")
    e = er[er.rule.isin(["left", "mid", "rand"])]
    tab = e.groupby(["path_kind", "alpha", "rule", "M"]).agg(median=("rel_proj", "median"), q95=("rel_proj", lambda v: v.quantile(0.95)),
                                                              rel_l2_med=("rel_l2", "median"), compl_gap_med=("compl_gap", "median"),
                                                              grads=("grads_path", "first")).reset_index()
    L.append(md(tab))
    L.append(f"## Khuyến nghị (tol = {qc['tol']})\n")
    L.append(md(pd.DataFrame(recommend(er, float(qc["tol"])))))
    L.append(f"Protocol hiện tại: rule = `{pr.get('rule')}`, M = {pr.get('M')}. Chọn M nhỏ nhất đạt tol cho mọi α sẽ dùng ở E1 "
             "rồi khoá bằng `bash scripts/run.sh infid-lock`. Điểm giữa không chính xác với ReLU; khoảng hở completeness ≈ 0 ở mọi quy tắc "
             "không phải bằng chứng hội tụ.\n")
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[quad] {rp}")
    return rp
