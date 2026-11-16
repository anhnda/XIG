"""Gom các bảng JSONL thành report.md (trung bình qua ảnh, kèm sai số chuẩn khi cần)."""

from __future__ import annotations

import collections
import json
import math
import os

import numpy as np


def _rows(out: str, table: str) -> list[dict]:
    p = os.path.join(out, f"{table}.jsonl")
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _m(v) -> float:
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    return float(v.mean()) if len(v) else float("nan")


def _se(v) -> float:
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    return float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")


def _med(v) -> float:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.median(v)) if len(v) else float("nan")


def _fmt(x, p=4):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{p}f}"


def report_a(out: str) -> list[str]:
    q = _rows(out, "a_quality")
    if not q:
        return []
    by = collections.defaultdict(list)
    for r in q:
        by[r["method"]].append(r)
    ref = {r["image"]: r["energy_total"] for r in q if r["method"] == "ig/canonical"}
    metrics = [m for m in ("energy_mask", "iou", "insdel_blur", "insdel_black", "road") if m in q[0]]
    L = [f"## A. Đường đi × quy tắc chia credit ({len(ref)} ảnh)\n",
         "| phương pháp | " + " | ".join(metrics) + " | năng lượng / IG (trung vị) | ‖A‖₁ / abs(ΣA) (trung vị) | trung vị abs residual |",
         "|---|" + "---|" * (len(metrics) + 3)]
    for k in sorted(by):
        rs = by[k]
        ratio = _med([r["energy_total"] / ref[r["image"]] for r in rs if ref.get(r["image"])])
        L.append(f"| {k} | " + " | ".join(f"{_fmt(_m([r.get(m) for r in rs]))} ± {_fmt(_se([r.get(m) for r in rs]), 3)}"
                                         for m in metrics)
                 + f" | {_fmt(ratio, 4)} | {_fmt(_med([r['cancel'] for r in rs]), 1)} | {_fmt(_med([abs(r['residual']) for r in rs]), 3)} |")
    # phân rã: hiệu ứng quy tắc theo từng đường (ghép cặp theo ảnh)
    paths = sorted({k.split("/")[0] for k in by})
    L += ["\n**Hiệu ứng quy tắc (idgi − canonical, ghép cặp theo ảnh), theo từng đường:**\n",
          "| đường | " + " | ".join(f"Δ{m}" for m in metrics) + " |", "|---|" + "---|" * len(metrics)]
    idx = {(r["method"], r["image"]): r for r in q}
    for p in paths:
        cells = []
        for m in metrics:
            d = [idx[(f"{p}/idgi", im)].get(m) - idx[(f"{p}/canonical", im)].get(m)
                 for (k, im) in idx if k == f"{p}/canonical" and (f"{p}/idgi", im) in idx
                 and idx[(f"{p}/idgi", im)].get(m) is not None and idx[(f"{p}/canonical", im)].get(m) is not None]
            cells.append(f"{_m(d):+.4f} ± {_fmt(_se(d), 4)}")
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    L += ["\n**Hiệu ứng đường (so với ig, cùng quy tắc, ghép cặp theo ảnh):**\n",
          "| đường | quy tắc | " + " | ".join(f"Δ{m}" for m in metrics) + " |", "|---|---|" + "---|" * len(metrics)]
    for p in paths:
        if p == "ig":
            continue
        for rule in ("canonical", "idgi"):
            cells = []
            for m in metrics:
                d = [idx[(f"{p}/{rule}", im)].get(m) - idx[(f"ig/{rule}", im)].get(m)
                     for (k, im) in idx if k == f"ig/{rule}" and (f"{p}/{rule}", im) in idx
                     and idx[(f"{p}/{rule}", im)].get(m) is not None and idx[(f"ig/{rule}", im)].get(m) is not None]
                cells.append(f"{_m(d):+.4f} ± {_fmt(_se(d), 4)}")
            L.append(f"| {p} | {rule} | " + " | ".join(cells) + " |")
    pr = _rows(out, "a_pairs")
    if pr:
        g = collections.defaultdict(list)
        for r in pr:
            g[(r["a"], r["b"])].append(r)
        L += ["\n**Độ tương đồng giữa các bản đồ** (cos có dấu khử nhiễu; Spearman và top-5% trên Σ_c|A|):\n",
              "| a | b | cos_deb | spearman | top-k |", "|---|---|---|---|---|"]
        for (a, b), rs in sorted(g.items(), key=lambda kv: -_m([r["cos_deb"] for r in kv[1]])):
            L.append(f"| {a} | {b} | {_fmt(_m([r['cos_deb'] for r in rs]), 3)} | "
                     f"{_fmt(_m([r['spearman'] for r in rs]), 3)} | {_fmt(_m([r['topk'] for r in rs]), 3)} |")
    return L


def report_b(out: str) -> list[str]:
    b = _rows(out, "b_traps")
    if not b:
        return []
    g = collections.defaultdict(list)
    for r in b:
        g[(r["shape"], r["u"], r["rho"], r["tau"])].append(r)
    methods = sorted({r["method"] for r in b})
    L = ["\n## B. Bẫy gradient\n",
         "R = ‖A(F+H) − A(F)‖ / ‖A(F)‖ (khử nhiễu MC). Với κ đã chọn, IG/canonical có R ≈ tau theo cấu tạo khi "
         "shape = line. `pred` = suy giảm dự đoán của SPI so với IG (chỉ line). H_noise, H_other = max |H|/|D| tại "
         "x + nhiễu và tại ảnh khác (kiểm bẫy có vô hại với hành vi không).\n"]
    for key in sorted(g):
        rs = g[key]
        shape, uk, rho, tau = key
        bym = collections.defaultdict(list)
        for r in rs:
            bym[r["method"]].append(r)
        n_img = len({r["image"] for r in rs})
        L += [f"\n**{shape}, u = {uk}, ρ = {rho}, τ = {tau}** ({n_img} ảnh; H_noise ≤ {_fmt(max(r['H_noise_rel'] for r in rs), 3)}, "
              f"H_other ≤ {_fmt(max(r['H_other_rel'] for r in rs), 3)}"
              + (f", sai số công thức IG {_fmt(_med([r.get('ig_formula_err') for r in rs if 'ig_formula_err' in r]), 3)}" if shape == "line" else "")
              + ")\n",
              "| phương pháp | R | R / thay đổi của IG | dự đoán | ΔE | ΔIoU |", "|---|---|---|---|---|---|"]
        for k in methods:
            if k not in bym:
                continue
            v = bym[k]
            pred = _m([r.get("pred_attenuation") for r in v]) if any("pred_attenuation" in r for r in v) else None
            L.append(f"| {k} | {_fmt(_m([r['R'] for r in v]), 3)} ± {_fmt(_se([r['R'] for r in v]), 3)} | "
                     f"{_fmt(_m([r['R_vs_ig_change'] for r in v]), 3)} | {_fmt(pred, 3) if pred is not None else ''} | "
                     f"{_fmt(_m([r['dE'] for r in v]), 4)} | {_fmt(_m([r['dIoU'] for r in v]), 4)} |")
    return L


def report_c(out: str) -> list[str]:
    c = _rows(out, "c_signal")
    if not c:
        return []
    g = collections.defaultdict(list)
    for r in c:
        g[(r["loc"], r["sign"], r["method"])].append(r)
    L = ["\n## C. Giữ tín hiệu đã biết\n",
         "retention = ⟨ΔA, A*⟩/‖A*‖² (1 = giữ đúng), err = ‖ΔA − A*‖/‖A*‖, leak = phần chuẩn của ΔA nằm ngoài R, "
         "sign_ok = tổng ΔA trên R cùng dấu với S. Phương pháp canonical tuyến tính nên retention = 1 theo cấu tạo.\n",
         "| vị trí | dấu | phương pháp | retention | err | leak | sign_ok |", "|---|---|---|---|---|---|---|"]
    for key in sorted(g):
        v = g[key]
        L.append(f"| {key[0]} | {key[1]:+d} | {key[2]} | {_fmt(_m([r['retention'] for r in v]), 3)} ± {_fmt(_se([r['retention'] for r in v]), 3)} | "
                 f"{_fmt(_m([r['err'] for r in v]), 3)} | {_fmt(_m([r['leak'] for r in v]), 3)} | "
                 f"{_m([float(r['sign_ok']) for r in v]):.2f} |")
    return L


def report_sanity(out: str) -> list[str]:
    s = _rows(out, "sanity")
    if not s:
        return []
    g = collections.defaultdict(list)
    for r in s:
        g[(r["depth"], r["layers"], r["method"])].append(r)
    L = ["\n## Sanity: khởi tạo lại tích luỹ theo tầng (giá trị cao = bản đồ KHÔNG đổi khi mô hình bị phá)\n",
         "| tầng | phương pháp | spearman | cos_deb | top-k |", "|---|---|---|---|---|"]
    for key in sorted(g):
        v = g[key]
        L.append(f"| {key[1]} | {key[2]} | {_fmt(_m([r['spearman'] for r in v]), 3)} | "
                 f"{_fmt(_m([r['cos_deb'] for r in v]), 3)} | {_fmt(_m([r['topk'] for r in v]), 3)} |")
    return L


def write(out: str) -> str:
    info = {}
    p = os.path.join(out, "run_info.json")
    if os.path.exists(p):
        with open(p) as fh:
            info = json.load(fh)
    L = [f"# attr_probe — {out}\n", f"config sha256: `{info.get('config_sha256', '?')[:16]}`, debug = {info.get('debug')}, "
         f"split = {info.get('split')}\n"]
    L += report_a(out) + report_b(out) + report_c(out) + report_sanity(out)
    path = os.path.join(out, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[report] -> {path}")
    return path
