"""
Data collapse (protocol 5.2, 6.6, Cổng 2).

Ba họ cùng E Z(t) = t và Var Z(t) = h·t(1−t), h = 1/(α+1):
    beta  spi_dp:alpha=α,node=rand
    mix   coord_mix:lam=h,node=rand
    line  line:s=2h      (A_s = (1−s)·A_IG + s·A_SPI(1), từ tham chiếu)

Chỉ số thu gọn cho cặp (f, g): CI_fg(h) = ‖A_f − A_g‖₂ / ‖A_beta − A_IG‖₂, trung vị qua ảnh;
kèm ‖A_f − A_g‖₂ / ‖A_IG‖₂. Chất lượng Q = IoU của tham chiếu. SE của ΔQ = SE ghép cặp qua ảnh
(sd/√n) — protocol không nói rõ SE nào; đây là lựa chọn của bộ code.

Tiêu chí (xét tại collapse.eval_alphas):
    thu gọn  : tại mọi h, mọi cặp có trung vị CI < ci_collapse và |ΔQ| < max(2·SE, dq_floor)
    tách rời : tại một h nào đó, có cặp với trung vị CI > ci_split, hoặc |ΔQ| > dq_split với CI 95% không chứa 0
    còn lại  : chưa kết luận
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import torch

from .config import beta_spec, canon, line_spec, mix_spec, num, parse_spec
from .reference import n_sub_of, top_level
from .samplers import make_sampler, s_eps
from .seeding import generator, seed_of
from .setting import Setting
from .stats import Table, pct_ci

FAMS = ("beta", "mix", "line")


class RefCache:
    def __init__(self, st: Setting):
        self.st, self.cfg = st, st.cfg
        self.top = {}

    def level(self, spec):
        if spec not in self.top:
            self.top[spec] = top_level(self.cfg, self.st, spec)
        return self.top[spec]

    def load(self, image, spec) -> torch.Tensor:
        name, p = parse_spec(spec)
        if name == "line":
            s = num(p["s"])
            ig = canon(self.cfg["gate1"]["ig_ref"])
            a1 = beta_spec(self.cfg["collapse"]["alpha_line"])
            return ((1 - s) * self.load(image, ig).double() + s * self.load(image, a1).double()).float()
        A, _ = self.st.store.load_level(image, spec, self.level(spec), n_sub_of(self.cfg, spec))
        return A


def family_specs(alpha) -> dict:
    return {"beta": beta_spec(alpha), "mix": mix_spec(alpha), "line": line_spec(alpha)}


def run_collapse(st: Setting) -> dict:
    cfg = st.cfg
    cc = cfg["collapse"]
    images = st.image_ids()
    tab = Table(st.store.rows("metrics"))
    rc = RefCache(st)
    ig = canon(cfg["gate1"]["ig_ref"])
    alphas = [int(a) for a in cc["alphas"]]
    eval_alphas = set(int(a) for a in cc["eval_alphas"])
    out = {"alphas": {}, "s_eps": {}}
    lines = [f"# Data collapse — {st.tag} — {len(images)} ảnh\n"]
    tbl = ["| h | cặp | trung vị CI_fg | trung vị ‖ΔA‖/‖A_IG‖ | Q_f | Q_g | ΔQ | SE | CI 95% ΔQ | thu gọn | tách |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    verdict_collapse_all, verdict_split_any = True, False

    for a in alphas:
        h = Fraction(1, a + 1)
        specs = family_specs(a)
        norms = {k: [] for k in ("ref",)}
        pair_ci = {pq: [] for pq in itertools.combinations(FAMS, 2)}
        pair_rel = {pq: [] for pq in pair_ci}
        for im in images:
            A_ig = rc.load(im, ig).double()
            A = {f: rc.load(im, specs[f]).double() for f in FAMS}
            den = float((A["beta"] - A_ig).norm())
            nig = float(A_ig.norm())
            norms["ref"].append(den)
            for f, g in pair_ci:
                num_ = float((A[f] - A[g]).norm())
                pair_ci[(f, g)].append(num_ / den if den > 0 else float("nan"))
                pair_rel[(f, g)].append(num_ / nig if nig > 0 else float("nan"))
        Q = {}
        for f in FAMS:
            spec = specs[f]
            L = 0 if f == "line" else rc.level(spec)
            Q[f] = tab.values(spec, L, "iou", images)
        rec = {"h": float(h), "specs": specs, "pairs": {}}
        for (f, g) in pair_ci:
            med_ci = float(np.nanmedian(pair_ci[(f, g)]))
            med_rel = float(np.nanmedian(pair_rel[(f, g)]))
            dq = Q[f] - Q[g]
            dq = dq[~np.isnan(dq)]
            mdq = float(dq.mean()) if len(dq) else float("nan")
            se = float(dq.std(ddof=1) / math.sqrt(len(dq))) if len(dq) > 1 else float("nan")
            lo, hi = pct_ci(dq, 0.95, int(cfg["gate1"]["n_boot"]), int(cfg["gate1"]["boot_seed"]))
            col = (med_ci < float(cc["ci_collapse"])) and (abs(mdq) < max(2 * se, float(cc["dq_floor"])))
            spl = (med_ci > float(cc["ci_split"])) or (abs(mdq) > float(cc["dq_split"]) and (lo > 0 or hi < 0))
            if a in eval_alphas:
                verdict_collapse_all &= bool(col)
                verdict_split_any |= bool(spl)
            rec["pairs"][f"{f}-{g}"] = {"median_ci": med_ci, "median_rel_ig": med_rel, "Q_f": float(np.nanmean(Q[f])),
                                        "Q_g": float(np.nanmean(Q[g])), "dQ": mdq, "se": se, "ci95": [lo, hi],
                                        "collapse": bool(col), "split": bool(spl), "evaluated": a in eval_alphas}
            tbl.append(f"| 1/{a + 1}{'' if a in eval_alphas else ' (không xét)'} | {f}–{g} | {med_ci:.3f} | "
                       f"{med_rel:.4f} | {np.nanmean(Q[f]):.4f} | {np.nanmean(Q[g]):.4f} | {mdq:+.4f} | {se:.4f} | "
                       f"[{lo:+.4f}, {hi:+.4f}] | {'✓' if col else '✗'} | {'✓' if spl else '✗'} |")
        rec["median_norm_beta_minus_ig"] = float(np.median(norms["ref"]))
        out["alphas"][str(a)] = rec

    if verdict_split_any:
        verdict = "SPLIT"
        msg = ("Tách rời → xác định yếu tố gây tách trước Giai đoạn 2: beta–mix khác nhau ⇒ hình dạng phân phối "
               "biên; chỉ line lệch (beta–mix gần nhau) ⇒ phụ thuộc giữa tọa độ.")
    elif verdict_collapse_all:
        verdict = "COLLAPSE"
        msg = "Thu gọn → Giai đoạn 2 với luận điểm 'mức trải quyết định'."
    else:
        verdict = "INCONCLUSIVE"
        msg = "Chưa kết luận → tăng ảnh hoặc làm dày lưới h ĐÚNG MỘT LẦN."
    out["verdict"], out["message"] = verdict, msg

    # S_ε
    se_cfg = cc["s_eps"]
    sl = ["\n## Thống kê hình dạng S_ε (ε = %s), Monte Carlo từ sampler\n" % cc["eps"],
          "| h | beta | mix | line | IG |", "|---|---|---|---|---|"]
    for a in alphas:
        vals = {}
        for f, spec in family_specs(a).items():
            name, p = parse_spec(spec)
            smp = make_sampler(name, p, int(se_cfg["N"]))
            gen = generator(seed_of(cfg["seed"], "s_eps", spec), "cpu")
            vals[f] = s_eps(smp, float(cc["eps"]), int(se_cfg["paths"]), (int(se_cfg["d"]),), gen,
                            chunk=int(se_cfg.get("chunk", 5)))
        vals["ig"] = s_eps(make_sampler("ig_mid", {}, int(se_cfg["N"])), float(cc["eps"]), 1, (1,),
                           generator(0, "cpu"))
        out["s_eps"][str(a)] = vals
        sl.append(f"| 1/{a + 1} | {vals['beta']:.4f} | {vals['mix']:.4f} | {vals['line']:.4f} | {vals['ig']:.4f} |")

    lines.insert(1, f"## Kết luận: **{verdict}**\n\n{msg}\n")
    lines += ["## Chỉ số thu gọn và chất lượng (IoU)\n"] + tbl + sl
    lines.append("\nTham chiếu: mức cao nhất đã đủ của từng spec; LINE từ tham chiếu IG và SPI(1).")
    text = "\n".join(lines) + "\n"
    path = st.store.write_report("collapse", text, out)
    print(text)
    print(f"[collapse] -> {path}")
    return out
