"""
Data collapse (protocol 5.2, 6.6, Cổng 2).

Ba họ cùng E Z(t) = t và Var Z(t) = h·t(1−t), h = 1/(α+1):
    beta  spi_dp:alpha=α,node=rand
    mix   coord_mix:lam=h,node=rand
    line  line:s=2h      (A_s = (1−s)·A_IG + s·A_SPI(1), từ tham chiếu)

Chỉ số thu gọn cho cặp (f, g): CI_fg(h) = ‖A_f − A_g‖₂ / ‖A_beta − A_IG‖₂, trung vị qua ảnh;
kèm ‖A_f − A_g‖₂ / ‖A_IG‖₂. Với collapse.debias: true, mọi ‖·‖² được khử nhiễu Monte Carlo
(trừ phương sai theo phần tử ước lượng từ sub-estimate), vì nhiễu theo phần tử của SPI ở 64 đường
cỡ 20–40% ‖A‖ và lấn át khác biệt thật giữa các họ. Chất lượng Q = IoU của tham chiếu. SE của ΔQ = SE ghép cặp qua ảnh
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
from .reference import debiased_sq, level_var, n_sub_of, top_level
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

    def load(self, image, spec) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (tham chiếu, phương sai Monte Carlo theo phần tử của nó), float64."""
        name, p = parse_spec(spec)
        if name == "line":
            s = num(p["s"])
            ig = canon(self.cfg["gate1"]["ig_ref"])
            a1 = beta_spec(self.cfg["collapse"]["alpha_line"])
            Ai, Vi = self.load(image, ig)
            A1, V1 = self.load(image, a1)
            return (1 - s) * Ai + s * A1, (1 - s) ** 2 * Vi + s ** 2 * V1
        A, subs = self.st.store.load_level(image, spec, self.level(spec), n_sub_of(self.cfg, spec),
                                           device=self.st.device)
        V = level_var(subs) if len(subs) > 1 else torch.zeros_like(A, dtype=torch.float64)
        return A.double(), V


def _sq(A, VA, B, VB, debias: bool, block: int) -> tuple[float, float]:
    """‖A − B‖² (khử nhiễu: trừ V_A + V_B) và SE của nó."""
    d2 = (A - B).pow(2)
    if not debias:
        return float(d2.sum()), 0.0
    return debiased_sq(d2, VA + VB, block)


def family_specs(alpha) -> dict:
    return {"beta": beta_spec(alpha), "mix": mix_spec(alpha), "line": line_spec(alpha)}


def run_collapse(st: Setting) -> dict:
    cfg = st.cfg
    cc = cfg["collapse"]
    images = st.image_ids()
    tab = Table(st.store.rows("metrics"))
    rc = RefCache(st)
    debias = bool(cc.get("debias", False))
    block = int(cfg["convergence"].get("block", 28))
    ig = canon(cfg["gate1"]["ig_ref"])
    alphas = [int(a) for a in cc["alphas"]]
    eval_alphas = set(int(a) for a in cc["eval_alphas"])
    qmetric = cc.get("q_metric", "iou")
    etab = Table(st.store.rows("energy"))
    if qmetric in ("energy_deb", "energy_raw", "iou_inf_sub"):
        if not etab.t:
            raise SystemExit("collapse.q_metric cần energy.jsonl: chạy `energy --group stage1b --line` trước")
        qtab = etab
    else:
        qtab = tab
    iou_tbl = ["\n## IoU (chỉ để đọc: bị nhiễu MC kéo xuống, mức chệch khác nhau giữa các họ)\n",
               "| h | họ | IoU tham chiếu | IoU ngoại suy trong mức (Q + (Q − Q̄_sub)/3) |", "|---|---|---|---|"]
    out = {"alphas": {}, "s_eps": {}, "q_metric": qmetric}
    lines = [f"# Data collapse — {st.tag} — {len(images)} ảnh\n",
             ("Chuẩn L2 khử nhiễu: ‖ΔA‖² − V_f − V_g (V từ sub-estimate). Ảnh có mẫu số không vượt 2·SE của "
              "chính nó bị loại khỏi CI_fg (NaN).\n") if debias else "Chuẩn L2 thô (chưa khử nhiễu).\n",
             f"Q = `{qmetric}`" + (" (tỉ lệ năng lượng Σ(Ā² − V) trong mask, không chệch do nhiễu MC).\n"
                                   if qmetric == "energy_deb" else ".\n")]
    tbl = ["| h | cặp | trung vị CI_fg | trung vị ‖ΔA‖/‖A_IG‖ | Q_f | Q_g | ΔQ | SE | CI 95% ΔQ | thu gọn | tách |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    verdict_collapse_all, verdict_split_any = True, False

    for a in alphas:
        h = Fraction(1, a + 1)
        specs = family_specs(a)
        norms = {k: [] for k in ("ref",)}
        pair_ci = {pq: [] for pq in itertools.combinations(FAMS, 2)}
        pair_rel = {pq: [] for pq in pair_ci}
        n_measurable = 0
        for im in images:
            A_ig, V_ig = rc.load(im, ig)
            AV = {f: rc.load(im, specs[f]) for f in FAMS}
            den_sq, den_se = _sq(AV["beta"][0], AV["beta"][1], A_ig, V_ig, debias, block)
            # mẫu số chỉ dùng được khi phần có hệ thống của ‖A_beta − A_IG‖² vượt rõ nhiễu của chính nó
            ok_den = den_sq > 0 and (not debias or den_sq > 2 * den_se)
            n_measurable += int(ok_den)
            den = den_sq ** 0.5 if ok_den else float("nan")
            nig = float(A_ig.norm())
            norms["ref"].append(den)
            for f, g in pair_ci:
                num_sq, _ = _sq(AV[f][0], AV[f][1], AV[g][0], AV[g][1], debias, block)
                num_ = max(0.0, num_sq) ** 0.5
                pair_ci[(f, g)].append(num_ / den if ok_den else float("nan"))
                pair_rel[(f, g)].append(num_ / nig if nig > 0 else float("nan"))
        Q, rec_iou = {}, {}
        for f in FAMS:
            spec = specs[f]
            L = 0 if f == "line" else rc.level(spec)
            Q[f] = qtab.values(spec, L, qmetric, images)
            rec_iou[f] = {"iou": float(np.nanmean(tab.values(spec, L, "iou", images))),
                          "iou_inf_sub": float(np.nanmean(etab.values(spec, L, "iou_inf_sub", images)))
                          if etab.t else float("nan")}
            iou_tbl.append(f"| 1/{a + 1} | {f} | {rec_iou[f]['iou']:.4f} | {rec_iou[f]['iou_inf_sub']:.4f} |")
        rec = {"h": float(h), "specs": specs, "pairs": {}, "iou": rec_iou}
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
        rec["median_norm_beta_minus_ig"] = float(np.nanmedian(norms["ref"])) if n_measurable else float("nan")
        rec["n_images_measurable_denominator"] = n_measurable
        tbl.append(f"| 1/{a + 1} | (mẫu số ‖A_beta − A_IG‖ đo được trên {n_measurable}/{len(images)} ảnh) "
                   "| | | | | | | | | |")
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
            gen = generator(seed_of(cfg["seed"], "s_eps", spec), st.device)
            vals[f] = s_eps(smp, float(cc["eps"]), int(se_cfg["paths"]), (int(se_cfg["d"]),), gen,
                            chunk=int(se_cfg.get("chunk", 5)))
        vals["ig"] = s_eps(make_sampler("ig_mid", {}, int(se_cfg["N"])), float(cc["eps"]), 1, (1,),
                           generator(0, st.device))
        out["s_eps"][str(a)] = vals
        sl.append(f"| 1/{a + 1} | {vals['beta']:.4f} | {vals['mix']:.4f} | {vals['line']:.4f} | {vals['ig']:.4f} |")

    lines.insert(1, f"## Kết luận: **{verdict}**\n\n{msg}\n")
    lines += [f"## Chỉ số thu gọn và chất lượng (Q = {qmetric})\n"] + tbl + iou_tbl + sl
    lines.append("\nTham chiếu: mức cao nhất đã đủ của từng spec; LINE từ tham chiếu IG và SPI(1).")
    text = "\n".join(lines) + "\n"
    path = st.store.write_report("collapse", text, out)
    print(text)
    print(f"[collapse] -> {path}")
    return out