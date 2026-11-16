"""
Tỉ lệ năng lượng trong mask, khử nhiễu MC (Q chính cho 1B, độ nhạy cho Cửa 1).

Chỉ đọc attribution đã lưu và mask, không cần mô hình hay GPU. Ghi <root>/energy.jsonl, một dòng cho
mỗi (ảnh, spec, mức): energy_deb (khử nhiễu, từ trung bình và phương sai của các sub-estimate),
energy_raw (thô) và iou_inf_sub (IoU ngoại suy trong cùng mức: Q + (Q − Q̄_sub)/3, chỉ hợp lệ khi
nhiễu nhỏ, lấy từ metrics.jsonl nếu có). Cột b = 'mean' để dùng chung lớp Table.

--line: thêm LINE(s), với V_s = (1−s)²·V_IG + s²·V_SPI(1) (hai tham chiếu độc lập).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .config import canon, frac_str
from .evaluate import _line_sources
from .metrics import energy_frac
from .reference import level_var, levels_of, n_sub_of, tasks_of, find_task
from .setting import Setting
from .stats import Table


def _iou_inf(tab: Table, spec, L, im, n_sub):
    q = tab.get(spec, L, "mean").get(im, {}).get("iou")
    subs = [tab.get(spec, L, b).get(im, {}).get("iou") for b in range(n_sub)]
    if q is None or n_sub < 2 or any(s is None for s in subs):
        return float("nan")
    return float(q + (q - np.mean(subs)) / 3.0)


def run_energy(st: Setting, group: str, line: bool = False) -> None:
    cfg, store = st.cfg, st.store
    images = st.image_ids()
    done = {(r["image"], r["spec"], r["level"]) for r in store.rows("energy")}
    tab = Table(store.rows("metrics"))
    n_new = 0
    for task in tasks_of(cfg, group):
        spec = task["spec"]
        B = n_sub_of(cfg, spec)
        for L, (K, N) in enumerate(levels_of(cfg, task, store)):
            for im in images:
                if (im, spec, L) in done or not all(store.has_attr(im, spec, L, b) for b in range(B)):
                    continue
                mask, valid = store.load_mask(im)
                A, subs = store.load_level(im, spec, L, B)
                V = level_var(subs) if B > 1 else None
                store.append("energy", {"image": im, "spec": spec, "level": L, "K": K, "N": N, "b": "mean",
                                        "energy_deb": energy_frac(A, mask, valid, V),
                                        "energy_raw": energy_frac(A, mask, valid, None),
                                        "iou_inf_sub": _iou_inf(tab, spec, L, im, B)})
                done.add((im, spec, L))
                n_new += 1
    if line:
        ig, a1, lv = _line_sources(cfg, st)
        B = n_sub_of(cfg, ig)
        Kig, Nig = levels_of(cfg, find_task(cfg, ig), store)[lv[ig]]
        K1, N1 = levels_of(cfg, find_task(cfg, a1), store)[lv[a1]]
        s_list = sorted({Fraction(2, int(a) + 1) for a in cfg["collapse"]["alphas"]}
                        | {Fraction(2, int(cfg["collapse"]["alpha_line"]) + 1)})
        for im in images:
            mask, valid = store.load_mask(im)
            Ai, si = store.load_level(im, ig, lv[ig], B)
            A1, s1 = store.load_level(im, a1, lv[a1], B)
            Vi, V1 = level_var(si), level_var(s1)
            for s in s_list:
                spec = canon(f"line:s={frac_str(s)}")
                if (im, spec, 0) in done:
                    continue
                sf = float(s)
                A = (1 - sf) * Ai.double() + sf * A1.double()
                V = (1 - sf) ** 2 * Vi + sf ** 2 * V1
                store.append("energy", {"image": im, "spec": spec, "level": 0, "K": 1, "N": Kig * Nig + K1 * N1,
                                        "b": "mean", "s": sf, "ig_level": lv[ig], "spi1_level": lv[a1],
                                        "energy_deb": energy_frac(A, mask, valid, V),
                                        "energy_raw": energy_frac(A, mask, valid, None),
                                        "iou_inf_sub": _iou_inf(tab, spec, 0, im, B)})
                n_new += 1
    print(f"[energy] {group} | {st.tag}: +{n_new} dòng -> {store.p('energy.jsonl')}")
