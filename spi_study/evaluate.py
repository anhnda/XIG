"""
Tính metric từ attribution đã lưu (tách khỏi bước tính attribution, chạy lại được).

Mỗi (ảnh, spec, mức) ghi một dòng b='mean' (tham chiếu của mức = trung bình các sub-estimate) và,
nếu metrics.on_subs, một dòng cho mỗi sub-estimate b = 0…n_sub−1 (để có sai số chuẩn của metric).

--line: thêm LINE(s) = (1 − s)·A_IG + s·A_SPI(1) từ các tham chiếu ở mức cao nhất đã đủ của
ig_shift và spi_dp α=1, với s = 2h, h = 1/(α+1) cho mọi α trong collapse.alphas. Sub-estimate của
LINE ghép cặp theo b.
"""

from __future__ import annotations

import time
from fractions import Fraction

import torch

from .config import beta_spec, canon, frac_str, line_spec
from .metrics import Evaluator
from .reference import complete_levels, find_task, levels_of, n_sub_of, tasks_of
from .setting import Setting


def _line_sources(cfg, st: Setting):
    ig = canon(cfg["gate1"]["ig_ref"])
    a1 = beta_spec(cfg["collapse"]["alpha_line"])
    images = st.image_ids()
    out = {}
    for spec in (ig, a1):
        task = find_task(cfg, spec)
        done = complete_levels(st.store, images, spec, levels_of(cfg, task, st.store), n_sub_of(cfg, spec))
        if not done:
            raise SystemExit(f"LINE cần tham chiếu {spec} cho mọi ảnh; chưa có")
        out[spec] = done[-1]
    return ig, a1, out


def run_eval(st: Setting, group: str, line: bool = False) -> None:
    cfg, store = st.cfg, st.store
    mc = cfg["metrics"]
    tasks = tasks_of(cfg, group)
    ds = st.dataset()
    bb = st.backbone()
    images = st.indices()
    done = store.metric_keys()
    seed = cfg["seed"]
    if line:
        ig, a1, lv = _line_sources(cfg, st)
        s_list = [Fraction(2, int(a) + 1) for a in cfg["collapse"]["alphas"]]
        if Fraction(2, int(cfg["collapse"]["alpha_line"]) + 1) not in s_list:
            s_list.append(Fraction(2, int(cfg["collapse"]["alpha_line"]) + 1))   # s = 1: chính SPI(1)
    print(f"[eval] {group} | {st.tag} | {len(images)} ảnh" + (" | +LINE" if line else ""))
    t_all = time.time()
    for n_img, i in enumerate(images):
        item = ds.get(i)
        im = item.image_id
        if not store.has_meta(im):
            print(f"  {im}: chưa có attribution, bỏ qua")
            continue
        meta = store.load_meta(im)
        x = item.x.to(st.device)
        ev = Evaluator(mc, bb, x, meta["class"], im, item.index, item.mask, item.valid,
                       float(meta["D"]), seed, expensive=True)

        def emit(spec, L, K, N, b, A, extra=None):
            if (im, spec, L, b) in done:
                return
            t0 = time.time()
            row = {"image": im, "index": item.index, "spec": spec, "level": L, "K": K, "N": N, "b": b,
                   "n_backward": (K * N if b == "mean" else (K * N) // max(1, n_sub)), "D": meta["D"]}
            if extra:
                row.update(extra)
            row.update(ev(A))
            row["eval_time"] = time.time() - t0
            store.append("metrics", row)

        for task in tasks:
            spec = task["spec"]
            n_sub = n_sub_of(cfg, spec)
            for L, (K, N) in enumerate(levels_of(cfg, task, store)):
                if not all(store.has_attr(im, spec, L, b) for b in range(n_sub)):
                    continue
                need = [("mean", None)]
                if mc.get("on_subs", True) and n_sub > 1:
                    need += [(b, b) for b in range(n_sub)]
                if all((im, spec, L, b) in done for b, _ in need):
                    continue
                A_mean, subs = store.load_level(im, spec, L, n_sub, device=st.device)
                for b, j in need:
                    emit(spec, L, K, N, b, A_mean if j is None else subs[j])
        if line:
            n_sub = n_sub_of(cfg, ig)
            A_ig, s_ig = store.load_level(im, ig, lv[ig], n_sub, device=st.device)
            A_1, s_1 = store.load_level(im, a1, lv[a1], n_sub, device=st.device)
            Kig, Nig = levels_of(cfg, find_task(cfg, ig), store)[lv[ig]]
            K1, N1 = levels_of(cfg, find_task(cfg, a1), store)[lv[a1]]
            for s in s_list:
                spec = canon(f"line:s={frac_str(s)}")
                sf = float(s)
                extra = {"ig_level": lv[ig], "spi1_level": lv[a1], "s": sf}
                emit(spec, 0, 1, Kig * Nig + K1 * N1, "mean", (1 - sf) * A_ig + sf * A_1, extra)
                if mc.get("on_subs", True):
                    for b in range(n_sub):
                        emit(spec, 0, 1, Kig * Nig + K1 * N1, b, (1 - sf) * s_ig[b] + sf * s_1[b], extra)
        print(f"  [{n_img + 1}/{len(images)}] {im} xong ({time.time() - t_all:.0f}s)")
    print(f"[eval] xong sau {time.time() - t_all:.0f}s")


__all__ = ["run_eval", "line_spec"]