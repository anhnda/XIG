"""
Tính attribution theo thang (K, N) và kiểm hội tụ (protocol 6.3).

Mỗi mức (K, N) được tính thành n_sub sub-estimate độc lập, mỗi cái K/n_sub đường (cùng chi phí với
một lần K đường). Tham chiếu của mức = trung bình các sub-estimate; sai số chuẩn của metric =
độ lệch chuẩn giữa sub-estimate / √n_sub. IG_mid (tất định) có một sub-estimate.

Tiêu chí hội tụ (giữa hai mức cao nhất đã đủ cho mọi ảnh):
    trung vị qua ảnh ‖A_top − A_prev‖₂ / ‖A_top‖₂ ≤ rel_l2   và   trung bình |IoU_top − IoU_prev| ≤ iou.
Không đạt → `converge --extend` thêm một mức (quy tắc `extend` của thang) rồi chạy lại `attr`.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .attribute import estimate, residual
from .config import canon, num, parse_spec
from .metrics import energy_frac, iou_topk, make_baseline, tie_perm
from .model import ImageGradF, endpoint
from .samplers import make_sampler
from .seeding import generator, seed_of
from .setting import Setting

EXIT_EXTENDED = 3


def tasks_of(cfg: dict, group: str) -> list[dict]:
    if group not in cfg["groups"]:
        raise SystemExit(f"nhóm '{group}' không có trong config.groups ({sorted(cfg['groups'])})")
    out = []
    for t in cfg["groups"][group]["tasks"]:
        t = dict(t)
        t["spec"] = canon(t["spec"])
        out.append(t)
    return out


def levels_of(cfg: dict, task: dict, store) -> list[tuple[int, int]]:
    spec = task["spec"]
    name, params = parse_spec(spec)
    if "ladder" in task:
        base = [tuple(map(int, l)) for l in cfg["ladders"][task["ladder"]]["levels"]]
        return base + [tuple(map(int, l)) for l in store.extensions().get(spec, [])]
    if "levels" in task:
        return [tuple(map(int, l)) for l in task["levels"]]
    if name == "ig_mid":
        return [(1, int(params["N"]))]
    raise SystemExit(f"task {spec}: cần 'ladder' hoặc 'levels'")


def n_sub_of(cfg: dict, spec: str) -> int:
    return 1 if parse_spec(spec)[0] == "ig_mid" else int(cfg["estimator"]["n_sub"])


def rng_device(cfg: dict, compute_device: torch.device) -> torch.device:
    return torch.device("cpu") if cfg["estimator"].get("rng", "device") == "cpu" else compute_device


# ─────────────────────────────────────────────────────────────────────────────
# meta theo ảnh
# ─────────────────────────────────────────────────────────────────────────────

def ensure_meta(st: Setting, item, x: torch.Tensor, xb: torch.Tensor) -> dict:
    store = st.store
    if store.has_meta(item.image_id):
        meta = store.load_meta(item.image_id)
        if meta["target_rule"] != st.target or meta["model"] != st.model_desc():
            raise SystemExit(f"meta {item.image_id} không khớp setting hiện tại")
        return meta
    bb = st.backbone()
    with torch.no_grad():
        logits = bb(x)[0]
        prob = torch.softmax(logits, dim=-1)
    pred = int(prob.argmax())
    cls = pred if st.target == "predicted" else int(item.label)
    fx, fb = endpoint(bb, x, xb, cls)
    meta = {"image": item.image_id, "index": item.index, "source": item.source, "class": cls,
            "pred": pred, "label": item.label, "prob_class": float(prob[cls]), "f_x": fx, "f_base": fb,
            "D": fx - fb, "target_rule": st.target, "model": st.model_desc(),
            "baseline": st.cfg["baseline"]}
    store.save_meta(item.image_id, meta, item.mask, item.valid)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# tính attribution
# ─────────────────────────────────────────────────────────────────────────────

def run_attr(st: Setting, group: str) -> None:
    cfg, store = st.cfg, st.store
    tasks = tasks_of(cfg, group)
    ds = st.dataset()
    bb = st.backbone()
    idx = st.indices()
    batch = int(cfg["model"]["grad_batch"])
    rdev = rng_device(cfg, st.device)
    print(f"[attr] {group} | {st.tag} | {len(idx)} ảnh | {len(tasks)} task | device={st.device} rng={rdev}")
    t_all = time.time()
    for n_img, i in enumerate(idx):
        item = ds.get(i)
        x = item.x.to(st.device)
        xb = make_baseline(x, cfg["baseline"])
        meta = ensure_meta(st, item, x, xb)
        D = float(meta["D"])
        gF = ImageGradF(bb, x, xb, meta["class"])
        shape = tuple(x.shape[1:])
        for task in tasks:
            spec = task["spec"]
            name, params = parse_spec(spec)
            B = n_sub_of(cfg, spec)
            for L, (K, N) in enumerate(levels_of(cfg, task, store)):
                if K % B:
                    raise SystemExit(f"{spec} mức {L}: K={K} không chia hết cho n_sub={B}")
                Kb = K // B
                sampler = make_sampler(name, params, N)
                for b in range(B):
                    if store.has_attr(item.image_id, spec, L, b):
                        continue
                    seed = seed_of(cfg["seed"], item.image_id, spec, L, K, N, b)
                    gen = generator(seed, rdev)
                    gF.n_bwd = 0
                    if st.device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                    A = estimate(gF, sampler, Kb, gen, shape, batch, dtype=x.dtype)
                    if st.device.type == "cuda":
                        torch.cuda.synchronize()
                    dt = time.time() - t0
                    res = residual(D, A)
                    store.save_attr(item.image_id, spec, L, b, A,
                                    {"spec": spec, "level": L, "K": K, "N": N, "b": b, "paths": Kb, "seed": seed})
                    store.append("compute", {"image": item.image_id, "index": item.index, "spec": spec,
                                             "level": L, "K": K, "N": N, "b": b, "paths": Kb,
                                             "n_backward": gF.n_bwd, "residual": res, "time": dt,
                                             "seed": seed, "D": D})
                    print(f"  [{n_img + 1}/{len(idx)}] {item.image_id} {spec} L{L}({K}×{N}) b{b}: "
                          f"bwd={gF.n_bwd} res={res:+.4f} t={dt:.1f}s")
    print(f"[attr] xong sau {time.time() - t_all:.0f}s")


# ─────────────────────────────────────────────────────────────────────────────
# hội tụ
# ─────────────────────────────────────────────────────────────────────────────

def complete_levels(store, images: list[str], spec: str, levels: list, n_sub: int) -> list[int]:
    return [L for L in range(len(levels)) if store.complete(images, spec, L, n_sub)]


def top_level(cfg: dict, st: Setting, spec: str, task: dict | None = None) -> int:
    """Mức cao nhất đã đủ cho mọi ảnh (dùng làm tham chiếu)."""
    if task is None:
        task = find_task(cfg, spec)
    levs = levels_of(cfg, task, st.store)
    done = complete_levels(st.store, st.image_ids(), spec, levs, n_sub_of(cfg, spec))
    if not done:
        raise SystemExit(f"{st.tag}: chưa có mức nào đủ cho {spec}")
    return done[-1]


def find_task(cfg: dict, spec: str) -> dict:
    spec = canon(spec)
    for g in cfg["groups"].values():
        for t in g["tasks"]:
            if canon(t["spec"]) == spec:
                t = dict(t)
                t["spec"] = spec
                return t
    raise SystemExit(f"không tìm thấy task cho {spec} trong config.groups")


def _next_level(cfg: dict, task: dict, levs: list, n_ext: int):
    rule = cfg["ladders"][task["ladder"]].get("extend", "double")
    if rule == "double":
        K, N = levs[-1]
        return (2 * K, 2 * N)
    if isinstance(rule, list):
        return tuple(rule[n_ext]) if n_ext < len(rule) else None
    raise SystemExit(f"extend không hợp lệ: {rule}")


def level_var(subs: list[torch.Tensor]) -> torch.Tensor:
    """Phương sai Monte Carlo theo phần tử của trung bình các sub-estimate: s²/n_sub (float64)."""
    S = torch.stack([s.double() for s in subs])
    return S.var(dim=0, unbiased=True) / S.shape[0]


def debiased_sq(diff_sq: torch.Tensor, var: torch.Tensor, block: int) -> tuple[float, float]:
    """Σ(diff² − var) và sai số chuẩn của nó theo khối block×block (khối coi như độc lập).
    diff_sq, var: (C, H, W)."""
    e = (diff_sq - var).sum(dim=0)                                   # (H, W)
    H, W = e.shape
    hb, wb = H // block, W // block
    blocks = e[:hb * block, :wb * block].reshape(hb, block, wb, block).sum(dim=(1, 3)).flatten()
    est = float(e.sum())
    se = float(blocks.std(unbiased=True)) * float(blocks.numel()) ** 0.5 if blocks.numel() > 1 else float("nan")
    return est, se


def rel_change(At, subs_t, Ap, subs_p, block: int) -> tuple[float, float, float, float]:
    """-> (rel L2 thô, rel L2 khử nhiễu, cận trên (ước lượng + 2·SE), nhiễu của A_top tương đối).

    E‖Ā_top − Ā_prev‖² = ‖bias_top − bias_prev‖² + V_top + V_prev (hai mức độc lập), nên
    ‖Ā_top − Ā_prev‖² − V_top − V_prev ước lượng không chệch phần thay đổi có hệ thống giữa hai mức."""
    At, Ap = At.double(), Ap.double()
    n = float(At.norm().clamp_min(1e-30))
    d2 = (At - Ap).pow(2)
    raw = float(d2.sum()) ** 0.5 / n
    if len(subs_t) < 2 or len(subs_p) < 2:
        return raw, raw, raw, 0.0
    vt, vp = level_var(subs_t), level_var(subs_p)
    est, se = debiased_sq(d2, vt + vp, block)
    deb = max(0.0, est) ** 0.5 / n
    hi = max(0.0, est + 2 * se) ** 0.5 / n
    return raw, deb, hi, float(vt.sum()) ** 0.5 / n


def run_converge(st: Setting, group: str, extend: bool = False) -> int:
    cfg, store = st.cfg, st.store
    cc = cfg["convergence"]
    mode = cc.get("mode", "raw")
    block = int(cc.get("block", 28))
    images = st.image_ids()
    qm = cc.get("q_metric", "iou")
    crit = ("‖Ā_top − Ā_prev‖/‖Ā_top‖ khử nhiễu (trừ V_top + V_prev từ sub-estimate)" if mode == "debiased"
            else "‖A_top − A_prev‖/‖A_top‖ thô")
    qcrit = (f"|trung bình có dấu ΔE_khử_nhiễu| ≤ {cc['q_tol']}" if qm == "energy_deb"
             else f"trung bình |ΔIoU| ≤ {cc['iou']}")
    lines = [f"# Hội tụ — {group} — {st.tag}\n",
             f"Tiêu chí ({mode}, {qm}): trung vị {crit} ≤ {cc['rel_l2']}, {qcrit}. {len(images)} ảnh.\n",
             "Cột 'cận trên' = trung vị của √(ước lượng + 2·SE)/‖Ā_top‖ (SE theo khối "
             f"{block}×{block}); 'nhiễu top' = √V_top/‖Ā_top‖. E = tỉ lệ năng lượng Σ(Ā² − V) trong mask. "
             "ΔIoU chỉ để đọc khi q_metric = energy_deb (IoU bị nhiễu MC làm chệch xuống).\n",
             "| spec | mức (K×N) top / prev | rel L2 thô | rel L2 khử nhiễu | cận trên | nhiễu top | "
             "ΔE (có dấu ± SE) | abs ΔE | abs ΔIoU | trạng thái |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    result, extended = {}, False
    seed = cfg["seed"]
    masks = {im: store.load_mask(im) for im in images if store.has_meta(im)}
    for task in tasks_of(cfg, group):
        if "ladder" not in task:
            continue
        spec = task["spec"]
        B = n_sub_of(cfg, spec)
        levs = levels_of(cfg, task, store)
        done = complete_levels(store, images, spec, levs, B)
        n_ext = len(store.extensions().get(spec, []))
        if len(done) < len(levs) or len(done) < 2:
            status = "chưa đủ mức (chạy attr)"
            result[spec] = {"status": "incomplete", "levels": levs, "complete": done}
            lines.append(f"| {spec} | – | – | – | – | – | – | – | – | {status} |")
            continue
        top, prev = done[-1], done[-2]
        raw, deb, hi, noise, diou, de = [], [], [], [], [], []
        for im in images:
            At, st_ = store.load_level(im, spec, top, B)
            Ap, sp_ = store.load_level(im, spec, prev, B)
            r, d, u, z = rel_change(At, st_, Ap, sp_, block)
            raw.append(r); deb.append(d); hi.append(u); noise.append(z)
            mask, valid = masks[im]
            perm = tie_perm(im, mask.numel(), seed)
            diou.append(abs(iou_topk(At, mask, perm, valid) - iou_topk(Ap, mask, perm, valid)))
            de.append(energy_frac(At, mask, valid, level_var(st_) if B > 1 else None)
                      - energy_frac(Ap, mask, valid, level_var(sp_) if B > 1 else None))
        med_raw, med_deb = float(np.median(raw)), float(np.median(deb))
        med_hi, med_noise = float(np.median(hi)), float(np.median(noise))
        mean_diou = float(np.nanmean(diou))
        de = np.array(de, dtype=np.float64)
        de_mean, de_abs = float(np.nanmean(de)), float(np.nanmean(np.abs(de)))
        de_se = float(np.nanstd(de, ddof=1) / np.sqrt(np.sum(~np.isnan(de)))) if len(de) > 1 else float("nan")
        med_rel = med_deb if mode == "debiased" else med_raw
        q_ok = abs(de_mean) <= float(cc["q_tol"]) if qm == "energy_deb" else mean_diou <= cc["iou"]
        ok = med_rel <= cc["rel_l2"] and q_ok
        status = "HỘI TỤ" if ok else "chưa hội tụ"
        if not ok:
            if extend and n_ext < int(cc["max_extra_levels"]):
                new = _next_level(cfg, task, levs, n_ext)
                if new is not None:
                    store.add_extension(spec, new)
                    extended = True
                    status += f" → thêm mức {new[0]}×{new[1]}"
                else:
                    status += " (hết quy tắc mở rộng)"
            elif n_ext >= int(cc["max_extra_levels"]):
                status += " (đã dùng hết số mức mở rộng: báo cáo kèm cờ)"
        result[spec] = {"status": "converged" if ok else "not_converged", "mode": mode, "top": top, "prev": prev,
                        "top_KN": levs[top], "prev_KN": levs[prev], "median_rel_l2": med_rel,
                        "median_rel_l2_raw": med_raw, "median_rel_l2_debiased": med_deb,
                        "median_rel_l2_upper": med_hi, "median_noise_top": med_noise,
                        "mean_abs_diou": mean_diou, "q_metric": qm, "mean_dE": de_mean, "se_dE": de_se,
                        "mean_abs_dE": de_abs, "rel_l2_raw_per_image": raw,
                        "rel_l2_debiased_per_image": deb, "n_extensions": n_ext}
        lines.append(f"| {spec} | {levs[top][0]}×{levs[top][1]} / {levs[prev][0]}×{levs[prev][1]} | "
                     f"{med_raw:.4f} | {med_deb:.4f} | {med_hi:.4f} | {med_noise:.4f} | "
                     f"{de_mean:+.4f} ± {de_se:.4f} | {de_abs:.4f} | {mean_diou:.4f} | {status} |")
    text = "\n".join(lines) + "\n"
    path = store.write_report(f"convergence_{group}", text, result)
    print(text)
    print(f"[converge] -> {path}")
    return EXIT_EXTENDED if extended else 0


def task_alpha(spec: str) -> float:
    return num(parse_spec(spec)[1]["alpha"])