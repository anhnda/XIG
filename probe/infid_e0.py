"""
E0 — `pathscore-infid-decomp`: audit PATHSCORE theo infidelity, KHÔNG gradient mới (chỉ forward cho ΔF_S).

Từ cache attr.pt (bản đồ từng đường SPI, R seed × K đường) của PATHSCORE, cho từng ảnh:
  1. Tái lập 96 ô gốc (cùng generator) và ΔF_S; so infid từng đường / SPI với scores.pt ⇒ xác minh đơn vị và cách tổng hợp
     54,1 / 25,6 (trung bình theo ảnh của tỷ số từng ảnh; tỷ số gộp chỉ là phân tích độ nhạy).
  2. Phân rã theo ảnh: TB infid các đường = infid(SPI) + phương sai giữa các đường (đẳng thức chính xác); σ̂² (ddof 1),
     L̂(μ) = infid(SPI) − σ̂²/K, tỷ trọng phương sai σ̂²/K trong infid(SPI).
  3. Đường cong infid(Â_K) theo K = 1..K (tập con ngẫu nhiên không hoàn lại) và hồi quy theo 1/K; C2: K·V(K) từ nhóm rời nhau.
  4. Trên tập ô MỚI (D quần thể, cùng phân phối): tập fit (n_fit) và tập đánh giá độc lập (n_eval): σ̂², L̂, J(K);
     sàn L*_free, L*_comp ở mức patch G×G (G ∈ grids), λ chọn bằng CV trên tập fit, báo in-sample và out-of-sample.
  5. rank(P) của 96 ô; khoảng hở completeness; infid_β (co giãn tối ưu — chỉ chẩn đoán thang đo).
  6. G0 (cổng phân bổ tài nguyên một chiều): dư địa J_indep(K_G0) − L*_comp,patch là CẬN DƯỚI của dư địa thật.
  7. Chẩn đoán làm mượt khối (không phải phương pháp mới): S_h A = trung bình A trong khối h×h px (bảo toàn tổng mỗi khối ⇒ bảo
     toàn ΣA; KHÔNG giữ dummy). Đo L_D(A), L_D(S_h A), L̂(S_h μ) trên tập ô đánh giá độc lập. Giảm mạnh ⇒ phân bố không gian ở
     độ phân giải cao đóng góp vào sai số; gần như không giảm ⇒ giả thuyết nhiễu tần số cao yếu đi.
Đầu ra: <out>/pilot/infid/e0/{e0_images.csv, e0_kcurve.csv, e0_groups.csv, INFID_E0_REPORT.md}.
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
from . import infid_stats as ST
from . import pathscore as PS
from . import sampling as SM
from .pathbank import settings

F64 = torch.float64
E0_VERSION = 1
E0_DEFAULT = {"n_fit": 1536, "n_eval": 512, "grids": [14, 28], "lams": [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1], "folds": 4,
              "subsets_per_K": 200, "K_g0": 16, "allow_stale_attr": False, "smooth": [8, 16, 32]}


def icfg(cfg: dict) -> dict:
    ic = cfg.get("infid")
    if not ic:
        raise SystemExit("config thiếu mục infid:")
    return ic


def e0cfg(cfg: dict) -> dict:
    return C._merge(E0_DEFAULT, icfg(cfg).get("e0", {}) or {})


def out_dir(cfg: dict, sub: str) -> str:
    d = os.path.join(C.out_dir(cfg, "pilot"), "infid", sub)
    os.makedirs(d, exist_ok=True)
    return d


def primary_spec(cfg: dict) -> tuple[str, dict]:
    D = icfg(cfg)["D"]
    name = D["primary"]
    return name, D["sets"][name]


def protocol(cfg: dict) -> dict:
    return icfg(cfg).get("protocol", {}) or {}


def e0_digest(cfg: dict, attr_dg: str) -> str:
    return IC._h({"v": E0_VERSION, "e0": e0cfg(cfg), "D": icfg(cfg)["D"], "attr": attr_dg, "Ks": protocol(cfg).get("Ks"),
                  "seed": cfg["seed"]})


# ─────────────────────────────────────────────────────────────────────────────
# một ảnh
# ─────────────────────────────────────────────────────────────────────────────

def kcurve(Y: torch.Tensor, d: torch.Tensor, draws: int, g: torch.Generator) -> list:
    """Y (K, n) của một seed: E infid(trung bình K' đường lấy không hoàn lại), K' = 1..K."""
    K = Y.shape[0]
    out = []
    for k in range(1, K + 1):
        if k == K:
            vals = IC.infid(Y.mean(0, keepdim=True), d)
        else:
            idx = torch.stack([torch.randperm(K, generator=g)[:k] for _ in range(int(draws))]).to(Y.device)
            vals = IC.infid(Y[idx].mean(1), d)
        out.append((k, float(vals.mean())))
    return out


def e0_image(ctx, cfg: dict, pc: dict, idx: int, cdir: str, attr: dict, scores: dict | None) -> dict:
    ec = e0cfg(cfg)
    pr = protocol(cfg)
    Ks = [int(k) for k in pr.get("Ks", [1, 2, 4, 8, 16])]
    E = IC.IEnv(ctx, idx)
    dev = E.x.device
    A = attr["A"]
    R, K = int(A.shape[0]), int(A.shape[1])
    a = A.to(dev).double().sum(2)                                   # (R, K, H, W) bản đồ pixel có dấu
    sumA = a.sum((2, 3))                                            # (R, K)
    row = {**E.info(), "R": R, "K": K}

    # 1. 96 ô gốc của PATHSCORE
    sub = PS.xcfg(pc)["subset"]
    boxes = IC.square_boxes(E.H, E.W, int(sub["n"]), [int(s) for s in sub["sizes"]], E.cgen("pathscore-subset", E.image))
    p96 = IC.Part("deletion", "black", torch.ones(len(boxes), dtype=F64), boxes[:, 2].clone(), boxes=boxes)
    p96.dF = IC.compute_dF(E, p96, E.ins_batch)
    ds96 = IC.DSet("pathscore96", [p96])
    d96 = ds96.dF.to(dev)
    Y96 = ds96.project(a.reshape(R * K, E.H, E.W), summed=True).view(R, K, -1)
    Lp96 = IC.infid(Y96.reshape(R * K, -1), d96).view(R, K)
    Ls96 = torch.stack([IC.infid(Y96[r].mean(0, keepdim=True), d96)[0] for r in range(R)])
    if scores is not None:
        cp = {(int(x["seed"]), int(x["path"])): float(x["infid"]) for x in scores["paths"] if "infid" in x}
        ce = {int(x["seed"]): float(x["infid"]) for x in scores["ens"] if "infid" in x}
        if cp:
            row["csv_absdiff_path_max"] = max(abs(float(Lp96[r, k]) - cp[(r, k)]) for r in range(R) for k in range(K) if (r, k) in cp)
        if ce:
            row["csv_absdiff_spi_max"] = max(abs(float(Ls96[r]) - ce[r]) for r in range(R) if r in ce)
    seeds = [IC.indep_core(Y96[r], d96, Ks) for r in range(R)]
    for key in ("mean_path_infid", "L_mean", "sigma2", "L_mu", "var_share_at_P", "identity_gap"):
        row[f"p96_{key}"] = float(np.mean([s[key] for s in seeds]))
    row["p96_identity_gap_absmax"] = float(max(abs(s["identity_gap"]) for s in seeds))
    row["p96_spi_infid"] = float(Ls96.mean())
    row["p96_den"] = float((d96 * d96).sum())
    pooled = IC.indep_core(Y96.reshape(R * K, -1), d96, Ks)
    for k in Ks:
        row[f"p96_J_{k}"] = pooled[f"J_{k}"]
    for k in (1, 2, 4, 8, 16):
        if f"KV_{k}" in pooled:
            row[f"p96_KV_{k}"] = pooled[f"KV_{k}"]
    row["rank_P96"] = IC.mask_rank(E, ds96)
    row["n96"] = int(d96.numel())
    row["p96_infid_beta_spi"] = float(np.mean([float(IC.infid_beta(Y96[r].mean(0, keepdim=True), d96)[0]) for r in range(R)]))
    gap = (sumA - E.D) / abs(E.D) if E.D else torch.full_like(sumA, math.nan)
    row["compl_gap_path_absmean"] = float(gap.abs().mean())
    row["compl_gap_spi_abs"] = float(((sumA.mean(1) - E.D) / abs(E.D)).abs().mean()) if E.D else math.nan

    g = E.cgen("infid-e0-kcurve", E.image)
    kc = []
    for r in range(R):
        cur = kcurve(Y96[r], d96, int(ec["subsets_per_K"]), g)
        xs = np.array([1.0 / k for k, _ in cur])
        ys = np.array([v for _, v in cur])
        slope, icp = np.polyfit(xs, ys, 1)
        for k, v in cur:
            kc.append({"image": E.image, "seed": r, "set": "pathscore96", "K": k, "J_emp": v,
                       "J_formula": seeds[r]["L_mu"] + seeds[r]["sigma2"] / k, "fit_slope": slope, "fit_intercept": icp,
                       "sigma2": seeds[r]["sigma2"], "L_mu": seeds[r]["L_mu"]})

    # 2. tập ô mới: fit / đánh giá
    name, spec = primary_spec(cfg)
    ds_f = IC.dset_cached(E, cdir, name, spec, int(ec["n_fit"]), "e0-fit", E.ins_batch)
    ds_e = IC.dset_cached(E, cdir, name, spec, int(ec["n_eval"]), "e0-eval", E.ins_batch)
    de = ds_e.dF.to(dev)
    Ye = ds_e.project(a.reshape(R * K, E.H, E.W), summed=True)
    ev = IC.indep_core(Ye, de, Ks)
    for key in ("mean_path_infid", "L_mean", "sigma2", "L_mu", "var_share_at_P", "den"):
        row[f"ev_{key}"] = ev[key]
    for k in Ks:
        row[f"ev_J_{k}"] = ev[f"J_{k}"]
        row[f"ev_var_{k}"] = ev["sigma2"] / k
    row["ev_spi_infid_seed"] = float(np.mean([float(IC.infid(Ye.view(R, K, -1)[r].mean(0, keepdim=True), de)[0]) for r in range(R)]))
    row["ev_infid_beta_spi"] = float(IC.infid_beta(Ye.mean(0, keepdim=True), de)[0])
    row["ev_infid_beta_path_mean"] = float(IC.infid_beta(Ye, de).mean())
    gr = []
    grp = ds_e.group.to(dev)
    for gv in sorted(set(grp.tolist())):
        ix = torch.nonzero(grp == gv, as_tuple=True)[0]
        if len(ix) >= 2:
            s = IC.indep_core(Ye[:, ix], de[ix], Ks)
            gr.append({"image": E.image, "group": int(gv), "n_sets": int(len(ix)), "share_den": s["den"] / ev["den"],
                       **{k: s[k] for k in ("sigma2", "L_mu", "mean_path_infid", "var_share_at_P")},
                       **{f"J_{k}": s[f"J_{k}"] for k in Ks}})
    Kg = int(ec["K_g0"])
    for G in ec["grids"]:
        Xf = IC.patch_design(E, ds_f, int(G))
        Xe = IC.patch_design(E, ds_e, int(G))
        fl = IC.floor_fit(Xf, ds_f.dF.to(dev), Xe, de, E.D, [float(x) for x in ec["lams"]], int(ec["folds"]),
                          E.cgen("infid-e0-cv", E.image, G))
        for k, v in fl.items():
            row[f"G{G}_{k}"] = v
        row[f"G{G}_margin_J{Kg}"] = ev.get(f"J_{Kg}", ev["L_mu"] + ev["sigma2"] / Kg) - fl["L_comp_oos"]
        row[f"G{G}_bias_part"] = ev["L_mu"] - fl["L_comp_oos"]
    sm = [{"image": E.image, "h": 1, "G": None, "L_mu": ev["L_mu"], "sigma2": ev["sigma2"], "L_spi": row["ev_spi_infid_seed"],
           "L_path": ev["mean_path_infid"], "infid_beta_spi": row["ev_infid_beta_spi"]}]
    for h in ec["smooth"]:
        G = max(1, int(round(E.H / int(h))))
        a_s = block_average(a.reshape(R * K, E.H, E.W), G)
        Ys = ds_e.project(a_s, summed=True)
        st = IC.indep_core(Ys, de, Ks, c2_sizes=())
        sm.append({"image": E.image, "h": int(h), "G": G, "L_mu": st["L_mu"], "sigma2": st["sigma2"],
                   "L_spi": float(np.mean([float(IC.infid(Ys.view(R, K, -1)[r].mean(0, keepdim=True), de)[0]) for r in range(R)])),
                   "L_path": st["mean_path_infid"], "infid_beta_spi": float(IC.infid_beta(Ys.mean(0, keepdim=True), de)[0]),
                   "sum_err": float((a_s.sum((1, 2)) - a.reshape(R * K, E.H, E.W).sum((1, 2))).abs().max())})
        for k in ("L_comp_oos", "L_free_oos"):
            if f"G{G}_{k}" in row:
                sm[-1][f"floor_{k}"] = row[f"G{G}_{k}"]
    return {"img": row, "kcurve": kc, "groups": gr, "smooth": sm}


def block_average(a: torch.Tensor, G: int) -> torch.Tensor:
    """a (P, H, W) → trung bình trong từng khối của lưới G×G (khối cuối nhận phần dư, như sampling.grid_pixel_labels).
    Bảo toàn tổng mỗi khối ⇒ bảo toàn tổng toàn ảnh."""
    P_, H, W = a.shape
    lab = SM.grid_pixel_labels(H, W, G).to(a.device)
    tot = torch.zeros(P_, G * G, dtype=a.dtype, device=a.device).index_add_(1, lab, a.reshape(P_, -1))
    cnt = torch.zeros(G * G, dtype=a.dtype, device=a.device).index_add_(0, lab, torch.ones_like(lab, dtype=a.dtype))
    return (tot / cnt.view(1, -1))[:, lab].view(P_, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# chạy
# ─────────────────────────────────────────────────────────────────────────────

def run(ctx) -> None:
    cfg = ctx.cfg
    S = settings(cfg)
    pc = PS.pcfg(cfg)
    ec = e0cfg(cfg)
    dg_ps = PS.digests(cfg, S, pc)
    ps_out = PS.out_dir(cfg)
    out = out_dir(cfg, "e0")
    idxs = ctx.sp["pilot"][:int(pc["n_images"])]
    rows = {"img": [], "kcurve": [], "groups": [], "smooth": []}
    t0, f0 = time.time(), SM.FwdCounter.n
    for j, idx in enumerate(idxs):
        image = f"seg{idx:05d}"
        ap = os.path.join(ps_out, "cache", image, "attr.pt")
        if not os.path.exists(ap):
            raise SystemExit(f"[e0] thiếu {ap} — chạy bash scripts/run.sh pathscore trước")
        attr = torch.load(ap, weights_only=False)
        if attr.get("digest") != dg_ps["attr"]:
            msg = f"[e0] {image}: attr.pt không khớp hash config hiện tại của PATHSCORE"
            if not ec["allow_stale_attr"]:
                raise SystemExit(msg + " (đặt infid.e0.allow_stale_attr: true nếu cố ý dùng cache cũ)")
            print("  CẢNH BÁO " + msg)
        sp = os.path.join(ps_out, "cache", image, "scores.pt")
        scores = torch.load(sp, weights_only=False) if os.path.exists(sp) else None
        cdir = os.path.join(out, "cache", image)
        os.makedirs(cdir, exist_ok=True)
        cp = os.path.join(cdir, "e0.pt")
        dg = e0_digest(cfg, str(attr.get("digest")))
        res = None
        if os.path.exists(cp):
            obj = torch.load(cp, weights_only=False)
            if obj.get("digest") == dg:
                res = obj["res"]
                print(f"  [e0] {image}: đã có, bỏ qua")
        if res is None:
            print(f"  [e0] ảnh {j + 1}/{len(idxs)} {image}", flush=True)
            ti = time.time()
            res = e0_image(ctx, cfg, pc, idx, cdir, attr, scores)
            torch.save({"digest": dg, "res": res}, cp)
            print(f"  [e0] {image}: {time.time() - ti:.0f}s", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for k in rows:
            rows[k] += [res[k]] if k == "img" else res.get(k, [])
    import pandas as pd
    for k, name in (("img", "e0_images"), ("kcurve", "e0_kcurve"), ("groups", "e0_groups"), ("smooth", "e0_smooth")):
        if rows[k]:
            pd.DataFrame(rows[k]).to_csv(os.path.join(out, f"{name}.csv"), index=False)
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"time": _dt.datetime.now().isoformat(timespec="seconds"), "images": [f"seg{i:05d}" for i in idxs],
                   "e0": ec, "D": icfg(cfg)["D"], "minutes": (time.time() - t0) / 60, "fwd": SM.FwdCounter.n - f0,
                   "pathscore_attr_digest": dg_ps["attr"], "debug": cfg["_debug"]}, fh, indent=1, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ci(t) -> str:
    m, lo, hi = t
    return f"{m:.4g} [{lo:.4g}, {hi:.4g}]" if np.isfinite(lo) else (f"{m:.4g}" if np.isfinite(m) else "—")


def write_report(cfg: dict) -> str:
    import pandas as pd
    from .report import _read, fmt, md
    out = out_dir(cfg, "e0")
    rp = os.path.join(out, "INFID_E0_REPORT.md")
    pr = protocol(cfg)
    ec = e0cfg(cfg)
    B, seed, al = int(pr.get("bootstrap", 10000)), int(pr.get("seed", 23)), float(pr.get("alpha", 0.05))
    im = _read(out, "e0_images")
    L = ["# E0 — Audit PATHSCORE theo infidelity (không gradient mới)\n"]
    if cfg.get("_debug"):
        L.append("> **DEBUG** — chỉ kiểm pipeline.\n")
    if im is None:
        L.append("_Chưa có kết quả — `bash scripts/run.sh pathscore-infid-decomp`._\n")
        open(rp, "w").write("\n".join(L) + "\n")
        return rp
    n = len(im)
    K = int(im.K.iloc[0])

    def ci(col, df=im):
        m, lo, hi = ST.boot_ci(df[col].values, B, seed, al)
        return m, lo, hi

    def cs(col, df=im, f="{:.4g}"):
        m, lo, hi = ci(col, df)
        return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]" if np.isfinite(lo) else f.format(m)

    L.append(f"{n} ảnh PATHSCORE, R = {int(im.R.iloc[0])} seed × K = {K} đường SPI (α, N như PATHSCORE; quy tắc trong cache = "
             "canon_rand, xem docs/P0_AUDIT.md). Mọi đại lượng tính theo TỪNG ảnh rồi trung bình theo ảnh; CI 95% bootstrap theo ảnh.\n")

    L.append("## 1. Tái lập đơn vị infid (96 ô gốc)\n")
    if "csv_absdiff_path_max" in im:
        L.append(f"- Sai khác tối đa so với scores.pt: từng đường {fmt(im.csv_absdiff_path_max.max())}, SPI "
                 f"{fmt(im.get('csv_absdiff_spi_max', pd.Series([np.nan])).max())} (≈ 0 ⇒ cùng định nghĩa, cùng thang đo).")
    L.append(f"- Đẳng thức TB infid đường = infid(SPI) + phương sai giữa các đường: sai khác tối đa {fmt(im.p96_identity_gap_absmax.max())}.")
    L.append(f"- rank(P) của 96 ô: min {int(im.rank_P96.min())}, max {int(im.rank_P96.max())} (= 96 ⇒ sàn tự do mức pixel trên đúng 96 ô "
             "gần như chắc chắn bằng 0; sàn phải định nghĩa theo D quần thể).\n")
    t = pd.DataFrame([
        {"đại lượng": "infid trung bình một đường", "trung bình theo ảnh": cs("p96_mean_path_infid")},
        {"đại lượng": f"infid của SPI (K = {K}, trung bình seed)", "trung bình theo ảnh": cs("p96_spi_infid")},
        {"đại lượng": "σ̂² một đường (ddof 1)", "trung bình theo ảnh": cs("p96_sigma2")},
        {"đại lượng": f"phần phương sai trong SPI, σ̂²/{K}", "trung bình theo ảnh": cs("p96_sigma2", im.assign(p96_sigma2=im.p96_sigma2 / K))},
        {"đại lượng": f"tỷ trọng σ̂²/{K} trong infid(SPI)", "trung bình theo ảnh": cs("p96_var_share_at_P")},
        {"đại lượng": "sàn + lệch, L̂(μ)", "trung bình theo ảnh": cs("p96_L_mu")},
        {"đại lượng": "infid_β của SPI (co giãn tối ưu, chẩn đoán)", "trung bình theo ảnh": cs("p96_infid_beta_spi")},
    ])
    L.append(md(t))
    pool = ST.boot_ratio((im.p96_sigma2 / K * im.p96_den).values, (im.p96_spi_infid * im.p96_den).values, B, seed, al)
    L.append(f"Phân tích độ nhạy (tỷ số gộp Σ tử / Σ mẫu, trọng số ảnh theo ΣΔF²): tỷ trọng phương sai {pool[0]:.4f} "
             f"[{pool[1]:.4f}, {pool[2]:.4f}].\n")
    kc = _read(out, "e0_kcurve")
    if kc is not None:
        L.append("## 2. infid(Â_K) theo K và C2\n")
        g = kc.groupby("K")[["J_emp", "J_formula"]].mean().reset_index()
        L.append(md(g))
        L.append(f"Hồi quy theo 1/K (trung bình ảnh × seed): độ dốc {kc.groupby(['image', 'seed']).fit_slope.first().mean():.4g} so với "
                 f"σ̂² {kc.groupby(['image', 'seed']).sigma2.first().mean():.4g}; chặn {kc.groupby(['image', 'seed']).fit_intercept.first().mean():.4g} "
                 f"so với L̂(μ) {kc.groupby(['image', 'seed']).L_mu.first().mean():.4g}. Với tập con không hoàn lại từ cùng K đường, độ dốc ≈ σ̂² theo "
                 "cấu trúc; kiểm tra thật của luật 1/K là C2 dưới đây (nhóm rời nhau).\n")
        kv = [c for c in im.columns if c.startswith("p96_KV_")]
        if kv:
            L.append(md(pd.DataFrame([{"K": int(c.split("_")[-1]), "K·V(K) (nhóm rời, TB ảnh)": im[c].mean(),
                                       "σ̂² (TB ảnh)": im.p96_sigma2.mean()} for c in kv])))
    L.append("## 3. Tập ô mới (D quần thể): phương sai, lệch, sàn\n")
    name, spec = primary_spec(cfg)
    L.append(f"D chính `{name}` = {spec}; {ec['n_fit']} ô fit, {ec['n_eval']} ô đánh giá độc lập mỗi ảnh. Sàn mức patch: λ chọn bằng CV "
             f"{ec['folds']}-fold trên tập fit, đánh giá out-of-sample trên tập đánh giá (in-sample < sàn quần thể < out-of-sample).\n")
    Kg = int(ec["K_g0"])
    rows = [{"đại lượng": "infid(SPI) trên tập đánh giá", "giá trị": cs("ev_spi_infid_seed")},
            {"đại lượng": "σ̂²", "giá trị": cs("ev_sigma2")},
            {"đại lượng": "L̂(μ)", "giá trị": cs("ev_L_mu")},
            {"đại lượng": f"J_indep({Kg})", "giá trị": cs(f"ev_J_{Kg}") if f"ev_J_{Kg}" in im else "—"}]
    for G in ec["grids"]:
        for k in ("L_free_in", "L_free_oos", "L_comp_in", "L_comp_oos"):
            c = f"G{G}_{k}"
            if c in im:
                rows.append({"đại lượng": f"{k} (patch {G}×{G})", "giá trị": cs(c)})
    L.append(md(pd.DataFrame(rows)))
    if "ev_infid_beta_spi" in im:
        L.append(f"- infid_β (SPI, tập đánh giá): {cs('ev_infid_beta_spi')} — chênh lớn với infid thô ⇒ một phần lệch là thang đo "
                 "(completeness ép tổng).")
    L.append(f"- Khoảng hở completeness |ΣA − ΔF|/|ΔF|: từng đường {cs('compl_gap_path_absmean')}, SPI {cs('compl_gap_spi_abs')} "
             "(= 0 không chứng tỏ attribution đã hội tụ).\n")
    gr = _read(out, "e0_groups")
    if gr is not None:
        L.append("Tách theo cỡ ô (tập đánh giá; `share_den` = tỷ phần ΣΔF² của cỡ ô đó):\n")
        L.append(md(gr.groupby("group")[["n_sets", "share_den", "sigma2", "L_mu", "var_share_at_P"]].mean().reset_index()))
    smd = _read(out, "e0_smooth")
    if smd is not None:
        L.append("## 3b. Chẩn đoán làm mượt khối S_h (bảo toàn tổng; tập ô đánh giá độc lập)\n")
        L.append("S_h A = trung bình A trong khối h×h px — ánh xạ về bản đồ hằng theo khối, cùng không gian nghiệm với sàn patch "
                 "(h = 8 ↔ lưới 28, h = 16 ↔ lưới 14). Đây là CHẨN ĐOÁN (không giữ dummy), không phải phương pháp SPI mới.\n")
        base = smd[smd.h == 1].set_index("image")
        tab = []
        for h, g in smd.groupby("h"):
            g = g.set_index("image")
            r = {"h (px)": int(h), "L̂(S_h μ)": _fmt_ci(ST.boot_ci(g.L_mu, B, seed, al)),
                 "L(S_h SPI)": _fmt_ci(ST.boot_ci(g.L_spi, B, seed, al)), "σ̂²(S_h)": _fmt_ci(ST.boot_ci(g.sigma2, B, seed, al)),
                 "infid_β(S_h SPI)": float(g.infid_beta_spi.mean())}
            if int(h) > 1:
                j = base.L_mu.reindex(g.index)
                r["log(L̂(μ)/L̂(S_h μ)) theo ảnh"] = _fmt_ci(ST.boot_ci(np.log(j.clip(lower=1e-12) / g.L_mu.clip(lower=1e-12)), B, seed, al))
                if "floor_L_comp_oos" in g and g.floor_L_comp_oos.notna().any():
                    r["L̂(S_h μ) − L*_comp,patch"] = _fmt_ci(ST.boot_ci(g.L_mu - g.floor_L_comp_oos, B, seed, al))
                r["max |ΔΣA|"] = float(g.sum_err.max())
            tab.append(r)
        L.append(md(pd.DataFrame(tab)))
        L.append("Đọc: L̂(S_h μ) giảm mạnh so với h = 1 ⇒ phân bố không gian ở độ phân giải dưới h px đóng góp lớn vào sai số "
                 "(phù hợp giả thuyết dương/âm tần số cao). Nếu L̂(S_h μ) vẫn xa sàn patch cùng lưới ⇒ còn sai lệch ở chính tổng theo khối "
                 "(độ lớn / hướng ở tần số thấp), không chỉ nhiễu tần số cao. infid_β = 1 − cos²(PA, ΔF) cho phần hướng.\n")
    L.append("## 4. G0 — cổng phân bổ tài nguyên (một chiều)\n")
    d0 = pr.get("delta0")
    for G in ec["grids"]:
        c = f"G{G}_margin_J{Kg}"
        if c not in im:
            continue
        m, lo, hi = ci(c)
        L.append(f"- Patch {G}×{G}: dư địa J_indep({Kg}) − L*_comp,patch (out-of-sample) = {m:.4g} [{lo:.4g}, {hi:.4g}] → "
                 f"**{ST.decide_g0(lo, d0)}** (δ₀ = {d0}).")
    L.append("\nVì L*_comp,pixel ≤ L*_comp,patch, dư địa trên là CẬN DƯỚI của dư địa thật; cận dưới nhỏ không chứng minh dư địa nhỏ. "
             "E0 chỉ đo ở α, K của PATHSCORE nên không loại trừ lợi ích ở K nhỏ hoặc α khác. E1a pilot phụ thuộc P0 và C3, không phụ thuộc G0.\n")
    L.append("## 5. Cỡ mẫu (xấp xỉ chuẩn, một phía, α = 0,05, power 0,8)\n")
    tab = []
    for k in [int(x) for x in pr.get("Ks", [1, 2, 4, 8, 16])]:
        c = f"ev_var_{k}"
        if c not in im:
            continue
        mu, sd = float(im[c].mean()), float(im[c].std(ddof=1))
        for frac in (0.25, 0.5):
            tab.append({"K": k, "TB phần phương sai σ̂²/K": mu, "sd theo ảnh": sd, "δ (phần của TB)": frac,
                        "n ảnh cần": ST.n_required(sd, frac * mu)})
    L.append("Phần phương sai σ̂²/K là TRẦN TRÊN của mức giảm mà ghép nối có thể đạt ở K (ΔV_D(K) ≤ σ̂²/K); với cache `rand`, σ̂² gồm cả "
             "nhiễu quadrature τ²_D (xem INFID_QUAD_REPORT.md) — phần `rev` có thể loại bỏ chỉ nằm trong σ̂² − τ²_D. sd lấy theo ảnh. Cỡ mẫu chính "
             "cho H1 phải tính lại từ sd của ΔV_D trong E1a pilot.\n")
    L.append(md(pd.DataFrame(tab)))
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[e0] {rp}")
    return rp
