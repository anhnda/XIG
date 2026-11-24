"""
budget2-audit — kiểm toán BUDGET2 trạng thái dev-only (không GPU, không chọn lại) → infid/INFID_BUDGET2_AUDIT.md
budget2-freeze — đóng băng BUDGET2 dev-only (sha256 dev CSV / lựa chọn / báo cáo / config / mã) → infid/budget2/FROZEN.json;
                 sau khi đóng băng, budget2-lock / budget2-test bị chặn (tập test fresh 1000– được giữ cho ứng viên mạnh hơn).

(1) Cổng mỗi B: tách P(cải thiện > 0) khỏi P(cải thiện > δ); ΔJ, g với CI percentile / BCa / Fieller; p một phía bootstrap-t cho
    g > 0 và g > δ; bỏ-một-ảnh; cỡ mẫu cần cho từng giả thuyết; ước lượng TRUNG THỰC bằng cross-fit theo ảnh (chọn ứng viên +
    tham chiếu trên nửa ảnh A, bộ ô CHỌN; đánh giá trên nửa B, bộ ô ĐÁNH GIÁ) và bootstrap có chọn lại (độ lạc quan do chọn).
(2) Kiểm J thực nghiệm của dev BUDGET2 (như budget-audit mục 3).
(3) Hiệu chuẩn chính quy trình kiểm J thực nghiệm bằng mô phỏng CPU: cặp đối xứng có đuôi nặng, J đúng biết trước; đo độ phủ
    thực của CI bootstrap percentile 95 % cho TB(J_thực nghiệm − J_công thức) với đúng cách lập nhóm rời, m, K, số ảnh như dev.
"""

from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import math
import os

import numpy as np
import torch

from . import budget2 as B2
from . import budget_opt as BO
from . import config as C
from . import infid_core as IC
from . import infid_stats as ST
from .budget_audit import bca, boot_t_p, emp_section, fieller
from .infid_posthoc import boot_idx, ci_of, fmt_ci

F64 = torch.float64


def frozen_path(cfg: dict) -> str:
    return os.path.join(B2.root(cfg), "FROZEN.json")


# ─────────────────────────────────────────────────────────────────────────────
# hàm thuần (unit test)
# ─────────────────────────────────────────────────────────────────────────────

def gain_stats(jr: np.ndarray, jc: np.ndarray, delta: float, n_boot: int, seed: int, al: float) -> dict:
    n = len(jr)
    idx = boot_idx(n, n_boot, seed)
    g = float((jr.mean() - jc.mean()) / jr.mean())
    gb = (jr[idx].mean(1) - jc[idx].mean(1)) / jr[idx].mean(1)
    dJ = jr - jc
    out = {"n": n, "J_ref": float(jr.mean()), "J_ch": float(jc.mean()), "dJ": float(dJ.mean()), "dJ_ci": ci_of(dJ[idx].mean(1), al),
           "g": g, "g_ci_pct": ci_of(gb, al),
           "g_ci_bca": bca(lambda Z: float((Z[:, 0].mean() - Z[:, 1].mean()) / Z[:, 0].mean()), np.stack([jr, jc], 1), n_boot, seed, al),
           "g_ci_fieller": fieller(jc, jr, np.zeros_like(jr), 1 - al), "P_g_gt_0": float((gb > 0).mean()),
           "P_g_gt_delta": float((gb > delta).mean()), "wins": int((dJ > 0).sum())}
    for nm, dl in (("0", 0.0), ("delta", delta)):
        D = dJ - dl * jr
        out[f"p_bt_{nm}"] = boot_t_p(D, n_boot, seed)
        eff = (g - dl) * float(jr.mean())
        out[f"n_req_{nm}"] = ST.n_required(float(D.std(ddof=1)), eff, al, 0.8, one_sided=True) if eff > 0 else math.nan
    loo = np.array([(np.delete(jr, i).mean() - np.delete(jc, i).mean()) / np.delete(jr, i).mean() for i in range(n)])
    out["g_loo"] = (float(loo.min()), float(loo.max()))
    return out


def crossfit(Ws: np.ndarray, We: np.ndarray, ref_fixed, chal: list, refs: list, n_split: int, seed: int) -> dict:
    """Ws, We: (ảnh × ứng viên) J trên bộ ô chọn / đánh giá, cùng thứ tự cột. Mỗi lần chia: nửa A chọn ứng viên tốt nhất trong `chal`
    và tham chiếu (cố định nếu ref_fixed không None, ngược lại tốt nhất trong `refs`) trên Ws; đánh giá g trên nửa B của We
    (và ngược lại). Trả phân phối g trung thực."""
    rng = np.random.default_rng(seed)
    n = Ws.shape[0]
    gs = []
    for _ in range(int(n_split)):
        perm = rng.permutation(n)
        h = n // 2
        for A, Bh in ((perm[:h], perm[h:]), (perm[h:], perm[:h])):
            c = chal[int(np.argmin(Ws[A][:, chal].mean(0)))]
            r = ref_fixed if ref_fixed is not None else refs[int(np.argmin(Ws[A][:, refs].mean(0)))]
            gs.append((We[Bh, r].mean() - We[Bh, c].mean()) / We[Bh, r].mean())
    gs = np.asarray(gs)
    return {"g_mean": float(gs.mean()), "g_q": (float(np.quantile(gs, 0.1)), float(np.quantile(gs, 0.9))),
            "P_gt_0": float((gs > 0).mean()), "gs": gs}


def selection_boot(Ws: np.ndarray, ref_fixed, chal: list, refs: list, n_boot: int, seed: int) -> dict:
    """Bootstrap có chọn lại: trong mỗi mẫu lặp, chọn lại ứng viên + tham chiếu rồi tính g trên chính mẫu đó (độ lạc quan của chọn)
    so với g của ứng viên cố định đã chọn trên toàn bộ ảnh."""
    idx = boot_idx(Ws.shape[0], n_boot, seed)
    M = Ws[idx].mean(1)                                                         # (B, cand)
    c = np.array(chal)[M[:, chal].argmin(1)]
    r = np.full(len(M), ref_fixed) if ref_fixed is not None else np.array(refs)[M[:, refs].argmin(1)]
    g = (M[np.arange(len(M)), r] - M[np.arange(len(M)), c]) / M[np.arange(len(M)), r]
    return {"g_mean": float(g.mean()), "choices": c}


def sim_coverage(n_img: int, m: int, Ks, n_cells: int, reps: int, n_boot: int, seed: int, rho: float = -0.3) -> list:
    """Độ phủ thực của CI percentile 95 % cho TB(J_emp − J_f) qua ảnh, J_emp / J_f như mã thật (BO.empirical_J, IC.pair_core).
    Mỗi ảnh: thang đo lognormal (đuôi nặng giữa ảnh); nhiễu đường có thang ngẫu nhiên (đuôi nặng trong ảnh), cặp tương quan rho
    (phân phối biên của hai thành viên bằng nhau — đúng điều kiện của rev). Nhiễu: chuẩn × √(Exp(1)) theo từng đường."""
    g = torch.Generator().manual_seed(int(seed))
    out = {}
    for _ in range(int(reps)):
        diffs = {}
        for _i in range(int(n_img)):
            s = float(torch.empty(1).log_normal_(0.0, 0.7, generator=g))
            mu = torch.randn(n_cells, generator=g, dtype=F64)
            d = mu + 0.5 * torch.randn(n_cells, generator=g, dtype=F64)

            def tnoise(shape):
                z = torch.randn(shape, generator=g, dtype=F64)
                w = torch.empty(shape[0], 1, dtype=F64).exponential_(1.0, generator=g)    # thang ngẫu nhiên theo đường ⇒ đuôi nặng
                return z * w.sqrt() * 1.5
            a, b_ = tnoise((m, n_cells)), tnoise((m, n_cells))
            y1 = mu + s * a
            y2 = mu + s * (rho * a + math.sqrt(1 - rho * rho) * b_)
            st = IC.pair_core(y1, y2, d, list(Ks))
            em = BO.empirical_J(y1, y2, d, Ks)
            for K in Ks:
                for cp in BO.COUPLINGS:
                    k = f"Jemp_{cp}_{K}"
                    if k in em and BO.coupling_ok(cp, K):
                        diffs.setdefault((cp, K), []).append(em[k] - BO.j_val(st["L_mu"], st["Ve"], st["Vo"], K, cp))
        for key, v in diffs.items():
            v = np.asarray(v)
            lo, hi = ci_of(v[boot_idx(len(v), n_boot, int(seed))].mean(1), 0.05)
            o = out.setdefault(key, {"cover": 0, "below": 0, "above": 0, "n": 0, "skew": []})
            o["n"] += 1
            o["cover"] += int(lo <= 0 <= hi)
            o["below"] += int(hi < 0)
            o["above"] += int(lo > 0)
            o["skew"].append(float(((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-300)))
    return [{"coupling": cp, "K": K, "lần lặp": o["n"], "độ phủ thực": o["cover"] / o["n"], "CI nằm dưới 0": o["below"] / o["n"],
             "CI nằm trên 0": o["above"] / o["n"], "skew TB của chênh theo ảnh": float(np.mean(o["skew"]))}
            for (cp, K), o in sorted(out.items())]


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo
# ─────────────────────────────────────────────────────────────────────────────

def gate_section(cfg: dict) -> list:
    import pandas as pd
    from .report import md
    b, pr = B2.b2(cfg), B2.pr2(cfg)
    nb, seed, al, dl = int(pr["bootstrap"]), int(pr["seed"]), float(pr["alpha"]), float(pr["delta_rel"])
    cp = b["coupling"]
    sel = B2.load_sel(cfg)
    jl, _ = B2._wide(cfg, "dev", "sel")
    je, _ = B2._wide(cfg, "dev", "eval")
    L = ["## 1. Kiểm toán cổng dev\n",
         f"g = (J̄_ref − J̄_new)/J̄_ref. Cổng đã đăng ký dùng P_boot(g > 0) ≥ {pr['gate_min_prob']} và điểm ước lượng g ≥ {pr['gate_min_gain']:.0%} — "
         f"tức là **sàng lọc**, không phải phép thử g > δ = {dl:.0%}. Bảng tách hai câu hỏi. Ứng viên và tham chiếu (nhánh B) được CHỌN "
         "trên cùng bộ ô chọn ⇒ g trên bộ chọn lạc quan; bộ ô đánh giá dùng chung đường và ảnh nên chỉ giảm một phần lạc quan; "
         "cross-fit theo ảnh mới là ước lượng trung thực.\n"]
    rows, cf_rows = [], []
    for B, s in sel["budgets"].items():
        B = int(B)
        g_sel = math.nan
        for set_name, J in (("chọn", jl), ("đánh giá", je)):
            W = J[(J.B == B) & (J.coupling == cp)].pivot_table(index="image", columns="cand", values="J", aggfunc="first")
            j = W[[s["ref"], s["challenger"]]].dropna()
            st = gain_stats(j[s["ref"]].values, j[s["challenger"]].values, dl, nb, seed, al)
            if set_name == "chọn":
                g_sel = st["g"]
            rows.append({"B": B, "nhánh": s["branch"], "bộ ô": set_name, "ứng viên": s["challenger"], "tham chiếu": s["ref"],
                         "ΔJ [CI]": fmt_ci(st["dJ"], *st["dJ_ci"]), "g [CI percentile]": fmt_ci(st["g"], *st["g_ci_pct"], "{:.3f}"),
                         "g [CI BCa]": fmt_ci(st["g"], *st["g_ci_bca"], "{:.3f}"), "g [CI Fieller]": fmt_ci(st["g"], *st["g_ci_fieller"], "{:.3f}"),
                         "P_boot(g > 0)": st["P_g_gt_0"], f"P_boot(g > {dl:g})": st["P_g_gt_delta"], "p bootstrap-t g > 0": st["p_bt_0"],
                         f"p bootstrap-t g > {dl:g}": st["p_bt_delta"], "n cần (g > 0, lực 0,8)": st["n_req_0"],
                         f"n cần (g > {dl:g})": st["n_req_delta"], "g bỏ-một-ảnh min / max": f"{st['g_loo'][0]:.3f} / {st['g_loo'][1]:.3f}",
                         "ảnh ứng viên tốt hơn": f"{st['wins']}/{st['n']}"})
        # cross-fit + bootstrap có chọn lại
        Ws = jl[(jl.B == B) & (jl.coupling == cp)].pivot_table(index="image", columns="cand", values="J", aggfunc="first")
        We = je[(je.B == B) & (je.coupling == cp)].pivot_table(index="image", columns="cand", values="J", aggfunc="first")
        cols = [c for c in Ws.columns if c in We.columns]
        Ws, We = Ws[cols].dropna(), We[cols].reindex(Ws[cols].dropna().index)
        ok = We.notna().all(1)
        Ws, We = Ws[ok], We[ok]
        names = list(Ws.columns)
        spi = [i for i, c in enumerate(names) if c.startswith("spi_")]
        if s["branch"] == "A":
            ref_fixed, refs = names.index(sel["incumbent"]), None
            chal = [i for i in spi if i != ref_fixed]
        else:
            ref_fixed, refs = None, spi
            chal = [i for i, c in enumerate(names) if c.startswith("hier_")]
        if not chal:
            continue
        cfit = crossfit(Ws.values, We.values, ref_fixed, chal, refs, 500, seed)
        sb = selection_boot(Ws.values, ref_fixed, chal, refs, nb, seed)
        ch_counts = {names[k]: float(v) / len(sb["choices"]) for k, v in zip(*np.unique(sb["choices"], return_counts=True))}
        top = sorted(ch_counts.items(), key=lambda x: -x[1])[:3]
        cf_rows.append({"B": B, "nhánh": s["branch"], "g trên bộ chọn (ứng viên cố định)": g_sel,
                        "g bootstrap có chọn lại (TB)": sb["g_mean"], "g cross-fit (TB)": cfit["g_mean"],
                        "g cross-fit (phân vị 10 % / 90 %)": f"{cfit['g_q'][0]:.3f} / {cfit['g_q'][1]:.3f}",
                        "P(g cross-fit > 0)": cfit["P_gt_0"],
                        "ứng viên được chọn lại (tỷ lệ)": "; ".join(f"{k} {v:.2f}" for k, v in top)})
    L.append(md(pd.DataFrame(rows)))
    if cf_rows:
        L.append("**Ước lượng sau khi trừ hiệu ứng chọn** (cross-fit: 500 lần chia đôi ảnh, chọn trên nửa này / bộ ô chọn, đánh giá trên nửa kia / "
                 "bộ ô đánh giá; bootstrap có chọn lại: chọn lại ứng viên + tham chiếu trong mỗi mẫu lặp):\n")
        L.append(md(pd.DataFrame(cf_rows)))
    L.append("Đọc: \"qua cổng\" chỉ nói ứng viên đáng xem tiếp. Muốn tuyên bố g > δ cần p bootstrap-t (g > δ) nhỏ trên ảnh MỚI với cỡ mẫu "
             "ở cột n cần; nếu g cross-fit ≈ δ hoặc nhỏ hơn thì không nên mở test xác nhận g > δ.\n")
    return L


def sim_section(cfg: dict) -> list:
    import pandas as pd
    from .report import md
    b, pr = B2.b2(cfg), B2.pr2(cfg)
    dbg = bool(cfg.get("_debug"))
    rows = sim_coverage(n_img=int(b["dev"]["n_images"]), m=int(b["dev"]["m"]), Ks=[int(k) for k in pr["check_K"]],
                        n_cells=64 if dbg else 256, reps=10 if dbg else 200, n_boot=200 if dbg else 2000, seed=int(pr["seed"]))
    return ["## 3. Hiệu chuẩn quy trình kiểm J thực nghiệm (mô phỏng CPU, J đúng = công thức kỳ vọng)\n",
            f"Cùng số ảnh ({b['dev']['n_images']}), số cặp m = {b['dev']['m']}, K {pr['check_K']}, cùng hàm `empirical_J` và `pair_core` như mã thật; "
            "đuôi nặng giữa ảnh (thang lognormal) và trong ảnh (phương sai đường ngẫu nhiên). Hai ước lượng cùng không chệch theo cấu "
            "trúc, nên độ phủ thực < 95 % hoặc lệch một phía (CI dưới / trên 0) đo trực tiếp sai lệch của CI percentile — không phải của "
            "công thức. So số CI trượt trong dữ liệu thật với tỷ lệ \"không phủ\" ở đây.\n", md(pd.DataFrame(rows))]


def run(cfg: dict) -> str:
    rp = os.path.join(C.out_dir(cfg, "pilot"), "infid", "INFID_BUDGET2_AUDIT.md")
    fz = frozen_path(cfg)
    L = ["# Kiểm toán BUDGET2 (dev-only; không GPU, không chọn lại)\n",
         f"Trạng thái: {'**đã đóng băng** (' + json.load(open(fz))['time'] + ')' if os.path.exists(fz) else 'chưa đóng băng'}; "
         f"khoá test: {'có' if os.path.exists(B2.lock_path(cfg)) else 'không'}.\n"]
    L += gate_section(cfg)
    v = B2.view(cfg)
    L += [x.replace("## 3. Kiểm J thực nghiệm", "## 2. Kiểm J thực nghiệm (dev BUDGET2)") for x in emp_section(v, stages=("dev",))]
    L += sim_section(cfg)
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[budget2-audit] {rp}")
    return rp


def freeze(cfg: dict) -> str:
    """Đóng băng BUDGET2 ở trạng thái dev-only. Chặn budget2-lock / budget2-test cho tới khi xoá FROZEN.json có chủ đích."""
    if os.path.exists(B2.lock_path(cfg)):
        raise SystemExit("BUDGET2 đã khoá test — không đóng băng dev-only được")
    root = B2.root(cfg)
    inf = os.path.join(C.out_dir(cfg, "pilot"), "infid")

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    files = sorted(glob.glob(os.path.join(root, "dev", "*.csv")) + glob.glob(os.path.join(root, "dev", "meta.json")) +
                   glob.glob(os.path.join(root, "selection_dev.json")) +
                   [p for p in (os.path.join(inf, "INFID_BUDGET2_REPORT.md"), os.path.join(inf, "INFID_BUDGET2_AUDIT.md")) if os.path.exists(p)])
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hs = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(src, "probe", "*.py"))):
        hs.update(open(f, "rb").read())
    b = B2.b2(cfg)
    sel = B2.load_sel(cfg)
    rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), "status": "dev-only; không test",
           "decision": {B: {"challenger": s["challenger"], "ref": s["ref"], "gain_rel": s["gain_rel"], "p_better": s["p_better"],
                            "pass": s["pass"]} for B, s in sel["budgets"].items()},
           "incumbent_kept": B2.cand(b["incumbent"], b["coupling"]),
           "test_images_reserved": {"split": b["test"]["split"], "offset": b["test"]["offset"], "max_n": B2.pr2(cfg)["n_max"]},
           "budget2_config_sha256": IC._h(b), "config": cfg["_path"], "source_sha256": hs.hexdigest(),
           "files": {os.path.relpath(f, inf): sha(f) for f in files}}
    p = frozen_path(cfg)
    if os.path.exists(p):
        rec["previous"] = json.load(open(p))
    with open(p, "w") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print(f"[budget2-freeze] {p}: {len(files)} file; budget2-lock / budget2-test bị chặn")
    return p
