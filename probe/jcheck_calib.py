"""
Hiệu chuẩn CI cho phép kiểm "J thực nghiệm = công thức" (CPU, không GPU).

Đại số (không phụ thuộc ΔF): với ng nhóm rời mỗi nhóm h = K/2 cặp phủ hết m cặp,
    J_emp,rev − J_f,rev = (1/ng)Σ_g‖Ē_g − Ē‖²/‖d‖² − V̂_e (2/K − 1/m),
hiệu của hai ước lượng không chệch của cùng một phương sai ⇒ E = 0 trên MỖI ảnh; phân phối của hiệu theo ảnh chỉ phụ thuộc phân phối
của các vector E_j = (y1_j + y2_j)/2 (thang đo giữa ảnh, đuôi trong ảnh, số chiều hiệu dụng của nhiễu trên các ô), m, K.
(indep tương tự với các thành viên y1 / y2.)

Mô phỏng dùng đúng `budget_opt.empirical_J` và `infid_core.pair_core`. Kịch bản lưới:
  giữa ảnh   : lognormal(σ) hoặc lấy lại từ thang đo thực (√V theo ảnh từ pairs.csv, chuẩn hoá theo cấu hình);
  trong ảnh  : gauss | exp (thang đường √Exp(1)) | t3 / t5 (thang đường t đa biến);
  số chiều hiệu dụng của nhiễu trên ô: r (nhúng r chiều ẩn vào n ô bằng ma trận ngẫu nhiên) — các ô chồng lấn mạnh ⇒ r nhỏ ⇒ ‖·‖²
  lệch mạnh (gần χ²_r);
  ρ giữa hai thành viên cặp: từ dữ liệu ((V_e − V_o)/(V_e + V_o) trung vị) hoặc 0.
Phương pháp CI cho TB qua ảnh: percentile, BCa, t, bootstrap-t. Quy tắc chọn (đặt trước khi áp lên dữ liệu thật): phương pháp có độ phủ
NHỎ NHẤT qua mọi kịch bản × (coupling, K) lớn nhất; hoà ⇒ độ rộng TB nhỏ hơn.
"""

from __future__ import annotations

import functools
import math
from statistics import NormalDist

import numpy as np
import torch

from . import budget_opt as BO
from . import infid_core as IC
from .budget_audit import t_ppf, t_sf
from .infid_posthoc import boot_idx, boot_p, holm

F64 = torch.float64
METHODS = ("percentile", "BCa", "t", "bootstrap-t")
DEFAULT_SCEN = [
    {"name": "gauss·r64·lognorm0.7", "between": 0.7, "within": "gauss", "r": 64},
    {"name": "exp·r16·lognorm0.7", "between": 0.7, "within": "exp", "r": 16},
    {"name": "t5·r4·lognorm0.7", "between": 0.7, "within": "t5", "r": 4},
    {"name": "t3·r4·lognorm1.0", "between": 1.0, "within": "t3", "r": 4},
    {"name": "t3·r1·lognorm1.0", "between": 1.0, "within": "t3", "r": 1},
    {"name": "dữ liệu·exp·r4", "between": "data", "within": "exp", "r": 4},
    {"name": "dữ liệu·t3·r1", "between": "data", "within": "t3", "r": 1},
]


@functools.lru_cache(maxsize=64)
def _tq(q: float, df: int) -> float:
    return t_ppf(q, df)


def ci_methods(v: np.ndarray, n_boot: int, seed: int, al: float = 0.05) -> dict:
    """{phương pháp: (lo, hi, p hai phía cho TB = 0)} cho trung bình của v (theo ảnh)."""
    v = np.asarray(v, float)
    n = len(v)
    m = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(n))
    X = v[boot_idx(n, n_boot, seed)]
    mb = X.mean(1)
    sb = X.std(1, ddof=1) / math.sqrt(n)
    out = {"percentile": (float(np.quantile(mb, al / 2)), float(np.quantile(mb, 1 - al / 2)), boot_p(mb))}
    nd = NormalDist()
    z0 = nd.inv_cdf(float(np.clip((mb < m).mean(), 1e-6, 1 - 1e-6)))
    dv = v - m
    s2 = float((dv ** 2).sum())
    acc = float((dv ** 3).sum() / (6 * s2 ** 1.5)) if s2 > 0 else 0.0
    qs = []
    for q in (al / 2, 1 - al / 2):
        z = nd.inv_cdf(q)
        qs.append(float(np.quantile(mb, min(max(nd.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))), 0.0), 1.0))))
    out["BCa"] = (qs[0], qs[1], boot_p(mb))
    tq = _tq(1 - al / 2, n - 1)
    T = m / se if se > 0 else math.inf
    out["t"] = (m - tq * se, m + tq * se, float(min(1.0, 2 * t_sf(abs(T), n - 1))))
    ok = sb > 0
    ts = (mb[ok] - m) / sb[ok]
    pbt = float(min(1.0, 2 * min((1 + (ts >= T).sum()) / (len(ts) + 1), (1 + (ts <= T).sum()) / (len(ts) + 1)))) if se > 0 else math.nan
    out["bootstrap-t"] = (m - float(np.quantile(ts, 1 - al / 2)) * se, m - float(np.quantile(ts, al / 2)) * se, pbt)
    return out


def _noise(g: torch.Generator, m: int, n_cells: int, r: int, within: str, Pm: torch.Tensor) -> torch.Tensor:
    z = torch.randn(m, r, generator=g, dtype=F64)
    if within == "exp":
        z = z * torch.empty(m, 1, dtype=F64).exponential_(1.0, generator=g).sqrt()
    elif within in ("t3", "t5"):
        df = int(within[1:])
        chi = (torch.randn(m, df, generator=g, dtype=F64) ** 2).sum(1, keepdim=True)
        z = z * (df / chi).sqrt() * math.sqrt((df - 2) / df)                     # chuẩn hoá phương sai = 1
    return z @ Pm                                                               # (m, n_cells)


def simulate(scen: dict, n_img: int, m: int, Ks, reps: int, n_boot: int, seed: int, n_cells: int = 64, rho: float = 0.0,
             data_scales=None, al: float = 0.05) -> list:
    """Độ phủ thực của từng phương pháp CI cho TB(J_emp − J_f) qua n_img ảnh (giá trị đúng = 0)."""
    g = torch.Generator().manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    r = int(min(scen["r"], n_cells))
    acc = {}
    for rep in range(int(reps)):
        diffs = {}
        for _i in range(int(n_img)):
            if scen["between"] == "data":
                s = float(rng.choice(data_scales))
            else:
                s = float(np.exp(rng.normal(0.0, float(scen["between"]))))
            Pm = torch.randn(r, n_cells, generator=g, dtype=F64) / math.sqrt(r)
            mu = torch.randn(n_cells, generator=g, dtype=F64)
            d = mu + 0.5 * torch.randn(n_cells, generator=g, dtype=F64)
            a = _noise(g, m, n_cells, r, scen["within"], Pm)
            b = _noise(g, m, n_cells, r, scen["within"], Pm)
            y1 = mu + s * a
            y2 = mu + s * (rho * a + math.sqrt(max(1 - rho * rho, 0.0)) * b)
            st = IC.pair_core(y1, y2, d, list(Ks))
            em = BO.empirical_J(y1, y2, d, Ks)
            for K in Ks:
                for cp in BO.COUPLINGS:
                    k = f"Jemp_{cp}_{K}"
                    if k in em and BO.coupling_ok(cp, K):
                        diffs.setdefault((cp, int(K)), []).append(em[k] - BO.j_val(st["L_mu"], st["Ve"], st["Vo"], K, cp))
        for key, v in diffs.items():
            v = np.asarray(v)
            cis = ci_methods(v, n_boot, int(seed) + rep, al)
            for meth, (lo, hi, _p) in cis.items():
                o = acc.setdefault((key, meth), {"cover": 0, "below": 0, "above": 0, "width": [], "n": 0, "skew": []})
                o["n"] += 1
                o["cover"] += int(lo <= 0 <= hi)
                o["below"] += int(hi < 0)
                o["above"] += int(lo > 0)
                o["width"].append(hi - lo)
                if meth == "percentile":
                    sd = v.std()
                    o["skew"].append(float(((v - v.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0)
    rows = []
    for ((cp, K), meth), o in sorted(acc.items()):
        cov = o["cover"] / o["n"]
        rows.append({"kịch bản": scen["name"], "coupling": cp, "K": K, "phương pháp": meth, "lần lặp": o["n"], "độ phủ": cov,
                     "SE Monte Carlo": math.sqrt(cov * (1 - cov) / o["n"]), "CI dưới 0": o["below"] / o["n"], "CI trên 0": o["above"] / o["n"],
                     "độ rộng TB": float(np.mean(o["width"])), "skew TB (theo ảnh)": float(np.mean(o["skew"])) if o["skew"] else math.nan})
    return rows


def choose_method(rows: list) -> dict:
    """Quy tắc đặt trước: max over phương pháp của min độ phủ (qua kịch bản × coupling × K); hoà (≤ 1 SE MC) ⇒ độ rộng tương đối
    (chia độ rộng của t trong cùng ô) nhỏ hơn."""
    import pandas as pd
    t = pd.DataFrame(rows)
    ref = t[t["phương pháp"] == "t"].set_index(["kịch bản", "coupling", "K"])["độ rộng TB"]
    t["độ rộng / t"] = [r["độ rộng TB"] / ref.get((r["kịch bản"], r["coupling"], r["K"]), np.nan) for _, r in t.iterrows()]
    s = t.groupby("phương pháp").agg(min_cov=("độ phủ", "min"), mean_cov=("độ phủ", "mean"), rel_w=("độ rộng / t", "mean"),
                                    se=("SE Monte Carlo", "max")).reset_index()
    best = s.min_cov.max()
    tie = s[s.min_cov >= best - s.se.max()]
    ch = tie.sort_values("rel_w").iloc[0]["phương pháp"]
    return {"chosen": ch, "summary": s.sort_values("min_cov", ascending=False).to_dict("records")}


def data_scales(p, dset: str) -> np.ndarray:
    """√V theo ảnh (V = V_e + V_o), chuẩn hoá theo TB của từng cấu hình — phân tán giữa ảnh thực."""
    q = p[(p.set == "eval") & (p.group == "all") & (p.dset == dset)]
    out = []
    for _, g in q.groupby("config"):
        v = np.sqrt((g.Ve + g.Vo).values)
        if np.isfinite(v).all() and v.mean() > 0:
            out.append(v / v.mean())
    return np.concatenate(out) if out else np.ones(1)


def data_rho(p, dset: str) -> float:
    q = p[(p.set == "eval") & (p.group == "all") & (p.dset == dset)]
    r = ((q.Ve - q.Vo) / (q.Ve + q.Vo)).replace([np.inf, -np.inf], np.nan).dropna()
    return float(r.median()) if len(r) else 0.0


def recheck(cfg_view: dict, stages, n_boot: int, seed: int, al: float, chosen: str, miss_rate: dict) -> list:
    """Làm lại kiểm J thực nghiệm trên dữ liệu thật với cả 4 phương pháp; quyết định theo phương pháp đã chọn; Holm trong giai đoạn;
    so số dòng trượt với tỷ lệ trượt MÔ PHỎNG của chính phương pháp đó (không phải 5 % danh nghĩa)."""
    import pandas as pd
    from .report import md
    pr = BO.bpr(cfg_view)
    Dp = BO.primary_D(cfg_view)
    L, summ = [], []
    for stage in stages:
        em = BO._read(stage, "emp", cfg_view)
        if em is None:
            continue
        em = em[em.dset == Dp]
        rows = []
        for c, gq in em.groupby("config"):
            for K in pr["check_K"]:
                for cp in BO.COUPLINGS:
                    a, b = f"Jemp_{cp}_{K}", f"Jf_{cp}_{K}"
                    if a not in gq or b not in gq or not gq[a].notna().any():
                        continue
                    v = (gq[a] - gq[b]).dropna().values
                    cis = ci_methods(v, n_boot, seed, al)
                    r = {"config": c, "coupling": cp, "K": int(K), "n": len(v), "chênh TB": float(v.mean()),
                         "chênh / TB J": float(v.mean() / gq[b].mean())}
                    for meth in METHODS:
                        lo, hi, _p = cis[meth]
                        r[f"{meth} loại 0"] = "✗" if not (lo <= 0 <= hi) else "✓"
                    lo, hi, pc = cis[chosen]
                    r[f"CI {chosen}"] = f"[{lo:.4g}, {hi:.4g}]"
                    r[f"p {chosen}"] = pc
                    rows.append(r)
        if not rows:
            continue
        ph = holm([r[f"p {chosen}"] for r in rows])
        for r, h in zip(rows, ph):
            r["p Holm"] = h
        t = pd.DataFrame(rows)
        exp_miss = float(np.mean([miss_rate.get((r["coupling"], r["K"]), 0.05) for r in rows]))
        summ.append({"giai đoạn": stage, "số dòng": len(rows), **{f"loại 0 ({m_})": int((t[f"{m_} loại 0"] == "✗").sum()) for m_ in METHODS},
                     f"kỳ vọng loại 0 nếu công thức đúng ({chosen}, theo mô phỏng)": exp_miss * len(rows),
                     "Holm < 0,05": int((t["p Holm"] < al).sum())})
        L += [f"#### {stage}\n", md(t)]
    if summ:
        L = [md(pd.DataFrame(summ))] + L
    return L

