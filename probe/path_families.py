"""
Họ đường ngẫu nhiên theo feature (PATH_FAMILIES, Phase 0 — chỉ CPU, không mô hình, không ảnh).

Mọi họ: γ_i(t) = x′_i + Z_i(t)(x_i − x′_i), lịch trình Z_i đơn điệu trên lưới N nút, Z(0) = 0, Z(1) = 1, độc lập giữa phần tử.
Đường thực là nội suy tuyến tính từng đoạn giữa các nút (đúng như pipeline INFID / BUDGET), nên quy tắc `rand` M = 1 vẫn không chệch
cho tích phân trên ĐƯỜNG ĐÃ RỜI RẠC HOÁ của mọi họ, kể cả họ có bước nhảy (bước nhảy nằm trọn trong một đoạn lưới).

  ig      Z_i(t_k) = t_k
  dp      số gia (Δ_1..Δ_N) ~ Dirichlet(α/N, …, α/N)                         [SPI hiện tại]
  ln      số gia = softmax_k(σ ε_k), ε ~ N(0, 1)  (logistic-normal)            [E10 "lognormal"]
  cmix    với xác suất λ: một bước nhảy trọn 0 → 1 tại đoạn k ~ U{1..N}; ngược lại số gia đều 1/N (= IG)   [COORD_MIX]
  smooth  Z_i(t) = t + δ s_i t(1 − t), s_i = ±1 đều, δ ≤ 1 (đơn điệu)          [E10 "smooth"]

Khớp mức exploration theo h = 1/(α + 1) của DP:
  cmix: λ = h khớp CHÍNH XÁC Var Z(t_k) = h t_k(1 − t_k) tại mọi nút (và E Z = t).
  ln:   σ hiệu chuẩn Monte Carlo để D̂ (hình thang trên nút) = D̂ của DP lưới (như E10).
  smooth: δ = √(5h) khớp D liên tục; chỉ khả thi khi δ ≤ 1 ⇔ h ≤ 1/5 ⇔ α ≥ 4 (KHÔNG đạt được vùng α = 1 của incumbent).

Lệnh: bash scripts/run.sh pathfam-audit  → runs_probe/pilot/infid/pathfam/PATH_FAMILIES_CHECKS.md
  (1) kiểm tính chất từng họ bằng Monte Carlo (đầu mút, đơn điệu, E Z = t, Var Z(t), D̂, bất biến khi đảo, phân phối thời điểm
      kích hoạt, độ nhọn, bước nhảy lớn nhất) trên lưới tham số đề xuất cho Phase 1;
  (2) phân tích lại E10 từ CSV nếu có (`path_families.e10_csv`): TOST thật sự (t và bootstrap) so với "chỉ gần nhau".
"""

from __future__ import annotations

import functools
import math
import os

import numpy as np
import torch

from . import config as C
from .paths import cum_nodes, spi_increments

F64 = torch.float64
FAMILIES = ("ig", "dp", "ln", "cmix", "smooth")

DEFAULT = {
    "alphas": [0.1, 0.3, 1, 3, 10],
    "N": [32, 64],
    "smooth_delta": [0.5, 0.674, 1.0],
    "mc_d": 20000,
    "calib_seed": 10,
    "e10_csv": None,
    "e10_margin": 0.02,
    "bootstrap": 10000,
    "seed": 53,
}


def pcfg(cfg: dict) -> dict:
    return C._merge(DEFAULT, cfg.get("path_families", {}) or {})


# ─────────────────────────────────────────────────────────────────────────────
# lịch trình
# ─────────────────────────────────────────────────────────────────────────────

def increments(fam: str, N: int, shape, gen: torch.Generator, p: dict) -> torch.Tensor:
    """Số gia (N, *shape) float64 trên gen.device; mỗi cột trên đơn hình."""
    dev = gen.device
    shape = tuple(shape)
    if fam == "ig":
        return torch.full((N,) + shape, 1.0 / N, dtype=F64, device=dev)
    if fam == "dp":
        return spi_increments(float(p["alpha"]), int(N), shape, gen)
    if fam == "ln":
        eps = torch.randn((N,) + shape, generator=gen, device=dev, dtype=F64)
        return torch.softmax(float(p["sigma"]) * eps, dim=0)
    if fam == "cmix":
        lam = float(p["lam"])
        jump = torch.rand(shape, generator=gen, device=dev, dtype=F64) < lam
        k = torch.randint(0, N, shape, generator=gen, device=dev)
        w = torch.full((N,) + shape, 1.0 / N, dtype=F64, device=dev)
        oh = torch.zeros((N,) + shape, dtype=F64, device=dev).scatter_(0, k.unsqueeze(0), 1.0)
        return torch.where(jump.unsqueeze(0), oh, w)
    if fam == "smooth":
        Z = schedule("smooth", N, shape, gen, p)
        return Z[1:] - Z[:-1]
    raise ValueError(fam)


def schedule(fam: str, N: int, shape, gen: torch.Generator, p: dict) -> torch.Tensor:
    """Z (N+1, *shape) float64."""
    if fam == "smooth":
        d = float(p["delta"])
        if not 0 <= d <= 1:
            raise ValueError(f"smooth: δ = {d} > 1 ⇒ không đơn điệu (Z' = 1 − δ < 0 tại đầu mút)")
        t = torch.linspace(0, 1, N + 1, device=gen.device, dtype=F64).view((-1,) + (1,) * len(tuple(shape)))
        s = 1.0 - 2.0 * (torch.rand(tuple(shape), generator=gen, device=gen.device, dtype=F64) < 0.5).double()
        return t + d * s * t * (1 - t)
    return cum_nodes(increments(fam, N, shape, gen, p))


def d_hat(Z: torch.Tensor) -> float:
    N = Z.shape[0] - 1
    t = torch.linspace(0, 1, N + 1, device=Z.device, dtype=F64).view((-1,) + (1,) * (Z.dim() - 1))
    e = ((Z.double() - t) ** 2).reshape(N + 1, -1).mean(1)
    return float(torch.trapezoid(e, dx=1.0 / N))


def d_dp_grid(alpha: float, N: int) -> float:
    t = torch.linspace(0, 1, N + 1, dtype=F64)
    return float(torch.trapezoid(t * (1 - t) / (alpha + 1), dx=1.0 / N))


def calibrate_ln(target: float, N: int, d: int, seed: int) -> float:
    """σ sao cho D̂ của ln trên lưới N = target (chia đôi trên log σ, nhiễu cố định ⇒ đơn điệu, tất định). Như E10.calibrate."""
    g = torch.Generator().manual_seed(int(seed))
    eps = torch.randn((N, d), generator=g, dtype=F64)

    def D(sig):
        return d_hat(cum_nodes(torch.softmax(sig * eps, dim=0)))
    lo, hi = math.log(1e-3), math.log(200.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        if D(math.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
    return math.exp((lo + hi) / 2)


def matched_grid(cfg: dict) -> list:
    """Lưới tham số khớp exploration theo α của DP (Phase 1). smooth chỉ ở các δ ≤ 1."""
    c = pcfg(cfg)
    out = []
    for N in c["N"]:
        for a in c["alphas"]:
            h = 1.0 / (float(a) + 1.0)
            out.append({"family": "dp", "N": int(N), "alpha": float(a), "h": h})
            out.append({"family": "cmix", "N": int(N), "lam": h, "h": h, "alpha_match": float(a)})
            sig = calibrate_ln(d_dp_grid(float(a), int(N)), int(N), int(c["mc_d"]), int(c["calib_seed"]))
            out.append({"family": "ln", "N": int(N), "sigma": sig, "h": h, "alpha_match": float(a)})
        for dl in c["smooth_delta"]:
            out.append({"family": "smooth", "N": int(N), "delta": float(dl), "h": float(dl) ** 2 / 5, "alpha_match": 5 / float(dl) ** 2 - 1})
    return out


@functools.lru_cache(maxsize=None)
def sigma_for(alpha: float, N: int, d: int = 20000, seed: int = 10) -> float:
    """σ của ln khớp D̂ lưới của DP(α, N) — tất định (seed, d cố định như E10), lưu đệm."""
    return calibrate_ln(d_dp_grid(float(alpha), int(N)), int(N), int(d), int(seed))


def family_params(c: dict) -> dict:
    """Tham số lịch trình từ cấu hình đã chuẩn hoá (sampling_infid.norm_conf)."""
    f = c["family"]
    if f == "smooth":
        return {"delta": float(c["delta"])}
    a = float(c["alpha"])
    if f in ("spi", "dp"):
        return {"alpha": a}
    if f == "ln":
        return {"sigma": sigma_for(a, int(c["N"]))}
    if f == "cmix":
        return {"lam": 1.0 / (a + 1.0)}
    raise ValueError(f)


# ─────────────────────────────────────────────────────────────────────────────
# kiểm tính chất (Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────────

def ks2(a: np.ndarray, b: np.ndarray) -> float:
    """KS hai mẫu; làm tròn 12 chữ số để các nguyên tử (smooth, cmix) không bị tách bởi sai số dấu phẩy động 1 ulp."""
    a, b = np.sort(np.round(a, 12)), np.sort(np.round(b, 12))
    v = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, v, side="right") / len(a) - np.searchsorted(b, v, side="right") / len(b)).max())


def properties(fam: str, N: int, p: dict, d: int, seed: int) -> dict:
    """Thống kê của d lịch trình độc lập (một phần tử mỗi cột)."""
    g = torch.Generator().manual_seed(int(seed))
    Z = schedule(fam, N, (d,), g, p)                                          # (N+1, d)
    R = 1.0 - Z.flip(0)                                                      # đảo
    t = torch.linspace(0, 1, N + 1, dtype=F64)
    dZ = Z[1:] - Z[:-1]
    h = p.get("h")
    v_emp = Z.var(1)
    out = {"endpoints_err": float(max(Z[0].abs().max(), (Z[-1] - 1).abs().max())), "min_increment": float(dZ.min()),
           "mean_err_max": float((Z.mean(1) - t).abs().max()), "D_hat": d_hat(Z)}
    if h is not None:
        tgt = h * t * (1 - t)
        out["var_rel_err_max"] = float(((v_emp - tgt).abs()[1:-1] / tgt[1:-1]).max())
    k = [N // 4, N // 2, (3 * N) // 4]
    out["ks_rev_max"] = max(ks2(Z[j].numpy(), R[j].numpy()) for j in k)
    out["ks_rev_ref_95"] = 1.36 * math.sqrt(2.0 / d)                         # ngưỡng KS 5 % (xấp xỉ) cho hai mẫu cỡ d
    # thời điểm kích hoạt: t* đầu tiên mà Z ≥ 1/2 (nội suy tuyến tính trong đoạn)
    above = Z >= 0.5
    kk = above.double().argmax(0).clamp_min(1)
    z0, z1 = Z[kk - 1, torch.arange(d)], Z[kk, torch.arange(d)]
    ts = ((kk - 1).double() + (0.5 - z0) / (z1 - z0).clamp_min(1e-300)) / N
    out["t_half_mean"] = float(ts.mean())
    out["t_half_sd"] = float(ts.std())
    out["t_half_q10_q90"] = f"{float(ts.quantile(0.1)):.3f} / {float(ts.quantile(0.9)):.3f}"
    mid = Z[N // 2] - 0.5
    out["kurt_Z_mid"] = float(((mid - mid.mean()) ** 4).mean() / (mid.var() ** 2 + 1e-300))
    out["max_jump_mean"] = float(dZ.max(0).values.mean())
    out["eff_steps"] = float((1.0 / (dZ ** 2).sum(0).clamp_min(1e-300)).mean())
    # tương quan giữa hai phần tử (phải ≈ 0: độc lập theo cấu trúc)
    Z2 = schedule(fam, N, (d,), torch.Generator().manual_seed(int(seed) + 1), p)
    out["corr_between_elems_mid"] = float(np.corrcoef(Z[N // 2].numpy(), Z2[N // 2].numpy())[0, 1])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# toy kiểm Mệnh đề (đa thức bậc ≤ 3: E[A] chỉ phụ thuộc họ qua D)
# ─────────────────────────────────────────────────────────────────────────────

def toy_poly(d: int, seed: int, quartic: float = 0.0, quintic: float = 0.0) -> dict:
    """F(x) = bᵀx + ½xᵀQx + (1/6)Σ T x³ (+ quartic·(1/24)Σ U x⁴ + quintic·(1/120)Σ W x⁵), các tensor đối xứng."""
    g = torch.Generator().manual_seed(int(seed))

    def sym(X):
        import itertools
        perms = list(itertools.permutations(range(X.dim())))
        return sum(X.permute(*p_) for p_ in perms) / len(perms)
    return {"b": torch.randn(d, generator=g, dtype=F64), "Q": sym(torch.randn(d, d, generator=g, dtype=F64)),
            "T": sym(torch.randn(d, d, d, generator=g, dtype=F64)) * 3.0,
            "U": sym(torch.randn(d, d, d, d, generator=g, dtype=F64)) * 3.0 * float(quartic),
            "W": sym(torch.randn(d, d, d, d, d, generator=g, dtype=F64)) * 3.0 * float(quintic)}


def poly_grad(P: dict, X: torch.Tensor) -> torch.Tensor:
    """∇F tại X (..., d)."""
    g = P["b"] + X @ P["Q"] + 0.5 * torch.einsum("ijk,...j,...k->...i", P["T"], X, X)
    if float(P["U"].abs().max()) > 0:
        g = g + torch.einsum("ijkl,...j,...k,...l->...i", P["U"], X, X, X) / 6.0
    if float(P["W"].abs().max()) > 0:
        g = g + torch.einsum("ijklm,...j,...k,...l,...m->...i", P["W"], X, X, X, X) / 24.0
    return g


def exact_attr(P: dict, Z: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """Attribution CHÍNH XÁC trên đường nội suy tuyến tính từng đoạn (Gauss–Legendre 3 điểm/đoạn, đúng khi ∇F bậc ≤ 5).
    Z (N+1, P, d), x′ = 0. -> (P, d)."""
    s = torch.tensor([0.5 - math.sqrt(0.15), 0.5, 0.5 + math.sqrt(0.15)], dtype=F64)
    w = torch.tensor([5 / 18, 8 / 18, 5 / 18], dtype=F64)
    X = Z * dx
    dX = X[1:] - X[:-1]
    A = torch.zeros(Z.shape[1:], dtype=F64)
    for q in range(3):
        A += w[q] * (poly_grad(P, X[:-1] + s[q] * dX) * dX).sum(0)
    return A


def d_interp(Z: torch.Tensor) -> float:
    """D = E ∫(Z − t)² dt CHÍNH XÁC trên đường nội suy tuyến tính (mỗi đoạn: (a² + ab + b²)/3 · 1/N)."""
    N = Z.shape[0] - 1
    t = torch.linspace(0, 1, N + 1, dtype=F64).view((-1,) + (1,) * (Z.dim() - 1))
    e = Z - t
    a, b = e[:-1], e[1:]
    return float(((a * a + a * b + b * b) / 3).sum(0).mean() / N)


TOY_FAMS = [("dp", {"alpha": 1.0}), ("cmix", {"lam": 0.5}), ("ln", {"alpha_match": 1.0}), ("dp", {"alpha": 3.0}),
            ("cmix", {"lam": 0.25}), ("smooth", {"delta": 1.0})]


def toy_check(d: int = 3, N: int = 8, n_paths: int = 200000, chunk: int = 50000, seed: int = 0, quartic: float = 0.0,
              quintic: float = 0.0) -> list:
    """E[A_f] − A_IG = D_f · a₁ với cùng a₁ cho mọi họ (bậc ≤ 3). a₁ ước lượng từ dp α = 1. Trả bảng z-score theo tọa độ."""
    P = toy_poly(d, 100 + seed, quartic, quintic)
    dx = torch.ones(d, dtype=F64)
    Zig = torch.linspace(0, 1, N + 1, dtype=F64).view(-1, 1, 1).expand(N + 1, 1, d)
    A_ig = exact_attr(P, Zig, dx)[0]
    res = []
    for k, (fam, p) in enumerate(TOY_FAMS):
        p = dict(p)
        if fam == "ln":
            p = {"sigma": calibrate_ln(d_dp_grid(p["alpha_match"], N), N, 20000, 10)}
        g = torch.Generator().manual_seed(1000 * (seed + 1) + k)
        s1 = torch.zeros(d, dtype=F64)
        s2 = torch.zeros(d, dtype=F64)
        Dsum, n = 0.0, 0
        while n < n_paths:
            m = min(chunk, n_paths - n)
            Z = schedule(fam, N, (m, d), g, p)
            A = exact_attr(P, Z, dx)
            s1 += A.sum(0)
            s2 += (A * A).sum(0)
            Dsum += d_interp(Z) * m
            n += m
        mean = s1 / n
        se = ((s2 / n - mean ** 2).clamp_min(0) / n).sqrt()
        res.append({"family": fam, "params": p, "D": Dsum / n, "delta": mean - A_ig, "se": se})
    ref = res[0]
    a1 = ref["delta"] / ref["D"]
    rows = []
    for r in res:
        pred = r["D"] * a1
        se = (r["se"] ** 2 + (r["D"] / ref["D"]) ** 2 * ref["se"] ** 2).sqrt()
        z = ((r["delta"] - pred) / se.clamp_min(1e-300))
        rows.append({"họ": r["family"], "tham số": ", ".join(f"{k_}={v:.4g}" for k_, v in r["params"].items()), "D (nội suy, chính xác)": r["D"],
                     "‖E A − A_IG‖": float(r["delta"].norm()), "‖E A − A_IG − D·a₁‖": float((r["delta"] - pred).norm()),
                     "max |z| theo tọa độ": float(z.abs().max()) if r is not ref else 0.0})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# phân tích lại E10 (nếu có CSV)
# ─────────────────────────────────────────────────────────────────────────────

def tost(v: np.ndarray, margin: float, n_boot: int, seed: int) -> dict:
    """Equivalence ±margin cho TB của hiệu theo ảnh: TOST t (df n − 1) và TOST bootstrap percentile (= CI 90 % trong ±margin)."""
    from .budget_audit import t_ppf, t_sf
    from .infid_posthoc import boot_idx
    n = len(v)
    m, se = float(v.mean()), float(v.std(ddof=1) / math.sqrt(n))
    p_lo = t_sf((m + margin) / se, n - 1) if se > 0 else 0.0
    p_hi = t_sf((margin - m) / se, n - 1) if se > 0 else 0.0
    bs = v[boot_idx(n, n_boot, seed)].mean(1)
    q = t_ppf(0.95, n - 1)
    return {"n": n, "mean": m, "median": float(np.median(v)), "ci90_t": (m - q * se, m + q * se),
            "ci90_boot": (float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))),
            "ci95_boot": (float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))),
            "p_tost_t": float(max(p_lo, p_hi)), "p_tost_boot": float(max((bs <= -margin).mean(), (bs >= margin).mean()))}


def kurtosis_check(N: int = 8, n_paths: int = 400000, chunk: int = 100000, seed: int = 0) -> list:
    """Phản ví dụ bậc 5: F(x) = x₀ (x₁ − ½)⁴ ⇒ E A₀ = E ∫ (Z₁ − ½)⁴ dZ₀ = ∫ E(Z₁(t) − ½)⁴ dt — phụ thuộc moment bậc 4 của Z(t),
    không chỉ D. dp(α) và cmix(λ = 1/(α + 1)) có CÙNG toàn bộ hàm hiệp phương sai (cùng D, cùng v(t)) nhưng khác độ nhọn ⇒ E A₀ khác."""
    s_ = torch.tensor([0.5 - math.sqrt(0.15), 0.5, 0.5 + math.sqrt(0.15)], dtype=F64)
    w = torch.tensor([5 / 18, 8 / 18, 5 / 18], dtype=F64)
    rows = []
    for k, (fam, p) in enumerate([("dp", {"alpha": 1.0}), ("cmix", {"lam": 0.5}),
                                  ("ln", {"sigma": calibrate_ln(d_dp_grid(1.0, N), N, 20000, 10)}), ("ig", {})]):
        g = torch.Generator().manual_seed(7000 + k)
        s1 = s2 = 0.0
        Dsum, n = 0.0, 0
        while n < n_paths:
            m = min(chunk, n_paths - n)
            Z = schedule(fam, N, (m, 2), g, p)
            dZ0 = Z[1:, :, 0] - Z[:-1, :, 0]
            A0 = torch.zeros(m, dtype=F64)
            for q in range(3):
                z1 = Z[:-1, :, 1] + s_[q] * (Z[1:, :, 1] - Z[:-1, :, 1])
                A0 += w[q] * ((z1 - 0.5) ** 4 * dZ0).sum(0)
            s1 += float(A0.sum())
            s2 += float((A0 * A0).sum())
            Dsum += d_interp(Z) * m
            n += m
        mean = s1 / n
        rows.append({"họ": fam, "tham số": ", ".join(f"{k_}={v:.4g}" for k_, v in p.items()), "D": Dsum / n, "E A₀": mean,
                     "SE": math.sqrt(max(s2 / n - mean * mean, 0.0) / n)})
    ref = rows[0]
    for r in rows:
        r["E A₀ − dp"] = r["E A₀"] - ref["E A₀"]
        r["z so với dp"] = 0.0 if r is ref else r["E A₀ − dp"] / math.sqrt(r["SE"] ** 2 + ref["SE"] ** 2)
    return rows


def toy_section(cfg: dict) -> list:
    import pandas as pd
    from .report import md
    n = 200000 if not cfg.get("_debug") else 20000
    L = ["## 1b. Toy kiểm Mệnh đề Phase 0 §5 (đa thức, tích phân đường chính xác, không mô hình)\n",
         f"d = 3 phần tử, N = 8, {n} đường mỗi họ. Attribution tính CHÍNH XÁC trên đường nội suy (Gauss–Legendre 3 điểm/đoạn); D tính chính "
         "xác trên đường nội suy. a₁ ước lượng từ dp α = 1; dự đoán E A_f − A_IG = D_f·a₁ cho mọi họ (|z| ≲ 4 nếu đúng). Lý thuyết: đúng "
         "với bậc ≤ 3 cho mọi họ có E Z = t, độc lập; đúng với bậc 4 khi thêm bất biến khi đảo (cả 4 họ đều có); bậc 5 KHÔNG đúng nói "
         "chung (phụ thuộc ∫v², ∫t²v, ∫tκ₃, ∫κ₄). Hệ số ngẫu nhiên bậc 5 cho đóng góp cumulant nhỏ hơn nhiễu MC ở cỡ mẫu này ⇒ xem phản "
         "ví dụ có chủ đích bên dưới.\n"]
    for lab, kw in (("bậc 3 (Mệnh đề)", {}), ("bậc 4 ngẫu nhiên", {"quartic": 1.0}), ("bậc 5 ngẫu nhiên", {"quintic": 1.0})):
        L.append(f"**{lab}**\n")
        L.append(md(pd.DataFrame(toy_check(n_paths=n, **kw))))
    L.append("**Phản ví dụ bậc 5 có chủ đích** F(x) = x₀(x₁ − ½)⁴: dp α = 1 và cmix λ = ½ có cùng hàm hiệp phương sai (cùng D) nhưng khác moment "
             "bậc 4 ⇒ E A₀ phải khác (|z| lớn); ln khớp D nhưng không khớp v(t) từng điểm.\n")
    L.append(md(pd.DataFrame(kurtosis_check(n_paths=2 * n))))
    return L


def e10_section(cfg: dict) -> list:
    import pandas as pd
    from .report import md
    c = pcfg(cfg)
    path = c.get("e10_csv")
    L = ["## 2. Phân tích lại E10 từ CSV\n"]
    if not path or not os.path.exists(path):
        L.append(f"Không thấy `e10_fam.csv` (`path_families.e10_csv` = `{path}`). Đặt đường dẫn tới `runs_probe/e10_fresh/e10_fam.csv` của "
                 "lần chạy E10 (máy cũ) rồi chạy lại lệnh — không cần GPU. Thiếu file này thì các con số E10 chỉ còn là trích dẫn trong "
                 "docs/SPI_Discuss.md, không tái lập được.\n")
        return L
    fam = pd.read_csv(path)
    mg = float(c["e10_margin"])
    nb, seed = int(c["bootstrap"]), int(c["seed"])
    rows = []
    for alt in ("smooth", "lognormal", "ig"):
        for met in ("iou", "insdel_blur", "insdel_black", "energy_mask", "stab_cos_elem"):
            if met not in fam:
                continue
            piv = fam.pivot_table(index="image", columns="family", values=met)
            if alt not in piv or "dp" not in piv:
                continue
            v = (piv[alt] - piv["dp"]).dropna().values
            r = tost(v, mg, nb, seed)
            eq = r["p_tost_t"] < 0.05 and r["p_tost_boot"] < 0.05
            diff = r["ci95_boot"][0] > 0 or r["ci95_boot"][1] < 0
            rows.append({"so sánh": f"{alt} − dp", "metric": met, "n": r["n"], "TB hiệu": r["mean"], "trung vị hiệu": r["median"],
                         "CI 90 % t": f"[{r['ci90_t'][0]:.4f}, {r['ci90_t'][1]:.4f}]",
                         "CI 90 % bootstrap": f"[{r['ci90_boot'][0]:.4f}, {r['ci90_boot'][1]:.4f}]",
                         "p TOST (t)": r["p_tost_t"], "p TOST (bootstrap)": r["p_tost_boot"],
                         "kết luận": ("tương đương ±%g (TOST)" % mg) if eq else ("khác (CI 95 % loại 0)" + (" và có thể vẫn trong biên" if abs(r["mean"]) < mg else ""))
                         if diff else "CHƯA KẾT LUẬN — không được gọi là tương đương"})
    L.append(f"Biên ±{mg} (tuyệt đối, như E10 đã đăng ký). TOST: tương đương khi CẢ hai phía một phía bác bỏ ở 5 % (⇔ CI 90 % nằm trong biên). "
             "Không hiệu chỉnh đa kiểm định cho kết luận \"tương đương ở mọi metric\" (phép giao–hợp IUT không cần), nhưng kết luận theo "
             "từng metric riêng thì chưa hiệu chỉnh. Lưu ý: E10 đo IoU / InsDel / ổn định của bản đồ TRUNG BÌNH 16 đường — trộn kỳ vọng và "
             "phương sai, và KHÔNG đo J.\n")
    L.append(md(pd.DataFrame(rows)))
    return L


# ─────────────────────────────────────────────────────────────────────────────
# chạy
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> str:
    import pandas as pd
    from .report import md
    c = pcfg(cfg)
    out = os.path.join(C.out_dir(cfg, "pilot"), "infid", "pathfam")
    os.makedirs(out, exist_ok=True)
    d = int(c["mc_d"]) if not cfg.get("_debug") else 2000
    grid = matched_grid(cfg)
    rows = []
    for k, p in enumerate(grid):
        pr = properties(p["family"], int(p["N"]), p, d, int(c["seed"]) + k)
        rows.append({**{kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in p.items()}, **pr})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "family_properties.csv"), index=False)
    L = ["# PATH_FAMILIES — kiểm tính chất họ đường (CPU) và phân tích lại E10\n",
         f"Monte Carlo {d} phần tử độc lập mỗi cấu hình. Lưới khớp exploration theo h = 1/(α + 1) của DP (α {c['alphas']}, N {c['N']}).\n",
         "## 1. Tính chất từng họ (đây là điều kiện cần trước khi đưa vào benchmark J)\n",
         "- `endpoints_err`, `min_increment` ≥ 0: hai đầu mút và đơn điệu (phải đúng chính xác).\n"
         "- `mean_err_max` = max_k |Ê Z(t_k) − t_k| (phải ≈ 0: E Z = t ⇒ cùng tâm IG); `var_rel_err_max`: sai lệch tương đối của Var Z(t_k) so "
         "với h t_k(1 − t_k) (cmix khớp chính xác theo cấu trúc; ln chỉ khớp D̂ tích phân, không khớp từng t; smooth có Var = δ² t²(1 − t)², "
         "hình dạng khác).\n"
         "- `ks_rev_max` so với `ks_rev_ref_95`: phân phối biên Z(t) và đường đảo 1 − Z(1 − t) trùng nhau (điều kiện để `rev` giữ phân phối "
         "biên và kỳ vọng attribution).\n"
         "- `t_half_*`: thời điểm kích hoạt (Z vượt 1/2) — trung bình, độ lệch chuẩn, phân vị; `kurt_Z_mid`: độ nhọn của Z(1/2) (3 = chuẩn; "
         "cmix lớn ⇒ ít phần tử lệch mạnh); `max_jump_mean`, `eff_steps`: mức dồn tiến độ vào một bước; `corr_between_elems_mid` ≈ 0.\n",
         md(t)]
    L += toy_section(cfg)
    L += e10_section(cfg)
    rp = os.path.join(out, "PATH_FAMILIES_CHECKS.md")
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[pathfam-audit] {rp}")
    return rp
