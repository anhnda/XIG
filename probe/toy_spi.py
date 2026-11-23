"""
E-toy — tái lập các giá trị giải tích của Phụ lục (proposal infidelity) bằng ĐÚNG sampler của repo (paths.spi_increments +
cum_nodes, sampling.reverse_nodes), với cả điểm trái và điểm giữa. Không cần mô hình / dữ liệu ảnh. Torch, GPU nếu có.

  toy        E-toy nhỏ: oracle Phụ lục (d = 2–4: Monte Carlo lớn có sai số chuẩn theo lô, hoặc tính chính xác khi đường tất
             định), hàm không đa tuyến (z_i²z_j, sigmoid, ReLU) đo ảnh hưởng của α lên attribution kỳ vọng.
  toy-large  E-toy lớn (d = 256): quét N, α, đồ thị tương tác (cùng dấu, khác dấu, thưa, dày, hub), cỡ ô; đối chiếu dấu của
             V_o − V_e với điều kiện T9; kiểm Cov(H_ij, H_ik) ≈ 0; mô phỏng coverage của CI tỷ số ρ khi ít ảnh.

Đầu ra: <out>/pilot/infid/toy/{toy_oracles.csv, toy_nonmultilinear.csv, toy_large.csv, toy_cov_h.csv, toy_coverage.csv,
TOY_REPORT.md}. Tiêu chí khớp: |ước lượng − oracle| ≤ max(z_tol · SE, atol + rtol · |oracle|).
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import math
import os
import time

import numpy as np
import torch

from . import config as C
from .infid_core import _solve
from .pathquad import offsets
from .paths import cum_nodes, spi_increments

F64 = torch.float64


def device(cfg: dict) -> torch.device:
    d = (cfg.get("infid", {}).get("toy", {}) or {}).get("device") or (cfg.get("local", {}) or {}).get("device", "cuda")
    if str(d).startswith("cuda") and not torch.cuda.is_available():
        d = "cpu"
    return torch.device(d)


def toy_out(cfg: dict) -> str:
    d = os.path.join(C.out_dir(cfg, "pilot"), "infid", "toy")
    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# hàm, đường, quy tắc
# ─────────────────────────────────────────────────────────────────────────────

class Quad:
    """f(z) = aᵀz + ½ zᵀBz, B đối xứng (Hessian). ∇f = a + Bz. f = Σ_{i<j} b_ij z_i z_j ⇔ B_ij = B_ji = b_ij, B_ii = 0;
    hạng c·z_i² ⇔ B_ii = 2c."""

    def __init__(self, a: torch.Tensor, Bm: torch.Tensor):
        self.a, self.B = a.double(), ((Bm + Bm.T) / 2).double()

    def to(self, dev):
        return Quad(self.a.to(dev), self.B.to(dev))

    def f(self, Z):
        return Z @ self.a + 0.5 * ((Z @ self.B) * Z).sum(-1)

    def grad(self, Z):
        return self.a + Z @ self.B


class Auto:
    def __init__(self, f, name: str = ""):
        self._f, self.name = f, name

    def to(self, dev):
        return self

    def f(self, Z):
        return self._f(Z)

    def grad(self, Z):
        with torch.enable_grad():
            z = Z.detach().clone().requires_grad_(True)
            (g,) = torch.autograd.grad(self._f(z).sum(), z)
        return g.detach()


def quad_from_terms(d: int, lin=None, cross=None, sq=None) -> Quad:
    a = torch.zeros(d, dtype=F64)
    Bm = torch.zeros(d, d, dtype=F64)
    for i, v in (lin or {}).items():
        a[i] += v
    for (i, j), v in (cross or {}).items():
        Bm[i, j] += v
        Bm[j, i] += v
    for i, v in (sq or {}).items():
        Bm[i, i] += 2 * v
    return Quad(a, Bm)


def dir_paths(alpha: float, N: int, B: int, d: int, gen) -> torch.Tensor:
    """(B, N+1, d): lịch trình Dirichlet độc lập theo phần tử, đúng sampler của repo."""
    return cum_nodes(spi_increments(float(alpha), int(N), (int(B), int(d)), gen)).permute(1, 0, 2).contiguous()


def rev(Z: torch.Tensor) -> torch.Tensor:
    """Z^R_k = 1 − Z_{N−k} (≡ sampling.reverse_nodes trên trục nút)."""
    return 1.0 - Z.flip(1)


def edge_path(order, d: int) -> torch.Tensor:
    Z = torch.zeros(1, d + 1, d, dtype=F64)
    for k, i in enumerate(order):
        Z[0, k + 1:, i] = 1.0
    return Z


def straight_path(d: int, N: int = 1) -> torch.Tensor:
    t = torch.linspace(0, 1, N + 1, dtype=F64).view(1, -1, 1)
    return t.expand(1, N + 1, d).clone()


def attribute(fn, Z: torch.Tensor, rule: str, M: int = 1, theta: float = 0.5, gl: int = 6, chunk: int = 8192,
              gen=None) -> torch.Tensor:
    """(B, d). rule ∈ left | mid | theta | exact (Gauss–Legendre `gl` điểm/đoạn = tích phân chính xác trên polyline cho đa thức
    bậc ≤ 2gl−1) | rand (như aw_stream: s = (m + U)/M, U ~ U(0,1) riêng cho mỗi (đường, đoạn, m), dùng chung mọi phần tử;
    mỗi lần gọi rút U mới ⇒ Z và RZ gọi riêng có nhiễu quadrature độc lập có điều kiện)."""
    B, N1, d = Z.shape
    N = N1 - 1
    if rule == "rand":
        if gen is None:
            raise ValueError("rule = rand cần gen")
        w = torch.full((int(M),), 1.0 / int(M), dtype=F64, device=Z.device)
    else:
        s, w = offsets("gl", gl) if rule == "exact" else offsets(rule, M, theta)
        s, w = s.to(Z.device), w.to(Z.device)
    out = []
    for b0 in range(0, B, chunk):
        Zb = Z[b0:b0 + chunk]
        dZ = Zb[:, 1:] - Zb[:, :-1]
        if rule == "rand":
            U = torch.rand(Zb.shape[0], N, int(M), generator=gen, device=gen.device, dtype=F64).to(Z.device)
            sb = (torch.arange(int(M), device=Z.device, dtype=F64).view(1, 1, -1) + U) / int(M)
            P = Zb[:, :-1, None, :] + sb[..., None] * dZ[:, :, None, :]
        else:
            P = Zb[:, :-1, None, :] + s.view(1, 1, -1, 1) * dZ[:, :, None, :]
        g = fn.grad(P.reshape(-1, d)).view(P.shape)
        out.append(((g * w.view(1, 1, -1, 1)).sum(2) * dZ).sum(1))
    return torch.cat(out)


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo có sai số chuẩn theo lô
# ─────────────────────────────────────────────────────────────────────────────

def _mse(vals) -> tuple:
    v = torch.tensor(vals, dtype=F64)
    return float(v.mean()), float(v.std(unbiased=True) / math.sqrt(len(v))) if len(v) > 1 else math.nan


def pair_mc(fn, d: int, alpha: float, N: int, rule: str, Pm: torch.Tensor, wts: torch.Tensor, nb: int, b: int, gen,
            theta: float = 0.5) -> dict:
    """V_e, V_o (THÔ, không chuẩn hoá — như Phụ lục), ρ, V_indep(2), V_rev(2), trung bình chiếu; SE theo nb lô × b cặp."""
    dev = gen.device
    Pm, wts = Pm.to(dev, F64), wts.to(dev, F64)
    Ve, Vo, rho, mu, dv = [], [], [], [], []
    for _ in range(int(nb)):
        Z = dir_paths(alpha, N, b, d, gen)
        y1 = attribute(fn, Z, rule, theta=theta, gen=gen) @ Pm.T
        y2 = attribute(fn, rev(Z), rule, theta=theta, gen=gen) @ Pm.T
        E, O = (y1 + y2) / 2, (y1 - y2) / 2
        ve = float((E.var(0, unbiased=True) * wts).sum())
        vo = float(((O ** 2).mean(0) * wts).sum())
        Ve.append(ve)
        Vo.append(vo)
        dv.append(vo - ve)
        rho.append(2 * ve / (ve + vo) if ve + vo > 0 else math.nan)
        mu.append(float((y1.mean(0) * wts).sum()))
    ve, se_e = _mse(Ve)
    vo, se_o = _mse(Vo)
    r = 2 * ve / (ve + vo) if ve + vo > 0 else math.nan
    _, se_r = _mse(rho)
    return {"Ve": (ve, se_e), "Vo": (vo, se_o), "rho": (r, se_r), "Vind2": ((ve + vo) / 2, math.hypot(se_e, se_o) / 2),
            "Vrev2": (ve, se_e), "mu": _mse(mu), "VoVe": _mse(dv)}


def row(toy, setting, rule, qty, est, se, oracle, ztol=4.0, atol=1e-9, rtol=1e-7, note=""):
    err = abs(est - oracle)
    tol = max(ztol * (se if np.isfinite(se) else 0.0), atol + rtol * abs(oracle))
    return {"toy": toy, "setting": setting, "rule": rule, "qty": qty, "est": est, "se": se, "oracle": oracle,
            "z": err / se if se and np.isfinite(se) and se > 0 else (0.0 if err <= tol else math.inf), "pass": bool(err <= tol),
            "note": note}


# ─────────────────────────────────────────────────────────────────────────────
# oracle Phụ lục
# ─────────────────────────────────────────────────────────────────────────────

def var_w_formula(alpha: float, N: int) -> float:
    return (N - 1) * (2 * (N + 1) * alpha + 3 * N) / (12 * N * N * (alpha + 1) ** 2)


def oracle_var_w(tc: dict, gen) -> list:
    f = quad_from_terms(2, cross={(0, 1): 1.0}).to(gen.device)
    rows = []
    nb, b = int(tc["nb"]), int(tc["b"])
    for alpha in tc["alphas"]:
        for N in tc["N_var_w"]:
            for rule in ("exact", "mid"):
                vals = []
                for _ in range(nb):
                    A = attribute(f, dir_paths(alpha, N, b, 2, gen), rule)
                    vals.append(float(A[:, 0].var(unbiased=True)))
                est, se = _mse(vals)
                rows.append(row("Var(w_ij)", f"f=z1z2, α={alpha:g}, N={N}", rule, "Var(w_12)", est, se, var_w_formula(alpha, N)))
    for (alpha, N, val) in ((2, 2, 0.041667), (1, 4, 0.085938), (10, 8, 0.015367), (3, 16, 0.045776)):
        rows.append(row("Var(w_ij) kiểm nhanh", f"α={alpha}, N={N}", "công thức", "Var(w_12)", var_w_formula(alpha, N), 0.0, val,
                        atol=6e-7, rtol=0))
    for alpha in tc["alphas"]:
        lim = (2 * alpha + 3) / (12 * (alpha + 1) ** 2)
        rows.append(row("T8 tiệm cận", f"α={alpha:g}, N=1e6", "công thức", "Var(w) → (2α+3)/(12(α+1)²)",
                        var_w_formula(alpha, 10 ** 6), 0.0, lim, atol=0, rtol=1e-5))
    return rows


def oracle_H(tc: dict, gen) -> list:
    rows = []
    nb, b = int(tc["nb"]), int(tc["b"])
    for alpha in (1.0, 10.0):
        for N in (2, 4, 16):
            mH, vH, cv = [], [], []
            for _ in range(nb):
                dl = spi_increments(alpha, N, (b, 3), gen)          # (N, b, 3): ba lịch trình độc lập
                H12 = (dl[:, :, 0] * dl[:, :, 1]).sum(0)
                H13 = (dl[:, :, 0] * dl[:, :, 2]).sum(0)
                mH.append(float(H12.mean()))
                vH.append(float(H12.var(unbiased=True)))
                cv.append(float(((H12 - H12.mean()) * (H13 - H13.mean())).sum() / (b - 1)))
            s = f"α={alpha:g}, N={N}"
            rows.append(row("T7 H_ij", s, "—", "E[H_ij]", *_mse(mH), 1.0 / N))
            rows.append(row("T7 H_ij", s, "—", "Var(H_ij)", *_mse(vH), (N - 1) / (N * N * (alpha + 1) ** 2)))
            rows.append(row("T7 H_ij", s, "—", "Cov(H_ij, H_ik)", *_mse(cv), 0.0))
    return rows


def oracle_left_sq(tc: dict, gen) -> list:
    f = quad_from_terms(1, sq={0: 1.0}).to(gen.device)
    rows = []
    for alpha in (0.1, 1.0, 10.0):
        for N in (2, 8, 64):
            vals = [float(attribute(f, dir_paths(alpha, N, int(tc["b"]), 1, gen), "left")[:, 0].mean()) for _ in range(int(tc["nb"]))]
            rows.append(row("Điểm trái, f = z²", f"α={alpha:g}, N={N}", "left", "E[Q_L]", *_mse(vals),
                            (1 - 1 / N) * alpha / (alpha + 1), note="đáp số đúng = 1"))
    return rows


def _P(d, sets):
    Pm = torch.zeros(len(sets), d, dtype=F64)
    for r, S in enumerate(sets):
        Pm[r, list(S)] = 1.0
    return Pm


def oracle_rho(tc: dict, gen) -> list:
    rows = []
    nb, b = int(tc["nb"]), int(tc["b"])
    f = quad_from_terms(2, cross={(0, 1): 1.0}).to(gen.device)
    one = torch.ones(1, dtype=F64)
    # ρ phụ thuộc D (điểm trái), N = 2, α = 2
    r = pair_mc(f, 2, 2.0, 2, "left", _P(2, [(0, 1)]), one, nb, b, gen)
    rows += [row("ρ phụ thuộc D", "f=z1z2, N=2, α=2, D={1,2}", "left", "ρ", *r["rho"], 2.0),
             row("ρ phụ thuộc D", "f=z1z2, N=2, α=2, D={1,2}", "left", "Var(rev), K=2", *r["Vrev2"], 1 / 36),
             row("ρ phụ thuộc D", "f=z1z2, N=2, α=2, D={1,2}", "left", "Var(indep), K=2", *r["Vind2"], 1 / 72)]
    r = pair_mc(f, 2, 2.0, 2, "left", _P(2, [(0,)]), one, nb, b, gen)
    rows.append(row("ρ phụ thuộc D", "f=z1z2, N=2, α=2, D={1}", "left", "ρ", *r["rho"], 2 / 7))
    # một cạnh cắt
    for N in (2, 4, 8):
        for alpha in (1.0, 2.0, 10.0):
            s = f"f=z1z2, D={{1}}, N={N}, α={alpha:g}"
            r = pair_mc(f, 2, alpha, N, "left", _P(2, [(0,)]), one, nb, b, gen)
            rows.append(row("ρ một cạnh cắt", s, "left", "ρ", *r["rho"], 6 / ((N + 1) * (2 * alpha + 3))))
            r = pair_mc(f, 2, alpha, N, "mid", _P(2, [(0,)]), one, nb, b, gen)
            rows.append(row("ρ một cạnh cắt", s, "mid", "ρ", *r["rho"], 0.0, atol=1e-8))
    # ngưỡng N*
    g4 = quad_from_terms(4, lin={0: 1.0}, cross={(0, 2): 1.0, (0, 3): -1.0, (1, 2): -1.0, (1, 3): 1.0}).to(gen.device)
    for N in (2, 4, 6, 8, 64):
        for alpha in (1.0, 5.0):
            s = f"f=z1+(z1−z2)(z3−z4), S={{1,2}}, N={N}, α={alpha:g}"
            r = pair_mc(g4, 4, alpha, N, "left", _P(4, [(0, 1)]), one, nb, b, gen)
            rows.append(row("Ngưỡng N*", s, "left", "ρ", *r["rho"], 6 / (N + 1)))
            r = pair_mc(g4, 4, alpha, N, "mid", _P(4, [(0, 1)]), one, nb, b, gen)
            rows.append(row("Ngưỡng N*", s, "mid", "ρ", *r["rho"], 0.0, atol=1e-8))
    # kiểm âm tính phạm vi T9: f = 10 z1² + z1 z2
    h = quad_from_terms(2, cross={(0, 1): 1.0}, sq={0: 10.0}).to(gen.device)
    s = "f=10z1²+z1z2, N=2, α=2, S={1}"
    r = pair_mc(h, 2, 2.0, 2, "left", _P(2, [(0,)]), one, nb, b, gen)
    t9 = t9_condition(torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=F64), [0], 2, 2.0)
    rows += [row("Kiểm âm tính T9", s, "left", "V_e", *r["Ve"], 107 / 48),
             row("Kiểm âm tính T9", s, "left", "V_o", *r["Vo"], 1 / 24),
             row("Kiểm âm tính T9", s, "left", "ρ", *r["rho"], 214 / 109,
                 note=f"điều kiện T9 (bỏ hạng z²) dự đoán rev {'thắng' if t9['rev_wins'] else 'thua'}")]
    r = pair_mc(h, 2, 2.0, 2, "mid", _P(2, [(0,)]), one, nb, b, gen)
    rows.append(row("Kiểm âm tính T9", s, "mid", "ρ", *r["rho"], 0.0, atol=1e-8))
    return rows


def oracle_theta(tc: dict, gen) -> list:
    """T6: phần lẻ O không đổi theo θ (từng đường, không chỉ phương sai) vì số hạng hiệu chỉnh (θ − ½)·B∘H chẵn."""
    gc = torch.Generator().manual_seed(5)
    d = 4
    Bm = torch.randn(d, d, generator=gc, dtype=F64)
    f = Quad(torch.randn(d, generator=gc, dtype=F64), Bm + Bm.T).to(gen.device)
    Pm = _P(d, [(0, 2), (1,), (0, 1, 3)]).to(gen.device)
    Z = dir_paths(3.0, 8, int(tc["b"]), d, gen)
    ref = None
    rows = []
    for th in (0.0, 0.25, 0.5, 1.0):
        O = (attribute(f, Z, "theta", theta=th) - attribute(f, rev(Z), "theta", theta=th)) @ Pm.T / 2
        if ref is None:
            ref = O
        rows.append(row("T6 họ θ", f"quadratic d=4, α=3, N=8, θ={th:g}", "theta", "max|O(θ) − O(0)| (từng đường)",
                        float((O - ref).abs().max()), 0.0, 0.0, atol=1e-10))
    # và số hạng hiệu chỉnh đúng (θ − ½)·Σ_j B_ij H_ij
    dl = Z[:, 1:] - Z[:, :-1]
    H = torch.einsum("bni,bnj->bij", dl, dl)
    corr = 0.25 * (f.B.unsqueeze(0) * H).sum(2)
    A75, Aex = attribute(f, Z, "theta", theta=0.75), attribute(f, Z, "exact")
    rows.append(row("T6 họ θ", "A^θ − A^exact = (θ − ½)·(B∘H)1, θ = 0,75", "theta", "max sai khác", float((A75 - Aex - corr).abs().max()),
                    0.0, 0.0, atol=1e-10))
    return rows


def oracle_c3(tc: dict, gen) -> list:
    """C3a: `rand` không chệch CÓ ĐIỀU KIỆN theo đường (cố định Z, trung bình theo U → tích phân chính xác trên polyline, f phi tuyến).
    C3b: nhiễu quadrature độc lập trong cặp (Z, RZ) cộng τ²/2 vào CẢ V_e lẫn V_o ⇒ V_o − V_e không đổi so với tích phân chính xác."""
    rows = []
    dev = gen.device
    nb, b = int(tc["nb"]), int(tc["b"])
    f = Auto(lambda Z: torch.sigmoid(4 * Z.sum(1) - 6) * (1 + Z[:, 0]), "sigmoid")
    Z1 = dir_paths(3.0, 8, 1, 3, gen)
    ex = attribute(f, Z1, "exact", gl=12)[0]
    acc = [attribute(f, Z1.expand(b, -1, -1), "rand", gen=gen).mean(0) for _ in range(nb)]
    acc = torch.stack(acc)
    mu, se = acc.mean(0), acc.std(0, unbiased=True) / math.sqrt(nb)
    for i in range(3):
        rows.append(row("C3a rand không chệch | Z", "f=σ(4Σz−6)(1+z1), d=3, N=8, α=3, 1 đường cố định", "rand", f"E_U[A_{i + 1}]",
                        float(mu[i]), float(se[i]), float(ex[i])))
    # C3b, f bậc 2: tích phân chính xác ⇒ V_e = 0 (T4), V_o = 1/24 ⇒ với rand: V_o − V_e = 1/24, V_e = τ²/2 > 0
    h = quad_from_terms(2, cross={(0, 1): 1.0}, sq={0: 10.0}).to(dev)
    one = torch.ones(1, dtype=F64)
    r = pair_mc(h, 2, 2.0, 2, "rand", _P(2, [(0,)]), one, nb, b, gen)
    s = "f=10z1²+z1z2, N=2, α=2, S={1}, U độc lập trong cặp"
    rows.append(row("C3b nhiễu độc lập trong cặp", s, "rand", "V_o − V_e", *r["VoVe"], 1 / 24))
    rows.append(row("C3b nhiễu độc lập trong cặp", s, "rand", "V_e (= τ²/2 > 0)", *r["Ve"], r["Ve"][0], note="chỉ báo, oracle = chính nó"))
    # C3b, f phi tuyến: V_o − V_e (rand) − V_o − V_e (chính xác) = 0
    g3 = Auto(lambda Z: torch.sigmoid(3 * Z[:, 0] * Z[:, 1] + Z[:, 2] - 1.5), "sig3")
    P3 = _P(3, [(0,), (1, 2)])
    w3 = torch.ones(2, dtype=F64)
    ra = pair_mc(g3, 3, 3.0, 4, "rand", P3, w3, nb, b // 4, gen)
    re = pair_mc(g3, 3, 3.0, 4, "exact", P3, w3, nb, b // 4, gen)
    rows.append(row("C3b nhiễu độc lập trong cặp", "f=σ(3z1z2+z3−1,5), d=3, N=4, α=3, D={1},{2,3}", "rand − exact",
                    "(V_o − V_e)_rand − (V_o − V_e)_exact", ra["VoVe"][0] - re["VoVe"][0], math.hypot(ra["VoVe"][1], re["VoVe"][1]), 0.0))
    return rows


def oracle_deterministic() -> list:
    rows = []
    # IG vs đường cạnh, f = z1² z2
    f = Auto(lambda Z: Z[:, 0] ** 2 * Z[:, 1])
    ig = attribute(f, straight_path(2, 1), "exact")[0]
    e = (attribute(f, edge_path([0, 1], 2), "exact")[0] + attribute(f, edge_path([1, 0], 2), "exact")[0]) / 2
    rows += [row("IG vs đường cạnh", "f=z1²z2", "exact", "IG_1", float(ig[0]), 0, 2 / 3),
             row("IG vs đường cạnh", "f=z1²z2", "exact", "IG_2", float(ig[1]), 0, 1 / 3),
             row("IG vs đường cạnh", "f=z1²z2", "exact", "TB cạnh_1 (= Shapley)", float(e[0]), 0, 0.5),
             row("IG vs đường cạnh", "f=z1²z2", "exact", "TB cạnh_2 (= Shapley)", float(e[1]), 0, 0.5)]
    # completeness không đủ để kiểm hội tụ
    g = Auto(lambda Z: Z[:, 0] ** 2 * Z[:, 1] - Z[:, 0] * Z[:, 1] ** 2)
    ex = attribute(g, straight_path(2, 1), "exact")[0]
    md = attribute(g, straight_path(2, 1), "mid")[0]
    rows += [row("Completeness ≠ hội tụ", "f=z1²z2 − z1z2², thẳng", "exact", "A_1", float(ex[0]), 0, 1 / 3),
             row("Completeness ≠ hội tụ", "f=z1²z2 − z1z2², thẳng", "mid (1 điểm)", "A_1", float(md[0]), 0, 1 / 4),
             row("Completeness ≠ hội tụ", "f=z1²z2 − z1z2², thẳng", "exact", "khoảng hở completeness", float(ex.sum()), 0, 0.0),
             row("Completeness ≠ hội tụ", "f=z1²z2 − z1z2², thẳng", "mid (1 điểm)", "khoảng hở completeness", float(md.sum()), 0, 0.0)]
    # phần chẵn có khi tích phân chính xác: f = z1 z2 z3, đường cạnh
    h = Auto(lambda Z: Z[:, 0] * Z[:, 1] * Z[:, 2])
    for o, exp in (((0, 1, 2), (0.5, 0.0, 0.5)), ((1, 0, 2), (0.0, 0.5, 0.5))):
        Zo = edge_path(list(o), 3)
        Zr = rev(Zo)
        pm = (attribute(h, Zo, "exact")[0] + attribute(h, Zr, "exact")[0]) / 2
        s = f"cặp ({'→'.join(str(i + 1) for i in o)}, đảo)"
        for i in range(3):
            rows.append(row("Phần chẵn khi tích phân chính xác", f"f=z1z2z3, {s}", "exact", f"TB cặp_{i + 1}", float(pm[i]), 0, exp[i]))
    # sàn có completeness: f = z1 z2, D xoá {1} hoặc {2} đều
    X = torch.eye(2, dtype=F64)
    dF = torch.tensor([1.0, 1.0], dtype=F64)              # f(1,1) − f(0,1) = f(1,1) − f(1,0) = 1
    af = _solve(X, dF, 0.0)
    ac = _solve(X, dF, 0.0, Dsum=1.0)                    # Σ A = f(1,1) − f(0,0) = 1
    L = lambda a: float(((X @ a - dF) ** 2).sum() / (dF ** 2).sum())
    rows += [row("Sàn có completeness", "f=z1z2, D xoá từng biến", "—", "L*_free", L(af), 0, 0.0),
             row("Sàn có completeness", "f=z1z2, D xoá từng biến", "—", "A*_free_1", float(af[0]), 0, 1.0),
             row("Sàn có completeness", "f=z1z2, D xoá từng biến", "—", "L*_comp", L(ac), 0, 0.25),
             row("Sàn có completeness", "f=z1z2, D xoá từng biến", "—", "A*_comp_1", float(ac[0]), 0, 0.5)]
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# hàm không đa tuyến: α đổi attribution kỳ vọng
# ─────────────────────────────────────────────────────────────────────────────

NONML = {
    "z1²z2": (2, lambda Z: Z[:, 0] ** 2 * Z[:, 1], 4),
    "z1z2²z3": (3, lambda Z: Z[:, 0] * Z[:, 1] ** 2 * Z[:, 2], 4),
    "sigmoid(4(z1+z2+z3)−6)": (3, lambda Z: torch.sigmoid(4 * Z.sum(1) - 6), 8),
    "relu(z1+z2−1)·(1+z3)": (3, lambda Z: torch.relu(Z[:, 0] + Z[:, 1] - 1) * (1 + Z[:, 2]), 16),
    "z1z2 (đa tuyến, đối chứng)": (2, lambda Z: Z[:, 0] * Z[:, 1], 2),
}


def nonmultilinear(tc: dict, gen) -> list:
    rows = []
    for name, (d, fz, gl) in NONML.items():
        f = Auto(fz, name)
        ig = attribute(f, straight_path(d, 64), "exact", gl=gl)[0]
        edges = torch.stack([attribute(f, edge_path(list(o), d), "exact", gl=gl)[0] for o in itertools.permutations(range(d))]).mean(0)
        for alpha in tc["alphas_nonml"]:
            acc = []
            for _ in range(int(tc["nb"])):
                acc.append(attribute(f, dir_paths(alpha, int(tc["N_nonml"]), int(tc["b_nonml"]), d, gen), "exact", gl=gl).mean(0))
            acc = torch.stack(acc)
            mu, se = acc.mean(0), acc.std(0, unbiased=True) / math.sqrt(acc.shape[0])
            for i in range(d):
                rows.append({"f": name, "alpha": alpha, "N": int(tc["N_nonml"]), "i": i + 1, "E_A": float(mu[i]), "se": float(se[i]),
                             "IG": float(ig[i]), "edge_avg": float(edges[i]), "gl": gl,
                             "note": "ReLU: Gauss–Legendre không chính xác tại điểm gãy" if "relu" in name else ""})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# E-toy lớn (d = 256): T9 và Cov(H_ij, H_ik)
# ─────────────────────────────────────────────────────────────────────────────

def make_graph(kind: str, d: int, g: torch.Generator) -> torch.Tensor:
    """Ma trận hệ số b_ij đối xứng, đường chéo 0 (chỉ tương tác chéo — đúng phạm vi T9)."""
    if kind == "dense_pos":
        b = torch.rand(d, d, generator=g, dtype=F64)
    elif kind == "dense_mixed":
        b = torch.randn(d, d, generator=g, dtype=F64)
    elif kind == "sparse_mixed":
        b = torch.randn(d, d, generator=g, dtype=F64) * (torch.rand(d, d, generator=g, dtype=F64) < 4.0 / d)
    elif kind == "sparse_pos":
        b = torch.rand(d, d, generator=g, dtype=F64) * (torch.rand(d, d, generator=g, dtype=F64) < 4.0 / d)
    elif kind == "hub":
        b = torch.randn(d, d, generator=g, dtype=F64) * (torch.rand(d, d, generator=g, dtype=F64) < 2.0 / d)
        b = torch.triu(b, 1)
        b = b + b.T
        hubs = torch.randperm(d, generator=g)[:8]
        hv = torch.randn(8, d, generator=g, dtype=F64)
        b[hubs, :] = hv
        b[:, hubs] = hv.T
        b.fill_diagonal_(0.0)
        return (b + b.T) / 2
    else:
        raise ValueError(kind)
    b = torch.triu(b, 1)
    return b + b.T


def t9_condition(b: torch.Tensor, S, N: int, alpha: float) -> dict:
    """(N−5)·B_cut + (N+1)(α+1)·G_S > 12·B_in ⇔ rev thắng (điểm trái, chỉ tương tác chéo, Dirichlet độc lập)."""
    d = b.shape[0]
    inS = torch.zeros(d, dtype=torch.bool)
    inS[list(S)] = True
    iu = torch.triu(torch.ones(d, d, dtype=torch.bool), 1)
    both = inS.view(-1, 1) & inS.view(1, -1)
    cut = inS.view(-1, 1) ^ inS.view(1, -1)
    B_in = float((b[iu & both] ** 2).sum())
    B_cut = float((b[iu & cut] ** 2).sum())
    s = (b * cut).sum(1)                                   # tổng hệ số các cạnh cắt tại mỗi đỉnh
    G_S = float((s ** 2).sum())
    lhs = (N - 5) * B_cut + (N + 1) * (alpha + 1) * G_S
    return {"B_in": B_in, "B_cut": B_cut, "G_S": G_S, "t9_lhs": lhs, "t9_rhs": 12 * B_in, "rev_wins": lhs > 12 * B_in}


def run_large(cfg: dict) -> None:
    tc = (cfg.get("infid", {}) or {}).get("toy", {}) or {}
    lc = tc.get("large", {}) or {}
    dev = device(cfg)
    out = toy_out(cfg)
    d = int(lc.get("d", 256))
    gen = C.generator(C.seed_of(cfg["seed"], "toy-large"), dev)
    gc = torch.Generator().manual_seed(C.seed_of(cfg["seed"], "toy-large-graph"))
    rows, t0 = [], time.time()
    for kind in lc.get("graphs", ["dense_pos", "dense_mixed", "sparse_mixed", "sparse_pos", "hub"]):
        b = make_graph(kind, d, gc)
        f = Quad(torch.zeros(d, dtype=F64), b).to(dev)
        for rep in range(int(lc.get("reps", 2))):
            for sz in lc.get("S_sizes", [8, 32, 128]):
                S = torch.randperm(d, generator=gc)[:int(sz)].tolist()
                Pm = _P(d, [S])
                for N in lc.get("N", [4, 16, 64]):
                    for alpha in lc.get("alphas", [1, 10, 100]):
                        t9 = t9_condition(b, S, int(N), float(alpha))
                        r = pair_mc(f, d, float(alpha), int(N), "left", Pm, torch.ones(1, dtype=F64), int(lc.get("nb", 8)),
                                    int(lc.get("b", 512)), gen)
                        rm = pair_mc(f, d, float(alpha), int(N), "mid", Pm, torch.ones(1, dtype=F64), 2, int(lc.get("b", 512)), gen)
                        diff, se = r["Vo"][0] - r["Ve"][0], math.hypot(r["Vo"][1], r["Ve"][1])
                        decisive = abs(diff) > 3 * se
                        rows.append({"graph": kind, "rep": rep, "S_size": int(sz), "N": int(N), "alpha": float(alpha), **t9,
                                     "Ve": r["Ve"][0], "Vo": r["Vo"][0], "VoVe": diff, "VoVe_se": se, "rho": r["rho"][0],
                                     "decisive": decisive, "mc_rev_wins": diff > 0,
                                     "agree": (diff > 0) == t9["rev_wins"] if decisive else None,
                                     "mid_Ve_rel": rm["Ve"][0] / max(rm["Ve"][0] + rm["Vo"][0], 1e-300)})
        print(f"  [toy-large] {kind}: {len(rows)} cấu hình, {time.time() - t0:.0f}s", flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out, "toy_large.csv"), index=False)
    # Cov(H_ij, H_ik) ở d = 256
    ch = []
    for alpha in lc.get("alphas", [1, 10, 100]):
        for N in lc.get("N", [4, 16, 64]):
            dl = spi_increments(float(alpha), int(N), (int(lc.get("b", 512)) * 8, d), gen)
            tr = torch.randint(0, d, (int(lc.get("cov_triples", 200)), 3), generator=gc)
            tr = tr[(tr[:, 0] != tr[:, 1]) & (tr[:, 0] != tr[:, 2]) & (tr[:, 1] != tr[:, 2])]
            Hij = (dl[:, :, tr[:, 0]] * dl[:, :, tr[:, 1]]).sum(0)
            Hik = (dl[:, :, tr[:, 0]] * dl[:, :, tr[:, 2]]).sum(0)
            n = Hij.shape[0]
            cv = ((Hij - Hij.mean(0)) * (Hik - Hik.mean(0))).sum(0) / (n - 1)
            vH = (int(N) - 1) / (int(N) ** 2 * (float(alpha) + 1) ** 2)
            se = vH / math.sqrt(n)                         # SE của hiệp phương sai mẫu khi độc lập ≈ σ²/√n
            ch.append({"alpha": float(alpha), "N": int(N), "n_triples": int(len(cv)), "mean_cov": float(cv.mean()),
                       "max_abs_z": float((cv.abs() / se).max()), "frac_abs_z_gt3": float(((cv.abs() / se) > 3).double().mean()),
                       "Var_H": vH})
    pd.DataFrame(ch).to_csv(os.path.join(out, "toy_cov_h.csv"), index=False)
    if lc.get("coverage", True):
        cov = coverage_study(cfg, dev)
        pd.DataFrame(cov).to_csv(os.path.join(out, "toy_coverage.csv"), index=False)
    with open(os.path.join(out, "meta_large.json"), "w") as fh:
        json.dump({"time": _dt.datetime.now().isoformat(timespec="seconds"), "minutes": (time.time() - t0) / 60, "large": lc,
                   "device": str(dev)}, fh, indent=1, default=str)
    write_report(cfg)


def coverage_study(cfg: dict, dev) -> list:
    """Coverage của CI bootstrap theo ảnh cho ρ (tỷ số Σ2V_e/Σ(V_e+V_o)) và cho ΔV (chênh lệch tuyệt đối) khi ít ảnh.
    'Ảnh' = toy bậc hai ngẫu nhiên (d = 16, có hạng z_i², điểm trái) với ô ngẫu nhiên; giá trị đúng từ Monte Carlo lớn."""
    from .infid_stats import boot_ci, boot_ratio
    cc = ((cfg.get("infid", {}) or {}).get("toy", {}) or {}).get("coverage", {}) or {}
    d, N, alpha = int(cc.get("d", 16)), int(cc.get("N", 8)), float(cc.get("alpha", 10))
    gen = C.generator(C.seed_of(cfg["seed"], "toy-coverage"), dev)
    gc = torch.Generator().manual_seed(C.seed_of(cfg["seed"], "toy-coverage-inst"))
    n_pool = int(cc.get("n_pool", 64))
    insts = []
    for _ in range(n_pool):
        Bm = torch.randn(d, d, generator=gc, dtype=F64)
        f = Quad(torch.randn(d, generator=gc, dtype=F64), Bm + Bm.T).to(dev)
        Pm = _P(d, [torch.randperm(d, generator=gc)[:4].tolist() for _ in range(6)])
        r = pair_mc(f, d, alpha, N, "left", Pm, torch.ones(6, dtype=F64), int(cc.get("truth_nb", 16)), int(cc.get("truth_b", 8192)), gen)
        insts.append((f, Pm, r["Ve"][0], r["Vo"][0]))
    out = []
    B = int(cc.get("bootstrap", 2000))
    for n_img in cc.get("n_images", [8, 16]):
        for m in cc.get("m", [8, 32]):
            hit_r, hit_d = [], []
            for rep in range(int(cc.get("reps", 200))):
                pick = torch.randint(0, n_pool, (int(n_img),), generator=gc).tolist()
                ve_t = sum(insts[i][2] for i in pick)
                vo_t = sum(insts[i][3] for i in pick)
                rho_t = 2 * ve_t / (ve_t + vo_t)
                dv_t = (vo_t - ve_t) / len(pick)
                ve, vo = [], []
                for i in pick:
                    f, Pm, _, _ = insts[i]
                    r = pair_mc(f, d, alpha, N, "left", Pm, torch.ones(6, dtype=F64), 1, int(m), gen)
                    ve.append(r["Ve"][0])
                    vo.append(r["Vo"][0])
                ve, vo = np.array(ve), np.array(vo)
                _, lo, hi = boot_ratio(2 * ve, ve + vo, B, rep)
                hit_r.append(lo <= rho_t <= hi)
                _, lo, hi = boot_ci(vo - ve, B, rep)
                hit_d.append(lo <= dv_t <= hi)
            out.append({"n_images": int(n_img), "m": int(m), "reps": len(hit_r), "coverage_rho": float(np.mean(hit_r)),
                        "coverage_dV": float(np.mean(hit_d)), "nominal": 0.95,
                        "note": "đích = trung bình trên đúng các 'ảnh' đã rút (điều kiện theo ảnh)"})
            print(f"  [toy-coverage] n = {n_img}, m = {m}: ρ {out[-1]['coverage_rho']:.3f}, ΔV {out[-1]['coverage_dV']:.3f}", flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# chạy + báo cáo
# ─────────────────────────────────────────────────────────────────────────────

TOY_DEFAULT = {"nb": 32, "b": 65536, "alphas": [1, 3, 10, 30, 100, 1000], "N_var_w": [2, 4, 8, 16],
               "alphas_nonml": [0.3, 1, 3, 10, 30, 100, 1000], "N_nonml": 16, "b_nonml": 32768, "z_tol": 4.0}


def tcfg(cfg: dict) -> dict:
    return C._merge(TOY_DEFAULT, ((cfg.get("infid", {}) or {}).get("toy", {}) or {}))


def run_small(cfg: dict) -> None:
    import pandas as pd
    tc = tcfg(cfg)
    dev = device(cfg)
    out = toy_out(cfg)
    gen = C.generator(C.seed_of(cfg["seed"], "toy-small"), dev)
    t0 = time.time()
    rows = []
    for name, fnc in (("deterministic", lambda: oracle_deterministic()), ("var_w", lambda: oracle_var_w(tc, gen)),
                      ("H", lambda: oracle_H(tc, gen)), ("left_sq", lambda: oracle_left_sq(tc, gen)),
                      ("rho", lambda: oracle_rho(tc, gen)), ("theta", lambda: oracle_theta(tc, gen)),
                      ("c3", lambda: oracle_c3(tc, gen))):
        r = fnc()
        rows += r
        print(f"  [toy] {name}: {sum(x['pass'] for x in r)}/{len(r)} khớp, {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "toy_oracles.csv"), index=False)
    nm = nonmultilinear(tc, gen)
    pd.DataFrame(nm).to_csv(os.path.join(out, "toy_nonmultilinear.csv"), index=False)
    with open(os.path.join(out, "meta_small.json"), "w") as fh:
        json.dump({"time": _dt.datetime.now().isoformat(timespec="seconds"), "minutes": (time.time() - t0) / 60, "toy": tc,
                   "device": str(dev), "n_pass": int(sum(r["pass"] for r in rows)), "n_total": len(rows)}, fh, indent=1, default=str)
    write_report(cfg)


def write_report(cfg: dict) -> str:
    from .report import _read, md
    out = toy_out(cfg)
    rp = os.path.join(out, "TOY_REPORT.md")
    L = ["# E-toy — tái lập oracle giải tích (proposal infidelity)\n"]
    if cfg.get("_debug"):
        L.append("> **DEBUG** — ít mẫu, chỉ kiểm pipeline.\n")
    o = _read(out, "toy_oracles")
    if o is not None:
        n_ok, n = int(o["pass"].sum()), len(o)
        L.append(f"## 1. Oracle Phụ lục: {n_ok}/{n} khớp\n")
        L.append("Khớp nếu |ước lượng − oracle| ≤ max(z_tol·SE, atol + rtol·|oracle|); SE theo lô Monte Carlo. Phương sai là THÔ "
                 "(không chuẩn hoá), như Phụ lục. Sampler = `paths.spi_increments` + `cum_nodes`; đảo = `1 − Z.flip`.\n")
        bad = o[~o["pass"]]
        if len(bad):
            L.append("**Không khớp** (code hoặc oracle cần xem lại trước E1):\n")
            L.append(md(bad[["toy", "setting", "rule", "qty", "est", "se", "oracle", "z"]]))
        tab = o.groupby("toy").agg(n=("pass", "size"), khớp=("pass", "sum"), max_z=("z", "max")).reset_index()
        L.append(md(tab))
        L.append("<details><summary>Toàn bộ dòng</summary>\n")
        L.append(md(o[["toy", "setting", "rule", "qty", "est", "se", "oracle", "pass", "note"]]))
        L.append("</details>\n")
    nm = _read(out, "toy_nonmultilinear")
    if nm is not None:
        L.append("## 2. Hàm không đa tuyến: attribution kỳ vọng theo α (tích phân chính xác trên polyline)\n")
        L.append("Với hàm đa tuyến mọi đường hoán đổi được cho cùng kỳ vọng (T1) — dòng đối chứng z1z2 phải phẳng theo α. Với hàm "
                 "không đa tuyến, α nhỏ → trung bình đường cạnh, α lớn → IG.\n")
        piv = nm.pivot_table(index=["f", "i"], columns="alpha", values="E_A").reset_index()
        ext = nm.groupby(["f", "i"])[["IG", "edge_avg"]].first().reset_index()
        L.append(md(piv.merge(ext, on=["f", "i"])))
    lg = _read(out, "toy_large")
    if lg is not None:
        dec = lg[lg.decisive == True]  # noqa: E712
        L.append("## 3. E-toy lớn (d = 256): dấu của V_o − V_e so với điều kiện T9 (điểm trái, chỉ tương tác chéo)\n")
        L.append(f"- {len(lg)} cấu hình; {len(dec)} có |V_o − V_e| > 3·SE; trong đó T9 khớp dấu "
                 f"{int(dec.agree.astype(bool).sum())}/{len(dec)}.")
        L.append(f"- Điểm giữa: max V_e/(V_e+V_o) = {lg.mid_Ve_rel.max():.2e} (T4: phải ≈ 0).\n")
        L.append(md(lg.groupby(["graph", "N"]).agg(n=("agree", "size"), decisive=("decisive", "sum"),
                                                     rev_wins_mc=("mc_rev_wins", "mean"), rho_med=("rho", "median")).reset_index()))
        if len(dec) and (dec.agree == False).any():  # noqa: E712
            L.append("Cấu hình bất đồng:\n")
            L.append(md(dec[dec.agree == False][["graph", "S_size", "N", "alpha", "B_in", "B_cut", "G_S", "VoVe", "VoVe_se"]]))  # noqa: E712
    ch = _read(out, "toy_cov_h")
    if ch is not None:
        L.append("## 4. Cov(H_ij, H_ik) ≈ 0 (T7, d = 256)\n")
        L.append(md(ch))
    cv = _read(out, "toy_coverage")
    if cv is not None:
        L.append("## 5. Coverage của CI bootstrap theo ảnh (danh nghĩa 95%)\n")
        L.append("ρ là tỷ số (có chệch hữu hạn mẫu) — luôn báo kèm CI cho chênh lệch tuyệt đối ΔV.\n")
        L.append(md(cv))
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[toy] {rp}")
    return rp
