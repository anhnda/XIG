"""
PATH_FAMILIES Phase 2 — kiểm số các định lý ở mức phân phối đường (CPU, không mô hình). docs/PATH_FAMILIES_THEORY.md.

  T2  F bậc 2: A_i(Z) − IG_i = Δ_i Σ_j H_ij Δ_j 𝔄_ij(Z) THEO TỪNG ĐƯỜNG, 𝔄_ij = ½∫(Z_j dZ_i − Z_i dZ_j) (diện tích Lévy);
      phân phối bất biến khi đảo ⇒ E 𝔄 = 0 ⇒ μ_Q = IG, kể cả khi các feature PHỤ THUỘC nhau và E Z ≠ t.
  T5  F bậc 2: A(Z) + A(RZ) = 2·IG theo từng đường ⇒ cặp `rev` khử HOÀN TOÀN phương sai đường bậc 2 (mọi Q).
  T4a F bậc 2, feature độc lập, E Z = t: Var ∫Z_j dZ_i = 2Γ + Λ theo hàm hiệp phương sai c(s, t) của MỘT lịch trình;
      c = h(min(s, t) − st) (dp, cmix): h/6 + h²/12; smooth: δ²/18.
  T4b F bậc 3: dp và cmix (cùng c, cùng μ) có phương sai attribution KHÁC nhau ⇒ phương sai phụ thuộc moment bậc cao hơn μ.
Lệnh: bash scripts/run.sh pathfam-theory → runs_probe/pilot/infid/pathfam/PATH_FAMILIES_THEORY_CHECKS.md
"""

from __future__ import annotations

import math
import os

import torch

from . import config as C
from . import path_families as PF

F64 = torch.float64


def quad_poly(d: int, seed: int) -> dict:
    P = PF.toy_poly(d, seed)
    P["T"] = torch.zeros_like(P["T"])
    P["U"] = torch.zeros_like(P["U"])
    P["W"] = torch.zeros_like(P["W"])
    return P


def levy_area(Z: torch.Tensor) -> torch.Tensor:
    """𝔄_ij = ½∫(Z_j dZ_i − Z_i dZ_j) chính xác trên đường nội suy tuyến tính. Z (N+1, P, d) -> (P, d, d)."""
    Zm = (Z[1:] + Z[:-1]) / 2                                                 # trung điểm đoạn (đúng cho tích phân tuyến tính)
    dZ = Z[1:] - Z[:-1]
    I = torch.einsum("npj,npi->pij", Zm, dZ)                                  # I[p, i, j] = ∫ Z_j dZ_i
    return 0.5 * (I - I.transpose(1, 2))


def schedules(kind: str, N: int, P: int, d: int, g: torch.Generator) -> torch.Tensor:
    """Các họ để kiểm T2 / T5: độc lập (dp, cmix, ln, smooth), PHỤ THUỘC bất biến khi đảo (group: feature trong nhóm dùng chung
    lịch trình dp), và KHÔNG bất biến khi đảo (power: Z_i = t^{a_i}, a cố định theo feature)."""
    if kind == "group":
        Zg = PF.schedule("dp", N, (P, (d + 1) // 2), g, {"alpha": 1.0})
        return Zg.repeat_interleave(2, dim=2)[..., :d]
    if kind == "power":
        t = torch.linspace(0, 1, N + 1, dtype=F64).view(-1, 1, 1)
        a = torch.tensor([0.5, 2.0, 1.0, 3.0][:d] + [1.0] * max(0, d - 4), dtype=F64)
        return (t ** a).expand(N + 1, P, d).clone()
    fam, p = {"dp": ("dp", {"alpha": 1.0}), "cmix": ("cmix", {"lam": 0.5}), "smooth": ("smooth", {"delta": 1.0}),
              "ln": ("ln", {"sigma": PF.sigma_for(1.0, N)})}[kind]
    return PF.schedule(fam, N, (P, d), g, p)


def t2_t5(N: int = 32, P: int = 100000, d: int = 4, seed: int = 0) -> list:
    Pq = quad_poly(d, 300 + seed)
    dx = torch.tensor([1.0, -0.7, 1.3, 0.5][:d], dtype=F64)
    Zig = torch.linspace(0, 1, N + 1, dtype=F64).view(-1, 1, 1).expand(N + 1, 1, d)
    ig = PF.exact_attr(Pq, Zig, dx)[0]
    H = Pq["Q"]
    rows = []
    for k, kind in enumerate(("dp", "cmix", "ln", "smooth", "group", "power")):
        g = torch.Generator().manual_seed(500 + k)
        Z = schedules(kind, N, P, d, g)
        A = PF.exact_attr(Pq, Z, dx)
        Ar = PF.exact_attr(Pq, 1.0 - Z.flip(0), dx)
        ar = levy_area(Z)
        pred = dx * torch.einsum("ij,pij,j->pi", H, ar, dx)
        mean_area = ar.mean(0)
        se_area = ar.std(0) / math.sqrt(P)
        det = float(se_area.max()) == 0.0                                     # họ tất định (power): không có nhiễu MC
        z = (mean_area / se_area.clamp_min(1e-300)).abs()
        z.fill_diagonal_(0)
        rev_inv = kind != "power"
        rows.append({"họ": kind, "phụ thuộc giữa feature": kind == "group", "bất biến khi đảo": rev_inv,
                     "max |A − IG − Δ(H∘𝔄)Δ| (từng đường)": float((A - ig - pred).abs().max()),
                     "max |A(Z) + A(RZ) − 2 IG| (từng đường)": float((A + Ar - 2 * ig).abs().max()),
                     "max |E 𝔄_ij|": float(mean_area.abs().max()), "max |z| của E 𝔄 (≈ 0 nếu bất biến)": "tất định" if det else float(z.max()),
                     "‖E A − IG‖": float((A.mean(0) - ig).norm())})
    return rows


def t4a(N: int = 32, P: int = 400000, seed: int = 0) -> list:
    rows = []
    for k, (kind, h, NN, PP) in enumerate((("dp", 0.5, N, P), ("cmix", 0.5, N, P), ("ln", 0.5, N, P), ("smooth", 0.2, N, P),
                                           ("dp", 0.5, 4 * N, P // 4), ("cmix", 0.5, 4 * N, P // 4))):
        g = torch.Generator().manual_seed(900 + k)
        Z = schedules(kind, NN, PP, 2, g)
        P_ = PP
        I = ((Z[1:, :, 1] + Z[:-1, :, 1]) / 2 * (Z[1:, :, 0] - Z[:-1, :, 0])).sum(0)       # ∫ Z_1 dZ_0
        v = float(I.var())
        se = v * math.sqrt(2.0 / (P_ - 1))
        theo = {"dp": h / 6 + h * h / 12, "cmix": h / 6 + h * h / 12, "smooth": 1.0 / 18}.get(kind)
        rows.append({"họ": kind, "N": NN, "h": h, "Var ∫Z₁dZ₀ (MC)": v, "SE": se, "lý thuyết (liên tục)": theo if theo is not None else math.nan,
                     "lệch tương đối": (v - theo) / theo if theo else math.nan, "D (nội suy)": PF.d_interp(Z),
                     "D liên tục": {"dp": h / 6, "cmix": h / 6, "smooth": 1.0 / 30, "ln": h / 6}[kind]})
    return rows


def t4b(N: int = 16, P: int = 200000, seed: int = 0) -> list:
    """F bậc 3: dp α = 1 và cmix λ = ½ — cùng μ (Mệnh đề), phương sai attribution khác."""
    Pc = PF.toy_poly(3, 100 + seed)
    dx = torch.ones(3, dtype=F64)
    out = []
    for k, (kind, p) in enumerate((("dp", {"alpha": 1.0}), ("cmix", {"lam": 0.5}))):
        g = torch.Generator().manual_seed(1300 + k)
        Z = PF.schedule(kind, N, (P, 3), g, p)
        A = PF.exact_attr(Pc, Z, dx)
        v = A.var(0)
        out.append({"họ": kind, "E A": A.mean(0), "SE E A": A.std(0) / math.sqrt(P), "Var A": v, "SE Var": v * math.sqrt(2.0 / (P - 1))})
    a, b = out
    zm = ((a["E A"] - b["E A"]) / (a["SE E A"] ** 2 + b["SE E A"] ** 2).sqrt()).abs().max()
    zv = ((a["Var A"] - b["Var A"]) / (a["SE Var"] ** 2 + b["SE Var"] ** 2).sqrt()).abs().max()
    return [{"so sánh": "cmix λ=½ − dp α=1, F bậc 3", "max |z| E A (≈ 0: cùng μ)": float(zm), "max |z| Var A (lớn: khác phương sai)": float(zv),
             "Var A dp": ", ".join(f"{x:.4f}" for x in a["Var A"].tolist()), "Var A cmix": ", ".join(f"{x:.4f}" for x in b["Var A"].tolist())}]


def run(cfg: dict) -> str:
    import pandas as pd
    from .report import md
    dbg = bool(cfg.get("_debug"))
    s = 10 if dbg else 1
    out = os.path.join(C.out_dir(cfg, "pilot"), "infid", "pathfam")
    os.makedirs(out, exist_ok=True)
    L = ["# PATH_FAMILIES Phase 2 — kiểm số các định lý ở mức phân phối đường (CPU)\n",
         "Tích phân đường chính xác trên đường nội suy tuyến tính (Gauss–Legendre), không mô hình. Định lý và chứng minh: "
         "docs/PATH_FAMILIES_THEORY.md.\n",
         "## T2 + T5 — F bậc 2: đồng nhất thức diện tích Lévy (từng đường), bất biến khi đảo ⇒ μ = IG, cặp rev chính xác\n",
         "`group`: các feature trong nhóm dùng chung một lịch trình dp (phụ thuộc, vẫn bất biến khi đảo). `power`: Z_i = t^{a_i} "
         "(a = 0,5; 2; 1; 3) — không bất biến khi đảo ⇒ E 𝔄 ≠ 0 ⇒ μ ≠ IG. Hai cột đầu phải ở mức sai số máy (~1e-13).\n",
         md(pd.DataFrame(t2_t5(P=100000 // s))),
         "## T4a — phương sai bậc 2 chỉ phụ thuộc hàm hiệp phương sai c(s, t)\n",
         "Var ∫Z₁dZ₀ (hai feature độc lập, N = 32). Lý thuyết liên tục: dp và cmix (cùng c = h(min − st)) h/6 + h²/12; smooth δ²/18. "
         "Lệch nhỏ còn lại là do rời rạc hoá N = 32 (c của đường nội suy ≠ c liên tục) và nhiễu MC. ln khớp D nhưng c khác ⇒ giá trị khác.\n",
         md(pd.DataFrame(t4a(P=400000 // s))),
         "## T4b — F bậc 3: cùng μ, khác phương sai\n",
         md(pd.DataFrame(t4b(P=200000 // s)))]
    rp = os.path.join(out, "PATH_FAMILIES_THEORY_CHECKS.md")
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[pathfam-theory] {rp}")
    return rp

