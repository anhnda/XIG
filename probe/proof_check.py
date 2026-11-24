"""
Kiểm toán chứng minh Phase 2 (CPU): symbolic (sympy) + Monte Carlo cho từng phát biểu của docs/PATH_FAMILIES_THEORY.md.

  S1  Hệ quả 3a / 3b: 4 mẫu chỉ số bậc 3 và 7 mẫu bậc 4 — tính TRỰC TIẾP E∫Π Z dZ_i (khai triển, kỳ vọng theo biến ngẫu nhiên, tích phân)
      cho họ "warp lệch" KHÔNG bất biến khi đảo và có κ₃ ≠ 0, so với công thức theo v, κ₃, D (hiệu phải = 0 chính xác).
  S2  Bất biến khi đảo ⇒ ∫ t v = D/2 và ∫ κ₃ = 0 (họ warp đối xứng).
  S3  Bậc 5: E∫Z_j⁴dZ_i = ∫ m₄ với m₄ = t⁴ + 6t²v + 4tκ₃ + 3v² + κ₄ (phụ thuộc κ₄, ∫v²).
  S4  Định lý 4: mọi kiểu chồng chỉ số của Cov(S_{e_j,i}, S_{e_k,l}) so với ±Γ, 0, 2Γ + Λ — symbolic trên họ hạng 2 (Λ ≠ 0).
  M1  Định lý 4 trên dp lưới (c có khối lượng trên đường chéo) — Monte Carlo, Γ̂ = Var ∫Z dt (đồng nhất thức ∬c = Var ∫Z).
  M2  Mệnh đề Shapley: lịch trình nhảy độc lập (λ = 1, N = 4096) ⇒ E A = giá trị Shapley của trò chơi v(S) = F(x′ + Δ⊙1_S).
Lệnh: bash scripts/run.sh pathfam-proofcheck → runs_probe/pilot/infid/pathfam/PATH_FAMILIES_PROOF_CHECKS.md
"""

from __future__ import annotations

import itertools
import math
import os

import torch

from . import config as C
from . import path_families as PF

F64 = torch.float64


# ─────────────────────────────────────────────────────────────────────────────
# symbolic
# ─────────────────────────────────────────────────────────────────────────────

def _warp_family(asym: bool):
    """Z = t + δ s φ(t); asym: φ = t(1 − t)², s ∈ {2 (1/3), −1 (2/3)} (E s = 0, E s³ = 2 ≠ 0); đối xứng: φ = t(1 − t), s = ±1."""
    import sympy as sp
    t = sp.symbols("t")
    if asym:
        phi = t * (1 - t) ** 2
        mom = {0: 1, 1: 0, 2: 2, 3: 2, 4: 6, 5: 10, 6: 22}                # E s^k với s ∈ {2 (1/3), −1 (2/3)}
        delta = sp.Rational(1, 2)
    else:
        phi = t * (1 - t)
        mom = {k: (1 if k % 2 == 0 else 0) for k in range(8)}
        delta = sp.Rational(1, 2)
    return t, phi, mom, delta


def _E(expr, svars, mom):
    """Kỳ vọng của đa thức theo các biến s độc lập cùng phân phối (moment cho trong mom)."""
    import sympy as sp
    p = sp.Poly(sp.expand(expr), *svars)
    out = 0
    for monom, coef in p.terms():
        term = coef
        for e in monom:
            term *= mom[e]
        out += term
    return sp.expand(out)


def pattern_check(asym: bool = True) -> list:
    import sympy as sp
    t, phi, mom, dl = _warp_family(asym)
    s = sp.symbols("s0:5")                                                   # s0 = feature i
    Z = [t + dl * si * phi for si in s]
    dZi = sp.diff(Z[0], t)
    v = sp.expand(dl ** 2 * mom[2] * phi ** 2)
    k3 = sp.expand(dl ** 3 * mom[3] * phi ** 3)
    D = sp.integrate(v, (t, 0, 1))
    Itv = sp.integrate(t * v, (t, 0, 1))
    Ik3 = sp.integrate(k3, (t, 0, 1))
    # (đơn thức theo chỉ số feature trong ∂_iF, công thức)
    pats = [("bậc 3: j ≠ k, cả hai ≠ i", (1, 2), sp.Rational(1, 3)),
            ("bậc 3: j = k ≠ i", (1, 1), sp.Rational(1, 3) + D),
            ("bậc 3: j = i ≠ k", (0, 1), sp.Rational(1, 3) - D / 2),
            ("bậc 3: j = k = i", (0, 0), sp.Rational(1, 3)),
            ("bậc 4: j, k, l khác nhau, ≠ i", (1, 2, 3), sp.Rational(1, 4)),
            ("bậc 4: j = k ≠ l, ≠ i", (1, 1, 2), sp.Rational(1, 4) + Itv),
            ("bậc 4: j = k = l ≠ i", (1, 1, 1), sp.Rational(1, 4) + 3 * Itv + Ik3),
            ("bậc 4: một = i, hai còn lại khác nhau", (0, 1, 2), sp.Rational(1, 4) - Itv),
            ("bậc 4: một = i, hai còn lại bằng nhau", (0, 1, 1), sp.Rational(1, 4)),
            ("bậc 4: hai = i", (0, 0, 1), sp.Rational(1, 4) - Itv - Ik3 / 3),
            ("bậc 4: ba = i", (0, 0, 0), sp.Rational(1, 4))]
    rows = []
    for name, idx, formula in pats:
        integrand = sp.Integer(1)
        for j in idx:
            integrand *= Z[j]
        lhs = sp.integrate(_E(integrand * dZi, s, mom), (t, 0, 1))
        rows.append({"mẫu": name, "trực tiếp": str(sp.nsimplify(lhs)), "công thức": str(sp.nsimplify(formula)),
                     "hiệu (phải = 0)": str(sp.simplify(lhs - formula))})
    rows.append({"mẫu": "tham số họ", "trực tiếp": f"D = {sp.nsimplify(D)}, ∫tv = {sp.nsimplify(Itv)}, ∫κ₃ = {sp.nsimplify(Ik3)}",
                 "công thức": "D/2 = " + str(sp.nsimplify(D / 2)), "hiệu (phải = 0)": "—"})
    return rows


def symmetry_check() -> dict:
    import sympy as sp
    t, phi, mom, dl = _warp_family(False)
    v = dl ** 2 * mom[2] * phi ** 2
    k3 = dl ** 3 * mom[3] * phi ** 3
    D = sp.integrate(v, (t, 0, 1))
    return {"∫ t v − D/2": str(sp.simplify(sp.integrate(t * v, (t, 0, 1)) - D / 2)), "∫ κ₃": str(sp.simplify(sp.integrate(k3, (t, 0, 1))))}


def deg5_check() -> dict:
    import sympy as sp
    t, phi, mom, dl = _warp_family(True)
    s = sp.symbols("s0:2")
    Zj = t + dl * s[1] * phi
    dZi = sp.diff(t + dl * s[0] * phi, t)
    lhs = sp.integrate(_E(Zj ** 4 * dZi, s, mom), (t, 0, 1))
    v = dl ** 2 * mom[2] * phi ** 2
    k3 = dl ** 3 * mom[3] * phi ** 3
    k4 = dl ** 4 * (mom[4] - 3 * mom[2] ** 2) * phi ** 4
    rhs = sp.integrate(t ** 4 + 6 * t ** 2 * v + 4 * t * k3 + 3 * v ** 2 + k4, (t, 0, 1))
    return {"E∫Z_j⁴dZ_i trực tiếp": str(sp.nsimplify(lhs)), "∫(t⁴ + 6t²v + 4tκ₃ + 3v² + κ₄)": str(sp.nsimplify(rhs)),
            "hiệu": str(sp.simplify(lhs - rhs)), "đóng góp κ₄": str(sp.nsimplify(sp.integrate(k4, (t, 0, 1)))),
            "đóng góp 3∫v²": str(sp.nsimplify(sp.integrate(3 * v ** 2, (t, 0, 1))))}


def cov_check() -> list:
    """Định lý 4 symbolic: họ hạng 2 Z = t + δ₁ s φ₁ + δ₂ r φ₂ (s, r độc lập ±1), E Z = t, Λ ≠ 0."""
    import sympy as sp
    t, u, w = sp.symbols("t u w")
    p1, p2 = t * (1 - t), t ** 2 * (1 - t)
    d1, d2 = sp.Rational(1, 4), sp.Rational(1, 3)
    n = 5
    s = sp.symbols(f"s0:{n}")
    r = sp.symbols(f"r0:{n}")
    mom = {k: (1 if k % 2 == 0 else 0) for k in range(10)}
    Z = [t + d1 * s[a] * p1 + d2 * r[a] * p2 for a in range(n)]

    def S(j, i):
        return sp.integrate(sp.expand(Z[j] * sp.diff(Z[i], t)), (t, 0, 1))
    allv = list(s) + list(r)

    def E(x):
        return _E(x, allv, mom)

    def cov(a, b):
        return sp.simplify(E(a * b) - E(a) * E(b))
    c = d1 ** 2 * p1.subs(t, u) * p1.subs(t, w) + d2 ** 2 * p2.subs(t, u) * p2.subs(t, w)
    G = sp.integrate(sp.integrate(c, (u, 0, 1)), (w, 0, 1))
    Lam = sp.integrate(sp.integrate(c * sp.diff(c, u, w), (u, 0, 1)), (w, 0, 1))
    cases = [("Var S_{e_j,i}", (1, 0), (1, 0), 2 * G + Lam), ("(j,i) & (k,i): chung integrator", (1, 0), (2, 0), G),
             ("(j,i) & (j,l): chung integrand", (1, 0), (1, 2), G), ("(j,i) & (i,l): integrator = integrand kia", (1, 0), (0, 2), -G),
             ("(j,i) & (l,j): integrand = integrator kia", (1, 0), (2, 1), -G), ("(j,i) & (i,j)", (1, 0), (0, 1), -(2 * G + Lam)),
             ("(j,i) & (k,l): rời nhau", (1, 0), (2, 3), sp.Integer(0))]
    rows = []
    for name, a, b, f in cases:
        val = cov(S(*a), S(*b))
        rows.append({"kiểu": name, "trực tiếp": str(sp.nsimplify(val)), "định lý": str(sp.nsimplify(f)), "hiệu": str(sp.simplify(val - f))})
    rows.append({"kiểu": "Γ, Λ của họ", "trực tiếp": f"Γ = {sp.nsimplify(G)}", "định lý": f"Λ = {sp.nsimplify(Lam)}", "hiệu": "—"})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

def dp_cov_mc(N: int = 64, P: int = 200000, h: float = 0.5, seed: int = 0) -> list:
    g = torch.Generator().manual_seed(seed)
    Z = PF.schedule("dp", N, (P, 4), g, {"alpha": 1.0 / h - 1.0})
    Zm = (Z[1:] + Z[:-1]) / 2
    dZ = Z[1:] - Z[:-1]
    I = torch.einsum("npj,npi->pij", Zm, dZ)                                  # I[p, i, j] = ∫Z_j dZ_i = S_{e_j, i}
    intZ = (Zm / N).sum(0)                                                   # ∫ Z dt (chính xác, tuyến tính từng đoạn)
    Gam = float(intZ.var(0).mean())                                          # Γ̂ = Var ∫Z dt = ∬ c

    def cv(a, b):
        x, y = I[:, a[1], a[0]], I[:, b[1], b[0]]
        return float(((x - x.mean()) * (y - y.mean())).mean())
    se = 1.0 / math.sqrt(P)
    rows = []
    var = cv((1, 0), (1, 0))
    for name, a, b, f in (("Var S_{e_j,i}", (1, 0), (1, 0), None), ("(j,i) & (k,i)", (1, 0), (2, 0), Gam), ("(j,i) & (j,l)", (1, 0), (1, 2), Gam),
                          ("(j,i) & (i,l)", (1, 0), (0, 2), -Gam), ("(j,i) & (l,j)", (1, 0), (2, 1), -Gam), ("(j,i) & (k,l)", (1, 0), (2, 3), 0.0)):
        val = cv(a, b)
        rows.append({"kiểu": name, "MC": val, "định lý (theo Γ̂)": f if f is not None else 2 * Gam + (var - 2 * Gam),
                     "lệch / độ lớn": (val - f) / max(abs(var), 1e-12) if f is not None else 0.0, "SE thô ≈": se * var * 2})
    rows.append({"kiểu": f"Γ̂ = Var∫Z (lý thuyết liên tục h/12 = {h / 12:.5f})", "MC": Gam, "định lý (theo Γ̂)": h / 12,
                 "lệch / độ lớn": (Gam - h / 12) / (h / 12), "SE thô ≈": float("nan")})
    rows.append({"kiểu": f"Λ̂ = Var − 2Γ̂ (lý thuyết liên tục h²/12 = {h * h / 12:.5f})", "MC": var - 2 * Gam, "định lý (theo Γ̂)": h * h / 12,
                 "lệch / độ lớn": (var - 2 * Gam - h * h / 12) / (h * h / 12), "SE thô ≈": float("nan")})
    return rows


def compress_jumps(k: torch.Tensor, d: int) -> torch.Tensor:
    """Lịch trình nhảy (feature j nhảy trọn trong đoạn k_j) với các đoạn đứng yên bị bỏ đi — tích phân đường không đổi khi bỏ đoạn đứng
    yên. Feature nhảy CÙNG đoạn đi thẳng cùng nhau trong một đoạn nén. Trả Z (d + 1, P, d)."""
    srt, _ = k.sort(1)
    new = torch.ones_like(srt)
    new[:, 1:] = (srt[:, 1:] != srt[:, :-1]).long()
    rank_sorted = new.cumsum(1) - 1                                          # hạng đặc của các giá trị đã sắp
    # hạng đặc của từng feature: số giá trị phân biệt nhỏ hơn k_j
    r = (k.unsqueeze(2) > srt.unsqueeze(1)).long()                           # (P, d, d): k_j > srt_m
    r = (r * new.unsqueeze(1)).sum(2)                                        # chỉ đếm vị trí đầu của mỗi giá trị phân biệt
    del rank_sorted
    g = torch.arange(d + 1).view(-1, 1, 1)
    return (r.unsqueeze(0) < g).double()                                     # Z_j tại nút g = 1{r_j < g}


def compress_check(d: int = 4, N: int = 16, P: int = 500) -> float:
    """Nén đoạn đứng yên không đổi attribution: so với lịch trình đầy đủ N đoạn."""
    g = torch.Generator().manual_seed(3)
    k = torch.randint(0, N, (P, d), generator=g)
    Zf = torch.stack([(k < n).double() for n in range(N + 1)])
    Pq = PF.toy_poly(d, 700)
    dx = torch.ones(d, dtype=F64)
    return float((PF.exact_attr(Pq, Zf, dx) - PF.exact_attr(Pq, compress_jumps(k, d), dx)).abs().max())


def shapley_mc(N: int = 4096, P: int = 100000, d: int = 4, seed: int = 0, chunk: int = 1000) -> list:
    """λ = 1 (mỗi feature nhảy trọn trong một đoạn ngẫu nhiên, độc lập), N rất lớn ⇒ đụng độ ≈ d²/(2N). E A so với Shapley chính xác."""
    Pq = PF.toy_poly(d, 700 + seed)
    dx = torch.ones(d, dtype=F64)

    def Fv(x):
        X = x.unsqueeze(0)
        q = (X * Pq["b"]).sum(-1) + 0.5 * torch.einsum("pi,ij,pj->p", X, Pq["Q"], X) + torch.einsum("ijk,pi,pj,pk->p", Pq["T"], X, X, X) / 6
        return float(q[0])
    phi = torch.zeros(d, dtype=F64)
    perms = list(itertools.permutations(range(d)))
    for pm in perms:
        cur = torch.zeros(d, dtype=F64)
        for i in pm:
            nxt = cur.clone()
            nxt[i] = 1.0
            phi[i] += Fv(nxt) - Fv(cur)
            cur = nxt
    phi /= len(perms)
    g = torch.Generator().manual_seed(800 + seed)
    s1 = torch.zeros(d, dtype=F64)
    s2 = torch.zeros(d, dtype=F64)
    for a in range(0, P, chunk):
        k = torch.randint(0, N, (min(chunk, P - a), d), generator=g)       # đoạn nhảy của từng feature (= cmix λ = 1)
        A = PF.exact_attr(Pq, compress_jumps(k, d), dx)
        s1 += A.sum(0)
        s2 += (A * A).sum(0)
    m = s1 / P
    se = ((s2 / P - m * m).clamp_min(0) / P).sqrt()
    return [{"feature": i, "E A (MC)": float(m[i]), "Shapley chính xác": float(phi[i]), "z": float((m[i] - phi[i]) / se[i]),
             "đụng độ kỳ vọng / đường": d * (d - 1) / (2 * N)} for i in range(d)]


def run(cfg: dict) -> str:
    import pandas as pd
    from .report import md
    dbg = bool(cfg.get("_debug"))
    out = os.path.join(C.out_dir(cfg, "pilot"), "infid", "pathfam")
    os.makedirs(out, exist_ok=True)
    L = ["# PATH_FAMILIES — kiểm toán chứng minh Phase 2 (symbolic + Monte Carlo, CPU)\n",
         "Tài liệu: docs/PATH_FAMILIES_THEORY.md, docs/PATH_FAMILIES_THEORY_AUDIT.md.\n",
         "## S1 — Hệ quả 3a / 3b: từng mẫu chỉ số, tính trực tiếp trên họ warp LỆCH (không bất biến khi đảo, κ₃ ≠ 0)\n",
         "Z_j = t + ½ s_j t(1 − t)², s ∈ {2 (xác suất 1/3), −1 (2/3)}: E Z = t, v(t) không đối xứng, κ₃ ≠ 0 ⇒ mọi số hạng ∫tv, ∫κ₃ của công "
         "thức đều khác 0, nên phép kiểm không suy biến.\n", md(pd.DataFrame(pattern_check(True))),
         "## S2 — bất biến khi đảo ⇒ ∫tv = D/2, ∫κ₃ = 0 (họ warp đối xứng φ = t(1 − t), s = ±1)\n", md(pd.DataFrame([symmetry_check()])),
         "## S3 — bậc 5 phụ thuộc κ₄ và ∫v² (họ warp lệch)\n", md(pd.DataFrame([deg5_check()])),
         "## S4 — Định lý 4: mọi kiểu chồng chỉ số (symbolic, họ hạng 2, Λ ≠ 0)\n", md(pd.DataFrame(cov_check())),
         "## M1 — Định lý 4 trên dp lưới (h = ½, N = 64, Monte Carlo)\n",
         "Γ̂ = Var ∫Z dt (đồng nhất thức ∬c = Var ∫Z dt, đúng cho mọi luật). Các hiệp phương sai phải bằng ±Γ̂ / 0 trong nhiễu MC; Γ̂, Λ̂ lệch "
         "nhẹ khỏi giá trị liên tục do rời rạc hoá.\n", md(pd.DataFrame(dp_cov_mc(P=20000 if dbg else 200000))),
         "## M2 — Mệnh đề Shapley: λ = 1, N = 4096, F bậc 3 ngẫu nhiên, d = 4\n",
         f"Đường được nén (bỏ đoạn đứng yên; sai khác so với lịch trình đầy đủ trên N = 16: {compress_check():.1e}). Đụng độ (hai feature nhảy "
         "cùng đoạn ⇒ đi thẳng cùng nhau) gây lệch O(d²/N), nhỏ hơn nhiễu MC ở đây.\n",
         md(pd.DataFrame(shapley_mc(P=10000 if dbg else 100000)))]
    rp = os.path.join(out, "PATH_FAMILIES_PROOF_CHECKS.md")
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[pathfam-proofcheck] {rp}")
    return rp
