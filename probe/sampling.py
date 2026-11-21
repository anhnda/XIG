"""
P4 — bộ lấy mẫu đường cho Real-CNN Stochastic Path Sampling Benchmark. Chỉ sinh lịch trình Z ∈ [0, 1] và các chẩn đoán
không cần mô hình (trừ LIME-grid và insertion/deletion, nhận hàm mô hình làm tham số). Không đổi gì ở paths.py.

Quy ước (giống SPI của P0–P3): đường = đường gấp khúc qua N+1 nút x_k = x′ + Z_k ⊙ Δx, Z_0 = 0, Z_N = 1. Tích phân chia
đúng theo từng đoạn (cầu phương phân tầng M điểm/đoạn, `mechanism.aw_stream`; M = 1 ≡ `canon_rand`).

Họ lịch trình
  spi     S0: DP độc lập theo phần tử — đúng `paths.spi_increments(α, N)` + `cum_nodes`, không đổi tham số.
  rev     antithetic Z^R(t) = 1 − Z(1 − t); trên nút Z^R_k = 1 − Z_{N−k} (≡ đảo thứ tự số gia). Dirichlet(α/N, …, α/N)
          đối xứng ⇒ số gia đảo cùng phân phối ⇒ MỖI đường đảo ~ đúng phân phối của đường gốc (q_SPI hoặc q_patch).
  patch   mỗi patch g của lưới G×G một lịch trình DP(α, N) (cùng α, N với S0) dùng chung cho mọi phần tử (pixel × kênh)
          của patch. "Temporal shift" của patch = vị trí số gia lớn nhất (anchor) a = ⌊N·φ⌋, φ ∈ [0, 1) do thiết kế gán.
          Xoay vòng số gia về anchor BẢO TOÀN phân phối DP: dãy số gia trao đổi được nên bất biến dưới phép xoay vòng ⇒
          vị trí argmax đều trên {0..N−1} và độc lập với "hình dạng" (dãy số gia nhìn từ argmax). Đặt lại argmax tại
          a ~ U{0..N−1} độc lập với hình dạng cho lại đúng phân phối DP.
  hier    (1 − η)·Z_patch[g(i)] + η·Z_spi[i]; Z_spi là lịch trình S0 cùng chỉ số đường (common random numbers).

Thiết kế gán pha φ_{k,g} (k: đường trong ensemble, g: patch), K_max = 2^m:
  indep   (S6) φ iid U(0, 1), rút mới cho mỗi đường                       — đối chứng độc lập, cùng operator
  fixed   (S5) φ_{k,g} = π_g cho cả ensemble                               — đối chứng gán cố định
  global  (S2) φ_{k,g} = frac(π_g + (z_g·b(k) mod K_max + ε_{k,g}) / K_max); z_g lẻ ngẫu nhiên; b(k) = đảo bit của k
  nbr     (S4) như global, z_g = 1 + 2·parity(i + j) + 4·r_g ⇒ hai patch kề cạnh luôn có z_g − z_h ≡ 2 (mod 4)
  lime    (S7) hạng của q_g + τ·ξ_{k,g} (q_g = hạng hệ số LIME; cao → kích hoạt muộn), φ = (hạng + U)/P
Với π_g iid U(0, 1) độc lập với phần còn lại, pha của MỘT đường là iid U(0, 1) theo patch ở indep / fixed / global / nbr
⇒ bốn thiết kế có cùng phân phối trên từng đường (cùng population operator q_patch); chỉ khác ghép nối giữa các đường.
global / nbr: với mọi tiền tố K = 2^j ≤ K_max, K pha của một patch rơi đúng một lần vào mỗi cung dài 1/K (tính từ π_g)
— Latin hypercube giữa các đường, lồng nhau theo K. nbr: với cặp kề cạnh, độ lệch pha (z_g − z_h)·b(k)/K_max nhận K/2
giá trị cách đều (mỗi giá trị hai lần) ⇒ thứ tự và độ chồng thời gian giữa hai patch kề được trải đều qua các đường;
global: ngẫu nhiên, có thể bằng hằng số cho một số cặp. lime: phân phối mỗi đường phụ thuộc mô hình ⇒ operator KHÁC.
"""

from __future__ import annotations

import math

import torch

from .paths import cum_nodes, spi_increments

F64 = torch.float64
DESIGNS = ("indep", "fixed", "global", "nbr", "lime")


# ─────────────────────────────────────────────────────────────────────────────
# lưới patch
# ─────────────────────────────────────────────────────────────────────────────

def is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def bitrev(k: int, bits: int) -> int:
    r = 0
    for _ in range(bits):
        r = (r << 1) | (k & 1)
        k >>= 1
    return r


def vdc(Kmax: int) -> torch.Tensor:
    """b(k) = đảo bit của k (m = log2 K_max bit), k = 0..K_max−1. Tiền tố K: {b(k)} = {j·K_max/K : j < K}."""
    if not is_pow2(Kmax):
        raise ValueError(f"K_max phải là luỹ thừa của 2: {Kmax}")
    m = Kmax.bit_length() - 1
    return torch.tensor([bitrev(k, m) for k in range(Kmax)], dtype=torch.long)


def grid_pixel_labels(H: int, W: int, g: int) -> torch.Tensor:
    """Nhãn patch (H·W,) theo hàng (i·g + j), cùng cách chia với `e9.patch_masks` (patch cuối nhận phần dư)."""
    ph, pw = H // g, W // g
    i = torch.clamp(torch.arange(H) // ph, max=g - 1)
    j = torch.clamp(torch.arange(W) // pw, max=g - 1)
    return (i.view(H, 1) * g + j.view(1, W)).flatten()


def grid_elem_labels(shape, g: int) -> torch.Tensor:
    """Nhãn patch cho mọi phần tử (C·H·W,), 3 kênh của một pixel cùng patch."""
    Cc, H, W = shape
    return grid_pixel_labels(H, W, g).repeat(Cc)


def grid_coords(g: int) -> torch.Tensor:
    i, j = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    return torch.stack([i.flatten(), j.flatten()], 1)


def cheb(g: int) -> torch.Tensor:
    c = grid_coords(g)
    return (c.view(-1, 1, 2) - c.view(1, -1, 2)).abs().max(-1).values


def pair_index(g: int, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Cặp (p < q). nbr4: kề cạnh; far: khoảng cách Chebyshev ≥ ⌈g/2⌉."""
    c = grid_coords(g)
    d1 = (c.view(-1, 1, 2) - c.view(1, -1, 2)).abs().sum(-1)
    dc = cheb(g)
    if kind == "nbr4":
        m = d1 == 1
    elif kind == "far":
        m = dc >= max(2, (g + 1) // 2)
    else:
        raise ValueError(kind)
    m = torch.triu(m, diagonal=1)
    p, q = torch.nonzero(m, as_tuple=True)
    return p, q


# ─────────────────────────────────────────────────────────────────────────────
# thiết kế gán pha
# ─────────────────────────────────────────────────────────────────────────────

def common_phase_noise(P: int, Kmax: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """π (P,) và ε (K_max, P), dùng chung giữa fixed / global / nbr (common random numbers)."""
    return torch.rand(P, generator=gen, dtype=F64), torch.rand(Kmax, P, generator=gen, dtype=F64)


def phase_design(kind: str, P: int, Kmax: int, gen: torch.Generator, g: int | None = None, common=None,
                 lime_rank: torch.Tensor | None = None, tau: float = 0.25) -> tuple[torch.Tensor, dict]:
    """-> φ (K_max, P) float64 CPU, thông tin thiết kế. `gen`: generator CPU riêng của thiết kế."""
    if not is_pow2(Kmax):
        raise ValueError(f"K_max phải là luỹ thừa của 2: {Kmax}")
    pi, eps = common if common is not None else common_phase_noise(P, Kmax, gen)
    if kind == "indep":
        return torch.rand(Kmax, P, generator=gen, dtype=F64), {}
    if kind == "fixed":
        return pi.view(1, P).expand(Kmax, P).clone(), {}
    if kind in ("global", "nbr"):
        if kind == "global":
            z = 2 * torch.randint(0, max(1, Kmax // 2), (P,), generator=gen) + 1
        else:
            if g is None or g * g != P:
                raise ValueError("nbr cần lưới g×g với g² = P")
            par = grid_coords(g).sum(1) % 2
            if float(torch.rand(1, generator=gen)) < 0.5:              # ngẫu nhiên hoá màu bàn cờ
                par = 1 - par
            z = 1 + 2 * par + 4 * torch.randint(0, max(1, Kmax // 4), (P,), generator=gen)
        b = vdc(Kmax)
        w = (torch.remainder(b.view(Kmax, 1) * z.view(1, P), Kmax).to(F64) + eps) / Kmax
        return torch.remainder(pi.view(1, P) + w, 1.0), {"z": z}
    if kind == "lime":
        if lime_rank is None:
            raise ValueError("lime cần lime_rank")
        xi = torch.randn(Kmax, P, generator=gen, dtype=F64)
        s = lime_rank.to(F64).view(1, P) + float(tau) * xi
        rk = s.argsort(1).argsort(1).to(F64)                          # 0 = sớm nhất, P−1 = muộn nhất
        U = torch.rand(Kmax, P, generator=gen, dtype=F64)
        return (rk + U) / P, {}
    raise ValueError(f"thiết kế không biết: {kind}")


def arc_coverage(phi: torch.Tensor, pi: torch.Tensor, K: int) -> torch.Tensor:
    """Số cung (độ dài 1/K, tính từ π_g) mà K pha đầu của mỗi patch rơi vào. (P,) — bằng K nếu phủ đủ."""
    rel = torch.remainder(phi[:K] - pi.view(1, -1), 1.0)
    s = torch.clamp((rel * K).floor().long(), max=K - 1)
    oh = torch.zeros(K, phi.shape[1], dtype=torch.long)
    oh.scatter_(0, s, 1)
    return (oh > 0).sum(0)


# ─────────────────────────────────────────────────────────────────────────────
# lịch trình
# ─────────────────────────────────────────────────────────────────────────────

def anchor_rotate(delta: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """delta (N, P) số gia; phase (P,) ∈ [0, 1). Xoay vòng từng cột để số gia lớn nhất nằm ở ô a = ⌊N·φ⌋."""
    N, P = delta.shape
    a0 = delta.argmax(0)
    a = torch.clamp((phase.to(delta.device).to(F64) * N).floor().long(), 0, N - 1)
    s = torch.remainder(a - a0, N)
    k = torch.arange(N, device=delta.device).view(N, 1)
    return delta.gather(0, torch.remainder(k - s.view(1, P), N))


def spi_schedule(alpha: float, N: int, shape, gen) -> torch.Tensor:
    """S0 — đúng sampler hiện tại. -> Z (N+1, *shape) float64 trên gen.device."""
    return cum_nodes(spi_increments(float(alpha), int(N), tuple(shape), gen))


def patch_schedule(alpha: float, N: int, phase: torch.Tensor, gen) -> torch.Tensor:
    """-> Z (N+1, P) float64: mỗi patch một DP(α, N), anchor tại ⌊N·φ_g⌋. Hình dạng chỉ phụ thuộc `gen`
    (dùng chung giữa các thiết kế ⇒ common random numbers), pha chỉ phụ thuộc thiết kế."""
    P = phase.shape[0]
    d = spi_increments(float(alpha), int(N), (P,), gen)
    return cum_nodes(anchor_rotate(d, phase))


def reverse_nodes(Z: torch.Tensor) -> torch.Tensor:
    """Z^R_k = 1 − Z_{N−k} (đường antithetic). Giữ đơn điệu và hai đầu mút."""
    return 1.0 - Z.flip(0)


def expand_patch(Zp: torch.Tensor, elem_lab: torch.Tensor, shape) -> torch.Tensor:
    """Zp (N+1, P) -> (N+1, *shape) theo nhãn phần tử."""
    return Zp[:, elem_lab.to(Zp.device)].view((Zp.shape[0],) + tuple(shape))


def mix(Zpatch_full: torch.Tensor, Zpix: torch.Tensor | None, eta: float) -> torch.Tensor:
    if eta <= 0 or Zpix is None:
        return Zpatch_full
    return (1.0 - eta) * Zpatch_full + eta * Zpix.to(Zpatch_full.dtype)


def nodes(x0: torch.Tensor, dx: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    return x0 + Z.to(dx.dtype) * dx


def midpoint_index(Z: torch.Tensor) -> torch.Tensor:
    """Chỉ số nút đầu tiên có Z ≥ ½ (theo phần tử / theo cột)."""
    return (Z >= 0.5).to(torch.uint8).argmax(0)


def patch_mean_schedule(Z: torch.Tensor, elem_lab: torch.Tensor, P: int) -> torch.Tensor:
    """Z (N+1, C,H,W) -> (N+1, P): lịch trình trung bình của các phần tử trong patch."""
    n1 = Z.shape[0]
    lab = elem_lab.to(Z.device)
    out = torch.zeros(n1, P, dtype=F64, device=Z.device)
    out.index_add_(1, lab, Z.reshape(n1, -1).double())
    cnt = torch.zeros(P, dtype=F64, device=Z.device).index_add_(0, lab, torch.ones_like(lab, dtype=F64))
    return out / cnt.view(1, P)


def patch_sum(A: torch.Tensor, elem_lab: torch.Tensor, P: int) -> torch.Tensor:
    lab = elem_lab.to(A.device)
    return torch.zeros(P, dtype=F64, device=A.device).index_add_(0, lab, A.reshape(-1).double())


def d_hat_cols(Z: torch.Tensor) -> float:
    """Trung bình theo phần tử/cột của ∫(Z − t)² dt (hình thang trên nút) — cùng định nghĩa với `e10.d_hat`."""
    N = Z.shape[0] - 1
    t = torch.linspace(0, 1, N + 1, device=Z.device, dtype=F64).view((-1,) + (1,) * (Z.dim() - 1))
    e = ((Z.double() - t) ** 2).reshape(N + 1, -1).mean(1)
    return float(torch.trapezoid(e, dx=1.0 / N))


# ─────────────────────────────────────────────────────────────────────────────
# chẩn đoán D2 — coverage của lịch trình
# ─────────────────────────────────────────────────────────────────────────────

def coverage_stats(tau: torch.Tensor, g: int, overlap_tol: float = 0.125) -> dict:
    """tau (K, P): thời điểm kích hoạt (Z ≥ ½) của lịch trình trung bình mỗi patch, trên K đường của ensemble."""
    tau = tau.double().cpu()
    K, P = tau.shape
    out = {"K_paths": K}
    out["tau_spread"] = float(tau.std(0, unbiased=True).mean()) if K > 1 else 0.0
    out["tau_range"] = float((tau.max(0).values - tau.min(0).values).mean())
    for name, nb in (("third", 3), ("kbin", K)):
        s = torch.clamp((tau * nb).floor().long(), 0, nb - 1)
        oh = torch.zeros(nb, P, dtype=torch.long)
        oh.scatter_(0, s, 1)
        out[f"cover_{name}"] = float(((oh > 0).sum(0).double() / min(nb, K)).mean())
    out["deficit_third"] = 1.0 - out["cover_third"]
    out["deficit_kbin"] = 1.0 - out["cover_kbin"]
    for kind in ("nbr4", "far"):
        p, q = pair_index(g, kind)
        if not len(p):
            continue
        a, b = tau[:, p], tau[:, q]
        before = (a < b).double() + 0.5 * (a == b).double()
        pb = before.mean(0)
        out[f"{kind}_order_balance"] = float((1 - 2 * (pb - 0.5).abs()).mean())
        out[f"{kind}_overlap"] = float(((a - b).abs() < overlap_tol).double().mean())
        out[f"{kind}_tie"] = float((a == b).double().mean())
    p, q = pair_index(g, "nbr4")
    cs = []
    for k in range(K):
        a, b = tau[k, p], tau[k, q]
        if float(a.std()) > 0 and float(b.std()) > 0:
            cs.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    out["nbr4_spatial_corr"] = sum(cs) / len(cs) if cs else float("nan")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# chẩn đoán D3 — tác động thứ tự theo khoảng cách (proxy, KHÔNG phải tương tác thật)
# ─────────────────────────────────────────────────────────────────────────────

def _order_effect(Cp: torch.Tensor, tau_g: torch.Tensor, tau_h: torch.Tensor, min_n: int = 2) -> torch.Tensor:
    """I[g, h] = mean(C_g | h kích hoạt trước g) − mean(C_g | h kích hoạt sau g), qua các đường."""
    before = (tau_h.unsqueeze(1) < tau_g.unsqueeze(2)).double()           # (n, g, h)
    after = (tau_h.unsqueeze(1) > tau_g.unsqueeze(2)).double()
    C = Cp.double().unsqueeze(2)
    nb, na = before.sum(0), after.sum(0)
    I = (C * before).sum(0) / nb.clamp_min(1) - (C * after).sum(0) / na.clamp_min(1)
    I[(nb < min_n) | (na < min_n)] = float("nan")
    return I


def residualize_own_time(Cp: torch.Tensor, tau: torch.Tensor, degree: int = 3) -> torch.Tensor:
    """Phần dư của C_g sau hồi quy (bình phương tối thiểu, qua các đường) lên đa thức bậc `degree` của τ_g.
    Bắt buộc: credit của g phụ thuộc thời điểm kích hoạt của CHÍNH g, và "h trước g" kéo theo τ_g lớn hơn — không khử thì
    hiệu ứng thời điểm của g bị đọc nhầm thành tác động của h (và mức nền hoán vị τ_h cũng mang nó)."""
    n, P = tau.shape
    R = torch.empty_like(Cp)
    for j in range(P):
        t = tau[:, j]
        X = torch.stack([t ** p for p in range(degree + 1)], 1)
        beta = torch.linalg.lstsq(X, Cp[:, j:j + 1]).solution
        R[:, j] = Cp[:, j] - (X @ beta)[:, 0]
    return R


def order_effect_by_distance(Cp: torch.Tensor, tau: torch.Tensor, g: int, D: float, n_null: int, gen) -> list:
    """Cp (n, P) credit theo patch của từng đường độc lập, tau (n, P). Trả |I|/|D| trung bình theo khoảng cách Chebyshev,
    I tính trên phần dư của C_g sau khi khử phụ thuộc vào τ_g, kèm mức nền (hoán vị thời điểm của h giữa các đường)."""
    Cp, tau = Cp.double().cpu(), tau.double().cpu()
    n, P = tau.shape
    dist = cheb(g)
    Cp = residualize_own_time(Cp, tau, degree=min(3, max(0, n // 8 - 1)))
    I = _order_effect(Cp, tau, tau)
    nulls = []
    for _ in range(int(n_null)):
        perm = torch.stack([torch.randperm(n, generator=gen) for _ in range(P)], 1)     # (n, P)
        nulls.append(_order_effect(Cp, tau, torch.gather(tau, 0, perm)))
    rows = []
    scale = abs(D) if D else 1.0
    for d in range(1, g):
        m = dist == d
        v = I[m].abs()
        v = v[torch.isfinite(v)]
        nv = torch.cat([x[m].abs() for x in nulls]) if nulls else torch.tensor([])
        nv = nv[torch.isfinite(nv)]
        rows.append({"dist": d, "n_pairs": int(m.sum()), "abs_I_rel": float(v.mean()) / scale if len(v) else float("nan"),
                     "null_abs_I_rel": float(nv.mean()) / scale if len(nv) else float("nan")})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# LIME trên lưới patch (S7, chỉ làm prior thứ tự mềm)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def lime_grid(F_fwd, x: torch.Tensor, x0: torch.Tensor, pix_lab: torch.Tensor, P: int, n: int, gen: torch.Generator,
              kernel_width: float = 0.25, ridge: float = 1.0, batch: int = 64) -> dict:
    """Mặt nạ nhị phân theo patch (1 = giữ x, 0 = baseline x′), y = F (logit lớp đích), kernel exp(−d²/w²) với d = khoảng
    cách cosine tới vector toàn 1 (như LIME ảnh), hồi quy ridge có trọng số (không phạt intercept)."""
    H, W = x.shape[-2:]
    Zb = torch.rand(n, P, generator=gen, dtype=F64) < 0.5
    Zb[0] = True
    lab = pix_lab.to(x.device)
    ys = []
    for i in range(0, n, batch):
        zb = Zb[i:i + batch].to(x.device)
        m = zb[:, lab].view(-1, 1, H, W)
        ys.append(F_fwd(torch.where(m, x, x0)).double().cpu())
        FwdCounter.n += int(zb.shape[0])
    y = torch.cat(ys)
    Zd = Zb.double()
    d = 1.0 - torch.sqrt(Zd.sum(1) / P)
    w = torch.exp(-d * d / (kernel_width ** 2))
    Dm = torch.cat([torch.ones(n, 1, dtype=F64), Zd], 1)
    reg = torch.eye(P + 1, dtype=F64) * float(ridge)
    reg[0, 0] = 0.0
    sol = torch.linalg.solve(Dm.T @ (w.view(-1, 1) * Dm) + reg, Dm.T @ (w * y))
    pred = Dm @ sol
    ybar = float((w * y).sum() / w.sum())
    r2 = 1.0 - float((w * (y - pred) ** 2).sum()) / max(float((w * (y - ybar) ** 2).sum()), 1e-30)
    beta = sol[1:]
    rank = beta.argsort().argsort().double() / max(P - 1, 1)
    return {"beta": beta, "intercept": float(sol[0]), "r2_weighted": r2, "rank": rank, "n_queries": n}


# ─────────────────────────────────────────────────────────────────────────────
# insertion / deletion — cùng thứ tự xếp hạng với `metrics.insdel`, trả riêng từng AUC và đường cong
# ─────────────────────────────────────────────────────────────────────────────

class FwdCounter:
    n = 0          # số ảnh qua mạng chỉ forward (insdel, LIME, F(x), F(x′)) — Counter.n của quad chỉ đếm forward+backward


@torch.no_grad()
def insdel_full(backbone, cls: int, x: torch.Tensor, base: torch.Tensor, A: torch.Tensor, n_steps: int, batch: int,
                signed: bool = False, curves: bool = False) -> dict:
    """signed = False: hạng theo Σ_c |A| (quy ước P0–P3, đúng `metrics.insdel`). signed = True: hạng theo Σ_c A (phụ).
    diff = AUC_ins − AUC_del của xác suất softmax lớp đích (≡ metrics.insdel khi signed = False)."""
    _, Cc, H, W = x.shape
    n = H * W
    m = A.sum(0) if signed else A.abs().sum(0)
    order = torch.argsort(m.flatten().to(x.device), descending=True)
    ranks = torch.empty(n, dtype=torch.long, device=x.device)
    ranks[order] = torch.arange(n, device=x.device)
    counts = torch.linspace(0, n, n_steps + 1, device=x.device).long()
    ip, dp, il, dl = [], [], [], []
    for i in range(0, counts.numel(), batch):
        c = counts[i:i + batch]
        mk = (ranks.view(1, -1) < c.view(-1, 1)).view(-1, 1, H, W).expand(-1, Cc, H, W)
        for imgs, P_, L_ in ((torch.where(mk, x.expand_as(mk), base.expand_as(mk)), ip, il),
                             (torch.where(mk, base.expand_as(mk), x.expand_as(mk)), dp, dl)):
            out = backbone(imgs)
            P_.append(torch.softmax(out, dim=-1)[:, cls].double())
            L_.append(out[:, cls].double())
            FwdCounter.n += int(imgs.shape[0])
    ip, dp, il, dl = (torch.cat(v) for v in (ip, dp, il, dl))
    auc = lambda v: float((v[:-1] + v[1:]).sum() / (2 * n_steps))
    res = {"ins": auc(ip), "del": auc(dp)}
    res["diff"] = res["ins"] - res["del"]
    if curves:
        res["curves"] = {"ins_prob": ip.cpu(), "del_prob": dp.cpu(), "ins_logit": il.cpu(), "del_logit": dl.cpu(),
                         "frac": (counts.double() / n).cpu()}
    return res