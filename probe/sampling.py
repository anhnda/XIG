"""
Lịch trình đường cho proposal infidelity: chỉ sinh Z ∈ [0, 1] (không cần mô hình), cộng InsDel cho metric ngoài.

Đường = gấp khúc qua N+1 nút x_k = x′ + Z_k ⊙ Δx, Z_0 = 0, Z_N = 1 (khớp SPI của paths.py).
  spi     DP độc lập theo phần tử (pixel × kênh) — đúng `paths.spi_increments(α, N)` + `cum_nodes`.
  rev     Z^R(t) = 1 − Z(1 − t); trên nút Z^R_k = 1 − Z_{N−k} (≡ đảo thứ tự số gia). Dirichlet(α/N, …) đối xứng ⇒ đường đảo
          cùng phân phối với đường gốc (C1a).
  patch   mỗi patch của lưới G×G một DP(α, N) dùng chung cho mọi phần tử của patch; số gia lớn nhất đặt ở ⌊N·φ_g⌋. Dãy số gia
          trao đổi được ⇒ xoay vòng về anchor đều giữ phân phối DP.
  pix     như spi nhưng anchor ⌊N·φ_i⌋ theo phần tử ⇒ mỗi đường vẫn ~ q_SPI (dùng cho thiết kế ghép giữ biên).
  hier    (1 − η)·patch + η·spi (cùng chỉ số đường).
Thiết kế gán pha giữa K đường (K luỹ thừa của 2): indep (φ iid), fixed, global (Latin hypercube giữa các đường, lồng nhau
theo K, z_g lẻ ngẫu nhiên), nbr (như global, hai ô / pixel kề cạnh có z_g − z_h ≡ 2 mod 4 ⇒ độ lệch pha trải đều). Với π
iid U(0,1), pha của MỘT đường iid U(0,1) ở mọi thiết kế ⇒ cùng phân phối từng đường; chỉ khác ghép nối giữa các đường.
"""

from __future__ import annotations

import torch

from .paths import cum_nodes, spi_increments

F64 = torch.float64
DESIGNS = ("indep", "fixed", "global", "nbr")


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


def common_phase_noise(P: int, Kmax: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """π (P,) và ε (K_max, P), dùng chung giữa fixed / global / nbr (common random numbers)."""
    return torch.rand(P, generator=gen, dtype=F64), torch.rand(Kmax, P, generator=gen, dtype=F64)


def phase_design(kind: str, P: int, Kmax: int, gen: torch.Generator, g: int | None = None, common=None) -> tuple[torch.Tensor, dict]:
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
    raise ValueError(f"thiết kế không biết: {kind}")


def pixel_phase_design(kind: str, shape, Kmax: int, gen: torch.Generator, common=None) -> tuple[torch.Tensor, dict]:
    """Thí nghiệm A — pha anchor RIÊNG cho từng phần tử (pixel × kênh), cân bằng giữa các đường, KHÔNG ép phần tử trong patch
    dùng chung lịch. φ_{k,i} = frac(π_i + (z_i·b(k) mod K_max + ε_{k,i}) / K_max), π_i iid U(0,1) theo phần tử.
      global  z_i lẻ ngẫu nhiên theo phần tử
      nbr     z_i = 1 + 2·parity(h + w) + 4·r_i: hai pixel kề cạnh luôn có z_i − z_j ≡ 2 (mod 4) ⇒ độ lệch pha giữa chúng
              trải đều qua các đường (cùng cơ chế với S4 nhưng ở cấp pixel; không có lưới patch)
    π_i iid ⇒ pha trong MỘT đường iid U(0,1) theo phần tử ⇒ kết hợp với hình dạng DP mới mỗi đường (anchor_rotate), mỗi đường
    có ĐÚNG phân phối q_SPI của S0. Với mọi tiền tố K = 2^j, mỗi phần tử phủ đúng một lần mỗi cung pha 1/K."""
    if not is_pow2(Kmax):
        raise ValueError(f"K_max phải là luỹ thừa của 2: {Kmax}")
    Cc, H, W = shape
    n = Cc * H * W
    pi, eps = common if common is not None else common_phase_noise(n, Kmax, gen)
    if kind == "global":
        z = 2 * torch.randint(0, max(1, Kmax // 2), (n,), generator=gen) + 1
    elif kind == "nbr":
        par = ((torch.arange(H).view(1, H, 1) + torch.arange(W).view(1, 1, W)) % 2).expand(Cc, H, W).reshape(-1)
        if float(torch.rand(1, generator=gen)) < 0.5:
            par = 1 - par
        z = 1 + 2 * par + 4 * torch.randint(0, max(1, Kmax // 4), (n,), generator=gen)
    else:
        raise ValueError(f"thiết kế pixel không biết: {kind}")
    w = (torch.remainder(vdc(Kmax).view(Kmax, 1) * z.view(1, n), Kmax).to(F64) + eps) / Kmax
    return torch.remainder(pi.view(1, n) + w, 1.0), {"z": z, "pi": pi}


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


def pixel_schedule(alpha: float, N: int, shape, phase: torch.Tensor, gen) -> torch.Tensor:
    """Thí nghiệm A: DP độc lập theo phần tử (dùng đúng số gia của S0 nếu cùng `gen` ⇒ common random numbers), chỉ đặt lại
    vị trí số gia lớn nhất của từng phần tử tại ⌊N·φ_i⌋. -> Z (N+1, *shape) float64, mỗi đường ~ q_SPI."""
    d = spi_increments(float(alpha), int(N), tuple(shape), gen)
    Z = cum_nodes(anchor_rotate(d.reshape(int(N), -1), phase))
    return Z.view((int(N) + 1,) + tuple(shape))


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


def d_hat_cols(Z: torch.Tensor) -> float:
    """Trung bình theo phần tử/cột của ∫(Z − t)² dt (hình thang trên nút) — cùng định nghĩa với `e10.d_hat`."""
    N = Z.shape[0] - 1
    t = torch.linspace(0, 1, N + 1, device=Z.device, dtype=F64).view((-1,) + (1,) * (Z.dim() - 1))
    e = ((Z.double() - t) ** 2).reshape(N + 1, -1).mean(1)
    return float(torch.trapezoid(e, dx=1.0 / N))


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
