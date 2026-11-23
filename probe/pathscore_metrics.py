"""
PATHSCORE — metric bổ sung bên cạnh InsDel. Mọi hàm nhận bản đồ có dấu A (C,H,W); bản đồ pixel m = Σ_c|A| (quy ước P0–P3);
hoà phá bằng `tie_perm` của ảnh (cùng hoán vị cho mọi đường ⇒ so sánh công bằng).

Định vị theo mask GT (không tốn forward):
  iou           top-k theo m, k = diện tích mask, IoU với mask                       (= metrics.iou)
  mass_in_mask  Σ_mask m / Σ m                                                      (relevance mass accuracy, Arras et al. 2022)
  energy_mask   Σ_mask Σ_c A² / Σ Σ_c A²                                            (= metrics.energy_mask với V = 0)
  pointing      1 nếu argmax m nằm trong mask nới `pointing_tol` px, ngược lại 0      (pointing game, Zhang et al. 2018)
  prec_top      |top_frac(m) ∩ mask| / |top_frac(m)|                                (precision@k)
  auroc_mask    AUC-ROC của m tách pixel trong / ngoài mask (Mann–Whitney, hạng trung bình khi hoà)
  ap_mask       average precision của m so với mask (ngưỡng tại các giá trị phân biệt của m)

Faithfulness (cần forward):
  road_morf     ROAD (Rong et al. 2022): xoá p% pixel QUAN TRỌNG NHẤT, điền bằng nội suy tuyến tính có nhiễu (noisy linear
                imputation: mỗi pixel bị xoá = trung bình có trọng số 1/6 láng giềng cạnh, 1/12 láng giềng chéo, giải đúng hệ tuyến
                tính bằng PCG), xác suất softmax lớp đích, trung bình theo p ∈ fracs. Nhỏ = tốt.
  road_lerf     như trên nhưng xoá p% pixel ÍT quan trọng nhất. Lớn = tốt.
  road          road_lerf − road_morf. Lớn = tốt (cùng tinh thần InsDel nhưng không dùng nền mask cố định).
  sens_n        Pearson qua các tập con S (ô vuông ngẫu nhiên) giữa Σ_{c, i∈S} A và ΔF_S = F(x) − F(x với S thay bằng x′)
                (sensitivity-n, Ancona et al. 2018; F = logit lớp đích, x′ = baseline tích phân đen). Lớn = tốt.
  infid         Σ_S (Σ_S A − ΔF_S)² / Σ_S ΔF_S²: infidelity (Yeh et al. 2019) với nhiễu "xoá ô vuông về x′", chuẩn hoá theo
                năng lượng ΔF, KHÔNG co giãn tối ưu (bản đồ IG/SPI đã có thang đo nhờ completeness). Nhỏ = tốt.
ΔF_S, các tập S, và nhiễu ROAD chỉ phụ thuộc ảnh (không phụ thuộc đường) ⇒ dùng chung cho mọi đường và cho SPI.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as Fnn

from .data import STD
from .metrics import iou as _iou

F64 = torch.float64
NAN = float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# tiện ích
# ─────────────────────────────────────────────────────────────────────────────

def pixmap(A: torch.Tensor) -> torch.Tensor:
    """(C,H,W) → (H,W) = Σ_c |A|, float64."""
    return A.double().abs().sum(0)


def order_desc(m: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Chỉ số pixel (phẳng) theo m giảm dần; hoà phá theo thứ tự trong perm (giống metrics._rank_top)."""
    v = m.flatten()
    perm = perm.to(v.device)
    return perm[torch.argsort(v[perm], descending=True, stable=True)]


def dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    mask = mask.bool()
    if r <= 0:
        return mask
    return Fnn.max_pool2d(mask.double()[None, None], 2 * r + 1, stride=1, padding=r)[0, 0] > 0


def avg_ranks(v: torch.Tensor) -> torch.Tensor:
    """Hạng tăng dần 1..n, hoà lấy hạng trung bình."""
    v = v.flatten()
    s, o = torch.sort(v, stable=True)
    _, inv, cnt = torch.unique_consecutive(s, return_inverse=True, return_counts=True)
    end = torch.cumsum(cnt, 0).to(F64)
    start = end - cnt.to(F64) + 1
    rs = ((start + end) / 2)[inv]
    r = torch.empty_like(rs)
    r[o] = rs
    return r


def auroc(m: torch.Tensor, mask: torch.Tensor) -> float:
    v, y = m.flatten(), mask.flatten().bool().to(m.device)
    P = int(y.sum())
    N = y.numel() - P
    if P == 0 or N == 0:
        return NAN
    r = avg_ranks(v.double())
    return float((r[y].sum() - P * (P + 1) / 2) / (P * N))


def average_precision(m: torch.Tensor, mask: torch.Tensor) -> float:
    """AP = Σ_t (R_t − R_{t−1}) · P_t, t chạy qua các ngưỡng phân biệt của m (giảm dần), giống sklearn."""
    v, y = m.flatten().double(), mask.flatten().to(m.device).double()
    P = float(y.sum())
    if P == 0:
        return NAN
    s, o = torch.sort(v, descending=True, stable=True)
    ys = y[o]
    _, cnt = torch.unique_consecutive(s, return_counts=True)
    last = torch.cumsum(cnt, 0) - 1                       # vị trí phần tử cuối của mỗi nhóm hoà
    tp = torch.cumsum(ys, 0)[last]
    prec = tp / (last + 1).to(F64)
    rec = tp / P
    drec = torch.diff(rec, prepend=rec.new_zeros(1))
    return float((drec * prec).sum())


# ─────────────────────────────────────────────────────────────────────────────
# định vị (không forward)
# ─────────────────────────────────────────────────────────────────────────────

def localization(A: torch.Tensor, mask: torch.Tensor, perm: torch.Tensor, top_frac: float, pointing_tol: int) -> dict:
    m = pixmap(A)
    mk = mask.to(m.device).bool()
    y = mk.flatten()
    n = y.numel()
    out = {"iou": _iou(A.float(), mk, perm)}
    tot = float(m.sum())
    out["mass_in_mask"] = float(m.flatten()[y].sum()) / tot if tot > 0 else NAN
    e = (A.double() ** 2).sum(0).flatten()
    et = float(e.sum())
    out["energy_mask"] = float(e[y].sum()) / et if et > 0 else NAN
    o = order_desc(m, perm)
    k = max(1, int(top_frac * n))
    out["prec_top"] = float(y[o[:k]].double().mean())
    out["pointing"] = float(dilate(mk, int(pointing_tol)).flatten()[o[0]]) if tot > 0 else NAN
    out["auroc_mask"] = auroc(m, mk)
    out["ap_mask"] = average_precision(m, mk)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ROAD — noisy linear imputation
# ─────────────────────────────────────────────────────────────────────────────

_KW = ((1 / 12, 1 / 6, 1 / 12), (1 / 6, 0.0, 1 / 6), (1 / 12, 1 / 6, 1 / 12))


def _pull_push(v: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Ước lượng ban đầu cho pixel bị xoá (kim tự tháp trung bình phần đã biết), để PCG hội tụ nhanh. v, K: (B,1,H,W)."""
    nums, dens = [v * K], [K]
    while min(nums[-1].shape[-2:]) > 1:
        nums.append(Fnn.avg_pool2d(nums[-1], 2, ceil_mode=True))
        dens.append(Fnn.avg_pool2d(dens[-1], 2, ceil_mode=True))
    est = nums[-1] / dens[-1].clamp_min(1e-12)
    for n_, d_ in zip(reversed(nums[:-1]), reversed(dens[:-1])):
        up = Fnn.interpolate(est, size=n_.shape[-2:], mode="bilinear", align_corners=False)
        est = n_ + (1 - d_).clamp(0, 1) * up               # n_ = d_ · (trung bình phần đã biết)
    return est


def road_impute(x: torch.Tensor, remove: torch.Tensor, z: torch.Tensor, noise_sd: torch.Tensor,
                iters: int = 1000, tol: float = 1e-5) -> torch.Tensor:
    """x (1,C,H,W) không gian đã chuẩn hoá; remove (B,H,W) bool (True = điền lại); z (B,C,H,W) ~ N(0,1);
    noise_sd (C,) độ lệch chuẩn nhiễu theo kênh (đơn vị chuẩn hoá). Trả (B,C,H,W) cùng dtype với x.
    Giải L_RR u = b (Laplacian có trọng số, Dirichlet tại pixel đã biết) bằng CG tiền điều kiện Jacobi, theo lô."""
    B = remove.shape[0]
    _, C, H, W = x.shape
    dev = x.device
    k = torch.tensor(_KW, dtype=F64, device=dev).view(1, 1, 3, 3)

    def conv(t):
        return Fnn.conv2d(t, k, padding=1)

    R = remove.to(dev).to(F64).view(B, 1, H, W).expand(B, C, H, W).reshape(B * C, 1, H, W)
    Kn = 1 - R
    v = x.to(F64).expand(B, C, H, W).reshape(B * C, 1, H, W)
    d = conv(torch.ones(1, 1, H, W, dtype=F64, device=dev))          # Σ trọng số láng giềng nằm trong ảnh
    b = conv(v * Kn) * R

    def Aop(u):
        return (d * u - conv(u * R)) * R

    dims = (1, 2, 3)
    u = _pull_push(v, Kn) * R
    r = b - Aop(u)
    z_ = r / d
    p = z_.clone()
    rz = (r * z_).sum(dims, keepdim=True)
    bn = b.flatten(1).norm(dim=1).clamp_min(1e-30).view(-1, 1, 1, 1)
    for _ in range(int(iters)):
        act = (r.flatten(1).norm(dim=1).view(-1, 1, 1, 1) / bn) > tol
        if not bool(act.any()):
            break
        Ap = Aop(p)
        pAp = (p * Ap).sum(dims, keepdim=True)
        alpha = torch.where(act & (pAp > 0), rz / pAp.clamp_min(1e-300), torch.zeros_like(rz))
        u = u + alpha * p
        r = r - alpha * Ap
        z_ = r / d
        rz_new = (r * z_).sum(dims, keepdim=True)
        beta = torch.where(act & (rz > 0), rz_new / rz.clamp_min(1e-300), torch.zeros_like(rz))
        p = z_ + beta * p
        rz = rz_new
    out = (v * Kn + u * R).view(B, C, H, W)
    out = out + R.view(B, C, H, W) * z.to(dev, F64) * noise_sd.to(dev, F64).view(1, C, 1, 1)
    return out.to(x.dtype)


def road_noise_sd(noise: float, device) -> torch.Tensor:
    """Nhiễu `noise` (đơn vị pixel [0,1]) đổi sang không gian chuẩn hoá ImageNet, theo kênh."""
    return float(noise) / torch.tensor(STD, dtype=F64, device=device)


def road_masks(A: torch.Tensor, perm: torch.Tensor, fracs) -> torch.Tensor:
    """(2·|fracs|, H, W) bool: MoRF (xoá p% quan trọng nhất) rồi LeRF (xoá p% ít quan trọng nhất)."""
    m = pixmap(A)
    H, W = m.shape
    n = H * W
    o = order_desc(m, perm)
    rank = torch.empty(n, dtype=torch.long, device=o.device)
    rank[o] = torch.arange(n, device=o.device)
    rem = [rank < int(round(p * n)) for p in fracs] + [rank >= n - int(round(p * n)) for p in fracs]
    return torch.stack(rem).view(-1, H, W)


@torch.no_grad()
def road(prob, x: torch.Tensor, A: torch.Tensor, perm: torch.Tensor, fracs, z: torch.Tensor, noise_sd: torch.Tensor,
         batch: int, iters: int, tol: float) -> dict:
    """prob: (B,3,H,W) → (B,) xác suất lớp đích. z: (2·|fracs|, C, H, W) nhiễu chuẩn DÙNG CHUNG cho mọi bản đồ của ảnh."""
    R = road_masks(A, perm, fracs)
    pr = []
    for i in range(0, R.shape[0], batch):
        xi = road_impute(x, R[i:i + batch], z[i:i + batch], noise_sd, iters, tol)
        pr.append(prob(xi).double())
    pr = torch.cat(pr)
    nf = len(fracs)
    morf, lerf = float(pr[:nf].mean()), float(pr[nf:].mean())
    return {"road_morf": morf, "road_lerf": lerf, "road": lerf - morf}


# ─────────────────────────────────────────────────────────────────────────────
# sensitivity-n / infidelity (xoá ô vuông về baseline tích phân)
# ─────────────────────────────────────────────────────────────────────────────

def subset_masks(H: int, W: int, n: int, sizes, gen: torch.Generator) -> torch.Tensor:
    """(n, H, W) bool: ô vuông cạnh sizes[s % len(sizes)], vị trí đều trong ảnh (generator CPU)."""
    S = torch.zeros(int(n), H, W, dtype=torch.bool)
    for s in range(int(n)):
        a = int(sizes[s % len(sizes)])
        a = max(1, min(a, H, W))
        t = int(torch.randint(0, H - a + 1, (1,), generator=gen))
        l_ = int(torch.randint(0, W - a + 1, (1,), generator=gen))
        S[s, t:t + a, l_:l_ + a] = True
    return S


@torch.no_grad()
def subset_deltas(fn, x: torch.Tensor, x0: torch.Tensor, S: torch.Tensor, F_x: float, batch: int) -> torch.Tensor:
    """ΔF_S = F(x) − F(x với S thay bằng x0), F = hàm mục tiêu (logit lớp đích). Trả (n,) float64."""
    S = S.to(x.device)
    out = []
    for i in range(0, S.shape[0], batch):
        m = S[i:i + batch, None].expand(-1, x.shape[1], -1, -1)
        out.append(fn(torch.where(m, x0.expand_as(m), x.expand_as(m))).double())
    return F_x - torch.cat(out)


def subset_scores(A: torch.Tensor, S: torch.Tensor, dF: torch.Tensor) -> dict:
    a = A.double().sum(0)
    S = S.to(a.device)
    dF = dF.to(a.device, F64)
    pred = (a[None] * S.to(F64)).sum((1, 2))
    pc, dc = pred - pred.mean(), dF - dF.mean()
    den = math.sqrt(float((pc * pc).sum()) * float((dc * dc).sum()))
    e = float((dF * dF).sum())
    return {"sens_n": float((pc * dc).sum()) / den if den > 0 else NAN,
            "infid": float(((pred - dF) ** 2).sum()) / e if e > 0 else NAN}
