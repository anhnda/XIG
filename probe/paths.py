"""
Các họ đường đi. Mỗi hàm trả về (dict quy tắc -> A, thông tin đầu-cuối).

straight : x_k = x′ + (k/N)·Δx                                  (IG)
spi      : x_k = x′ + Z_k ⊙ Δx, Z ~ quá trình Dirichlet theo phần tử trên lưới N ô, trung bình K đường
           chia n_sub nhóm độc lập (cho phương sai MC). Cùng seed cho mọi biến thể mô hình (CRN).
guided   : Guided IG (Kapishnikov et al., CVPR 2021), cài lại theo thuật toán tham chiếu của thư viện
           PAIR `saliency` (steps=128, fraction=0.25, max_dist=0.02). Đường phụ thuộc mô hình.
blur     : BlurIG (Xu et al., CVPR 2020): x_k = GaussianBlur(x, σ_k), σ_k = σ_max·(1 − k/N). Điểm đầu là
           ảnh bị làm mờ mạnh, KHÔNG phải ảnh đen; completeness theo f(x) − f(blur_σmax(x)).
           Bước dùng hiệu trạng thái x_{k+1} − x_k thay cho đạo hàm theo σ nhân Δσ (tương đương khi N lớn).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import generator, seed_of
from .integrate import RULES, accumulate, fg, run_states

F64 = torch.float64


# ─────────────────────────────────────────────────────────────────────────────
# làm mờ Gaussian (tách được, reflect)
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_blur(x: torch.Tensor, sigma: float, truncate: float = 4.0) -> torch.Tensor:
    """x: (B, C, H, W). σ = 0 trả lại x."""
    if sigma <= 0:
        return x
    H, W = x.shape[-2:]
    r = min(int(math.ceil(truncate * sigma)), H - 1, W - 1)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t * t / (2 * sigma * sigma))
    k = k / k.sum()
    C = x.shape[1]
    y = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)


# ─────────────────────────────────────────────────────────────────────────────
# SPI: số gia Dirichlet trong log-space, nút cộng dồn chính xác
# ─────────────────────────────────────────────────────────────────────────────

def gamma_ge1(k: float, size, gen) -> torch.Tensor:
    """Gamma(k, 1), k ≥ 1, Marsaglia–Tsang, float64, mọi nguồn ngẫu nhiên đi qua `gen`."""
    d = k - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    n_tot = int(torch.tensor(size).prod())
    out = torch.empty(n_tot, dtype=F64, device=gen.device)
    todo = torch.arange(n_tot, device=gen.device)
    while todo.numel():
        n = todo.numel()
        x = torch.randn(n, generator=gen, device=gen.device, dtype=F64)
        u = torch.rand(n, generator=gen, device=gen.device, dtype=F64)
        v = 1.0 + c * x
        pos = v > 0
        v3 = torch.where(pos, v ** 3, torch.ones_like(v))
        ok = pos & (torch.log(u) < 0.5 * x * x + d - d * v3 + d * torch.log(v3))
        out[todo[ok]] = d * v3[ok]
        todo = todo[~ok]
    return out.view(tuple(size))


def spi_nodes(alpha: float, N: int, shape, gen) -> torch.Tensor:
    """Z tại nút k = 0..N, (N+1, *shape), float64. Nửa trên tính bằng phần bù để giữ chính xác sát 1."""
    a = alpha / N
    size = (N,) + tuple(shape)
    L = torch.log(gamma_ge1(a + 1.0, size, gen)) + \
        torch.log(torch.rand(size, generator=gen, device=gen.device, dtype=F64).clamp_min(1e-300)) / a
    delta = torch.softmax(L, dim=0)
    m = N // 2
    zero = torch.zeros_like(delta[:1])
    fwd = torch.cumsum(delta[:m], dim=0)
    tail = torch.flip(torch.cumsum(torch.flip(delta[m + 1:], [0]), dim=0), [0])
    Z = torch.cat([zero, fwd, 1.0 - tail, zero + 1.0], dim=0).clamp_(0.0, 1.0)
    return torch.cummax(Z, dim=0).values


# ─────────────────────────────────────────────────────────────────────────────
# các phương pháp
# ─────────────────────────────────────────────────────────────────────────────

def straight(fn, x, x0, mc, batch, **_):
    N = int(mc["steps"])
    t = torch.linspace(0, 1, N + 1, device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
    out = run_states(fn, x0 + t * (x - x0), batch)
    return out, {"start": x0}


def spi(fn, x, x0, mc, batch, seed_key=(), **_):
    """Trả mean, phương sai MC theo phần tử (V = s²/n_sub) và các sub-estimate."""
    N, K, B = int(mc["steps"]), int(mc["paths"]), int(mc.get("n_sub", 4))
    if K % B:
        raise SystemExit(f"spi: paths={K} phải chia hết cho n_sub={B}")
    dx = x - x0
    subs = {r: [] for r in RULES}
    f0 = fN = None
    for b in range(B):
        acc = {r: torch.zeros_like(x[0], dtype=F64) for r in RULES}
        for p in range(K // B):
            gen = generator(seed_of(*seed_key, "spi", mc["alpha"], N, b, p), x.device)
            Z = spi_nodes(float(mc["alpha"]), N, x.shape[1:], gen).to(x.dtype)
            o = run_states(fn, x0 + Z * dx, batch)
            for r in RULES:
                acc[r] += o[r].double()
            f0, fN = o["_f0"], o["_fN"]
        for r in RULES:
            subs[r].append((acc[r] / (K // B)).float())
    out = {}
    for r in RULES:
        S = torch.stack(subs[r]).double()
        out[r] = S.mean(0).float()
        out[r + "_var"] = (S.var(0, unbiased=True) / B).float()
        out[r + "_subs"] = subs[r]
    out["_f0"], out["_fN"] = f0, fN
    return out, {"start": x0}


def blur(fn, x, x0, mc, batch, **_):
    N, smax = int(mc["steps"]), float(mc["max_sigma"])
    sig = [smax * (1 - k / N) for k in range(N + 1)]
    X = torch.cat([gaussian_blur(x, s) for s in sig])
    out = run_states(fn, X, batch)
    return out, {"start": X[:1]}


def guided(fn, x, x0, mc, batch, **_):
    """Guided IG trong tọa độ z = (x − x′)/Δx theo từng phần tử. Ghi lại các trạng thái đầu mỗi bước."""
    N, frac, md = int(mc["steps"]), float(mc["fraction"]), float(mc["max_dist"])
    eps = 1e-9
    dx = (x - x0)[0]
    adx = dx.abs()
    flat = adx.flatten()
    zero = adx == 0
    z = torch.zeros_like(dx)
    z[zero] = 1.0                                   # phần tử không đổi: coi như đã ở đích, không bao giờ được chọn
    l1_total = float(adx.sum())
    states, fs, gs = [], [], []
    for step in range(N):
        xc = (x0[0] + z * dx).unsqueeze(0)
        f, g = fg(fn, xc, batch)
        states.append(xc[0]); fs.append(f[0]); gs.append(g[0])
        g = g[0]
        alpha = (step + 1) / N
        a_min, a_max = max(alpha - md, 0.0), min(alpha + md, 1.0)
        l1_target = l1_total * (1 - alpha)
        for _ in range(10_000):
            z = torch.where(~zero & (z < a_min), torch.full_like(z, a_min), z)
            l1_cur = float((adx * (1 - z)).sum())
            if math.isclose(l1_target, l1_cur, rel_tol=eps, abs_tol=eps):
                break
            ga = g.abs().clone()
            ga[(z >= a_max) | zero] = float("inf")
            fin = torch.isfinite(ga)
            if not bool(fin.any()):
                break
            # như bản tham chiếu: phân vị trên MỌI phần tử, kể cả các phần tử đã tới đích (gán inf)
            thr = torch.quantile(ga.flatten().double(), frac, interpolation="lower")
            s = fin & (ga <= thr)
            l1_s = float((adx * (a_max - z) * s).sum())
            gamma = (l1_cur - l1_target) / l1_s if l1_s > 0 else float("inf")
            if gamma > 1.0:
                z = torch.where(s, torch.full_like(z, a_max), z)
            else:
                z = torch.where(s, z + gamma * (a_max - z), z)
                break
    xN = (x0[0] + z * dx).unsqueeze(0)
    fN, gN = fg(fn, xN, batch)
    X = torch.stack(states + [xN[0]])
    f = torch.stack(fs + [fN[0]])
    out = accumulate(X, f, torch.stack(gs))
    out["_f0"], out["_fN"] = float(f[0]), float(f[-1])
    out["_end_gap"] = float((xN - x).abs().max())
    return out, {"start": x0}


KINDS = {"straight": straight, "spi": spi, "blur": blur, "guided": guided}


def run_method(name: str, mc: dict, fn, x, x0, batch, seed_key=()):
    return KINDS[mc["kind"]](fn, x, x0, mc, batch, seed_key=seed_key)
