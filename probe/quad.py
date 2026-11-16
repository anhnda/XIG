"""
Đánh giá mạng trên một đường cho trước và các quy tắc chia credit, tính trên CÙNG đường (cùng nút, cùng điểm đầu-cuối).

Một đường là dãy nút x_0..x_N; giữa hai nút là đoạn thẳng (nội suy tuyến tính). Mạng được đánh giá:
    - tại các nút x_k:                          f(x_k), g_k = ∂f/∂x(x_k)          (N+1 lượt)
    - tại điểm cầu phương trên mỗi đoạn:        x_k + U_k·Δx_k                     (N lượt mỗi loại)
        mid  : U_k = ½
        rand : U_k ~ U(0,1), một giá trị cho mỗi đoạn, dùng chung mọi phần tử (không chệch với tích phân
               trên đoạn nội suy — KHÔNG bảo đảm đoạn nội suy trùng đường liên tục, ví dụ đường DP)

Quy tắc (tên cố định, không tự rơi sang quy tắc khác):
    canon_left   Σ_k g(x_k) ⊙ Δx_k                      Riemann trái (bản cũ)
    canon_mid    Σ_k g(x_k + ½Δx_k) ⊙ Δx_k
    canon_rand   Σ_k g(x_k + U_kΔx_k) ⊙ Δx_k
    idgi         Σ_k g_k²/Σ_j g_{k,j}² · ΔF_k,  g_k tại nút x_k, không gian input đã chuẩn hoá (đúng bài IDGI)
    idgi_raw     như idgi nhưng gradient theo pixel thô [0,1]: g/std_c (trọng số chia std_c²)
    idgi_rand    như idgi nhưng g lấy tại điểm cầu phương rand của chính đoạn đó
với ΔF_k = F(x_{k+1}) − F(x_k). Bước có Σ_j g_{k,j}² ≤ tol·max_k Σ_j g² bị BỎ (không chia epsilon); ΔF của
nó được báo trong `dropped_dF` và tự xuất hiện trong residual.

Thành phần cộng thêm c (tín hiệu S hoặc bẫy H) được áp mà không cần đánh giá lại mạng:
    F̃ = F + c, ∇F̃ = g + ∇c  (c có value/grad giải tích hoặc autograd riêng)
— hợp lệ khi đường không phụ thuộc mô hình (IG, SPI, BlurIG). Guided IG phải chạy lại (xem pilot.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

F64 = torch.float64


class Counter:
    n = 0          # số lượt forward+backward của mạng


def fg(fn, X: torch.Tensor, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """X: (M, C, H, W) -> f (M,) float64, g (M, C, H, W)."""
    fs, gs = [], []
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            out = fn(xb)
            (g,) = torch.autograd.grad(out.sum(), xb)
        fs.append(out.detach().double())
        gs.append(g.detach())
        Counter.n += int(xb.shape[0])
    return torch.cat(fs), torch.cat(gs)


@dataclass
class Evals:
    X: torch.Tensor                 # (N+1, C, H, W) nút
    f: torch.Tensor                 # (N+1,) float64
    g: torch.Tensor                 # (N, C, H, W) gradient tại nút 0..N−1
    Q: dict = field(default_factory=dict)   # tên -> (P (N,…), gP (N,…), U (N,))

    @property
    def N(self) -> int:
        return self.X.shape[0] - 1


def quad_offsets(kind: str, N: int, gen, device) -> torch.Tensor:
    if kind == "mid":
        return torch.full((N,), 0.5, device=device)
    if kind == "rand":
        return torch.rand(N, generator=gen, device=gen.device).to(device)
    raise ValueError(f"cầu phương không biết: {kind}")


def evaluate(fn, X: torch.Tensor, batch: int, quads=("rand",), gen=None,
             f_nodes=None, g_nodes=None) -> Evals:
    N = X.shape[0] - 1
    if f_nodes is None:
        f, g = fg(fn, X, batch)
        g = g[:N]
    else:
        f, g = f_nodes, g_nodes
    ev = Evals(X, f, g)
    for q in quads:
        if q == "left":
            continue
        U = quad_offsets(q, N, gen, X.device)
        P = X[:-1] + U.view(-1, *([1] * (X.dim() - 1))).to(X.dtype) * (X[1:] - X[:-1])
        _, gP = fg(fn, P, batch)
        ev.Q[q] = (P, gP, U)
    return ev


def shift(ev: Evals, comp) -> Evals:
    """Evals của F + c trên cùng đường, không đánh giá lại mạng."""
    if comp is None:
        return ev
    f = ev.f + comp.value(ev.X).double()
    g = ev.g + comp.grad(ev.X[:-1])
    Q = {q: (P, gP + comp.grad(P), U) for q, (P, gP, U) in ev.Q.items()}
    return Evals(ev.X, f, g, Q)


def unshift(ev: Evals, comp) -> Evals:
    """Ngược lại: từ Evals của F + c (ví dụ Guided IG chạy trên F + c) lấy phần của F trên cùng đường."""
    if comp is None:
        return ev
    f = ev.f - comp.value(ev.X).double()
    g = ev.g - comp.grad(ev.X[:-1])
    Q = {q: (P, gP - comp.grad(P), U) for q, (P, gP, U) in ev.Q.items()}
    return Evals(ev.X, f, g, Q)


def idgi(G: torch.Tensor, dF: torch.Tensor, tol: float, scale=None) -> tuple[torch.Tensor, dict]:
    Gd = G.double()
    if scale is not None:
        Gd = Gd / scale.double()
    w = Gd * Gd
    s = w.flatten(1).sum(1)
    smax = float(s.max()) if s.numel() else 0.0
    keep = s > tol * smax if smax > 0 else torch.zeros_like(s, dtype=torch.bool)
    shape = (-1,) + (1,) * (G.dim() - 1)
    if bool(keep.any()):
        A = (w[keep] / s[keep].view(shape) * dF[keep].view(shape)).sum(0)
    else:
        A = torch.zeros_like(Gd[0])
    return A.to(G.dtype), {"n_dropped": int((~keep).sum()), "dropped_dF": float(dF[~keep].sum())}


def rules(ev: Evals, tol: float = 1e-12, std=None) -> tuple[dict, dict]:
    """-> ({tên quy tắc: A (C,H,W)}, thông tin). std: (C,1,1) cho idgi_raw (None = bỏ)."""
    dX = ev.X[1:] - ev.X[:-1]
    dF = ev.f[1:] - ev.f[:-1]
    out = {"canon_left": (ev.g.double() * dX.double()).sum(0).to(ev.g.dtype)}
    for q, (P, gP, U) in ev.Q.items():
        out[f"canon_{q}"] = (gP.double() * dX.double()).sum(0).to(gP.dtype)
    out["idgi"], i0 = idgi(ev.g, dF, tol)
    info = {"idgi": i0, "D": float(ev.f[-1] - ev.f[0])}
    if std is not None:
        out["idgi_raw"], info["idgi_raw"] = idgi(ev.g, dF, tol, scale=std)
    if "rand" in ev.Q:
        out["idgi_rand"], info["idgi_rand"] = idgi(ev.Q["rand"][1], dF, tol)
    return out, info


def idgi_diagnostics(ev: Evals, mask: torch.Tensor, top_frac: float = 0.01) -> dict:
    """Theo từng bước k (gradient tại nút): q_k(M), entropy và top-1% mass của p = g²/Σg², ΔF_k."""
    w = ev.g.double() ** 2
    s = w.flatten(1).sum(1).clamp_min(1e-300)
    p = w.flatten(1) / s.view(-1, 1)
    m = mask.to(w.device).expand_as(w[0]).flatten().double()
    q = (p * m).sum(1)
    ent = -(p * torch.log(p.clamp_min(1e-300))).sum(1)
    k = max(1, int(top_frac * p.shape[1]))
    top = p.topk(k, dim=1).values.sum(1)
    dF = (ev.f[1:] - ev.f[:-1])
    a = dF.abs()
    P_M = float((a * q).sum() / a.sum()) if float(a.sum()) > 0 else float("nan")
    qm = float((a * q).sum() / a.sum()) if float(a.sum()) > 0 else float("nan")
    q_sd = float(torch.sqrt((a * (q - qm) ** 2).sum() / a.sum())) if float(a.sum()) > 0 else float("nan")
    return {"q": q.cpu(), "entropy": ent.cpu(), "top1": top.cpu(), "dF": dF.cpu(), "P_M": P_M, "q_sd_w": q_sd,
            "dF_pos": float(dF.clamp_min(0).sum()), "dF_neg": float(dF.clamp_max(0).sum()),
            "max_entropy": float(torch.log(torch.tensor(float(p.shape[1]))))}