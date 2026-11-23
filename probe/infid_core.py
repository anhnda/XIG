"""
Lõi chung cho proposal "Lấy mẫu đường SPI theo infidelity" (RQ1 / RQ2): phân phối nhiễu D, phép chiếu p_Sᵀ A, ΔF_S,
infidelity chuẩn hoá, bộ ước lượng chỉ dùng cặp `rev` (T10), phép thử sign-flip (C1c), sàn L*_free / L*_comp ở mức patch.

Quy ước (khớp pathscore_metrics.subset_scores):
  dự đoán      p_Sᵀ A = c_S · Σ_{i∈S} Σ_c A_{c,i}          (c_S = 1 cho xoá / chèn toàn phần, c cho xoá phân số)
  infid        L(A) = Σ_S (p_Sᵀ A − ΔF_S)² / Σ_S ΔF_S²     (W_D đều trên các ô; chuẩn hoá theo TỪNG ảnh)
  phương sai   V = E Σ_S (p_Sᵀ(Â − μ))² / Σ_S ΔF_S²        (cùng mẫu số ⇒ E L(Â) = L(μ) + V cộng được)

Phân tích chẵn / lẻ (Khung lý thuyết): E_j = (T(Z_j) + T(RZ_j))/2, O_j = (T(Z_j) − T(RZ_j))/2. Từ m cặp độc lập:
  V̂_e = Σ_j ‖P(E_j − Ē)‖² / ((m−1)·Σ ΔF²)       V̂_o = Σ_j ‖P O_j‖² / (m·Σ ΔF²)        (E[O] = 0 khi C1 đúng)
  L̂(μ) = L(Ē) − V̂_e / m                          J_indep(K) = L̂ + (V̂_e + V̂_o)/K       J_rev(K) = L̂ + 2V̂_e/K
  ΔV_D(K) = (V̂_o − V̂_e)/K                        ρ̂_D = 2V̂_e / (V̂_e + V̂_o)

Các kiểu D (mỗi ô S một bộ (z_from, z_to) trong không gian z, x = x′ + z ⊙ Δx khi nền = baseline tích phân):
  deletion   x_S ← x_S − c·(x_S − bg_S)       ΔF = F(x) − F(x̃),     p_S = c·1_S      (c = 1: xoá toàn phần)
  insertion  x̃ = bg + 1_S ⊙ (x − bg)          ΔF = F(x̃) − F(bg),    p_S = 1_S
  symmetric  hợp (nửa xoá, nửa chèn) với trọng số đều trên các ô
Nền `black` = baseline tích phân ⇒ p_S đúng nghĩa Yeh et al. Nền `blur` không nằm trên đoạn x′–x ⇒ dùng dự đoán chỉ thị
p_S = c·1_S (quy ước giống InsDel nền blur) — ghi rõ trong báo cáo, không phải infid "đúng nghĩa".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import torch

from . import config as C
from .data import MEAN, STD
from .metrics import make_base, tie_perm
from .model import TargetFn
from . import sampling as SM

F64 = torch.float64
NAN = float("nan")


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# môi trường một ảnh
# ─────────────────────────────────────────────────────────────────────────────

class IEnv:
    """Môi trường một ảnh (x, x′ đen, F, lớp dự đoán, mask, hoán vị hoà)."""

    def __init__(self, ctx, idx: int):
        self.ctx = ctx
        it, x, mask, cls = ctx.load(idx)
        SM.FwdCounter.n += 1
        self.it, self.x, self.mask, self.cls, self.image = it, x, mask, cls, it.image
        self.fn = TargetFn(ctx.bb, cls)
        self.x0 = make_base(x, "black")
        self.dx = x - self.x0
        self.shape = tuple(self.dx.shape[1:])
        self.H, self.W = self.shape[1], self.shape[2]
        self.perm = tie_perm(it.image, self.H * self.W, ctx.seed, ctx.dev)
        ic = ctx.cfg["metrics"]["insdel"]
        self.ic = ic
        self.ins_batch = int(ic["batch"])
        with torch.no_grad():
            f = self.fn(torch.cat([self.x0, x])).double()
        SM.FwdCounter.n += 2
        self.F0, self.F1 = float(f[0]), float(f[1])
        self.D = self.F1 - self.F0
        self._bg = {"black": self.x0}

    def bg(self, name: str) -> torch.Tensor:
        if name not in self._bg:
            if name != "blur":
                raise ValueError(f"nền không biết: {name}")
            self._bg[name] = make_base(self.x, "blur", self.ic["blur_kernel"], self.ic["blur_sigma"])
        return self._bg[name]

    def gen(self, *parts):
        return self.ctx.gen(*parts)

    def cgen(self, *parts) -> torch.Generator:
        return torch.Generator().manual_seed(C.seed_of(self.ctx.seed, *parts))

    def info(self) -> dict:
        return {"image": self.image, "index": self.it.index, "cls": self.cls, "F_x": self.F1, "F_base": self.F0, "D": self.D,
                "mask_area": float(self.mask.float().mean())}


# ─────────────────────────────────────────────────────────────────────────────
# tập ô S
# ─────────────────────────────────────────────────────────────────────────────

def square_boxes(H: int, W: int, n: int, sizes, gen: torch.Generator) -> torch.Tensor:
    """(n, 4) long [t, l, h, w]: ĐÚNG thuật toán của pathscore_metrics.subset_masks (cùng gen ⇒ cùng ô)."""
    out = torch.zeros(int(n), 4, dtype=torch.long)
    for s in range(int(n)):
        a = int(sizes[s % len(sizes)])
        a = max(1, min(a, H, W))
        t = int(torch.randint(0, H - a + 1, (1,), generator=gen))
        l_ = int(torch.randint(0, W - a + 1, (1,), generator=gen))
        out[s] = torch.tensor([t, l_, a, a])
    return out


def box_masks(boxes: torch.Tensor, H: int, W: int, device) -> torch.Tensor:
    b = boxes.to(device)
    ys = torch.arange(H, device=device).view(1, H, 1)
    xs = torch.arange(W, device=device).view(1, 1, W)
    t, l_, h, w = (b[:, i].view(-1, 1, 1) for i in range(4))
    return (ys >= t) & (ys < t + h) & (xs >= l_) & (xs < l_ + w)


def slic_lite(x: torch.Tensor, n_segments: int, compactness: float = 10.0, iters: int = 10) -> torch.Tensor:
    """Superpixel kiểu SLIC rút gọn trên GPU: k-means trên (màu RGB [0,1]·100, toạ độ·m/S), khởi tạo lưới đều, gán toàn cục
    (n_seg × HW nhỏ). Không ép liên thông. x: (1,3,H,W) đã chuẩn hoá ImageNet. Trả nhãn liên tiếp (H, W) long."""
    _, Cc, H, W = x.shape
    dev = x.device
    m = torch.tensor(MEAN, device=dev, dtype=F64).view(3, 1, 1)
    sd = torch.tensor(STD, device=dev, dtype=F64).view(3, 1, 1)
    img = (x[0].double() * sd + m).clamp(0, 1) * 100.0
    S = math.sqrt(H * W / max(int(n_segments), 1))
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=F64), torch.arange(W, device=dev, dtype=F64), indexing="ij")
    feat = torch.cat([img.reshape(3, -1), (torch.stack([yy, xx]) * (float(compactness) / S)).reshape(2, -1)], 0).T   # (HW, 5)
    gy = max(1, int(round(H / S)))
    gx = max(1, int(round(W / S)))
    cy = ((torch.arange(gy, device=dev, dtype=F64) + 0.5) * H / gy).long().clamp(max=H - 1)
    cx = ((torch.arange(gx, device=dev, dtype=F64) + 0.5) * W / gx).long().clamp(max=W - 1)
    idx = (cy.view(-1, 1) * W + cx.view(1, -1)).flatten()
    cent = feat[idx].clone()
    lab = None
    for _ in range(int(iters)):
        d = torch.cdist(feat, cent)
        lab = d.argmin(1)
        cnt = torch.zeros(cent.shape[0], device=dev, dtype=F64).index_add_(0, lab, torch.ones_like(lab, dtype=F64))
        s = torch.zeros_like(cent).index_add_(0, lab, feat)
        keep = cnt > 0
        cent = torch.where(keep.view(-1, 1), s / cnt.clamp_min(1).view(-1, 1), cent)
    _, lab = torch.unique(lab, return_inverse=True)
    return lab.view(H, W)


@dataclass
class Part:
    """Một khối ô đồng nhất. boxes (n,4) HOẶC (seg_map (n,), seg_id (n,), maps (n_maps,H,W))."""
    kind: str                              # deletion | insertion
    bg: str                                # black | blur
    coef: torch.Tensor                     # (n,) float64
    group: torch.Tensor                    # (n,) long — cỡ ô (px) hoặc số superpixel của bản đồ
    boxes: torch.Tensor | None = None
    seg_map: torch.Tensor | None = None
    seg_id: torch.Tensor | None = None
    maps: torch.Tensor | None = None
    dF: torch.Tensor | None = None         # (n,) float64

    @property
    def n(self) -> int:
        return int(self.coef.numel())

    def masks(self, i0: int, i1: int, H: int, W: int, device) -> torch.Tensor:
        if self.boxes is not None:
            return box_masks(self.boxes[i0:i1], H, W, device)
        mp = self.maps.to(device)[self.seg_map[i0:i1].to(device)]
        return mp == self.seg_id[i0:i1].to(device).view(-1, 1, 1)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("kind", "bg", "coef", "group", "boxes", "seg_map", "seg_id", "maps", "dF")}


@dataclass
class DSet:
    name: str
    parts: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return sum(p.n for p in self.parts)

    @property
    def dF(self) -> torch.Tensor:
        return torch.cat([p.dF.double().cpu() for p in self.parts])

    @property
    def group(self) -> torch.Tensor:
        return torch.cat([p.group.cpu() for p in self.parts])

    @property
    def kind_vec(self) -> list:
        return sum(([p.kind] * p.n for p in self.parts), [])

    def project(self, A: torch.Tensor, summed: bool = False) -> torch.Tensor:
        """A: (C,H,W) | (P,C,H,W) | (H,W) / (P,H,W) nếu summed. → (P, n) float64 (P = 1 nếu A một bản đồ)."""
        a = A.double()
        if not summed:
            a = a.sum(-3)
        if a.dim() == 2:
            a = a.unsqueeze(0)
        return torch.cat([project_part(p, a) for p in self.parts], 1)

    def to_dict(self) -> dict:
        return {"name": self.name, "parts": [p.to_dict() for p in self.parts]}

    @staticmethod
    def from_dict(d: dict) -> "DSet":
        return DSet(d["name"], [Part(**p) for p in d["parts"]])


def project_part(p: Part, a: torch.Tensor) -> torch.Tensor:
    """a (P, H, W) float64 bản đồ pixel (đã cộng kênh) → (P, n) = c_S · Σ_{i∈S} a_i."""
    P_, H, W = a.shape
    dev = a.device
    if p.boxes is not None:
        I = torch.zeros(P_, H + 1, W + 1, dtype=F64, device=dev)
        I[:, 1:, 1:] = a.cumsum(1).cumsum(2)
        b = p.boxes.to(dev)
        t, l_, h, w = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        s = I[:, t + h, l_ + w] - I[:, t, l_ + w] - I[:, t + h, l_] + I[:, t, l_]
    else:
        maps = p.maps.to(dev)
        s = torch.empty(P_, p.n, dtype=F64, device=dev)
        sm, si = p.seg_map.to(dev), p.seg_id.to(dev)
        af = a.reshape(P_, -1)
        for mi in range(maps.shape[0]):
            sel = sm == mi
            if not bool(sel.any()):
                continue
            lab = maps[mi].reshape(-1)
            nseg = int(lab.max()) + 1
            tot = torch.zeros(P_, nseg, dtype=F64, device=dev).index_add_(1, lab, af)
            s[:, sel] = tot[:, si[sel]]
    return s * p.coef.to(dev).view(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# xây D cho một ảnh
# ─────────────────────────────────────────────────────────────────────────────

def _forward_values(E: IEnv, imgs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = E.fn(imgs).double()
    SM.FwdCounter.n += int(imgs.shape[0])
    return out


@torch.no_grad()
def compute_dF(E: IEnv, p: Part, batch: int) -> torch.Tensor:
    x, bg = E.x, E.bg(p.bg)
    out = []
    Fbg = None
    if p.kind == "insertion":
        Fbg = float(_forward_values(E, bg)[0]) if p.bg != "black" else E.F0
    for i0 in range(0, p.n, batch):
        i1 = min(p.n, i0 + batch)
        mb = p.masks(i0, i1, E.H, E.W, x.device).unsqueeze(1)                              # (b,1,H,W) bool
        m = mb.to(x.dtype)
        cc = p.coef[i0:i1]
        c = cc.to(x.device, x.dtype).view(-1, 1, 1, 1)
        if p.kind == "deletion":
            if bool((cc == 1).all()):
                me = mb.expand(-1, x.shape[1], -1, -1)                                       # đúng phép thay của subset_deltas
                xt = torch.where(me, bg.expand_as(me), x.expand_as(me))
            else:
                xt = x - c * m * (x - bg)
            out.append(E.F1 - _forward_values(E, xt))
        elif p.kind == "insertion":
            me = mb.expand(-1, x.shape[1], -1, -1)
            xt = torch.where(me, x.expand_as(me), bg.expand_as(me))
            out.append(_forward_values(E, xt) - Fbg)
        else:
            raise ValueError(p.kind)
    return torch.cat(out).cpu()


def make_parts(E: IEnv, spec: dict, n: int, stream: str) -> list:
    """spec (một mục infid.D.sets): {kind, shape: square|superpixel, sizes | n_segments, bg, c}. `stream` tách tập chọn / đánh giá."""
    kind = spec.get("kind", "deletion")
    bg = spec.get("bg", "black")
    shape = spec.get("shape", "square")
    c = float(spec.get("c", 1.0))
    if kind == "symmetric":
        h = int(n) // 2
        a = make_parts(E, {**spec, "kind": "deletion"}, h, stream + "/sym-del")
        b = make_parts(E, {**spec, "kind": "insertion", "c": 1.0}, int(n) - h, stream + "/sym-ins")
        return a + b
    if kind == "fraction":
        kind = "deletion"
    if kind not in ("deletion", "insertion"):
        raise ValueError(f"D.kind không biết: {kind}")
    g = E.cgen("infid-D", stream, E.image)
    if shape == "square":
        sizes = [int(s) for s in spec.get("sizes", [16, 32, 56])]
        boxes = square_boxes(E.H, E.W, n, sizes, g)
        return [Part(kind, bg, torch.full((int(n),), c, dtype=F64), boxes[:, 2].clone(), boxes=boxes)]
    if shape == "superpixel":
        nsegs = [int(s) for s in spec.get("n_segments", [50, 100, 200])]
        maps = torch.stack([slic_lite(E.x, k, float(spec.get("compactness", 10.0)), int(spec.get("iters", 10))).cpu()
                            for k in nsegs])
        sm = torch.randint(0, len(nsegs), (int(n),), generator=g)
        nseg_real = torch.tensor([int(maps[i].max()) + 1 for i in range(len(nsegs))])
        si = (torch.rand(int(n), generator=g, dtype=F64) * nseg_real[sm].double()).long()
        return [Part(kind, bg, torch.full((int(n),), c, dtype=F64), torch.tensor(nsegs)[sm], seg_map=sm, seg_id=si, maps=maps)]
    raise ValueError(f"D.shape không biết: {shape}")


def build_dset(E: IEnv, name: str, spec: dict, n: int, stream: str, batch: int) -> DSet:
    parts = make_parts(E, spec, n, stream)
    for p in parts:
        p.dF = compute_dF(E, p, batch)
    return DSet(name, parts)


def dset_cached(E: IEnv, cdir: str, name: str, spec: dict, n: int, stream: str, batch: int, extra_key=None) -> DSet:
    """Cache ΔF theo (ảnh, spec, n, stream, model). Hash đổi ⇒ tính lại."""
    os.makedirs(cdir, exist_ok=True)
    dg = _h({"spec": spec, "n": int(n), "stream": stream, "seed": E.ctx.seed, "model": E.ctx.cfg["model"],
             "data": E.ctx.cfg["data"], "insdel": E.ic, "extra": extra_key, "v": 1})
    p = os.path.join(cdir, f"D_{name}_{stream.replace('/', '-')}.pt")
    if os.path.exists(p):
        obj = torch.load(p, weights_only=False)
        if obj.get("digest") == dg:
            return DSet.from_dict(obj["dset"])
    ds = build_dset(E, name, spec, n, stream, batch)
    torch.save({"digest": dg, "dset": ds.to_dict()}, p)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# infidelity, phân rã
# ─────────────────────────────────────────────────────────────────────────────

def infid(y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """y (P, n) hoặc (n,), d (n,) → (P,) L chuẩn hoá."""
    d = d.to(y.device, F64)
    y = y.double()
    return ((y - d) ** 2).sum(-1) / (d * d).sum()


def infid_beta(y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Chẩn đoán thang đo: min_β L(βA) = 1 − (yᵀd)² / (yᵀy · dᵀd). Không dùng làm objective."""
    d = d.to(y.device, F64)
    y = y.double()
    yy = (y * y).sum(-1)
    return 1.0 - (y * d).sum(-1) ** 2 / (yy.clamp_min(1e-300) * (d * d).sum())


def _gidx(groups, n):
    out = {"all": None}
    if groups is not None:
        for g in sorted(set(int(v) for v in groups.tolist())):
            out[f"g{g}"] = torch.nonzero(groups == g, as_tuple=True)[0]
    return out


def num_den(Yb: torch.Tensor, d: torch.Tensor, var_mean_raw: float, L_mu: float) -> dict:
    """Tách L_D(μ) = E_D / C_D (sửa đổi 3): C_D = E_D[ΔF_S²] (mẫu số), E_D = E_D[(p_Sᵀμ − ΔF_S)²] (tử số, đã khử phương sai),
    và độ lớn của dự đoán so với ΔF: rms_ratio = sqrt(E_D[(p_Sᵀμ)²] / C_D) (‖Pμ‖² ước lượng không chệch = ‖PȲ‖² − tr Var(PȲ)),
    beta_opt = ⟨Pμ, ΔF⟩ / ‖Pμ‖² (hệ số co giãn tối ưu), cos_bar = cos(PȲ, ΔF) (từ ước lượng, hơi chệch về 0)."""
    n = int(d.numel())
    C_D = float((d * d).sum()) / n
    pm2 = max(float((Yb * Yb).sum()) - float(var_mean_raw), 0.0)
    dot = float((Yb * d).sum())
    yn = float(Yb.norm())
    return {"C_D": C_D, "E_D": L_mu * C_D, "rms_ratio": math.sqrt(pm2 / n / C_D) if C_D > 0 else NAN,
            "beta_opt": dot / pm2 if pm2 > 0 else NAN, "cos_bar": dot / (yn * math.sqrt(n * C_D)) if yn > 0 and C_D > 0 else NAN}


def pair_core(y1: torch.Tensor, y2: torch.Tensor, d: torch.Tensor, Ks, n_flip: int = 0, gen=None,
              jackknife: bool = False) -> dict:
    """T10 trên m cặp (y1 = P T(Z_j), y2 = P T(RZ_j)), (m, n) float64. Mọi phương sai chia cho Σ ΔF² của chính tập ô."""
    m = int(y1.shape[0])
    if m < 2:
        raise ValueError("cần m ≥ 2 cặp")
    d = d.to(y1.device, F64)
    den = float((d * d).sum())
    E = (y1 + y2) / 2
    O = (y1 - y2) / 2
    Eb = E.mean(0)
    Ve_raw = float(((E - Eb) ** 2).sum()) / (m - 1)
    Vo_raw = float((O ** 2).sum()) / m
    Ve, Vo = Ve_raw / den, Vo_raw / den
    L_Eb = float(((Eb - d) ** 2).sum()) / den
    L_mu = L_Eb - Ve / m
    out = {"m": m, "n_sets": int(d.numel()), "den": den, "Ve": Ve, "Vo": Vo, "Vpath": Ve + Vo, "L_Ebar": L_Eb, "L_mu": L_mu,
           "rho": 2 * Ve / (Ve + Vo) if Ve + Vo > 0 else NAN,
           "Ve_raw": Ve_raw, "Vo_raw": Vo_raw, "L_Ebar_raw": L_Eb * den}
    out.update(num_den(Eb, d, Ve_raw / m, L_mu))
    for K in Ks:
        out[f"J_indep_{K}"] = L_mu + (Ve + Vo) / K
        out[f"J_rev_{K}"] = L_mu + 2 * Ve / K
        out[f"dV_{K}"] = (Vo - Ve) / K
    Ob = O.mean(0)
    T = float((Ob ** 2).sum()) / den
    out["signflip_T"] = T
    if n_flip > 0:
        s = (torch.randint(0, 2, (int(n_flip), m), generator=gen, device=gen.device if gen is not None else "cpu")
             .to(O.device, F64) * 2 - 1)
        Ts = ((s @ O / m) ** 2).sum(1) / den
        out["signflip_p"] = float((1 + (Ts >= T - 1e-15 * max(T, 1e-300)).sum()) / (n_flip + 1))
    if jackknife:
        idx = torch.arange(m, device=y1.device)
        dvs = []
        for j in range(m):
            k = idx != j
            Ej, Oj = E[k], O[k]
            ve = float(((Ej - Ej.mean(0)) ** 2).sum()) / (m - 2) / den if m > 2 else NAN
            vo = float((Oj ** 2).sum()) / (m - 1) / den
            dvs.append(vo - ve)
        dv = torch.tensor(dvs, dtype=F64)
        out["jk_se_VoVe"] = float(math.sqrt((m - 1) / m * float(((dv - dv.mean()) ** 2).sum()))) if m > 2 else NAN
    return out


def pair_stats(y1, y2, d, Ks, groups=None, n_flip: int = 0, gen=None, jackknife: bool = False) -> dict:
    """{nhóm: pair_core} — "all" + từng cỡ ô (báo cáo tách theo cỡ ô)."""
    res = {}
    for g, ix in _gidx(groups, y1.shape[1]).items():
        if ix is None:
            res[g] = pair_core(y1, y2, d, Ks, n_flip, gen, jackknife)
        elif len(ix) >= 2:
            ixd = ix.to(y1.device)
            res[g] = pair_core(y1[:, ixd], y2[:, ixd], d.to(y1.device)[ixd], Ks, 0, None, False)
    return res


def indep_core(Y: torch.Tensor, d: torch.Tensor, Ks, c2_sizes=(1, 2, 4, 8, 16)) -> dict:
    """P đường iid (P, n): σ̂² chuẩn hoá (ddof 1), L̂(μ) = L(Ȳ) − σ̂²/P, J(K) = L̂ + σ̂²/K, đẳng thức trung bình infid đường,
    C2: K·V(K) từ các nhóm rời nhau."""
    P = int(Y.shape[0])
    d = d.to(Y.device, F64)
    den = float((d * d).sum())
    Yb = Y.mean(0)
    s2 = float(((Y - Yb) ** 2).sum()) / (P - 1) / den
    L_mean = float(((Yb - d) ** 2).sum()) / den
    Lp = infid(Y, d)
    out = {"P": P, "den": den, "sigma2": s2, "L_mean": L_mean, "L_mu": L_mean - s2 / P, "mean_path_infid": float(Lp.mean()),
           "identity_gap": float(Lp.mean()) - (L_mean + s2 * (P - 1) / P), "var_share_at_P": (s2 / P) / L_mean if L_mean > 0 else NAN}
    out.update(num_den(Yb, d, s2 * den / P, out["L_mu"]))
    for K in Ks:
        out[f"J_{K}"] = out["L_mu"] + s2 / K
    for K in c2_sizes:
        ng = P // K
        if ng >= 2:
            G = Y[:ng * K].view(ng, K, -1).mean(1)
            VK = float(((G - G.mean(0)) ** 2).sum()) / (ng - 1) / den
            out[f"KV_{K}"] = K * VK
            out[f"KV_{K}_ngroups"] = ng
    return out


# ─────────────────────────────────────────────────────────────────────────────
# sàn ở mức patch
# ─────────────────────────────────────────────────────────────────────────────

def patch_design(E: IEnv, ds: DSet, G: int, chunk: int = 256) -> torch.Tensor:
    """X (n, G²) với X[S, g] = c_S · |S ∩ g| / |g|: dự đoán của bản đồ hằng theo patch (a_g chia đều trong patch, Σ_g a_g = ΣA)."""
    dev = E.x.device
    lab = SM.grid_pixel_labels(E.H, E.W, G).to(dev)
    P = G * G
    oh = torch.zeros(E.H * E.W, P, device=dev, dtype=torch.float32)
    oh[torch.arange(E.H * E.W, device=dev), lab] = 1.0
    cnt = oh.sum(0).double()
    rows = []
    for p in ds.parts:
        for i0 in range(0, p.n, chunk):
            i1 = min(p.n, i0 + chunk)
            mk = p.masks(i0, i1, E.H, E.W, dev).reshape(i1 - i0, -1).float()
            rows.append((mk @ oh).double() / cnt.view(1, -1) * p.coef[i0:i1].to(dev).view(-1, 1))
    return torch.cat(rows)


def _solve(X, d, lam, Dsum=None):
    p = X.shape[1]
    Gm = X.T @ X
    sc = float(torch.diagonal(Gm).mean()) if p else 1.0
    Gm = Gm + (float(lam) + 1e-10) * max(sc, 1e-300) * torch.eye(p, dtype=F64, device=X.device)   # jitter: ô phủ kém ⇒ suy biến
    b = X.T @ d
    if Dsum is None:
        return torch.linalg.solve(Gm, b)
    one = torch.ones(p, 1, dtype=F64, device=X.device)
    Kk = torch.cat([torch.cat([Gm, one], 1), torch.cat([one.T, torch.zeros(1, 1, dtype=F64, device=X.device)], 1)], 0)
    rhs = torch.cat([b, torch.tensor([float(Dsum)], dtype=F64, device=X.device)])
    return torch.linalg.solve(Kk, rhs)[:p]


def floor_fit(Xf: torch.Tensor, df: torch.Tensor, Xe: torch.Tensor, de: torch.Tensor, Dsum: float, lams, folds: int,
              gen: torch.Generator) -> dict:
    """L*_free và L*_comp ở mức patch: chọn λ bằng CV k-fold trên tập fit, khớp lại toàn bộ fit, đánh giá trên tập ô độc lập.
    In-sample < sàn quần thể < out-of-sample (kỳ vọng) — báo cả hai; G0 dùng out-of-sample (thận trọng cho quyết định "đi")."""
    Xf, Xe = Xf.double(), Xe.double()
    df, de = df.to(Xf.device, F64), de.to(Xf.device, F64)
    n = Xf.shape[0]
    perm = torch.randperm(n, generator=gen).to(Xf.device)
    fid = torch.arange(n, device=Xf.device)[perm] % max(2, int(folds))
    out = {}
    for name, comp in (("free", None), ("comp", Dsum)):
        best, best_l = None, math.inf
        cv = {}
        for lam in lams:
            ls = []
            for f in range(max(2, int(folds))):
                tr, va = fid != f, fid == f
                a = _solve(Xf[tr], df[tr], lam, comp)
                ls.append(float(((Xf[va] @ a - df[va]) ** 2).sum() / (df[va] ** 2).sum()))
            cv[lam] = sum(ls) / len(ls)
            if cv[lam] < best_l:
                best_l, best = cv[lam], lam
        a = _solve(Xf, df, best, comp)
        out[f"L_{name}_in"] = float(((Xf @ a - df) ** 2).sum() / (df ** 2).sum())
        out[f"L_{name}_oos"] = float(((Xe @ a - de) ** 2).sum() / (de ** 2).sum())
        out[f"L_{name}_cv"] = best_l
        out[f"lam_{name}"] = float(best)
        out[f"sumA_{name}"] = float(a.sum())
        out[f"_coef_{name}"] = a.detach()
    return out


def coef_stats(a: torch.Tensor, D: float, prefix: str = "") -> dict:
    """Chẩn đoán triệt tiêu của một bản đồ hệ số a (patch hoặc pixel, đã cộng kênh):
      C_cancel = ‖a‖₁ / (|Σa| + ε)          (= 1 khi mọi hệ số cùng dấu; lớn ⇒ khối dương / âm lớn triệt nhau)
      r_neg    = Σ max(−a, 0) / (Σ|a| + ε)  (tỷ lệ khối lượng âm)
      l1_over_D = ‖a‖₁ / |ΔF| ; max_over_D = max|a| / |ΔF| ; max_over_mean = max|a| / mean|a| ; frac_neg = tỷ lệ phần tử âm."""
    a = a.double().flatten()
    l1 = float(a.abs().sum())
    eps = 1e-12 * (abs(D) + 1.0)
    aD = max(abs(D), eps)
    mx = float(a.abs().max()) if a.numel() else float("nan")
    return {f"{prefix}C_cancel": l1 / (abs(float(a.sum())) + eps), f"{prefix}r_neg": float((-a).clamp_min(0).sum()) / (l1 + eps),
            f"{prefix}l1_over_D": l1 / aD, f"{prefix}max_over_D": mx / aD,
            f"{prefix}max_over_mean": mx / max(l1 / max(a.numel(), 1), eps), f"{prefix}frac_neg": float((a < 0).double().mean())}


def mask_rank(E: IEnv, ds: DSet) -> int:
    """rank(P) của ma trận chỉ thị ô (n × HW) ở mức pixel (trên đúng tập ô cố định, sàn tự do pixel = 0 nếu rank = n và n ≤ HW)."""
    dev = E.x.device
    M = torch.cat([p.masks(0, p.n, E.H, E.W, dev).reshape(p.n, -1).double() for p in ds.parts])
    return int(torch.linalg.matrix_rank(M @ M.T))
