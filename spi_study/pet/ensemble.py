"""
Giai đoạn 2: báo cáo mô hình, V_model(h), phép thử faithfulness của phần bị loại (protocol 5.3, 7).

pet-models : accuracy, tỉ lệ đồng thuận dự đoán theo ảnh (toàn bộ test), ECE (15 bin), phân phối D_m
             trên 50 ảnh đã chọn (lớp đích = nhãn thật).
ensemble   : đọc tham chiếu của mọi seed (thư mục <out>/pet_s<m>_label), tính trên Ã = A / D_m, chỉ ảnh có
             |D_m| ≥ pet.d_min ở mọi mô hình:
    V_model(h) = trung bình ảnh của (1/M) Σ_m ‖Ã_m − mean_{m'≠m} Ã_{m'}‖² / d  −  phần phương sai estimator
                 (phần estimator ước lượng từ sub-estimate: Var(level mean) = s²_sub / n_sub, chuẩn hoá D_m²)
    cos        trung bình cosine theo cặp mô hình.
    Hai nghiệm tầm thường: cực tiểu tại IG; cực tiểu của họ beta không tốt hơn LINE.
    V1: V(α) − V(IG) = 2·Cov_m(I, D_α) + V(D_α) (dạng thực nghiệm trên Ã, không trừ phương sai estimator).
    Faithfulness phần bị loại: R_m = (IG − SPI_α)(F_m) − mean_{m'≠m}(IG − SPI_α)(F_{m'}); xoá top-p pixel theo
    Σ_c|R_m| (thay bằng blur), đo độ giảm xác suất lớp đích trên chính m và trên các mô hình khác, chia cho
    độ giảm khi xoá ngẫu nhiên cùng số pixel. ≈ 1 trên m: SPI đang khử nhiễu; ≫ 1: SPI xoá hành vi thật của m.
"""

from __future__ import annotations

import itertools
import math
import os

import numpy as np
import torch

from ..config import beta_spec, canon, line_spec, mix_spec
from ..data import PetTest, read_index
from ..metrics import make_baseline
from ..model import TargetProb, load_pet
from ..reference import n_sub_of, top_level
from ..setting import Setting, pet_ckpt
from ..storage import Store


# ─────────────────────────────────────────────────────────────────────────────
# báo cáo mô hình
# ─────────────────────────────────────────────────────────────────────────────

def _ece(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(1)
    acc = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += float(m.float().mean()) * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return e


def run_models_report(cfg: dict, out: str, root: str, device) -> dict:
    from torchvision.datasets import OxfordIIITPet
    from .train import predict, test_transform
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = [int(s) for s in cfg["pet"]["seeds"]]
    te = OxfordIIITPet(root, split="test", target_types="category",
                       transform=test_transform(int(cfg["data"]["pet"]["size"])), download=True)
    dl = torch.utils.data.DataLoader(te, batch_size=128, shuffle=False, num_workers=4)
    preds, rec = {}, {"models": {}}
    lines = ["# Mô hình Pet — Giai đoạn 2\n", f"init = {cfg['pet']['init']}\n",
             "| seed | test acc | ECE (15 bin) | D_m trung vị (50 ảnh) | D_m min | số ảnh |D_m| < d_min |",
             "|---|---|---|---|---|---|"]
    idx = read_index(os.path.join(out, "pet_index.json"))
    ds = PetTest(root, int(cfg["data"]["pet"]["size"]))
    for s in seeds:
        m = load_pet(pet_ckpt(out, s), dev)
        logits, labels = predict(m, dl, dev)
        probs = torch.softmax(logits, 1)
        preds[s] = logits.argmax(1)
        acc = float((preds[s] == labels).float().mean())
        ece = _ece(probs, labels)
        Ds = []
        for i in idx:
            it = ds.get(i)
            x = it.x.to(dev)
            xb = make_baseline(x, cfg["baseline"])
            with torch.no_grad():
                o = m(torch.cat([x, xb]))[:, it.label]
            Ds.append(float(o[0] - o[1]))
        Ds = np.array(Ds)
        small = int((np.abs(Ds) < float(cfg["pet"]["d_min"])).sum())
        rec["models"][s] = {"acc": acc, "ece": ece, "D": Ds.tolist()}
        lines.append(f"| {s} | {acc:.4f} | {ece:.4f} | {np.median(Ds):.3f} | {Ds.min():.3f} | {small} |")
    lines += ["\n## Tỉ lệ đồng thuận dự đoán theo ảnh (toàn bộ test)\n", "| cặp | đồng thuận |", "|---|---|"]
    rec["agreement"] = {}
    for a, b in itertools.combinations(seeds, 2):
        ag = float((preds[a] == preds[b]).float().mean())
        rec["agreement"][f"{a}-{b}"] = ag
        lines.append(f"| {a}–{b} | {ag:.4f} |")
    text = "\n".join(lines) + "\n"
    st = Store(os.path.join(out, "pet"))
    path = st.write_report("models", text, rec)
    print(text)
    print(f"[pet-models] -> {path}")
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# V_model
# ─────────────────────────────────────────────────────────────────────────────

class Models:
    """Tham chiếu theo từng mô hình (setting pet_s<m>_label)."""

    def __init__(self, cfg: dict, out: str, seeds: list[int], device=None):
        self.cfg = cfg
        self.sts = {s: Setting(cfg, "pet", out, device or "cpu", pet_seed=s, target="label") for s in seeds}
        self.seeds = seeds

    def images(self) -> list[str]:
        return self.sts[self.seeds[0]].image_ids()

    def D(self, s, im) -> float:
        return float(self.sts[s].store.load_meta(im)["D"])

    def ref(self, s, im, spec) -> tuple[torch.Tensor, torch.Tensor | None]:
        """-> (tham chiếu, tổng phương sai estimator của level mean theo phần tử) — chưa chuẩn hoá D."""
        st = self.sts[s]
        name = spec.split(":")[0]
        if name == "line":
            from ..config import num, parse_spec
            sv = num(parse_spec(spec)[1]["s"])
            ig = canon(self.cfg["gate1"]["ig_ref"])
            a1 = beta_spec(self.cfg["collapse"]["alpha_line"])
            Ai, vi = self.ref(s, im, ig)
            A1, v1 = self.ref(s, im, a1)
            return (1 - sv) * Ai + sv * A1, (1 - sv) ** 2 * vi + sv ** 2 * v1
        B = n_sub_of(self.cfg, spec)
        L = top_level(self.cfg, st, spec)
        A, subs = st.store.load_level(im, spec, L, B)
        S = torch.stack(subs).double()
        var = S.var(dim=0, unbiased=True) / B if B > 1 else torch.zeros_like(S[0])
        return A.double(), var


def _vmodel(Atil: list[torch.Tensor], vtil: list[torch.Tensor]) -> tuple[float, float, float]:
    """-> (V thô, phần estimator, cos trung bình) cho một ảnh."""
    M = len(Atil)
    d = Atil[0].numel()
    tot = sum(Atil)
    raw, est = 0.0, 0.0
    for m in range(M):
        loo = (tot - Atil[m]) / (M - 1)
        raw += float((Atil[m] - loo).pow(2).sum()) / d
        est += float(vtil[m].sum() + sum(vtil[k].sum() for k in range(M) if k != m) / (M - 1) ** 2) / d
    cos = []
    for a, b in itertools.combinations(range(M), 2):
        cos.append(float(torch.nn.functional.cosine_similarity(Atil[a].flatten(), Atil[b].flatten(), dim=0)))
    return raw / M, est / M, float(np.mean(cos))


def run_ensemble(cfg: dict, out: str, root: str, device) -> dict:
    seeds = [int(s) for s in cfg["pet"]["seeds"]]
    if len(seeds) < 2:
        raise SystemExit("cần ≥ 2 seed")
    mods = Models(cfg, out, seeds)
    images = mods.images()
    dmin = float(cfg["pet"]["d_min"])
    keep = [im for im in images if all(abs(mods.D(s, im)) >= dmin for s in seeds)]
    ig = canon(cfg["gate1"]["ig_ref"])
    alphas = [int(a) for a in cfg["collapse"]["alphas"]]
    fams = {"beta": beta_spec, "mix": mix_spec, "line": line_spec}
    rec = {"n_images": len(images), "n_kept": len(keep), "d_min": dmin, "V": {}, "cos": {}}

    def stat(spec):
        V, E, C = [], [], []
        for im in keep:
            At, vt = [], []
            for s in seeds:
                A, v = mods.ref(s, im, spec)
                Dm = mods.D(s, im)
                At.append(A / Dm)
                vt.append(v / Dm ** 2)
            raw, est, cos = _vmodel(At, vt)
            V.append(raw - est)
            E.append(est)
            C.append(cos)
        return float(np.mean(V)), float(np.mean(E)), float(np.mean(C))

    v_ig = stat(ig)
    rec["V"]["ig"] = {"0": v_ig[0]}
    rec["cos"]["ig"] = {"0": v_ig[2]}
    lines = [f"# Giai đoạn 2 — ổn định qua {len(seeds)} mô hình (seed {seeds})\n",
             f"{len(keep)}/{len(images)} ảnh có |D_m| ≥ {dmin} ở mọi mô hình. Ã = A/D_m.\n",
             "| h | họ | V_model (đã trừ estimator) | phần estimator | cos |", "|---|---|---|---|---|",
             f"| 0 (IG) | ig | {v_ig[0]:.4e} | {v_ig[1]:.2e} | {v_ig[2]:.4f} |"]
    for f, mk in fams.items():
        rec["V"][f], rec["cos"][f] = {}, {}
        for a in alphas:
            v = stat(mk(a))
            key = f"1/{a + 1}"
            rec["V"][f][key], rec["cos"][f][key] = v[0], v[2]
            lines.append(f"| {key} | {f} | {v[0]:.4e} | {v[1]:.2e} | {v[2]:.4f} |")

    # nghiệm tầm thường
    best = {f: min(rec["V"][f].items(), key=lambda kv: kv[1]) for f in fams}
    beta_min = best["beta"][1]
    triv_ig = v_ig[0] <= beta_min
    triv_line = beta_min >= best["line"][1]
    rec["argmin"] = {f: best[f][0] for f in fams}
    rec["trivial_min_at_ig"], rec["trivial_not_better_than_line"] = bool(triv_ig), bool(triv_line)
    lines += ["\n## Nghiệm tầm thường (tiêu chí dừng)\n",
              f"- Cực tiểu tại IG: {'CÓ (tầm thường)' if triv_ig else 'không'} "
              f"(V_IG = {v_ig[0]:.4e}, min V_beta = {beta_min:.4e} tại h = {best['beta'][0]})",
              f"- Họ beta không tốt hơn LINE: {'CÓ (tầm thường)' if triv_line else 'không'} "
              f"(min V_line = {best['line'][1]:.4e} tại h = {best['line'][0]})",
              f"- argmin theo họ: {rec['argmin']}  (cùng hình dạng qua các họ là điều kiện Cổng 3)",
              "- Pilot 3 seed KHÔNG được dùng làm bằng chứng tổng quát hoá.\n"]

    # V1
    lines += ["## Phân rã V1 trên Ã (không trừ phương sai estimator)\n",
              "| α | V(α) − V(IG) | 2·Cov(I, D_α) | V(D_α) |", "|---|---|---|---|"]
    rec["V1"] = {}
    for a in alphas:
        lhs, cov2, vd = [], [], []
        for im in keep:
            I = torch.stack([mods.ref(s, im, ig)[0] / mods.D(s, im) for s in seeds])
            S = torch.stack([mods.ref(s, im, beta_spec(a))[0] / mods.D(s, im) for s in seeds])
            Dd = S - I
            var = lambda X: float(X.var(dim=0, unbiased=True).mean())
            lhs.append(var(S) - var(I))
            cov2.append(float(2 * ((I - I.mean(0)) * (Dd - Dd.mean(0))).sum(0).div(len(seeds) - 1).mean()))
            vd.append(var(Dd))
        rec["V1"][str(a)] = {"lhs": float(np.mean(lhs)), "2cov": float(np.mean(cov2)), "vd": float(np.mean(vd))}
        lines.append(f"| {a} | {np.mean(lhs):.4e} | {np.mean(cov2):.4e} | {np.mean(vd):.4e} |")

    # faithfulness
    rec["faith"] = run_faith(cfg, mods, keep, root, device)
    fl = ["\n## Faithfulness của phần bị loại (α = %s)\n" % cfg["pet"]["faith"]["alpha"],
          "Tỉ lệ = độ giảm xác suất khi xoá theo |R_m| / độ giảm khi xoá ngẫu nhiên, trung bình qua ảnh.\n",
          "| p | trên chính m | trên mô hình khác |", "|---|---|---|"]
    for p, v in rec["faith"]["by_frac"].items():
        fl.append(f"| {p} | {v['own']:.3f} | {v['other']:.3f} |")
    lines += fl
    text = "\n".join(lines) + "\n"
    st = Store(os.path.join(out, "pet"))
    path = st.write_report("ensemble", text, rec)
    print(text)
    print(f"[ensemble] -> {path}")
    return rec


@torch.no_grad()
def run_faith(cfg: dict, mods: Models, keep: list[str], root: str, device) -> dict:
    fc = cfg["pet"]["faith"]
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = mods.seeds
    ig = canon(cfg["gate1"]["ig_ref"])
    spi = beta_spec(fc["alpha"])
    nets = {s: load_pet(pet_ckpt(mods.sts[s].out, s), dev) for s in seeds}
    ds = PetTest(root, int(cfg["data"]["pet"]["size"]))
    ic = cfg["metrics"]["insdel"]
    fracs = [float(p) for p in fc["fracs"]]
    acc = {p: {"own": [], "other": []} for p in fracs}
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    for im in keep:
        meta = mods.sts[seeds[0]].store.load_meta(im)
        it = ds.get(int(meta["index"]))
        x = it.x.to(dev)
        blur = make_baseline(x, "blur", ic["blur_kernel"], ic["blur_sigma"])
        H, W = x.shape[-2:]
        Dl = {s: (mods.ref(s, im, ig)[0] - mods.ref(s, im, spi)[0]) / mods.D(s, im) for s in seeds}
        tot = sum(Dl.values())
        for m in seeds:
            R = Dl[m] - (tot - Dl[m]) / (len(seeds) - 1)
            order = torch.argsort(R.abs().sum(0).flatten(), descending=True)
            for p in fracs:
                k = int(round(p * H * W))

                def drop(idx_pix, net):
                    msk = torch.zeros(H * W, dtype=torch.bool)
                    msk[idx_pix[:k]] = True
                    msk = msk.view(1, 1, H, W).to(dev)
                    xd = torch.where(msk, blur, x)
                    pm = TargetProb(net, it.label)
                    return float(pm(x)[0] - pm(xd)[0])

                rnd = [torch.randperm(H * W, generator=g) for _ in range(int(fc["n_random"]))]
                for s2 in seeds:
                    dr = drop(order.cpu(), nets[s2])
                    base = float(np.mean([drop(r, nets[s2]) for r in rnd]))
                    ratio = dr / base if abs(base) > 1e-6 else float("nan")
                    acc[p]["own" if s2 == m else "other"].append(ratio)
    return {"alpha": fc["alpha"], "by_frac": {str(p): {"own": float(np.nanmean(v["own"])),
                                                       "other": float(np.nanmean(v["other"]))}
                                              for p, v in acc.items()}}
