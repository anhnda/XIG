"""
Shared helpers for experiment scripts: model loading, method specs, legacy
wrappers, per-image evaluation rows, JSONL i/o.

Method spec syntax (command line):
    name                    e.g. ig, idgi, guided_ig, dnoise, pixel_random
    name:k=v,k=v            e.g. dnoise:spread=0.75,tile=4   ig:N=200,est=gl,Q=4
    legacy_<m>[:fix_df=1]   original repo code (ig, idig, guided_ig, blurig)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Callable, Optional

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathattr import Oracle                                   # noqa: E402
from pathattr import methods as M                             # noqa: E402
from pathattr import metrics as MET                           # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# models
# ─────────────────────────────────────────────────────────────────────────────

class TargetLogit(nn.Module):
    """backbone -> logit of one class, shape (B,)."""

    def __init__(self, backbone: nn.Module, target: int):
        super().__init__()
        self.backbone, self.target = backbone, int(target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[:, self.target]


class TargetProb(nn.Module):
    """backbone -> softmax probability of one class, shape (B,)."""

    def __init__(self, backbone: nn.Module, target: int):
        super().__init__()
        self.backbone, self.target = backbone, int(target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.backbone(x), dim=-1)[:, self.target]


def load_backbone(name: str, device: torch.device) -> nn.Module:
    import torchvision.models as tvm
    table = {
        "resnet50": lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2),
        "resnet50_v1": lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1),
        "vgg16": lambda: tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1),
        "densenet121": lambda: tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1),
        "vit_b_16": lambda: tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1),
        "swin_b": lambda: tvm.swin_b(weights=tvm.Swin_B_Weights.IMAGENET1K_V1),
        "convnext_base": lambda: tvm.convnext_base(weights=tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1),
        "efficientnet_b0": lambda: tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
        "mobilenet_v2": lambda: tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2),
        "inception_v3": lambda: tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1),
    }
    if name not in table:
        raise ValueError(f"unknown model '{name}', choose from {sorted(table)}")
    m = table[name]()
    if name == "inception_v3":          # as in KIG compare_methods.load_model
        m.aux_logits = False
        m.AuxLogits = None
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# method specs
# ─────────────────────────────────────────────────────────────────────────────

def _parse_value(v: str):
    lv = v.lower()
    if lv in ("true", "false"):
        return lv == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_spec(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    params = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        params[k.strip()] = _parse_value(v.strip())
    return name.strip(), params


LEGACY = ("legacy_ig", "legacy_idig", "legacy_guided_ig", "legacy_blurig")


def _legacy_fn(name: str) -> Callable:
    """Wrap an original-repo method as fn(oracle, xb, x, gen, **params)."""
    import utility

    def run(oracle, xb, x, gen=None, fix_df: int = 0, **params):
        if name == "legacy_ig":
            from ig import compute_ig as f
        elif name == "legacy_idig":
            from idig import compute_idig as f
        elif name == "legacy_guided_ig":
            from guided_ig import compute_guided_ig as f
        else:
            from blurig import compute_blurig as f   # copied verbatim from KIG-main
        old = utility.FIX_DF_ALIGNMENT
        utility.FIX_DF_ALIGNMENT = bool(fix_df)
        try:
            p = {"baseline": xb, "N": 50}
            p.update(params)
            res = f(oracle.model, x, p)
        finally:
            utility.FIX_DF_ALIGNMENT = old
        mu = [float(m) for m in (s.mu_k for s in res.steps)]
        info = {"Q": res.Q, "Var_nu": res.Var_nu, "mu_argmax": int(max(range(len(mu)), key=mu.__getitem__)),
                "mu_support": int(sum(m > 1e-3 for m in mu)), "mu": mu}
        return M.Attribution(res.attributions.detach(), info)

    return run


def resolve(spec: str) -> tuple[Callable, dict, bool]:
    """spec -> (fn, params, stochastic)."""
    name, params = parse_spec(spec)
    if name in LEGACY:
        return _legacy_fn(name), params, False
    fn, base, stoch = M.resolve(name)
    base.update(params)
    if str(base.get("est", "left")).startswith("random") or base.get("path") == "dnoise":
        stoch = True
    return fn, base, stoch


# ─────────────────────────────────────────────────────────────────────────────
# evaluation of one attribution
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor],
             attr: torch.Tensor, eval_bases: dict, n_steps: int = 100,
             prob_model: Optional[nn.Module] = None, infd_n: int = 0,
             infd_sigma: float = 0.1, road: bool = False,
             gen: Optional[torch.Generator] = None, infd_base: Optional[torch.Tensor] = None,
             infd_mode: str = "delta", infd_cell: int = 16) -> dict:
    """insdel_<b>: legacy normalised logit AUC; insdelp_<b>: raw softmax-prob AUC;
    optional infidelity (Gaussian kernel, infd_n samples) and approximate ROAD."""
    row = {}
    for b, base in eval_bases.items():
        s = MET.insertion_deletion(model, x, base, attr, n_steps=n_steps)
        row[f"ins_{b}"], row[f"del_{b}"], row[f"insdel_{b}"] = s["ins"], s["del"], s["insdel"]
        if prob_model is not None:
            p = MET.insertion_deletion(prob_model, x, base, attr, n_steps=n_steps, normalize=False)
            row[f"insp_{b}"], row[f"delp_{b}"], row[f"insdelp_{b}"] = p["ins"], p["del"], p["insdel"]
    if mask is not None:
        row.update(MET.localisation(attr, mask))
    row["checker"] = MET.checker_energy(attr, mask)
    if infd_n > 0:
        row.update(MET.infidelity(model, x, attr, sigma=infd_sigma, n=infd_n, gen=gen,
                                  base=infd_base, mode=infd_mode, cell=infd_cell))
    if road:
        if prob_model is None:
            raise ValueError("ROAD needs the probability model")
        row.update(MET.road(prob_model, x, attr, gen=gen))
    return row


def run_one(fn: Callable, params: dict, oracle: Oracle, xb: torch.Tensor,
            x: torch.Tensor, seed: int) -> tuple[Optional[torch.Tensor], dict]:
    """Run a method with counters and timing; never raises (error goes in the row)."""
    gen = torch.Generator().manual_seed(int(seed))
    torch.manual_seed(int(seed))              # legacy code uses the global RNG
    oracle.reset()
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    try:
        res = fn(oracle, xb, x, gen, **params)
        if x.is_cuda:
            torch.cuda.synchronize()
        n_fwd, n_bwd = oracle.counts()
        return res.attr, {"time": time.time() - t0, "n_fwd": n_fwd, "n_bwd": n_bwd,
                          "info": dict(res.info), "error": None}
    except Exception as e:  # noqa: BLE001
        return None, {"time": time.time() - t0, "n_fwd": None, "n_bwd": None, "info": {},
                      "error": f"{type(e).__name__}: {e}",
                      "trace": traceback.format_exc(limit=3)}


# ─────────────────────────────────────────────────────────────────────────────
# data arguments shared by the P0/P1 scripts
# ─────────────────────────────────────────────────────────────────────────────

def add_data_args(ap) -> None:
    ap.add_argument("--data", choices=["seg", "folder"], required=True)
    ap.add_argument("--seg-mat", default=None)
    ap.add_argument("--images", default=None)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--prep", default="resize", choices=["resize", "crop"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--start", type=int, default=0, help="skip the first images of the index list")
    ap.add_argument("--index-file", default=None)
    ap.add_argument("--index-seed", type=int, default=0)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--baseline", default="black", choices=["black", "gray", "white", "blur"])
    ap.add_argument("--device", default=None)


def open_data(args):
    """-> (device, backbone, iterator over (idx, x, mask, name), n_images)"""
    from pathattr import data as DATA
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone = load_backbone(args.model, dev)
    size = input_size(args.model)
    if args.data == "seg":
        ds = DATA.ImageNetSeg(args.seg_mat, size=size)
    else:
        ds = DATA.FolderImages(args.images, args.masks, size=size, prep=args.prep)
    idx = DATA.select_indices(len(ds), args.n, args.index_seed, args.index_file,
                              start=getattr(args, "start", 0))
    return dev, backbone, DATA.iterate(ds, idx), len(idx)


def input_size(model_name: str) -> int:
    """299 for Inception v3 (Resize 342 + CenterCrop 299 with --prep crop), else 224."""
    return 299 if model_name == "inception_v3" else 224


def target_model(backbone: nn.Module, x: torch.Tensor) -> tuple[TargetLogit, int, float]:
    with torch.no_grad():
        prob = torch.softmax(backbone(x), dim=-1)[0]
    cls = int(prob.argmax())
    return TargetLogit(backbone, cls), cls, float(prob[cls])


# ─────────────────────────────────────────────────────────────────────────────
# JSONL
# ─────────────────────────────────────────────────────────────────────────────

def load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None:
                    done.add((r["image"], r["method"], r["seed"]))
    return done


def append_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
