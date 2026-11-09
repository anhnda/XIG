"""
Shared helpers for experiment scripts: model loading, method specs, legacy
wrappers, per-image evaluation rows, JSONL i/o.

Method spec syntax (command line):
    name                    e.g. ig, idgi, guided_ig, dnoise, pixel_random
    name:k=v,k=v            e.g. dnoise:spread=0.75,tile=4   ig:N=200,est=gl,Q=4
    legacy_<m>[:fix_df=1]   original repo code (ig, idig, guided_ig, lig_idig, lig)
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
    }
    if name not in table:
        raise ValueError(f"unknown model '{name}', choose from {sorted(table)}")
    m = table[name]().to(device).eval()
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


LEGACY = ("legacy_ig", "legacy_idig", "legacy_guided_ig", "legacy_lig_idig", "legacy_lig")


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
        elif name == "legacy_lig_idig":
            from lig_idig import compute_lig_idig as f
        else:
            from lig import compute_lig as f
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
        return _legacy_fn(name), params, name in ("legacy_lig",)
    fn, base, stoch = M.resolve(name)
    base.update(params)
    if str(base.get("est", "left")).startswith("random"):
        stoch = True
    return fn, base, stoch


# ─────────────────────────────────────────────────────────────────────────────
# evaluation of one attribution
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor],
             attr: torch.Tensor, eval_bases: dict, n_steps: int = 100) -> dict:
    row = {}
    for b, base in eval_bases.items():
        s = MET.insertion_deletion(model, x, base, attr, n_steps=n_steps)
        row[f"ins_{b}"], row[f"del_{b}"], row[f"insdel_{b}"] = s["ins"], s["del"], s["insdel"]
    if mask is not None:
        row.update(MET.localisation(attr, mask))
    row["checker"] = MET.checker_energy(attr, mask)
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
    if args.data == "seg":
        ds = DATA.ImageNetSeg(args.seg_mat)
    else:
        ds = DATA.FolderImages(args.images, args.masks, prep=args.prep)
    idx = DATA.select_indices(len(ds), args.n, args.index_seed, args.index_file)
    return dev, backbone, DATA.iterate(ds, idx), len(idx)


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
