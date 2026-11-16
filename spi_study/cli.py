"""
python -m spi_study [--config config.yaml] <lệnh> [tuỳ chọn]

Lệnh
  lock          khoá config (ghi SHA-256 vào config.yaml.lock)
  index         tạo danh sách ảnh (ImageNet-Seg: n ảnh mới không thuộc 200 ảnh cũ; Pet: n_test ảnh test)
  power         power analysis từ JSONL cũ (Giai đoạn 0)
  attr          tính attribution của một nhóm task (chạy tiếp được)
  converge      kiểm hội tụ của thang; --extend thêm một mức nếu chưa hội tụ (mã thoát 3 = cần chạy lại attr)
  eval          tính metric từ attribution đã lưu; --line thêm LINE(s)
  energy        tỉ lệ năng lượng khử nhiễu trong mask (CPU, chỉ đọc attribution); --line thêm LINE(s)
  gate1         Cửa 1: thống kê và quyết định
  collapse      Giai đoạn 1B: chỉ số thu gọn, Q_f(h), S_ε, kết luận Cổng 2
  pet-train     huấn luyện một ResNet-18 Pet (--seed)
  pet-models    báo cáo accuracy / đồng thuận / ECE / D_m của các mô hình Pet
  ensemble      Giai đoạn 2: V_model(h), nghiệm tầm thường, V1, faithfulness phần bị loại

Tuỳ chọn chung: --out, --device, --setting imagenet|pet, --pet-seed, --limit (chỉ n ảnh đầu: smoke test,
nên kèm --out khác), --mat (gtsegs_ijcv.mat), --pet-root.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import config as C


def _common(ap):
    ap.add_argument("--out", default=None, help="thư mục kết quả (mặc định local.out)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--setting", default="imagenet", choices=["imagenet", "pet"])
    ap.add_argument("--pet-seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="chỉ dùng n ảnh đầu danh sách (smoke test)")
    ap.add_argument("--mat", default=None, help="đường dẫn gtsegs_ijcv.mat (mặc định local.imagenet_mat)")
    ap.add_argument("--pet-root", default=None)


def _setting(cfg, args, target="predicted"):
    from .setting import Setting
    return Setting(cfg, args.setting, args.out, args.device, args.pet_seed, target, args.mat,
                   args.pet_root, args.limit)


def _target(cfg, group):
    return cfg["groups"][group].get("target", "predicted")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="spi_study", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lock")
    p.add_argument("--force", action="store_true")
    p.add_argument("--reason", default="")

    p = sub.add_parser("index"); _common(p)
    p.add_argument("--old-index", default=None, help="runs/seg200.json cũ (mặc định local.old_index_file)")

    p = sub.add_parser("power"); _common(p)
    p.add_argument("--old-jsonl", default=None)

    for name in ("attr", "converge", "eval", "energy"):
        p = sub.add_parser(name); _common(p)
        p.add_argument("--group", required=True)
        if name == "converge":
            p.add_argument("--extend", action="store_true")
        if name in ("eval", "energy"):
            p.add_argument("--line", action="store_true")

    p = sub.add_parser("gate1"); _common(p)
    p.add_argument("--group", default="stage1a")
    p = sub.add_parser("collapse"); _common(p)
    p.add_argument("--group", default="stage1b")

    p = sub.add_parser("pet-train"); _common(p)
    p.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("pet-models"); _common(p)
    p = sub.add_parser("ensemble"); _common(p)

    args = ap.parse_args(argv)
    cfg = C.load(args.config)

    if args.cmd == "lock":
        C.write_lock(cfg, force=args.force, reason=args.reason)
        return

    if args.cmd != "power":                 # power chạy TRƯỚC khi khoá: nó quyết định n_images
        C.require_lock(cfg)
    loc = cfg.get("local", {}) or {}
    out = args.out or loc.get("out", "runs")
    pet_root = args.pet_root or loc.get("pet_root", "data/pet")

    if args.cmd == "index":
        from . import data as D
        if args.setting == "imagenet":
            st = _setting(cfg, args)
            ds = st.dataset()
            D.make_imagenet_index(len(ds), cfg["data"]["imagenet"],
                                  args.old_index or loc.get("old_index_file"), st.index_file())
        else:
            ds = D.PetTest(pet_root, cfg["data"]["pet"]["size"])
            D.make_pet_index(len(ds), cfg["data"]["pet"], os.path.join(out, "pet_index.json"))
        return

    if args.cmd == "power":
        from .power import run_power
        run_power(cfg, args.old_jsonl or loc.get("old_jsonl"), os.path.join(out, "imagenet"))
        return

    if args.cmd == "attr":
        from .reference import run_attr
        run_attr(_setting(cfg, args, _target(cfg, args.group)), args.group)
        return

    if args.cmd == "converge":
        from .reference import run_converge
        sys.exit(run_converge(_setting(cfg, args, _target(cfg, args.group)), args.group, args.extend))

    if args.cmd == "eval":
        from .evaluate import run_eval
        run_eval(_setting(cfg, args, _target(cfg, args.group)), args.group, line=args.line)
        return

    if args.cmd == "energy":
        from .energy import run_energy
        run_energy(_setting(cfg, args, _target(cfg, args.group)), args.group, line=args.line)
        return

    if args.cmd == "gate1":
        from .gate1 import run_gate1
        run_gate1(_setting(cfg, args, _target(cfg, args.group)), args.group)
        return

    if args.cmd == "collapse":
        from .collapse import run_collapse
        run_collapse(_setting(cfg, args, _target(cfg, args.group)))
        return

    if args.cmd == "pet-train":
        from .model import set_precision
        from .pet.train import run_train
        set_precision(cfg["model"]["allow_tf32"])
        run_train(cfg, args.seed, out, pet_root, args.device)
        return

    if args.cmd == "pet-models":
        from .pet.ensemble import run_models_report
        run_models_report(cfg, out, pet_root, args.device)
        return

    if args.cmd == "ensemble":
        from .pet.ensemble import run_ensemble
        run_ensemble(cfg, out, pet_root, args.device)
        return


if __name__ == "__main__":
    main()