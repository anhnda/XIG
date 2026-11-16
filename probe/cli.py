"""
python -m probe [--config config.yaml] [--debug] [--split pilot|confirm] [--device cuda] <lệnh>

  lock     khoá config (bắt buộc trước --split confirm)
  info     in danh sách ảnh của các split và ước tính số eval
  a        thí nghiệm A: đường đi × quy tắc chia credit
  b        thí nghiệm B: bẫy gradient
  c        thí nghiệm C: giữ tín hiệu đã biết
  sanity   khởi tạo lại tích luỹ theo tầng
  report   gom kết quả thành report.md
  all      a → b → c → sanity → report
"""

from __future__ import annotations

import argparse

from . import config as C


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="probe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--debug", action="store_true", help="vài ảnh, ít bước; ghi vào <out>_debug")
    ap.add_argument("--split", default="pilot", choices=["pilot", "confirm"])
    ap.add_argument("--device", default=None)
    ap.add_argument("cmd", choices=["lock", "info", "a", "b", "c", "sanity", "report", "all"])
    ap.add_argument("--reason", default="")
    args = ap.parse_args(argv)
    cfg = C.load(args.config, debug=args.debug)

    if args.cmd == "lock":
        C.write_lock(cfg, args.reason)
        return
    if args.split == "confirm":
        if args.debug:
            raise SystemExit("--split confirm không đi cùng --debug")
        C.require_lock(cfg)
    if args.cmd == "report":
        from .report import write
        write(C.out_dir(cfg, args.split))
        return

    from . import experiments as E
    ctx = E.Ctx(cfg, args.split, args.device)
    if args.cmd == "info":
        b, q = E.evals_per_variant(cfg)
        print({k: (v if isinstance(v, int) else len(v)) for k, v in ctx.sp.items()})
        print(f"mỗi biến thể mô hình, mỗi ảnh: {b} eval theo lô + {q} eval tuần tự (Guided IG)")
        nb = len(cfg["exp_b"]["shapes"]) * len(cfg["exp_b"]["u"]) * len(cfg["exp_b"]["rho"]) * len(cfg["exp_b"]["tau"])
        print(f"số biến thể: A 1, B {nb} (+ gốc), C 4 (+ gốc), sanity {len(cfg['sanity']['cascade'])} (+ gốc)")
        return
    steps = {"a": [E.run_a], "b": [E.run_b], "c": [E.run_c], "sanity": [E.run_sanity],
             "all": [E.run_a, E.run_b, E.run_c, E.run_sanity]}[args.cmd]
    for f in steps:
        f(ctx)
    from .report import write
    write(ctx.out)


if __name__ == "__main__":
    main()
