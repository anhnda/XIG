"""
python -m probe [--config config.yaml] [--debug] [--device cuda] <lệnh>

  estimate  đo tốc độ GPU, in ước tính thời gian conv/main (không chạy thí nghiệm)
  conv      P0: hội tụ theo N (left/mid/rand, IDGI các biến thể)
  main      P1 + P2 + P3 trong một lượt mỗi ảnh
  report    đồ thị + ATTR_PROBE_PILOT_REPORT.md từ các CSV đã có
  all       conv → main → report
  mech-e1      E1: kiểm định số học của phân rã (đường cố định, gộp bước nhảy DP, toy)
  mech-e23     E2 + E3: phân rã cơ chế trên ResNet (SPI K đường, IG) và trên bẫy H
  mech-report  MECHANISM_REPORT.md + đồ thị từ CSV đã có
  mech         mech-e1 → mech-e23 → mech-report
  cross-gate   E1′: between do nhiễu cầu phương (M lồng nhau) + toy bậc hai/bậc năm
  cross-e4     E4a+b: IG, SPI, Guided IG, BlurIG cùng thước đo exposure (mode, trạng thái, IDGI)
  cross-e4c    E4c: quét mức lệch nhịp (SPI α)
  cross-report CROSS_REPORT.md + đồ thị từ CSV đã có
  cross        cross-gate → cross-e4 → cross-e4c → cross-report
  e5           E5: residual avoidance, hoán vị lịch trình Guided, đường λ, tín hiệu → E5_REPORT.md
  e5-report    E5_REPORT.md + đồ thị từ CSV đã có
  e6           E6: A (cô lập gradient guidance, K=1) + B (topology chia sẻ lịch trình) → E6_REPORT.md
  e6-report    E6_REPORT.md + đồ thị từ CSV đã có
  e7           E7: nhóm đồng bộ cùng cỡ (block vs random), độ bền theo N, Guided ở cầu phương dày → E7_REPORT.md
  e7-report    E7_REPORT.md + đồ thị từ CSV đã có
  e8           E8: xác nhận held-out các phép so đã tiền đăng ký (split confirm, 50 ảnh) → E8_REPORT.md
  e8-report    E8_REPORT.md + đồ thị từ CSV đã có

Kết quả: <local.out>[_debug]/pilot/. Chạy lại sẽ bỏ qua ảnh đã có trong cache (cùng hash config).
"""

from __future__ import annotations

import argparse

from . import config as C


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="probe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--debug", action="store_true", help="ghi đè bằng mục debug: (ít ảnh, ít bước); ghi vào <out>_debug")
    ap.add_argument("--device", default=None)
    ap.add_argument("cmd", choices=["estimate", "conv", "main", "report", "all",
                                    "mech-e1", "mech-e23", "mech-report", "mech",
                                    "cross-gate", "cross-e4", "cross-e4c", "cross-report", "cross", "e5", "e5-report",
                                    "e6", "e6-report", "e7", "e7-report",
                                    "e8", "e8-report"])
    args = ap.parse_args(argv)
    cfg = C.load(args.config, debug=args.debug)

    if args.cmd == "report":
        from .report import write
        write(C.out_dir(cfg, "pilot"), cfg)
        return
    if args.cmd.startswith("e8"):
        from . import e8 as E8
        from .e8_report import write as w8
        if args.cmd == "e8":
            from .pilot import Ctx
            E8.run(Ctx(cfg, args.device))
        w8(E8.e8_out(cfg), cfg)
        return
    if args.cmd.startswith("e7"):
        from . import e7 as E7
        from .e7_report import write as w7
        if args.cmd == "e7":
            from .pilot import Ctx
            E7.run(Ctx(cfg, args.device))
        w7(E7.e7_out(cfg), cfg)
        return
    if args.cmd.startswith("e6"):
        from . import e6 as E6
        from .e6_report import write as w6
        if args.cmd == "e6":
            from .pilot import Ctx
            E6.run(Ctx(cfg, args.device))
        w6(E6.e6_out(cfg), cfg)
        return
    if args.cmd.startswith("e5"):
        from . import e5 as E5
        from .e5_report import write as ewrite
        if args.cmd == "e5":
            from .pilot import Ctx
            E5.run(Ctx(cfg, args.device))
        ewrite(E5.e5_out(cfg), cfg)
        return
    if args.cmd.startswith("cross"):
        from . import cross as X
        from .cross_report import write as cwrite
        if args.cmd != "cross-report":
            from .pilot import Ctx
            ctx = Ctx(cfg, args.device)
            for st in ("gate", "e4", "e4c"):
                if args.cmd in (f"cross-{st}", "cross"):
                    X.run(ctx, st)
        cwrite(X.cross_out(cfg), cfg)
        return
    if args.cmd.startswith("mech"):
        from . import mechanism as Mx
        from .mech_report import write as mwrite
        if args.cmd != "mech-report":
            from .pilot import Ctx
            ctx = Ctx(cfg, args.device)
            if args.cmd in ("mech-e1", "mech"):
                Mx.run(ctx, "e1")
            if args.cmd in ("mech-e23", "mech"):
                Mx.run(ctx, "e23")
        mwrite(Mx.mech_out(cfg), cfg)
        return

    from . import pilot as P
    ctx = P.Ctx(cfg, args.device)
    if args.cmd == "estimate":
        b = P.bench(ctx)
        e = P.estimate(cfg, b)
        print(f"{b['per_eval'] * 1e3:.2f} ms/eval (lô {ctx.batch}), {b['per_seq'] * 1e3:.1f} ms/bước Guided")
        print(f"conv ≈ {e['conv_min']:.1f} phút ({cfg['convergence']['n_images']} ảnh)")
        print(f"main ≈ {e['main_min']:.1f} phút ({cfg['exp_a']['n_images']} ảnh A; tín hiệu {e['n_signal_variants']} biến thể × "
              f"{cfg['exp_c']['n_images']} ảnh; bẫy {e['n_trap_variants']} biến thể × {cfg['exp_b']['n_images']} ảnh)")
        print(f"mỗi ảnh A: {e['evals_per_image_A']} eval theo lô + {e['seq_per_image_A']} eval tuần tự")
        print({k: (v if isinstance(v, int) else len(v)) for k, v in ctx.sp.items()})
        return
    if args.cmd in ("conv", "all"):
        P.run_stage(ctx, "conv")
    if args.cmd in ("main", "all"):
        P.run_stage(ctx, "main")
    from .report import write
    write(ctx.out, cfg)


if __name__ == "__main__":
    main()