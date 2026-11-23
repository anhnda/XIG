"""
python -m probe [--config config.yaml] [--debug] [--device cuda] <lệnh>

Proposal "Lấy mẫu đường SPI theo infidelity" (mục infid:, kết quả <out>/pilot/infid/):
  toy                      E-toy nhỏ: oracle Phụ lục + hàm không đa tuyến (không cần mô hình) → infid/toy/TOY_REPORT.md
  toy-large                E-toy lớn d = 256: dấu V_o − V_e so với T9, Cov(H_ij, H_ik), coverage CI tỷ số
  toy-report               dựng lại TOY_REPORT.md
  pathscore                (đầu vào E0) bản đồ từng đường SPI + điểm → <out>/pilot/pathscore/ + ATTR_PROBE_PATHSCORE_REPORT.md
  pathscore-report         dựng lại báo cáo PATHSCORE từ CSV
  pathscore-infid-decomp   E0: audit PATHSCORE từ attr.pt (không gradient mới) → infid/e0/INFID_E0_REPORT.md
  pathscore-infid-report   dựng lại báo cáo E0
  quad-check               E-quadrature (C3): lưới lồng nhau N → 2N → … tới tolerance → infid/quad/INFID_QUAD_REPORT.md
  quad-check-report        dựng lại báo cáo E-quadrature
  sampling-infid-pilot     E1a pilot (cặp rev + nhánh indep nhỏ; C1c, C2, chọn m)
  infid-lock               khoá protocol (K chính, δ, quy tắc tích phân, D, cấu hình E1/E2) → infid/infid_prereg.json
  sampling-infid-e1a       E1a chính (chỉ cặp rev; tuỳ chọn global/nbr) — cần khoá
  sampling-infid-e1b-dev   E1b trên ảnh dev + bộ ô chọn → chọn cấu hình (e1b_selection.json) — cần khoá
  sampling-infid-e1b-test  E1b trên ảnh test + bộ ô đánh giá độc lập (H3) — cần khoá
  sampling-infid-e2        E2 kiểm định ngoài mẫu (nhiều D, metric ngoài, tuỳ chọn ViT) — cần khoá
  sampling-infid-report    dựng lại infid/INFID_SAMPLING_REPORT.md từ CSV
Chạy lại sẽ bỏ qua phần đã có trong cache (cùng hash config).
"""

from __future__ import annotations

import argparse

from . import config as C

CMDS = ["toy", "toy-large", "toy-report", "pathscore", "pathscore-report", "pathscore-infid-decomp", "pathscore-infid-report",
        "quad-check", "quad-check-report", "sampling-infid-pilot", "infid-lock", "sampling-infid-e1a", "sampling-infid-e1b-dev",
        "sampling-infid-e1b-test", "sampling-infid-e2", "sampling-infid-report"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="probe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--debug", action="store_true", help="ghi đè bằng mục debug: (ít ảnh, ít bước); ghi vào <out>_debug")
    ap.add_argument("--device", default=None)
    ap.add_argument("cmd", choices=CMDS)
    args = ap.parse_args(argv)
    cfg = C.load(args.config, debug=args.debug)
    cmd = args.cmd

    def ctx():
        from .ctx import Ctx
        return Ctx(cfg, args.device)

    if cmd.startswith("toy"):
        from . import toy_spi as TS
        {"toy": TS.run_small, "toy-large": TS.run_large, "toy-report": TS.write_report}[cmd](cfg)
    elif cmd.startswith("pathscore-infid"):
        from . import infid_e0 as E0
        if cmd == "pathscore-infid-decomp":
            E0.run(ctx())
        E0.write_report(cfg)
    elif cmd.startswith("pathscore"):
        from . import pathscore as PS
        if cmd == "pathscore":
            PS.run(ctx())
        PS.write_report(cfg)
    elif cmd.startswith("quad-check"):
        from . import infid_quad as IQ
        if cmd == "quad-check":
            IQ.run(ctx())
        IQ.write_report(cfg)
    elif cmd == "infid-lock":
        from . import sampling_infid as SI
        SI.lock(cfg)
    elif cmd == "sampling-infid-report":
        from . import sampling_infid as SI
        SI.write_report(cfg)
    else:
        from . import sampling_infid as SI
        SI.run(ctx(), cmd[len("sampling-infid-"):])


if __name__ == "__main__":
    main()
