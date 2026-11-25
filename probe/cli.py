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
  infid-posthoc            kiểm toán sau kiểm định (không GPU): H3b, E2 theo cặp, metric ngoài, J(K), độ nhạy C_D
                           → infid/INFID_POSTHOC_REPORT.md
  infid-freeze             đóng băng: sha256 mọi CSV / báo cáo / khoá / config / mã nguồn → infid/FROZEN.json
Nghiên cứu BUDGET (mục budget:, docs/PROTOCOL_BUDGET.md; kết quả infid/budget/ + infid/INFID_BUDGET_REPORT.md):
  budget-dev1              G1: α × coupling × K ở ngân sách B (dev) → budget/selection_dev1.json
  budget-dev2              G2: N, họ patch/hier, G ở cùng B (dev, quy tắc dừng theo chiều) → budget/selection_dev2.json
  budget-lock              khoá tiêu chí + lựa chọn cuối → budget/budget_prereg.json
  budget-test              G3 + G4: ảnh test mới, θ*(B) vs SPI α = 10 / α = 0,1 / IG cùng ngân sách, 8 D, metric ngoài — cần khoá
  budget-report            dựng lại INFID_BUDGET_REPORT.md
  budget-audit             kiểm toán BUDGET 1 (không GPU): P_select đầy đủ, hiệu chuẩn p của R, kiểm J thực nghiệm → INFID_BUDGET_AUDIT.md
Vòng BUDGET2 (mục budget2:, docs/PROTOCOL_BUDGET2.md; kết quả infid/budget2/ + infid/INFID_BUDGET2_REPORT.md):
  budget2-dev              nhánh A (SPI α × N nhỏ, B 256 / 1024) + nhánh B dev-only (hier, B 2048 / 4096); cổng so với incumbent
  budget2-lock             khoá cổng, ngân sách được kiểm, cỡ mẫu test (từ SD theo cặp trên dev)
  budget2-test             kiểm định trên ảnh mới — cần khoá
  budget2-report           dựng lại INFID_BUDGET2_REPORT.md
  budget2-audit            kiểm toán dev-only (không GPU): cổng g > 0 vs g > δ, cross-fit, kiểm J + mô phỏng độ phủ → INFID_BUDGET2_AUDIT.md
  budget2-freeze           đóng băng BUDGET2 dev-only → budget2/FROZEN.json; chặn budget2-lock / budget2-test
PATH_FAMILIES (docs/PATH_FAMILIES_AUDIT.md, docs/PATH_FAMILIES_PROTOCOL.md):
  pathfam-audit            Phase 0, CPU: tính chất 4 họ đường (dp, ln, cmix, smooth), toy Mệnh đề bậc ≤ 3/4/5, phân tích lại E10
                           → infid/pathfam/PATH_FAMILIES_CHECKS.md
  pathfam-plan             CPU: kế hoạch cuối + kiểm toán chi phí gradient (pipeline thật, hàm giả) → infid/pathfam/PATHFAM_PLAN.md
  pathfam-dev              Phase 1 dev: 20 cấu hình × 32 ảnh mới; chọn θ*_F theo J_rev; lực NI → infid/pathfam/selection_dev.json
  pathfam-lock             khoá lựa chọn, kiểm định, n test (từ lực dev) → infid/pathfam/pathfam_prereg.json
  pathfam-test             Phase 1 test (cần khoá) → infid/INFID_PATHFAM_REPORT.md;  pathfam-report: dựng lại báo cáo
  pathfam-verify           CPU: kiểm toán độc lập từ pairs.csv (J, g, NI / sup / TOST, Holm, khoá) → infid/pathfam/PATHFAM_VERIFY.md
  pathfam-diag             CPU: J ↔ metric ngoài (Kendall τ theo ảnh, mức B và mức μ, smooth) → infid/pathfam/PATHFAM_J_VS_EXTERNAL.md
  pathfam-theory           CPU: kiểm số định lý Phase 2 (diện tích Lévy, rev chính xác bậc 2, phương sai) → infid/pathfam/PATH_FAMILIES_THEORY_CHECKS.md
  pathfam-proofcheck       CPU: kiểm toán chứng minh Phase 2 (sympy + Monte Carlo) → infid/pathfam/PATH_FAMILIES_PROOF_CHECKS.md
  pathfam-signdiag         GPU, hậu kiểm: chẩn đoán có dấu / không dấu trên ảnh test đã khoá → infid/pathfam/PATHFAM_SIGNED_UNSIGNED.md
  pathfam-signdiag2        GPU, hậu kiểm vòng 2: thang đo β (trong / ngoài mẫu), so sánh cùng h-N-K, InsDel theo nền, completeness
                           → infid/pathfam/PATHFAM_SIGNDIAG2_AUDIT.md;  pathfam-signdiag2-report: dựng lại (CPU)
  pathfam-final-audit      CPU: β = (‖d‖/‖y‖)·cos (ảnh / chung / ngoài mẫu), mô phỏng luật K^{−1/2} completeness, khoá chẩn đoán
                           → infid/pathfam/PATHFAM_FINAL_AUDIT.md + DIAG_LOCK.json
  pathfam-freeze           đóng băng PATHFAM (sha256 artifact, seed, σ) → infid/pathfam/FROZEN.json; chặn dev / lock / test
Chạy lại sẽ bỏ qua phần đã có trong cache (cùng hash config).
"""

from __future__ import annotations

import argparse

from . import config as C

CMDS = ["toy", "toy-large", "toy-report", "pathscore", "pathscore-report", "pathscore-infid-decomp", "pathscore-infid-report",
        "quad-check", "quad-check-report", "sampling-infid-pilot", "infid-lock", "sampling-infid-e1a", "sampling-infid-e1b-dev",
        "sampling-infid-e1b-test", "sampling-infid-e2", "sampling-infid-report", "infid-posthoc", "infid-freeze",
        "budget-dev1", "budget-dev2", "budget-lock", "budget-test", "budget-report", "budget-audit",
        "budget2-dev", "budget2-lock", "budget2-test", "budget2-report", "budget2-audit", "budget2-freeze",
        "pathfam-audit", "pathfam-plan", "pathfam-dev", "pathfam-lock", "pathfam-test", "pathfam-report",
        "pathfam-verify", "pathfam-diag", "pathfam-theory", "pathfam-freeze",
        "pathfam-proofcheck", "pathfam-signdiag", "pathfam-signdiag-report",
        "pathfam-signdiag2", "pathfam-signdiag2-report", "pathfam-final-audit"]


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
    elif cmd == "infid-posthoc":
        from . import infid_posthoc as PH
        PH.run(cfg)
    elif cmd == "infid-freeze":
        from . import infid_posthoc as PH
        PH.freeze(cfg)
    elif cmd == "budget-audit":
        from . import budget_audit as BA
        BA.run(cfg)
    elif cmd == "pathfam-audit":
        from . import path_families as PF
        PF.run(cfg)
    elif cmd in ("pathfam-verify", "pathfam-diag", "pathfam-freeze"):
        from . import pathfam_verify as PV
        {"pathfam-verify": PV.verify, "pathfam-diag": PV.diag, "pathfam-freeze": PV.freeze}[cmd](cfg)
    elif cmd == "pathfam-theory":
        from . import path_theory as PT
        PT.run(cfg)
    elif cmd == "pathfam-proofcheck":
        from . import proof_check as PCK
        PCK.run(cfg)
    elif cmd == "pathfam-final-audit":
        from . import pathfam_final as PFF
        PFF.run(cfg)
    elif cmd in ("pathfam-signdiag2", "pathfam-signdiag2-report"):
        from . import signdiag2 as SD2
        SD2.run(ctx()) if cmd == "pathfam-signdiag2" else SD2.report(cfg)
    elif cmd in ("pathfam-signdiag", "pathfam-signdiag-report"):
        from . import signdiag as SD
        SD.run(ctx()) if cmd == "pathfam-signdiag" else SD.report(cfg)
    elif cmd.startswith("pathfam-"):
        from . import pathfam as PFM
        if cmd == "pathfam-plan":
            PFM.plan(cfg)
        elif cmd == "pathfam-lock":
            PFM.lock(cfg)
        elif cmd == "pathfam-report":
            PFM.write_report(cfg)
        else:
            PFM.run(ctx(), cmd[len("pathfam-"):])
    elif cmd in ("budget2-audit", "budget2-freeze"):
        from . import budget2_audit as B2A
        (B2A.run if cmd == "budget2-audit" else B2A.freeze)(cfg)
    elif cmd.startswith("budget2-"):
        from . import budget2 as B2
        if cmd == "budget2-lock":
            B2.lock(cfg)
        elif cmd == "budget2-report":
            B2.write_report(cfg)
        else:
            B2.run(ctx(), cmd[len("budget2-"):])
    elif cmd.startswith("budget-"):
        from . import budget_opt as BO
        if cmd == "budget-lock":
            BO.lock(cfg)
        elif cmd == "budget-report":
            BO.write_report(cfg)
        else:
            BO.run(ctx(), cmd[len("budget-"):])
    elif cmd == "sampling-infid-report":
        from . import sampling_infid as SI
        SI.write_report(cfg)
    else:
        from . import sampling_infid as SI
        SI.run(ctx(), cmd[len("sampling-infid-"):])


if __name__ == "__main__":
    main()
