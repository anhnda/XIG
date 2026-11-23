#!/usr/bin/env bash
# bash scripts/run.sh <lệnh>   — proposal "Lấy mẫu đường SPI theo infidelity" (kết quả runs_probe/pilot/infid/)
#   test                     unit test (CPU) → runs_probe/pilot/infid/tests_report.txt
#   toy | toy-large          E-toy nhỏ (oracle Phụ lục) | lớn (d = 256, T9, Cov(H), coverage) — không cần mô hình
#   pathscore                (đầu vào E0) bản đồ từng đường SPI → runs_probe/pilot/pathscore/
#   pathscore-infid-decomp   E0 (không gradient mới)
#   quad-check               E-quadrature / C3
#   sampling-infid-pilot     E1a pilot
#   infid-lock               khoá protocol (sau khi điền K_main, δ₁–δ₃, rule/M)
#   sampling-infid-e1a | sampling-infid-e1b-dev | sampling-infid-e1b-test | sampling-infid-e2   (cần khoá)
#   *-report                 dựng lại báo cáo từ CSV: toy-report, pathscore-report, pathscore-infid-report, quad-check-report,
#                            sampling-infid-report
#   <lệnh>-debug             bản debug của mọi lệnh trên (ít ảnh, ít bước; ghi vào runs_probe_debug/)
# env: CFG=config.yaml, EXTRA="--device cuda:1"
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:?lệnh: test|toy|toy-large|pathscore|pathscore-infid-decomp|quad-check|sampling-infid-pilot|infid-lock|sampling-infid-e1a|sampling-infid-e1b-dev|sampling-infid-e1b-test|sampling-infid-e2|*-report|*-debug}
CFG=${CFG:-config.yaml}
EXTRA=${EXTRA:-}
PY="python -m probe --config $CFG $EXTRA"
case "$STAGE" in
  test)
    OUTD=$(python -c "import yaml; c=yaml.safe_load(open('$CFG')); print((c.get('local') or {}).get('out','runs_probe'))")/pilot/infid
    mkdir -p "$OUTD"
    { echo "# test — $(date -Iseconds)"; python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())";
      python -m pytest -q tests 2>&1; } | tee "$OUTD/tests_report.txt"
    exit "${PIPESTATUS[0]}" ;;
  *-debug) $PY --debug "${STAGE%-debug}" ;;
  *) $PY "$STAGE" ;;
esac
