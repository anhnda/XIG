#!/usr/bin/env bash
# bash scripts/run.sh <stage>
#   test       unit test (CPU, vài giây) → tests_report.txt (báo cáo sẽ chép vào)
#   debug      toàn pipeline ở chế độ debug (vài ảnh, ít bước) → runs_probe_debug/pilot/ATTR_PROBE_PILOT_REPORT.md
#   estimate   đo tốc độ GPU, in ước tính thời gian (không chạy thí nghiệm)
#   conv       P0 hội tụ
#   main       P1 + P2 + P3
#   pilot      conv → main → report   (đầy đủ)
#   report     dựng lại đồ thị + báo cáo từ CSV đã có (DEBUG=1 cho thư mục debug)
#   mech-debug E1–E3 ở chế độ debug → runs_probe_debug/mechanism/MECHANISM_REPORT.md
#   mech       E1 → E2+E3 → MECHANISM_REPORT.md   (runs_probe/mechanism/)
#   mech-e1 | mech-e23 | mech-report   từng bước (mech-report: DEBUG=1 cho thư mục debug)
#   cross-debug  E1′ + E4 ở chế độ debug → runs_probe_debug/cross/CROSS_REPORT.md
#   cross        E1′ gate → E4 → E4c → CROSS_REPORT.md   (runs_probe/cross/)
#   cross-gate | cross-e4 | cross-e4c | cross-report   từng bước (cross-report: DEBUG=1 cho thư mục debug)
# env: CFG=config.yaml, EXTRA="--device cuda:1"
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:?stage: test|debug|estimate|conv|main|pilot|report|mech-debug|mech|mech-e1|mech-e23|mech-report|cross-debug|cross|cross-gate|cross-e4|cross-e4c|cross-report}
CFG=${CFG:-config.yaml}
EXTRA=${EXTRA:-}
PY="python -m probe --config $CFG $EXTRA"
case "$STAGE" in
  test)     python -m pytest -q tests 2>&1 | tee tests_report.txt ;;
  debug)    $PY --debug all ;;
  estimate) $PY estimate ;;
  conv)     $PY conv ;;
  main)     $PY main ;;
  pilot)    $PY all ;;
  report)   if [ "${DEBUG:-0}" = "1" ]; then $PY --debug report; else $PY report; fi ;;
  mech-debug) $PY --debug mech ;;
  mech|mech-e1|mech-e23) $PY "$STAGE" ;;
  mech-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug mech-report; else $PY mech-report; fi ;;
  cross-debug) $PY --debug cross ;;
  cross|cross-gate|cross-e4|cross-e4c) $PY "$STAGE" ;;
  cross-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug cross-report; else $PY cross-report; fi ;;
  *) echo "stage không biết: $STAGE"; exit 1 ;;
esac