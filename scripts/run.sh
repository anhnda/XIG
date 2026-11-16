#!/usr/bin/env bash
# bash scripts/run.sh <stage>
#   test     unit test trên toy (CPU, < 1 phút)
#   debug    toàn bộ A, B, C, sanity ở chế độ debug (2 ảnh, ít bước) → runs_probe_debug/pilot/report.md
#   info     danh sách ảnh và số eval mỗi biến thể
#   a|b|c|sanity   từng thí nghiệm trên pilot (in ước tính thời gian trước khi chạy)
#   pilot    a → b → c → sanity trên pilot
#   lock     khoá config (REASON="...")
#   confirm  a → b → c → sanity trên 50 ảnh xác nhận (cần lock)
#   report   gom lại report.md (SPLIT=pilot|confirm)
# env: CFG=config.yaml, EXTRA="--device cuda:1"
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:?stage: test|debug|info|a|b|c|sanity|pilot|lock|confirm|report}
CFG=${CFG:-config.yaml}
EXTRA=${EXTRA:-}
PY="python -m probe --config $CFG $EXTRA"
case "$STAGE" in
  test)    python -m pytest -q tests ;;
  debug)   $PY --debug all ;;
  info)    $PY info ;;
  a|b|c|sanity) $PY "$STAGE" ;;
  pilot)   $PY all ;;
  lock)    $PY lock --reason "${REASON:?REASON=\"...\" bắt buộc}" ;;
  confirm) $PY --split confirm all ;;
  report)  $PY --split "${SPLIT:-pilot}" report ;;
  *) echo "stage không biết: $STAGE"; exit 1 ;;
esac
