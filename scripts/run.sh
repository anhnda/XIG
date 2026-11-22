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
#   cross-gate | cross-e4 | cross-e4c | cross-e4d | cross-report   từng bước (cross-report: DEBUG=1 cho thư mục debug)
#   cross-e4d-debug   E4d ở chế độ debug
#   e5-debug   E5 ở chế độ debug → runs_probe_debug/e5/E5_REPORT.md
#   e5         E5 → runs_probe/e5/E5_REPORT.md ;  e5-report: dựng lại báo cáo (DEBUG=1 cho thư mục debug)
#   e6-debug   E6 ở chế độ debug → runs_probe_debug/e6_pilot/E6_REPORT.md
#   e6         E6 → runs_probe/e6_<split>/E6_REPORT.md ;  e6-report: dựng lại báo cáo (DEBUG=1 cho thư mục debug)
#   e7-debug   E7 ở chế độ debug → runs_probe_debug/e7/E7_REPORT.md
#   e7         E7 → runs_probe/e7/E7_REPORT.md ;  e7-report: dựng lại báo cáo (DEBUG=1 cho thư mục debug)
#   e8-debug   E8 ở chế độ debug (2 ảnh PILOT, không đụng held-out) → runs_probe_debug/e8_pilot/E8_REPORT.md
#   e8         E8 xác nhận held-out → runs_probe/e8_confirm/E8_REPORT.md ;  e8-report: dựng lại (DEBUG=1 cho debug)
#   e9-debug   E9 ở chế độ debug → runs_probe_debug/e9/E9_REPORT.md
#   e9         E9 → runs_probe/e9/E9_REPORT.md ;  e9-report: dựng lại (DEBUG=1 cho debug)
#   ocs-debug  Output Convergence Study ở chế độ debug → runs_probe_debug/ocs_pilot/OCS_REPORT.md
#   ocs        Output Convergence Study → runs_probe/ocs_<split>/OCS_REPORT.md ;  ocs-report: dựng lại (DEBUG=1 cho debug)
#   e10-debug  E10 ở chế độ debug (1 ảnh pilot, không đụng ảnh fresh)
#   e10        E10 trên ảnh fresh, tiền đăng ký → runs_probe/e10_fresh/E10_REPORT.md ;  e10-report: dựng lại (DEBUG=1 cho debug)
#   e11-debug  E11 ở chế độ debug
#   e11        E11: LIG Phase 1 trên mẫu đường cho trước → runs_probe/e11_<split>/E11_REPORT.md ;  e11-report: dựng lại
#   e12-debug  E12 ở chế độ debug (1 ảnh pilot)
#   e12        E12 trên ảnh fresh, tiền đăng ký → runs_probe/e12_fresh/E12_REPORT.md ;  e12-report: dựng lại
#   P4 — Real-CNN Stochastic Path Sampling Benchmark:
#   sampling-test      unit test P4 + toàn bộ test cũ (CPU) → runs_probe/pilot/sampling/tests_report.txt
#   sampling-debug     P4 ở chế độ debug (2 ảnh, N = 8, K ∈ {2,4}) → runs_probe_debug/pilot/ATTR_PROBE_SAMPLING_REPORT.md
#   sampling-estimate  đo GPU, in ước tính từng khối + kế hoạch theo ngân sách (không chạy thí nghiệm)
#   sampling-pilot     P4 đầy đủ (ref → primary → khối phụ trong ngân sách) → runs_probe/pilot/ATTR_PROBE_SAMPLING_REPORT.md
#   sampling-report    dựng lại thống kê + đồ thị + báo cáo P4 từ CSV (DEBUG=1 cho thư mục debug)
#   sampling-quad      thí nghiệm C: cùng tập đường, M ∈ sampling.quad_check.M trên vài ảnh (DEBUG=1 cho debug)
#   PATHSCORE — eval riêng (tách khỏi P4): InsDel từng đường SPI, đánh giá chéo nền mask blur/black/white/grey/mean
#   pathscore          → runs_probe/pilot/pathscore/ + runs_probe/pilot/ATTR_PROBE_PATHSCORE_REPORT.md
#   pathscore-debug    bản debug (ít ảnh, N = 8) → runs_probe_debug/pilot/ATTR_PROBE_PATHSCORE_REPORT.md
#   pathscore-report   dựng lại báo cáo từ CSV (DEBUG=1 cho thư mục debug)
# env: CFG=config.yaml, EXTRA="--device cuda:1"
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:?stage: test|debug|estimate|conv|main|pilot|report|mech-debug|mech|mech-e1|mech-e23|mech-report|cross-debug|cross|cross-gate|cross-e4|cross-e4c|cross-e4d|cross-e4d-debug|cross-report|e5-debug|e5|e5-report|e6-debug|e6|e6-report|e7-debug|e7|e7-report|e8-debug|e8|e8-report|e9-debug|e9|e9-report|ocs-debug|ocs|ocs-report|e10-debug|e10|e10-report|e11-debug|e11|e11-report|e12-debug|e12|e12-report|sampling-test|sampling-debug|sampling-estimate|sampling-pilot|sampling-report|sampling-quad|pathscore|pathscore-debug|pathscore-report}
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
  cross|cross-gate|cross-e4|cross-e4c|cross-e4d) $PY "$STAGE" ;;
  cross-e4d-debug) $PY --debug cross-e4d ;;
  cross-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug cross-report; else $PY cross-report; fi ;;
  e5-debug) $PY --debug e5 ;;
  e5) $PY e5 ;;
  e5-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug e5-report; else $PY e5-report; fi ;;
  e6-debug) $PY --debug e6 ;;
  e6) $PY e6 ;;
  e6-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug e6-report; else $PY e6-report; fi ;;
  e7-debug) $PY --debug e7 ;;
  e7) $PY e7 ;;
  e7-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug e7-report; else $PY e7-report; fi ;;
  e8-debug) $PY --debug e8 ;;
  e8) $PY e8 ;;
  e8-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug e8-report; else $PY e8-report; fi ;;
  e9-debug) $PY --debug e9 ;;
  e9) $PY e9 ;;
  e9-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug e9-report; else $PY e9-report; fi ;;
  ocs-debug) $PY --debug ocs ;;
  ocs) $PY ocs ;;
  ocs-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug ocs-report; else $PY ocs-report; fi ;;
  e10-debug) $PY --debug e10 ;;
  e10) $PY e10 ;;
  e10-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug e10-report; else $PY e10-report; fi ;;
  e11-debug) $PY --debug e11 ;;
  e11) $PY e11 ;;
  e11-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug e11-report; else $PY e11-report; fi ;;
  e12-debug) $PY --debug e12 ;;
  e12) $PY e12 ;;
  e12-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug e12-report; else $PY e12-report; fi ;;
  sampling-test)
    OUTD=$(python -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); print((c.get('local') or {}).get('out','runs_probe'))")/pilot/sampling
    mkdir -p "$OUTD"
    { echo "# P4 sampling-test — $(date -Iseconds)"; python -c "import torch,torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)";
      python -m pytest -q tests/test_sampling.py tests/test_pathscore.py tests/test_units.py 2>&1; } | tee "$OUTD/tests_report.txt"
    exit "${PIPESTATUS[0]}" ;;
  sampling-debug) $PY --debug sampling ;;
  sampling-estimate) $PY sampling-estimate ;;
  sampling-pilot) $PY sampling ;;
  sampling-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug sampling-report; else $PY sampling-report; fi ;;
  sampling-quad) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug sampling-quad; else $PY sampling-quad; fi ;;
  pathscore) $PY pathscore ;;
  pathscore-debug) $PY --debug pathscore ;;
  pathscore-report) if [ "${DEBUG:-0}" = "1" ]; then $PY --debug pathscore-report; else $PY pathscore-report; fi ;;
  *) echo "stage không biết: $STAGE"; exit 1 ;;
esac