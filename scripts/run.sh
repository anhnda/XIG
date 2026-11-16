#!/usr/bin/env bash
# Chạy theo giai đoạn của protocol.
#
#   bash scripts/run.sh <stage> [config.yaml]
#
# Giai đoạn 0
#   test        pytest U1–U9 trên toy (CPU). Phải đạt hết trước mọi lần chạy ResNet.
#   test_slow   thêm các test chậm (U4 phân biệt 251/720 vs 252/720, bias của node=mid)
#   power       power analysis từ JSONL cũ (local.old_jsonl) → ghi n vào config TRƯỚC khi lock
#   lock        khoá config
#   index       chọn n ảnh mới từ gtsegs_ijcv (không thuộc 200 ảnh cũ)
#   smoke       2 ảnh, thang rút gọn tự nhiên của nhóm stage1a, ghi vào runs_smoke/ (kiểm đường ống)
# Giai đoạn 1A
#   s1a         attr + converge (tự thêm mức nếu chưa hội tụ) + eval + gate1   → runs/imagenet/gate1.md
#   s1a_post    tính lại energy + hội tụ + gate1 từ attribution đã có (không GPU)
#   explore     khám phá α = 1e3, 1e4, 1e5 (ngoài protocol)
#   s1a_bq      (tuỳ chọn) SPI_BQ 8×24, α ∈ {10, 100}
# Giai đoạn 1B (chỉ khi gate1 = POSITIVE)
#   s1b         attr + converge + eval --line + collapse                       → runs/imagenet/collapse.md
# Pet
#   pet_train   huấn luyện ResNet-18 cho mọi seed trong SEEDS (mặc định "0 1 2")
#   pet_models  báo cáo accuracy / đồng thuận / ECE / D_m (cần pet_index: chạy pet_index trước)
#   pet_index   chọn 50 ảnh test Pet
#   pet_s1a     Cửa 1 trên Pet, seed PET_SEED (mặc định 0)
#   s2          Giai đoạn 2: stage2 cho mọi seed + ensemble                    → runs/pet/ensemble.md
#
# env: SEEDS="0 1 2", PET_SEED=0, OUT=runs (ghi đè local.out), EXTRA="--device cuda:1"
set -euo pipefail

STAGE=${1:?stage: test|test_slow|power|lock|index|smoke|s1a|s1a_post|explore|s1a_bq|s1b|pet_train|pet_index|pet_models|pet_s1a|s2}
CFG=${2:-config.yaml}
SEEDS=${SEEDS:-"0 1 2"}
PET_SEED=${PET_SEED:-0}
EXTRA=${EXTRA:-}
OUTARG=""
[ -n "${OUT:-}" ] && OUTARG="--out $OUT"

cd "$(dirname "$0")/.."
PY="python -m spi_study --config $CFG"

# attr → converge --extend, lặp lại khi converge thêm mức (mã 3)
attr_loop() {
  local G=$1; shift
  for _ in 1 2 3 4 5; do
    $PY attr --group "$G" $OUTARG $EXTRA "$@"
    set +e
    $PY converge --group "$G" --extend $OUTARG $EXTRA "$@"
    local rc=$?
    set -e
    if [ $rc -eq 3 ]; then echo "[run] đã thêm mức thang, chạy lại attr"; continue; fi
    return $rc
  done
}

case "$STAGE" in
  test)       python -m pytest -q tests -m "not slow" ;;
  test_slow)  python -m pytest -q tests ;;
  power)      $PY power $OUTARG ;;
  lock)       $PY lock ;;
  index)      $PY index $OUTARG $EXTRA ;;
  smoke)
    O=${OUT:-runs_smoke}
    $PY index --out "$O" $EXTRA
    $PY attr --group stage1a --limit 2 --out "$O" $EXTRA
    set +e; $PY converge --group stage1a --limit 2 --out "$O" $EXTRA; set -e
    $PY eval --group stage1a --limit 2 --out "$O" $EXTRA
    echo "[smoke] xem $O/imagenet/compute.jsonl và metrics.jsonl (residual, thời gian, backward)" ;;
  s1a)
    attr_loop stage1a
    $PY eval --group stage1a $OUTARG $EXTRA
    $PY energy --group stage1a $OUTARG $EXTRA
    $PY gate1 $OUTARG $EXTRA ;;
  s1a_post)   # chỉ tính lại báo cáo 1A từ attribution đã có (CPU): energy, hội tụ, gate1
    $PY energy --group stage1a $OUTARG $EXTRA
    set +e; $PY converge --group stage1a $OUTARG $EXTRA; set -e
    $PY gate1 $OUTARG $EXTRA ;;
  explore)    # khám phá: SPI α = 1e3, 1e4, 1e5 (bình nguyên hay chuyển tiếp về IG)
    attr_loop explore_alpha
    $PY eval --group explore_alpha $OUTARG $EXTRA
    $PY energy --group explore_alpha $OUTARG $EXTRA ;;
  s1a_bq)
    $PY attr --group stage1a_bq $OUTARG $EXTRA
    $PY eval --group stage1a_bq $OUTARG $EXTRA ;;
  s1b)
    attr_loop stage1b
    $PY eval --group stage1b --line $OUTARG $EXTRA
    $PY energy --group stage1b --line $OUTARG $EXTRA
    $PY collapse $OUTARG $EXTRA ;;
  pet_train)
    for s in $SEEDS; do $PY pet-train --seed "$s" $OUTARG $EXTRA; done ;;
  pet_index)  $PY index --setting pet $OUTARG $EXTRA ;;
  pet_models) $PY pet-models $OUTARG $EXTRA ;;
  pet_s1a)
    attr_loop stage1a --setting pet --pet-seed "$PET_SEED"
    $PY eval --group stage1a --setting pet --pet-seed "$PET_SEED" $OUTARG $EXTRA
    $PY energy --group stage1a --setting pet --pet-seed "$PET_SEED" $OUTARG $EXTRA
    $PY gate1 --setting pet --pet-seed "$PET_SEED" $OUTARG $EXTRA ;;
  s2)
    for s in $SEEDS; do
      attr_loop stage2 --setting pet --pet-seed "$s"
      $PY eval --group stage2 --line --setting pet --pet-seed "$s" $OUTARG $EXTRA
    done
    $PY ensemble $OUTARG $EXTRA ;;
  *) echo "stage không biết: $STAGE"; exit 1 ;;
esac