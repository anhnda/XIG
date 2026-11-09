#!/usr/bin/env bash
# Re-run every revision experiment from gtsegs_ijcv.mat alone (images + masks are inside it).
#
#   bash scripts/run_all.sh /path/to/gtsegs_ijcv.mat <stage> [out_dir]
#
# stages (run in this order; check the output of `calib` before going on):
#   test    pytest on toy models (seconds)
#   calib   IG / Guided IG / dnoise on 200 images → IG and Guided IG must match the slide
#           (Ins−Del 0.282 / IoU 0.423 and 0.463 / 0.491); if not, retry with BASE=gray
#   dnoise  sweep of the unknown dnoise settings (find the one matching 0.465 / 0.513)
#   b1      impact of the Δf bug on IDGI / LIG-IDIG / LIG (first 50 images, slow: LIG)
#   p0      Phase-1 μ: closed form vs Adam vs legacy
#   p1      fixed dnoise path, change the estimator (decides branch A or B)
#   p2      path families at equal gradient budget (IG, dnoise, pixel paths, SPI)
#   all     everything above in order
#
# env: BASE=black|gray (path + ins/del baseline, default black), MODEL=resnet50, SEEDS="0 1 2"
set -euo pipefail

MAT=${1:?usage: run_all.sh /path/to/gtsegs_ijcv.mat <stage> [out_dir]}
STAGE=${2:?stage: test|calib|dnoise|b1|p0|p1|p2|all}
OUT=${3:-runs}
BASE=${BASE:-black}
MODEL=${MODEL:-resnet50}
SEEDS=${SEEDS:-"0 1 2"}

cd "$(dirname "$0")/.."
mkdir -p "$OUT"
IDX="$OUT/seg200.json"                      # fixed list of 200 images, created on first use
DATA="--data seg --seg-mat $MAT --index-file $IDX --model $MODEL"
METRICS="insdel_${BASE} insdel_blur overlap iou pointing checker"

run_test() { python -m pytest -q tests; }

run_calib() {
  python -m experiments.run $DATA --n 200 --baseline "$BASE" --insdel "$BASE" blur \
    --methods ig guided_ig legacy_guided_ig dnoise pixel_random block14 \
    --seeds $SEEDS --out "$OUT/calib_${BASE}.jsonl"
  python -m experiments.summarize "$OUT/calib_${BASE}.jsonl" --ref ig --metrics $METRICS \
    | tee "$OUT/calib_${BASE}.md"
}

run_dnoise() {
  python -m experiments.run $DATA --n 200 --baseline "$BASE" --insdel "$BASE" blur \
    --methods dnoise "dnoise:spread=0.25" "dnoise:spread=0.75" "dnoise:spread=1" \
              "dnoise:thresh=stratified" "dnoise:thresh=random" "dnoise:tile=4" "dnoise:tile=16" \
              "dnoise:noise=0.02" "dnoise:noise=0.05" "dnoise:per_channel=true" "dnoise:est=mid" \
    --seeds $SEEDS --out "$OUT/dnoise_${BASE}.jsonl"
  python -m experiments.summarize "$OUT/dnoise_${BASE}.jsonl" --ref dnoise --metrics $METRICS \
    | tee "$OUT/dnoise_${BASE}.md"
}

run_b1() {
  # gray = the baseline the old batch benchmark actually used
  python -m experiments.run $DATA --n 50 --baseline gray --insdel gray \
    --methods legacy_idig "legacy_idig:fix_df=1" idgi \
              legacy_lig_idig "legacy_lig_idig:fix_df=1" \
              legacy_lig "legacy_lig:fix_df=1" \
    --seeds 0 --out "$OUT/b1.jsonl"
  python -m experiments.summarize "$OUT/b1.jsonl" --ref legacy_idig \
    --metrics insdel_gray overlap iou | tee "$OUT/b1.md"
}

run_p0() {
  python -m experiments.p0_mu_projection $DATA --n 200 --baseline "$BASE" --out "$OUT/p0_mu.jsonl"
}

run_p1() {
  python -m experiments.p1_estimators $DATA --n 200 --baseline "$BASE" --insdel "$BASE" blur \
    --path dnoise --seeds $SEEDS --out "$OUT/p1.jsonl"
  python -m experiments.summarize "$OUT/p1.jsonl" --ref "dnoise|gl" \
    --metrics $METRICS rel_err_vs_gl spearman_vs_gl | tee "$OUT/p1.md"
}

run_p2() {
  python -m experiments.run $DATA --n 200 --baseline "$BASE" --insdel "$BASE" blur \
    --methods "ig:N=24" "ig:N=50" "ig:N=200" "ig:N=24,est=gl,Q=2" guided_ig \
              dnoise pixel_random pixel_strat block14 "dnoise:K=2,N=12" \
              "spi:alpha=0.01" "spi:alpha=0.1" "spi:alpha=1" "spi:alpha=10" \
    --seeds $SEEDS --out "$OUT/p2.jsonl"
  python -m experiments.summarize "$OUT/p2.jsonl" --ref "ig:N=50" --metrics $METRICS \
    | tee "$OUT/p2.md"
}

case "$STAGE" in
  test)   run_test ;;
  calib)  run_calib ;;
  dnoise) run_dnoise ;;
  b1)     run_b1 ;;
  p0)     run_p0 ;;
  p1)     run_p1 ;;
  p2)     run_p2 ;;
  all)    run_test; run_calib; run_dnoise; run_b1; run_p0; run_p1; run_p2 ;;
  *)      echo "unknown stage '$STAGE'"; exit 1 ;;
esac
