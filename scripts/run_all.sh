#!/usr/bin/env bash
# Path-attribution experiments (LIG dropped). Images + masks come from gtsegs_ijcv.mat.
#
#   bash scripts/run_all.sh /path/to/gtsegs_ijcv.mat <stage> [out_dir]
#
# stages (in this order):
#   test        pytest on toy models (seconds) — run first, nothing has been executed yet
#   calib       IG / Guided IG / dnoise on 200 images (protocol check vs the slides)
#   p1          estimator audit: same path, left / mid / random / random_cv vs Gauss–Legendre
#               (dnoise, SPI α=0.01 — does the estimator inherit the Shapley limit? — and SPI α=1)
#   p2          path families at an equal gradient budget (≈192 backward) + KernelSHAP,
#               with ins/del, localisation, infidelity, approximate ROAD
#   sigma       E6: smooth IG with σ in pixel units, black vs gray baseline
#   alpha       E1: within-α spread vs between-α gap, against saturation
#   tpa_calib   TPA sweep on the calibration slice (images 0–49), fidelity only
#   tpa_select  choose α / ε per family by held-out fidelity → $OUT/tpa_chosen.json
#   tpa_test    chosen variants on the test slice (images 50–199), all metrics
#   agree       E5: Kendall τ between metric rankings (p2 and tpa_test)
#   nlp         E7 on SST-2 (needs transformers + datasets; MODEL_NLP to change the model)
#   all         everything above in order
#
# env: BASE=black|gray (path + ins/del baseline, default black), MODEL=resnet50, SEEDS="0 1 2",
#      PARTS (TPA fit partitions, default slic:64 — needs scikit-image; use grid:28 without it)
set -euo pipefail

MAT=${1:?usage: run_all.sh /path/to/gtsegs_ijcv.mat <stage> [out_dir]}
STAGE=${2:?stage: test|calib|p1|p2|sigma|alpha|tpa_calib|tpa_select|tpa_test|agree|nlp|all}
OUT=${3:-runs}
BASE=${BASE:-black}
MODEL=${MODEL:-resnet50}
SEEDS=${SEEDS:-"0 1 2"}
PARTS=${PARTS:-"slic:64"}
MODEL_NLP=${MODEL_NLP:-"textattack/bert-base-uncased-SST-2"}

cd "$(dirname "$0")/.."
mkdir -p "$OUT"
IDX="$OUT/seg200.json"                      # fixed list of 200 images, created on first use
DATA="--data seg --seg-mat $MAT --index-file $IDX --model $MODEL"
METRICS="insdel_${BASE} insdelp_${BASE} insdel_blur overlap iou pointing infd road"
TAG="${MODEL}_${BASE}"

run_test() { python -m pytest -q tests; }

run_calib() {
  python -m experiments.run $DATA --n 200 --baseline "$BASE" --insdel "$BASE" blur \
    --methods ig guided_ig legacy_guided_ig dnoise pixel_random block14 \
    --seeds $SEEDS --out "$OUT/calib_${TAG}.jsonl"
  python -m experiments.summarize "$OUT/calib_${TAG}.jsonl" --ref ig \
    --metrics insdel_${BASE} insdel_blur overlap iou pointing checker | tee "$OUT/calib_${TAG}.txt"
}

run_p1() {
  for P in dnoise "spi:alpha=0.01" "spi:alpha=1"; do
    F=$(echo "$P" | tr ':=' '__')
    python -m experiments.p1_estimators $DATA --n 200 --baseline "$BASE" --insdel "$BASE" \
      --path "$P" --seeds $SEEDS --out "$OUT/p1_${F}_${TAG}.jsonl"
    python -m experiments.summarize "$OUT/p1_${F}_${TAG}.jsonl" --ref "${P}|gl" \
      --metrics insdel_${BASE} iou residual rel_err_vs_gl spearman_vs_gl | tee "$OUT/p1_${F}_${TAG}.txt"
  done
}

run_p2() {
  python -m experiments.run $DATA --n 200 --baseline "$BASE" --insdel "$BASE" blur \
    --methods "ig:N=50" "ig:N=192,est=mid" "idgi:N=192,node=mid" guided_ig legacy_blurig dnoise \
              "smooth_ig:N=24,K=8,sigma=0.3,unit=abs" "noise_only:N=24,K=8,sigma=0.3,unit=abs" \
              "spi:alpha=0.1,K=8,est=mid" "spi:alpha=1,K=8,est=mid" "spi:alpha=10,K=8,est=mid" \
              "kernelshap:parts=${PARTS},M=256" \
    --infd-n 64 --road --seeds $SEEDS --out "$OUT/p2_${TAG}.jsonl"
  python -m experiments.summarize "$OUT/p2_${TAG}.jsonl" --ref "ig:N=50" --metrics $METRICS residual \
    | tee "$OUT/p2_${TAG}.txt"
}

run_sigma() {
  for B in black gray; do
    python -m experiments.run $DATA --n 100 --baseline "$B" --insdel "$B" \
      --methods "smooth_ig:N=24,K=8,unit=abs,sigma=0.05" "smooth_ig:N=24,K=8,unit=abs,sigma=0.1" \
                "smooth_ig:N=24,K=8,unit=abs,sigma=0.2" "smooth_ig:N=24,K=8,unit=abs,sigma=0.4" \
                "smooth_ig:N=24,K=8,unit=abs,sigma=0.8" "smooth_ig:N=24,K=8,unit=delta,sigma=0.3" \
                "smooth_ig:N=24,K=8,unit=delta,sigma=1.0" \
      --seeds $SEEDS --out "$OUT/sigma_${MODEL}_${B}.jsonl"
    python -m experiments.summarize "$OUT/sigma_${MODEL}_${B}.jsonl" --metrics insdel_${B} iou \
      | tee "$OUT/sigma_${MODEL}_${B}.txt"
  done
}

run_alpha() {
  python -m experiments.alpha_disagreement $DATA --n 200 --baseline "$BASE" \
    --alphas 0.1 1 10 100 --K 16 --out "$OUT/alpha_${TAG}.jsonl" --save-maps "$OUT/alpha_maps_${TAG}" \
    | tee "$OUT/alpha_${TAG}.txt"
}

run_tpa_calib() {
  python -m experiments.tpa_sweep $DATA --start 0 --n 50 --baseline "$BASE" --parts "$PARTS" \
    --alphas 0.1 1 10 --eps 1 0.1 0.01 0 --kappas 1 --seeds 0 --out "$OUT/tpa_calib_${TAG}.jsonl"
}

run_tpa_select() {
  python -m experiments.select_tpa "$OUT/tpa_calib_${TAG}.jsonl" --out "$OUT/tpa_chosen_${TAG}.json" \
    | tee "$OUT/tpa_select_${TAG}.txt"
}

run_tpa_test() {
  python -m experiments.tpa_sweep $DATA --start 50 --n 150 --baseline "$BASE" --parts "$PARTS" \
    --alphas 0.1 1 10 --eps 1 0.1 0.01 0 --kappas 1 --seeds $SEEDS \
    --variants-file "$OUT/tpa_chosen_${TAG}.json" --eval --insdel "$BASE" blur --infd-n 64 --road \
    --out "$OUT/tpa_test_${TAG}.jsonl"
  REF=$(python -c "import json;print([v for v in json.load(open('$OUT/tpa_chosen_${TAG}.json')) if v.endswith('|uniform')][0])")
  python -m experiments.summarize "$OUT/tpa_test_${TAG}.jsonl" --ref "$REF" \
    --metrics hold_rel tr_rel $METRICS | tee "$OUT/tpa_test_${TAG}.txt"
}

run_agree() {
  python -m experiments.metric_agreement "$OUT/p2_${TAG}.jsonl" \
    --metrics insdel_${BASE} insdelp_${BASE} insdel_blur iou overlap pointing infd road \
    | tee "$OUT/agree_p2_${TAG}.txt"
  if [ -f "$OUT/tpa_test_${TAG}.jsonl" ]; then
    python -m experiments.metric_agreement "$OUT/tpa_test_${TAG}.jsonl" \
      --metrics hold_rel tr_rel insdel_${BASE} iou infd road | tee "$OUT/agree_tpa_${TAG}.txt"
  fi
}

run_nlp() {
  M_TAG=$(echo "$MODEL_NLP" | tr '/' '_')
  python -m experiments.nlp.run_nlp --dataset sst2 --model "$MODEL_NLP" --n 872 --max-len 128 \
    --methods "ig:N=50" "ig:N=192" "spi:alpha=0.1" "spi:alpha=1" "spi:alpha=10" \
              "smooth:sigma=0.05" "smooth:sigma=0.1" "smooth:sigma=0.2" \
              "tpa:alpha=1,mode=uniform" "tpa:alpha=1,mode=gibbs,eps=0.1" \
              "tpa:alpha=1,mode=core,eps=0.1" "tpa:alpha=1,mode=r,eps=0.1,kappa=1" "kernelshap:M=256" \
    --seeds 0 --ref-cache "$OUT/nlp_ref_sst2_${M_TAG}" --out "$OUT/nlp_sst2_${M_TAG}.jsonl"
  python -m experiments.summarize "$OUT/nlp_sst2_${M_TAG}.jsonl" --ref "ig:N=50" \
    --metrics ref_spearman ref_kendall ref_l1 ref_top20_overlap comp suff hold_rel residual \
    | tee "$OUT/nlp_sst2_${M_TAG}.txt"
}

case "$STAGE" in
  test)       run_test ;;
  calib)      run_calib ;;
  p1)         run_p1 ;;
  p2)         run_p2 ;;
  sigma)      run_sigma ;;
  alpha)      run_alpha ;;
  tpa_calib)  run_tpa_calib ;;
  tpa_select) run_tpa_select ;;
  tpa_test)   run_tpa_test ;;
  agree)      run_agree ;;
  nlp)        run_nlp ;;
  all)        run_test; run_calib; run_p1; run_p2; run_sigma; run_alpha
              run_tpa_calib; run_tpa_select; run_tpa_test; run_agree; run_nlp ;;
  *)          echo "unknown stage '$STAGE'"; exit 1 ;;
esac
