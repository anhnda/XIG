#!/usr/bin/env bash
# Path-attribution experiments (LIG dropped). Images + masks come from gtsegs_ijcv.mat.
#
#   bash scripts/run_all.sh /path/to/gtsegs_ijcv.mat <stage> [out_dir]
#
# stages (in this order):
#   test        pytest on toy models (seconds)
#   calib       IG / Guided IG / dnoise on 200 images (protocol check vs the slides)
#   p1          estimator audit: same path, left / mid / random / random_cv vs Gauss–Legendre
#   p2          path families at an equal gradient budget (≈192 backward) + KernelSHAP,
#               ins/del, localisation, infidelity, approximate ROAD; attributions saved
#   infd        re-evaluate p2's saved attributions: infidelity per 16 px cell, σ = 0.3 and 1.0
#   axis        SPI α axis with the EXACT Dirichlet sampler: α = 10 … 1e5 vs IG (α = ∞)
#   tpa_calib   TPA sweep on images 0–49 for each budget split (K, N) ∈ {8·24, 16·12, 32·6}, α ∈ {1, 3, 10, 30, 100}
#   tpa_select  choose α / ε per family and per (K, N) by held-out fidelity
#   alpha       E1: within-α spread vs debiased between-α gap (run with MODEL=vit_b_16, swin_b too)
#   tpa_test    chosen variants on images 50–199, all metrics   (KN="8 24" picks the split)
#   sigma       E6: smooth IG with σ in pixel units, black vs gray baseline
#   agree       E5: Kendall τ between metric rankings (p2, axis, tpa_test)
#   nlp         E7 on SST-2 (needs transformers + datasets; MODEL_NLP to change the model)
#   all         everything above in order
# analysis stages (not in 'all'):
#   theory      CPU toy check of SPI = IG + C/(α+1) + O(1/α²)   (NPATH paths per α, default 20000)
#   theory_sweep  deterministic: cos(correction, Shapley − IG) over NSEED random toys (default 200)
#   theory_d    same check as 'theory' for d = 5, 20, 80 (does the expansion degrade with d?)
#   analyze     why SPI ranks high: f along line / SPI paths + saved axis attributions (after axis)
# Notes: SPI uses the exact Dirichlet sampler (the old stick sampler capped atoms at 256, so
# α ≥ 100 in p2 is not the stated α — the 'axis' stage replaces those rows).
#
# env: BASE=black|gray (path + ins/del baseline, default black), MODEL=resnet50, SEEDS="0 1 2",
#      PARTS (TPA fit partitions, default slic:64 — needs scikit-image; use grid:28 without it),
#      KN (budget split used by tpa_test, default "8 24"),
#      NIMG (override the number of images of any stage: smoke test with NIMG=2 SEEDS=0)
set -euo pipefail

MAT=${1:?usage: run_all.sh /path/to/gtsegs_ijcv.mat <stage> [out_dir]}
STAGE=${2:?stage: test|calib|p1|p2|infd|axis|tpa_calib|tpa_select|alpha|tpa_test|sigma|agree|nlp|all|theory|theory_sweep|theory_d|analyze}
OUT=${3:-runs}
BASE=${BASE:-black}
MODEL=${MODEL:-resnet50}
SEEDS=${SEEDS:-"0 1 2"}
PARTS=${PARTS:-"slic:64"}
MODEL_NLP=${MODEL_NLP:-"textattack/bert-base-uncased-SST-2"}
KN=${KN:-"8 24"}
KNS=("8 24" "16 12" "32 6")
ALPHAS_TPA="1 3 10 30 100"

cd "$(dirname "$0")/.."
mkdir -p "$OUT"
IDX="$OUT/seg200.json"                      # fixed list of 200 images, created on first use
DATA="--data seg --seg-mat $MAT --index-file $IDX --model $MODEL"
METRICS="insdelp_${BASE} insdelp_blur insdel_${BASE} overlap iou pointing road"
TAG="${MODEL}_${BASE}"

run_test() { python -m pytest -q tests; }

run_calib() {
  python -m experiments.run $DATA --n ${NIMG:-200} --baseline "$BASE" --insdel "$BASE" blur \
    --methods ig guided_ig legacy_guided_ig dnoise pixel_random block14 \
    --seeds $SEEDS --out "$OUT/calib_${TAG}.jsonl"
  python -m experiments.summarize "$OUT/calib_${TAG}.jsonl" --ref ig \
    --metrics insdel_${BASE} insdelp_${BASE} insdel_blur insdelp_blur overlap iou pointing checker \
    | tee "$OUT/calib_${TAG}.txt"
}

run_p1() {
  for P in dnoise "spi:alpha=0.01" "spi:alpha=1"; do
    F=$(echo "$P" | tr ':=' '__')
    python -m experiments.p1_estimators $DATA --n ${NIMG:-200} --baseline "$BASE" --insdel "$BASE" \
      --path "$P" --seeds $SEEDS --out "$OUT/p1_${F}_${TAG}.jsonl"
    python -m experiments.summarize "$OUT/p1_${F}_${TAG}.jsonl" --ref "${P}|gl" \
      --metrics insdel_${BASE} iou residual rel_err_vs_gl spearman_vs_gl | tee "$OUT/p1_${F}_${TAG}.txt"
  done
}

run_p2() {
  # equal budget ≈192 backward + cheap variants at 24/50 backward; full attributions saved
  python -m experiments.run $DATA --n ${NIMG:-200} --baseline "$BASE" --insdel "$BASE" blur \
    --methods "ig:N=50" "ig:N=192,est=mid" "idgi:N=192,node=mid" guided_ig legacy_blurig dnoise \
              "dnoise:K=8,est=mid" "spi:alpha=10,K=1,est=mid" \
              "smooth_ig:N=24,K=8,sigma=0.3,unit=abs" "noise_only:N=24,K=8,sigma=0.3,unit=abs" \
              "spi:alpha=0.1,K=8,est=mid" "spi:alpha=1,K=8,est=mid" "spi:alpha=10,K=8,est=mid" \
              "kernelshap:parts=${PARTS},M=256" \
    --infd-n 64 --infd-mode delta --infd-cell 16 --road --seeds $SEEDS \
    --save-attr "$OUT/attr_p2_${TAG}" --out "$OUT/p2_${TAG}.jsonl"
  python -m experiments.summarize "$OUT/p2_${TAG}.jsonl" --ref "ig:N=50" --metrics $METRICS residual \
    | tee "$OUT/p2_${TAG}.txt"
}

run_infd() {
  for S in 0.3 1.0; do
    python -m experiments.reeval $DATA --n ${NIMG:-200} --attr-dir "$OUT/attr_p2_${TAG}" --insdel \
      --infd-n 64 --infd-cell 16 --infd-sigma $S --out "$OUT/p2_infd16_s${S}_${TAG}.jsonl"
    python -m experiments.summarize "$OUT/p2_infd16_s${S}_${TAG}.jsonl" --ref "dnoise:K=8,est=mid" \
      --metrics infd | tee "$OUT/p2_infd16_s${S}_${TAG}.txt"
  done
}

run_axis() {
  python -m experiments.run $DATA --n ${NIMG:-200} --baseline "$BASE" --insdel "$BASE" blur \
    --methods "spi:alpha=10,K=8,est=mid" "spi:alpha=100,K=8,est=mid" "spi:alpha=300,K=8,est=mid" \
              "spi:alpha=1000,K=8,est=mid" "spi:alpha=10000,K=8,est=mid" "spi:alpha=100000,K=8,est=mid" \
              "ig:N=24,est=mid" \
    --road --seeds $SEEDS \
    --save-attr "$OUT/attr_axis_${TAG}" --out "$OUT/axis_${TAG}.jsonl"
  python -m experiments.summarize "$OUT/axis_${TAG}.jsonl" --ref "ig:N=24,est=mid" \
    --metrics insdelp_${BASE} insdelp_blur insdel_${BASE} iou overlap road residual \
    | tee "$OUT/axis_${TAG}.txt"
}

run_sigma() {
  for B in black gray; do
    python -m experiments.run $DATA --n ${NIMG:-100} --baseline "$B" --insdel "$B" \
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
  python -m experiments.alpha_disagreement $DATA --n ${NIMG:-50} --baseline "$BASE" \
    --alphas 1 10 100 1000 --K 32 --est mid --resolutions element pixel grid:16 grid:32 \
    --out "$OUT/alpha_${TAG}.jsonl" --save-maps "$OUT/alpha_maps_${TAG}" \
    | tee "$OUT/alpha_${TAG}.txt"
}

run_tpa_calib() {
  for kn in "${KNS[@]}"; do set -- $kn
    python -m experiments.tpa_sweep $DATA --start 0 --n ${NIMG:-50} --baseline "$BASE" --parts "$PARTS" \
      --alphas $ALPHAS_TPA --K $1 --N $2 --eps 1 0.1 0.01 0 --kappas 1 0.1 0.01 --seeds 0 \
      --out "$OUT/tpa_calib_${TAG}_K$1.jsonl"
  done
}

run_tpa_select() {
  for kn in "${KNS[@]}"; do set -- $kn
    python -m experiments.select_tpa "$OUT/tpa_calib_${TAG}_K$1.jsonl" \
      --out "$OUT/tpa_chosen_${TAG}_K$1.json" | tee "$OUT/tpa_select_${TAG}_K$1.txt"
  done
}

run_tpa_test() {
  set -- $KN
  CH="$OUT/tpa_chosen_${TAG}_K$1.json"
  python -m experiments.tpa_sweep $DATA --start 50 --n ${NIMG:-150} --baseline "$BASE" --parts "$PARTS" \
    --alphas $ALPHAS_TPA --K $1 --N $2 --eps 1 0.1 0.01 0 --kappas 1 0.1 0.01 --seeds $SEEDS \
    --variants-file "$CH" --eval --insdel "$BASE" blur --road \
    --out "$OUT/tpa_test_${TAG}_K$1.jsonl"
  REF=$(python -c "import json;print([v for v in json.load(open('$CH')) if v.endswith('|uniform')][0])")
  python -m experiments.summarize "$OUT/tpa_test_${TAG}_K$1.jsonl" --ref "$REF" \
    --metrics hold_rel tr_rel $METRICS | tee "$OUT/tpa_test_${TAG}_K$1.txt"
}

run_agree() {
  python -m experiments.metric_agreement "$OUT/p2_${TAG}.jsonl" \
    --metrics insdel_${BASE} insdelp_${BASE} insdel_blur iou overlap pointing infd road \
    | tee "$OUT/agree_p2_${TAG}.txt"
  if [ -f "$OUT/axis_${TAG}.jsonl" ]; then
    python -m experiments.metric_agreement "$OUT/axis_${TAG}.jsonl" \
      --metrics insdelp_${BASE} insdelp_blur iou overlap road | tee "$OUT/agree_axis_${TAG}.txt"
  fi
  set -- $KN
  if [ -f "$OUT/tpa_test_${TAG}_K$1.jsonl" ]; then
    python -m experiments.metric_agreement "$OUT/tpa_test_${TAG}_K$1.jsonl" \
      --metrics hold_rel tr_rel insdelp_${BASE} iou road | tee "$OUT/agree_tpa_${TAG}.txt"
  fi
}

run_theory() {
  # CPU only, no data needed: SPI = IG + C/(α+1) + O(1/α²) on toy functions (d = 5)
  python -m experiments.verify_spi_expansion --toys mlp saturating satint multilinear --ds 5 \
    --alphas 3 10 30 100 300 --n ${NPATH:-20000} | tee "$OUT/theory_spi_expansion.txt"
}

run_theory_sweep() {
  # deterministic: does the large-α correction point towards Shapley? (NSEED toys per family × d)
  python -m experiments.sweep_spi_theory --families mlp mlpn saturating satint --ds 5 10 15 \
    --seeds ${NSEED:-200} --out "$OUT/theory_sweep.jsonl" | tee "$OUT/theory_sweep.txt"
}

run_theory_d() {
  # does the expansion degrade with d? mlpn: pre-activation noise O(1); mlp: grows like √d
  python -m experiments.verify_spi_expansion --toys mlpn mlp --ds 5 20 80 \
    --alphas 10 30 100 --small-alphas --n ${NPATH:-20000} | tee "$OUT/theory_d.txt"
}

run_analyze() {
  # forward-only profiles of f along line / SPI paths + saved axis attributions (needs stage axis)
  python -m experiments.analyze_spi $DATA --n ${NIMG:-200} --baseline "$BASE" \
    --attr-dir "$OUT/attr_axis_${TAG}" --axis "$OUT/axis_${TAG}.jsonl" \
    --prof-alphas 0.1 1 10 100 1000 --prof-K 4 --adv-alpha 100 \
    --metrics insdelp_${BASE} insdelp_blur insdel_${BASE} iou overlap road \
    --out "$OUT/analyze_${TAG}.jsonl" | tee "$OUT/analyze_${TAG}.txt"
}

run_nlp() {
  M_TAG=$(echo "$MODEL_NLP" | tr '/' '_')
  python -m experiments.nlp.run_nlp --dataset sst2 --model "$MODEL_NLP" --n ${NIMG:-872} --max-len 128 \
    --methods "ig:N=50" "ig:N=192" "spi:alpha=1" "spi:alpha=10" "spi:alpha=100" \
              "smooth:sigma=0.05" "smooth:sigma=0.1" "smooth:sigma=0.2" \
              "tpa:alpha=10,mode=uniform" "tpa:alpha=10,mode=gibbs,eps=0.1" \
              "tpa:alpha=10,mode=core,eps=0.1" "tpa:alpha=10,mode=r,eps=0.1,kappa=1" "kernelshap:M=256" \
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
  infd)       run_infd ;;
  axis)       run_axis ;;
  tpa_calib)  run_tpa_calib ;;
  tpa_select) run_tpa_select ;;
  alpha)      run_alpha ;;
  tpa_test)   run_tpa_test ;;
  sigma)      run_sigma ;;
  agree)      run_agree ;;
  nlp)        run_nlp ;;
  theory)     run_theory ;;
  theory_sweep) run_theory_sweep ;;
  theory_d)   run_theory_d ;;
  analyze)    run_analyze ;;
  all)       run_test; run_calib; run_p1; run_p2; run_infd; run_axis; run_tpa_calib; run_tpa_select
              run_alpha; run_tpa_test; run_sigma; run_agree; run_nlp ;;
  *)          echo "unknown stage '$STAGE'"; exit 1 ;;
esac
