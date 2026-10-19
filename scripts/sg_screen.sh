#!/usr/bin/env bash
# fold-0 screening of StressGuard v2 variants: bash scripts/sg_screen.sh <tag> <extra overrides...>
cd "$(dirname "$0")/.."
tag=$1; shift
SYM="train.loss.c_fn=1.0 train.loss.c_fp=1.0 train.loss.use_sev_weight=false train.loss.use_gamma_weight=false"
OMP_NUM_THREADS=2 python3 -W ignore scripts/sg_train.py --dataset ${DS:-synthetic_regime} --seed ${SEED:-42} --folds ${FOLDS:-0} --out results/sg_screen --tag $tag \
  --override $SYM "$@" > results/sg_screen/log_$tag.txt 2>&1
grep -h "fold [0-9] @" results/sg_screen/log_$tag.txt | sed "s/^/[$tag] /" | cut -c1-260
