#!/usr/bin/env bash
# Full StressGuard experiment grid. Usage: bash scripts/sg_run_all.sh [datasets...]   (default: synthetic_regime)
# Real datasets need their raw files under data/raw/<key>/ (see stressguard/data/recipes.py); missing ones are skipped with a message.
set -u
cd "$(dirname "$0")/.."
DATASETS=("${@:-synthetic_regime}")
SEEDS=(42 2024 2025 2026)
ABL=(erm no_irm vrex no_econ static_thr no_ctx no_cvar no_sev_head linear_surrogate no_env_balance)
BASE=(zscore iforest recon_ae dlinear_recon)
for ds in "${DATASETS[@]}"; do
  python3 scripts/sg_make_stress_splits.py --dataset "$ds" || { echo "[skip] $ds (raw data missing?)"; continue; }
  for s in "${SEEDS[@]}"; do
    python3 scripts/sg_train.py --dataset "$ds" --seed "$s"
    for a in "${ABL[@]}"; do python3 scripts/sg_train.py --config "configs/stressguard/ablations/$a.yaml" --dataset "$ds" --seed "$s"; done
    for b in "${BASE[@]}"; do
      python3 scripts/sg_baseline.py --config "configs/stressguard/baselines/$b.yaml" --dataset "$ds" --seed "$s" --decision static
      python3 scripts/sg_baseline.py --config "configs/stressguard/baselines/$b.yaml" --dataset "$ds" --seed "$s" --decision dynamic
    done
  done
done
python3 scripts/sg_make_tables.py
python3 scripts/sg_motivation_fig.py
