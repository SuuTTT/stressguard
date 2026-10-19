"""StressGuard: cost-sensitive invariant anomaly detection for trading under extreme volatility cascades.

Package layout (maps the brief's `c225/date/zyy` layout onto this repository)
---------------------------------------------------------------------------
stressguard.data       regime partition (Normal vs Stress environments), multi-scale context priors,
                       the multi-regime jump-diffusion / Hawkes simulator, RegimeEventDataset
stressguard.models     stress-invariant encoder (IRM / VREx), severity head, cost-sensitive dynamic decision boundary
stressguard.backtest   hedge-overlay execution with the shared non-linear cost model (sqrt impact, half-spread, delay)
stressguard.train      environment-batched trainer with annealed invariance penalty and worst-environment model selection
stressguard.eval       regime-sliced metric matrix: Delta-AUC_stress, tail P/R/F1, VaR/ES, economic cost curves
stressguard.baselines  unified score-only baseline contract + 10 benchmark slots with verification status

Shared infrastructure is imported from `warn2trade` (Panel schema, purged walk-forward splits, CostModel, metrics).
"""
__version__ = "0.1.0"
