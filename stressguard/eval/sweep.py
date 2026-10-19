"""Risk-aversion frontier: re-apply the dynamic decision rule at several c_FN / c_FP ratios (inference only) and report the
regime-sliced detection + trading matrix per ratio. Training is done once (at the config's ratio); the frontier shows how the
same calibrated detector trades return for drawdown as the asymmetry grows.
"""
from __future__ import annotations
from typing import Any, Dict, Sequence
import numpy as np
from .metrics import summarize_fold


def ratio_sweep(trainer, loader, cost_model, ratios: Sequence[float], c_fp: float, bt: Dict[str, Any], k_frac: float = 0.01) -> Dict[str, Dict[str, Any]]:
    model = trainer.model
    c_fn0, c_fp0, dyn0 = model.c_fn, model.c_fp, model.dynamic
    out: Dict[str, Dict[str, Any]] = {}
    try:
        model.dynamic = True
        for r in ratios:
            model.set_costs(float(r) * c_fp, c_fp)
            df = trainer.predict(loader, cost_model)
            s = summarize_fold(df, dict(bt, cost_model=cost_model), k_frac)
            out[f"{float(r):g}"] = {"detection": {sl: {k: s["detection"][sl].get(k) for k in ("alert_rate", "alert_precision", "alert_recall")} for sl in ("full", "normal", "stress")},
                                    "trading": s["trading"]}
    finally:
        model.set_costs(c_fn0, c_fp0)
        model.dynamic = dyn0
    return out
