#!/usr/bin/env python3
"""Build and persist the Normal/Stress partition for a dataset -> data/stress_splits/<dataset>.json.

    python scripts/sg_make_stress_splits.py --dataset synthetic_regime
The rule (vol percentile OR tail index OR drawdown speed) is calibrated on the first `split.train` bars; the recipe's curated
calendar (stressguard/data/recipes.py) is OR-ed in for real datasets whose `times` are timestamps.
"""
from __future__ import annotations
import os
import numpy as np
from sg_common import sg_parser, load_full_config, build_panel_and_folds, ROOT, log
from stressguard.data import RegimeLabeler, market_return, apply_calendar, write_stress_split


def main():
    args = sg_parser("Make stress splits").parse_args()
    cfg = load_full_config(args)
    builder, panel, folds = build_panel_and_folds(cfg)
    r = market_return(panel.mid)
    lab = RegimeLabeler.from_cfg(cfg["regime"]["labeler"])
    res = lab.fit_transform(r, np.arange(min(int(cfg["split"]["train"]), panel.T)))
    env = res["env"]
    cal = getattr(builder, "stress_calendar", [])
    if cal and np.issubdtype(panel.times.dtype, np.integer) and panel.times.max() > 10**12:
        env = apply_calendar(panel.times, env, cal, "or")
    extra = {"vol_thr": lab.vol_thr_, "rule": cfg["regime"]["labeler"], "calendar": cal}
    if "env_true" in panel.meta:
        tr, pr = panel.meta["env_true"].astype(bool), env.astype(bool)
        tp = float((tr & pr).sum())
        extra["agreement_with_truth"] = {"precision": tp / max(pr.sum(), 1), "recall": tp / max(tr.sum(), 1), "f1": 2 * tp / max(pr.sum() + tr.sum(), 1)}
    rec = write_stress_split(os.path.join(ROOT, "data", "stress_splits", f"{args.dataset}.json"), args.dataset, env, panel.times, extra)
    log.info(f"{args.dataset}: {rec['n_bars']} bars, stress frac={rec['stress_frac']:.2%}, {len(rec['stress_runs'])} stress runs" + (f", F1 vs truth={extra['agreement_with_truth']['f1']:.2f}" if "agreement_with_truth" in extra else ""))


if __name__ == "__main__":
    main()
