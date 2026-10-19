#!/usr/bin/env python3
"""Feature-reliance diagnostic: permute one input feature over time (test window only) and measure the AUC change per regime.
A detector that leans on the stress-dependent spurious feature loses AUC when it is permuted; an invariant detector should not.

    python scripts/sg_reliance.py --run results/stressguard/synthetic_regime/erm/seed42 --feature sp
    python scripts/sg_reliance.py --run results/stressguard/synthetic_regime/stressguard/seed42 --feature ofi
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import numpy as np
import torch
from sg_common import build_panel_and_folds, cost_model_from_cfg, regime_env, fold_data, dims_with_ctx, save_json, log
from warn2trade.utils import set_seed, MODELS
from warn2trade.train import resolve_device
from warn2trade.backtest import metrics as M
from stressguard.data import build_context
from stressguard.train import StressTrainer
import stressguard.models  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--feature", default="sp")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    cfg = json.load(open(os.path.join(args.run, "config.json")))["config"]
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    dataset = cfg["dataset"]["name"]
    set_seed(seed, cfg.get("deterministic", True))
    device = resolve_device(cfg.get("device", "auto"))
    builder, panel, folds = build_panel_and_folds(cfg)
    names = panel.meta.get("feature_names", [f"f{j}" for j in range(panel.price_feat.shape[-1])])
    j = names.index(args.feature)
    ctx, _ = build_context(panel, cfg["context"])
    rows = []
    for fold in folds:
        ck = os.path.join(args.run, f"model_fold{fold.k}.pt")
        if not os.path.exists(ck):
            continue
        set_seed(seed + fold.k)
        reg = regime_env(cfg, panel, fold, dataset)
        model = MODELS.get(cfg["model"]["name"])(dims_with_ctx(panel, ctx), cfg["model"])
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        res = {"fold": fold.k, "feature": args.feature}
        for mode in ("intact", "permuted"):
            pan = panel
            if mode == "permuted":
                pan = copy.copy(panel)
                pf = panel.price_feat.copy()
                rng = np.random.default_rng(seed)
                lo, hi = int(fold.test[0]) - int(cfg["data"]["lookback"]), int(fold.test[-1]) + 1
                for i in range(panel.N):
                    pf[lo:hi, i, j] = rng.permutation(pf[lo:hi, i, j])
                pan.price_feat = pf
            _, loaders = fold_data(cfg, pan, fold, reg["env"], ctx, reg["stress_score"], seed, reg.get("irm_env"))
            tr = StressTrainer(cfg, model, loaders, cost_model_from_cfg(cfg), device, builder.bars_per_year)
            df = tr.predict(loaders["test"])
            from stressguard.eval.metrics import slice_masks
            for sl, m in slice_masks(df).items():
                res[f"AUC_{sl}_{mode}"] = M.auc_roc(df["y"].to_numpy()[m], df["score"].to_numpy()[m]) if m.sum() else float("nan")
        for sl in [k for k in ("full", "normal", "stress", "severe", "true_severe") if f"AUC_{k}_intact" in res]:
            res[f"drop_{sl}"] = res[f"AUC_{sl}_intact"] - res[f"AUC_{sl}_permuted"]
        sev = f" | true_severe {res['AUC_true_severe_intact']:.3f}->{res['AUC_true_severe_permuted']:.3f}" if "AUC_true_severe_intact" in res else ""
        log.info(f"fold {fold.k} feature={args.feature} | AUC normal {res['AUC_normal_intact']:.3f}->{res['AUC_normal_permuted']:.3f} "
                 f"| stress {res['AUC_stress_intact']:.3f}->{res['AUC_stress_permuted']:.3f}{sev} | drop normal={res['drop_normal']:.3f} stress={res['drop_stress']:.3f}")
        rows.append(res)
    save_json({"run": args.run, "feature": args.feature, "folds": rows}, os.path.join(args.run, f"reliance_{args.feature}.json"))


if __name__ == "__main__":
    main()
