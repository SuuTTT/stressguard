#!/usr/bin/env python3
"""Re-evaluate a saved run from its checkpoints with the current metric code (no retraining): rewrites eval blocks in fold*.json
and the preds / decile / cost-curve CSVs. Use after changing stressguard/eval.

    python scripts/sg_reeval.py --run results/stressguard/synthetic_regime/erm/seed42
"""
from __future__ import annotations
import argparse
import json
import os
import torch
from sg_common import build_panel_and_folds, cost_model_from_cfg, regime_env, fold_data, dims_with_ctx, bt_context, save_json, bps_key, log
from warn2trade.utils import set_seed, MODELS
from warn2trade.train import resolve_device
from stressguard.data import build_context
from stressguard.train import StressTrainer
from stressguard.eval import summarize_fold, auc_by_decile, cost_curve, ratio_sweep
import stressguard.models  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--no-sweep", action="store_true")
    args = p.parse_args()
    cfg = json.load(open(os.path.join(args.run, "config.json")))["config"]
    seed, dataset = int(cfg["seed"]), cfg["dataset"]["name"]
    set_seed(seed, cfg.get("deterministic", True))
    device = resolve_device(cfg.get("device", "auto"))
    builder, panel, folds = build_panel_and_folds(cfg)
    ctx, _ = build_context(panel, cfg["context"])
    main_bps = bps_key(cfg["cost"]["commission_bps"])
    grid = sorted(set([float(b) for b in cfg["eval"]["cost_grid_bps"]] + [float(cfg["cost"]["commission_bps"])]))
    for fold in folds:
        ck, fj = os.path.join(args.run, f"model_fold{fold.k}.pt"), os.path.join(args.run, f"fold{fold.k}.json")
        if not (os.path.exists(ck) and os.path.exists(fj)):
            continue
        set_seed(seed + fold.k)
        reg = regime_env(cfg, panel, fold, dataset)
        _, loaders = fold_data(cfg, panel, fold, reg["env"], ctx, reg["stress_score"], seed, reg.get("irm_env"))
        model = MODELS.get(cfg["model"]["name"])(dims_with_ctx(panel, ctx), cfg["model"])
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        tr = StressTrainer(cfg, model, loaders, cost_model_from_cfg(cfg), device, builder.bars_per_year)
        bt = bt_context(cfg, panel, fold, reg["env"], builder)
        res = json.load(open(fj))
        res["eval"] = {}
        for bps in grid:
            df = tr.predict(loaders["test"], cost_model_from_cfg(cfg, bps))
            res["eval"][bps_key(bps)] = summarize_fold(df, dict(bt, cost_model=cost_model_from_cfg(cfg, bps)), float(cfg["eval"].get("k_frac", 0.01)))
            if bps_key(bps) == main_bps:
                df.to_csv(os.path.join(args.run, f"preds_fold{fold.k}.csv"), index=False)
                auc_by_decile(df, "stress_score", int(cfg["eval"].get("n_deciles", 10))).to_csv(os.path.join(args.run, f"auc_decile_fold{fold.k}.csv"), index=False)
                cost_curve(df, cfg["eval"]["cost_ratios"], float(cfg["model"]["decision"].get("c_fp", 1.0))).to_csv(os.path.join(args.run, f"cost_curve_fold{fold.k}.csv"), index=False)
        if not args.no_sweep:
            res["ratio_sweep"] = ratio_sweep(tr, loaders["test"], cost_model_from_cfg(cfg), cfg["eval"]["cost_ratios"], float(cfg["model"]["decision"].get("c_fp", 1.0)),
                                             bt, float(cfg["eval"].get("k_frac", 0.01)))
        det = res["eval"][main_bps]["detection"]
        log.info(f"[{os.path.basename(os.path.dirname(args.run))}] fold {fold.k} | AUC normal={det['normal'].get('AUC_ROC', float('nan')):.3f} stress={det['stress'].get('AUC_ROC', float('nan')):.3f} "
                 f"severe={det.get('severe', {}).get('AUC_ROC', float('nan')):.3f} true_severe={det.get('true_severe', {}).get('AUC_ROC', float('nan')):.3f}")
        save_json(res, fj)


if __name__ == "__main__":
    main()
