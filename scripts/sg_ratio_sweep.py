#!/usr/bin/env python3
"""Offline risk-aversion frontier for a saved StressGuard run (needs model_fold*.pt): writes ratio_sweep_fold{k}.json per fold.

    python scripts/sg_ratio_sweep.py --run results/stressguard/synthetic_regime/stressguard/seed42
"""
from __future__ import annotations
import argparse
import json
import os
import torch
from sg_common import build_panel_and_folds, cost_model_from_cfg, regime_env, fold_data, dims_with_ctx, bt_context, save_json, log
from warn2trade.utils import set_seed, MODELS
from warn2trade.train import resolve_device
from stressguard.data import build_context
from stressguard.train import StressTrainer
from stressguard.eval import ratio_sweep
import stressguard.models  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ratios", nargs="*", type=float, default=None)
    args = p.parse_args()
    cfg = json.load(open(os.path.join(args.run, "config.json")))["config"]
    ratios = args.ratios or cfg["eval"]["cost_ratios"]
    seed, dataset = int(cfg["seed"]), cfg["dataset"]["name"]
    set_seed(seed, cfg.get("deterministic", True))
    device = resolve_device(cfg.get("device", "auto"))
    builder, panel, folds = build_panel_and_folds(cfg)
    ctx, _ = build_context(panel, cfg["context"])
    for fold in folds:
        ck = os.path.join(args.run, f"model_fold{fold.k}.pt")
        if not os.path.exists(ck):
            continue
        set_seed(seed + fold.k)
        reg = regime_env(cfg, panel, fold, dataset)
        _, loaders = fold_data(cfg, panel, fold, reg["env"], ctx, reg["stress_score"], seed, reg.get("irm_env"))
        model = MODELS.get(cfg["model"]["name"])(dims_with_ctx(panel, ctx), cfg["model"])
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        tr = StressTrainer(cfg, model, loaders, cost_model_from_cfg(cfg), device, builder.bars_per_year)
        bt = bt_context(cfg, panel, fold, reg["env"], builder)
        sw = ratio_sweep(tr, loaders["test"], cost_model_from_cfg(cfg), ratios, float(cfg["model"]["decision"].get("c_fp", 1.0)), bt)
        for r, v in sw.items():
            t = v["trading"]
            log.info(f"fold {fold.k} ratio={r}: hedge_rate={t['overlay']['hedge_rate']:.3f} ARR={t['hedged']['ARR']:.3%} SR={t['hedged']['SR']:.2f} MDD base={t['base']['MDD']:.2%} hedged={t['hedged']['MDD']:.2%} "
                     f"MDD_stress hedged={t.get('hedged_stress', {}).get('MDD', float('nan')):.2%} ES99={t['hedged']['ES99']:.4f}")
        save_json({"fold": fold.k, "ratio_sweep": sw}, os.path.join(args.run, f"ratio_sweep_fold{fold.k}.json"))


if __name__ == "__main__":
    main()
