#!/usr/bin/env python3
"""Run a score-only baseline through the shared decision layer and overlay backtest.

    python scripts/sg_baseline.py --config configs/stressguard/baselines/recon_ae.yaml --dataset synthetic_regime --seed 42
    python scripts/sg_baseline.py --config configs/stressguard/baselines/iforest.yaml --decision dynamic   # plug-in dynamic rule on top
"""
from __future__ import annotations
import math
import os
import time
import numpy as np
import pandas as pd
import torch
from sg_common import (sg_parser, load_full_config, build_panel_and_folds, cost_model_from_cfg, save_json, bps_key, regime_env, fold_data,
                       dims_with_ctx, bt_context, run_dir_for, log)
from warn2trade.utils import set_seed, BASELINES
from warn2trade.train import resolve_device
from warn2trade.train.trainer import to_device
from stressguard.data import build_context
from stressguard.models.stressguard import attach_gamma
from stressguard.models.losses import bayes_threshold, hedge_ratio
from stressguard.backtest import OverlayBacktester
from stressguard.eval import summarize_fold, auc_by_decile, cost_curve
import stressguard.baselines  # noqa: F401


@torch.no_grad()
def collect(bl, loader, device, cost_model, latency, tau, a, b, static_thr, decision, dec_cfg):
    bt = OverlayBacktester(cost_model, latency, tau)
    frames = []
    for batch in loader:
        batch = attach_gamma(to_device(batch, device), cost_model)
        s = bl.score(batch).float()
        p = torch.sigmoid(a * s + b)
        sev_hat = float(dec_cfg.get("sev_plugin_k", 2.0)) * batch["sigma"] * math.sqrt(tau)
        c_fn, c_fp, h_max = float(dec_cfg.get("c_fn", 5.0)), float(dec_cfg.get("c_fp", 1.0)), float(dec_cfg.get("h_max", 1.0))
        if decision == "dynamic":
            thr = bayes_threshold(sev_hat, batch["gamma"], c_fn, c_fp)
            h = hedge_ratio(p, sev_hat, batch["gamma"], c_fn, c_fp, h_max)
        else:
            thr = torch.full_like(p, static_thr)
            h = h_max * (p >= thr).float()
        frames.append(bt.collect({"p": p, "thr": thr, "h": h, "sev_hat": sev_hat}, batch))
    df = pd.concat(frames, ignore_index=True)
    df["static_thr"] = static_thr
    return df


@torch.no_grad()
def scores(bl, loader, device, cost_model):
    ys, ss = [], []
    for batch in loader:
        batch = attach_gamma(to_device(batch, device), cost_model)
        ss.append(bl.score(batch).float().cpu()); ys.append(batch["y"].cpu())
    return torch.cat(ys).numpy(), torch.cat(ss).numpy()


def main():
    p = sg_parser("Run a StressGuard baseline", "configs/stressguard/baselines/recon_ae.yaml")
    p.add_argument("--decision", default="static", choices=["static", "dynamic"], help="static = val-fitted best-F1 threshold; dynamic = plug-in cost-sensitive rule")
    p.add_argument("--folds", nargs="*", type=int, default=None, help="run only these fold ids")
    args = p.parse_args()
    cfg = load_full_config(args)
    set_seed(args.seed, cfg.get("deterministic", True))
    device = resolve_device(args.device)
    builder, panel, folds = build_panel_and_folds(cfg)
    if args.folds:
        folds = [f for f in folds if f.k in set(args.folds)]
    ctx, _ = build_context(panel, cfg["context"])
    name = cfg["baseline"]["name"]
    tag = args.tag or (name if args.decision == "static" else f"{name}+dyn")
    run_dir = run_dir_for(args.out, args.dataset, tag, args.seed)
    tau, latency = int(cfg["model"]["tau"]), int(cfg["data"]["latency"])
    main_bps = bps_key(cfg["cost"]["commission_bps"])
    grid = sorted(set([float(b) for b in cfg["eval"]["cost_grid_bps"]] + [float(cfg["cost"]["commission_bps"])]))
    from stressguard.baselines import calibrate_scores, fit_static_threshold
    results = []
    for fold in folds:
        set_seed(args.seed + fold.k)
        reg = regime_env(cfg, panel, fold, args.dataset)
        ds, loaders = fold_data(cfg, panel, fold, reg["env"], ctx, reg["stress_score"], args.seed, reg.get("irm_env"))
        bl = BASELINES.get(name)(cfg["baseline"], dims_with_ctx(panel, ctx), device)
        t0 = time.time()
        bl.fit(loaders["train"], loaders["val"])
        cm = cost_model_from_cfg(cfg)
        vy, vs = scores(bl, loaders["val"], device, cm)
        a, b = calibrate_scores(vs, vy)
        static_thr = fit_static_threshold(1.0 / (1.0 + np.exp(-(a * vs + b))), vy)
        bt = bt_context(cfg, panel, fold, reg["env"], builder)
        res = {"fold": fold.k, "train_sec": time.time() - t0, "baseline": {"name": name, "status": bl.status, "paper": bl.paper, "decision": args.decision},
               "static_thr": static_thr, "platt": [a, b], "regime": {k: v for k, v in reg.items() if k not in ("env", "stress_score", "irm_env")}, "eval": {}}
        for bps in grid:
            df = collect(bl, loaders["test"], device, cost_model_from_cfg(cfg, bps), latency, tau, a, b, static_thr, args.decision, cfg["model"]["decision"])
            res["eval"][bps_key(bps)] = summarize_fold(df, dict(bt, cost_model=cost_model_from_cfg(cfg, bps)), float(cfg["eval"].get("k_frac", 0.01)))
            if bps_key(bps) == main_bps:
                df.to_csv(os.path.join(run_dir, f"preds_fold{fold.k}.csv"), index=False)
                auc_by_decile(df, "stress_score", int(cfg["eval"].get("n_deciles", 10))).to_csv(os.path.join(run_dir, f"auc_decile_fold{fold.k}.csv"), index=False)
                cost_curve(df, cfg["eval"]["cost_ratios"], float(cfg["model"]["decision"].get("c_fp", 1.0))).to_csv(os.path.join(run_dir, f"cost_curve_fold{fold.k}.csv"), index=False)
        det, tr = res["eval"][main_bps]["detection"], res["eval"][main_bps]["trading"]
        log.info(f"[{tag}] fold {fold.k} | AUC normal={det['normal'].get('AUC_ROC', float('nan')):.3f} stress={det['stress'].get('AUC_ROC', float('nan')):.3f} "
                 f"dAUC={det['dAUC_stress']:.3f} | MDD base={tr['base']['MDD']:.2%} hedged={tr['hedged']['MDD']:.2%} SR={tr['hedged']['SR']:.2f} hedge_rate={tr['overlay']['hedge_rate']:.3f}")
        save_json(res, os.path.join(run_dir, f"fold{fold.k}.json"))
        results.append(res)
        print(f"progress {len(results)}/{len(folds)}", flush=True)
    save_json({"config": cfg, "n_folds": len(results)}, os.path.join(run_dir, "config.json"))
    log.info(f"done: {len(results)} folds -> {run_dir}")


if __name__ == "__main__":
    main()
