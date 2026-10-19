#!/usr/bin/env python3
"""Train + evaluate StressGuard (or an ablation config) with purged walk-forward validation and regime-sliced reporting.

    python scripts/sg_train.py --dataset synthetic_regime --seed 42
    python scripts/sg_train.py --config configs/stressguard/ablations/erm.yaml --dataset synthetic_regime --seed 42
    python scripts/sg_train.py --dataset lobster_sg --override train.batch_size=512 cost.commission_bps=10
"""
from __future__ import annotations
import os
import time
from sg_common import (sg_parser, load_full_config, build_panel_and_folds, cost_model_from_cfg, save_json, bps_key, regime_env, fold_data,
                       dims_with_ctx, bt_context, run_dir_for, log)
from warn2trade.utils import set_seed, MODELS
from warn2trade.train import resolve_device
from stressguard.data import build_context
from stressguard.train import StressTrainer
from stressguard.eval import summarize_fold, auc_by_decile, cost_curve, ratio_sweep
import stressguard.models  # noqa: F401  (registers "stressguard")


def main():
    p = sg_parser("Train StressGuard")
    p.add_argument("--folds", nargs="*", type=int, default=None, help="run only these fold ids")
    args = p.parse_args()
    cfg = load_full_config(args)
    set_seed(args.seed, cfg.get("deterministic", True))
    device = resolve_device(args.device)
    builder, panel, folds = build_panel_and_folds(cfg)
    if args.folds:
        folds = [f for f in folds if f.k in set(args.folds)]
    ctx, ctx_names = build_context(panel, cfg["context"])
    tag = args.tag or cfg["model"].get("tag", cfg["model"]["name"])
    run_dir = run_dir_for(args.out, args.dataset, tag, args.seed)
    log.info(f"dataset={args.dataset} T={panel.T} N={panel.N} folds={len(folds)} ctx={ctx_names} device={device} run_dir={run_dir}")
    main_bps = bps_key(cfg["cost"]["commission_bps"])
    grid = sorted(set([float(b) for b in cfg["eval"]["cost_grid_bps"]] + [float(cfg["cost"]["commission_bps"])]))
    results = []
    save_json({"config": cfg, "ctx_names": ctx_names}, os.path.join(run_dir, "config.json"))     # written first so interrupted runs stay visible
    for fold in folds:
        set_seed(args.seed + fold.k)
        reg = regime_env(cfg, panel, fold, args.dataset)
        ds, loaders = fold_data(cfg, panel, fold, reg["env"], ctx, reg["stress_score"], args.seed, reg.get("irm_env"))
        log.info(f"{fold} | train={len(ds['train'])} envs={ds['train'].env_counts()} pos/env={ds['train'].pos_rate_by_env()} "
                 f"| stress frac train={reg['stress_frac_train']:.2%} test={reg['stress_frac_test']:.2%}" + (f" | regime-rule F1 vs truth={reg['agreement']['f1']:.2f}" if "agreement" in reg else ""))
        model = MODELS.get(cfg["model"]["name"])(dims_with_ctx(panel, ctx), cfg["model"])
        gcfg = cfg["model"].get("gate", {})
        gate_info = None
        if gcfg.get("enabled", False):
            from stressguard.models.gate import channel_reliability
            gate_info = channel_reliability(panel.price_feat, panel.labels, reg["stress_score"], fold.train,
                                            gcfg.get("quantiles", [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]), float(gcfg.get("extrap", 0.5)),
                                            float(gcfg.get("min_calm", 2.0)), float(gcfg.get("power", 1.0)), z_trend=float(gcfg.get("z_trend", 2.0)))
            gate_info = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in gate_info.items()}
            if gcfg.get("mode", "global") == "adaptive":
                model.enc.set_adaptive(gate_info)                 # v4: per-bar suppression at inference only
            else:
                model.enc.set_gate(gate_info["gate"])             # v3: global gate
            names = panel.meta.get("feature_names", [f"f{j}" for j in range(panel.price_feat.shape[-1])])
            log.info("SECG gate: " + " ".join(f"{n}={g:.2f}" for n, g in zip(names, gate_info["gate"])))
        trainer = StressTrainer(cfg, model, loaders, cost_model_from_cfg(cfg), device, builder.bars_per_year)
        t0 = time.time()
        fit = trainer.fit()
        bt = bt_context(cfg, panel, fold, reg["env"], builder)
        import torch
        torch.save(model.state_dict(), os.path.join(run_dir, f"model_fold{fold.k}.pt"))
        res = {"fold": fold.k, "train_sec": time.time() - t0, "history": fit["history"], "static_thr": float(model.static_thr), "calibration": fit.get("calibration"), "gate": gate_info,
               "regime": {k: v for k, v in reg.items() if k not in ("env", "stress_score", "irm_env")}, "eval": {}}
        for bps in grid:
            df = trainer.predict(loaders["test"], cost_model_from_cfg(cfg, bps))
            res["eval"][bps_key(bps)] = summarize_fold(df, dict(bt, cost_model=cost_model_from_cfg(cfg, bps)), float(cfg["eval"].get("k_frac", 0.01)))
            if bps_key(bps) == main_bps:
                df.to_csv(os.path.join(run_dir, f"preds_fold{fold.k}.csv"), index=False)
                auc_by_decile(df, "stress_score", int(cfg["eval"].get("n_deciles", 10))).to_csv(os.path.join(run_dir, f"auc_decile_fold{fold.k}.csv"), index=False)
                cost_curve(df, cfg["eval"]["cost_ratios"], float(cfg["model"]["decision"].get("c_fp", 1.0))).to_csv(os.path.join(run_dir, f"cost_curve_fold{fold.k}.csv"), index=False)
        res["ratio_sweep"] = ratio_sweep(trainer, loaders["test"], cost_model_from_cfg(cfg), cfg["eval"]["cost_ratios"], float(cfg["model"]["decision"].get("c_fp", 1.0)),
                                         bt, float(cfg["eval"].get("k_frac", 0.01)))
        m = res["eval"][main_bps]
        det, tr = m["detection"], m["trading"]
        log.info(f"fold {fold.k} @ {main_bps}bps | AUC full={det['full'].get('AUC_ROC', float('nan')):.3f} normal={det['normal'].get('AUC_ROC', float('nan')):.3f} "
                 f"stress={det['stress'].get('AUC_ROC', float('nan')):.3f} dAUC={det['dAUC_stress']:.3f} TailF1_stress={det['stress'].get('TailF1', float('nan')):.3f} "
                 f"| MDD base={tr['base']['MDD']:.2%} hedged={tr['hedged']['MDD']:.2%} SR hedged={tr['hedged']['SR']:.2f} Calmar={tr['hedged']['Calmar']:.2f} "
                 f"hedge_rate={tr['overlay']['hedge_rate']:.3f}")
        save_json(res, os.path.join(run_dir, f"fold{fold.k}.json"))
        results.append(res)
        print(f"progress {len(results)}/{len(folds)}", flush=True)
    save_json({"config": cfg, "n_folds": len(results), "ctx_names": ctx_names}, os.path.join(run_dir, "config.json"))
    log.info(f"done: {len(results)} folds -> {run_dir}")


if __name__ == "__main__":
    main()
