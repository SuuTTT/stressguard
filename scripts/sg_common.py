"""Shared plumbing for the StressGuard scripts (scripts/sg_*.py)."""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from warn2trade.data import build_loader, WalkForwardSplitter
from warn2trade.utils import get_logger, load_config, deep_merge, apply_overrides, DATASETS
from warn2trade.backtest import CostModel
from stressguard.data import RegimeLabeler, market_return, apply_calendar, load_stress_split, build_context, RegimeEventDataset, env_balanced_loader

log = get_logger("sg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "stressguard")
DATASET_DIRS = (os.path.join(ROOT, "configs", "stressguard", "datasets"), os.path.join(ROOT, "configs", "datasets"))   # own track first


def dataset_config_path(name: str) -> str:
    for d in DATASET_DIRS:
        p = os.path.join(d, f"{name}.yaml")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no dataset config {name}.yaml under {DATASET_DIRS}")


def load_full_config(args) -> Dict[str, Any]:
    cfg = load_config(args.config, args.override)
    ds_cfg = load_config(dataset_config_path(args.dataset))
    cfg = deep_merge(cfg, {"dataset": ds_cfg})
    if "horizons" in ds_cfg:
        cfg.setdefault("model", {})["horizons"] = ds_cfg["horizons"]
    if "split" in ds_cfg:
        cfg["split"] = deep_merge(cfg.get("split", {}), ds_cfg["split"])
    cfg = apply_overrides(cfg, args.override)      # CLI overrides win over dataset defaults
    cfg["seed"], cfg["device"] = args.seed, args.device
    if getattr(args, "max_folds", None) is not None:
        cfg["split"]["max_folds"] = args.max_folds
    return cfg


def build_panel_and_folds(cfg: Dict[str, Any]):
    builder = DATASETS.get(cfg["dataset"]["name"])(cfg["dataset"])
    panel = builder.build_panel()
    sp = cfg["split"]
    hmax = max(cfg["model"]["horizons"]) + cfg["data"]["latency"] + cfg["data"].get("exec_max", 4)
    purge = max(int(sp.get("purge", 0)), hmax)
    splitter = WalkForwardSplitter(panel.T, sp["train"], sp["val"], sp["test"], sp.get("step"), purge, sp.get("embargo", 0),
                                   sp.get("anchored", False), sp.get("max_folds"))
    return builder, panel, list(splitter)


def cost_model_from_cfg(cfg: Dict[str, Any], bps: float = None) -> CostModel:
    c = cfg["cost"]
    return CostModel(bps if bps is not None else c["commission_bps"], c.get("use_half_spread", True), c.get("impact_eta", 0.1), c.get("delay_zeta", 0.05), c.get("notional", 1.0))


def bps_key(bps) -> str:
    return f"{float(bps):g}"


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


def sg_parser(desc: str, default_cfg: str = "configs/stressguard/stressguard.yaml") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=os.path.join(ROOT, default_cfg))
    p.add_argument("--dataset", default="synthetic_regime", help="configs/stressguard/datasets/<name>.yaml (falls back to configs/datasets/)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=RESULTS)
    p.add_argument("--tag", default=None)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--override", nargs="*", default=[], help="dotted overrides a.b=c")
    return p


def regime_env(cfg: Dict[str, Any], panel, fold, dataset: str) -> Dict[str, Any]:
    """Fit the regime rule on the fold's TRAIN bars only, label every bar, OR the curated calendar if present."""
    r = market_return(panel.mid)
    lab = RegimeLabeler.from_cfg(cfg["regime"]["labeler"])
    res = lab.fit_transform(r, fold.train)
    env = res["env"]
    cal = load_stress_split(os.path.join(ROOT, "data", "stress_splits", f"{dataset}.json")) if cfg["regime"].get("use_calendar", True) else None
    if cal:
        env = apply_calendar(panel.times, env, cal, cfg["regime"].get("calendar_mode", "or"))
    if cfg["regime"].get("source", "rule") == "truth" and "env_true" in panel.meta:
        env = panel.meta["env_true"].astype(np.int8)
    irm_q = cfg["regime"].get("irm_quantiles")          # e.g. [0.5, 0.8, 0.95] -> 4 training environments by stress-score quantile (train-fitted)
    if irm_q:
        edges = np.nanquantile(res["stress_score"][fold.train], irm_q)
        irm_env = np.digitize(np.nan_to_num(res["stress_score"]), edges).astype(np.float32)
    else:
        irm_env = env.astype(np.float32)
    out = {"env": env, "irm_env": irm_env, "stress_score": res["stress_score"], "vol_thr": lab.vol_thr_, "stress_frac_train": float(env[fold.train].mean()),
           "stress_frac_test": float(env[fold.test].mean())}
    if "env_true" in panel.meta:
        tr = panel.meta["env_true"].astype(bool)
        pr = env.astype(bool)
        tp = float((tr & pr).sum())
        out["agreement"] = {"precision": tp / max(pr.sum(), 1), "recall": tp / max(tr.sum(), 1), "f1": 2 * tp / max(pr.sum() + tr.sum(), 1)}
    return out


def fold_data(cfg: Dict[str, Any], panel, fold, env: np.ndarray, ctx: np.ndarray, stress_score: np.ndarray, seed: int, irm_env: np.ndarray = None):
    d, m = cfg["data"], cfg["model"]
    rng = np.random.default_rng(seed)
    aux = {"stress_true": panel.meta["u"]} if "u" in panel.meta else {}
    if irm_env is not None:
        aux["irm_env"] = irm_env
    mk = lambda idx, neg: RegimeEventDataset(panel, idx, env, ctx, int(m["tau"]), d["lookback"], m["horizons"], d["latency"], d.get("exec_max", 4),
                                             d.get("vol_window", 60), neg_subsample=neg, rng=rng, stress_score=stress_score, aux=aux)
    ds = {"train": mk(fold.train, d.get("neg_subsample")), "val": mk(fold.val, None), "test": mk(fold.test, None)}
    bs, nw = int(cfg["train"]["batch_size"]), int(d.get("num_workers", 0))
    loaders = {"train": env_balanced_loader(ds["train"], bs, nw, seed, bool(cfg["train"].get("env_balance", True))),
               "val": build_loader(ds["val"], bs, False, nw, seed), "test": build_loader(ds["test"], bs, False, nw, seed)}
    return ds, loaders


def dims_with_ctx(panel, ctx: np.ndarray) -> Dict[str, int]:
    dims = panel.dims()
    dims["ctx"] = int(ctx.shape[-1])
    return dims


def bt_context(cfg: Dict[str, Any], panel, fold, env: np.ndarray, builder) -> Dict[str, Any]:
    """Everything the bar-level hedge book needs for one fold (cost_model is filled in per commission level)."""
    return {"panel": panel, "test": fold.test, "env_bars": env[fold.test], "latency": int(cfg["data"]["latency"]), "tau": int(cfg["model"]["tau"]),
            "bars_per_year": builder.bars_per_year, "h_max": float(cfg["model"].get("decision", {}).get("h_max", 1.0)), "vol_window": int(cfg["data"].get("vol_window", 60))}


def test_bars(panel, fold, env: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Per-bar base log returns (n_bars, N) and env over the fold's test window, plus t0."""
    lr = np.log(panel.mid[fold.test]) - np.log(panel.mid[fold.test - 1])
    return lr, env[fold.test], int(fold.test[0])


def run_dir_for(out: str, dataset: str, tag: str, seed: int) -> str:
    d = os.path.join(out, dataset, tag, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d
