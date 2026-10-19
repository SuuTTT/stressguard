"""Multi-scale regime context c_{t,i}: conditioning prior for detection sensitivity + friction inputs for the decision rule.

Every column is causal (trailing windows only) and normalised by its own trailing median so the same code serves
daily equities, 100 ms LOB bars and crypto depth streams:
    rv_w     log realised vol over w bars (w in `windows`) minus trailing median      volatility-surface proxy (term structure)
    depth    log depth at touch (liq[..., 1]) minus trailing median                   LOB depletion
    spread   log relative half-spread (liq[..., 0]) minus trailing median             friction
    xcorr    trailing correlation of the asset return with the market median return  cross-asset contagion
    + panel.meta["ctx_extra"] (T, N, k): implied-vol skew / VIX term structure (OptionMetrics, CBOE), funding rates
      (Binance), MOVE index (UST), or the simulator's noisy forward-looking skew proxy. Names in meta["ctx_extra_names"].
"""
from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np
import pandas as pd

from warn2trade.data import Panel


def _rel_log(x: pd.DataFrame, norm_window: int, minp: int) -> pd.DataFrame:
    lx = np.log(x.clip(lower=1e-8))
    return lx - lx.rolling(norm_window, min_periods=minp).median()


def default_context(panel: Panel, windows: Sequence[int] = (5, 20, 60), norm_window: int = 250, corr_window: int = 20) -> Tuple[np.ndarray, List[str]]:
    T, N = panel.mid.shape
    ret = pd.DataFrame(np.vstack([np.zeros((1, N)), np.diff(np.log(panel.mid), axis=0)]))
    mkt = ret.median(axis=1)
    cols, names = [], []
    minp = max(5, norm_window // 5)
    for w in windows:
        rv = ret.rolling(w, min_periods=max(2, w // 2)).std()
        cols.append(_rel_log(rv, norm_window, minp)); names.append(f"rv{w}")
    cols.append(_rel_log(pd.DataFrame(panel.liq[..., 1]), norm_window, minp)); names.append("depth")
    cols.append(_rel_log(pd.DataFrame(panel.liq[..., 0]), norm_window, minp)); names.append("spread")
    cols.append(ret.rolling(corr_window, min_periods=corr_window // 2).corr(mkt)); names.append("xcorr")
    ctx = np.stack([c.to_numpy() for c in cols], -1)
    ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.clip(ctx, -6.0, 6.0), names


def build_context(panel: Panel, cfg: dict) -> Tuple[np.ndarray, List[str]]:
    ctx, names = default_context(panel, tuple(cfg.get("windows", (5, 20, 60))), int(cfg.get("norm_window", 250)), int(cfg.get("corr_window", 20)))
    extra = panel.meta.get("ctx_extra")
    if extra is not None and cfg.get("use_extra", True):
        ctx = np.concatenate([ctx, np.asarray(extra, dtype=np.float32)], -1)
        names = names + list(panel.meta.get("ctx_extra_names", [f"extra{k}" for k in range(extra.shape[-1])]))
    return ctx, names
