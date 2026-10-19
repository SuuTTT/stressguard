"""Environment partition: label every bar as Normal (0) or Stress (1) from causal market-wide statistics.

Three stress signals, all trailing (no look-ahead), computed on the cross-sectional median log return r_t:
    (1) realised-volatility percentile   sigma_t >= Q_q(sigma)   with Q_q fitted on the TRAIN window only
    (2) Hill tail index                  alpha_hat_t <= Q_{tail_quantile}(alpha_hat) on TRAIN (left tail turns heavy; an absolute
                                         cap `tail_alpha_crit` is used instead when tail_quantile is null)
    (3) drawdown speed                   sum_{k<W} r_{t-k} <= -k_dd * sigma_t * sqrt(W)
Stress_t = (1) OR (2) OR (3), then run-length smoothed (the smoothing looks up to min_run-1 bars ahead: the label defines
environments and evaluation slices ex post and is never a model input). A curated event calendar can override the rule for real datasets
(2010-05-06, 2015-08-24, 2020-02-24..2020-04-15, 2022-01..2022-10 hike cascade, 2024-08-05 yen-carry unwind, ...).
The environment index is the ONLY place where regime knowledge enters training: it defines the IRM environments.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


def market_return(mid: np.ndarray) -> np.ndarray:
    lr = np.full(mid.shape, np.nan)
    lr[1:] = np.log(mid[1:]) - np.log(mid[:-1])
    with np.errstate(all="ignore"):
        return np.nanmedian(lr, axis=1)


def rolling_std(x: np.ndarray, window: int, minp: Optional[int] = None) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=minp or max(2, window // 2)).std().to_numpy()


def hill_tail_index(x: np.ndarray, window: int = 250, k_frac: float = 0.1, stride: int = 1) -> np.ndarray:
    """Trailing Hill estimator of the left-tail index of losses (-x): alpha = [mean_{i<=k} log(X_(i)/X_(k+1))]^-1.
    Smaller alpha = heavier tail. `stride` > 1 evaluates every `stride` bars and forward-fills (large T)."""
    T = len(x)
    out = np.full(T, np.nan)
    losses = -x
    for t in range(window - 1, T, stride):
        seg = losses[t - window + 1 : t + 1]
        seg = seg[np.isfinite(seg) & (seg > 0)]
        k = max(int(k_frac * len(seg)), 5)
        if len(seg) <= k + 1:
            continue
        s = np.sort(seg)[::-1]
        m = np.log(s[:k] / s[k]).mean()
        out[t] = 1.0 / m if m > 0 else np.nan
    if stride > 1:
        out = pd.Series(out).ffill().to_numpy()
    return out


def drawdown_speed(x: np.ndarray, sigma: np.ndarray, window: int = 5) -> np.ndarray:
    """Trailing W-bar cumulative return in units of sigma*sqrt(W) (negative = fast sell-off)."""
    cum = pd.Series(x).rolling(window, min_periods=window).sum().to_numpy()
    return cum / (sigma * np.sqrt(window) + 1e-12)


def smooth_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Drop stress runs shorter than min_run and fill normal gaps shorter than min_run."""
    m = mask.copy()
    if min_run <= 1:
        return m
    for target in (True, False):          # first remove short stress runs, then short gaps
        t = 0
        while t < len(m):
            if m[t] == target:
                s = t
                while t < len(m) and m[t] == target:
                    t += 1
                if t - s < min_run:
                    m[s:t] = not target
            else:
                t += 1
    return m


@dataclass
class RegimeLabeler:
    vol_window: int = 20
    vol_quantile: float = 0.8
    tail_window: int = 250
    tail_k_frac: float = 0.1
    tail_alpha_crit: float = 3.0
    tail_quantile: Optional[float] = 0.1
    tail_stride: int = 1
    dd_window: int = 5
    dd_k: float = 2.5
    min_run: int = 3
    use: Sequence[str] = ("vol", "tail", "dd")
    vol_thr_: Optional[float] = None
    tail_thr_: Optional[float] = None

    def fit(self, r: np.ndarray, train_idx: np.ndarray) -> "RegimeLabeler":
        vol = rolling_std(r, self.vol_window)
        self.vol_thr_ = float(np.nanquantile(vol[train_idx], self.vol_quantile))
        if "tail" in self.use:
            tail = hill_tail_index(r, self.tail_window, self.tail_k_frac, self.tail_stride)
            ok = np.isfinite(tail[train_idx])
            self.tail_thr_ = float(np.nanquantile(tail[train_idx][ok], self.tail_quantile)) if (self.tail_quantile is not None and ok.any()) else float(self.tail_alpha_crit)
        return self

    def transform(self, r: np.ndarray) -> Dict[str, np.ndarray]:
        assert self.vol_thr_ is not None, "call fit(r, train_idx) first"
        vol = rolling_std(r, self.vol_window)
        stress = np.zeros(len(r), dtype=bool)
        out: Dict[str, np.ndarray] = {"vol": vol}
        if "vol" in self.use:
            stress |= np.nan_to_num(vol >= self.vol_thr_, nan=False)
        if "tail" in self.use:
            tail = hill_tail_index(r, self.tail_window, self.tail_k_frac, self.tail_stride)
            out["tail"] = tail
            stress |= np.nan_to_num(tail <= (self.tail_thr_ if self.tail_thr_ is not None else self.tail_alpha_crit), nan=False)
        if "dd" in self.use:
            dd = drawdown_speed(r, vol, self.dd_window)
            out["dd"] = dd
            stress |= np.nan_to_num(dd <= -self.dd_k, nan=False)
        stress = smooth_runs(stress, self.min_run)
        out["env"] = stress.astype(np.int8)
        out["stress_score"] = np.nan_to_num(vol / self.vol_thr_, nan=0.0)   # continuous intensity for decile plots
        return out

    def fit_transform(self, r: np.ndarray, train_idx: np.ndarray) -> Dict[str, np.ndarray]:
        return self.fit(r, train_idx).transform(r)

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "RegimeLabeler":
        keys = {f.name for f in cls.__dataclass_fields__.values()} - {"vol_thr_", "tail_thr_"}
        return cls(**{k: (tuple(v) if k == "use" else v) for k, v in (cfg or {}).items() if k in keys})


def apply_calendar(times: np.ndarray, env: np.ndarray, calendar: Sequence[Tuple[Any, Any]], mode: str = "or") -> np.ndarray:
    """Override / augment rule-based labels with curated [start, end] windows (bar indices, or ISO dates if `times` are ns)."""
    env = env.copy()
    cal_mask = np.zeros(len(env), dtype=bool)
    for start, end in calendar:
        if isinstance(start, str):
            ts = pd.to_datetime(times)
            cal_mask |= (ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end).normalize() + pd.Timedelta(days=1))   # end day inclusive (intraday bars)
        else:
            cal_mask[int(start) : int(end) + 1] = True
    if mode == "replace":
        return cal_mask.astype(np.int8)
    return (env.astype(bool) | cal_mask).astype(np.int8)


def _runs(env: np.ndarray) -> List[Tuple[int, int]]:
    runs, t = [], 0
    while t < len(env):
        if env[t]:
            s = t
            while t < len(env) and env[t]:
                t += 1
            runs.append((int(s), int(t - 1)))
        else:
            t += 1
    return runs


def write_stress_split(path: str, dataset: str, env: np.ndarray, times: np.ndarray, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist the Normal/Stress partition under data/stress_splits/<dataset>.json (bar ranges + timestamps)."""
    runs = _runs(env.astype(bool))
    rec = {"dataset": dataset, "n_bars": int(len(env)), "stress_frac": float(env.mean()),
           "stress_runs": [{"start": s, "end": e, "t_start": str(times[s]), "t_end": str(times[e])} for s, e in runs]}
    rec.update(extra or {})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def load_stress_split(path: str) -> Optional[List[Tuple[int, int]]]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rec = json.load(f)
    return [(r["start"], r["end"]) for r in rec.get("stress_runs", [])]
