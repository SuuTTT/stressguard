"""Regime-sliced metric matrix (everything reported separately for Normal, Stress and Full sample).

A. Detection      AUC-ROC, AUC-PR, P@1%, R@1%, Tail-F1 (F1 at the top-1% operating point), best-F1, alert precision/recall at the
                  model's own decision rule; Delta-AUC_stress = AUC_normal - AUC_stress; AUC by stress-score decile (Fig. 1B).
B. Trading        hedged vs unhedged equal-weight portfolio: ARR, SR, SoR, Calmar, MDD, VaR99, ES99, turnover, net PnL,
                  evaluated on all bars and on Stress bars only; MDD reduction = 1 - |MDD_hedged| / |MDD_base|.
C. Economic cost  FN cost = c_FN y (1-d) s, FP cost = c_FP (1-y) d gamma for the dynamic rule, the val-fitted static rule and an
                  oracle static rule, swept over c_FN / c_FP ratios (Fig. "Economic Cost Curve").
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from warn2trade.backtest import metrics as M
from ..backtest.overlay import OverlayBacktester

SLICES = ("full", "normal", "stress", "severe", "true_severe")
SEVERE_Q = 0.95          # "severe" = top-5 % of the test window by stress score (deployable definition)
TRUE_SEVERE_U = 0.6      # simulator ground truth: stress intensity u >= 0.6 (where the shortcut's sign has flipped)


def slice_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    env = df["env"].to_numpy()
    out = {"full": np.ones(len(df), bool), "normal": env == 0, "stress": env == 1}
    if "stress_score" in df and np.isfinite(df["stress_score"].to_numpy()).any():
        ss = df["stress_score"].to_numpy()
        out["severe"] = ss >= np.nanquantile(ss, SEVERE_Q)
    if "stress_true" in df and np.isfinite(df["stress_true"].to_numpy()).any():
        out["true_severe"] = df["stress_true"].to_numpy() >= TRUE_SEVERE_U
    return out


def detection_block(y: np.ndarray, s: np.ndarray, k_frac: float = 0.01, d: Optional[np.ndarray] = None) -> Dict[str, float]:
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n = len(y)
    if n == 0:
        return {"n": 0}
    k = max(1, int(round(k_frac * n)))
    out = {"n": n, "n_pos": int(y.sum()), "AUC_ROC": M.auc_roc(y, s), "AUC_PR": M.auc_pr(y, s),
           "P@1%": M.precision_at_k(y, s, k), "R@1%": M.recall_at_k(y, s, k)}
    pred = np.zeros(n, bool)
    pred[np.argsort(-s)[:k]] = True
    tp = float((pred & (y == 1)).sum())
    pr, rc = tp / k, tp / max(y.sum(), 1.0)
    out["TailF1"] = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0
    out["F1_best"], out["thr_best"] = M.best_f1(y, s)
    if d is not None:
        d = np.asarray(d, bool)[ok]
        tp = float((d & (y == 1)).sum())
        out["alert_rate"] = float(d.mean())
        out["alert_precision"] = tp / max(d.sum(), 1.0)
        out["alert_recall"] = tp / max(y.sum(), 1.0)
    return out


def regime_detection(df: pd.DataFrame, k_frac: float = 0.01) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    d = (df["h_dec"].to_numpy() > 0) if "h_dec" in df else (df["h"].to_numpy() > 0)
    for name, m in slice_masks(df).items():
        out[name] = detection_block(df["y"].to_numpy()[m], df["score"].to_numpy()[m], k_frac, d[m]) if m.sum() > 0 else {"n": 0}
    out["dAUC_stress"] = out["normal"].get("AUC_ROC", np.nan) - out["stress"].get("AUC_ROC", np.nan)
    out["dAUCPR_stress"] = out["normal"].get("AUC_PR", np.nan) - out["stress"].get("AUC_PR", np.nan)
    for sl in ("severe", "true_severe"):
        out[f"dAUC_{sl}"] = out["normal"].get("AUC_ROC", np.nan) - out.get(sl, {}).get("AUC_ROC", np.nan)
    return out


def auc_by_decile(df: pd.DataFrame, col: str = "stress_score", n: int = 10) -> pd.DataFrame:
    x = df[col].to_numpy()
    if not np.isfinite(x).any() or np.nanstd(x) == 0:
        col, x = "sigma", df["sigma"].to_numpy()
    edges = np.nanquantile(x, np.linspace(0, 1, n + 1))
    rows = []
    for k in range(n):
        m = (x >= edges[k]) & ((x <= edges[k + 1]) if k == n - 1 else (x < edges[k + 1]))
        y, s = df["y"].to_numpy()[m], df["score"].to_numpy()[m]
        rows.append({"decile": k + 1, "lo": edges[k], "hi": edges[k + 1], "n": int(m.sum()), "n_pos": int(y.sum()),
                     "AUC_ROC": M.auc_roc(y, s) if m.sum() else np.nan, "AUC_PR": M.auc_pr(y, s) if m.sum() else np.nan, "by": col})
    return pd.DataFrame(rows)


def var_es(r: np.ndarray, q: float = 0.99) -> Tuple[float, float]:
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return float("nan"), float("nan")
    cut = np.quantile(r, 1.0 - q)
    tail = r[r <= cut]
    return float(-cut), float(-tail.mean()) if len(tail) else float("nan")


def trading_block(r: np.ndarray, bars_per_year: int, to: Optional[np.ndarray] = None) -> Dict[str, float]:
    d = M.summarize_trading(r, bars_per_year, to)
    d["VaR99"], d["ES99"] = var_es(r, 0.99)
    d["NetPnL"] = float(np.nansum(r))
    d["n_bars"] = int(len(r))
    return d


def stress_run_mdd(r: np.ndarray, mask: np.ndarray) -> float:
    """Worst drawdown inside any contiguous run of `mask` bars (equity reset at each run start)."""
    worst, b = 0.0, 0
    while b < len(mask):
        if mask[b]:
            e = b
            while e < len(mask) and mask[e]:
                e += 1
            if e - b >= 2:
                worst = min(worst, M.max_drawdown(r[b:e]))
            b = e
        else:
            b += 1
    return float(worst)


def overlay_trading(df: pd.DataFrame, bt: Dict):
    """bt = dict(panel, test, env_bars, cost_model, latency, tau, bars_per_year, h_max, vol_window). Capped bar-level hedge book."""
    bpy = bt["bars_per_year"]
    hb = OverlayBacktester.hedge_book(df, bt["panel"], bt["test"], bt["cost_model"], bt["latency"], bt["tau"], bt.get("h_max", 1.0), bt.get("vol_window", 60))
    base, hedged = hb["base"], hb["base"] + hb["overlay"]
    alert = (df["h_dec"].to_numpy() > 0)
    out = {"base": trading_block(base, bpy), "hedged": trading_block(hedged, bpy, hb["turnover"]),
           "overlay": {"NetPnL": float(hb["overlay"].sum()), "cost": float(hb["cost"].sum()), "alert_rate": float(alert.mean()),
                       "hedge_rate": float((hb["pos"] > 0).mean()), "mean_pos": float(hb["pos"].mean()), "max_pos": float(hb["pos"].max()),
                       "frozen_rate": hb["frozen_rate"]}}
    m = bt["env_bars"].astype(bool)
    for name, mm in (("stress", m), ("normal", ~m)):
        if mm.sum() >= 5:
            out[f"base_{name}"], out[f"hedged_{name}"] = trading_block(base[mm], bpy), trading_block(hedged[mm], bpy)
            out[f"base_{name}"]["MDD"], out[f"hedged_{name}"]["MDD"] = stress_run_mdd(base, mm), stress_run_mdd(hedged, mm)   # per contiguous run
    for key, bkey, hkey in (("MDD_reduction", "base", "hedged"), ("MDD_reduction_stress", "base_stress", "hedged_stress")):
        if bkey in out and out[bkey]["MDD"] < 0:
            out[key] = 1.0 - abs(out[hkey]["MDD"]) / abs(out[bkey]["MDD"])
    return out, {"base": base, "hedged": hedged, "overlay": hb["overlay"]}


def cost_curve(df: pd.DataFrame, ratios: Sequence[float], c_fp: float = 1.0, n_grid: int = 200) -> pd.DataFrame:
    rows = []
    masks = slice_masks(df)
    y, s, sev, gam = df["y"].to_numpy(), df["score"].to_numpy(), df["sev"].to_numpy(), df["gamma"].to_numpy()
    sh = df["sev_hat"].to_numpy()
    st = df["static_thr"].to_numpy() if "static_thr" in df else None
    grid = np.unique(np.quantile(s[np.isfinite(s)], np.linspace(0, 1, n_grid)))
    for ratio in ratios:
        r = float(ratio) * c_fp                                    # c_FN = ratio * c_FP (same convention as ratio_sweep)
        rules = {"dynamic": s >= c_fp * gam / (c_fp * gam + r * sh)}
        if st is not None:
            rules["static_valF1"] = s >= st
        for name, m in masks.items():
            if m.sum() == 0:
                continue
            costs_oracle = [((r * y[m] * (1 - (s[m] >= g)) * sev[m]).mean() + (c_fp * (1 - y[m]) * (s[m] >= g) * gam[m]).mean()) for g in grid]
            rules["static_oracle"] = s >= grid[int(np.argmin(costs_oracle))]      # oracle fitted INSIDE the slice (true lower bound for static rules)
            for rule, d in rules.items():
                dm, ym = d[m], y[m]
                fn, fp = float((r * ym * (1 - dm) * sev[m]).mean()), float((c_fp * (1 - ym) * dm * gam[m]).mean())
                rows.append({"ratio": float(ratio), "rule": rule, "slice": name, "fn_cost": fn, "fp_cost": fp, "total": fn + fp,
                             "alert_rate": float(dm.mean()), "recall": float((dm & (ym == 1)).sum() / max(ym.sum(), 1))})
    return pd.DataFrame(rows)


def summarize_fold(df: pd.DataFrame, bt: Dict, k_frac: float = 0.01) -> Dict[str, Dict]:
    trading, _ = overlay_trading(df, bt)
    return {"detection": regime_detection(df, k_frac), "trading": trading}
