"""Score-only baseline contract. A baseline produces an anomaly score per decision point; the SAME decision layer
(val-fitted static threshold = the F1-tuned rule prior work uses, or our plug-in dynamic rule) and the SAME overlay backtest
turn scores into hedges, so the comparison isolates (i) representation robustness and (ii) the decision rule.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

from warn2trade.backtest import metrics as M


class ScoreBaseline:
    name: str = "base"
    paper: str = ""
    official_repo: str = ""
    status: str = "implemented"       # implemented | reimplementation | wrapper-needed | placeholder-unverified
    substitute: str = ""

    def __init__(self, cfg: Dict[str, Any], dims: Dict[str, int], device: torch.device):
        self.cfg, self.dims, self.device = cfg, dims, device

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:  # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def score(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def to(self, device):
        self.device = device
        return self


def calibrate_scores(val_scores: np.ndarray, val_y: np.ndarray) -> Tuple[float, float]:
    """Platt scaling: p = sigmoid(a * score + b) fitted on the validation fold so the dynamic rule can use calibrated probabilities."""
    ok = np.isfinite(val_scores)
    if val_y[ok].sum() == 0 or val_y[ok].sum() == ok.sum():
        return 1.0, 0.0
    s = val_scores[ok].astype(np.float64)
    mu, sd = float(s.mean()), float(s.std()) + 1e-8                     # standardise first: Platt on raw logits is ill-conditioned
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(((s - mu) / sd)[:, None], val_y[ok])
    a, b = float(lr.coef_[0, 0]) / sd, float(lr.intercept_[0]) - float(lr.coef_[0, 0]) * mu / sd
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        return 1.0, 0.0
    return a, b


def fit_static_threshold(val_p: np.ndarray, val_y: np.ndarray) -> float:
    _, thr = M.best_f1(val_y, val_p)
    return float(thr)
