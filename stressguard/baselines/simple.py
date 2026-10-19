"""In-repo, runnable baselines representing the main anomaly-scoring families (fully implemented, no external code):
    zscore        rule: max |z| of the latest bar's features                              (classical control)
    iforest       Isolation Forest on [last bar features, window mean/std]                 (Liu et al. 2008)
    recon_ae      window reconstruction MLP auto-encoder trained on Normal bars; score = reconstruction error
                  (the reconstruction family: USAD / TranAD / Anomaly-Transformer-style criterion without association discrepancy)
    dlinear_recon DLinear one-step forecaster (Zeng et al. AAAI 2023); score = forecast-error norm   (forecasting-error family)
The ERM same-backbone detector (the collapse baseline) is configs/stressguard/ablations/erm.yaml, trained by the StressGuard trainer.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import IsolationForest

from warn2trade.train.trainer import to_device
from warn2trade.utils.registry import BASELINES
from .base import ScoreBaseline


def _feats(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    x = batch["price"]
    return torch.cat([x[:, -1], x.mean(1), x.std(1), batch["liq"]], -1)


@BASELINES.register("zscore")
class ZScore(ScoreBaseline):
    name, paper, status = "zscore", "rule-based control", "implemented"

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        xs = torch.cat([b["price"][:, -1] for b in train])
        self.mu, self.sd = xs.mean(0).to(self.device), xs.std(0).to(self.device) + 1e-6

    @torch.no_grad()
    def score(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return ((batch["price"][:, -1] - self.mu) / self.sd).abs().max(-1).values


@BASELINES.register("iforest")
class IForest(ScoreBaseline):
    name, paper, status = "iforest", "Liu, Ting, Zhou. Isolation Forest. ICDM 2008", "implemented"

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        xs = torch.cat([_feats(b) for b in train]).numpy()
        self.model = IsolationForest(n_estimators=int(self.cfg.get("n_estimators", 200)), contamination="auto", random_state=0).fit(xs)

    @torch.no_grad()
    def score(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        s = -self.model.score_samples(_feats(batch).cpu().numpy())
        return torch.as_tensor(s, dtype=torch.float32, device=batch["price"].device)


class _WindowAE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, latent: int = 16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, in_dim))

    def forward(self, x):
        return self.dec(self.enc(x))


@BASELINES.register("recon_ae")
class ReconAE(ScoreBaseline):
    name, paper, status = "recon_ae", "window reconstruction auto-encoder (USAD/TranAD-style criterion)", "reimplementation"

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        L, F = next(iter(train))["price"].shape[1:]
        self.net = _WindowAE(L * F, int(self.cfg.get("hidden", 128)), int(self.cfg.get("latent", 16))).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.cfg.get("lr", 1e-3)))
        for _ in range(int(self.cfg.get("epochs", 5))):
            for b in train:
                b = to_device(b, self.device)
                x = b["price"].flatten(1)
                m = b["y"] < 0.5 if self.cfg.get("normal_only", True) else torch.ones_like(b["y"], dtype=torch.bool)
                if m.sum() < 2:
                    continue
                loss = ((self.net(x[m]) - x[m]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def score(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["price"].flatten(1)
        return ((self.net(x) - x) ** 2).mean(-1)


@BASELINES.register("dlinear_recon")
class DLinearRecon(ScoreBaseline):
    name, paper, official_repo, status = "dlinear_recon", "Zeng et al. Are Transformers Effective for Time Series Forecasting? (DLinear) AAAI 2023", "https://github.com/cure-lab/LTSF-Linear", "reimplementation"

    def fit(self, train: DataLoader, val: Optional[DataLoader] = None) -> None:
        L, F = next(iter(train))["price"].shape[1:]
        self.L, self.F = L, F
        self.lin = nn.Linear(L - 1, 1).to(self.device)
        opt = torch.optim.Adam(self.lin.parameters(), lr=1e-3)
        for _ in range(int(self.cfg.get("epochs", 5))):
            for b in train:
                x = b["price"].to(self.device)
                pred = self.lin(x[:, :-1].transpose(1, 2)).squeeze(-1)
                loss = ((pred - x[:, -1]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def score(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["price"]
        pred = self.lin(x[:, :-1].transpose(1, 2)).squeeze(-1)
        return ((pred - x[:, -1]) ** 2).mean(-1)
