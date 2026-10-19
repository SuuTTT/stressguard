"""RegimeEventDataset: EventDataset + environment index, context prior and crash severity for every decision point."""
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from warn2trade.data import Panel, EventDataset
from warn2trade.utils.seed import seed_worker


def adverse_move(mid: np.ndarray, tau: int, latency: int = 0) -> np.ndarray:
    """s_t = max(0, -min_{1<=k<=tau} [log mid_{t+latency+k} - log mid_{t+latency}]): worst drawdown over the HEDGEABLE window
    (from the fill bar t+latency). latency=0 measures from the decision bar."""
    lm = np.log(mid)
    T = lm.shape[0]
    worst = np.full(mid.shape, np.inf)
    for k in range(1, tau + 1):
        worst[: T - k] = np.minimum(worst[: T - k], lm[k:] - lm[: T - k])
    worst[T - tau :] = np.nan
    s = np.clip(-worst, 0.0, None)
    if latency > 0:
        out = np.full(mid.shape, np.nan)
        out[: T - latency] = s[latency:]
        return out
    return s


class RegimeEventDataset(EventDataset):
    """Adds to each item:  env (int, 0 = normal / 1 = stress), ctx (Fc,), sev (crash severity |dP_{t+tau}| in log-return units)."""

    def __init__(self, panel: Panel, time_idx: np.ndarray, env: np.ndarray, ctx: np.ndarray, tau: int, lookback: int,
                 horizons: Sequence[int], latency: int = 1, exec_max: int = 4, vol_window: int = 60,
                 neg_subsample: Optional[float] = None, rng: Optional[np.random.Generator] = None,
                 stress_score: Optional[np.ndarray] = None, aux: Optional[Dict[str, np.ndarray]] = None):
        assert tau + latency <= max(horizons) + latency + exec_max, "tau must fit inside the fut_ret window"
        super().__init__(panel, time_idx, lookback, horizons, latency, exec_max, vol_window, True, neg_subsample, rng)
        self.env, self.ctx, self.tau = env.astype(np.int64), ctx.astype(np.float32), int(tau)
        self.stress_score = np.zeros(len(env), dtype=np.float32) if stress_score is None else np.nan_to_num(stress_score).astype(np.float32)
        self.aux = {k: np.asarray(v, dtype=np.float32) for k, v in (aux or {}).items()}      # (T,) per-bar extras, e.g. stress_true
        self._sev = adverse_move(panel.mid, tau, latency)
        ok = ~np.isnan(self._sev[self.index[:, 0], self.index[:, 1]])
        self.index = self.index[ok]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        t, i = self.index[idx]
        item["env"] = torch.tensor(int(self.env[t]))
        item["ctx"] = torch.as_tensor(self.ctx[t, i], dtype=torch.float32)
        item["sev"] = torch.tensor(float(self._sev[t, i]), dtype=torch.float32)
        item["stress_score"] = torch.tensor(float(self.stress_score[t]), dtype=torch.float32)
        for k, v in self.aux.items():
            item[k] = torch.tensor(float(v[t]), dtype=torch.float32)
        return item

    def env_counts(self) -> Dict[int, int]:
        e = self.env[self.index[:, 0]]
        return {int(k): int(v) for k, v in zip(*np.unique(e, return_counts=True))}

    def pos_rate_by_env(self) -> Dict[int, float]:
        e, y = self.env[self.index[:, 0]], self.p.labels[self.index[:, 0], self.index[:, 1]]
        return {int(k): float(y[e == k].mean()) for k in np.unique(e)}


def env_balanced_loader(ds: RegimeEventDataset, batch_size: int, num_workers: int = 0, seed: int = 42, balance: bool = True) -> DataLoader:
    """Training loader. With balance=True each environment is sampled with equal total probability so every mini-batch
    carries enough Stress samples for the per-environment IRM penalty (the rare-regime problem)."""
    g = torch.Generator()
    g.manual_seed(seed)
    if balance and len(ds.env_counts()) > 1:
        counts = ds.env_counts()
        e = ds.env[ds.index[:, 0]]
        w = torch.as_tensor([1.0 / counts[int(k)] for k in e], dtype=torch.double)
        sampler = WeightedRandomSampler(w, num_samples=len(ds), replacement=True, generator=g)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, worker_init_fn=seed_worker, drop_last=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g, drop_last=False)
