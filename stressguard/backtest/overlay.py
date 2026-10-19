"""Hedge-overlay execution. The investor holds the asset; the detector's hedge ratio h in [0, h_max] reduces exposure for `tau` bars.

Timing (latency delta >= 1): decision at bar t -> hedge filled at the close of t + delta -> lifted at the close of t + delta + tau
(delayed to the next tradable bar if the exit bar is halted). If the entry bar is untradable the hedge is NOT placed (frozen).
    R      = cumulative log return from fill to exit
    pnl    = - h R - cost_in(h) - cost_out(h)        incremental PnL of the overlay vs. staying unhedged
    hedged = (1 - h) R - costs                        hedged holding-period return (for drawdown protection metrics)
Costs come from warn2trade.backtest.CostModel: commission + half-spread + sqrt market impact (depth-aware -> grows under stress) + delay drift.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
import torch

from warn2trade.backtest import CostModel
from warn2trade.backtest.engine import _next_tradable_index


def overlay_pnl(h: torch.Tensor, batch: Dict[str, torch.Tensor], cost_model: CostModel, latency: int, tau: int) -> Dict[str, torch.Tensor]:
    fut_ret, fut_liq, fut_trad, sigma = batch["fut_ret"], batch["fut_liq"], batch["fut_tradable"], batch["sigma"]
    B, Hmax = fut_ret.shape
    entry_j, exit_j = latency - 1, latency + tau - 1
    assert exit_j < Hmax, "fut_ret window too short for latency + tau"
    cum = torch.cumsum(fut_ret, 1)
    nxt = _next_tradable_index(fut_trad)
    exit_idx = nxt[:, exit_j]
    bi = torch.arange(B, device=h.device)
    R = cum[bi, exit_idx] - cum[:, entry_j]
    one = torch.ones_like(h)
    active = h > 0
    h_c = torch.where(active, h, torch.full_like(h, 1e-6))        # sqrt impact has an infinite derivative at 0: evaluate the cost on a
    cost_in = cost_model(h_c, fut_liq[:, entry_j, 0], sigma, fut_liq[:, entry_j, 2], one)        # strictly positive copy and mask
    cost_out = cost_model(h_c, fut_liq[bi, exit_idx, 0], sigma, fut_liq[bi, exit_idx, 2], one)
    frozen = ~fut_trad[:, entry_j]
    cost = torch.where(frozen | ~active, torch.zeros_like(h), cost_in + cost_out)
    pnl = torch.where(frozen, torch.zeros_like(h), -h * R - cost)
    h_eff = torch.where(frozen, torch.zeros_like(h), h)
    return {"pnl": pnl, "R": R, "cost": cost, "frozen": frozen, "h_eff": h_eff, "exit_j": exit_idx, "hedged": (1.0 - h_eff) * R - cost}


class OverlayBacktester:
    def __init__(self, cost_model: CostModel, latency: int, tau: int):
        self.cost_model, self.latency, self.tau = cost_model, latency, tau

    @torch.no_grad()
    def collect(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> pd.DataFrame:
        r = overlay_pnl(out["h"], batch, self.cost_model, self.latency, self.tau)
        g = lambda k: r[k].detach().cpu().numpy()
        b = lambda k: batch[k].detach().cpu().numpy()
        o = lambda k: out[k].detach().cpu().numpy() if k in out else np.full(len(r["pnl"]), np.nan)
        return pd.DataFrame({"t": b("t"), "asset": b("asset"), "y": b("y"), "env": b("env"), "score": o("p"), "p_raw": o("p_raw"), "thr": o("thr"),
                             "h": g("h_eff"), "h_dec": out["h"].detach().cpu().numpy(), "sev_hat": o("sev_hat"), "sev": b("sev"), "sigma": b("sigma"), "gamma": b("gamma"),
                             "R": g("R"), "pnl": g("pnl"), "hedged": g("hedged"), "cost": g("cost"), "frozen": g("frozen"),
                             "exit_t": b("t") + 1 + g("exit_j"), "stress_score": b("stress_score") if "stress_score" in batch else np.nan,
                             "stress_true": b("stress_true") if "stress_true" in batch else np.nan})

    @staticmethod
    def hedge_book(df: pd.DataFrame, panel, test_idx: np.ndarray, cost_model: CostModel, latency: int, tau: int, h_max: float = 1.0,
                   vol_window: int = 60) -> Dict[str, np.ndarray]:
        """Bar-level hedge book over the test window with ONE position per asset, capped at h_max.

        A decision h at bar t asks for a hedge of size h on bars [t+latency, t+latency+tau). Overlapping requests on the same asset
        do not stack: the target position is the max active request (alerts extend / top up the hedge). The position held over
        bar b earns -pos * r_b; costs are charged on |change in position| at the bar's liquidity; a halted asset keeps its position.
        Requests whose window leaves the test window are truncated at its end (no post-window returns are booked).
        Series are per unit of an equal-weight portfolio (1/N per asset)."""
        from warn2trade.data.labels import realized_vol
        t0, n, N = int(test_idx[0]), len(test_idx), panel.mid.shape[1]
        tgt = np.zeros((n, N))
        d = df[df["h_dec"] > 0]
        t, a, h = d["t"].to_numpy().astype(int), d["asset"].to_numpy().astype(int), np.minimum(d["h_dec"].to_numpy(), h_max)
        for k in range(tau):
            b = t - t0 + latency + k
            ok = (b >= 0) & (b < n)
            np.maximum.at(tgt, (b[ok], a[ok]), h[ok])
        lr = np.nan_to_num(np.log(panel.mid[test_idx]) - np.log(panel.mid[test_idx - 1]))
        sigma = np.nan_to_num(realized_vol(panel.mid, vol_window)[test_idx])
        liq, trad = panel.liq[test_idx], panel.tradable[test_idx]
        held, pos = np.zeros(N), np.zeros((n, N))
        overlay, cost, to = np.zeros(n), np.zeros(n), np.zeros(n)
        one = torch.ones(N, dtype=torch.float64)
        for b in range(n):
            overlay[b] = -(held * lr[b]).sum() / N                    # position carried into bar b earns bar b's return
            new = np.where(trad[b], tgt[b], held)
            dw = np.abs(new - held)
            if dw.any():
                c = cost_model(torch.as_tensor(np.where(dw > 0, dw, 1e-6)), torch.as_tensor(liq[b, :, 0].astype(np.float64)),
                               torch.as_tensor(sigma[b]), torch.as_tensor(liq[b, :, 2].astype(np.float64)), one).numpy()
                cost[b] = np.where(dw > 0, c, 0.0).sum() / N
                to[b] = dw.sum() / N
            held = new
            pos[b] = held
        return {"overlay": overlay - cost, "cost": cost, "turnover": to, "pos": pos, "base": lr.mean(axis=1),
                "frozen_rate": float(((tgt > 0) & ~trad).sum() / max((tgt > 0).sum(), 1))}
