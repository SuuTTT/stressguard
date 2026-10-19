"""Dataset 10 - Synthetic Multi-Regime Financial Simulator (Markov regime switching x Merton jump-diffusion x Hawkes).

Controlled ground truth for the stress-collapse mechanism. Per bar t a stress intensity u_t in [0, 1] (0 = calm) drives:
    volatility        sigma_{t,i} = sigma_i (1 + vol_mult u_t)
    jump intensity    lambda_{t,i} = lambda_0 (1 + jump_mult u_t) + Hawkes self-excitation (alpha, beta)
    common crashes    with prob ~ lambda_0 jump_mult u_t common_frac the whole universe jumps together (contagion)
    LOB depletion     depth x (1 - 0.8 u_t), spread x (1 + spread_mult u_t), random halts when u_t > 0.7
Crash event        = down-jump of size >= crash_thr * sigma_i at bar t_j. Label y = 1 on [t_j - lead, t_j - 1].
INVARIANT precursor  ofi: rises linearly over the `lead` bars before every crash with the same mechanism in every regime
                     (noise grows with u_t -> SNR drops under stress, as in real order flow).
SPURIOUS feature     sp = beta(u_t) y + N(0,1) with beta(u) = sp_beta0 (1 - sp_flip u): strongly predictive in calm
                     markets, uninformative or sign-flipped in severe stress. ERM latches onto it; IRM should not.
Context extras       ivskew: noisy causal proxy of the current regime (what an options skew gives you), regime prior only.
Stress episodes are planted with a fixed severity cycle; a test window is out-of-distribution only when its worst episode exceeds
the worst one in its training window (not guaranteed for every fold: report per-fold train/test max severity, meta["episodes"]).
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd

from warn2trade.data import Panel, BaseDatasetBuilder
from warn2trade.utils.registry import DATASETS


@DATASETS.register("synthetic_regime")
class SyntheticRegimeBuilder(BaseDatasetBuilder):
    name, freq, bars_per_year, horizons = "synthetic_regime", "daily", 252, (1, 2, 3, 5, 10, 20)

    def build_panel(self) -> Panel:
        c = self.cfg
        rng = np.random.default_rng(int(c.get("seed", 0)))
        T, N = int(c.get("T", 6000)), int(c.get("N", 12))
        lead = int(c.get("lead", 3))
        # ---------------------------------------------------------------- stress intensity path
        u = np.zeros(T)
        episodes = []
        sevs = list(c.get("episode_severities", [0.35, 0.9, 0.55, 1.0, 0.45, 0.85, 0.6, 0.95]))
        gap_lo, gap_hi = c.get("episode_gap", [350, 650])
        len_lo, len_hi = c.get("episode_len", [40, 120])
        t, k = int(c.get("warmup", 300)), 0
        while True:
            t += int(rng.integers(gap_lo, gap_hi))
            L = int(rng.integers(len_lo, len_hi))
            if t + L >= T - 100:
                break
            s = sevs[k % len(sevs)]
            k += 1
            x = np.arange(L) / L
            shape = np.exp(-((x - 0.25) / 0.35) ** 2)
            u[t : t + L] = np.maximum(u[t : t + L], s * shape / shape.max())
            episodes.append({"start": int(t), "end": int(t + L), "severity": float(s)})
            t += L
        env_true = (u > float(c.get("stress_cut", 0.2))).astype(np.int8)
        # ---------------------------------------------------------------- vol, depth, spread
        sigma0 = rng.uniform(0.008, 0.02, N)
        sig = sigma0[None, :] * (1.0 + float(c.get("vol_mult", 3.0)) * u[:, None])
        depth = (1.0 - 0.8 * u)[:, None] * np.exp(0.15 * rng.standard_normal((T, N)))
        spread = np.tile(rng.uniform(5e-4, 2e-3, N), (T, 1)) * (1.0 + float(c.get("spread_mult", 6.0)) * u[:, None]) * np.exp(0.1 * rng.standard_normal((T, N)))
        # ---------------------------------------------------------------- Hawkes jumps + common crashes
        lam0, jump_mult = float(c.get("jump_rate", 0.004)), float(c.get("jump_mult", 12.0))
        a_h, b_h = float(c.get("hawkes_alpha", 0.3)), float(c.get("hawkes_beta", 0.5))
        p_up, jmean = float(c.get("p_up", 0.25)), float(c.get("jump_mean", 4.0))
        common_frac = float(c.get("common_frac", 0.5))
        exc = np.zeros(N)
        jumps = np.zeros((T, N))
        common = np.zeros(T, dtype=bool)
        for tt in range(T):
            lam = np.clip(lam0 * (1.0 + jump_mult * u[tt]) + exc, 0.0, 0.5)
            fire = rng.random(N) < lam
            if rng.random() < lam0 * jump_mult * u[tt] * common_frac:
                fire |= rng.random(N) < 0.7
                common[tt] = True
            if fire.any():
                sign = np.where(rng.random(N) < p_up, 1.0, -1.0)
                size = (jmean + 3.0 * u[tt]) * np.abs(rng.normal(1.0, 0.3, N)) * sigma0
                jumps[tt] = np.where(fire, sign * size, 0.0)
            exc = exc * np.exp(-b_h) + a_h * fire
        ret = sig * rng.standard_normal((T, N)) + jumps - 0.0005 * u[:, None]
        mid = 100.0 * np.exp(np.cumsum(ret, axis=0))
        # ---------------------------------------------------------------- crash labels + invariant precursor
        crash = jumps <= -float(c.get("crash_thr", 3.0)) * sigma0[None, :]
        labels = np.zeros((T, N), dtype=np.int8)
        btp = np.full((T, N), np.nan)
        noise_mult, a_ofi = float(c.get("noise_mult", 0.5)), float(c.get("ofi_strength", 2.0))
        ofi = rng.standard_normal((T, N)) * (1.0 + noise_mult * u[:, None])
        for t_j, i in zip(*np.nonzero(crash)):
            lo = max(0, t_j - lead)
            if lo == t_j:
                continue
            labels[lo:t_j, i] = 1
            btp[lo:t_j, i] = t_j - np.arange(lo, t_j)
            ofi[lo:t_j, i] += a_ofi * np.linspace(0.4, 1.0, t_j - lo)
        # ---------------------------------------------------------------- spurious, stress-dependent feature
        beta = float(c.get("sp_beta0", 2.0)) * (1.0 - float(c.get("sp_flip", 1.6)) * u)
        sp = beta[:, None] * labels + rng.standard_normal((T, N))
        # ---------------------------------------------------------------- observable features
        rdf = pd.DataFrame(ret)
        rv20 = rdf.rolling(20, min_periods=10).std().to_numpy()
        rv20 = np.where(np.isnan(rv20), sigma0[None, :], rv20)
        zret = ret / rv20
        big = (np.abs(ret) > 3.0 * rv20).astype(float)
        njump = pd.DataFrame(big).rolling(20, min_periods=1).sum().to_numpy()
        price_feat = np.stack([zret, ofi, sp, np.log(depth), np.log(spread * 1e3), np.log(rv20 / sigma0[None, :]), njump], -1).astype(np.float32)
        liq = np.stack([spread / 2.0, depth, 0.02 * depth], -1).astype(np.float32)
        tradable = ~(rng.random((T, N)) < 0.15 * np.clip((u - 0.7) / 0.3, 0.0, 1.0)[:, None])
        # synthetic 5-level book: [bid_px_rel, bid_sz, ask_px_rel, ask_sz] x 5
        levels = int(c.get("lob_levels", 5))
        lob = np.zeros((T, N, levels * 4), dtype=np.float32)
        for l in range(levels):
            lob[..., 4 * l + 0] = -(spread / 2.0 + l * 1e-4)
            lob[..., 4 * l + 1] = depth * np.exp(-0.3 * l) * np.exp(0.2 * rng.standard_normal((T, N)))
            lob[..., 4 * l + 2] = spread / 2.0 + l * 1e-4
            lob[..., 4 * l + 3] = depth * np.exp(-0.3 * l) * np.exp(0.2 * rng.standard_normal((T, N)))
        # options-like skew proxy: noisy EWMA of the CURRENT stress intensity (no look-ahead), shared across assets
        ivskew = pd.Series(0.6 * u + 0.3 * rng.standard_normal(T)).ewm(span=5).mean().to_numpy()   # causal: current/past intensity only
        ctx_extra = np.repeat(ivskew[:, None, None], N, axis=1).astype(np.float32)
        panel = Panel(times=np.arange(T, dtype=np.int64), assets=[f"SYN{i:02d}" for i in range(N)], mid=mid,
                      price_feat=price_feat, liq=liq, tradable=tradable, labels=labels, bars_to_peak=btp, lob=lob,
                      meta={"freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": list(self.horizons),
                            "env_true": env_true, "u": u.astype(np.float32), "episodes": episodes, "n_crashes": int(crash.sum()),
                            "common_crash_bars": int(common.sum()), "feature_names": ["zret", "ofi", "sp", "log_depth", "log_spread", "log_rv20", "njump20"],
                            "ctx_extra": ctx_extra, "ctx_extra_names": ["ivskew"], "spurious_beta": beta.astype(np.float32)})
        panel.validate()
        return panel
