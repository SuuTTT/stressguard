"""Real-data builders that run on files already present in this checkout (no new downloads).

binance_um   Binance USD-M perpetual futures BTCUSDT / ETHUSDT / SOLUSDT, public Binance Vision 5-minute klines
             (data_raw/zerotrigger/binance_um_5m, 2025-06 .. 2026-09), resampled to `bar_min`-minute bars. Contains the
             2025-10-10 liquidation cascade. Order-flow precursor = taker-buy imbalance.
us_equity    FRED daily NASDAQ Composite (1990+), S&P 500 and DJIA (2016+, NaN before), with VIX, 10y-2y term slope and WTI oil
             as market-wide context (lagged one day). Crises: 1998, 2000-02, 2008, 2010, 2011, 2015, 2018, 2020, 2022, 2025.
             Note: FRED redistributes S&P / Dow Jones index data under the index owners' terms; cite FRED series ids.

Crash label (both): y_t = 1 if the worst drawdown over bars (t, t+H] reaches k_crash * sigma_t * sqrt(H), sigma_t = trailing
realised vol. Forward-looking by construction; a training target only. All features are trailing.
Spread / depth for the cost model are PROXIES (no quotes in these files): half-spread = max(floor, spread_k * bar range) and
depth = trailing-median-relative volume. They are reported as such.
"""
from __future__ import annotations
import glob
import os
from typing import List
import numpy as np
import pandas as pd

from warn2trade.data import Panel, BaseDatasetBuilder
from warn2trade.utils.registry import DATASETS


def _crash_labels(mid: np.ndarray, sigma: np.ndarray, H: int, k: float) -> np.ndarray:
    lm = np.log(mid)
    T = len(lm)
    worst = np.full(mid.shape, np.inf)
    for h in range(1, H + 1):
        worst[: T - h] = np.minimum(worst[: T - h], lm[h:] - lm[: T - h])
    worst[T - H :] = np.nan
    with np.errstate(invalid="ignore"):
        y = (-worst >= k * sigma * np.sqrt(H)) & np.isfinite(worst) & np.isfinite(sigma)
    return y.astype(np.int8)


def _z(x: pd.DataFrame, w: int) -> pd.DataFrame:
    m, s = x.rolling(w, min_periods=w // 2).mean(), x.rolling(w, min_periods=w // 2).std()
    return ((x - m) / (s + 1e-12)).clip(-8, 8)


@DATASETS.register("binance_um")
class BinanceUMBuilder(BaseDatasetBuilder):
    name, freq = "binance_um", "15min"
    bars_per_year = 365 * 24 * 4
    horizons = (1, 2, 4, 8, 16, 32)

    def build_panel(self) -> Panel:
        c = self.cfg
        root = c.get("root_binance", "data_raw/zerotrigger/binance_um_5m")
        syms: List[str] = list(c.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]))
        bar = f"{int(c.get('bar_min', 15))}min"
        frames = {}
        for s in syms:
            files = sorted(glob.glob(os.path.join(root, s, "*.csv")))
            if not files:
                raise FileNotFoundError(f"no klines for {s} under {root}")
            cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
            rd = lambda f: (lambda d0: d0 if "open_time" in d0.columns else pd.read_csv(f, header=None, names=cols))(pd.read_csv(f))
            d = pd.concat([rd(f) for f in files], ignore_index=True)
            d = d[pd.to_numeric(d["open_time"], errors="coerce").notna()]
            d.index = pd.to_datetime(d["open_time"].astype("int64"), unit="ms")
            d = d[~d.index.duplicated()].sort_index()
            num = d[["open", "high", "low", "close", "volume", "count", "taker_buy_volume"]].astype(float)
            r = num.resample(bar, label="right", closed="left")
            frames[s] = pd.DataFrame({"close": r["close"].last(), "high": r["high"].max(), "low": r["low"].min(), "volume": r["volume"].sum(),
                                      "count": r["count"].sum(), "taker": r["taker_buy_volume"].sum()})
        idx = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
        F = {s: f.loc[idx].ffill() for s, f in frames.items()}
        P = lambda col: pd.DataFrame({s: F[s][col] for s in syms})
        close, high, low, vol, cnt, taker = P("close"), P("high"), P("low"), P("volume"), P("count"), P("taker")
        ret = np.log(close).diff().fillna(0.0)
        day = int(c.get("vol_window_bars", 96))
        rv = ret.rolling(day, min_periods=day // 2).std().bfill()
        imb = (2.0 * taker / vol.replace(0, np.nan) - 1.0).fillna(0.0)
        feats = {
            "zret": (ret / rv).clip(-10, 10),
            "ofi": imb,                                                    # taker-buy imbalance (order-flow precursor)
            "ofi_ema": imb.ewm(span=8).mean(),
            "logvol_z": _z(np.log1p(vol), 4 * day),
            "count_z": _z(np.log1p(cnt), 4 * day),
            "range_z": (np.log(high / low) / rv).clip(0, 20),
            "rv_ratio": np.log(ret.rolling(8, min_periods=4).std().bfill() / rv).clip(-4, 4),
            "dd_day": (np.log(close) - np.log(close.rolling(day, min_periods=1).max())) / (rv * np.sqrt(day)),
        }
        names = list(feats)
        price_feat = np.stack([feats[k].to_numpy() for k in names], -1).astype(np.float32)
        hs = np.maximum(float(c.get("spread_floor", 1e-4)), float(c.get("spread_k", 0.05)) * np.log(high / low).to_numpy())
        depth = (vol / vol.rolling(4 * day, min_periods=day).median()).bfill().clip(0.05, 20).to_numpy()
        liq = np.stack([hs, depth, 0.01 * depth], -1).astype(np.float32)
        mid = close.to_numpy()
        H, k = int(c.get("crash_H", 8)), float(c.get("crash_k", 3.0))
        labels = _crash_labels(mid, rv.to_numpy(), H, k)
        T, N = mid.shape
        panel = Panel(times=pd.DatetimeIndex(idx).asi8.astype(np.int64), assets=syms, mid=mid, price_feat=np.nan_to_num(price_feat), liq=liq,
                      tradable=np.ones((T, N), bool), labels=labels, bars_to_peak=np.full((T, N), np.nan),
                      meta={"freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": list(self.horizons), "feature_names": names,
                            "label_rule": f"worst drawdown over {H} bars >= {k} sigma sqrt(H)", "spread_proxy": True})
        panel.validate()
        return panel


@DATASETS.register("us_equity")
class USEquityDailyBuilder(BaseDatasetBuilder):
    name, freq = "us_equity", "daily"
    bars_per_year = 252
    horizons = (1, 2, 3, 5, 10, 20)

    def _read(self, sid: str) -> pd.Series:
        for root in self.cfg.get("fred_roots", ["data_raw/sentinel/fred", "data_raw/zerotrigger/fred"]):
            p = os.path.join(root, f"{sid}.csv")
            if os.path.exists(p):
                d = pd.read_csv(p)
                s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
                s.index = pd.to_datetime(d.iloc[:, 0])
                return s
        raise FileNotFoundError(f"FRED series {sid} not found")

    def build_panel(self) -> Panel:
        c = self.cfg
        syms = list(c.get("symbols", ["NASDAQCOM", "SP500", "DJIA"]))
        start = pd.Timestamp(c.get("start", "1990-01-02"))
        px = pd.DataFrame({s: self._read(s) for s in syms})
        px = px[px.index >= start]
        px = px[px["NASDAQCOM"].notna()] if "NASDAQCOM" in px else px.dropna(how="all")
        px = px.ffill(limit=3)
        ret = np.log(px).diff()
        rv = ret.rolling(60, min_periods=20).std()
        vix = self._read("VIXCLS").reindex(px.index).ffill()
        slope = (self._read("DGS10") - self._read("DGS2")).reindex(px.index).ffill()
        oil = np.log(self._read("DCOILWTICO").where(lambda s: s > 0)).diff(5).reindex(px.index).ffill()
        feats = {
            "zret": (ret / rv).clip(-10, 10),
            "zret5": (ret.rolling(5).sum() / (rv * np.sqrt(5))).clip(-10, 10),
            "rv_ratio": np.log(ret.rolling(5, min_periods=3).std() / rv).clip(-4, 4),
            "dd60": (np.log(px) - np.log(px.rolling(60, min_periods=10).max())) / (rv * np.sqrt(60)),
            "absret_z": _z(ret.abs(), 250),
        }
        names = list(feats)
        price_feat = np.nan_to_num(np.stack([feats[k].to_numpy() for k in names], -1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        # zero-fill missing features (warm-up bars, and windows that start before SP500/DJIA exist on FRED, 2016-10);
        # rows where the asset itself does not exist are still dropped through NaN mid
        lag = lambda s: s.shift(1)                                           # market-wide context lagged one day (publication timing)
        extra = pd.DataFrame({"vix_z": _z(np.log(vix), 250), "vix_chg": np.log(vix).diff(5), "term_slope": slope, "oil_5d": oil}).apply(lag)
        T, N = px.shape
        ctx_extra = np.repeat(np.nan_to_num(extra.to_numpy(dtype=np.float32))[:, None, :], N, axis=1)
        mid = px.to_numpy()
        hs = np.full((T, N), float(c.get("half_spread", 1e-4)))
        liq = np.stack([hs, np.ones((T, N)), np.full((T, N), 0.01)], -1).astype(np.float32)
        H, k = int(c.get("crash_H", 10)), float(c.get("crash_k", 2.5))
        labels = _crash_labels(mid, rv.to_numpy(), H, k)
        panel = Panel(times=px.index.asi8.astype(np.int64), assets=syms, mid=mid, price_feat=price_feat, liq=liq,
                      tradable=~np.isnan(mid), labels=labels, bars_to_peak=np.full((T, N), np.nan),
                      meta={"freq": self.freq, "bars_per_year": self.bars_per_year, "horizons": list(self.horizons), "feature_names": names,
                            "ctx_extra": ctx_extra, "ctx_extra_names": list(extra.columns), "label_rule": f"worst drawdown over {H} days >= {k} sigma sqrt(H)"})
        panel.validate()
        return panel
