"""Datasets 1-9: builders for the real benchmarks. Each records the source, licence status, the bar/feature recipe and the
stress calendar used for the Normal/Stress partition. Builders raise with a precise instruction when raw files are absent;
`stress_calendar` is consumed by scripts/sg_make_stress_splits.py so the partition is reproducible and auditable.

All recipes produce the shared Panel: mid (T,N), price_feat (T,N,Fp), liq (T,N,3), tradable, labels, bars_to_peak, optional
lob (T,N,Fl) and meta["ctx_extra"] (T,N,k) for market-wide priors (IV skew, funding, MOVE).
Crash labels (unless a curated table exists): y_t = 1 if the worst forward drawdown within tau bars exceeds k_crash * sigma_t * sqrt(tau)
AND is accompanied by a volume / trade-count z-score > k_vol; extended `lead` bars backwards (warn2trade.data.labels.abnormal_move_labels).
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple
from warn2trade.data import Panel, BaseDatasetBuilder
from warn2trade.utils.registry import DATASETS


class RecipeBuilder(BaseDatasetBuilder):
    source: str = ""
    licence: str = ""
    required: Tuple[str, ...] = ()
    recipe: str = ""
    stress_calendar: List[Tuple[str, str]] = []

    def _require(self) -> None:
        missing = [f for f in self.required if not os.path.exists(os.path.join(self.root, f))]
        if missing:
            raise FileNotFoundError(f"[{self.name}] missing {missing} under {self.root}. Source: {self.source} ({self.licence}).\nRecipe: {self.recipe}")

    def build_panel(self) -> Panel:
        self._require()
        raise NotImplementedError(f"[{self.name}] raw files found; implement the parser described in `recipe`.")

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update({"source": self.source, "licence": self.licence, "stress_calendar": self.stress_calendar})
        return d


def _recipe(key, freq, bpy, horizons, source, licence, required, recipe, calendar):
    @DATASETS.register(key)
    class _R(RecipeBuilder):
        pass
    _R.__name__ = key
    _R.name, _R.freq, _R.bars_per_year, _R.horizons = key, freq, bpy, horizons
    _R.source, _R.licence, _R.required, _R.recipe, _R.stress_calendar = source, licence, tuple(required), recipe, calendar
    return _R


LOB_BPY = int(252 * 6.5 * 3600 * 10)    # 100 ms bars, 6.5 h NASDAQ session (calendar dates below are UTC days)
EQ_STRESS = [("2010-05-06", "2010-05-07"), ("2015-08-24", "2015-08-26"), ("2018-02-05", "2018-02-09"), ("2020-02-24", "2020-04-15"),
             ("2022-01-18", "2022-03-15"), ("2022-06-09", "2022-06-17"), ("2022-09-13", "2022-10-13"), ("2024-08-02", "2024-08-07"),
             ("2025-04-03", "2025-04-10")]   # extend with 2025-2026 tail events after verification against the volatility rule
_recipe("lobster_sg", "lob", LOB_BPY, (10, 20, 50, 100, 300, 600), "LOBSTER (lobsterdata.com) NASDAQ level-10 messages + order books",
        "academic licence", ["lobster"], "100 ms bars; mid=(b1+a1)/2; price_feat=[OFI (Cont et al. 2014), mid ret z, trade imbalance, queue imbalance, "
        "cancel ratio, depth slope]; lob=levels x [bid_px_rel, bid_sz, ask_px_rel, ask_sz]; liq=[half-spread/mid, touch depth/median, bar vol/day vol]; "
        "tradable=False at auctions or spread>20 ticks; crash labels from forward mid drawdown + trade-count burst; stress = calendar OR vol rule on 1-min mid.",
        [("2020-03-09", "2020-03-20"), ("2024-08-05", "2024-08-05")])
_recipe("fi2010", "lob", int(252 * 8.5 * 3600 * 10), (10, 20, 50, 100), "FI-2010 (Ntakaris et al. 2018) 5 Nordic stocks, 10 days, 10 levels",
        "public (CC BY)", ["fi2010"], "use the provided z-score normalised 144-d feature rows; mid from level-1 quotes; labels = provided k-horizon "
        "down-move class (treated as crash proxy) AND forward drawdown rule; stress partition = top-quintile realised vol days (no calendar: 2010-06 only).",
        [])
_recipe("spx_surface", "daily", 252, (1, 2, 3, 5, 10, 20), "OptionMetrics IvyDB / CBOE S&P 500 volatility surface (VIX, SKEW, VVIX, term structure)",
        "licensed (OptionMetrics) / CBOE public indices (VIX9D starts 2011-01: use VIX3M-VIX as term slope before) / CME ES futures (DataMine licence) for liquidity", ["spx_surface"], "daily SPX + VIX/SKEW/VVIX/VIX9D/VIX3M; price_feat=[ret z, VIX chg, skew chg, "
        "term slope, VVIX chg]; ctx_extra=[25d skew, term slope, VVIX]; liq from ES futures proxies; labels = forward 10-day drawdown > 2.5 sigma sqrt(10).",
        EQ_STRESS)
_recipe("binance_l2", "1min", 365 * 24 * 60, (1, 5, 15, 30, 60), "Binance BTC/USDT + ETH/USDT L2 depth snapshots (public REST/WebSocket archives, e.g. Tardis)",
        "Binance Vision public archive (klines, trades, bookTicker, funding; futures bookDepth from 2023) + Tardis.dev subscription for L2 snapshots and liquidations; DVOL is a Deribit index (from 2021-03, masked before)", ["binance_l2"], "1-min bars from depth20 snapshots; price_feat=[ret z, OFI, depth imbalance, liquidation count z, "
        "funding rate chg]; lob=20 levels; ctx_extra=[funding rate, open interest chg, DVOL]; labels = forward 30-min drawdown > 3 sigma; stress calendar = "
        "2020-03-12, 2021-05-19, 2022-05-09..13 (UST), 2022-06-13..18 (3AC), 2022-11-08..10 (FTX), 2024-08-05, 2025-02-03.",
        [("2020-03-12", "2020-03-13"), ("2021-05-19", "2021-05-20"), ("2022-05-09", "2022-05-13"), ("2022-06-13", "2022-06-18"), ("2022-11-08", "2022-11-10"), ("2024-08-05", "2024-08-05"), ("2025-02-03", "2025-02-03")])
_recipe("csi300_futures", "1min", 244 * 240, (1, 5, 15, 30, 60), "CSI 300 / SSE 50 index futures (IF/IH) level-1 tick (CFFEX via vendor, e.g. Wind/JoinQuant/RQData)",
        "vendor licence", ["csi300_futures"], "1-min bars; price_feat=[ret z, OI chg, trade imbalance, basis z, limit-distance]; liq=[half-spread, touch depth, vol/ADV]; "
        "tradable=False at price limits; labels = forward 30-min drawdown rule; stress calendar = 2015-06-15..2015-09-02, 2016-01-04..07 (circuit breaker), 2020-02-03, 2024-02-05..08 (small-cap quant crash: CSI 300 itself rallied, treat as dispersion stress or add IC/IM), 2024-10-08..10.",
        [("2015-06-15", "2015-09-02"), ("2016-01-04", "2016-01-07"), ("2020-02-03", "2020-02-04"), ("2024-02-05", "2024-02-08"), ("2024-10-08", "2024-10-10")])
_recipe("nyse_taq", "1min", 252 * 390, (1, 5, 15, 30, 60), "NYSE TAQ (WRDS) trades and NBBO quotes, S&P 100 constituents",
        "WRDS subscription", ["nyse_taq"], "1-min NBBO bars; price_feat=[ret z, quoted-spread chg, trade imbalance, quote-update rate z, quote-based OFI (Cont et al. 2014)]; "
        "tradable=False during LULD halts; labels = forward 15-min drawdown rule; stress calendar = 2010-05-06, 2015-08-24, 2018-02-05, 2020-03, 2024-08-05.",
        EQ_STRESS)
_recipe("fxcm_fx", "1min", 260 * 1440, (1, 5, 15, 30, 60), "FXCM / Dukascopy EUR/USD, USD/JPY tick data (free historical downloads)",
        "public (terms of use)", ["fxcm_fx"], "1-min mid bars from bid/ask ticks; price_feat=[ret z, spread chg, tick-rate z, realised skew]; liq=[half-spread, tick-rate, 1]; "
        "labels = forward 30-min move > 3 sigma (two-sided; macro shocks); stress calendar = 2015-01-15 (SNB), 2016-06-24 (Brexit), 2016-10-07 (GBP flash), 2019-01-03 (JPY flash), 2020-03, 2022-09-22 (MoF intervention), 2024-08-05.",
        [("2015-01-15", "2015-01-16"), ("2016-06-24", "2016-06-27"), ("2016-10-06", "2016-10-07"), ("2019-01-02", "2019-01-03"), ("2020-03-09", "2020-03-20"), ("2022-09-22", "2022-09-23"), ("2024-08-05", "2024-08-06")])
_recipe("ust_curve", "daily", 252, (1, 2, 3, 5, 10, 20), "US Treasury par yield curve (FRED/Treasury) + CME 10Y futures ticks (optional) + MOVE index",
        "FRED/Treasury public; ICE BofA MOVE is licensed (ICE Data Services); CME DataMine licence for futures ticks", ["ust_curve"], "daily tenors 3M..30Y; price_feat per tenor=[yield chg z, curvature chg, 2s10s slope chg, MOVE chg]; "
        "liq from futures bid-ask if available else constant; labels = flight-to-safety: forward 5-day |yield chg| > 3 sigma with MOVE spike; stress calendar = "
        "2013-05-22..06-25 (taper), 2019-09-17 (repo), 2020-03-09..03-23 (dash for cash), 2022-09-26..10-14 (gilt spill-over), 2023-03-09..03-17 (SVB), 2025-04-07..04-11.",
        [("2013-05-22", "2013-06-25"), ("2019-09-17", "2019-09-18"), ("2020-03-09", "2020-03-23"), ("2022-09-26", "2022-10-14"), ("2023-03-09", "2023-03-17"), ("2025-04-07", "2025-04-11")])
_recipe("flashcrash_bench", "1min", 252 * 390, (1, 5, 15, 30, 60), "Curated Flash-Crash Benchmark: 1-min bars around 2010-05-06, 2015-08-24, 2019-01-03 (FX), 2020-03-09/12/16, "
        "2021-02-25 (UST), 2022-05-02 (Nordic fat finger), 2024-08-05, plus 2025-2026 events to be verified",
        "assembled from the sources above", ["flashcrash_bench"], "each event = [-5 days, +2 days] window of 1-min bars for the affected instruments; labels = curated "
        "onset timestamps (bars_to_peak exact); stress = event window; normal = the surrounding days. Every window is its own walk-forward fold (train on earlier events only).",
        [("2010-05-06", "2010-05-06"), ("2015-08-24", "2015-08-24"), ("2019-01-02", "2019-01-03"), ("2020-03-09", "2020-03-16"), ("2021-02-25", "2021-02-25"), ("2022-05-02", "2022-05-02"), ("2024-08-05", "2024-08-05")])
