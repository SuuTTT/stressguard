"""StressGuard unit + smoke tests. Run with `python3 -m pytest tests/test_stressguard.py -q` or `python3 tests/test_stressguard.py`."""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warn2trade.utils import DATASETS, MODELS, set_seed
from warn2trade.backtest import CostModel
from stressguard.data import RegimeLabeler, market_return, build_context, RegimeEventDataset, adverse_move, env_balanced_loader
from stressguard.models import losses as Lz
from stressguard.models.stressguard import attach_gamma
from stressguard.backtest import overlay_pnl
from stressguard.eval import regime_detection, cost_curve, auc_by_decile
import stressguard.models  # noqa: F401
import stressguard.baselines  # noqa: F401

SYN = dict(name="synthetic_regime", seed=1, T=2500, N=6, warmup=150, episode_gap=[200, 300], episode_len=[40, 80])


def _panel():
    return DATASETS.get("synthetic_regime")(SYN).build_panel()


def test_synthetic_panel_valid_and_stress_is_ood():
    p = _panel()
    p.validate()
    u = p.meta["u"]
    assert p.labels.sum() > 0 and p.meta["n_crashes"] > 0
    assert (p.meta["env_true"] == 1).mean() > 0.03
    beta = p.meta["spurious_beta"]
    assert beta[u == 0].min() > 0 and beta[u > 0.8].max() < 0.2   # spurious correlation flips under severe stress


def test_regime_labeler_tracks_truth():
    p = _panel()
    r = market_return(p.mid)
    res = RegimeLabeler(vol_quantile=0.9, tail_quantile=0.1, use=("vol", "tail", "dd")).fit_transform(r, np.arange(1200))
    env, tr = res["env"].astype(bool), p.meta["env_true"].astype(bool)
    tp = (env & tr).sum()
    f1 = 2 * tp / max(env.sum() + tr.sum(), 1)
    assert f1 > 0.5, f1
    assert 0.0 < env.mean() < 0.6


def test_adverse_move():
    mid = np.exp(np.cumsum(np.array([[0.0], [-0.01], [-0.02], [0.05], [0.0]]), 0))
    s = adverse_move(mid, 2)
    assert np.isclose(s[0, 0], 0.03) and np.isclose(s[1, 0], 0.02) and np.isnan(s[-1, 0])


def test_bayes_threshold_and_hedge_consistency():
    sev = torch.tensor([0.05, 0.05, 0.01]); gam = torch.tensor([0.005, 0.01, 0.005])
    thr = Lz.bayes_threshold(sev, gam, 5.0, 1.0)
    assert (thr > 0).all() and (thr < 1).all() and thr[1] > thr[0] and thr[2] > thr[0]   # more friction / less severity -> higher bar
    p = thr + torch.tensor([0.1, -0.05, 0.3])
    h = Lz.hedge_ratio(p, sev, gam, 5.0, 1.0, 1.0)
    assert h[0] > 0 and h[1] == 0 and h[2] > 0 and (h <= 1).all()
    assert torch.isclose(Lz.hedge_ratio(torch.ones(3), sev, gam, 5.0, 1.0, 1.0), torch.ones(3)).all()


def test_cost_sensitive_loss_recovers_bce_and_linear_form():
    logit = torch.randn(32); y = (torch.rand(32) > 0.7).float(); sev = torch.rand(32) * 0.1; gam = torch.rand(32) * 0.01
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
    sym = Lz.cost_sensitive_loss(logit, y, sev, gam, 1.0, 1.0, use_sev_weight=False, use_gamma_weight=False)
    assert torch.allclose(sym, bce, atol=1e-6)
    lin = Lz.cost_sensitive_loss(logit, y, sev, gam, 5.0, 1.0, surrogate="linear", normalize=False)
    p = torch.sigmoid(logit)
    assert torch.allclose(lin, 5.0 * y * (1 - p) * sev + (1 - y) * p * gam, atol=1e-6)


def test_irm_penalty_positive_and_differentiable():
    phi = torch.randn(64, 4, requires_grad=True)
    w = torch.randn(4, requires_grad=True)
    logit = phi @ w
    y = (torch.rand(64) > 0.5).float()
    pen = Lz.irm_penalty(logit, lambda lg: torch.nn.functional.binary_cross_entropy_with_logits(lg, y))
    assert pen.item() >= 0
    pen.backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    with torch.no_grad():
        assert Lz.irm_penalty(logit.detach(), lambda lg: lg.mean()).item() == 0.0


def _data(p, env, ctx, lo, hi, tau=5):
    return RegimeEventDataset(p, np.arange(lo, hi), env, ctx, tau, 16, [1, 2, 3, 5], 1, 2, 20)


def test_dataset_and_model_forward_and_overlay():
    set_seed(0)
    p = _panel()
    env = RegimeLabeler(vol_quantile=0.9).fit_transform(market_return(p.mid), np.arange(1200))["env"]
    ctx, names = build_context(p, {"windows": [5, 20], "norm_window": 100})
    assert ctx.shape == (p.T, p.N, 6) and names[-1] == "ivskew"
    ds = _data(p, env, ctx, 200, 900)
    item = ds[0]
    for k in ("price", "liq", "lob", "ctx", "env", "sev", "fut_ret", "fut_liq", "fut_tradable", "stress_score"):
        assert k in item, k
    loader = env_balanced_loader(ds, 64, 0, 0, True)
    b = next(iter(loader))
    dims = p.dims(); dims["ctx"] = ctx.shape[-1]
    cfg = {"tau": 5, "encoder": {"d_model": 32, "patch_len": 4, "stride": 2, "price_layers": 1, "n_heads": 2}, "decision": {"c_fn": 5.0, "c_fp": 1.0}}
    model = MODELS.get("stressguard")(dims, cfg)
    cm = CostModel()
    out = model(attach_gamma(b, cm))
    assert out["logit"].shape == (len(b["y"]),) and out["sev_q"].shape == (len(b["y"]), 4)
    assert (out["sev_q"][:, 1:] >= out["sev_q"][:, :-1]).all()                   # monotone quantiles
    assert ((out["h"] > 0) == (out["p"] > out["thr"])).all()                       # hedge iff p above dynamic threshold
    r = overlay_pnl(out["h"], b, cm, 1, 5)
    assert r["pnl"].shape == out["h"].shape and (r["pnl"][r["frozen"]] == 0).all()
    assert torch.allclose(r["pnl"][~r["frozen"]], (-r["h_eff"] * r["R"] - r["cost"])[~r["frozen"]], atol=1e-6)


def test_end_to_end_tiny_training():
    from stressguard.train import StressTrainer
    from warn2trade.data import build_loader
    set_seed(0)
    p = _panel()
    env = RegimeLabeler(vol_quantile=0.9).fit_transform(market_return(p.mid), np.arange(1200))["env"]
    ctx, _ = build_context(p, {"windows": [5, 20], "norm_window": 100})
    ds = {"train": _data(p, env, ctx, 200, 1200), "val": _data(p, env, ctx, 1210, 1500), "test": _data(p, env, ctx, 1510, 2300)}
    loaders = {"train": env_balanced_loader(ds["train"], 128, 0, 0, True), "val": build_loader(ds["val"], 256, False), "test": build_loader(ds["test"], 256, False)}
    dims = p.dims(); dims["ctx"] = ctx.shape[-1]
    cfg = {"data": {"latency": 1}, "regime": {"n_env": 2},
           "model": {"name": "stressguard", "tau": 5, "encoder": {"d_model": 32, "patch_len": 4, "stride": 2, "price_layers": 1, "n_heads": 2}, "decision": {"c_fn": 5.0, "c_fp": 1.0}},
           "train": {"epochs": 2, "lr": 1e-3, "patience": 2, "loss": {"invariance": "irm", "lam_irm": 2.0, "irm_warmup": 1, "lam_sev": 1.0, "lam_cvar": 0.1}}}
    model = MODELS.get("stressguard")(dims, cfg["model"])
    tr = StressTrainer(cfg, model, loaders, CostModel(), torch.device("cpu"), 252)
    fit = tr.fit()
    assert len(fit["history"]) == 2 and all(np.isfinite(h["total"]) for h in fit["history"])
    df = tr.predict(loaders["test"])
    for c in ("score", "thr", "h", "sev_hat", "sev", "env", "gamma", "pnl", "hedged", "static_thr"):
        assert c in df.columns, c
    det = regime_detection(df)
    assert "AUC_ROC" in det["full"] and np.isfinite(det["dAUC_stress"]) or det["stress"].get("n", 0) == 0
    cc = cost_curve(df, [1, 5, 20])
    assert set(cc["rule"]) == {"dynamic", "static_valF1", "static_oracle"} and (cc["total"] >= 0).all()
    assert len(auc_by_decile(df, "stress_score", 5)) == 5


def test_baselines_score_shapes():
    from warn2trade.utils import BASELINES
    from warn2trade.data import build_loader
    set_seed(0)
    p = _panel()
    env = RegimeLabeler(vol_quantile=0.9).fit_transform(market_return(p.mid), np.arange(1200))["env"]
    ctx, _ = build_context(p, {"windows": [5, 20], "norm_window": 100})
    ds = _data(p, env, ctx, 200, 700)
    loader = build_loader(ds, 128, False)
    dims = p.dims(); dims["ctx"] = ctx.shape[-1]
    for name in ("zscore", "iforest", "recon_ae", "dlinear_recon"):
        bl = BASELINES.get(name)({"epochs": 1}, dims, torch.device("cpu"))
        bl.fit(loader)
        b = attach_gamma(next(iter(loader)), CostModel())
        s = bl.score(b)
        assert s.shape == (len(b["y"]),) and torch.isfinite(s).all(), name
    for slot in ("moderntcn", "mamba_finad"):                     # anomaly_transformer is now a reimplementation (reimpl.py)
        cls = BASELINES.get(slot)
        assert cls.status in ("wrapper-needed", "placeholder-unverified")


def test_hedge_book_caps_overlapping_hedges_and_calendar_end_day():
    import pandas as pd
    from stressguard.backtest import OverlayBacktester
    from stressguard.data import apply_calendar
    p = _panel()
    test_idx = np.arange(1000, 1200)
    t = np.repeat(np.arange(1010, 1060), 1)                       # 50 consecutive alerts on asset 0, each asking for a 10-bar hedge
    df = pd.DataFrame({"t": t, "asset": 0, "h_dec": 1.0})
    hb = OverlayBacktester.hedge_book(df, p, test_idx, CostModel(), 1, 10, h_max=1.0)
    assert hb["pos"].max() <= 1.0 + 1e-9                          # never stacked beyond h_max
    assert hb["pos"][:, 1:].sum() == 0 and hb["turnover"].sum() > 0
    held = hb["pos"][:, 0] > 0
    tradable = p.tradable[test_idx, 0]
    assert held[12:68][tradable[12:68]].mean() > 0.9                # covered from first fill to last exit
    ts = pd.date_range("2024-08-02 09:30", "2024-08-06 16:00", freq="1min").values.astype("int64")
    env = apply_calendar(ts, np.zeros(len(ts), np.int8), [("2024-08-05", "2024-08-05")])
    days = pd.to_datetime(ts).normalize()
    assert env[days == pd.Timestamp("2024-08-05")].all() and not env[days != pd.Timestamp("2024-08-05")].any()


def test_reimpl_baselines_fit_and_score():
    from warn2trade.utils import BASELINES
    from warn2trade.data import build_loader
    set_seed(0)
    p = _panel()
    env = RegimeLabeler(vol_quantile=0.9).fit_transform(market_return(p.mid), np.arange(1200))["env"]
    ctx, _ = build_context(p, {"windows": [5, 20], "norm_window": 100})
    loader = build_loader(_data(p, env, ctx, 200, 500), 128, False)
    dims = p.dims(); dims["ctx"] = ctx.shape[-1]
    for name in ("anomaly_transformer", "tranad", "dcdetector", "timesnet"):
        bl = BASELINES.get(name)({"epochs": 1, "patch_sizes": [4]}, dims, torch.device("cpu"))
        assert bl.status == "reimplementation" and bl.official_repo.startswith("https://github.com/"), name
        bl.fit(loader)
        assert all(np.isfinite(h) for h in bl.history), name
        b = attach_gamma(next(iter(loader)), CostModel())
        s = bl.score(b)
        assert s.shape == (len(b["y"]),) and torch.isfinite(s).all() and s.std() > 0, name


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(f"{len(fns)} tests passed")
