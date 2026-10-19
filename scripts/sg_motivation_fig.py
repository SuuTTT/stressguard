#!/usr/bin/env python3
"""Paper figures from results/stressguard (static PNG + PDF; matplotlib).

Fig. 1  (A) AUC-ROC by stress-score decile per detector: prior detectors collapse toward 0.5 in the top deciles.
        (B) Return-to-drawdown frontier: annualised return vs |MDD| of the hedged portfolio, one point per model (iso-Calmar rays).
Fig. 2  Economic cost curve on the Stress slice: total asymmetric cost vs c_FN/c_FP for the dynamic rule, the val-fitted static rule
        and the oracle static rule.
    python scripts/sg_motivation_fig.py --out results/stressguard --models stressguard erm recon_ae iforest dlinear_recon zscore
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sg_common import RESULTS
from sg_make_tables import load_runs

# validated categorical palette (fixed slot order; identity is never colour-alone: every series is also direct-labelled / marker-coded)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LABEL = {"stressguard": "StressGuard (ours)", "erm": "ERM same backbone", "recon_ae": "Recon-AE", "iforest": "Isolation Forest",
         "dlinear_recon": "DLinear-recon", "zscore": "z-score rule", "no_irm": "w/o IRM", "no_econ": "w/o L_econ", "static_thr": "static thr.",
         "no_ctx": "w/o context", "vrex": "V-REx", "no_cvar": "w/o CVaR", "no_sev_head": "severity plug-in", "linear_surrogate": "linear surrogate"}


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig1(runs: pd.DataFrame, deciles: pd.DataFrame, models, dataset: str, out: str):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=160)
    fig.patch.set_facecolor(SURF)
    # ---- (A) AUC by stress decile
    d = deciles[(deciles["dataset"] == dataset) & (deciles["model"].isin(models))]
    g = d.groupby(["model", "decile"])["AUC_ROC"].agg(["mean", "std"]).reset_index()
    for i, m in enumerate(models):
        gm = g[g["model"] == m]
        if gm.empty:
            continue
        c = SERIES[i % len(SERIES)]
        a.plot(gm["decile"], gm["mean"], color=c, lw=1.8 if m == "stressguard" else 1.4, marker=MARKERS[i % 8], ms=4.5, label=LABEL.get(m, m), zorder=3 if m == "stressguard" else 2)
        if gm["std"].notna().any():
            a.fill_between(gm["decile"], gm["mean"] - gm["std"].fillna(0), gm["mean"] + gm["std"].fillna(0), color=c, alpha=0.10, lw=0)
    a.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
    a.text(10.1, 0.5, "random", color=INK2, fontsize=7, va="center")
    a.set_xlabel("stress-score decile (1 = calm, 10 = extreme)", color=INK2, fontsize=8.5)
    a.set_ylabel("AUC-ROC", color=INK2, fontsize=8.5)
    a.set_title("A. Discriminative power across volatility regimes", color=INK, fontsize=9.5, loc="left")
    a.set_xticks(range(1, 11)); a.set_ylim(0.35, 1.0); a.set_xlim(0.7, 11.2)
    a.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    _style(a)
    # ---- (B) return-to-drawdown frontier
    r = runs[(runs["dataset"] == dataset) & runs["main"]]
    pts = r.groupby("model")[["ARR/hedged", "MDD/hedged", "ARR/hedged".replace("hedged", "base"), "MDD/base"]].mean()
    base = pts[["ARR/base", "MDD/base"]].mean()
    xs = [abs(base["MDD/base"])] + [abs(pts.loc[m, "MDD/hedged"]) for m in models if m in pts.index]
    ys = [base["ARR/base"]] + [pts.loc[m, "ARR/hedged"] for m in models if m in pts.index]
    xmax, ymax = max(xs) * 1.25 + 1e-9, max(max(ys), 1e-6) * 1.25
    for cal in (0.5, 1.0, 2.0, 4.0):
        b.plot([0, xmax], [0, cal * xmax], color=GRID, lw=0.8, ls=(0, (2, 2)), zorder=1)
        if cal * xmax <= ymax:
            b.text(xmax, cal * xmax, f"Calmar {cal:g}", color=INK2, fontsize=6.5, ha="right", va="bottom")
    b.scatter([xs[0]], [ys[0]], s=46, marker="x", color=INK2, zorder=3)
    b.annotate("Unhedged", (xs[0], ys[0]), xytext=(4, -9), textcoords="offset points", fontsize=7, color=INK2)
    for i, m in enumerate(models):
        if m not in pts.index:
            continue
        x, y = abs(pts.loc[m, "MDD/hedged"]), pts.loc[m, "ARR/hedged"]
        b.scatter([x], [y], s=46, marker=MARKERS[i % 8], color=SERIES[i % len(SERIES)], edgecolor=SURF, linewidth=1.2, zorder=4)
        b.annotate(LABEL.get(m, m), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7, color=INK)
    b.set_xlim(0, xmax); b.set_ylim(min(0, min(ys) * 1.2), ymax)
    b.set_xlabel("maximum drawdown |MDD| (log-return)", color=INK2, fontsize=8.5)
    b.set_ylabel("annualised return (hedged portfolio)", color=INK2, fontsize=8.5)
    b.set_title("B. Return-to-drawdown frontier (hedged overlay)", color=INK, fontsize=9.5, loc="left")
    _style(b)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig1_motivation_{dataset}.{ext}"), facecolor=SURF)
    plt.close(fig)


def fig2(curves: pd.DataFrame, models, dataset: str, out: str):
    c = curves[(curves["dataset"] == dataset) & (curves["slice"] == "stress") & (curves["model"].isin(models))]
    if c.empty:
        return
    g = c.groupby(["model", "rule", "ratio"])["total"].mean().reset_index()
    fig, axes = plt.subplots(1, len(models), figsize=(2.9 * len(models), 3.0), dpi=160, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURF)
    rules = [("dynamic", SERIES[0], "o", "dynamic (cost-sensitive)"), ("static_valF1", SERIES[1], "s", "static (val best-F1)"), ("static_oracle", INK2, "^", "static oracle (test-fitted)")]
    for ax, m in zip(axes, models):
        for rule, col, mk, lab in rules:
            gm = g[(g["model"] == m) & (g["rule"] == rule)]
            if gm.empty:
                continue
            ax.plot(gm["ratio"], gm["total"], color=col, lw=1.4, marker=mk, ms=4, label=lab)
        ax.set_xscale("log"); ax.set_title(LABEL.get(m, m), color=INK, fontsize=9, loc="left")
        ax.set_xlabel("c_FN / c_FP", color=INK2, fontsize=8.5)
        _style(ax)
    axes[0].set_ylabel("expected economic cost (Stress slice)", color=INK2, fontsize=8.5)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig2_cost_curve_{dataset}.{ext}"), facecolor=SURF)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=RESULTS)
    p.add_argument("--dataset", default="synthetic_regime")
    p.add_argument("--models", nargs="*", default=["stressguard", "erm", "recon_ae", "iforest", "dlinear_recon", "zscore"])
    args = p.parse_args()
    runs, curves, deciles, sweeps = load_runs(args.out)
    if runs.empty:
        print("no runs under", args.out); return
    models = [m for m in args.models if m in set(runs["model"])]
    fdir = os.path.join(args.out, "figures"); os.makedirs(fdir, exist_ok=True)
    fig1(runs, deciles, models, args.dataset, fdir)
    fig2(curves, [m for m in models if m in ("stressguard", "erm", "recon_ae", "iforest")][:4], args.dataset, fdir)
    print("figures ->", fdir, sorted(os.listdir(fdir)))


if __name__ == "__main__":
    main()
