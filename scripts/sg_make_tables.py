#!/usr/bin/env python3
"""Aggregate results/stressguard/<dataset>/<tag>/seed*/fold*.json into the paper tables (mean +- std over seeds x folds).

    python scripts/sg_make_tables.py --out results/stressguard
Writes results/stressguard/tables/{detection,trading,cost_grid,cost_curve}.{csv,md}
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
from sg_common import RESULTS, bps_key

DET_COLS = [("full", "AUC_ROC"), ("normal", "AUC_ROC"), ("stress", "AUC_ROC"), ("severe", "AUC_ROC"), ("true_severe", "AUC_ROC"), ("stress", "AUC_PR"), ("stress", "P@1%"), ("stress", "R@1%"), ("stress", "TailF1"), ("stress", "alert_precision"), ("stress", "alert_recall")]
TR_COLS = [("base", "ARR"), ("base", "SR"), ("hedged", "ARR"), ("hedged", "SR"), ("hedged", "SoR"), ("hedged", "Calmar"), ("hedged", "MDD"), ("base", "MDD"), ("hedged_stress", "MDD"), ("base_stress", "MDD"),
           ("hedged", "VaR99"), ("hedged", "ES99"), ("base", "ES99"), ("hedged", "NetPnL"), ("hedged", "Turnover"), ("overlay", "hedge_rate")]


def load_runs(out: str):
    rows, curves, deciles, sweeps = [], [], [], []
    for cfg_path in glob.glob(os.path.join(out, "*", "*", "seed*", "config.json")):
        run_dir = os.path.dirname(cfg_path)
        seed = os.path.basename(run_dir)
        tag = os.path.basename(os.path.dirname(run_dir))
        dataset = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
        cfg = json.load(open(cfg_path))["config"]
        main_bps = bps_key(cfg["cost"]["commission_bps"])
        for fp in sorted(glob.glob(os.path.join(run_dir, "fold*.json"))):
            r = json.load(open(fp))
            for bps, ev in r["eval"].items():
                row = {"dataset": dataset, "model": tag, "seed": seed, "fold": r["fold"], "bps": float(bps), "main": bps == main_bps}
                for sl, k in DET_COLS:
                    row[f"{k}/{sl}"] = ev["detection"].get(sl, {}).get(k, np.nan)
                row["dAUC_stress"] = ev["detection"].get("dAUC_stress", np.nan)
                row["dAUC_severe"] = ev["detection"].get("dAUC_severe", np.nan)
                row["dAUC_true_severe"] = ev["detection"].get("dAUC_true_severe", np.nan)
                for sl, k in TR_COLS:
                    row[f"{k}/{sl}"] = ev["trading"].get(sl, {}).get(k, np.nan)
                row["MDD_reduction"] = ev["trading"].get("MDD_reduction", np.nan)
                row["MDD_reduction_stress"] = ev["trading"].get("MDD_reduction_stress", np.nan)
                rows.append(row)
            k = r["fold"]
            sw = r.get("ratio_sweep")
            sp = os.path.join(run_dir, f"ratio_sweep_fold{k}.json")
            if sw is None and os.path.exists(sp):
                sw = json.load(open(sp)).get("ratio_sweep")
            for ratio, v in (sw or {}).items():
                t = v["trading"]
                sweeps.append({"dataset": dataset, "model": tag, "seed": seed, "fold": k, "ratio": float(ratio), "hedge_rate": t["overlay"]["hedge_rate"],
                               "ARR": t["hedged"]["ARR"], "ARR_base": t["base"]["ARR"], "SR": t["hedged"]["SR"], "SR_base": t["base"]["SR"], "Calmar": t["hedged"]["Calmar"], "MDD": t["hedged"]["MDD"], "MDD_base": t["base"]["MDD"],
                               "MDD_stress": t.get("hedged_stress", {}).get("MDD", np.nan), "MDD_base_stress": t.get("base_stress", {}).get("MDD", np.nan),
                               "ES99": t["hedged"]["ES99"], "ES99_base": t["base"]["ES99"], "NetPnL": t["hedged"]["NetPnL"], "Turnover": t["hedged"].get("Turnover", np.nan),
                               "alert_recall_stress": v["detection"]["stress"].get("alert_recall", np.nan), "alert_precision_stress": v["detection"]["stress"].get("alert_precision", np.nan)})
            for name, store in (("cost_curve", curves), ("auc_decile", deciles)):
                p = os.path.join(run_dir, f"{name}_fold{k}.csv")
                if os.path.exists(p):
                    d = pd.read_csv(p); d["dataset"], d["model"], d["seed"], d["fold"] = dataset, tag, seed, k; store.append(d)
    return pd.DataFrame(rows), (pd.concat(curves) if curves else pd.DataFrame()), (pd.concat(deciles) if deciles else pd.DataFrame()), pd.DataFrame(sweeps)


def agg(df: pd.DataFrame, cols, by=("dataset", "model")) -> pd.DataFrame:
    g = df.groupby(list(by))
    out = g[cols].mean().round(4)
    sd = g[cols].std().round(4)
    n = g.size().rename("n_runs")
    fmt = out.copy().astype(object)
    for c in cols:
        fmt[c] = [f"{m:.3f} ± {s:.3f}" if np.isfinite(s) else f"{m:.3f}" for m, s in zip(out[c], sd[c])]
    return fmt.join(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=RESULTS)
    args = p.parse_args()
    runs, curves, deciles, sweeps = load_runs(args.out)
    if runs.empty:
        print("no runs found under", args.out); return
    tdir = os.path.join(args.out, "tables"); os.makedirs(tdir, exist_ok=True)
    main = runs[runs["main"]]
    det_cols = [f"{k}/{sl}" for sl, k in DET_COLS] + ["dAUC_stress", "dAUC_severe", "dAUC_true_severe"]
    tr_cols = [f"{k}/{sl}" for sl, k in TR_COLS] + ["MDD_reduction", "MDD_reduction_stress"]
    t1, t2 = agg(main, det_cols), agg(main, tr_cols)
    t3 = runs.pivot_table(index=["dataset", "model"], columns="bps", values=["SR/hedged", "MDD/hedged", "NetPnL/hedged"], aggfunc="mean").round(4)
    t1.to_csv(os.path.join(tdir, "detection.csv")); t2.to_csv(os.path.join(tdir, "trading.csv")); t3.to_csv(os.path.join(tdir, "cost_grid.csv"))
    runs.to_csv(os.path.join(tdir, "all_runs.csv"), index=False)
    md = ["# StressGuard result tables (mean ± std over seeds × folds, main commission level)\n", "## Table 1 - Detection by regime (AUC/full, AUC/normal, AUC/stress, ΔAUC_stress = normal − stress; tail metrics on the Stress slice)\n",
          t1.to_markdown(), "\n## Table 2 - Hedge-overlay trading & downside protection (hedged vs unhedged equal-weight portfolio)\n", t2.to_markdown(),
          "\n## Table 3 - Cost-grid robustness (hedged SR / MDD / NetPnL per commission bps; model trained at the main level only)\n", t3.to_markdown()]
    if not curves.empty:
        cc = curves.groupby(["dataset", "model", "slice", "rule", "ratio"])[["fn_cost", "fp_cost", "total", "alert_rate", "recall"]].mean().reset_index()
        cc.to_csv(os.path.join(tdir, "cost_curve.csv"), index=False)
        piv = cc[cc["slice"] == "stress"].pivot_table(index=["dataset", "model", "rule"], columns="ratio", values="total").round(5)
        md += ["\n## Table 4 - Economic cost curve on the Stress slice: total cost (FN + FP) per c_FN/c_FP ratio, dynamic vs val-fitted static vs oracle static\n", piv.to_markdown()]
    if not deciles.empty:
        dd = deciles.groupby(["dataset", "model", "decile"])["AUC_ROC"].mean().unstack("decile").round(3)
        dd.to_csv(os.path.join(tdir, "auc_by_decile.csv"))
        md += ["\n## Table 5 - AUC by stress-score decile (Fig. 1B source)\n", dd.to_markdown()]
    if not sweeps.empty:
        sw = sweeps.groupby(["dataset", "model", "ratio"])[["hedge_rate", "ARR", "ARR_base", "SR", "SR_base", "Calmar", "MDD", "MDD_base", "MDD_stress", "MDD_base_stress", "ES99", "ES99_base", "NetPnL", "alert_recall_stress"]].mean().round(4)
        sw.to_csv(os.path.join(tdir, "ratio_frontier.csv"))
        md += ["\n## Table 6 - Risk-aversion frontier: dynamic rule re-applied at each c_FN/c_FP on the trained detector (hedged portfolio; MDD_base for reference)\n", sw.to_markdown()]
    open(os.path.join(tdir, "tables.md"), "w").write("\n".join(md))
    print("\n".join(md))
    print(f"\nwritten -> {tdir}")


if __name__ == "__main__":
    main()
