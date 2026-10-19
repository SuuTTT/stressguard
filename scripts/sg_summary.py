#!/usr/bin/env python3
"""Mean over folds (and seeds) per dataset x model of the regime-sliced AUCs and hedged trading metrics, plus paired win-rates vs ERM.

    python scripts/sg_summary.py --out results [--ref erm]
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

def rows(out):
    R = []
    for f in glob.glob(os.path.join(out, "*", "*", "seed*", "fold*.json")):
        p = f.split(os.sep); ds, tag, seed, fold = p[-4], p[-3], p[-2], int(p[-1][4:-5])
        r = json.load(open(f)); ev = r["eval"].get("5") or next(iter(r["eval"].values()))
        d, t = ev["detection"], ev["trading"]
        row = {"dataset": ds, "model": tag, "seed": seed, "fold": fold}
        for sl in ("full", "normal", "stress", "severe", "true_severe"):
            row[f"AUC_{sl}"] = d.get(sl, {}).get("AUC_ROC", np.nan)
            row[f"PR_{sl}"] = d.get(sl, {}).get("AUC_PR", np.nan)
        row["TailF1_stress"] = d.get("stress", {}).get("TailF1", np.nan)
        for k in ("ARR", "SR", "MDD", "ES99", "Calmar"):
            row[f"{k}_h"] = t["hedged"].get(k, np.nan); row[f"{k}_b"] = t["base"].get(k, np.nan)
        row["MDD_stress_h"] = t.get("hedged_stress", {}).get("MDD", np.nan); row["MDD_stress_b"] = t.get("base_stress", {}).get("MDD", np.nan)
        row["hedge_rate"] = t["overlay"].get("hedge_rate", np.nan)
        R.append(row)
    return pd.DataFrame(R)

def main():
    a = argparse.ArgumentParser(); a.add_argument("--out", default="results"); a.add_argument("--ref", default="erm"); a = a.parse_args()
    df = rows(a.out)
    if df.empty:
        print("no runs"); return
    cols = ["AUC_normal", "AUC_stress", "AUC_severe", "AUC_true_severe", "PR_stress", "PR_severe", "ARR_h", "SR_h", "MDD_h", "MDD_stress_h", "ES99_h", "hedge_rate"]
    pd.set_option("display.width", 250)
    for ds, g in df.groupby("dataset"):
        m = g.groupby("model")[cols].mean().round(3); m["n"] = g.groupby("model").size()
        base = g.groupby("model")[["ARR_b", "MDD_b", "ES99_b"]].mean().round(3).iloc[0]
        print(f"\n=== {ds}   (unhedged ARR {base.ARR_b}, MDD {base.MDD_b}, ES99 {base.ES99_b})")
        print(m.sort_values("AUC_severe", ascending=False).to_string())
        if a.ref in set(g.model):
            ref = g[g.model == a.ref].set_index(["seed", "fold"])
            wins = []
            for mod, gm in g.groupby("model"):
                j = gm.set_index(["seed", "fold"]).join(ref, rsuffix="_ref", how="inner")
                if len(j) == 0 or mod == a.ref: continue
                wins.append({"model": mod, "n": len(j), **{f"win_{c}": float(np.nanmean(j[c] > j[c + "_ref"])) for c in ("AUC_severe", "AUC_stress", "MDD_h")}})
            if wins: print(pd.DataFrame(wins).set_index("model").round(2).to_string())
    df.to_csv(os.path.join(os.path.dirname(a.out.rstrip("/*")) or ".", "summary_rows.csv"), index=False)

if __name__ == "__main__":
    main()
