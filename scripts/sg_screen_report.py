#!/usr/bin/env python3
"""Print detection AUC slices + hedged trading summary for screening runs (fold jsons)."""
import glob, json, sys
roots = sys.argv[1:] or ["results/sg_screen", "results/stressguard"]
for root in roots:
    for f in sorted(glob.glob(f"{root}/*/*/seed*/fold*.json")):
        r = json.load(open(f)); ev = r["eval"].get("5") or next(iter(r["eval"].values()))
        d, t = ev["detection"], ev["trading"]
        parts = f.split("/")
        print(f"{parts[-4][:16]:16s} {parts[-3][:22]:22s} {parts[-2]} {parts[-1][:5]} | " + " ".join(f"{k[:6]}={d.get(k,{}).get('AUC_ROC',float('nan')):.3f}" for k in ("normal","stress","severe","true_severe"))
              + f" | ARR {t['base']['ARR']:+.3f}->{t['hedged']['ARR']:+.3f} MDD {t['base']['MDD']:.3f}->{t['hedged']['MDD']:.3f} hedge={t['overlay']['hedge_rate']:.3f}")
