#!/usr/bin/env python3
"""Emit one shell command per (dataset, model, seed, fold) job for the remote runner (scripts/sg_remote_run.sh).

    python scripts/sg_jobs.py --stage screen > jobs_screen.txt
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.environ.get("SG_PY", "/date/zyy/venvs/hh/bin/python")
NF = {"synthetic_regime": 6, "binance_um": 8, "us_equity": 8}          # walk-forward folds per dataset (checked by sg_count_folds)
MODELS = {"erm": "configs/stressguard/ablations/erm.yaml", "sgv1": "configs/stressguard/stressguard.yaml",
          "sgv3": "configs/stressguard/stressguard_v3.yaml", "sgv3_noctx": "configs/stressguard/ablations/v3_noctx.yaml",
          "sgv3_nogate": "configs/stressguard/ablations/v3_nogate.yaml", "sgv3_irm": "configs/stressguard/ablations/v3_irm.yaml",
          "sgv3_regcal": "configs/stressguard/ablations/v3_regcal.yaml", "sgv3_econ": "configs/stressguard/ablations/v3_econ.yaml",
          "sgv3_ex1": "configs/stressguard/ablations/v3_extrap1.yaml", "sgv3_ex2": "configs/stressguard/ablations/v3_extrap2.yaml"}
BASELINES = ["zscore", "iforest", "recon_ae", "dlinear_recon", "anomaly_transformer", "tranad", "dcdetector", "timesnet"]


def train_job(out, ds, tag, cfg, seed, k):
    log = f"{out}/logs/{ds}__{tag}__s{seed}__f{k}.log"
    return f"OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 {PY} -W ignore scripts/sg_train.py --device cpu --config {cfg} --dataset {ds} --seed {seed} --folds {k} --out {out} --tag {tag} > {log} 2>&1"


def base_job(out, ds, b, seed, k, decision):
    tag = b if decision == "static" else f"{b}+dyn"
    log = f"{out}/logs/{ds}__{tag}__s{seed}__f{k}.log"
    return (f"OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 {PY} -W ignore scripts/sg_baseline.py --device cpu --config configs/stressguard/baselines/{b}.yaml "
            f"--dataset {ds} --seed {seed} --folds {k} --decision {decision} --out {out} > {log} 2>&1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="screen", choices=["screen", "full"])
    p.add_argument("--out", default="results")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--datasets", nargs="*", default=list(NF))
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    a = p.parse_args()
    seeds = a.seeds or ([42] if a.stage == "screen" else [42, 2024, 2025, 2026])
    models = a.models or list(MODELS)
    jobs = []
    for ds in a.datasets:
        for seed in seeds:
            for k in range(NF[ds]):
                for m in models:
                    if m in MODELS:
                        jobs.append(train_job(a.out, ds, m, MODELS[m], seed, k))
                    elif m.endswith("+dyn"):
                        jobs.append(base_job(a.out, ds, m[:-4], seed, k, "dynamic"))
                    else:
                        jobs.append(base_job(a.out, ds, m, seed, k, "static"))
    print("\n".join(jobs))


if __name__ == "__main__":
    main()
