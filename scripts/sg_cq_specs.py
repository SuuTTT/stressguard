#!/usr/bin/env python3
"""Write cq job specs for the StressGuard grid (one job per dataset x model x seed, all walk-forward folds inside).

    python3 scripts/sg_cq_specs.py --work StressGuard --commit <sha> --out specs/ [--seeds 42] [--smoke-id j-...]
Each job: CPU-only (gpus 0, cpus 2, ram_gb 4), writes only to $CQ_OUTPUT_DIR, prints n/N progress to run.log, and declares every
fold json in outputs.expect. Code from github.com/SuuTTT/stressguard (code.git); project home /home/sudingli/cq-home/stressguard on c225 (cq copies it to the job host).
"""
import argparse, json, os
H = "/home/sudingli/cq-home/stressguard"
PY = f"{H}/env/bin/python"
NF = {"synthetic_regime": 6, "binance_um": 8, "us_equity": 8}
MODELS = {"erm": "configs/stressguard/ablations/erm.yaml", "sgv1": "configs/stressguard/stressguard.yaml",
          "sgv3": "configs/stressguard/stressguard_v3.yaml", "sgv3_noctx": "configs/stressguard/ablations/v3_noctx.yaml",
          "sgv3_nogate": "configs/stressguard/ablations/v3_nogate.yaml", "sgv3_irm": "configs/stressguard/ablations/v3_irm.yaml",
          "sgv3_regcal": "configs/stressguard/ablations/v3_regcal.yaml", "sgv3_econ": "configs/stressguard/ablations/v3_econ.yaml",
          "sgv3_ex1": "configs/stressguard/ablations/v3_extrap1.yaml", "sgv3_ex2": "configs/stressguard/ablations/v3_extrap2.yaml",
          "sgv4": "configs/stressguard/stressguard_v4.yaml", "sgv4_noaug": "configs/stressguard/ablations/v4_noaug.yaml",
          "sgv4_augonly": "configs/stressguard/ablations/v4_augonly.yaml", "sgv4_z3": "configs/stressguard/ablations/v4_z3.yaml",
          "sgv5": "configs/stressguard/stressguard_v5.yaml", "sgv5_uni": "configs/stressguard/ablations/v5_uni.yaml",
          "sgv5_k8": "configs/stressguard/ablations/v5_k8.yaml", "sgv5_z1": "configs/stressguard/ablations/v5_z1.yaml"}
BASELINES = ["zscore", "iforest", "recon_ae", "dlinear_recon", "anomaly_transformer", "tranad", "dcdetector", "timesnet"]


def spec(work, commit, batch, name, ds, tag, seed, cmd, nfolds, ram=4, hours=2.0, depends=None):
    run = f"{ds}/{tag}/seed{seed}"
    shell = f"OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 {PY} -W ignore {cmd} --out $CQ_OUTPUT_DIR 2>&1 | tee $CQ_OUTPUT_DIR/run.log; test ${{PIPESTATUS[0]}} -eq 0"
    s = {"name": f"stressguard {batch}: {name}", "work": work, "python": PY,
         "command": ["bash", "-lc", shell],
         "code": {"git": "SuuTTT/stressguard", "commit": commit},
         "home": {"name": "stressguard", "from": "c225"},
         "resources": {"gpus": 0, "cpus": 2, "ram_gb": ram},
         "limits": {"max_hours": hours, "max_attempts": 2},
         "outputs": {"dest": f"{H}/results/{batch}/{run}", "expect": [{"path": f"{run}/fold{k}.json"} for k in range(nfolds)],
                     "progress": {"path": "run.log", "stall_min": 30}},
         "workload": {"bound": "cpu"}}
    if depends:
        s["depends_on"] = [depends]
    return s


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--work", required=True); a.add_argument("--commit", required=True); a.add_argument("--out", default="specs")
    a.add_argument("--batch", default="screen"); a.add_argument("--seeds", nargs="*", type=int, default=[42])
    a.add_argument("--datasets", nargs="*", default=list(NF)); a.add_argument("--models", nargs="*", default=list(MODELS) + BASELINES)
    a.add_argument("--smoke-id", default=None); a.add_argument("--only-smoke", action="store_true")
    a.add_argument("--smoke-deps", nargs="*", default=[], help="job ids the smoke job waits for (env build, data copy)")
    o = a.parse_args()
    os.makedirs(o.out, exist_ok=True)
    smoke = spec(o.work, o.commit, "smoke", "sgv5 synthetic fold0 2 epochs", "synthetic_regime", "sgv5", 42,
                 "scripts/sg_train.py --device cpu --config configs/stressguard/stressguard_v5.yaml --dataset synthetic_regime --seed 42 --folds 0 --tag sgv5 --override train.epochs=2",
                 1, ram=4, hours=0.5)
    if o.smoke_deps:
        smoke["depends_on"] = list(o.smoke_deps)
    json.dump(smoke, open(os.path.join(o.out, "smoke.json"), "w"), indent=1)
    if o.only_smoke:
        return
    n = 0
    for ds in o.datasets:
        for seed in o.seeds:
            for m in o.models:
                if m in MODELS:
                    cmd = f"scripts/sg_train.py --device cpu --config {MODELS[m]} --dataset {ds} --seed {seed} --tag {m}"
                    tag = m
                else:
                    b, dec = (m[:-4], "dynamic") if m.endswith("+dyn") else (m, "static")
                    tag = b if dec == "static" else f"{b}+dyn"
                    cmd = f"scripts/sg_baseline.py --device cpu --config configs/stressguard/baselines/{b}.yaml --dataset {ds} --seed {seed} --decision {dec}"
                hours = 3.0 if ds == "binance_um" else 2.0
                s = spec(o.work, o.commit, o.batch, f"{ds} {tag} s{seed}", ds, tag, seed, cmd, NF[ds], ram=4 if ds == "binance_um" else 2, hours=hours, depends=o.smoke_id)
                json.dump(s, open(os.path.join(o.out, f"{o.batch}__{ds}__{tag.replace('+', '_')}__s{seed}.json"), "w"), indent=1)
                n += 1
    print(f"wrote smoke + {n} specs to {o.out}")


if __name__ == "__main__":
    main()
