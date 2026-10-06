"""A04 - Learning-rate / local-epoch survey across methods (Section 5).

Baselines: constants read from the audited source + every report header.
FedHAD: per-client LR and epochs parsed from every report (constant per client
across rounds under static partitions). Also inventories the only existing
'reduced fixed-epoch' evidence: the historical test5_ablacao arms with
USE_DYNAMIC_EPOCHS=False, which run exactly 1 local epoch (FedHAD_Final_2.0.py fit()).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import ROOT, OUT, CAMPAIGN_05, CAMPAIGN_001, parse_report, seed_of, alpha_of

SRC = ROOT / "algoritmos"


def consts():
    rows = []
    for f in ["FedAVG_Final.py", "FedAvgM_Final.py", "FedProx_Final.py", "FedNova_Final.py", "FedHAD_Final_2.0.py"]:
        t = (SRC / f).read_text(encoding="utf-8")
        g = lambda k: (re.search(rf"^{k}\s*=\s*([^\s#]+)", t, re.M) or [None, None])[1]
        rows.append({"file": f, "LOCAL_LR/BASE_LR": g("LOCAL_LR") or g("BASE_LR"), "MIN_LR": g("MIN_LR"),
                     "LR_DECAY_FACTOR": g("LR_DECAY_FACTOR"), "LOCAL_EPOCHS": g("LOCAL_EPOCHS") or g("BASE_EPOCHS"),
                     "MIN_EPOCHS": g("MIN_EPOCHS"), "LOCAL_MOMENTUM": g("LOCAL_MOMENTUM") or "0.9 (hard-coded in train())",
                     "SERVER_MOMENTUM": g("SERVER_MOMENTUM"), "SERVER_LR": g("SERVER_LR"),
                     "FEDPROX_MU": g("FEDPROX_MU"), "scheduler": "none" if "lr_scheduler" not in t else "present",
                     "LR env override": "yes" if "FL_BASE_LR" in t else "no"})
    return pd.DataFrame(rows)


def fedhad_lr():
    rows = []
    for camp in (CAMPAIGN_05, CAMPAIGN_001):
        base = camp / "FedHAD-results"
        for p in sorted(base.rglob("*.txt")):
            r = parse_report(p)
            cl = r["clients"]
            if not cl or not any("lr_avg" in c for c in cl.values()):
                continue
            n = np.array([c["n"] for c in cl.values()], float)
            lr = np.array([c.get("lr_avg", np.nan) for c in cl.values()])
            ep = np.array([c.get("epochs_avg", np.nan) for c in cl.values()])
            live = n > 0
            rows.append({"campaign": camp.name, "tag": r.get("EXPERIMENT_TAG"), "dataset": r["dataset"],
                         "alpha": alpha_of(p), "seed": seed_of(p), "signature": r.get("HEURISTICS_SIGNATURE"),
                         "epochs_decay": r.get("EPOCHS_DECAY_FACTOR"), "lr_decay": r.get("LR_DECAY_FACTOR"),
                         "rounds": r.get("NUM_ROUNDS"), "clients": len(cl),
                         "lr_min": lr[live].min(), "lr_max": lr[live].max(), "lr_median": np.median(lr[live]),
                         "lr_wmean": (lr[live] * n[live]).sum() / n[live].sum(),
                         "ep_min": ep[live].min(), "ep_max": ep[live].max(),
                         "ep_wmean": (ep[live] * n[live]).sum() / n[live].sum(),
                         "empty_clients": int((~live).sum()),
                         "acc_final": r["acc_final"], "TF": (r["total_flops"] or 0) / 1e12})
    return pd.DataFrame(rows)


def main():
    c = consts()
    f = fedhad_lr()
    f.to_csv(OUT / "a04_fedhad_lr_per_run.csv", index=False)
    s = f.groupby(["campaign", "tag", "dataset", "alpha", "signature", "epochs_decay", "lr_decay"], dropna=False).agg(
        n=("seed", "size"), lr_min=("lr_min", "min"), lr_max=("lr_max", "max"), lr_median=("lr_median", "median"),
        lr_wmean=("lr_wmean", "mean"), ep_min=("ep_min", "min"), ep_max=("ep_max", "max"), ep_wmean=("ep_wmean", "mean"),
        empty_client_runs=("empty_clients", lambda x: int((x > 0).sum())),
        acc=("acc_final", "mean"), TF=("TF", "mean")).reset_index()
    s.to_csv(OUT / "a04_fedhad_lr_summary.csv", index=False)
    # the E0 arms of the historical ablation: 1 fixed epoch
    e0 = s[s.signature.fillna("").str.startswith("E0")]
    with open(OUT / "a04_summary.txt", "w") as fh:
        fh.write(c.to_string(index=False) + "\n\n")
        fh.write(s.to_string(index=False) + "\n\n")
        fh.write("Historical arms with USE_DYNAMIC_EPOCHS=False (1 fixed local epoch):\n" + e0.to_string(index=False) + "\n")
    print(c.to_string(index=False))
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
