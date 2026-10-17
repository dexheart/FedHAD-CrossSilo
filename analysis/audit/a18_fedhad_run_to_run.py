# -*- coding: utf-8 -*-
"""Run-to-run variation of the complete FedHAD policy across the control batteries
(Section 5.6, "Bitwise reproducibility").

Three batteries execute the complete policy on CIFAR-10, seeds 42-71, alpha in
{0.1, 0.01}, with the same seed, partition, checksummed checkpoint and worker
seeding: the step-preserving permutation (whose FedHAD runs the tuned-control
evaluation reuses), the uniform-learning-rate control and the global-gradient
diagnostic (Dirichlet level). For each pair of batteries the script reports the
mean and maximum absolute difference in final accuracy over the 60 seed-level pairs.

Run: python3 -I analysis/audit/a18_fedhad_run_to_run.py
"""
import itertools
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parent / "out" / "a18_fedhad_run_to_run.csv"


def main():
    s = pd.read_csv(RES / "step_preserving_permutation/analysis/per_seed.csv")
    s = s[["alpha", "seed", "acc_full_pp"]].rename(columns={"acc_full_pp": "step_permutation"})
    l = pd.read_csv(RES / "lr_uniform_control/analysis/runs_tidy.csv")
    l = l[l.arm == "full_fedhad"][["alpha", "seed", "final_acc_pp"]].rename(columns={"final_acc_pp": "lr_uniform"})
    g = pd.read_csv(RES / "global_gradient_diagnostic/analysis/accuracy.csv")
    g = g[(g.partition == "dirichlet") & (g.arm == "full_fedhad")][["alpha", "seed", "accuracy"]]
    g = g.rename(columns={"accuracy": "global_gradient"})
    m = s.merge(l, on=["alpha", "seed"]).merge(g, on=["alpha", "seed"])
    rows = []
    for a, b in itertools.combinations(["step_permutation", "lr_uniform", "global_gradient"], 2):
        d = (m[a] - m[b]).abs()
        rows.append({"battery_a": a, "battery_b": b, "pairs": len(d),
                     "mean_abs_diff_pp": round(d.mean(), 3), "max_abs_diff_pp": round(d.max(), 3)})
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
