"""A09 - Epochs vs optimizer steps and first-order objective reweighting
(Section 3).

For every client of the principal configurations (main campaign, alpha=0.5 tree and
FEMNIST): tau_k = E_k * floor(n_k / 32) executed optimizer steps (drop_last=True).
Under the plain-SGD first-order approximation used in the manuscript, the aggregate
update is  sum_k p_k * eta_k * tau_k * grad F_k,  p_k = n_k / N (sample weighting).
The effective objective weight is therefore q_k ∝ p_k eta_k tau_k, versus the
intended q_k = p_k. (Local momentum 0.9 multiplies every eta_k by the same 1/(1-beta)
in steady state and does not change the relative weights.) We report
TV(q, p) = 0.5 * sum |q_k - p_k| for the fixed-epoch baseline and for FedHAD, and
the within-seed spread of tau_k versus E_k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT, CAMPAIGN_05, parse_report, seed_of, alpha_of


def tv(q, p):
    return 0.5 * np.abs(q / q.sum() - p / p.sum()).sum()


def main():
    cells = [("test1_convergencia", "CIFAR10", 0.5), ("test2_robustez_alpha", "CIFAR10", 1.0),
             ("test2_robustez_alpha", "CIFAR10", 0.1), ("test2_robustez_alpha", "CIFAR10", 0.01),
             ("test1_convergencia", "MNIST", 0.5), ("test2_robustez_alpha", "MNIST", 0.01),
             ("test8_femnist", "FEMNIST", None)]
    per_seed, per_client = [], []
    for tag, ds, a in cells:
        d = CAMPAIGN_05 / "FedHAD-results" / tag / f"{ds}-Standard"
        for p in sorted(d.glob("*.txt")):
            if a is not None and abs(alpha_of(p) - a) > 1e-9:
                continue
            r = parse_report(p)
            cl = pd.DataFrame(r["clients"]).T
            cl = cl[cl.n > 0].astype(float)
            b = np.floor(cl.n / 32)
            tau_h = cl.epochs_avg * b
            tau_b = 5 * b
            pk = cl.n.to_numpy()
            q_h = (pk * cl.lr_avg * tau_h).to_numpy()
            q_b = (pk * 0.01 * tau_b).to_numpy()
            per_seed.append({"tag": tag, "dataset": ds, "alpha": a, "seed": seed_of(p), "clients": len(cl),
                             "E_ratio_max_min": cl.epochs_avg.max() / cl.epochs_avg.min(),
                             "tau_ratio_max_min_fedhad": tau_h.max() / max(tau_h.min(), 1),
                             "tau_min_fedhad": tau_h.min(), "tau_max_fedhad": tau_h.max(),
                             "samples_dropped_pct": 100 * (1 - (b * 32).sum() / cl.n.sum()),
                             "TV_fixed5_vs_p": tv(q_b, pk), "TV_fedhad_vs_p": tv(q_h, pk),
                             "maxweight_p": pk.max() / pk.sum(),
                             "maxweight_fixed5": q_b.max() / q_b.sum(), "maxweight_fedhad": q_h.max() / q_h.sum()})
            for (cid, row), th in zip(cl.iterrows(), tau_h):
                per_client.append({"tag": tag, "dataset": ds, "alpha": a, "seed": seed_of(p), "client": cid,
                                   "n": row.n, "batches": int(row.n // 32), "E": row.epochs_avg, "lr": row.lr_avg,
                                   "tau": th, "eta_tau": row.lr_avg * th})
    s = pd.DataFrame(per_seed)
    s.to_csv(OUT / "a09_steps_reweighting_per_seed.csv", index=False)
    pd.DataFrame(per_client).to_csv(OUT / "a09_steps_per_client.csv", index=False)
    g = s.groupby(["dataset", "alpha"], dropna=False).mean(numeric_only=True).drop(columns="seed")
    txt = g.round(4).to_string()
    open(OUT / "a09_summary.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
