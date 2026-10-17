# -*- coding: utf-8 -*-
"""Step-preserving permutation: small clients and sensitivity analysis.

1. Lists the clients with fewer than B = 32 training samples that entered the
   permutation, with the budget they gave and received (round 1 telemetry).
2. Recomputes the primary contrast Full - permuted at both levels and the
   sensitivity analysis that excludes the seeds whose processed examples differ
   between the arms, with 95% percentile bootstrap CI (10,000 resamples, seed
   20260930), d_z, Wilcoxon p and Holm over the two levels.
3. Reports per-arm means over the retained seeds (updates, examples, executed
   TFLOPs) to show they are matched exactly.

Run: /usr/bin/python3 -I analysis/audit/a17_step_permutation_sensitivity.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "step_preserving_permutation"


def boot(x, level=0.95):
    rng = np.random.default_rng(20260930)
    m = x[rng.integers(0, len(x), (10_000, len(x)))].mean(1)
    a = (1 - level) / 2
    return np.quantile(m, a), np.quantile(m, 1 - a)


def holm(p):
    p = np.asarray(p, float)
    order, m, run, adj = np.argsort(p), len(p), 0.0, np.empty(len(p))
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def summary(x):
    lo, hi = boot(x)
    return dict(n=len(x), mean=x.mean(), ci_lo=lo, ci_hi=hi, dz=x.mean() / x.std(ddof=1),
                wins=int((x > 0).sum()), losses=int((x < 0).sum()), p=wilcoxon(x).pvalue)


def main():
    tel = pd.concat(pd.read_csv(f) for f in glob.glob(
        str(RES / "raw/step_permuted_lrclient__CIFAR10__alpha-*__seed-*/telemetry.csv")))
    r1 = tel[tel["round"] == 1]
    small = r1[(r1.n_k > 0) & (r1.n_k < 32)]
    print("Clients with 0 < n_k < 32 in the permuted arm (round 1):")
    print(small[["alpha", "seed", "client_id", "n_k", "n_batches", "tau_original", "tau_received",
                 "examples_processed"]].to_string(index=False))

    d = pd.read_csv(RES / "analysis/per_seed.csv")
    affected = d[d.examples_mismatch_pct != 0][["alpha", "seed", "examples_mismatch_pct", "diff_acc_pp"]]
    print("\nSeeds whose processed examples differ between arms:\n", affected.to_string(index=False))
    excl = set(affected.seed[affected.alpha == 0.01])

    a01 = d[d.alpha == 0.01]
    rows = {
        "primary 0.01": summary(a01.diff_acc_pp.to_numpy()),
        "primary 0.1": summary(d[d.alpha == 0.1].diff_acc_pp.to_numpy()),
        "sensitivity 0.01": summary(a01[~a01.seed.isin(excl)].diff_acc_pp.to_numpy()),
    }
    ph_primary = holm([rows["primary 0.01"]["p"], rows["primary 0.1"]["p"]])
    ph_sens = holm([rows["sensitivity 0.01"]["p"], rows["primary 0.1"]["p"]])
    rows["primary 0.01"]["p_holm"], rows["primary 0.1"]["p_holm"] = ph_primary
    rows["sensitivity 0.01"]["p_holm"] = ph_sens[0]
    print("\n", pd.DataFrame(rows).T.round(4).to_string())

    kept = a01[~a01.seed.isin(excl)]
    print("\nRetained seeds, accuracy mean ± sd: full %.2f ± %.2f, permuted %.2f ± %.2f" % (
        kept.acc_full_pp.mean(), kept.acc_full_pp.std(), kept.acc_perm_pp.mean(), kept.acc_perm_pp.std()))
    disp = pd.read_csv(RES / "analysis/displacement.csv")
    dk = disp[(disp.alpha == 0.01) & (~disp.seed.isin(excl))]
    print("sum p tau ratio median [min, max]: %.2f [%.2f, %.2f]" % (
        dk.ratio_sum_p_tau.median(), dk.ratio_sum_p_tau.min(), dk.ratio_sum_p_tau.max()))
    from scipy.stats import spearmanr
    print("\nSpearman(paired difference, ratio), per level (Section 6.11):")
    for a in (0.01, 0.1):
        x = disp[disp.alpha == a]
        r1 = spearmanr(x.diff_acc_pp, x.ratio_sum_p_tau)
        r2 = spearmanr(x.diff_acc_pp, x.ratio_sum_p_eta_tau)
        print(f"  alpha={a}: vs sum p*tau rho={r1.statistic:+.2f} (p={r1.pvalue:.4f}); "
              f"vs sum p*eta*tau rho={r2.statistic:+.2f} (p={r2.pvalue:.4f}), "
              f"median ratio p*eta*tau={x.ratio_sum_p_eta_tau.median():.2f}")
    tot = []
    for arm in ("full_fedhad", "step_permuted_lrclient"):
        for f in glob.glob(str(RES / f"raw/{arm}__CIFAR10__alpha-0.01__seed-*/telemetry.csv")):
            t = pd.read_csv(f)
            s = int(t.seed.iloc[0])
            if s in excl or s > 71:
                continue
            tot.append((arm, s, t.steps_executed.sum(), t.examples_processed.sum(), t.flops_executed.sum() / 1e12))
    T = pd.DataFrame(tot, columns=["arm", "seed", "updates", "examples", "tflops_exec"])
    print(T.groupby("arm")[["updates", "examples", "tflops_exec"]].mean())


if __name__ == "__main__":
    main()
