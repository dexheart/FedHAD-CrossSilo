"""A06 - Independent recomputation of the paired statistics and Holm adjustment within
defensible comparison families (Section 5.6).

Final accuracy = last centralised accuracy of each raw report (own parser). Pairs are
aligned by seed (inner join), never by row order. Wilcoxon: two-sided,
zero_method='wilcox'. CI: 10,000-resample percentile bootstrap of the paired mean
(seed = unit). Raw p-values are kept; Holm is added per family, never across
families.

Families (declared here, before looking at the adjusted values):
  F1 principal  - CIFAR-10, 10 rounds, FedHAD vs {FedAvg, FedAvgM, FedProx, FedNova},
                  alpha in {1.0, 0.5, 0.1, 0.01}: 16 tests (supports the headline
                  heterogeneity claim).
  F2 secondary  - other datasets, 10 rounds: MNIST and FashionMNIST (4 alphas x 4
                  baselines) and FEMNIST (4 baselines): 36 tests.
  F3 persistence- 50 rounds: CIFAR-10 alpha=0.5, CIFAR-10 alpha=0.01, FEMNIST vs
                  {FedAvg, FedAvgM, FedProx} (FedNova was not run): 9 tests.
  F4 components - controlled ablation Q1-Q5b x 2 alphas: 12 tests (pre-declared
                  questions in component_analysis_ablation/README.md).
  Exploratory (no Holm; descriptive only): drift/prior correlations, decay grid,
                  historical 8-arm ablation, communication and client-count families.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import OUT, REV, CAMPAIGN_05, CAMPAIGN_001, parse_report, seed_of, alpha_of

RNG = np.random.default_rng(20260929)
BASE = ["FedAVG", "FedAvgM", "FedProx", "FedNova"]


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k])
        adj[k] = min(run, 1.0)
    return adj


def accs(camp, method, tag, ds, alpha=None, rounds=None):
    d = camp / f"{method}-results" / tag / (f"{ds}-Standard" if method == "FedHAD" else ds)
    out = {}
    for p in sorted(d.glob("*.txt")):
        if alpha is not None and (alpha_of(p) is None or abs(alpha_of(p) - alpha) > 1e-9):
            continue
        r = parse_report(p)
        if rounds is not None and int(r.get("NUM_ROUNDS", 0)) != rounds:
            continue
        out[seed_of(p)] = r["acc_final"]
    return pd.Series(out, dtype=float)


def test(a: pd.Series, b: pd.Series):
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    d = 100 * (j.iloc[:, 0] - j.iloc[:, 1]).to_numpy()
    boot = d[RNG.integers(0, len(d), (10000, len(d)))].mean(1)
    w = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"n": len(d), "mean_pp": d.mean(), "median_pp": np.median(d),
            "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
            "dz": d.mean() / d.std(ddof=1), "p_raw": w.pvalue,
            "wins_fedhad": int((d > 0).sum()), "wins_other": int((d < 0).sum())}


def main():
    rows = []
    # F1
    for a, tag in [(1.0, "test2_robustez_alpha"), (0.5, "test1_convergencia"), (0.1, "test2_robustez_alpha"), (0.01, "test2_robustez_alpha")]:
        fh = accs(CAMPAIGN_05, "FedHAD", tag, "CIFAR10", a)
        for b in BASE:
            rows.append({"family": "F1_principal_CIFAR10_10r", "dataset": "CIFAR10", "alpha": a, "baseline": b,
                         **test(fh, accs(CAMPAIGN_05, b, tag, "CIFAR10", a))})
    # F2
    for ds in ["MNIST", "FashionMNIST"]:
        for a, tag in [(1.0, "test2_robustez_alpha"), (0.5, "test1_convergencia"), (0.1, "test2_robustez_alpha"), (0.01, "test2_robustez_alpha")]:
            fh = accs(CAMPAIGN_05, "FedHAD", tag, ds, a)
            for b in BASE:
                rows.append({"family": "F2_secondary_10r", "dataset": ds, "alpha": a, "baseline": b,
                             **test(fh, accs(CAMPAIGN_05, b, tag, ds, a))})
    fh = accs(CAMPAIGN_05, "FedHAD", "test8_femnist", "FEMNIST")
    for b in BASE:
        rows.append({"family": "F2_secondary_10r", "dataset": "FEMNIST", "alpha": np.nan, "baseline": b,
                     **test(fh, accs(CAMPAIGN_05, b, "test8_femnist", "FEMNIST"))})
    # F3
    for camp, ds, tag, a, lab in [(CAMPAIGN_05, "CIFAR10", "test7_plato", 0.5, "CIFAR10_a0.5"),
                                  (CAMPAIGN_001, "CIFAR10", "test7_plato", 0.01, "CIFAR10_a0.01"),
                                  (CAMPAIGN_001, "FEMNIST", "test8_femnist", None, "FEMNIST")]:
        fh = accs(camp, "FedHAD", tag, ds, a, rounds=50)
        for b in ["FedAVG", "FedAvgM", "FedProx"]:
            rows.append({"family": "F3_persistence_50r", "dataset": lab, "alpha": a, "baseline": b,
                         **test(fh, accs(camp, b, tag, ds, a, rounds=50))})
    # F4 from the ablation raw finals
    fm = pd.read_csv(REV / "component_analysis_ablation/tables/final_metrics_official.csv")
    pairs = [("full_fedhad", "fixed_matched", "Q1"), ("full_fedhad", "permuted_allocation", "Q2"),
             ("epoch_only", "fixed_matched", "Q3"), ("lr_only_matched", "fixed_matched", "Q4"),
             ("full_fedhad", "epoch_only", "Q5a"), ("full_fedhad", "lr_only_matched", "Q5b")]
    for a in (0.01, 0.1):
        for x, y, q in pairs:
            A = fm[(fm.variant == x) & (fm.alpha == a)].set_index("seed").acc_final
            B = fm[(fm.variant == y) & (fm.alpha == a)].set_index("seed").acc_final
            rows.append({"family": "F4_components", "dataset": "CIFAR10", "alpha": a, "baseline": f"{q}: {x} - {y}", **test(A, B)})
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for f, g in df.groupby("family"):
        df.loc[g.index, "p_holm"] = holm(g.p_raw.values)
    # sensitivity: gatekeeping split of F4 (Q1 declared as the headline test in the ablation README)
    f4 = df.family == "F4_components"
    q1 = f4 & df.baseline.str.startswith("Q1")
    df["p_holm_F4_split"] = np.nan
    df.loc[q1, "p_holm_F4_split"] = holm(df.loc[q1, "p_raw"].values)
    df.loc[f4 & ~q1, "p_holm_F4_split"] = holm(df.loc[f4 & ~q1, "p_raw"].values)
    df["decision_raw_0.05"] = np.where(df.p_raw < 0.05, "reject", "not rejected")
    df["decision_holm_0.05"] = np.where(df.p_holm < 0.05, "reject", "not rejected")
    df["changes_with_holm"] = df["decision_raw_0.05"] != df["decision_holm_0.05"]
    df.to_csv(OUT / "a06_holm_families.csv", index=False)
    pd.set_option("display.width", 250)
    txt = df.round(4).to_string(index=False)
    open(OUT / "a06_summary.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
