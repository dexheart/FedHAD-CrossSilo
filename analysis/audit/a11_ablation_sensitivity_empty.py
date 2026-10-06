"""A11 - Sensitivity analysis of the component analysis excluding the five alpha=0.01
seeds that contain an empty client (46, 48, 51, 52, 57). Read-only.

The primary analysis (30 seeds per alpha) is NOT replaced. The family definition is the
one used in the manuscript: F4 = the 12 contrasts Q1-Q5b x 2 alphas (Holm), plus the hierarchical
variant (Q1 alone, then the remaining 10). alpha=0.1 has no empty client, so its rows are
identical to the primary analysis; they are kept inside the family so that the Holm
adjustment is computed over the same 12 tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import OUT, REV

EMPTY = {46, 48, 51, 52, 57}
PAIRS = [("full_fedhad", "fixed_matched", "Q1"), ("full_fedhad", "permuted_allocation", "Q2"),
         ("epoch_only", "fixed_matched", "Q3"), ("lr_only_matched", "fixed_matched", "Q4"),
         ("full_fedhad", "epoch_only", "Q5a"), ("full_fedhad", "lr_only_matched", "Q5b")]


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(run, 1.0)
    return adj


def stats(d, seed):
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    w = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"n": len(d), "mean_pp": d.mean(), "median_pp": np.median(d),
            "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
            "W": w.statistic, "p_raw": w.pvalue, "dz": d.mean() / d.std(ddof=1),
            "wins_first": int((d > 0).sum()), "wins_second": int((d < 0).sum()), "ties": int((d == 0).sum())}


def run(fm, exclude):
    rows = []
    for a in (0.01, 0.1):
        g = fm[fm.alpha == a]
        if exclude and a == 0.01:
            g = g[~g.seed.isin(EMPTY)]
        piv = g.pivot(index="seed", columns="variant", values="acc_final")
        for x, y, q in PAIRS:
            d = 100 * (piv[x] - piv[y]).dropna().to_numpy()
            rows.append({"Q": q, "contrast": f"{x} - {y}", "alpha": a, **stats(d, 20260929)})
    r = pd.DataFrame(rows)
    r["p_holm_F4"] = holm(r.p_raw)
    q1 = r.Q == "Q1"
    r["p_holm_split"] = np.nan
    r.loc[q1, "p_holm_split"] = holm(r.loc[q1, "p_raw"])
    r.loc[~q1, "p_holm_split"] = holm(r.loc[~q1, "p_raw"])
    return r


def main():
    fm = pd.read_csv(REV / "component_analysis_ablation/tables/final_metrics_official.csv")
    prim, sens = run(fm, False), run(fm, True)
    m = prim.merge(sens, on=["Q", "contrast", "alpha"], suffixes=("_30", "_excl"))
    m["decision_30"] = np.where(m.p_holm_F4_30 < 0.05, "reject", "n.s.")
    m["decision_excl"] = np.where(m.p_holm_F4_excl < 0.05, "reject", "n.s.")
    m["decision_split_30"] = np.where(m.p_holm_split_30 < 0.05, "reject", "n.s.")
    m["decision_split_excl"] = np.where(m.p_holm_split_excl < 0.05, "reject", "n.s.")
    m["raw_decision_30"] = np.where(m.p_raw_30 < 0.05, "reject", "n.s.")
    m["raw_decision_excl"] = np.where(m.p_raw_excl < 0.05, "reject", "n.s.")
    m.to_csv(OUT / "a11_ablation_sensitivity_empty.csv", index=False)
    cols = ["Q", "alpha", "n_30", "mean_pp_30", "n_excl", "mean_pp_excl", "median_pp_excl", "ci_lo_excl", "ci_hi_excl",
            "dz_excl", "wins_first_excl", "wins_second_excl", "p_raw_30", "p_raw_excl", "p_holm_F4_30", "p_holm_F4_excl",
            "p_holm_split_30", "p_holm_split_excl", "decision_30", "decision_excl", "decision_split_30", "decision_split_excl"]
    txt = m[cols].round(4).to_string(index=False)
    open(OUT / "a11_summary.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
