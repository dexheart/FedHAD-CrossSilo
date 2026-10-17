# -*- coding: utf-8 -*-
"""Multiplicity control for the equivalence
and non-inferiority decisions of the pre-registered batteries.

The batteries classify each contrast with Holm-adjusted Wilcoxon tests for
superiority and with unadjusted 90% bootstrap intervals (TOST at 5%) for
equivalence and non-inferiority. This script recomputes, from the per-seed
results, the equivalence / non-inferiority decisions under multiplicity
control within the same families used for superiority:

  * Bonferroni-adjusted bootstrap interval: TOST at level 0.05/m corresponds
    to the (1 - 2*0.05/m) percentile interval;
  * Holm over the TOST Wilcoxon p-values (max of the two one-sided p) for
    equivalence, and over the lower one-sided p for non-inferiority.

Bootstrap: percentile over seeds, 10,000 resamples, generator seed 20260930,
as in the pre-registrations. Output: analysis/audit/out/a16_equivalence_multiplicity.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "out" / "a16_equivalence_multiplicity.csv"

BOOT_RESAMPLES = 10_000
BOOT_SEED = 20260930
ALPHA = 0.05


def boot_ci(d, level):
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(BOOT_RESAMPLES, len(d)))
    means = d[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))


def w(x, alt):
    x = np.asarray(x, float)
    if np.all(x == 0):
        return 1.0
    return float(wilcoxon(x, alternative=alt, zero_method="wilcox").pvalue)


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def contrasts():
    """Yield (battery, family, alpha, label, delta, d) with d in percentage points."""
    # Tuned controls
    t = pd.read_csv(RES / "tuned_controls_fedprox_validation/evaluate_analysis/per_seed.csv")
    margins = {0.1: 0.75, 0.01: 1.0}
    for a in (0.1, 0.01):
        s = t[t.alpha == a]
        yield "tuned", "primary", a, "FedHAD - tuned", margins[a], (s.acc_fedhad - s.acc_tuned).to_numpy()
    for a in (0.1, 0.01):
        s = t[t.alpha == a]
        yield "tuned", "secondary", a, "FedHAD - default", margins[a], (s.acc_fedhad - s.acc_default).to_numpy()
        yield "tuned", "secondary", a, "tuned - default", margins[a], (s.acc_tuned - s.acc_default).to_numpy()
    s = t[(t.alpha == 0.01) & (t.seed <= 71)]
    yield "tuned", "sensitivity30", 0.01, "FedHAD - tuned", 1.0, (s.acc_fedhad - s.acc_tuned).to_numpy()

    # Uniform-learning-rate control
    r = pd.read_csv(RES / "lr_uniform_control/analysis/runs_tidy.csv")
    # "level" marks primary (1) vs secondary (2) arms, not a subset of runs
    piv = r.pivot_table(index=["alpha", "seed"], columns="arm", values="final_acc_pp")
    for a in (0.1, 0.01):
        p = piv.loc[a]
        yield "lru", f"primary@{a}", a, "Full - U2", margins[a], (p.full_fedhad - p.u2).to_numpy()
        for A, B, lab in (("lr_only_matched", "u1", "LR-only - U1"),
                          ("full_fedhad", "u2_arith", "Full - U2-arith"),
                          ("full_fedhad", "inv", "Full - INV")):
            yield "lru", f"secondary@{a}", a, lab, margins[a], (p[A] - p[B]).to_numpy()

    # Global-gradient diagnostic, H3 allocation on equal-size partitions
    g = pd.read_csv(RES / "global_gradient_diagnostic/analysis/accuracy.csv")
    g = g[g.partition == "equal"].pivot_table(index=["alpha", "seed"], columns="arm", values="accuracy")
    for a in (0.1, 0.2):
        p = g.loc[a]
        yield "gg_H3", "primary", a, "Full - INV epochs", 0.75, (p.full_fedhad - p.inv_epochs).to_numpy()


def main():
    rows = [dict(battery=b, family=f, alpha=a, contrast=c, delta=dl, d=d)
            for b, f, a, c, dl, d in contrasts()]
    df = pd.DataFrame(rows)
    out = []
    for (b, f), grp in df.groupby(["battery", "family"], sort=False):
        m = len(grp)
        recs = []
        for _, r in grp.iterrows():
            d, dl = np.asarray(r.d, float), r.delta
            ci90 = boot_ci(d, 0.90)
            lvl_adj = 1.0 - 2.0 * ALPHA / m
            ci_adj = boot_ci(d, lvl_adj)
            p_lo, p_up = w(d + dl, "greater"), w(d - dl, "less")
            recs.append(dict(
                battery=b, family=f, m=m, alpha=r.alpha, contrast=r.contrast, n=len(d), delta=dl,
                mean=d.mean(), ci90_lo=ci90[0], ci90_hi=ci90[1],
                adj_level=lvl_adj, ciadj_lo=ci_adj[0], ciadj_hi=ci_adj[1],
                p_sup=w(d, "two-sided"), p_tost=max(p_lo, p_up), p_ni=p_lo,
                eq_unadj=(ci90[0] > -dl) and (ci90[1] < dl),
                eq_bonf=(ci_adj[0] > -dl) and (ci_adj[1] < dl),
                ni_unadj=ci90[0] > -dl,
                ni_bonf=ci_adj[0] > -dl,
            ))
        ph_sup = holm([x["p_sup"] for x in recs])
        ph_tost = holm([x["p_tost"] for x in recs])
        ph_ni = holm([x["p_ni"] for x in recs])
        for x, a1, a2, a3 in zip(recs, ph_sup, ph_tost, ph_ni):
            x.update(p_sup_holm=a1, p_tost_holm=a2, p_ni_holm=a3,
                     eq_tost_holm=a2 < ALPHA, ni_holm=a3 < ALPHA)
            out.append(x)

    # Alternative family for the primary uniform-LR contrast: both alphas together
    alt = [x for x in out if x["battery"] == "lru" and x["family"].startswith("primary")]
    sub = df[(df.battery == "lru") & df.family.str.startswith("primary")]
    for x, (_, r) in zip(alt, sub.iterrows()):
        d = np.asarray(r.d, float)
        lo, hi = boot_ci(d, 0.95)
        out.append({**x, "family": "primary(both alphas)", "m": 2, "adj_level": 0.95,
                    "ciadj_lo": lo, "ciadj_hi": hi,
                    "eq_bonf": (lo > -x["delta"]) and (hi < x["delta"]), "ni_bonf": lo > -x["delta"]})
    ph = holm([x["p_tost"] for x in alt])
    for x, p in zip(out[-2:], ph):
        x["p_tost_holm"] = p
        x["eq_tost_holm"] = p < ALPHA
    phs = holm([x["p_sup"] for x in alt])
    for x, p in zip(out[-2:], phs):
        x["p_sup_holm"] = p

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(OUT, index=False, float_format="%.4f")
    cols = ["battery", "family", "m", "alpha", "contrast", "n", "delta", "mean",
            "ci90_lo", "ci90_hi", "adj_level", "ciadj_lo", "ciadj_hi",
            "p_sup_holm", "p_tost_holm", "p_ni_holm",
            "eq_unadj", "eq_bonf", "eq_tost_holm", "ni_unadj", "ni_bonf", "ni_holm"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(pd.DataFrame(out)[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
