# -*- coding: utf-8 -*-
"""
Statistics and the pre-registered classification rule of the LR-uniform battery.
Pure functions (numpy + scipy); unit-tested in lr_uniform_control/tests.

Differences are paired, per seed, in percentage points: d_i = arm_A_i - arm_B_i.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import wilcoxon

BOOT_RESAMPLES = 10_000
BOOT_SEED = 20260930
Z_90 = 1.6448536269514722          # two-sided 90% normal quantile (one-sided 95%)

CATEGORY = {
    1: "1: heterogeneity-based LR assignment superior",
    2: "2: practically equivalent (within +/- delta)",
    3: "3: uniform LR superior",
    4: "4: inconclusive",
}


def bootstrap_ci(d, level, resamples=BOOT_RESAMPLES, seed=BOOT_SEED):
    """Percentile bootstrap CI of the mean over seeds (resampling seeds)."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    means = d[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))


def d_z(d):
    d = np.asarray(d, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def _wilcoxon(x, alternative):
    x = np.asarray(x, dtype=float)
    if np.all(x == 0):
        return 1.0
    return float(wilcoxon(x, alternative=alternative, zero_method="wilcox").pvalue)


def wilcoxon_two_sided(d):
    return _wilcoxon(d, "two-sided")


def tost_wilcoxon(d, delta):
    """Two one-sided Wilcoxon tests against the equivalence bounds.
    p_lower: H0 median <= -delta vs H1 > -delta  (test on d + delta, 'greater')
    p_upper: H0 median >= +delta vs H1 < +delta  (test on d - delta, 'less')"""
    d = np.asarray(d, dtype=float)
    return _wilcoxon(d + delta, "greater"), _wilcoxon(d - delta, "less")


def holm(pvalues):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return [float(x) for x in adj]


def seeds_needed(d, delta):
    """Simple extrapolation: seeds for which the 90% interval (normal approximation,
    half-width z_0.95 * sd / sqrt(n)) would fit inside +/- delta around the observed
    mean. None when |mean| >= delta (more seeds alone cannot place it inside)."""
    d = np.asarray(d, dtype=float)
    room = delta - abs(float(d.mean()))
    sd = float(d.std(ddof=1))
    if room <= 0:
        return None
    if sd == 0:
        return len(d)
    return int(math.ceil((Z_90 * sd / room) ** 2))


def classify(*, p_holm, ci95, ci90, delta, alpha_level=0.05):
    """Pre-registered rule, applied mechanically:
      1 if p_holm < 0.05 and the whole 95% CI is above zero;
      3 if p_holm < 0.05 and the whole 95% CI is below zero;
      2 if neither, and the whole 90% CI lies inside (-delta, +delta);
      4 otherwise (inconclusive)."""
    lo95, hi95 = ci95
    lo90, hi90 = ci90
    if p_holm < alpha_level and lo95 > 0:
        return 1
    if p_holm < alpha_level and hi95 < 0:
        return 3
    if -delta < lo90 and hi90 < delta:
        return 2
    return 4


def contrast_summary(d, *, delta):
    """Everything the table needs for one contrast, except the Holm-adjusted p (which
    depends on the family) and the category (which needs it)."""
    d = np.asarray(d, dtype=float)
    ci95 = bootstrap_ci(d, 0.95)
    ci90 = bootstrap_ci(d, 0.90)
    p_low, p_up = tost_wilcoxon(d, delta)
    return {
        "n": int(len(d)), "mean_diff_pp": float(d.mean()), "sd_diff_pp": float(d.std(ddof=1)),
        "ci95": ci95, "ci90": ci90, "d_z": d_z(d),
        "p_wilcoxon": wilcoxon_two_sided(d),
        "p_tost_lower": p_low, "p_tost_upper": p_up,
        "n_favour_A": int((d > 0).sum()), "n_favour_B": int((d < 0).sum()), "n_ties": int((d == 0).sum()),
        "ci90_half_width": float((ci90[1] - ci90[0]) / 2.0),
        "seeds_needed_for_equivalence": seeds_needed(d, delta),
        "delta": float(delta),
    }
