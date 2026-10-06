#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing the uniform-global-prior assumption embedded in H_k (Sections 6.4 and 6.8 of the manuscript).

OFFLINE analysis: no training and no federated simulation is executed. Nothing
outside this directory is written.

Background
----------
The heterogeneity score used throughout the paper is a normalised coefficient of
variation of a client's per-class sample counts:

    H_k = ( std(counts) / mean(counts) ) / sqrt(K - 1)

This implicitly measures departure from a *uniform* class prior:
it is zero exactly when the client's classes are equally represented. In a
naturally federated dataset the global prior is not uniform, so a client that
perfectly mirrors the federation can still receive a high H_k. The assumption is
not true in general; the manuscript states it explicitly (Sections 1, 2 and 4.2).

What this script does
---------------------
Part 1 (Dirichlet, CIFAR-10) - does an alternative heterogeneity metric predict
client drift better than the CV? Four per-client metrics are computed and each is
correlated (Spearman) against the logged cosine similarity, per method and per
alpha, with bootstrap confidence intervals resampled over seeds.

  IMPORTANT: CIFAR-10's empirical global prior is UNIFORM by construction (5,000
  images per class). Part 1 therefore CANNOT test the failure mode of a non-uniform prior -
  it characterises the regime in which the uniform hypothesis is satisfied, and
  asks only whether a different functional form predicts drift better.

  The three baselines (FedAvg, FedAvgM, FedProx) are reported first, because there
  H_k is purely observational. In FedHAD, H_k -> E_k/LR -> update -> cos_sim is a
  causal path, so the within-FedHAD correlation is not an independent validation.

Part 2 (FEMNIST) - no drift telemetry exists for FEMNIST, so NO correlation with
drift is attempted and none is implied. The empirical global prior is estimated,
its departure from uniform quantified, and the *ranking* disagreement between the
CV (uniform reference) and the Jensen-Shannon divergence (empirical global
reference) is measured over an aggregated pool of distinct writers.

Part 2b - the campaign's actual FEMNIST federation (10 client writers, 30 seeds):
a within-federation, budget-preserving counterfactual rank remapping. The exact
multiset of epoch budgets assigned by the CV is redistributed by the JS ranking,
so sum(E_k) is identical before and after WITHIN those 10 clients. This is a
counterfactual reallocation, never an algorithm: it is not "FedHAD-JS", nothing
was trained with it, and it makes no claim about drift or accuracy.

What Part 2 establishes is an APPLICABILITY BOUNDARY for the CV when the global
prior is non-uniform - not the superiority of JS.

Metric orientation
------------------
All four metrics are oriented so that a HIGHER value means MORE heterogeneous,
which matches the sign convention of the existing drift analysis:

  cv_norm     normalised coefficient of variation (the current H_k)
  entropy_het 1 - H(p_k)/log(K), i.e. one minus the normalised entropy
  js_global   Jensen-Shannon divergence (base 2) against the EMPIRICAL global prior
  tv_global   total variation distance (= L1 / 2) against the same empirical prior
  js_uniform  Jensen-Shannon divergence against the UNIFORM prior, reported so the
              effect of changing the reference distribution can be read directly

Data sources (read-only)
------------------------
  data_partition_manifest/artifacts/partitions/partition_manifest.csv
  data_partition_manifest/artifacts/partitions/femnist_heldout_manifest.csv
  drift_diagnostics_10seeds/1.1_diagnostico_drift/drift_telemetry.csv

One deviation, and why
----------------------
The 1e-6 validation guard cannot be satisfied from the partition manifest alone.
The manifest records each Dirichlet client's FULL partition, whereas the logged
H_k is computed over the 90% TRAINING SPLIT of that partition (verified: the
telemetry's n_samples is exactly 0.9 x the manifest's n_k). Recomputing the CV
from manifest counts therefore reproduces the logged H_k only to ~2e-3.

So the Dirichlet class counts are regenerated offline, deterministically, by
importing the campaign's own partitioning code and re-running the split - no
training. On that basis the guard passes: max absolute error 7.4e-08 over 147
informative clients (the 3 M2 clients compare sentinel 0.0 to 0.0 and are not
evidence), with the largest RELATIVE error at 0.82x float32 machine epsilon and
no case above 2x eps32 - i.e. the residual is the float32 representation error of
the logged value, which the campaign produces via torch.bincount(...).float().

Because H_k is exactly constant across rounds and methods when the partition is
static (verified: zero within-client span over all 150 clients), validating one
round validates all 6,000 telemetry rows.

The manifest remains an INDEPENDENT CROSS-CHECK rather than a second copy of the
same quantity: it records the full client partition, the telemetry the 90%
training split. The script reports their offset and rank correlation; this is a
consistency check, not an identity.

Reproducibility note (implementation, not a scientific result):
`dirichlet_split_noniid(dataset, n_clients, alpha, seed=SEED)` binds `seed` as a
DEFAULT ARGUMENT at import time, and `load_data()` never passes it. Setting
`module.SEED` after import therefore silently regenerates the seed-42 partition.
The campaign is unaffected (it sets FL_SEED before importing), but any offline
reuse must import once per seed, which is what this script does.

Usage
-----
    python analysis/global_prior_analysis/analyze_prior_assumption.py
    python analysis/global_prior_analysis/analyze_prior_assumption.py --bootstrap 2000
    python analysis/global_prior_analysis/analyze_prior_assumption.py --skip-regen   # manifest basis only
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

MANIFEST_CSV = (PROJECT_ROOT / "results" / "data_partition_manifest" /
                "artifacts" / "partitions" / "partition_manifest.csv")
HELDOUT_CSV = (PROJECT_ROOT / "results" / "data_partition_manifest" /
               "artifacts" / "partitions" / "femnist_heldout_manifest.csv")
DRIFT_CSV = (PROJECT_ROOT / "results" / "drift_diagnostics_10seeds" /
             "1.1_diagnostico_drift" / "drift_telemetry.csv")

OUT_ROOT = PROJECT_ROOT / "results" / "global_prior_analysis"   # outputs (code lives in analysis/)
TABLES = OUT_ROOT / "tables"
FIGURES = OUT_ROOT / "figures"
DATA = OUT_ROOT / "data"

# Clients that ended a round with no samples. Documented in the drift report as
# M2 and excluded from the analysis (not from the experiment): they stayed in the
# federation with weight n_k/N = 0, and their H_k / cos_sim / update_norm are
# sentinel zeros rather than measurements.
M2_EXCLUSIONS = {(0.01, 46, 1), (0.01, 48, 4), (0.01, 51, 2)}

ALPHAS = [1.0, 0.1, 0.01]
METHODS = ["FedAVG", "FedAvgM", "FedProx", "FedHAD"]
METHOD_LABEL = {"FedAVG": "FedAvg", "FedAvgM": "FedAvgM",
                "FedProx": "FedProx", "FedHAD": "FedHAD"}
METRICS = ["cv_norm", "entropy_het", "js_global", "tv_global", "js_uniform"]
METRIC_LABEL = {
    "cv_norm": "Normalised CV (current $H_k$)",
    "entropy_het": "1 - normalised entropy",
    "js_global": "JS vs empirical global prior",
    "tv_global": "Total variation vs global prior",
    "js_uniform": "JS vs uniform prior",
}

# FedHAD's epoch allocation rule (FedHAD_Final_2.0.py: define_epochs_dinamicas).
BASE_EPOCHS, MIN_EPOCHS, EPOCHS_DECAY_FACTOR = 5, 2, 3.0

# Okabe-Ito colourblind-safe qualitative palette.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
             "#56B4E9", "#F0E442", "#000000"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _probs(counts):
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum()
    return c / total if total > 0 else c


def cv_norm(counts, n_classes):
    c = np.asarray(counts, dtype=np.float64)
    if c.size == 0 or c.mean() == 0:
        return 0.0
    raw = c.std(ddof=0) / c.mean()
    return float(np.clip(raw / np.sqrt(n_classes - 1), 0.0, 1.0))


def entropy_het(counts, n_classes):
    """1 - H(p)/log(K): 0 when perfectly balanced, 1 when concentrated."""
    p = _probs(counts)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    h = -(p * np.log(p)).sum() / np.log(n_classes)
    return float(np.clip(1.0 - h, 0.0, 1.0))


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits; 0 = identical, 1 = disjoint support."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return float(np.clip(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0, 1.0))


def total_variation(p, q):
    """L1/2 distance in [0, 1]."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p, q = p / p.sum(), q / q.sum()
    return float(0.5 * np.abs(p - q).sum())


def all_metrics(counts, global_prior, n_classes):
    p = _probs(counts)
    uniform = np.full(n_classes, 1.0 / n_classes)
    return {
        "cv_norm": cv_norm(counts, n_classes),
        "entropy_het": entropy_het(counts, n_classes),
        "js_global": js_divergence(p, global_prior),
        "tv_global": total_variation(p, global_prior),
        "js_uniform": js_divergence(p, uniform),
    }


def cv_norm_float32(counts, n_classes):
    """
    The CV as the campaign computes it: `torch.bincount(...).float()` is float32,
    so std and mean are evaluated in single precision before the result leaves
    torch. Used only to confirm that the residual of the validation guard is
    floating-point precision rather than a reconstruction error.
    """
    c = np.asarray(counts, dtype=np.float32)
    if c.size == 0 or c.mean() == 0:
        return 0.0
    raw = float(c.std(ddof=0) / c.mean())
    return float(np.clip(raw / np.sqrt(n_classes - 1), 0.0, 1.0))


def epochs_from_h(h):
    """FedHAD's rule, reproduced exactly."""
    if h == 0.0:
        return BASE_EPOCHS
    return max(MIN_EPOCHS, int(round(BASE_EPOCHS - EPOCHS_DECAY_FACTOR * h)))


def within_federation_remap(epochs_cv, js_values):
    """
    Counterfactual rank remapping with a rigorously preserved budget.

    The EXACT multiset of epoch values that the CV assigned is kept and
    redistributed according to the JS ranking: the writer that diverges least
    from the global prior receives the largest budget, mirroring the direction of
    FedHAD's rule (lower heterogeneity -> more local epochs). By construction
    sum(E_k) is identical before and after, so the only thing that changes is
    WHICH writer holds WHICH budget.

    This is a counterfactual reallocation, not an algorithm: it is not FedHAD-JS,
    it was never trained, and it makes no claim about accuracy or drift.
    """
    epochs_cv = np.asarray(epochs_cv)
    js_values = np.asarray(js_values, dtype=float)
    budgets_desc = np.sort(epochs_cv)[::-1]          # largest budgets first
    order_by_js = np.argsort(js_values, kind="stable")  # least divergent first
    remapped = np.empty_like(epochs_cv)
    remapped[order_by_js] = budgets_desc
    assert remapped.sum() == epochs_cv.sum(), "budget must be preserved exactly"
    return remapped


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def bootstrap_ci_over_seeds(df, metric_col, target_col, n_boot, rng):
    """
    Percentile bootstrap CI for Spearman rho, resampling SEEDS (not the
    client-round observations, which are not independent within a seed).
    """
    seeds = np.array(sorted(df["seed"].unique()))
    if len(seeds) < 2:
        return np.nan, np.nan
    by_seed = {s: g for s, g in df.groupby("seed")}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        sample = pd.concat([by_seed[s] for s in pick], ignore_index=True)
        r = spearman(sample[metric_col].to_numpy(), sample[target_col].to_numpy())
        if not np.isnan(r):
            stats.append(r)
    if not stats:
        return np.nan, np.nan
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# --------------------------------------------------------------------------- #
# Read-only loaders
# --------------------------------------------------------------------------- #
def parse_counts(s):
    return np.array([int(x) for x in str(s).split(";")], dtype=np.int64)


@contextlib.contextmanager
def quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def load_sources():
    for p in (MANIFEST_CSV, HELDOUT_CSV, DRIFT_CSV):
        if not p.exists():
            raise FileNotFoundError(f"Required read-only source not found: {p}")
    manifest = pd.read_csv(MANIFEST_CSV)
    heldout = pd.read_csv(HELDOUT_CSV)
    drift = pd.read_csv(DRIFT_CSV)
    return manifest, heldout, drift


# --------------------------------------------------------------------------- #
# Part 1 - Dirichlet / CIFAR-10
# --------------------------------------------------------------------------- #
def regenerate_train_split_counts(seeds, alphas, n_clients=5, n_classes=10):
    """
    Regenerate the exact per-client TRAINING-SPLIT class counts by importing the
    campaign's own code. No training is executed - only partitioning.

    One import per seed: `dirichlet_split_noniid` binds `seed` as a default
    argument at import time, so patching module.SEED afterwards has no effect,
    whereas ALPHA is read at call time and can be varied freely.
    """
    from torch.utils.data import Subset

    os.environ["FL_DATASET"] = "CIFAR10"
    os.environ["FL_USE_ENERGY"] = "false"
    os.environ["FL_CLIENT_SETUP"] = str(n_clients)
    sys.path.insert(0, str(PROJECT_ROOT / "algoritmos"))
    prev_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)  # the method scripts resolve "./data" relatively
    rows = []
    try:
        for seed in seeds:
            os.environ["FL_SEED"] = str(seed)
            spec = importlib.util.spec_from_file_location(
                f"_prior_m{seed}", PROJECT_ROOT / "algoritmos" / "FedAVG_Final.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"_prior_m{seed}"] = mod
            with quiet():
                spec.loader.exec_module(mod)
            for alpha in alphas:
                mod.ALPHA = float(alpha)
                with quiet():
                    trainloaders, _val, _test = mod.load_data()
                for cid, dl in enumerate(trainloaders):
                    sub = dl.dataset
                    idx = list(getattr(sub, "indices", []))
                    while isinstance(getattr(sub, "dataset", None), Subset):
                        sub = sub.dataset
                        parent = list(sub.indices)
                        idx = [parent[i] for i in idx]
                    base = sub.dataset if hasattr(sub, "dataset") else sub
                    labels = np.asarray(base.targets)[idx] if idx else np.array([], int)
                    counts = np.bincount(labels, minlength=n_classes).astype(int)
                    rows.append({
                        "alpha": float(alpha), "seed": int(seed), "client_id": cid,
                        "n_k_train": int(len(idx)),
                        "label_counts": ";".join(str(c) for c in counts.tolist()),
                    })
            del mod
            sys.modules.pop(f"_prior_m{seed}", None)
    finally:
        os.chdir(prev_cwd)
    return pd.DataFrame(rows)


def manifest_partition_counts(manifest, n_clients=5):
    """Full-partition counts for CIFAR-10 from the read-only manifest."""
    c = manifest[(manifest.dataset == "CIFAR10") &
                 (manifest.n_clients == n_clients) &
                 (manifest.method == "FedAvg")]
    return c[["alpha", "seed", "client_id", "n_k", "label_counts"]].rename(
        columns={"n_k": "n_k_partition", "label_counts": "label_counts_partition"})


def run_guard(counts_df, drift, tolerance=1e-6, n_classes=10):
    """Recomputed CV must reproduce the logged H_k."""
    logged = (drift[drift["round"] == 1]
              .groupby(["alpha", "seed", "client_id"], as_index=False)["H_k"].first())
    merged = counts_df.merge(logged, on=["alpha", "seed", "client_id"], how="inner")
    merged["cv_recomputed"] = [cv_norm(parse_counts(s), n_classes)
                               for s in merged["label_counts"]]
    merged["abs_diff"] = (merged["cv_recomputed"] - merged["H_k"]).abs()
    return merged, float(merged["abs_diff"].max()), bool(
        (merged["abs_diff"] < tolerance).all())


def build_dirichlet_metrics(counts_df, global_prior, n_classes=10):
    recs = []
    for _, r in counts_df.iterrows():
        counts = parse_counts(r["label_counts"])
        m = all_metrics(counts, global_prior, n_classes)
        m.update({"alpha": r["alpha"], "seed": int(r["seed"]),
                  "client_id": int(r["client_id"]), "n_k_train": int(r["n_k_train"]),
                  "label_counts": r["label_counts"]})
        recs.append(m)
    return pd.DataFrame(recs)


def correlate_dirichlet(metrics_df, drift, n_boot, rng):
    obs = drift.merge(metrics_df.drop(columns=["label_counts"]),
                      on=["alpha", "seed", "client_id"], how="inner")
    mask = [(round(a, 4), int(s), int(c)) not in M2_EXCLUSIONS
            for a, s, c in zip(obs.alpha, obs.seed, obs.client_id)]
    obs_kept = obs[mask].copy()

    rows = []
    for method in METHODS:
        for alpha in ALPHAS:
            sub = obs_kept[(obs_kept.metodo == method) & (obs_kept.alpha == alpha)]
            for metric in METRICS:
                rho = spearman(sub[metric].to_numpy(), sub["cos_sim"].to_numpy())
                lo, hi = bootstrap_ci_over_seeds(sub, metric, "cos_sim", n_boot, rng)
                rows.append({
                    "method": METHOD_LABEL[method], "alpha": alpha, "metric": metric,
                    "n_obs": len(sub), "n_seeds": sub["seed"].nunique(),
                    "spearman_rho": rho, "ci95_low": lo, "ci95_high": hi,
                })
    return pd.DataFrame(rows), obs, obs_kept


# --------------------------------------------------------------------------- #
# Part 2 - FEMNIST
# --------------------------------------------------------------------------- #
def femnist_writer_table(manifest, heldout, n_classes=62):
    """
    Distinct FEMNIST writers with their class counts.

    Held-out writers carry their complete data and form a large, unbiased sample
    of the writer population; the campaign's client writers are also included and
    flagged, since they are the ones that actually receive an epoch budget.
    """
    rows = []
    for _, r in heldout.iterrows():
        rows.append({"writer": str(r["writer_natural_id"]), "role": "heldout_test",
                     "seed": int(r["seed"]), "n_k": int(r["n_k"]),
                     "label_counts": r["label_counts"]})
    fem = manifest[(manifest.dataset == "FEMNIST") & (manifest.method == "FedAvg")]
    for _, r in fem.iterrows():
        rows.append({"writer": str(r["writer_natural_id"]), "role": "client",
                     "seed": int(r["seed"]), "n_k": int(r["n_k"]),
                     "label_counts": r["label_counts"]})
    df = pd.DataFrame(rows)
    # One row per writer: the first occurrence (client role wins if present).
    df["role_rank"] = (df["role"] == "client").astype(int)
    df = (df.sort_values(["writer", "role_rank", "seed"], ascending=[True, False, True])
            .drop_duplicates(subset=["writer"], keep="first")
            .drop(columns=["role_rank"]).reset_index(drop=True))
    return df


def femnist_global_prior(writers_df, n_classes=62):
    total = np.zeros(n_classes, dtype=np.int64)
    for s in writers_df["label_counts"]:
        total += parse_counts(s)
    return total / total.sum(), total


def analyse_femnist(writers_df, prior, n_classes=62):
    recs = []
    for _, r in writers_df.iterrows():
        counts = parse_counts(r["label_counts"])
        m = all_metrics(counts, prior, n_classes)
        m.update({"writer": r["writer"], "role": r["role"], "n_k": int(r["n_k"]),
                  "n_classes_covered": int((counts > 0).sum())})
        recs.append(m)
    df = pd.DataFrame(recs)
    # To compare epoch allocations fairly, JS is mapped onto the CV's marginal
    # distribution by quantile matching (the JS value at rank r is replaced by the
    # CV value at rank r). This holds the total epoch budget and the shape of the
    # allocation FIXED, so every difference that remains is caused purely by the
    # two metrics ranking writers differently - which is the question being asked.
    # A min-max rescale would instead conflate re-ranking with a change of scale.
    order = np.argsort(np.argsort(df["js_global"].to_numpy()))
    cv_sorted = np.sort(df["cv_norm"].to_numpy())
    df["js_global_quantile_mapped"] = cv_sorted[order]
    df["epochs_cv"] = [epochs_from_h(h) for h in df["cv_norm"]]
    df["epochs_js"] = [epochs_from_h(h) for h in df["js_global_quantile_mapped"]]
    df["epochs_differ"] = df["epochs_cv"] != df["epochs_js"]
    for col, src in (("band_cv", "cv_norm"), ("band_js", "js_global")):
        df[col] = pd.qcut(df[src], 3, labels=["low", "medium", "high"])
    df["band_changed"] = df["band_cv"].astype(str) != df["band_js"].astype(str)
    return df


def analyse_campaign_federation(manifest, prior, n_classes=62):
    """
    Item (7): the campaign's ACTUAL FEMNIST federation - the 10 client writers -
    analysed per seed with a within-federation, budget-preserving rank remapping.

    The pooled 2,095-writer analysis answers a population-level question; this one
    answers the deployment-level question, on the federation the campaign really
    runs. Every one of the 30 seeds is evaluated, because the 90/10 split (and so
    each writer's class counts) changes with the seed.
    """
    fem = manifest[(manifest.dataset == "FEMNIST") & (manifest.method == "FedAvg")]
    per_writer, per_seed = [], []
    for seed, grp in fem.groupby("seed"):
        grp = grp.sort_values("client_id")
        counts = [parse_counts(s) for s in grp["label_counts"]]
        cv = np.array([cv_norm(c, n_classes) for c in counts])
        js = np.array([js_divergence(_probs(c), prior) for c in counts])
        e_cv = np.array([epochs_from_h(h) for h in cv])
        e_js = within_federation_remap(e_cv, js)
        rho = spearman(cv, js)
        per_seed.append({
            "seed": int(seed), "n_clients": len(grp),
            "total_epochs_cv": int(e_cv.sum()), "total_epochs_js": int(e_js.sum()),
            "budget_preserved": bool(e_cv.sum() == e_js.sum()),
            "n_writers_changed": int((e_cv != e_js).sum()),
            "spearman_cv_vs_js": rho,
        })
        for i, (_, r) in enumerate(grp.iterrows()):
            per_writer.append({
                "seed": int(seed), "client_id": int(r["client_id"]),
                "writer_natural_id": r["writer_natural_id"],
                "n_k": int(r["n_k"]), "cv_norm": cv[i], "js_global": js[i],
                "epochs_cv": int(e_cv[i]), "epochs_js_remapped": int(e_js[i]),
                "changed": bool(e_cv[i] != e_js[i]),
            })
    return pd.DataFrame(per_writer), pd.DataFrame(per_seed)


def normalization_sensitivity(fem, n_classes=62):
    """
    Item (8): how the pooled headline depends on how JS is put on the CV's scale.
    Only quantile matching preserves the aggregate budget; the other two inflate
    it, which conflates re-ranking with simply training more.
    """
    cv = fem["cv_norm"].to_numpy()
    js = fem["js_global"].to_numpy()
    e_cv = np.array([epochs_from_h(h) for h in cv])
    variants = {
        "quantile_matching": np.sort(cv)[np.argsort(np.argsort(js))],
        "min_max": ((js - js.min()) / (js.max() - js.min())
                    if js.max() > js.min() else np.zeros_like(js)),
        "raw_js": js,
    }
    rows = []
    for name, mapped in variants.items():
        e = np.array([epochs_from_h(h) for h in mapped])
        rows.append({
            "normalisation": name,
            "pct_writers_changed": 100 * float((e != e_cv).mean()),
            "total_epochs_cv": int(e_cv.sum()), "total_epochs_variant": int(e.sum()),
            "budget_preserved": bool(e.sum() == e_cv.sum()),
        })
    return pd.DataFrame(rows)


def risk_register():
    """Item (14): every methodological risk, explicitly classified."""
    rows = [
        ("Leakage (predictive)", "resolved",
         "No held-out target is predicted from itself. The CIFAR-10 global prior is "
         "definitional (the training set the partitions are drawn from), not an "
         "inferential estimate; the FEMNIST prior is pooled over 2,095 writers, of "
         "which the 10 campaign clients are 0.48%."),
        ("Circularity (FedHAD)", "residual",
         "In FedHAD H_k drives E_k and the LR, so H_k -> E_k/LR -> update -> cos_sim "
         "is a causal path absent from the other methods. The FedHAD correlation is "
         "therefore not an independent validation. FedAvg/FedAvgM/FedProx are the "
         "clean estimates and are reported first."),
        ("Scale dependence of the pooled headline", "resolved",
         "Quantile matching is the only normalisation that preserves the aggregate "
         "budget; min-max and raw JS inflate it. Sensitivity of all three is reported "
         "so the figure is never read as scale-free."),
        ("Budget invariance claimed per scenario", "resolved",
         "The pooled quantile mapping preserves the budget globally, NOT within any "
         "subset. The campaign federation is therefore analysed separately, with a "
         "remapping that preserves the budget exactly inside the 10 clients."),
        ("Scope: pool vs campaign federation", "resolved",
         "The 2,095 writers are an aggregated pool, not the campaign's federation of "
         "10. Both are now reported, labelled, and never conflated."),
        ("Mixed basis in the FEMNIST pool", "documentation",
         "Client writers enter with their 90% training split, held-out writers with "
         "100% of their data (10 of 2,095 rows). Negligible for the prior; documented "
         "because the 10 client rows are not strictly comparable to the other 2,085."),
        ("Guard passing on sentinel values", "documentation",
         "3 of the 150 clients are M2 (n_k=0) and compare 0.0 against 0.0. The "
         "informative sample size of the guard is 147, not 150."),
        ("Deduplication of held-out writers", "resolved",
         "Verified: a held-out writer's class counts are identical across all 30 "
         "scenarios (0 divergences), so keeping the first occurrence is lossless."),
        ("SEED as a default argument", "documentation",
         "`dirichlet_split_noniid(..., seed=SEED)` binds SEED at import time and "
         "`load_data()` never passes it, so patching module.SEED after import "
         "silently regenerates the seed-42 partition. An implementation and "
         "reproducibility note for offline reuse, not a scientific result. The "
         "campaign is unaffected: it sets FL_SEED before importing."),
        ("Overinterpretation of Part 2", "resolved",
         "FEMNIST establishes an applicability boundary for the CV under a "
         "non-uniform global prior. It does not establish that JS predicts drift or "
         "improves accuracy: there is no FEMNIST drift telemetry and no training was "
         "run with JS."),
    ]
    return pd.DataFrame(rows, columns=["risk", "status", "assessment"])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "figure.autolayout": True,
    })
    return plt


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_priors(cifar_prior, femnist_prior, plt):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, (prior, title, k) in zip(axes, [
            (cifar_prior, "CIFAR-10 (Dirichlet benchmark)", 10),
            (femnist_prior, "FEMNIST (naturally federated)", 62)]):
        ax.bar(np.arange(k), prior, color=OKABE_ITO[0], width=0.9,
               label="Empirical global prior")
        ax.axhline(1.0 / k, color=OKABE_ITO[1], ls="--", lw=1.2,
                   label="Uniform prior")
        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.legend(frameon=False, fontsize=7)
    save_fig(fig, "F1_global_prior_vs_uniform")


def fig_spearman(corr, plt):
    methods = sorted(corr["method"].unique())
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(9.6, 3.1), sharey=True)
    width = 0.8 / len(METRICS)
    for ax, alpha in zip(axes, ALPHAS):
        for j, metric in enumerate(METRICS):
            sub = corr[(corr.alpha == alpha) & (corr.metric == metric)]
            sub = sub.set_index("method").reindex(methods)
            x = np.arange(len(methods)) + j * width - 0.4 + width / 2
            err = np.vstack([
                (sub.spearman_rho - sub.ci95_low).to_numpy(),
                (sub.ci95_high - sub.spearman_rho).to_numpy()])
            ax.bar(x, sub.spearman_rho, width=width, color=OKABE_ITO[j],
                   label=METRIC_LABEL[metric] if alpha == ALPHAS[0] else None)
            ax.errorbar(x, sub.spearman_rho, yerr=np.abs(err), fmt="none",
                        ecolor="0.25", elinewidth=0.8, capsize=1.8)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_title(rf"$\alpha = {alpha:g}$")
        ax.axhline(0, color="0.3", lw=0.8)
    axes[0].set_ylabel(r"Spearman $\rho$ vs cosine similarity")
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.13))
    save_fig(fig, "F2_dirichlet_spearman_by_metric")


def fig_femnist_scatter(fem, extreme, plt):
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    clients = fem[fem.role == "client"]
    others = fem[fem.role != "client"]
    ax.scatter(others.cv_norm, others.js_global, s=9, alpha=0.45,
               color=OKABE_ITO[0], edgecolors="none", label="Held-out writers")
    ax.scatter(clients.cv_norm, clients.js_global, s=26, color=OKABE_ITO[1],
               edgecolors="black", linewidths=0.4, label="Campaign client writers")
    if extreme is not None:
        ax.scatter([extreme.cv_norm], [extreme.js_global], s=80, marker="*",
                   color=OKABE_ITO[3], edgecolors="black", linewidths=0.5,
                   zorder=5, label=r"Mirrors the prior, loses an epoch")
        ax.annotate(extreme.writer, (extreme.cv_norm, extreme.js_global),
                    textcoords="offset points", xytext=(8, -10), fontsize=7)
    ax.set_xlabel(r"Normalised CV (current $H_k$, uniform reference)")
    ax.set_ylabel("JS divergence vs empirical global prior")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save_fig(fig, "F3_femnist_cv_vs_js")


def fig_epoch_impact(fem, plt):
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    tab = (fem.groupby(["epochs_cv", "epochs_js"]).size()
              .rename("n").reset_index())
    grid = tab.pivot(index="epochs_cv", columns="epochs_js", values="n").fillna(0)
    im = ax.imshow(grid.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel(r"$E_k$ allocated using JS (global prior)")
    ax.set_ylabel(r"$E_k$ allocated using CV (uniform)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = int(grid.to_numpy()[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > grid.to_numpy().max() * 0.55 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Writers")
    save_fig(fig, "F4_femnist_epoch_allocation")


def fig_federation_remap(fed_writers, fed_seeds, plt):
    """Item (7), visualised: who gets reallocated inside the real federation."""
    freq = (fed_writers.groupby("writer_natural_id")["changed"].sum()
            .sort_values(ascending=False))
    n_seeds = len(fed_seeds)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    colors = [OKABE_ITO[1] if v > 0 else OKABE_ITO[0] for v in freq.values]
    ax.barh(range(len(freq)), freq.values, color=colors)
    ax.set_yticks(range(len(freq)), freq.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Seeds in which $E_k$ is reallocated (of {n_seeds})")
    ax.set_title("Campaign client writers", fontsize=9)

    ax = axes[1]
    ax.hist(fed_seeds.n_writers_changed, bins=np.arange(-0.5, 11.5, 1),
            color=OKABE_ITO[0], edgecolor="white")
    ax.set_xlabel("Writers reallocated per seed (of 10)")
    ax.set_ylabel("Seeds")
    ax.set_title("Budget preserved exactly in every seed", fontsize=9)
    save_fig(fig, "F5_within_federation_remap")


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def write_tex_federation(fed_writers, fed_seeds, trans, path):
    """Item (7) as a standalone supplementary table."""
    n_seeds = len(fed_seeds)
    rows = []
    for _, r in trans.iterrows():
        tag = "unchanged" if r.epochs_cv == r.epochs_js_remapped else "reallocated"
        rows.append(f"{int(r.epochs_cv)} & {int(r.epochs_js_remapped)} & "
                    f"{int(r.n_writer_seed)} & {tag} \\\\")
    changed_pairs = int(fed_writers.changed.sum())
    tex = f"""% Generated by analyze_prior_assumption.py -- do not edit by hand.
% Requires \\usepackage{{booktabs}}.
\\begin{{table}}[t]
\\centering
\\caption{{Counterfactual, budget-preserving rank remapping within the campaign's
FEMNIST federation. The exact multiset of local-epoch budgets assigned by the
normalised CV is preserved and redistributed according to each writer's
Jensen--Shannon divergence from the empirical global prior, over all {n_seeds}
seeds. This is a counterfactual reallocation, not an algorithm: nothing was
trained, and no claim is made about drift or accuracy.}}
\\label{{tab:federation-remap}}
\\begin{{tabular}}{{rrrl}}
\\toprule
$E_k$ (CV) & $E_k$ (remapped) & (writer, seed) pairs & Outcome \\\\
\\midrule
{chr(10).join(rows)}
\\midrule
\\multicolumn{{2}}{{l}}{{Total}} & {len(fed_writers)} & {changed_pairs} reallocated \\\\
\\bottomrule
\\end{{tabular}}

\\vspace{{2pt}}
{{\\footnotesize\\raggedright
\\textit{{Note.}} The federation has 10 client writers. The total local-epoch budget
is identical before and after the remapping in every one of the {n_seeds} seeds
({sorted(set(fed_seeds.total_epochs_cv))[0]} epochs), so the only thing that changes is which writer
holds which budget. On average {fed_seeds.n_writers_changed.mean():.1f} of the 10 writers are
reallocated per seed (min {int(fed_seeds.n_writers_changed.min())}, max {int(fed_seeds.n_writers_changed.max())}). The rank correlation between
the two heterogeneity measures inside the federation is
{fed_seeds.spearman_cv_vs_js.mean():.3f} on average
(range {fed_seeds.spearman_cv_vs_js.min():.3f}--{fed_seeds.spearman_cv_vs_js.max():.3f}), i.e. lower than over the
aggregated writer pool, so the disagreement is if anything sharper on the
federation that was actually run.
\\par}}
\\end{{table}}
"""
    path.write_text(tex, encoding="utf-8")


def write_tex(corr, fem_summary, guard_max, guard_pass, cifar_dev, femnist_dev, path):
    best = (corr.assign(absrho=corr.spearman_rho.abs())
                .sort_values("absrho", ascending=False)
                .groupby(["method", "alpha"], as_index=False).first())
    lines = []
    for alpha in ALPHAS:
        for method in sorted(corr["method"].unique()):
            sub = corr[(corr.alpha == alpha) & (corr.method == method)]
            cv = sub[sub.metric == "cv_norm"].iloc[0]
            b = best[(best.alpha == alpha) & (best.method == method)].iloc[0]
            lines.append(
                f"{method} & {alpha:g} & {int(cv.n_obs)} & "
                f"{cv.spearman_rho:.3f} & [{cv.ci95_low:.3f}, {cv.ci95_high:.3f}] & "
                f"\\texttt{{{b.metric.replace('_', chr(92) + '_')}}} & {b.spearman_rho:.3f} \\\\")

    tex = f"""% Generated by analyze_prior_assumption.py -- do not edit by hand.
% Requires \\usepackage{{booktabs}}.
\\begin{{table}}[t]
\\centering
\\caption{{Testing the uniform-global-prior assumption embedded in $H_k$.
Part 1 (CIFAR-10, Dirichlet): Spearman correlation between each heterogeneity
metric and the logged cosine similarity, with 95\\% bootstrap intervals resampled
over the ten seeds. ``Best metric'' is the one with the largest $|\\rho|$ for that
row. Part 2 (FEMNIST) is reported in the text, since no drift telemetry exists
for that dataset.}}
\\label{{tab:prior-assumption}}
\\begin{{tabular}}{{llrrlrr}}
\\toprule
Method & $\\alpha$ & $n$ & \\multicolumn{{2}}{{c}}{{Normalised CV (current $H_k$)}} & \\multicolumn{{2}}{{c}}{{Best metric}} \\\\
\\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}}
 & & & $\\rho$ & 95\\% CI & metric & $\\rho$ \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}

\\vspace{{2pt}}
{{\\footnotesize\\raggedright
\\textit{{Note.}} $H_k$ is a normalised coefficient of variation, which measures
departure from a \\emph{{uniform}} class prior. The empirical global prior of
CIFAR-10 is exactly uniform (5{{,}}000 images per class; maximum deviation
{cifar_dev:.1e}), so on this benchmark the assumption is satisfied by construction
and the JS divergences against the empirical and the uniform prior coincide.
FEMNIST, being naturally federated, departs from uniform by up to
{femnist_dev:.4f} in probability, which is the regime in which the assumption fails.
CIFAR-10 therefore characterises the regime in which the uniform hypothesis holds;
it cannot test the failure mode of a non-uniform prior. The FEMNIST results establish an
applicability boundary for the CV under a non-uniform global prior; they do NOT
establish that JS predicts drift or improves accuracy, since no FEMNIST drift
telemetry exists and no model was trained with JS.
All metrics are oriented so that larger values mean more heterogeneous.
Three (alpha, seed, client) pairs with $n_k=0$ (documented as M2) are excluded.
Validation guard: the recomputed CV reproduces the logged $H_k$ to
{guard_max:.1e} ({'passed' if guard_pass else 'FAILED'} at the 1e-6 tolerance).
\\par}}
\\end{{table}}
"""
    path.write_text(tex, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="Bootstrap resamples for the confidence intervals.")
    ap.add_argument("--skip-regen", action="store_true",
                    help="Use only the manifest (full-partition) basis; the 1e-6 "
                         "guard cannot pass in this mode, by construction.")
    args = ap.parse_args()

    for d in (TABLES, FIGURES, DATA):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260906)
    plt = setup_mpl()

    print("=" * 78)
    print("PRIOR-ASSUMPTION ANALYSIS (offline; no training is executed)")
    print("=" * 78)
    manifest, heldout, drift = load_sources()
    print(f"Sources read (read-only):\n  {MANIFEST_CSV.name}  {manifest.shape}"
          f"\n  {HELDOUT_CSV.name}  {heldout.shape}\n  {DRIFT_CSV.name}  {drift.shape}")

    # ---------------------------- Part 1 ----------------------------------- #
    print("\n" + "-" * 78)
    print("PART 1 - Dirichlet / CIFAR-10")
    print("-" * 78)

    part_counts = manifest_partition_counts(manifest)
    drift_keys = drift[["alpha", "seed", "client_id"]].drop_duplicates()
    joined = drift_keys.merge(part_counts, on=["alpha", "seed", "client_id"],
                              how="left", indicator=True)
    match_rate = float((joined["_merge"] == "both").mean())
    print(f"Join (dataset, alpha, seed, client_id): "
          f"{(joined['_merge'] == 'both').sum()}/{len(joined)} = {100 * match_rate:.1f}%")
    if match_rate < 0.999:
        print("ABORTING: join match rate is not ~100%; the two sources do not "
              "describe the same clients.")
        return 2

    if args.skip_regen:
        counts_df = part_counts.rename(columns={
            "n_k_partition": "n_k_train", "label_counts_partition": "label_counts"})
        counts_df = counts_df[counts_df.alpha.isin(ALPHAS)]
        basis = "manifest full client partition"
    else:
        seeds = sorted(drift["seed"].unique())
        print(f"Regenerating exact training-split counts for {len(seeds)} seeds x "
              f"{len(ALPHAS)} alphas (partitioning only, no training)...", flush=True)
        counts_df = regenerate_train_split_counts(seeds, ALPHAS)
        basis = "regenerated 90% training split (matches the telemetry)"
    print(f"Metric basis: {basis}")

    guard_df, guard_max, guard_pass = run_guard(counts_df, drift)
    # Items (1) and (2): a formal statement of what the guard establishes.
    guard_df["is_m2"] = [(round(a, 4), int(s), int(c)) in M2_EXCLUSIONS
                         for a, s, c in zip(guard_df.alpha, guard_df.seed,
                                            guard_df.client_id)]
    guard_df["cv_float32"] = [cv_norm_float32(parse_counts(s), 10)
                              for s in guard_df["label_counts"]]
    guard_df["rel_diff"] = (guard_df["abs_diff"] /
                            guard_df["H_k"].abs().replace(0, np.nan))
    informative = guard_df[~guard_df.is_m2]
    eps32 = float(np.finfo(np.float32).eps)
    rel = informative["rel_diff"].replace([np.inf, -np.inf], np.nan).dropna()
    span = (drift.groupby(["alpha", "seed", "client_id"])["H_k"]
                 .agg(lambda v: v.max() - v.min()))
    print(f"Validation guard  : max |recomputed CV - logged H_k| = {guard_max:.3e} "
          f"over {len(guard_df)} clients -> {'PASS' if guard_pass else 'FAIL'} at 1e-6")
    print(f"  informative cases : {len(informative)}/{len(guard_df)} "
          f"({int(guard_df.is_m2.sum())} M2 clients compare sentinel 0.0 to 0.0 and "
          f"are not evidence)")
    print(f"  max / median error: {informative.abs_diff.max():.3e} / "
          f"{informative.abs_diff.median():.3e}")
    print(f"  relative error max/median: {rel.max():.3e} / {rel.median():.3e}")
    print(f"  float32 machine epsilon  : {eps32:.3e}  -> the largest relative error is "
          f"{rel.max() / eps32:.2f}x eps32, and {int((rel > 2 * eps32).sum())}/{len(rel)} "
          f"cases exceed 2x eps32")
    print(f"  => the residual is float32 representation error in the logged value "
          f"(the campaign computes the counts via torch.bincount(...).float()), "
          f"not a reconstruction error")
    print(f"  H_k constant across the {drift['round'].nunique()} rounds and "
          f"{drift['metodo'].nunique()} methods: max within-client span = "
          f"{span.max():.3e} over {len(span)} clients, so this guard covers all "
          f"{len(drift)} telemetry rows")
    guard_df.to_csv(TABLES / "T1_guard_validation.csv", index=False)

    if not guard_pass:
        print("  NOTE: the guard did not pass. With --skip-regen this is expected: "
              "the manifest records the full partition while the telemetry's H_k is "
              "computed over the 90% training split.")

    # Cross-check between the two bases.
    cross = counts_df.merge(part_counts, on=["alpha", "seed", "client_id"], how="inner")
    cross["cv_basis_used"] = [cv_norm(parse_counts(s), 10) for s in cross["label_counts"]]
    cross["cv_manifest"] = [cv_norm(parse_counts(s), 10)
                            for s in cross["label_counts_partition"]]
    cross["is_m2"] = [(round(a, 4), int(s), int(c)) in M2_EXCLUSIONS
                      for a, s, c in zip(cross.alpha, cross.seed, cross.client_id)]
    keep = cross[~cross.is_m2]
    basis_rho = spearman(keep["cv_basis_used"], keep["cv_manifest"])
    basis_maxdiff = float((keep["cv_basis_used"] - keep["cv_manifest"]).abs().max())
    print(f"Basis cross-check : max |CV(train split) - CV(full partition)| = "
          f"{basis_maxdiff:.3e}; Spearman between bases = {basis_rho:.6f} "
          f"(excluding the {int(cross.is_m2.sum())} M2 clients, whose single sample "
          f"falls entirely into the validation split, giving CV=1 on one basis and "
          f"the sentinel 0 on the other)")
    cross.to_csv(TABLES / "T1b_basis_cross_check.csv", index=False)

    # Empirical global prior of CIFAR-10. Estimated from the FULL partitions (the
    # complete training set, 5,000 images per class by construction), not from the
    # 90% training split, whose subsampling noise would masquerade as prior skew.
    ref = part_counts[(part_counts.alpha == ALPHAS[0]) &
                      (part_counts.seed == part_counts.seed.min())]
    cifar_total = np.zeros(10, dtype=np.int64)
    for s in ref["label_counts_partition"]:
        cifar_total += parse_counts(s)
    cifar_prior = cifar_total / cifar_total.sum()
    cifar_dev = float(np.abs(cifar_prior - 0.1).max())
    split_total = np.zeros(10, dtype=np.int64)
    for s in counts_df[(counts_df.alpha == ALPHAS[0]) &
                       (counts_df.seed == counts_df.seed.min())]["label_counts"]:
        split_total += parse_counts(s)
    split_dev = float(np.abs(split_total / split_total.sum() - 0.1).max())
    print(f"CIFAR-10 empirical global prior: counts per class = "
          f"{sorted(set(cifar_total.tolist()))}, max deviation from uniform = {cifar_dev:.2e}"
          f"  ({'UNIFORM by construction' if cifar_dev < 1e-9 else 'non-uniform'})")
    print(f"  (the 90% training split deviates by {split_dev:.2e}, which is subsampling "
          f"noise, not prior skew)")

    metrics_df = build_dirichlet_metrics(counts_df, cifar_prior)
    metrics_df.to_csv(DATA / "dirichlet_client_metrics.csv", index=False)

    print(f"Correlating vs cos_sim ({args.bootstrap} bootstrap resamples over seeds)...",
          flush=True)
    corr, obs_all, obs_kept = correlate_dirichlet(metrics_df, drift, args.bootstrap, rng)
    corr.to_csv(TABLES / "T2_dirichlet_spearman.csv", index=False)
    print(f"  observations: {len(obs_all)} -> {len(obs_kept)} after excluding the "
          f"{len(M2_EXCLUSIONS)} M2 (n_k=0) client-seed pairs")

    # Item (10): lead with the three baselines, where H_k does not control the
    # optimisation and the correlation is therefore causally clean.
    print("\nSpearman rho vs cosine similarity - BASELINES (H_k does not steer training):")
    for method in ["FedAvg", "FedAvgM", "FedProx"]:
        print(f"  {method}")
        print(f"    {'alpha':>6} " + " ".join(f"{m:>14}" for m in METRICS))
        for alpha in ALPHAS:
            sub = corr[(corr.method == method) & (corr.alpha == alpha)].set_index("metric")
            print(f"    {alpha:>6} " + " ".join(f"{sub.loc[m, 'spearman_rho']:>14.3f}"
                                                for m in METRICS))
    print("\n  FedHAD (reported separately: H_k -> E_k/LR -> update -> cos_sim is a")
    print("  causal path, so this correlation is NOT an independent validation)")
    print(f"    {'alpha':>6} " + " ".join(f"{m:>14}" for m in METRICS))
    for alpha in ALPHAS:
        sub = corr[(corr.method == "FedHAD") & (corr.alpha == alpha)].set_index("metric")
        print(f"    {alpha:>6} " + " ".join(f"{sub.loc[m, 'spearman_rho']:>14.3f}"
                                            for m in METRICS))

    # ---------------------------- Part 2 ----------------------------------- #
    print("\n" + "-" * 78)
    print("PART 2 - FEMNIST (no drift telemetry; no correlation attempted)")
    print("-" * 78)
    writers = femnist_writer_table(manifest, heldout)
    fem_prior, fem_total = femnist_global_prior(writers)
    fem_dev = float(np.abs(fem_prior - 1 / 62).max())
    print(f"Distinct writers: {len(writers)} "
          f"({(writers.role == 'client').sum()} campaign clients, "
          f"{(writers.role == 'heldout_test').sum()} held-out) | "
          f"{int(fem_total.sum())} samples")
    print(f"Empirical global prior: min={fem_prior.min():.5f} max={fem_prior.max():.5f} "
          f"(uniform = {1 / 62:.5f}); max/min ratio = {fem_prior.max() / fem_prior.min():.1f}x")
    print(f"Max deviation from uniform: {fem_dev:.5f} | "
          f"JS(global || uniform) = {js_divergence(fem_prior, np.full(62, 1 / 62)):.5f}")

    fem = analyse_femnist(writers, fem_prior)
    fem.to_csv(DATA / "femnist_writer_metrics.csv", index=False)

    rank_rho = spearman(fem["cv_norm"].to_numpy(), fem["js_global"].to_numpy())
    n_band = int(fem["band_changed"].sum())
    n_ep = int(fem["epochs_differ"].sum())
    print(f"\nRanking disagreement CV (uniform ref) vs JS (global ref)")
    print(f"  SCOPE: the aggregated pool of {len(fem)} distinct writers - NOT the "
          f"campaign's 10-client federation, which is analysed separately below.")
    print(f"  Spearman between the two rankings          : {rank_rho:.4f}")
    print(f"  Pool writers changing H_k tercile band     : {n_band}/{len(fem)} "
          f"({100 * n_band / len(fem):.1f}%)")
    print(f"  Pool writers whose E_k would differ        : {n_ep}/{len(fem)} "
          f"({100 * n_ep / len(fem):.1f}%)  [quantile matching; budget preserved "
          f"GLOBALLY over the pool, not within any subset]")

    # Item (8): the pooled headline is contingent on the normalisation.
    sens = normalization_sensitivity(fem)
    sens.to_csv(TABLES / "T6_normalisation_sensitivity.csv", index=False)
    print("\n  Sensitivity of that pooled figure to how JS is placed on the CV scale:")
    for _, r in sens.iterrows():
        print(f"    {r.normalisation:20} {r.pct_writers_changed:5.1f}%  "
              f"budget {r.total_epochs_variant} vs {r.total_epochs_cv} "
              f"({'preserved' if r.budget_preserved else 'INFLATED'})")
    print("    Only quantile matching isolates re-ranking; the others inflate the "
          "budget, conflating reordering with simply training more.")

    # The case the critique predicts: a writer that closely mirrors the global
    # prior (low JS) and is nevertheless scored as heterogeneous by the CV.
    # Searched among the 5% most prior-like writers, taking the highest CV.
    def pct(col, value):
        return 100 * (fem[col] < value).mean()

    closest = fem.loc[fem["js_global"].idxmin()]
    prior_like = fem.nsmallest(max(1, int(0.05 * len(fem))), "js_global")
    penalised = prior_like.loc[prior_like["cv_norm"].idxmax()]
    # The operationally meaningful case: among writers whose epoch budget actually
    # changes, the one whose CV rank most overstates its true divergence.
    changed = fem[fem["epochs_differ"]].copy()
    changed["rank_gap"] = (changed["cv_norm"].rank(pct=True) -
                           changed["js_global"].rank(pct=True))
    extreme = changed.loc[changed["rank_gap"].idxmax()]

    print(f"\nMost prior-like writer overall (lowest JS): {closest.writer} "
          f"JS={closest.js_global:.4f}, CV={closest.cv_norm:.4f} "
          f"(CV percentile {pct('cv_norm', closest.cv_norm):.1f})")
    print(f"Among the 5% most prior-like writers, the one the CV penalises most: "
          f"{penalised.writer} JS={penalised.js_global:.4f} "
          f"(pct {pct('js_global', penalised.js_global):.1f}) vs "
          f"CV={penalised.cv_norm:.4f} (pct {pct('cv_norm', penalised.cv_norm):.1f})")
    print(f"Most extreme case that actually changes the epoch budget:")
    print(f"  {extreme.writer}: JS={extreme.js_global:.4f} (percentile "
          f"{pct('js_global', extreme.js_global):.1f}) but CV={extreme.cv_norm:.4f} "
          f"(percentile {pct('cv_norm', extreme.cv_norm):.1f})")
    print(f"  n_k={int(extreme.n_k)}, classes covered={int(extreme.n_classes_covered)}/62, "
          f"E_k(CV)={int(extreme.epochs_cv)} vs E_k(JS)={int(extreme.epochs_js)}")

    # ---- Item (7): the campaign's actual federation -------------------------- #
    print("\n" + "-" * 78)
    print("PART 2b - COUNTERFACTUAL RANK REMAPPING WITHIN THE CAMPAIGN FEDERATION")
    print("-" * 78)
    print("Within-federation, budget-preserving rank remapping over the 10 writers the")
    print("campaign actually uses as clients, evaluated on all 30 seeds. The exact")
    print("multiset of E_k values assigned by the CV is preserved and redistributed by")
    print("the JS ranking. This is a counterfactual reallocation, NOT an algorithm:")
    print("it is not 'FedHAD-JS', nothing was trained, and it claims nothing about")
    print("accuracy or drift.")
    fed_writers, fed_seeds = analyse_campaign_federation(manifest, fem_prior)
    fed_writers.to_csv(DATA / "campaign_federation_remap.csv", index=False)
    fed_seeds.to_csv(TABLES / "T7_campaign_federation_remap.csv", index=False)

    print(f"\n  Seeds evaluated                 : {len(fed_seeds)}")
    print(f"  Budget preserved in every seed  : {bool(fed_seeds.budget_preserved.all())}"
          f"  (total epochs before = after in {int(fed_seeds.budget_preserved.sum())}"
          f"/{len(fed_seeds)} seeds)")
    print(f"  Total epochs per seed           : "
          f"{sorted(set(fed_seeds.total_epochs_cv))} (CV) vs "
          f"{sorted(set(fed_seeds.total_epochs_js))} (remapped)")
    print(f"  Writers changing E_k per seed   : mean "
          f"{fed_seeds.n_writers_changed.mean():.1f} of 10 "
          f"(min {fed_seeds.n_writers_changed.min()}, "
          f"max {fed_seeds.n_writers_changed.max()})")
    print(f"  Spearman CV vs JS within the federation: mean "
          f"{fed_seeds.spearman_cv_vs_js.mean():.3f} "
          f"[{fed_seeds.spearman_cv_vs_js.min():.3f}, "
          f"{fed_seeds.spearman_cv_vs_js.max():.3f}]")

    trans = (fed_writers.groupby(["epochs_cv", "epochs_js_remapped"]).size()
             .rename("n_writer_seed").reset_index())
    trans.to_csv(TABLES / "T8_federation_transition_matrix.csv", index=False)
    print(f"\n  Transition matrix over {len(fed_writers)} (writer, seed) pairs:")
    for _, r in trans.iterrows():
        tag = "unchanged" if r.epochs_cv == r.epochs_js_remapped else "CHANGED"
        print(f"    E_k {int(r.epochs_cv)} -> {int(r.epochs_js_remapped)} : "
              f"{int(r.n_writer_seed):3d}  ({tag})")
    changed = fed_writers[fed_writers.changed]
    print(f"\n  Writers that change in at least one seed: "
          f"{changed.writer_natural_id.nunique()} of "
          f"{fed_writers.writer_natural_id.nunique()}")
    if len(changed):
        top = (changed.groupby("writer_natural_id").size()
               .sort_values(ascending=False).head(5))
        print("  Most frequently reallocated writers (seeds affected):")
        for w, n in top.items():
            print(f"    {w}: {n}/{len(fed_seeds)} seeds")

    risks = risk_register()
    risks.to_csv(TABLES / "T9_risk_register.csv", index=False)
    print(f"\nRisk register: {len(risks)} entries "
          f"({(risks.status == 'resolved').sum()} resolved, "
          f"{(risks.status == 'residual').sum()} residual, "
          f"{(risks.status == 'documentation').sum()} documentation-only) "
          f"-> tables/T9_risk_register.csv")

    fem_summary = pd.DataFrame([{
        "n_writers": len(fem), "spearman_cv_vs_js": rank_rho,
        "writers_band_changed": n_band, "pct_band_changed": 100 * n_band / len(fem),
        "writers_epochs_changed": n_ep, "pct_epochs_changed": 100 * n_ep / len(fem),
        "prior_max_dev_from_uniform": fem_dev,
        "prior_max_min_ratio": float(fem_prior.max() / fem_prior.min()),
        "js_global_vs_uniform": js_divergence(fem_prior, np.full(62, 1 / 62)),
        "most_prior_like_writer": closest.writer,
        "most_prior_like_js": float(closest.js_global),
        "most_prior_like_cv": float(closest.cv_norm),
        "extreme_writer": extreme.writer,
        "extreme_js": float(extreme.js_global),
        "extreme_cv": float(extreme.cv_norm),
        "extreme_epochs_cv": int(extreme.epochs_cv),
        "extreme_epochs_js": int(extreme.epochs_js),
    }])
    fem_summary.to_csv(TABLES / "T4_femnist_ranking_disagreement.csv", index=False)
    pd.DataFrame({"class": np.arange(62), "count": fem_total,
                  "empirical_prior": fem_prior,
                  "uniform_prior": 1 / 62}).to_csv(
        TABLES / "T3_femnist_prior.csv", index=False)
    (fem.groupby(["epochs_cv", "epochs_js"]).size().rename("n_writers")
        .reset_index().to_csv(TABLES / "T5_femnist_epoch_impact.csv", index=False))

    # ---------------------------- outputs ---------------------------------- #
    print("\nWriting figures and LaTeX...", flush=True)
    fig_priors(cifar_prior, fem_prior, plt)
    fig_spearman(corr, plt)
    fig_femnist_scatter(fem, extreme, plt)
    fig_epoch_impact(fem, plt)
    fig_federation_remap(fed_writers, fed_seeds, plt)
    write_tex_federation(fed_writers, fed_seeds, trans,
                         TABLES / "federation_remap.tex")
    write_tex(corr, fem_summary, guard_max, guard_pass, cifar_dev, fem_dev,
              TABLES / "prior_analysis.tex")

    print("\nFiles written under", OUT_ROOT)
    for p in sorted(list(TABLES.iterdir()) + list(FIGURES.iterdir()) + list(DATA.iterdir())):
        print(f"  {p.relative_to(OUT_ROOT)}  ({p.stat().st_size:,} bytes)")
    return 0 if guard_pass or args.skip_regen else 1


if __name__ == "__main__":
    sys.exit(main())
