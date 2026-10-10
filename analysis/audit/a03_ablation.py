"""A03 - Controlled ablation: permuted allocation, budgets, FedHAD vs LR-only
(Section 6.10, Tables 14 and 15).

Ground truth = the per-round, per-client telemetry CSVs of component_analysis_ablation/
raw/official (what each client actually executed), cross-checked with the
fingerprints. Nothing is re-trained.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import REV, OUT

RAW = REV / "component_analysis_ablation" / "raw" / "official"
ARMS = ["full_fedhad", "fixed_matched", "permuted_allocation", "epoch_only", "lr_only_matched"]
BATCH = 32
RNG = np.random.default_rng(20260929)
NBOOT = 10000
TIE_PP = 0.10  # |diff| below this (pp) is reported as a practical tie (descriptive only)


def load():
    tel, fps = [], {}
    for a in ("0.01", "0.1"):
        for s in range(42, 72):
            for arm in ARMS:
                t = pd.read_csv(RAW / f"telemetry__{arm}__CIFAR10__alpha-{a}__seed-{s}.csv")
                tel.append(t)
                fps[(arm, float(a), s)] = json.load(open(RAW / f"fingerprint__{arm}__CIFAR10__alpha-{a}__seed-{s}.json"))
    return pd.concat(tel, ignore_index=True), fps


def boot_ci(x):
    x = np.asarray(x)
    idx = RNG.integers(0, len(x), (NBOOT, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def paired(a, b, label):
    d = 100 * (np.asarray(a) - np.asarray(b))
    lo, hi = boot_ci(d)
    w = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    return {"contrast": label, "n": len(d), "mean_pp": d.mean(), "median_pp": np.median(d),
            "boot95_lo": lo, "boot95_hi": hi, "wilcoxon_W": w.statistic, "p_two_sided": w.pvalue,
            "dz": d.mean() / d.std(ddof=1),
            "wins_first": int((d > TIE_PP).sum()), "wins_second": int((d < -TIE_PP).sum()),
            "ties_|d|<=0.1pp": int((np.abs(d) <= TIE_PP).sum()), "exact_zero": int((d == 0).sum())}


def main():
    tel, fps = load()
    # ---- per client x round integrity: executed quantities ----
    # Examples actually processed: training loaders use drop_last=(n_train > 1), so every update
    # processes a full batch of BATCH examples, except for a client with a single training sample,
    # whose only (incomplete) batch is kept and processes one example per update.
    tel["processed_examples"] = tel.minibatch_updates * np.minimum(BATCH, tel.n_k)
    tel["n_batches_implied"] = tel.minibatch_updates / tel.epochs
    tel["flops_full_per_sample"] = tel.flops_locais / (tel.n_k * tel.epochs)
    per_run = tel.groupby(["variant", "alpha", "seed"]).agg(
        updates=("minibatch_updates", "sum"), processed=("processed_examples", "sum"),
        flops_nominal=("flops_locais", "sum"), epochs_sum_per_round=("epochs", lambda s: s.sum() / 10),
        acc_final=("acc_centralized_round", "last")).reset_index()
    ff = tel.flops_full_per_sample.median()
    per_run["flops_executed"] = per_run.processed * ff
    ref = per_run[per_run.variant == "full_fedhad"].set_index(["alpha", "seed"])
    for c in ["updates", "processed", "flops_nominal", "flops_executed"]:
        per_run[f"{c}_pct_vs_full"] = per_run.apply(
            lambda r: 100 * (r[c] / ref.loc[(r.alpha, r.seed), c] - 1), axis=1)
    per_run.to_csv(OUT / "a03_per_run_budget.csv", index=False)

    # final accuracy: use the fingerprint-independent final_metrics table too
    fm = pd.read_csv(REV / "component_analysis_ablation" / "tables" / "final_metrics_official.csv")
    chk = per_run.merge(fm, left_on=["variant", "alpha", "seed"], right_on=["variant", "alpha", "seed"])
    acc_mismatch = int((abs(chk.acc_final_x - chk.acc_final_y) > 1e-12).sum())

    # ---- per client: permutation detail, LR attachment ----
    rows = []
    lr_follow = {"stays_with_client": 0, "moves_with_epochs": 0, "cells": 0}
    for a in (0.01, 0.1):
        for s in range(42, 72):
            fpf = fps[("full_fedhad", a, s)]
            fpp = fps[("permuted_allocation", a, s)]
            tp = tel[(tel.variant == "permuted_allocation") & (tel.alpha == a) & (tel.seed == s) & (tel["round"] == 1)].set_index("client_id")
            tf = tel[(tel.variant == "full_fedhad") & (tel.alpha == a) & (tel.seed == s) & (tel["round"] == 1)].set_index("client_id")
            lr_follow["cells"] += 1
            # LR in perm == LR_full of same client?
            same_client = all(abs(tp.loc[k, "learning_rate"] - tf.loc[k, "learning_rate"]) < 1e-15 for k in tf.index)
            lr_follow["stays_with_client"] += int(same_client)
            for k in sorted(tf.index):
                nk, nb = int(tf.loc[k, "n_k"]), int(tf.loc[k, "n_batches_implied"])
                ef, ep = int(tf.loc[k, "epochs"]), int(tp.loc[k, "epochs"])
                rows.append({
                    "alpha": a, "seed": s, "client": k, "H": tf.loc[k, "H_k"], "n_k": nk, "batches": nb,
                    "E_full": ef, "E_perm": ep, "E_fixed": fpf["E_fixed"][k],
                    "LR_full": tf.loc[k, "learning_rate"], "LR_perm": tp.loc[k, "learning_rate"],
                    "LR_if_it_followed_epochs": None,
                    "steps_full": ef * nb, "steps_perm": ep * nb,
                    "examples_full": ef * nb * BATCH, "examples_perm": ep * nb * BATCH,
                    "flops_nominal_full": ff * nk * ef, "flops_nominal_perm": ff * nk * ep,
                    "lr_x_steps_full": tf.loc[k, "learning_rate"] * ef * nb,
                    "lr_x_steps_perm": tp.loc[k, "learning_rate"] * ep * nb,
                })
    pc = pd.DataFrame(rows)
    # which LR the epochs "came from": the client that held E_perm[k] under E_full is ambiguous with
    # repeated values; report the LR attached to epochs only as the multiset-free statement below.
    pc.to_csv(OUT / "a03_permutation_per_client.csv", index=False)

    pcs = pc.groupby(["alpha", "seed"]).agg(
        steps_full=("steps_full", "sum"), steps_perm=("steps_perm", "sum"),
        ex_full=("examples_full", "sum"), ex_perm=("examples_perm", "sum"),
        fl_full=("flops_nominal_full", "sum"), fl_perm=("flops_nominal_perm", "sum"),
        lrs_full=("lr_x_steps_full", "sum"), lrs_perm=("lr_x_steps_perm", "sum"),
        clients_changed=("E_full", lambda s: 0)).reset_index()
    pcs["clients_changed"] = pc.assign(c=pc.E_full != pc.E_perm).groupby(["alpha", "seed"]).c.sum().values
    for c in ("steps", "ex", "fl", "lrs"):
        pcs[f"{c}_mismatch_pct"] = 100 * (pcs[f"{c}_perm"] / pcs[f"{c}_full"] - 1)
    pcs.to_csv(OUT / "a03_permutation_per_seed.csv", index=False)

    # ---- contrasts ----
    acc = per_run.pivot_table(index=["alpha", "seed"], columns="variant", values="acc_final")
    contrasts = []
    pairs = [("full_fedhad", "fixed_matched", "Q1"), ("full_fedhad", "permuted_allocation", "Q2"),
             ("epoch_only", "fixed_matched", "Q3"), ("lr_only_matched", "fixed_matched", "Q4"),
             ("full_fedhad", "epoch_only", "Q5a"), ("full_fedhad", "lr_only_matched", "Q5b")]
    for a in (0.01, 0.1):
        A = acc.loc[a]
        for x, y, q in pairs:
            r = paired(A[x], A[y], f"{x} - {y}")
            r.update({"alpha": a, "Q": q})
            contrasts.append(r)
    ct = pd.DataFrame(contrasts)
    ct.to_csv(OUT / "a03_contrasts_recomputed.csv", index=False)

    # Q2 adjusted for the budget covariate (descriptive): regress diff on steps mismatch
    q2 = []
    for a in (0.01, 0.1):
        A = acc.loc[a]
        d = 100 * (A.full_fedhad - A.permuted_allocation)
        m = pcs[pcs.alpha == a].set_index("seed").steps_mismatch_pct.reindex(d.index)
        b = np.polyfit(m.values, d.values, 1)
        sub = d[m.abs() <= 1.0]
        q2.append({"alpha": a, "slope_pp_per_pct_steps": b[0], "intercept_pp_at_zero_mismatch": b[1],
                   "n_|mismatch|<=1%": len(sub), "mean_diff_subset": sub.mean() if len(sub) else np.nan,
                   "wilcoxon_p_subset": wilcoxon(sub).pvalue if len(sub) >= 6 else np.nan,
                   "perm_fewer_steps_cells": int((m < 0).sum())})
    pd.DataFrame(q2).to_csv(OUT / "a03_q2_budget_adjustment.csv", index=False)

    # ---- FedHAD vs LR-only, seed by seed ----
    seedrows = []
    for a in (0.01, 0.1):
        for s in range(42, 72):
            f, l = fps[("full_fedhad", a, s)], fps[("lr_only_matched", a, s)]
            pf = per_run[(per_run.variant == "full_fedhad") & (per_run.alpha == a) & (per_run.seed == s)].iloc[0]
            pl = per_run[(per_run.variant == "lr_only_matched") & (per_run.alpha == a) & (per_run.seed == s)].iloc[0]
            seedrows.append({
                "alpha": a, "seed": s, "acc_full": pf.acc_final, "acc_lr_only": pl.acc_final,
                "diff_pp": 100 * (pf.acc_final - pl.acc_final),
                "E_full": ";".join(map(str, f["E_full"])), "E_lr_only(=E_fixed)": ";".join(map(str, f["E_fixed"])),
                "LR_both": ";".join(f"{x:.5f}" for x in f["LR_full"]),
                "updates_full": pf.updates, "updates_lr_only": pl.updates,
                "examples_full": pf.processed, "examples_lr_only": pl.processed,
                "TF_nominal_full": pf.flops_nominal / 1e12, "TF_nominal_lr_only": pl.flops_nominal / 1e12,
                "TF_exec_full": pf.flops_executed / 1e12, "TF_exec_lr_only": pl.flops_executed / 1e12,
                "updates_mismatch_pct": 100 * (pl.updates / pf.updates - 1),
            })
    sr = pd.DataFrame(seedrows)
    sr.to_csv(OUT / "a03_fedhad_vs_lronly_per_seed.csv", index=False)

    # Arms in absolute terms
    absr = per_run.groupby(["alpha", "variant"]).agg(
        acc_mean=("acc_final", "mean"), acc_sd=("acc_final", "std"),
        updates=("updates", "mean"), processed=("processed", "mean"),
        TF_nominal=("flops_nominal", lambda s: s.mean() / 1e12),
        TF_exec=("flops_executed", lambda s: s.mean() / 1e12),
        upd_mis_med=("updates_pct_vs_full", "median"),
        upd_mis_min=("updates_pct_vs_full", "min"), upd_mis_max=("updates_pct_vs_full", "max"),
        upd_absmis_med=("updates_pct_vs_full", lambda s: s.abs().median()),
        within1=("updates_pct_vs_full", lambda s: int((s.abs() <= 1).sum()))).reset_index()
    absr.to_csv(OUT / "a03_arms_absolute.csv", index=False)

    # reference: 5-epoch fixed baseline at identical data partition (main campaign; different init protocol)
    # budget only (the accuracies are not paired because initialisation protocol differs)
    base5 = []
    for a in (0.01, 0.1):
        for s in range(42, 72):
            f = fps[("full_fedhad", a, s)]
            u5 = 5 * sum(f["n_batches"]) * 10
            base5.append({"alpha": a, "seed": s, "updates_5epoch": u5})
    b5 = pd.DataFrame(base5).groupby("alpha").updates_5epoch.mean()

    with open(OUT / "a03_summary.txt", "w") as fh:
        fh.write(f"telemetry rows={len(tel)}; runs={per_run.shape[0]}; acc mismatches vs final_metrics table={acc_mismatch}\n")
        fh.write(f"flops per sample (full) from telemetry: {ff:.0f}\n")
        fh.write(f"LR under permutation stays with the ORIGINAL client in {lr_follow['stays_with_client']}/{lr_follow['cells']} cells\n\n")
        fh.write(absr.to_string(index=False) + "\n\n")
        fh.write("5-epoch fixed budget (updates) at same partitions:\n" + b5.to_string() + "\n\n")
        fh.write(ct.to_string(index=False) + "\n\n")
        fh.write(pcs.groupby("alpha")[["steps_mismatch_pct", "ex_mismatch_pct", "fl_mismatch_pct", "lrs_mismatch_pct", "clients_changed"]].describe().T.to_string() + "\n\n")
        fh.write(pd.DataFrame(q2).to_string(index=False) + "\n\n")
        fh.write(sr.to_string(index=False) + "\n")
    print(open(OUT / "a03_summary.txt").read()[:9000])


if __name__ == "__main__":
    main()
