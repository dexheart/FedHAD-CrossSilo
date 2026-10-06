#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidation of the step-preserving permutation and of the tuning (validation) phase
of the tuned FedProx controls.

Reads results/step_preserving_permutation/manifest.csv and
results/tuned_controls_fedprox_validation/tune/validation_summary.csv and writes, to
results/step_preserving_permutation/analysis/:

  per_arm.csv        absolute accuracy and executed budget per (alpha, arm)
  per_seed.csv       paired per-seed records with budget mismatch
  contrasts.csv      Full - step_permuted_lrclient, paired over seeds (Holm over alphas)
  displacement.csv   per seed: sum_k p_k tau_k and sum_k p_k eta_k tau_k of each arm
                     (p_k = n_k / N, the aggregation weight), and Spearman(H_k, n_k)
  tune_table.csv     FedHAD vs every static FedProx grid point on validation seeds
  report.md          the numbers above as text

Usage:  python analysis/tuned_controls_and_step_permutation/analyze_step_permutation_and_tuning.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
STEP = REPO / "results" / "step_preserving_permutation"
TUNE = REPO / "results" / "tuned_controls_fedprox_validation" / "tune"
OUT = STEP / "analysis"
N_BOOT, BOOT_SEED = 10_000, 20260930


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def boot_ci(d, level):
    rng = np.random.default_rng(BOOT_SEED)
    means = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
    lo = (1 - level) / 2
    return float(np.quantile(means, lo)), float(np.quantile(means, 1 - lo))


def load_step():
    m = pd.read_csv(STEP / "manifest.csv")
    # FedHAD runs of seeds outside this experiment (e.g. 77-106 of the experiment-2
    # evaluation) share the directory; keep only the seeds with the permuted arm
    perm = m.loc[m["arm"] == "step_permuted_lrclient", ["alpha", "seed"]].drop_duplicates()
    m = m.merge(perm, on=["alpha", "seed"])
    bad = m[m["status"] != "completed"]
    if len(bad):
        sys.exit(f"ERROR: {len(bad)} step_permutation runs not completed")
    if m["init_sha256"].isna().any():
        sys.exit("ERROR: run without init checkpoint checksum")
    m["acc_pp"] = 100 * m["acc_centralized"]
    m["val_pp"] = 100 * m["acc_distributed_val"]
    m["tflops_nominal"] = m["flops_nominal"] / 1e12
    m["tflops_executed"] = m["flops_executed"] / 1e12
    return m


def paired(m):
    a = m[m["arm"] == "full_fedhad"].set_index(["alpha", "seed"])
    b = m[m["arm"] == "step_permuted_lrclient"].set_index(["alpha", "seed"])
    if not a.index.equals(b.index.sort_values()) and set(a.index) != set(b.index):
        sys.exit("ERROR: unpaired seeds between arms")
    b = b.loc[a.index]
    if (a["init_sha256"] != b["init_sha256"]).any():
        sys.exit("ERROR: paired runs do not share the initial checkpoint")
    ps = pd.DataFrame({
        "acc_full_pp": a["acc_pp"], "acc_perm_pp": b["acc_pp"],
        "diff_acc_pp": a["acc_pp"] - b["acc_pp"],
        "diff_val_pp": a["val_pp"] - b["val_pp"],
        "steps_full": a["total_steps"], "steps_perm": b["total_steps"],
        "steps_mismatch_pct": 100 * (b["total_steps"] - a["total_steps"]) / a["total_steps"],
        "examples_mismatch_pct": 100 * (b["total_examples"] - a["total_examples"]) / a["total_examples"],
        "tflops_exec_mismatch_pct": 100 * (b["flops_executed"] - a["flops_executed"]) / a["flops_executed"],
        "tflops_nominal_mismatch_pct": 100 * (b["flops_nominal"] - a["flops_nominal"]) / a["flops_nominal"],
        "perm_mapping": b["perm_mapping_receiver_to_donor"], "init_sha256": a["init_sha256"],
    }).reset_index()
    return ps


def contrasts(ps):
    rows = []
    for alpha, g in ps.groupby("alpha"):
        d = g["diff_acc_pp"].to_numpy()
        w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        t = stats.ttest_1samp(d, 0.0)
        tci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        rows.append({
            "alpha": alpha, "n_seeds": len(d), "contrast": "Full - StepPerm(lr stays with client)",
            "mean_diff_pp": d.mean(), "median_diff_pp": float(np.median(d)), "sd_diff_pp": d.std(ddof=1),
            "ci95_t_lo": tci[0], "ci95_t_hi": tci[1],
            "ci95_boot_lo": boot_ci(d, .95)[0], "ci95_boot_hi": boot_ci(d, .95)[1],
            "ci90_boot_lo": boot_ci(d, .90)[0], "ci90_boot_hi": boot_ci(d, .90)[1],
            "dz": d.mean() / d.std(ddof=1),
            "wins_full": int((d > 0).sum()), "wins_perm": int((d < 0).sum()), "ties": int((d == 0).sum()),
            "p_wilcoxon": w.pvalue, "p_ttest": t.pvalue,
            "mean_diff_val_pp": g["diff_val_pp"].mean(),
            "max_abs_steps_mismatch_pct": g["steps_mismatch_pct"].abs().max(),
            "max_abs_examples_mismatch_pct": g["examples_mismatch_pct"].abs().max(),
            "max_abs_tflops_exec_mismatch_pct": g["tflops_exec_mismatch_pct"].abs().max(),
            "mean_tflops_nominal_mismatch_pct": g["tflops_nominal_mismatch_pct"].mean(),
            "max_abs_tflops_nominal_mismatch_pct": g["tflops_nominal_mismatch_pct"].abs().max(),
        })
    c = pd.DataFrame(rows)
    c["p_wilcoxon_holm"] = holm(c["p_wilcoxon"])
    return c


def displacement(ps):
    """Aggregation-weighted work per arm. The permutation preserves sum_k tau_k exactly
    but not sum_k p_k tau_k, which is what reaches the aggregate under FedAvg weights."""
    rows = []
    for f in sorted((STEP / "raw").glob("*/telemetry.csv")):
        t = pd.read_csv(f)
        t = t[t["round"] == t["round"].min()]          # the plan is fixed across rounds
        p = t["n_k"] / t["n_k"].sum()
        rows.append({"arm": t["arm"].iloc[0], "alpha": float(t["alpha"].iloc[0]), "seed": int(t["seed"].iloc[0]),
                     "sum_p_tau": float((p * t["steps_executed"]).sum()),
                     "sum_p_eta_tau": float((p * t["learning_rate"] * t["steps_executed"]).sum()),
                     "spearman_H_n": float(stats.spearmanr(t["H_k"], t["n_k"])[0])})
    d = pd.DataFrame(rows).merge(ps[["alpha", "seed"]], on=["alpha", "seed"])
    a = d[d["arm"] == "full_fedhad"].set_index(["alpha", "seed"])
    b = d[d["arm"] == "step_permuted_lrclient"].set_index(["alpha", "seed"]).loc[a.index]
    out = pd.DataFrame({"ratio_sum_p_tau": b["sum_p_tau"] / a["sum_p_tau"],
                        "ratio_sum_p_eta_tau": b["sum_p_eta_tau"] / a["sum_p_eta_tau"],
                        "spearman_H_n_full": a["spearman_H_n"]}).reset_index()
    return out.merge(ps[["alpha", "seed", "diff_acc_pp"]], on=["alpha", "seed"])


def per_arm(m):
    return (m.groupby(["alpha", "arm"])
             .agg(n=("seed", "size"), acc_mean_pp=("acc_pp", "mean"), acc_sd_pp=("acc_pp", "std"),
                  val_mean_pp=("val_pp", "mean"),
                  steps_mean=("total_steps", "mean"), examples_mean=("total_examples", "mean"),
                  tflops_nominal_mean=("tflops_nominal", "mean"),
                  tflops_executed_mean=("tflops_executed", "mean"))
             .reset_index())


def tune_table():
    v = pd.read_csv(TUNE / "validation_summary.csv")
    v["val_pp"] = 100 * v["val_acc_mean"]
    v["test_pp_NOT_FOR_SELECTION"] = 100 * v["test_acc_mean_NOT_FOR_SELECTION"]
    ref = v[v["arm"] == "full_fedhad"].set_index("alpha")
    v["steps_vs_fedhad"] = v.apply(lambda r: r["steps_mean"] / ref.loc[r["alpha"], "steps_mean"], axis=1)
    v["val_minus_fedhad_pp"] = v.apply(lambda r: r["val_pp"] - ref.loc[r["alpha"], "val_pp"], axis=1)
    prop = json.loads((TUNE / "selection_proposal.json").read_text())
    return v, prop


def fmt(x, n=2):
    return f"{x:+.{n}f}"


def report(pa, c, ps, v, prop, disp):
    L = ["# Step-preserving permutation and tuning phase — consolidated results", "",
         "Substrate: CIFAR-10, 5 clients, 10 rounds, FedProx mu=0.01 inside FedHAD arms, common "
         "initial checkpoint per seed (SHA-256 recorded), seeds 42-71.", "",
         "## Experiment 1 — step-preserving permuted allocation", "",
         "`step_permuted_lrclient`: the executed optimizer-step budgets tau_k of full FedHAD are "
         "permuted among clients (derangement); each client KEEPS its own FedHAD learning rate.", "",
         "| alpha | arm | n | acc (%) | sd | steps | examples | TFLOPs exec | TFLOPs nominal |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in pa.iterrows():
        L.append(f"| {r.alpha:g} | {r.arm} | {r.n} | {r.acc_mean_pp:.2f} | {r.acc_sd_pp:.2f} | "
                 f"{r.steps_mean:.0f} | {r.examples_mean:.0f} | {r.tflops_executed_mean:.1f} | {r.tflops_nominal_mean:.1f} |")
    L += ["", "| alpha | mean diff (pp) | 95% CI boot | 95% CI t | dz | W/L | p Wilcoxon | p Holm | "
          "max abs mismatch steps / examples / exec FLOPs | nominal FLOPs mismatch mean (max) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in c.iterrows():
        L.append(f"| {r.alpha:g} | {fmt(r.mean_diff_pp)} | [{fmt(r.ci95_boot_lo)}, {fmt(r.ci95_boot_hi)}] | "
                 f"[{fmt(r.ci95_t_lo)}, {fmt(r.ci95_t_hi)}] | {r.dz:.2f} | {r.wins_full}/{r.wins_perm} | "
                 f"{r.p_wilcoxon:.4f} | {r.p_wilcoxon_holm:.4f} | "
                 f"{r.max_abs_steps_mismatch_pct:.2f}% / {r.max_abs_examples_mismatch_pct:.2f}% / "
                 f"{r.max_abs_tflops_exec_mismatch_pct:.2f}% | {fmt(r.mean_tflops_nominal_mismatch_pct)}% "
                 f"({r.max_abs_tflops_nominal_mismatch_pct:.2f}%) |")
    L += ["", "### Aggregation-weighted work (p_k = n_k / N)", "",
          "| alpha | perm/full sum p*tau median [min, max] | perm/full sum p*eta*tau median | seeds with ratio < 1 | "
          "Spearman(diff, ratio p*eta*tau) | median Spearman(H_k, n_k) in Full |", "|---|---|---|---|---|---|"]
    for alpha, g in disp.groupby("alpha"):
        rho, pr = stats.spearmanr(g["diff_acc_pp"], g["ratio_sum_p_eta_tau"])
        L.append(f"| {alpha:g} | {g.ratio_sum_p_tau.median():.2f} [{g.ratio_sum_p_tau.min():.2f}, {g.ratio_sum_p_tau.max():.2f}] | "
                 f"{g.ratio_sum_p_eta_tau.median():.2f} | {int((g.ratio_sum_p_tau < 1).sum())}/{len(g)} | "
                 f"{rho:+.2f} (p={pr:.4f}) | {g.spearman_H_n_full.median():+.2f} |")
    L += ["", "## Experiment 2 — tuned static FedProx (validation phase only, seeds 72-76)", "",
          "Evaluate phase NOT executed (no frozen_config.json). Numbers below are 5 validation "
          "seeds; they select a configuration, they are not the evaluation.", "",
          "| alpha | config | val acc (%) | val - FedHAD (pp) | steps / FedHAD |", "|---|---|---|---|---|"]
    for _, r in v.sort_values(["alpha", "steps_mean", "lr"]).iterrows():
        cfg = "FedHAD" if r.arm == "full_fedhad" else f"FedProx lr={r.lr:g} E={int(r.epochs)}"
        L.append(f"| {r.alpha:g} | {cfg} | {r.val_pp:.2f} | {fmt(r.val_minus_fedhad_pp)} | {r.steps_vs_fedhad:.2f} |")
    L += ["", "Pre-specified `budget_matched` proposal (steps <= 1.02 x FedHAD, best validation accuracy):"]
    for e in prop["by_alpha_method"]:
        b = e["budget_matched"]
        L.append(f"- alpha={e['alpha']:g}: FedProx lr={b['lr']:g}, E={b['epochs']} "
                 f"(val {100*b['val_acc_mean']:.2f}% vs FedHAD {100*e['fedhad_reference']['val_acc_mean']:.2f}%; "
                 f"steps {b['steps_mean']:.0f} vs {e['fedhad_reference']['steps_mean']:.0f})")
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_step()
    ps = paired(m)
    c = contrasts(ps)
    pa = per_arm(m)
    v, prop = tune_table()
    disp = displacement(ps)
    disp.to_csv(OUT / "displacement.csv", index=False)
    pa.to_csv(OUT / "per_arm.csv", index=False)
    ps.to_csv(OUT / "per_seed.csv", index=False)
    c.to_csv(OUT / "contrasts.csv", index=False)
    v.to_csv(OUT / "tune_table.csv", index=False)
    text = report(pa, c, ps, v, prop, disp)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
