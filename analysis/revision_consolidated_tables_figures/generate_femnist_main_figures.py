#!/usr/bin/env python3
"""Generate publication figures for the 50-round FEMNIST main campaign.

The script is deliberately read-only with respect to experiment results.  It
parses the 120 raw logs stored in main_campaign_default_alpha_0_01/resultados_cluster.zip,
checks their protocol and terminal values against the consolidated master CSV,
and writes only derived CSVs, figures, and documentation below
revision_consolidated_tables_figures/figures/femnist_main.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "results" / "main_campaign_default_alpha_0_01" / "resultados_cluster.zip"
MASTER = ROOT / "results" / "revision_consolidated_tables_figures" / "data" / "master_results_summary.csv"
OUT = ROOT / "results" / "revision_consolidated_tables_figures" / "figures" / "femnist_main"

METHODS = ["FedAVG", "FedAvgM", "FedProx", "FedHAD"]
DISPLAY = {"FedAVG": "FedAvg", "FedAvgM": "FedAvgM", "FedProx": "FedProx", "FedHAD": "FedHAD"}
COLORS = {"FedAVG": "#1F77B4", "FedAvgM": "#9467BD", "FedProx": "#2CA02C", "FedHAD": "#D62728"}
MARKERS = {"FedAVG": "o", "FedAvgM": "^", "FedProx": "D", "FedHAD": "s"}
TARGET_CANDIDATES = [0.60, 0.65, 0.70, 0.75]
MIN_REACH_FRACTION = 0.80

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.32,
        "axes.axisbelow": True,
        "font.family": "sans-serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def extract_section(text: str, heading: str):
    match = re.search(
        rf"--- {re.escape(heading)} ---\s*(.*?)(?=\n--- |\Z)", text, flags=re.S
    )
    if not match:
        raise ValueError(f"Missing section: {heading}")
    return ast.literal_eval(match.group(1).strip())


def extract_scalar(text: str, label: str, cast=float):
    match = re.search(rf"^{re.escape(label)}:\s*(.+?)\s*$", text, flags=re.M)
    if not match:
        raise ValueError(f"Missing scalar: {label}")
    return cast(match.group(1))


def load_logs() -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict] = []
    round_rows: list[dict] = []
    with ZipFile(ARCHIVE) as archive:
        members = sorted(
            member
            for member in archive.namelist()
            if "test8_femnist" in member and member.endswith(".txt")
        )
        if len(members) != 120:
            raise RuntimeError(f"Expected 120 FEMNIST logs, found {len(members)}")

        for member in members:
            text = archive.read(member).decode("utf-8", errors="strict")
            method_match = re.search(r"/(Fed(?:AVG|AvgM|Prox|HAD))__", member)
            seed_match = re.search(r"__Seed-(\d+)\.txt$", member)
            if not method_match or not seed_match:
                raise RuntimeError(f"Unexpected log name: {member}")
            method, seed = method_match.group(1), int(seed_match.group(1))
            if method not in METHODS:
                raise RuntimeError(f"Method outside the approved set: {method}")

            required = {
                "EXPERIMENT_TAG": "test8_femnist",
                "Dataset": "FEMNIST",
                "NUM_ROUNDS": "50",
                "NUM_CLIENTS": "10",
                "CLIENT_SETUP": "10",
                "COMMUNICATION_DELAY": "0.05",
            }
            for label, expected in required.items():
                actual = str(extract_scalar(text, label, str)).strip()
                if actual != expected:
                    raise RuntimeError(f"Protocol mismatch in {member}: {label}={actual}")
            if "PARTITIONING: NATURAL (per-writer)" not in text:
                raise RuntimeError(f"Non-natural FEMNIST partition in {member}")

            metrics = extract_section(text, "History (metrics, centralized)")
            losses = dict(extract_section(text, "History (loss, centralized)"))
            fit = extract_section(text, "History (metrics, distributed, fit)")
            accuracies = dict(metrics["accuracy"])
            round_flops = dict(fit["total_flops_round"])
            if set(accuracies) != set(range(0, 51)) or set(losses) != set(range(0, 51)):
                raise RuntimeError(f"Incomplete centralized history in {member}")
            if set(round_flops) != set(range(1, 51)):
                raise RuntimeError(f"Incomplete FLOP history in {member}")

            cumulative = 0.0
            round_rows.append(
                {
                    "campaign": "base_alpha_0.01/test8_femnist",
                    "method": method,
                    "seed": seed,
                    "round": 0,
                    "accuracy": float(accuracies[0]),
                    "loss": float(losses[0]),
                    "round_tflops": 0.0,
                    "accumulated_tflops": 0.0,
                    "source_member": member,
                }
            )
            for rnd in range(1, 51):
                cost = float(round_flops[rnd]) / 1e12
                cumulative += cost
                round_rows.append(
                    {
                        "campaign": "base_alpha_0.01/test8_femnist",
                        "method": method,
                        "seed": seed,
                        "round": rnd,
                        "accuracy": float(accuracies[rnd]),
                        "loss": float(losses[rnd]),
                        "round_tflops": cost,
                        "accumulated_tflops": cumulative,
                        "source_member": member,
                    }
                )

            reported_total = float(
                re.search(r"Total acumulado da execução:\s*([0-9.]+) TFLOPs", text).group(1)
            )
            if not np.isclose(cumulative, reported_total, atol=5e-4):
                raise RuntimeError(
                    f"Per-round and reported total FLOPs disagree in {member}: "
                    f"{cumulative:.9f} vs {reported_total:.3f}"
                )
            seed_rows.append(
                {
                    "campaign": "base_alpha_0.01/test8_femnist",
                    "method": method,
                    "seed": seed,
                    "rounds": 50,
                    "clients": 10,
                    "partitioning": "natural (per-writer)",
                    "final_accuracy": float(accuracies[50]),
                    "best_accuracy": max(float(v) for v in accuracies.values()),
                    "total_training_tflops": cumulative,
                    "reported_total_tflops_3dp": reported_total,
                    "source_member": member,
                }
            )

    seeds = pd.DataFrame(seed_rows).sort_values(["method", "seed"]).reset_index(drop=True)
    rounds = pd.DataFrame(round_rows).sort_values(["method", "seed", "round"]).reset_index(drop=True)
    for method in METHODS:
        observed = sorted(seeds.loc[seeds.method == method, "seed"].tolist())
        if observed != list(range(42, 72)):
            raise RuntimeError(f"Expected seeds 42--71 for {method}, found {observed}")
    return seeds, rounds


def validate_against_master(seeds: pd.DataFrame) -> None:
    master = pd.read_csv(MASTER)
    q = master[
        (master.source_campaign == "base_alpha_0.01")
        & (master.experiment_tag == "test8_femnist")
        & (master.dataset == "FEMNIST")
        & (master.method.isin(METHODS))
        & (master.rounds == 50)
    ][["method", "seed", "final_accuracy", "tflops"]].copy()
    q["seed"] = q.seed.astype(int)
    if len(q) != 120 or q.groupby("method").seed.nunique().to_dict() != {m: 30 for m in METHODS}:
        raise RuntimeError("The consolidated master does not contain the expected 4 x 30 campaign rows")
    merged = seeds.merge(q, on=["method", "seed"], suffixes=("_raw", "_master"), validate="one_to_one")
    if not np.allclose(merged.final_accuracy_raw, merged.final_accuracy_master, rtol=0, atol=1e-12):
        raise RuntimeError("Raw and consolidated final accuracies disagree")
    if not np.allclose(merged.reported_total_tflops_3dp, merged.tflops, rtol=0, atol=5e-4):
        raise RuntimeError("Raw and consolidated total TFLOPs disagree")


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], color=COLORS[m], marker=MARKERS[m], lw=2.0,
            markersize=8 if m == "FedHAD" else 7,
            markeredgecolor="black" if m == "FedHAD" else "white",
            label=DISPLAY[m],
        )
        for m in METHODS
    ]


def summary(seeds: pd.DataFrame) -> pd.DataFrame:
    return (
        seeds.groupby("method", sort=False)
        .agg(
            n_seeds=("seed", "size"),
            final_accuracy_mean=("final_accuracy", "mean"),
            final_accuracy_sd=("final_accuracy", "std"),
            total_training_tflops_mean=("total_training_tflops", "mean"),
            total_training_tflops_sd=("total_training_tflops", "std"),
        )
        .reindex(METHODS)
        .reset_index()
    )


def plot_pareto(
    stats: pd.DataFrame,
    xlabel: str = "Total FLOPs (TFLOPs)",
    ylabel: str = "Final accuracy",
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    # FedProx and FedAvg have nearly identical coordinates.  Drawing the
    # diamond first and the smaller circle second makes both glyphs visible
    # without changing either x coordinate (no jitter or cosmetic offset).
    plot_order = ["FedAvgM", "FedProx", "FedAVG", "FedHAD"]
    rows = stats.set_index("method")
    for method in plot_order:
        row = rows.loc[method]
        ax.errorbar(
            row.total_training_tflops_mean,
            100 * row.final_accuracy_mean,
            xerr=row.total_training_tflops_sd,
            yerr=100 * row.final_accuracy_sd,
            fmt=MARKERS[method], color=COLORS[method], capsize=4,
            markersize=12 if method == "FedHAD" else (7 if method == "FedAVG" else 9),
            markeredgecolor="black" if method == "FedHAD" else "white",
            markeredgewidth=1.5 if method == "FedHAD" else 1.0, zorder=4,
        )
        label_offsets = {
            "FedAVG": (10, -8),
            "FedAvgM": (10, -19),
            "FedProx": (10, 10),
            "FedHAD": (8, 8),
        }
        ax.annotate(
            DISPLAY[method],
            (row.total_training_tflops_mean, 100 * row.final_accuracy_mean),
            xytext=label_offsets[method], textcoords="offset points",
            fontsize=10, color=COLORS[method], weight="bold" if method == "FedHAD" else "normal",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(
        stats.total_training_tflops_mean.min() - 0.3,
        stats.total_training_tflops_mean.max() + 1.1,
    )
    save(fig, "01_pareto_final_accuracy_vs_tflops")


def plot_seed_scatter(seeds: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for method in METHODS:
        g = seeds[seeds.method == method]
        ax.scatter(
            g.total_training_tflops, 100 * g.final_accuracy,
            s=64 if method == "FedHAD" else 43, marker=MARKERS[method],
            color=COLORS[method], alpha=0.72 if method == "FedHAD" else 0.55,
            edgecolor="black" if method == "FedHAD" else "white",
            linewidth=0.8, label=DISPLAY[method], zorder=4 if method == "FedHAD" else 3,
        )
    ax.set_xlabel("Total training compute (TFLOPs)")
    ax.set_ylabel("Final centralized accuracy (%)")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "02_seed_scatter_accuracy_vs_tflops")


def plot_bars(stats: pd.DataFrame) -> None:
    x = np.arange(len(METHODS))
    labels = [DISPLAY[m] for m in METHODS]
    colors = [COLORS[m] for m in METHODS]

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    bars = ax.bar(
        x, 100 * stats.final_accuracy_mean, yerr=100 * stats.final_accuracy_sd,
        color=colors, width=0.68, capsize=4, edgecolor="black", linewidth=0.8,
    )
    bars[-1].set_linewidth(1.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Final centralized accuracy (%)")
    ax.set_ylim(0, max(100 * (stats.final_accuracy_mean + stats.final_accuracy_sd)) * 1.14)
    for i, row in stats.iterrows():
        ax.text(i, 100 * (row.final_accuracy_mean + row.final_accuracy_sd) + 1.2,
                f"{100 * row.final_accuracy_mean:.2f}", ha="center", fontsize=10)
    save(fig, "03_final_accuracy_errorbars")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    bars = ax.bar(
        x, stats.total_training_tflops_mean, yerr=stats.total_training_tflops_sd,
        color=colors, width=0.68, capsize=4, edgecolor="black", linewidth=0.8,
    )
    bars[-1].set_linewidth(1.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Total training compute (TFLOPs)")
    ax.set_ylim(0, max(stats.total_training_tflops_mean + stats.total_training_tflops_sd) * 1.15)
    for i, row in stats.iterrows():
        ax.text(i, row.total_training_tflops_mean + row.total_training_tflops_sd + 1.0,
                f"{row.total_training_tflops_mean:.2f}", ha="center", fontsize=10)
    save(fig, "04_total_training_tflops_errorbars")


def trajectory_stats(rounds: pd.DataFrame) -> pd.DataFrame:
    return (
        rounds.groupby(["method", "round"], sort=False)
        .agg(
            n_seeds=("seed", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            loss_mean=("loss", "mean"),
            loss_sd=("loss", "std"),
            accumulated_tflops_mean=("accumulated_tflops", "mean"),
            accumulated_tflops_sd=("accumulated_tflops", "std"),
        )
        .reset_index()
    )


def plot_compute_trajectory(ts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    for method in METHODS:
        g = ts[ts.method == method].sort_values("round")
        x = g.accumulated_tflops_mean.to_numpy()
        y = (100 * g.accuracy_mean).to_numpy()
        sd = (100 * g.accuracy_sd.fillna(0)).to_numpy()
        ax.plot(x, y, color=COLORS[method], marker=MARKERS[method], markevery=5,
                markersize=5.5, lw=2.1, label=DISPLAY[method])
        ax.fill_between(x, y - sd, y + sd, color=COLORS[method], alpha=0.11, lw=0)
        sparse = g[(g["round"] > 0) & (g["round"] % 10 == 0)]
        ax.errorbar(
            sparse.accumulated_tflops_mean, 100 * sparse.accuracy_mean,
            xerr=sparse.accumulated_tflops_sd, fmt="none", ecolor=COLORS[method],
            alpha=0.65, capsize=2, lw=0.9,
        )
    ax.set_xlabel("Accumulated training compute (TFLOPs)")
    ax.set_ylabel("Centralized accuracy (%)")
    ax.legend(handles=legend_handles(), frameon=False, loc="lower right")
    save(fig, "05_accuracy_vs_accumulated_tflops")


def compute_targets(rounds: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    records: list[dict] = []
    total_seeds = 30
    for target in TARGET_CANDIDATES:
        for method in METHODS:
            values = []
            for _, g in rounds[(rounds.method == method) & (rounds["round"] > 0)].groupby("seed"):
                reached = g[g.accuracy >= target].sort_values("round")
                if not reached.empty:
                    values.append(float(reached.iloc[0].accumulated_tflops))
            records.append(
                {
                    "method": method,
                    "target_accuracy": target,
                    "reached_seeds": len(values),
                    "total_seeds": total_seeds,
                    "reach_fraction": len(values) / total_seeds,
                    "tflops_mean_reached_only": np.mean(values) if values else np.nan,
                    "tflops_sd_reached_only": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                    "tflops_median_reached_only": np.median(values) if values else np.nan,
                }
            )
    target_df = pd.DataFrame(records)
    retained = [
        target
        for target in TARGET_CANDIDATES
        if target_df[target_df.target_accuracy == target].reach_fraction.min() >= MIN_REACH_FRACTION
    ]
    return target_df, retained


def plot_cost_to_target(target_df: pd.DataFrame, targets: list[float]) -> None:
    q = target_df[target_df.target_accuracy.isin(targets)]
    x = np.arange(len(targets))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for j, method in enumerate(METHODS):
        g = q[q.method == method].set_index("target_accuracy").loc[targets]
        xpos = x + (j - 1.5) * width
        bars = ax.bar(
            xpos, g.tflops_mean_reached_only, yerr=g.tflops_sd_reached_only.fillna(0),
            width=width, color=COLORS[method], edgecolor="black",
            linewidth=1.4 if method == "FedHAD" else 0.7, capsize=3, label=DISPLAY[method],
        )
        for bar, reached in zip(bars, g.reached_seeds):
            ax.text(
                bar.get_x() + bar.get_width() / 2, 0.55 * bar.get_height(),
                f"{int(reached)}/30", ha="center", va="center", fontsize=7.4,
                color="white", weight="bold",
            )
    ax.set_xticks(x, [f"{100*t:.0f}%" for t in targets])
    ax.set_xlabel("Target centralized accuracy")
    ax.set_ylabel("TFLOPs to first reach target")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.subplots_adjust(bottom=0.20)
    fig.text(
        0.5, 0.025, "Mean ± 1 SD among reaching seeds; labels show reached/30",
        ha="center", va="bottom", fontsize=9, color="0.3",
    )
    save(fig, "06_cost_to_target")


def plot_round_trajectories(ts: pd.DataFrame) -> None:
    for metric, stem, ylabel in [
        ("accuracy", "07_accuracy_over_communication_rounds", "Centralized accuracy (%)"),
        ("loss", "08_loss_over_communication_rounds", "Centralized loss"),
    ]:
        fig, ax = plt.subplots(figsize=(6.7, 4.8))
        for method in METHODS:
            g = ts[ts.method == method].sort_values("round")
            mean = g[f"{metric}_mean"].to_numpy() * (100 if metric == "accuracy" else 1)
            sd = g[f"{metric}_sd"].fillna(0).to_numpy() * (100 if metric == "accuracy" else 1)
            r = g["round"].to_numpy()
            ax.plot(r, mean, color=COLORS[method], marker=MARKERS[method], markevery=5,
                    markersize=5.5, lw=2.1, label=DISPLAY[method])
            ax.fill_between(r, mean - sd, mean + sd, color=COLORS[method], alpha=0.11, lw=0)
        ax.set_xlabel("Communication round")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, int(ts["round"].max()))
        ax.legend(handles=legend_handles(), frameon=False, loc="best")
        save(fig, stem)


def plot_target_reliability(rounds: pd.DataFrame, targets: list[float]) -> None:
    fig, axes = plt.subplots(1, len(targets), figsize=(11.2, 3.8), sharey=True)
    max_cost = float(rounds.accumulated_tflops.max())
    x_end = 1.01 * max_cost
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        for method in METHODS:
            hits = []
            for _, g in rounds[(rounds.method == method) & (rounds["round"] > 0)].groupby("seed"):
                reached = g[g.accuracy >= target].sort_values("round")
                if not reached.empty:
                    hits.append(float(reached.iloc[0].accumulated_tflops))
            hits = np.sort(hits)
            x = np.r_[0, hits, x_end]
            y = np.r_[0, np.arange(1, len(hits) + 1) / 30, len(hits) / 30]
            ax.step(x, y, where="post", color=COLORS[method], lw=2.0, label=DISPLAY[method])
        ax.set_title(f"Target: {100*target:.0f}%")
        ax.set_xlabel("Accumulated TFLOPs")
        ax.set_xlim(0, x_end)
        ax.set_ylim(0, 1.03)
    axes[0].set_ylabel("Fraction of seeds reaching target")
    axes[-1].legend(handles=legend_handles(), frameon=False, loc="lower right")
    save(fig, "09_target_attainment_probability_vs_tflops")


def write_readme(stats: pd.DataFrame, target_df: pd.DataFrame, targets: list[float]) -> None:
    target_text = ", ".join(f"{100*t:.0f}%" for t in targets)
    lines = [
        "# FEMNIST figures of the main campaign (50 rounds)",
        "",
        "## Scope and audit",
        "",
        "These artifacts use only `main_campaign_default_alpha_0_01/test8_femnist` (read from "
        "`resultados_cluster.zip`, a byte-for-byte copy of the folder): 50 rounds, 10 clients, "
        "natural partitioning by writer, communication delay 0.05 and seeds 42--71. There are "
        "exactly 30 runs for each of FedAvg, FedAvgM, FedProx and FedHAD. No training was run. "
        "The terminal values were checked against `data/master_results_summary.csv`; the "
        "accumulated cost was reconstructed by summing the values recorded in `total_flops_round`.",
        "",
        "All error bars and shaded bands show the mean +/- one sample standard deviation over 30 "
        "seeds. Accuracy and loss are centralized evaluation metrics. No Accuracy/TFLOP ratio is used.",
        "",
        "## Purpose of each figure",
        "",
        "1. `01_pareto_final_accuracy_vs_tflops`: main view of the accuracy--cost trade-off; shows "
        "both uncertainties and highlights FedHAD without artificially offsetting the points. The "
        "reading of the upper-left region is left to the caption.",
        "2. `02_seed_scatter_accuracy_vs_tflops`: shows the whole distribution across runs, not only "
        "the aggregated means.",
        "3. `03_final_accuracy_errorbars`: compares predictive performance alone.",
        "4. `04_total_training_tflops_errorbars`: compares training cost alone.",
        "5. `05_accuracy_vs_accumulated_tflops`: shows learning as the measured cost accumulates; the "
        "line uses the mean accumulated cost and the mean accuracy per round, the vertical band is the "
        "SD of accuracy and the sparse horizontal bars are the SD of cost.",
        f"6. `06_cost_to_target`: compares the cost to first reach {target_text}. Means and SDs are "
        "conditional on the seeds that reach the target, and each bar reports reached/30.",
        "7. `07_accuracy_over_communication_rounds`: compares convergence in the communication domain.",
        "8. `08_loss_over_communication_rounds`: checks whether the accuracy evidence is consistent "
        "with the dynamics of the centralized loss.",
        "9. `09_target_attainment_probability_vs_tflops`: complements the conditional cost-to-target "
        "means with the empirical probability of reaching each target under a cost budget.",
        "",
        "## Use in the manuscript",
        "",
        "None of these figures appears in the current manuscript. The 50-round FEMNIST block is "
        "reported in Table 11 (Section 6.7): differences between FedHAD and the three baselines "
        "below 0.15 percentage points, none detected after the Holm adjustment, and a reduction of "
        "about 10% in nominal computation (55.67 against 61.97 TFLOPs). An undetected difference is "
        "not treated as equivalence. The Accuracy/TFLOP ratio is not used in the manuscript.",
        "",
        "## Suggested caption",
        "",
        "Final accuracy against total training computation on FEMNIST after 50 communication "
        "rounds. Points show means over 30 seeds and error bars denote one standard deviation. "
        "Points toward the upper-left region represent a more favorable accuracy--computation "
        "trade-off.",
        "",
        "## Target rule",
        "",
        f"The candidate targets were 60%, 65%, 70% and 75%. A target appears only when every method "
        f"reaches it in at least {100*MIN_REACH_FRACTION:.0f}% of the seeds. This retains {target_text}; "
        "the full audit, including omitted targets, is in `femnist_cost_to_target.csv`.",
        "",
        "## Aggregate values used",
        "",
        "| Method | n | Final accuracy, mean +/- SD (%) | Total TFLOPs, mean +/- SD |",
        "|---|---:|---:|---:|",
    ]
    for _, row in stats.iterrows():
        lines.append(
            f"| {DISPLAY[row.method]} | {int(row.n_seeds)} | "
            f"{100*row.final_accuracy_mean:.3f} +/- {100*row.final_accuracy_sd:.3f} | "
            f"{row.total_training_tflops_mean:.3f} +/- {row.total_training_tflops_sd:.3f} |"
        )
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds, rounds = load_logs()
    validate_against_master(seeds)
    stats = summary(seeds)
    ts = trajectory_stats(rounds)
    targets_df, retained_targets = compute_targets(rounds)
    if retained_targets != [0.60, 0.65, 0.70]:
        raise RuntimeError(f"Unexpected defensible target set: {retained_targets}")

    seeds.to_csv(OUT / "femnist_seed_summary.csv", index=False, float_format="%.12g")
    rounds.to_csv(OUT / "femnist_round_history.csv", index=False, float_format="%.12g")
    stats.to_csv(OUT / "femnist_aggregate_statistics.csv", index=False, float_format="%.12g")
    ts.to_csv(OUT / "femnist_trajectory_statistics.csv", index=False, float_format="%.12g")
    targets_df.to_csv(OUT / "femnist_cost_to_target.csv", index=False, float_format="%.12g")

    plot_pareto(stats)
    plot_seed_scatter(seeds)
    plot_bars(stats)
    plot_compute_trajectory(ts)
    plot_cost_to_target(targets_df, retained_targets)
    plot_round_trajectories(ts)
    plot_target_reliability(rounds, retained_targets)
    write_readme(stats, targets_df, retained_targets)
    print(f"Generated 9 figures (PDF + PNG) and audit CSVs in {OUT}")


if __name__ == "__main__":
    main()
