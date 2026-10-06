#!/usr/bin/env python3
"""Generate the FEMNIST figure suite from main_campaign_default_alpha_0_5.

Despite the source-directory name, FEMNIST uses natural per-writer
partitioning and therefore has no Dirichlet alpha.  This script keeps this
10-round historical campaign separate from the 50-round campaign.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import generate_femnist_main_figures as common
from femnist_compute_accounting import FIXED_EPOCH_METHODS, ROUND_TFLOPS, TOTAL_TFLOPS


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "main_campaign_default_alpha_0_5"
OUT = ROOT / "results" / "revision_consolidated_tables_figures" / "figures" / "femnist_main_base_alpha_0_5"
CAMPAIGN = "base_alpha_0.5/test8_femnist"
ROUNDS = 10
TARGET_CANDIDATES = [0.50, 0.55, 0.60, 0.65]
MIN_REACH_FRACTION = 0.80


def load_logs() -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict] = []
    round_rows: list[dict] = []
    selected: list[tuple[str, Path]] = []
    for method in common.METHODS:
        selected.extend(
            (method, path)
            for path in sorted((SOURCE / f"{method}-results" / "test8_femnist").rglob("*.txt"))
        )
    if len(selected) != 120:
        raise RuntimeError(f"Expected 120 approved FEMNIST logs, found {len(selected)}")

    for method, path in selected:
        text = path.read_text(encoding="utf-8", errors="strict")
        seed_match = re.search(r"__Seed-(\d+)\.txt$", path.name)
        if not seed_match:
            raise RuntimeError(f"Unexpected log name: {path}")
        seed = int(seed_match.group(1))
        required = {
            "EXPERIMENT_TAG": "test8_femnist",
            "Dataset": "FEMNIST",
            "NUM_ROUNDS": str(ROUNDS),
            "NUM_CLIENTS": "10",
            "CLIENT_SETUP": "10",
            "COMMUNICATION_DELAY": "0.05",
        }
        for label, expected in required.items():
            actual = str(common.extract_scalar(text, label, str)).strip()
            if actual != expected:
                raise RuntimeError(f"Protocol mismatch in {path}: {label}={actual}")
        if "PARTITIONING: NATURAL (per-writer)" not in text:
            raise RuntimeError(f"Non-natural FEMNIST partition in {path}")

        metrics = common.extract_section(text, "History (metrics, centralized)")
        losses = dict(common.extract_section(text, "History (loss, centralized)"))
        fit = common.extract_section(text, "History (metrics, distributed, fit)")
        accuracies = dict(metrics["accuracy"])
        round_flops = dict(fit["total_flops_round"])
        if set(accuracies) != set(range(ROUNDS + 1)) or set(losses) != set(range(ROUNDS + 1)):
            raise RuntimeError(f"Incomplete centralized history in {path}")
        if set(round_flops) != set(range(1, ROUNDS + 1)):
            raise RuntimeError(f"Incomplete FLOP history in {path}")

        relative = path.relative_to(ROOT).as_posix()
        cumulative = 0.0
        observed_cumulative = 0.0
        round_rows.append(
            {
                "campaign": CAMPAIGN,
                "method": method,
                "seed": seed,
                "round": 0,
                "accuracy": float(accuracies[0]),
                "loss": float(losses[0]),
                "round_tflops": 0.0,
                "accumulated_tflops": 0.0,
                "reported_round_tflops": 0.0,
                "reported_accumulated_tflops": 0.0,
                "source_member": relative,
            }
        )
        for rnd in range(1, ROUNDS + 1):
            reported_cost = float(round_flops[rnd]) / 1e12
            observed_cumulative += reported_cost
            cost = ROUND_TFLOPS if method in FIXED_EPOCH_METHODS else reported_cost
            cumulative += cost
            round_rows.append(
                {
                    "campaign": CAMPAIGN,
                    "method": method,
                    "seed": seed,
                    "round": rnd,
                    "accuracy": float(accuracies[rnd]),
                    "loss": float(losses[rnd]),
                    "round_tflops": cost,
                    "accumulated_tflops": cumulative,
                    "reported_round_tflops": reported_cost,
                    "reported_accumulated_tflops": observed_cumulative,
                    "source_member": relative,
                }
            )
        reported_match = re.search(
            r"Total acumulado da execução:\s*([0-9.]+) TFLOPs", text
        )
        if not reported_match:
            raise RuntimeError(f"Missing reported total FLOPs in {path}")
        reported_total = float(reported_match.group(1))
        if not np.isclose(observed_cumulative, reported_total, atol=5e-4):
            raise RuntimeError(
                f"Per-round and reported total FLOPs disagree in {path}: "
                f"{observed_cumulative:.9f} vs {reported_total:.3f}"
            )
        seed_rows.append(
            {
                "campaign": CAMPAIGN,
                "method": method,
                "seed": seed,
                "rounds": ROUNDS,
                "clients": 10,
                "partitioning": "natural (per-writer)",
                "final_accuracy": float(accuracies[ROUNDS]),
                "best_accuracy": max(float(v) for v in accuracies.values()),
                "total_training_tflops": cumulative,
                "reported_total_training_tflops": observed_cumulative,
                "reported_total_tflops_3dp": reported_total,
                "source_member": relative,
            }
        )

    seeds = pd.DataFrame(seed_rows).sort_values(["method", "seed"]).reset_index(drop=True)
    rounds = pd.DataFrame(round_rows).sort_values(["method", "seed", "round"]).reset_index(drop=True)
    for method in common.METHODS:
        observed = sorted(seeds.loc[seeds.method == method, "seed"].tolist())
        if observed != list(range(42, 72)):
            raise RuntimeError(f"Expected seeds 42--71 for {method}, found {observed}")
    return seeds, rounds


def validate_against_master(seeds: pd.DataFrame) -> None:
    master = pd.read_csv(common.MASTER)
    q = master[
        (master.source_campaign == "base_alpha_0.5")
        & (master.experiment_tag == "test8_femnist")
        & (master.dataset == "FEMNIST")
        & (master.method.isin(common.METHODS))
        & (master.rounds == ROUNDS)
    ][["method", "seed", "final_accuracy", "tflops", "tflops_reported"]].copy()
    q["seed"] = q.seed.astype(int)
    if len(q) != 120 or q.groupby("method").seed.nunique().to_dict() != {
        method: 30 for method in common.METHODS
    }:
        raise RuntimeError("Consolidated master does not contain the expected 4 x 30 rows")
    merged = seeds.merge(q, on=["method", "seed"], suffixes=("_raw", "_master"), validate="one_to_one")
    if not np.allclose(merged.final_accuracy_raw, merged.final_accuracy_master, rtol=0, atol=1e-12):
        raise RuntimeError("Raw and consolidated final accuracies disagree")
    if not np.allclose(merged.reported_total_tflops_3dp, merged.tflops_reported, rtol=0, atol=5e-4):
        raise RuntimeError("Raw and consolidated reported TFLOPs disagree")
    # FedHAD's master value is parsed from the 3-decimal total printed in the
    # historical log, whereas the trajectory sums full per-round integers.
    if not np.allclose(merged.total_training_tflops, merged.tflops, rtol=0, atol=5e-4):
        raise RuntimeError("Nominal figure and consolidated publication TFLOPs disagree")


def write_readme(stats: pd.DataFrame, targets: list[float]) -> None:
    target_text = ", ".join(f"{100*t:.0f}%" for t in targets)
    lines = [
        "# FEMNIST figures of the base-alpha-0_5 campaign (10 rounds)",
        "",
        "## Scope and audit",
        "",
        "These artifacts use only `main_campaign_default_alpha_0_5/test8_femnist`: 10 rounds, "
        "10 clients, natural partitioning by writer, delay 0.05 and seeds 42--71. "
        "The folder name is historical; FEMNIST does not use a Dirichlet alpha in this campaign. "
        "There are exactly 30 runs for each approved method. No training was run. "
        "The terminal values were checked against `data/master_results_summary.csv`. For the "
        "methods with 5 fixed epochs, the publication cost is the nominal scheduled workload: "
        f"116602368 FLOPs/sample x 2126 samples x 5 epochs x 10 rounds = {TOTAL_TFLOPS:.10f} "
        "TFLOPs. The values actually summed from the logs remain in the `reported_*` columns.",
        "The canonical logs of FedProx/seed 66 and FedHAD/seed 53 are the complete re-executions, "
        "run on the server with the same configuration and seed because the original executions on "
        "the workstation were left incomplete by failures of that machine (Section 5.8 of the "
        "manuscript); the partial logs were preserved with the suffix "
        "`discarded_incomplete_2026-09-10` and fall outside the `*.txt` pattern used in this analysis.",
        "",
        "All error bars and bands show the mean +/- one sample standard deviation over 30 "
        "seeds. Accuracy and loss are centralized. No Accuracy/TFLOP ratio is used.",
        "",
        "## Purpose of each figure",
        "",
        "1. `01_pareto_final_accuracy_vs_tflops`: main trade-off between final accuracy and cost, "
        "with uncertainty on both axes and FedHAD highlighted, without offsetting the points.",
        "2. `02_seed_scatter_accuracy_vs_tflops`: full distribution of the 30 runs.",
        "3. `03_final_accuracy_errorbars`: predictive performance alone.",
        "4. `04_total_training_tflops_errorbars`: computational cost alone.",
        "5. `05_accuracy_vs_accumulated_tflops`: learning trajectory in the domain of the measured cost.",
        f"6. `06_cost_to_target`: TFLOPs to reach {target_text}, with reached/30 stated explicitly.",
        "7. `07_accuracy_over_communication_rounds`: convergence per communication round.",
        "8. `08_loss_over_communication_rounds`: dynamics of the centralized loss.",
        "9. `09_target_attainment_probability_vs_tflops`: empirical probability of reaching each "
        "target under a computational budget.",
        "",
        "## Use in the manuscript",
        "",
        "`01_pareto_final_accuracy_vs_tflops` is Figure 15 of the manuscript (Section 6.8). The other "
        "figures in this folder do not appear in the manuscript. At 10 rounds FedHAD is 0.83 "
        "percentage points behind FedProx, a difference that survives the Holm adjustment, with 10.2% "
        "less nominal computation; the executed totals are 10.30 and 11.38 TFLOPs. The Accuracy/TFLOP "
        "ratio is not used in the manuscript.",
        "",
        "## Suggested caption",
        "",
        "Final accuracy against total training computation on FEMNIST after 10 communication "
        "rounds. Points show means over 30 seeds and error bars denote one standard deviation. "
        "Points toward the upper-left region represent a more favorable accuracy--computation "
        "trade-off.",
        "",
        "## Target rule",
        "",
        f"The candidates were 50%, 55%, 60% and 65%. A target is plotted only when every method "
        f"reaches it in at least {100*MIN_REACH_FRACTION:.0f}% of the seeds. Retained: "
        f"{target_text}; the full audit is in `femnist_cost_to_target.csv`.",
        "",
        "## Aggregate values used",
        "",
        "| Method | n | Final accuracy, mean +/- SD (%) | Total TFLOPs, mean +/- SD |",
        "|---|---:|---:|---:|",
    ]
    for _, row in stats.iterrows():
        lines.append(
            f"| {common.DISPLAY[row.method]} | {int(row.n_seeds)} | "
            f"{100*row.final_accuracy_mean:.3f} +/- {100*row.final_accuracy_sd:.3f} | "
            f"{row.total_training_tflops_mean:.3f} +/- {row.total_training_tflops_sd:.3f} |"
        )
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common.OUT = OUT
    common.TARGET_CANDIDATES = TARGET_CANDIDATES
    common.MIN_REACH_FRACTION = MIN_REACH_FRACTION

    seeds, rounds = load_logs()
    validate_against_master(seeds)
    stats = common.summary(seeds)
    trajectories = common.trajectory_stats(rounds)
    targets_df, retained = common.compute_targets(rounds)
    if retained != [0.50, 0.55, 0.60]:
        raise RuntimeError(f"Unexpected defensible target set: {retained}")

    seeds.to_csv(OUT / "femnist_seed_summary.csv", index=False, float_format="%.12g")
    rounds.to_csv(OUT / "femnist_round_history.csv", index=False, float_format="%.12g")
    stats.to_csv(OUT / "femnist_aggregate_statistics.csv", index=False, float_format="%.12g")
    trajectories.to_csv(OUT / "femnist_trajectory_statistics.csv", index=False, float_format="%.12g")
    targets_df.to_csv(OUT / "femnist_cost_to_target.csv", index=False, float_format="%.12g")

    common.plot_pareto(
        stats,
        xlabel="Total FLOPs (TFLOPs)",
        ylabel="Final accuracy",
    )
    common.plot_seed_scatter(seeds)
    common.plot_bars(stats)
    common.plot_compute_trajectory(trajectories)
    common.plot_cost_to_target(targets_df, retained)
    common.plot_round_trajectories(trajectories)
    common.plot_target_reliability(rounds, retained)
    write_readme(stats, retained)
    print(f"Generated 9 figures (PDF + PNG) and audit CSVs in {OUT}")


if __name__ == "__main__":
    main()
