#!/usr/bin/env python3
"""Generate the paired long-horizon table and per-alpha convergence figures.

Only the canonical, non-recursive test7_plato directory of each method is read.  This is
intentional: main_campaign_default_alpha_0_5 contains a duplicated nested FedAvgM-results tree.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "analysis" / "main_campaign") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis" / "main_campaign"))

from analisar_resultados import parse_file  # noqa: E402
from figuras_artigo import agregar_convergencia_std, fig_convergencia  # noqa: E402


OUT = ROOT / "results" / "revision_consolidated_tables_figures"
TABLES = OUT / "tables"
FIGURES = OUT / "figures" / "long_horizon"
DATA = OUT / "data"

METHODS = ("FedAVG", "FedAvgM", "FedProx", "FedHAD")
BASELINES = ("FedAVG", "FedAvgM", "FedProx")
DISPLAY = {"FedAVG": "FedAvg", "FedAvgM": "FedAvgM", "FedProx": "FedProx", "FedHAD": "FedHAD"}
CAMPAIGNS = {
    "0.5": ROOT / "results" / "main_campaign_default_alpha_0_5",
    "0.01": ROOT / "results" / "main_campaign_default_alpha_0_01",
}


def source_dir(campaign: Path, method: str) -> Path:
    dataset_dir = "CIFAR10-Standard" if method == "FedHAD" else "CIFAR10"
    return campaign / f"{method}-results" / "test7_plato" / dataset_dir


def read_campaign(alpha: str, campaign: Path) -> tuple[pd.DataFrame, dict[str, list[Path]]]:
    rows: list[dict] = []
    paths_by_method: dict[str, list[Path]] = {}
    expected_seeds = set(range(42, 72))

    for method in METHODS:
        directory = source_dir(campaign, method)
        paths = sorted(directory.glob("*.txt"))
        if len(paths) != 30:
            raise RuntimeError(f"{alpha} {method}: expected 30 canonical files, found {len(paths)} in {directory}")
        paths_by_method[method] = paths
        parsed = [parse_file(path) for path in paths]
        seeds = [int(row["seed"]) for row in parsed]
        if len(seeds) != len(set(seeds)):
            raise RuntimeError(f"{alpha} {method}: duplicate seed in canonical directory")
        if set(seeds) != expected_seeds:
            raise RuntimeError(f"{alpha} {method}: seeds are {sorted(seeds)}, expected 42--71")
        for path, row in zip(paths, parsed):
            if row["metodo"] != method or row["dataset"] != "CIFAR10":
                raise RuntimeError(f"metadata mismatch in {path}")
            if int(row["num_rounds"]) != 50 or row["experiment_tag"] != "test7_plato":
                raise RuntimeError(f"non-long-horizon report in {path}")
            if not np.isclose(float(row["alpha"]), float(alpha)):
                raise RuntimeError(f"alpha mismatch in {path}: {row['alpha']} != {alpha}")
            if row["acuracia_final"] is None or row["total_tflops"] is None:
                raise RuntimeError(f"missing final accuracy or FLOPs in {path}")
            rows.append({
                "alpha": alpha,
                "method": method,
                "seed": int(row["seed"]),
                "accuracy": float(row["acuracia_final"]),
                "tflops": float(row["total_tflops"]),
                "source": str(path.relative_to(ROOT)),
            })

    frame = pd.DataFrame(rows)
    seed_sets = {method: set(frame.loc[frame.method == method, "seed"]) for method in METHODS}
    if len({frozenset(seeds) for seeds in seed_sets.values()}) != 1:
        raise RuntimeError(f"{alpha}: methods do not share the same seeds")
    return frame, paths_by_method


def format_p(value: float) -> str:
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.4f}" if value < 0.01 else f"{value:.3f}"


def generate_table(summary: pd.DataFrame) -> str:
    lines = [
        r"% Generated from the raw per-seed test7_plato reports by generate_long_horizon_alpha_comparison.py.",
        r"% Requires \usepackage{booktabs}.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Long-horizon comparison on CIFAR-10 after 50 communication rounds. Accuracy is mean $\pm$ sample standard deviation over the same 30 seeds (42--71). $\Delta$Acc is FedHAD minus the indicated baseline in percentage points, and $\Delta$FLOPs is the relative training-compute reduction of FedHAD with respect to that baseline.}",
        r"\label{tab:long_horizon_alpha_comparison}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"$\alpha$ & Baseline & Baseline acc. (\%) & FedHAD acc. (\%) & $\Delta$Acc (pp) & $p_{W}$ & TFLOPs (HAD/base) & $\Delta$FLOPs \\",
        r"\midrule",
    ]
    for alpha_index, alpha in enumerate(("0.5", "0.01")):
        block = summary.loc[summary.alpha == alpha]
        for _, row in block.iterrows():
            lines.append(
                f"{alpha} & {row['baseline_display']} & "
                f"{row['baseline_acc_mean_pct']:.2f} $\\pm$ {row['baseline_acc_std_pct']:.2f} & "
                f"{row['fedhad_acc_mean_pct']:.2f} $\\pm$ {row['fedhad_acc_std_pct']:.2f} & "
                f"{row['delta_acc_pp']:+.2f} & {format_p(row['wilcoxon_p'])} & "
                f"{row['fedhad_tflops_mean']:.2f}/{row['baseline_tflops_mean']:.2f} & "
                f"{row['delta_flops_pct']:+.2f}\\% \\\\" 
            )
        if alpha_index == 0:
            lines.append(r"\addlinespace")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"{\footnotesize\raggedright \textit{Note.} $p_W$ is the raw two-sided paired Wilcoxon signed-rank $p$-value (SciPy defaults: \texttt{zero\_method=wilcox}, \texttt{correction=False}, \texttt{method=auto}). Positive values of both deltas favor FedHAD. The two $\alpha$ blocks are separate experimental campaigns and were not pooled.\par}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def generate_figure_snippet(alpha_slug: str, metric: str) -> None:
    filename = f"{metric}_convergence_50rounds_alpha{alpha_slug}"
    ylabel = "centralized accuracy" if metric == "accuracy" else "centralized loss"
    text = "\n".join([
        r"% Requires \usepackage{graphicx}.",
        r"\begin{figure}[t]",
        r"\centering",
        rf"\includegraphics[width=\linewidth]{{{filename}.pdf}}",
        rf"\caption{{CIFAR-10 {ylabel} over 50 communication rounds for $\alpha={alpha_slug.replace('_', '.')}$. Curves show the mean and shaded bands show $\pm 1$ sample standard deviation over 30 seeds.}}",
        rf"\label{{fig:long_horizon_{metric}_alpha{alpha_slug}}}",
        r"\end{figure}",
        "",
    ])
    (FIGURES / f"{filename}.tex").write_text(text, encoding="utf-8")


def main() -> None:
    all_rows: list[pd.DataFrame] = []
    sources: dict[str, dict[str, list[Path]]] = {}
    for alpha, campaign in CAMPAIGNS.items():
        frame, paths = read_campaign(alpha, campaign)
        all_rows.append(frame)
        sources[alpha] = paths

    raw = pd.concat(all_rows, ignore_index=True)
    results: list[dict] = []
    for alpha in ("0.5", "0.01"):
        block = raw.loc[raw.alpha == alpha]
        had = block.loc[block.method == "FedHAD", ["seed", "accuracy", "tflops"]]
        had = had.rename(columns={"accuracy": "accuracy_had", "tflops": "tflops_had"})
        for baseline in BASELINES:
            base = block.loc[block.method == baseline, ["seed", "accuracy", "tflops"]]
            base = base.rename(columns={"accuracy": "accuracy_base", "tflops": "tflops_base"})
            paired = had.merge(base, on="seed", validate="one_to_one").sort_values("seed")
            if len(paired) != 30:
                raise RuntimeError(f"{alpha} {baseline}: expected 30 paired seeds, found {len(paired)}")
            differences = 100.0 * (paired.accuracy_had - paired.accuracy_base)
            statistic, p_value = wilcoxon(differences)
            had_tflops = float(paired.tflops_had.mean())
            base_tflops = float(paired.tflops_base.mean())
            results.append({
                "dataset": "CIFAR-10",
                "alpha": alpha,
                "baseline": baseline,
                "baseline_display": DISPLAY[baseline],
                "n": len(paired),
                "fedhad_acc_mean_pct": 100.0 * paired.accuracy_had.mean(),
                "fedhad_acc_std_pct": 100.0 * paired.accuracy_had.std(ddof=1),
                "baseline_acc_mean_pct": 100.0 * paired.accuracy_base.mean(),
                "baseline_acc_std_pct": 100.0 * paired.accuracy_base.std(ddof=1),
                "delta_acc_pp": differences.mean(),
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_p": float(p_value),
                "fedhad_tflops_mean": had_tflops,
                "baseline_tflops_mean": base_tflops,
                "delta_flops_pct": 100.0 * (base_tflops - had_tflops) / base_tflops,
                "fedhad_source_dir": str(source_dir(CAMPAIGNS[alpha], "FedHAD").relative_to(ROOT)),
                "baseline_source_dir": str(source_dir(CAMPAIGNS[alpha], baseline).relative_to(ROOT)),
                "paired_seeds": "42-71",
            })

        figure_data = {method: agregar_convergencia_std(sources[alpha][method]) for method in METHODS}
        alpha_slug = alpha.replace(".", "_")
        fig_convergencia(
            figure_data,
            "acc",
            "Centralized accuracy",
            FIGURES / f"accuracy_convergence_50rounds_alpha{alpha_slug}",
            ylim=(0, 1.02),
            marcar_round=10,
        )
        fig_convergencia(
            figure_data,
            "loss",
            "Centralized loss",
            FIGURES / f"loss_convergence_50rounds_alpha{alpha_slug}",
            marcar_round=10,
        )
        generate_figure_snippet(alpha_slug, "accuracy")
        generate_figure_snippet(alpha_slug, "loss")

    summary = pd.DataFrame(results)
    DATA.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DATA / "long_horizon_alpha_comparison.csv", index=False, float_format="%.10g")

    table_text = generate_table(summary)
    TABLES.mkdir(parents=True, exist_ok=True)
    table_path = TABLES / "tab_long_horizon_alpha_comparison.tex"
    table_path.write_text(table_text, encoding="utf-8")
    shutil.copy2(table_path, OUT / table_path.name)

    print(summary[[
        "alpha", "baseline_display", "n", "fedhad_acc_mean_pct", "baseline_acc_mean_pct",
        "delta_acc_pp", "wilcoxon_p", "fedhad_tflops_mean", "baseline_tflops_mean",
        "delta_flops_pct",
    ]].to_string(index=False))
    print(f"\nTable: {table_path.relative_to(ROOT)}")
    print(f"Figures: {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
