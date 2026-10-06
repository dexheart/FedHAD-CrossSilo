#!/usr/bin/env python3
"""Generate paired FedHAD-versus-FedNova outputs from the raw reports."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "main_campaign"))
from analisar_resultados import parse_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from femnist_compute_accounting import publication_tflops

CAMPAIGN = "base_alpha_0.5"
RAW_ROOT = ROOT / "results" / "main_campaign_default_alpha_0_5"
MASTER = ROOT / "results" / "revision_consolidated_tables_figures/data/master_results_summary.csv"
MANIFEST = ROOT / "results" / "data_partition_manifest/artifacts/partitions/partition_verification.csv"
OUT = ROOT / "results" / "revision_consolidated_tables_figures/tables"


@dataclass(frozen=True)
class Scenario:
    dataset: str
    raw_dataset: str
    setting: str
    tag: str
    alpha: float | None
    clients: int
    partition: str


SCENARIOS = (
    Scenario("MNIST", "MNIST", r"$\alpha=1.0$", "test2_robustez_alpha", 1.0, 5, "dirichlet"),
    Scenario("MNIST", "MNIST", r"$\alpha=0.1$", "test2_robustez_alpha", 0.1, 5, "dirichlet"),
    Scenario("MNIST", "MNIST", r"$\alpha=0.01$", "test2_robustez_alpha", 0.01, 5, "dirichlet"),
    Scenario("FashionMNIST", "FashionMNIST", r"$\alpha=1.0$", "test2_robustez_alpha", 1.0, 5, "dirichlet"),
    Scenario("FashionMNIST", "FashionMNIST", r"$\alpha=0.1$", "test2_robustez_alpha", 0.1, 5, "dirichlet"),
    Scenario("FashionMNIST", "FashionMNIST", r"$\alpha=0.01$", "test2_robustez_alpha", 0.01, 5, "dirichlet"),
    Scenario("CIFAR-10", "CIFAR10", r"$\alpha=1.0$", "test2_robustez_alpha", 1.0, 5, "dirichlet"),
    Scenario("CIFAR-10", "CIFAR10", r"$\alpha=0.1$", "test2_robustez_alpha", 0.1, 5, "dirichlet"),
    Scenario("CIFAR-10", "CIFAR10", r"$\alpha=0.01$", "test2_robustez_alpha", 0.01, 5, "dirichlet"),
    Scenario("FEMNIST", "FEMNIST", "natural federation", "test8_femnist", None, 10, "natural"),
)


def arm_dir(method: str, s: Scenario) -> Path:
    name = f"{s.raw_dataset}-Standard" if method == "FedHAD" else s.raw_dataset
    return RAW_ROOT / f"{method}-results" / s.tag / name


def arm_raw(method: str, s: Scenario) -> pd.DataFrame:
    folder = arm_dir(method, s)
    if not folder.is_dir():
        raise FileNotFoundError(f"Raw folder absent: {folder}")
    records = []
    for path in sorted(folder.glob("*.txt")):
        value = parse_file(path)
        if not value or any(value.get(key) is None for key in ("seed", "acuracia_final", "total_tflops")):
            continue
        if s.alpha is None:
            match = value.get("alpha_key") == "natural"
        else:
            try:
                match = np.isclose(float(value.get("alpha")), s.alpha)
            except (TypeError, ValueError):
                match = False
        if match:
            reported_tflops = float(value["total_tflops"])
            records.append({"seed": int(value["seed"]), "accuracy": float(value["acuracia_final"]),
                            "tflops": publication_tflops(
                                reported_tflops,
                                source_campaign=CAMPAIGN,
                                experiment_tag=s.tag,
                                dataset=s.raw_dataset,
                                method=method,
                                rounds=10,
                            ),
                            "tflops_reported": reported_tflops,
                            "path": str(path.relative_to(ROOT))})
    output = pd.DataFrame(records)
    if output.empty or output.seed.duplicated().any():
        raise RuntimeError(f"Invalid raw reports for {method}, {s.dataset}, {s.setting}")
    return output.sort_values("seed").reset_index(drop=True)


def arm_master(master: pd.DataFrame, method: str, s: Scenario) -> pd.DataFrame:
    q = master[(master.source_campaign == CAMPAIGN) & (master.experiment_tag == s.tag)
               & (master.dataset == s.raw_dataset) & (master.method == method)].copy()
    alpha = pd.to_numeric(q.alpha, errors="coerce")
    q = q[alpha.isna()] if s.alpha is None else q[np.isclose(alpha, s.alpha, equal_nan=False)]
    return q[["seed", "final_accuracy", "tflops", "tflops_reported"]].sort_values("seed").reset_index(drop=True)


def require_master_match(master: pd.DataFrame, raw: pd.DataFrame, method: str, s: Scenario) -> None:
    consolidated = arm_master(master, method, s)
    joined = raw.merge(consolidated, on="seed", how="outer", indicator=True)
    bad = ((joined["_merge"] != "both")
           | ~np.isclose(joined.accuracy, joined.final_accuracy, atol=1e-12, rtol=0)
           | ~np.isclose(joined.tflops_x, joined.tflops_y, atol=1e-12, rtol=0)
           | ~np.isclose(joined.tflops_reported_x, joined.tflops_reported_y,
                        atol=1e-12, rtol=0))
    if bad.any():
        raise RuntimeError(f"RAW/CONSOLIDATED DIVERGENCE for {method}, {s.dataset}, {s.setting}: {joined.loc[bad].to_dict('records')}")


def partition_status(manifest: pd.DataFrame, s: Scenario, seeds: list[int]) -> str:
    q = manifest[(manifest.dataset == s.raw_dataset) & (manifest.partitioning == s.partition)
                 & (manifest.n_clients == s.clients)].copy()
    alpha = pd.to_numeric(q.alpha, errors="coerce")
    q = q[alpha.isna()] if s.alpha is None else q[np.isclose(alpha, s.alpha, equal_nan=False)]
    q = q[q.seed.isin(seeds)]
    checks = ("n_methods", "sha_indices_identical", "sha_counts_identical")
    valid = (len(q) == len(seeds) and set(q.seed) == set(seeds)
             and (q.n_methods >= 5).all() and q.sha_indices_identical.all() and q.sha_counts_identical.all())
    return (f"confirmed: {len(q)}/{len(seeds)} manifest seeds, identical indices/counts across 5 methods"
            if valid else f"UNCONFIRMED: {len(q)}/{len(seeds)} matching manifest rows")


def calculate(s: Scenario, had: pd.DataFrame, nova: pd.DataFrame, manifest: pd.DataFrame) -> tuple[dict, dict, str]:
    had_seeds, nova_seeds = set(had.seed), set(nova.seed)
    paired = sorted(had_seeds & nova_seeds)
    if paired != sorted(had_seeds) or paired != sorted(nova_seeds):
        raise RuntimeError(f"Incomplete pairing: {s.dataset}, {s.setting}")
    joined = had.merge(nova, on="seed", suffixes=("_fedhad", "_fednova")).sort_values("seed")
    diff = 100 * (joined.accuracy_fedhad.to_numpy() - joined.accuracy_fednova.to_numpy())
    rng = np.random.default_rng(20260906)
    sampled = diff[rng.integers(0, len(diff), size=(10_000, len(diff)))].mean(axis=1)
    ci_low, ci_high = np.percentile(sampled, [2.5, 97.5])
    test = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    dz = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) else np.nan
    hf, nf = joined.tflops_fedhad.to_numpy(), joined.tflops_fednova.to_numpy()
    summary = {"dataset": s.dataset, "setting": s.setting, "n": len(joined),
               "fedhad_acc_mean": 100 * joined.accuracy_fedhad.mean(), "fedhad_acc_std": 100 * joined.accuracy_fedhad.std(ddof=1),
               "fednova_acc_mean": 100 * joined.accuracy_fednova.mean(), "fednova_acc_std": 100 * joined.accuracy_fednova.std(ddof=1),
               "delta_acc_pp": float(np.mean(diff)), "fedhad_tflops_mean": float(np.mean(hf)), "fedhad_tflops_std": float(np.std(hf, ddof=1)),
               "fednova_tflops_mean": float(np.mean(nf)), "fednova_tflops_std": float(np.std(nf, ddof=1)),
               "delta_flops_pct": float(100 * (np.mean(nf) - np.mean(hf)) / np.mean(nf))}
    stats = {"dataset": s.dataset, "setting": s.setting, "n": len(joined), "mean_paired_difference": float(np.mean(diff)),
             "ci95_low": float(ci_low), "ci95_high": float(ci_high), "wilcoxon_statistic": float(test.statistic),
             "wilcoxon_p": float(test.pvalue), "paired_effect_size": float(dz)}
    return summary, stats, partition_status(manifest, s, paired)


def write_tex(summary: pd.DataFrame) -> None:
    statistics = pd.read_csv(OUT / "fedhad_vs_fednova_statistics.csv")
    table = summary.merge(statistics, on=["dataset", "setting", "n"], validate="one_to_one")

    def pvalue(value: float) -> str:
        return r"$<0.001$" if value < 0.001 else f"{value:.3f}"

    def signed(value: float) -> str:
        return f"{value:+.2f}"

    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Paired FedHAD versus FedNova comparison across datasets and heterogeneity regimes. Accuracy is mean $\pm$ standard deviation over 30 paired seeds. $\Delta$Acc is FedHAD minus FedNova in percentage points; brackets give the 95\% bootstrap interval of the paired mean. $p_{\mathrm{W}}$ is the two-sided Wilcoxon signed-rank $p$-value. Positive $\Delta$FLOPs denotes training-FLOP savings by FedHAD.}",
             r"\label{tab:fednova_comparison}", r"\scriptsize", r"\setlength{\tabcolsep}{3.2pt}",
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r"\textbf{Dataset} &", r"\textbf{Setting} &", r"\textbf{FedHAD acc.} &", r"\textbf{FedNova acc.} &",
             r"\textbf{$\Delta$Acc (pp) [95\% CI]} &", r"\textbf{$p_{\mathrm{W}}$} &",
             r"\textbf{TFLOPs (HAD / Nova)} &", r"\textbf{$\Delta$FLOPs} \\", r"\midrule"]
    for _, row in table.iterrows():
        setting = row.setting.replace(r"$\alpha=", r"$\alpha = ")
        interval = f'[{signed(row.ci95_low)}, {signed(row.ci95_high)}]'
        lines.append(
            f'{row.dataset} & {setting} & '
            f'{row.fedhad_acc_mean:.2f} $\\pm$ {row.fedhad_acc_std:.2f} & '
            f'{row.fednova_acc_mean:.2f} $\\pm$ {row.fednova_acc_std:.2f} & '
            f'{signed(row.delta_acc_pp)} {interval} & {pvalue(row.wilcoxon_p)} & '
            f'{row.fedhad_tflops_mean:.2f} / {row.fednova_tflops_mean:.2f} & '
            f'{row.delta_flops_pct:+.1f}\\% \\\\'
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (OUT / "tab_fedhad_vs_fednova.tex").write_text("\n".join(lines), encoding="utf-8")


def alpha_half(master: pd.DataFrame) -> list[str]:
    result = []
    for label, raw in (("MNIST", "MNIST"), ("FashionMNIST", "FashionMNIST"), ("CIFAR-10", "CIFAR10")):
        s = Scenario(label, raw, "alpha=0.5", "test1_convergencia", 0.5, 5, "dirichlet")
        had, nova = arm_raw("FedHAD", s), arm_raw("FedNova", s)
        require_master_match(master, had, "FedHAD", s)
        require_master_match(master, nova, "FedNova", s)
        result.append(f"- {label}, alpha=0.5: exists separately in test1_convergencia; {len(set(had.seed) & set(nova.seed))} paired seeds; excluded from the main table.")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    master, manifest = pd.read_csv(MASTER), pd.read_csv(MANIFEST)
    summary_rows, stats_rows = [], []
    audit = ["# FedHAD versus FedNova raw-data audit", "", "Primary raw campaign: main_campaign_default_alpha_0_5. Every raw seed was verified against the consolidated master CSV before output generation.", ""]
    for s in SCENARIOS:
        had, nova = arm_raw("FedHAD", s), arm_raw("FedNova", s)
        require_master_match(master, had, "FedHAD", s)
        require_master_match(master, nova, "FedNova", s)
        summary, stats, partition = calculate(s, had, nova, manifest)
        summary_rows.append(summary); stats_rows.append(stats)
        paired = sorted(set(had.seed) & set(nova.seed))
        details = [f"{s.dataset} | {s.setting}", f"  FedHAD path:  {arm_dir('FedHAD', s).relative_to(ROOT)}",
                   f"  FedNova path: {arm_dir('FedNova', s).relative_to(ROOT)}", f"  FedHAD seeds ({len(had)}): {had.seed.tolist()}",
                   f"  FedNova seeds ({len(nova)}): {nova.seed.tolist()}", f"  paired seeds ({len(paired)}): {paired}",
                   f"  FedHAD accuracy: {summary['fedhad_acc_mean']:.6f} +/- {summary['fedhad_acc_std']:.6f}%",
                   f"  FedNova accuracy: {summary['fednova_acc_mean']:.6f} +/- {summary['fednova_acc_std']:.6f}%",
                   f"  FedHAD TFLOPs: {summary['fedhad_tflops_mean']:.8f} +/- {summary['fedhad_tflops_std']:.8f}",
                   f"  FedNova TFLOPs: {summary['fednova_tflops_mean']:.8f} +/- {summary['fednova_tflops_std']:.8f}",
                   f"  partition: {partition}"]
        print("\n".join([""] + details))
        audit.extend([f"## {s.dataset} — {s.setting}", ""] + [f"- {line.strip()}" for line in details[1:]] + [""])
    summary, stats = pd.DataFrame(summary_rows), pd.DataFrame(stats_rows)
    summary.to_csv(OUT / "fedhad_vs_fednova_summary.csv", index=False, float_format="%.10g")
    stats.to_csv(OUT / "fedhad_vs_fednova_statistics.csv", index=False, float_format="%.10g")
    write_tex(summary)
    base = alpha_half(master)
    print("\nSeparate alpha=0.5 base configuration (excluded from main table):\n" + "\n".join(base))
    audit.extend(["## Separate alpha=0.5 base configuration", ""] + base + ["", "## Statistical methods", "", "Differences are FedHAD minus FedNova in percentage points. The 95% interval is a 10,000-resample percentile bootstrap of the paired mean, NumPy RNG seed 20260906, matching the ablation pipeline. Wilcoxon is two-sided paired signed-rank with zero_method='wilcox'; paired effect size is dz = mean paired difference / sample standard deviation.", ""])
    (OUT / "fedhad_vs_fednova_audit.md").write_text("\n".join(audit), encoding="utf-8")
    for file in ("fedhad_vs_fednova_summary.csv", "fedhad_vs_fednova_statistics.csv", "tab_fedhad_vs_fednova.tex", "fedhad_vs_fednova_audit.md"):
        print(f"Wrote {OUT.relative_to(ROOT) / file}")


if __name__ == "__main__":
    main()
