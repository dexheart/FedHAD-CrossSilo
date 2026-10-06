#!/usr/bin/env python3
"""Create the compact raw-data FedHAD versus all-baselines LaTeX table."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "main_campaign"))
from analisar_resultados import parse_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from femnist_compute_accounting import publication_tflops

RAW = ROOT / "results" / "main_campaign_default_alpha_0_5"
MANIFEST = ROOT / "results" / "data_partition_manifest/artifacts/partitions/partition_verification.csv"
OUT = ROOT / "results" / "revision_consolidated_tables_figures/tables/tab_fedhad_vs_all_baselines_compact.tex"
METHODS = ("FedHAD", "FedAVG", "FedAvgM", "FedProx", "FedNova")


@dataclass(frozen=True)
class Scenario:
    label: str
    dataset: str
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
    Scenario("FEMNIST", "FEMNIST", "natural", "test8_femnist", None, 10, "natural"),
)


def folder(method: str, s: Scenario) -> Path:
    # FedHAD uses the Standard directory suffix; every baseline uses plain dataset names.
    suffix = f"{s.dataset}-Standard" if method == "FedHAD" else s.dataset
    return RAW / f"{method}-results" / s.tag / suffix


def load(method: str, s: Scenario) -> pd.DataFrame:
    rows = []
    for path in sorted(folder(method, s).glob("*.txt")):
        report = parse_file(path)
        if not report or any(report.get(key) is None for key in ("seed", "acuracia_final", "total_tflops")):
            continue
        if s.alpha is None:
            matches = report.get("alpha_key") == "natural"
        else:
            try:
                matches = np.isclose(float(report.get("alpha")), s.alpha)
            except (TypeError, ValueError):
                matches = False
        if matches:
            reported = float(report["total_tflops"])
            tflops = publication_tflops(
                reported,
                source_campaign="base_alpha_0.5",
                experiment_tag=s.tag,
                dataset=s.dataset,
                method=method,
                rounds=10,
            )
            rows.append((int(report["seed"]), float(report["acuracia_final"]), tflops))
    result = pd.DataFrame(rows, columns=["seed", "accuracy", "tflops"])
    if len(result) != 30 or result.seed.duplicated().any():
        raise RuntimeError(f"Expected 30 unique raw reports for {method}, {s.label}, {s.setting}; found {len(result)}.")
    return result.sort_values("seed").reset_index(drop=True)


def require_partitions(manifest: pd.DataFrame, s: Scenario, seeds: set[int]) -> None:
    rows = manifest[(manifest.dataset == s.dataset) & (manifest.partitioning == s.partition)
                    & (manifest.n_clients == s.clients)].copy()
    alpha = pd.to_numeric(rows.alpha, errors="coerce")
    rows = rows[alpha.isna()] if s.alpha is None else rows[np.isclose(alpha, s.alpha, equal_nan=False)]
    rows = rows[rows.seed.isin(seeds)]
    valid = (len(rows) == 30 and set(rows.seed) == seeds and (rows.n_methods >= 5).all()
             and rows.sha_indices_identical.all() and rows.sha_counts_identical.all())
    if not valid:
        raise RuntimeError(f"Partition pairing is not confirmed for {s.label}, {s.setting}.")


def cell(frame: pd.DataFrame, bold: bool = False) -> str:
    value = (
        rf"\shortstack{{{100 * frame.accuracy.mean():.2f} $\pm$ {100 * frame.accuracy.std(ddof=1):.2f} \\"
        rf"{frame.tflops.mean():.2f}}}"
    )
    return rf"\textbf{{{value}}}" if bold else value


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{FedHAD and baselines across datasets and heterogeneity regimes. Each cell reports final accuracy in percent (top: mean $\pm$ standard deviation over 30 paired seeds) and mean training TFLOPs (bottom). FedHAD is bolded. All methods use identical client partitions for each seed.}",
        r"\label{tab:fedhad_all_baselines_compact}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Setting} & \textbf{FedHAD} & \textbf{FedAvg} & \textbf{FedAvgM} & \textbf{FedProx} & \textbf{FedNova} \\",
        r"\midrule",
    ]
    for i, scenario in enumerate(SCENARIOS):
        frames = {method: load(method, scenario) for method in METHODS}
        seeds = set(frames["FedHAD"].seed)
        if any(set(frame.seed) != seeds for frame in frames.values()):
            raise RuntimeError(f"Seed sets are not fully paired for {scenario.label}, {scenario.setting}.")
        require_partitions(manifest, scenario, seeds)
        setting = scenario.setting.replace(r"$\alpha=", r"$\alpha = ")
        lines.append(" & ".join([scenario.label, setting] + [cell(frames[m], m == "FedHAD") for m in METHODS]) + r" \\")
        if i in (2, 5, 8):
            lines.append(r"\midrule")
        print(f"{scenario.label} {scenario.setting}: 30 fully paired seeds; partitions confirmed.")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
