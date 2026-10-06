#!/usr/bin/env python3
"""Independent raw-data audit of the FedHAD versus FedNova article table."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "main_campaign"))
from analisar_resultados import parse_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from femnist_compute_accounting import publication_tflops

RAW_ROOT = ROOT / "results" / "main_campaign_default_alpha_0_5"
TABLES = ROOT / "results" / "revision_consolidated_tables_figures/tables"
MANIFEST = ROOT / "results" / "data_partition_manifest/artifacts/partitions/partition_verification.csv"
SUMMARY = TABLES / "fedhad_vs_fednova_summary.csv"
STATISTICS = TABLES / "fedhad_vs_fednova_statistics.csv"
LATEX = TABLES / "tab_fedhad_vs_fednova.tex"
BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 20260906, 10_000


@dataclass(frozen=True)
class Scenario:
    dataset: str
    raw_dataset: str
    setting: str
    tag: str
    alpha: float | None
    clients: int
    partitioning: str


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


def folder(method: str, s: Scenario) -> Path:
    suffix = f"{s.raw_dataset}-Standard" if method == "FedHAD" else s.raw_dataset
    return RAW_ROOT / f"{method}-results" / s.tag / suffix


def raw_arm(method: str, s: Scenario) -> pd.DataFrame:
    rows = []
    for path in sorted(folder(method, s).glob("*.txt")):
        r = parse_file(path)
        if not r or any(r.get(k) is None for k in ("seed", "acuracia_final", "total_tflops")):
            continue
        if s.alpha is None:
            selected = r.get("alpha_key") == "natural"
        else:
            try:
                selected = np.isclose(float(r.get("alpha")), s.alpha)
            except (TypeError, ValueError):
                selected = False
        if selected:
            reported_tflops = float(r["total_tflops"])
            rows.append({"seed": int(r["seed"]), "accuracy": float(r["acuracia_final"]),
                         "tflops": publication_tflops(
                             reported_tflops,
                             source_campaign="base_alpha_0.5",
                             experiment_tag=s.tag,
                             dataset=s.raw_dataset,
                             method=method,
                             rounds=10,
                         ),
                         "source_file": str(path.relative_to(ROOT)),
                         "parsed_tag": r.get("experiment_tag")})
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No raw reports for {method}, {s.dataset}, {s.setting}")
    return out.sort_values("seed").reset_index(drop=True)


def partitions(manifest: pd.DataFrame, s: Scenario, seeds: set[int]) -> tuple[bool, str]:
    q = manifest[(manifest.dataset == s.raw_dataset) & (manifest.partitioning == s.partitioning)
                 & (manifest.n_clients == s.clients)].copy()
    alpha = pd.to_numeric(q.alpha, errors="coerce")
    q = q[alpha.isna()] if s.alpha is None else q[np.isclose(alpha, s.alpha, equal_nan=False)]
    q = q[q.seed.isin(seeds)]
    ok = (len(q) == len(seeds) and set(q.seed) == seeds and (q.n_methods >= 5).all()
          and q.sha_indices_identical.all() and q.sha_counts_identical.all())
    detail = (f"{len(q)}/{len(seeds)} matching manifest rows; indices/counts identical across >=5 methods"
              if ok else f"{len(q)}/{len(seeds)} matching manifest rows; partition confirmation failed")
    return ok, detail


def bootstrap_ci(d: np.ndarray) -> tuple[float, float]:
    # Same paired percentile-bootstrap convention as paired_stats in analyze_ablation.py.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(BOOTSTRAP_RESAMPLES)])
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


def matches(a: float, b: float) -> bool:
    # Summary CSV uses float_format="%.10g"; allow its final serialized digit.
    return bool(np.isclose(a, b, rtol=0, atol=1e-7))


def latex_has_row(source: str, s: Scenario, x: dict) -> bool:
    setting = s.setting.replace(r"$\alpha=", r"$\alpha = ")
    p = r"$<0.001$" if x["wilcoxon_p"] < 0.001 else f"{x['wilcoxon_p']:.3f}"
    row = (
        f'{s.dataset} & {setting} & {x["fedhad_acc_mean"]:.2f} $\\pm$ {x["fedhad_acc_std"]:.2f} & '
        f'{x["fednova_acc_mean"]:.2f} $\\pm$ {x["fednova_acc_std"]:.2f} & '
        f'{x["mean_diff_pp"]:+.2f} [{x["ci95_low"]:+.2f}, {x["ci95_high"]:+.2f}] & {p} & '
        f'{x["fedhad_tflops_mean"]:.2f} / {x["fednova_tflops_mean"]:.2f} & {x["delta_flops_pct"]:+.1f}\\% \\\\'
    )
    return row in source


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    summary, statistics = pd.read_csv(SUMMARY), pd.read_csv(STATISTICS)
    latex = LATEX.read_text(encoding="utf-8")
    audit_rows, pair_rows = [], []
    report = [
        "# FedHAD versus FedNova audit", "",
        "This independent audit reads raw per-seed reports and does not modify raw data, consolidation code, or the LaTeX table.", "",
        "## Methods", "",
        "Paired differences use d_i = 100 * (FedHAD_i - FedNova_i), in percentage points.",
        f"Bootstrap is a paired percentile bootstrap of d_i: {BOOTSTRAP_RESAMPLES:,} resamples, NumPy RNG seed {BOOTSTRAP_SEED}, matching the paired_stats convention in component_analysis_ablation/analyze_ablation.py.",
        f"Wilcoxon uses SciPy {scipy.__version__}: two-sided, zero_method='wilcox', correction=False, method='auto' (the current pipeline default).", "",
        "## Per-scenario results", "",
    ]
    for s in SCENARIOS:
        had, nova = raw_arm("FedHAD", s), raw_arm("FedNova", s)
        hs, ns = set(had.seed), set(nova.seed)
        paired = sorted(hs & ns)
        complete = hs == ns == set(paired)
        no_mix = (set(had.parsed_tag.dropna()) <= {s.tag} and set(nova.parsed_tag.dropna()) <= {s.tag}
                  and all(s.tag in p for p in list(had.source_file) + list(nova.source_file)))
        partition_ok, partition_detail = partitions(manifest, s, set(paired))
        joined = had.merge(nova, on="seed", suffixes=("_fedhad", "_fednova"), validate="one_to_one").sort_values("seed")
        if not complete or len(joined) != 30:
            raise RuntimeError(f"Pairing failure: {s.dataset} {s.setting}")
        d = 100 * (joined.accuracy_fedhad.to_numpy() - joined.accuracy_fednova.to_numpy())
        ci_low, ci_high = bootstrap_ci(d)
        w = wilcoxon(joined.accuracy_fedhad.to_numpy(), joined.accuracy_fednova.to_numpy(),
                     zero_method="wilcox", correction=False, alternative="two-sided", method="auto")
        hf, nf = joined.tflops_fedhad.to_numpy(), joined.tflops_fednova.to_numpy()
        x = {
            "fedhad_acc_mean": float(100 * joined.accuracy_fedhad.mean()), "fedhad_acc_std": float(100 * joined.accuracy_fedhad.std(ddof=1)),
            "fednova_acc_mean": float(100 * joined.accuracy_fednova.mean()), "fednova_acc_std": float(100 * joined.accuracy_fednova.std(ddof=1)),
            "mean_diff_pp": float(d.mean()), "median_diff_pp": float(np.median(d)), "std_diff_pp": float(d.std(ddof=1)),
            "min_diff_pp": float(d.min()), "max_diff_pp": float(d.max()), "positive_differences": int((d > 0).sum()),
            "negative_differences": int((d < 0).sum()), "zero_differences": int((d == 0).sum()),
            "ci95_low": ci_low, "ci95_high": ci_high, "wilcoxon_statistic": float(w.statistic), "wilcoxon_p": float(w.pvalue),
            "fedhad_tflops_mean": float(hf.mean()), "fedhad_tflops_std": float(hf.std(ddof=1)),
            "fednova_tflops_mean": float(nf.mean()), "fednova_tflops_std": float(nf.std(ddof=1)),
            "delta_flops_pct": float(100 * (nf.mean() - hf.mean()) / nf.mean()),
        }
        sr = summary[(summary.dataset == s.dataset) & (summary.setting == s.setting)].iloc[0]
        tr = statistics[(statistics.dataset == s.dataset) & (statistics.setting == s.setting)].iloc[0]
        summary_match = all(matches(x[k], float(sr[c])) for k, c in (
            ("fedhad_acc_mean", "fedhad_acc_mean"), ("fedhad_acc_std", "fedhad_acc_std"),
            ("fednova_acc_mean", "fednova_acc_mean"), ("fednova_acc_std", "fednova_acc_std"),
            ("mean_diff_pp", "delta_acc_pp"), ("fedhad_tflops_mean", "fedhad_tflops_mean"),
            ("fedhad_tflops_std", "fedhad_tflops_std"), ("fednova_tflops_mean", "fednova_tflops_mean"),
            ("fednova_tflops_std", "fednova_tflops_std"), ("delta_flops_pct", "delta_flops_pct")))
        statistics_match = all(matches(x[k], float(tr[c])) for k, c in (
            ("mean_diff_pp", "mean_paired_difference"), ("ci95_low", "ci95_low"), ("ci95_high", "ci95_high"),
            ("wilcoxon_statistic", "wilcoxon_statistic"), ("wilcoxon_p", "wilcoxon_p")))
        latex_match = latex_has_row(latex, s, x)
        audit_rows.append({
            "dataset": s.dataset, "setting": s.setting,
            "fedhad_raw_path": str(folder("FedHAD", s).relative_to(ROOT)), "fednova_raw_path": str(folder("FedNova", s).relative_to(ROOT)),
            "fedhad_seeds": ",".join(map(str, sorted(hs))), "fednova_seeds": ",".join(map(str, sorted(ns))),
            "paired_seeds": ",".join(map(str, paired)), "n_fedhad": len(had), "n_fednova": len(nova), "n_paired": len(joined),
            "fedhad_duplicate_seeds": int(had.seed.duplicated().sum()), "fednova_duplicate_seeds": int(nova.seed.duplicated().sum()),
            "complete_pairing": complete, "no_experiment_family_mixing": no_mix, "partition_paired": partition_ok,
            "partition_detail": partition_detail, "wilcoxon_zero_method": "wilcox", "wilcoxon_correction": False,
            "wilcoxon_method": "auto", "bootstrap_method": "paired percentile bootstrap",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
            **x, "summary_match": summary_match, "statistics_match": statistics_match, "latex_match": latex_match})
        for r in joined.itertuples(index=False):
            pair_rows.append({"dataset": s.dataset, "setting": s.setting, "seed": int(r.seed),
                              "fedhad_accuracy": r.accuracy_fedhad, "fednova_accuracy": r.accuracy_fednova,
                              "paired_difference_pp": 100 * (r.accuracy_fedhad - r.accuracy_fednova),
                              "fedhad_tflops": r.tflops_fedhad, "fednova_tflops": r.tflops_fednova,
                              "fedhad_source_file": r.source_file_fedhad, "fednova_source_file": r.source_file_fednova})
        report.extend([
            f"### {s.dataset} — {s.setting}", "",
            f"- Raw sources: FedHAD: {folder('FedHAD', s).relative_to(ROOT)}; FedNova: {folder('FedNova', s).relative_to(ROOT)}.",
            f"- Seeds, FedHAD/FedNova/paired: {sorted(hs)} / {sorted(ns)} / {paired}.",
            f"- Duplicates, FedHAD/FedNova: {int(had.seed.duplicated().sum())}/{int(nova.seed.duplicated().sum())}; complete pairing: {complete}; one experimental family: {no_mix}.",
            f"- Partitions: {partition_detail}.",
            f"- Accuracy (%), FedHAD/FedNova: {x['fedhad_acc_mean']:.9f} +/- {x['fedhad_acc_std']:.9f} / {x['fednova_acc_mean']:.9f} +/- {x['fednova_acc_std']:.9f}.",
            f"- d (pp): mean {x['mean_diff_pp']:+.9f}; median {x['median_diff_pp']:+.9f}; SD {x['std_diff_pp']:.9f}; min/max [{x['min_diff_pp']:+.9f}, {x['max_diff_pp']:+.9f}]; positive/negative/zero {x['positive_differences']}/{x['negative_differences']}/{x['zero_differences']}.",
            f"- Wilcoxon: statistic {x['wilcoxon_statistic']:.9g}; p {x['wilcoxon_p']:.12g}; zero_method=wilcox; correction=False; method=auto.",
            f"- Paired bootstrap CI95 (pp): [{x['ci95_low']:+.9f}, {x['ci95_high']:+.9f}]; {BOOTSTRAP_RESAMPLES:,} resamples; seed {BOOTSTRAP_SEED}.",
            f"- FLOPs, FedHAD/FedNova/Delta: {x['fedhad_tflops_mean']:.9f} / {x['fednova_tflops_mean']:.9f} / {x['delta_flops_pct']:+.9f}%.",
            f"- Consolidated summary/statistics/LaTeX match: {summary_match}/{statistics_match}/{latex_match}.", ""])
        if s.dataset == "FashionMNIST" and s.alpha == 0.01:
            f = pd.DataFrame(pair_rows[-30:])[["seed", "paired_difference_pp"]]
            report.extend(["#### FashionMNIST alpha=0.01 paired differences", "",
                           "Ordered by seed; values are unrounded raw-pair calculations in pp.", "",
                           "    " + f.to_string(index=False, float_format=lambda v: f"{v:+.12f}").replace("\n", "\n    "), "",
                           "The bootstrap interval targets the arithmetic paired mean, while Wilcoxon ranks signed absolute differences and tests a location shift. Mixed signs and rank mass can yield a positive mean CI while the two-sided Wilcoxon p-value remains above 0.05. This is a legitimate difference in inferential targets, not a pairing or arithmetic error.", ""])
    audit, pairs = pd.DataFrame(audit_rows), pd.DataFrame(pair_rows)
    audit.to_csv(TABLES / "fedhad_vs_fednova_audit.csv", index=False, float_format="%.12g")
    pairs.to_csv(TABLES / "fedhad_vs_fednova_pairwise_differences.csv", index=False, float_format="%.12g")
    passed = audit[["complete_pairing", "no_experiment_family_mixing", "partition_paired", "summary_match", "statistics_match", "latex_match"]].all().all()
    report.extend(["## Conclusion", "", f"Overall audit status: {'PASS' if passed else 'FAIL'}.",
                   "No values were rounded before mean/SD, paired differences, bootstrap, Wilcoxon, or FLOP calculations.",
                   "The same 30 seed pairs were used for every calculation within each scenario.", ""])
    (TABLES / "fedhad_vs_fednova_audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Audit status: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
