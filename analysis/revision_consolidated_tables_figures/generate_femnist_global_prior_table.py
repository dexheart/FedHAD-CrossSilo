#!/usr/bin/env python3
"""Create the article table summarising FEMNIST's empirical global class prior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRIOR_TABLES = ROOT / "results" / "global_prior_analysis" / "tables"
OUT_TABLE = ROOT / "results" / "revision_consolidated_tables_figures" / "tables" / "tabG_femnist_global_prior.tex"
OUT_DATA = ROOT / "results" / "revision_consolidated_tables_figures" / "data" / "blockG_femnist_global_prior_table.csv"


def main() -> None:
    prior = pd.read_csv(PRIOR_TABLES / "T3_femnist_prior.csv")
    ranking = pd.read_csv(PRIOR_TABLES / "T4_femnist_ranking_disagreement.csv").iloc[0]
    sensitivity = pd.read_csv(PRIOR_TABLES / "T6_normalisation_sensitivity.csv")
    quantile = sensitivity.loc[sensitivity.normalisation == "quantile_matching"].iloc[0]

    if len(prior) != 62:
        raise RuntimeError(f"Expected 62 FEMNIST classes, found {len(prior)}")
    if not np.isclose(prior.empirical_prior.sum(), 1.0, atol=1e-12):
        raise RuntimeError("FEMNIST empirical prior does not sum to one")
    if not np.allclose(prior.uniform_prior, 1.0 / 62.0):
        raise RuntimeError("Uniform reference is inconsistent with 62 classes")

    total_examples = int(prior["count"].sum())
    min_prior = float(prior.empirical_prior.min())
    max_prior = float(prior.empirical_prior.max())
    ratio = max_prior / min_prior
    max_deviation = float(np.max(np.abs(prior.empirical_prior - prior.uniform_prior)))

    checks = {
        "prior_max_min_ratio": ratio,
        "prior_max_dev_from_uniform": max_deviation,
    }
    for key, calculated in checks.items():
        if not np.isclose(calculated, float(ranking[key]), atol=1e-12):
            raise RuntimeError(f"Recalculated {key} disagrees with T4: {calculated} vs {ranking[key]}")
    if not bool(quantile.budget_preserved):
        raise RuntimeError("The selected quantile matching row does not preserve the epoch budget")

    audit = pd.DataFrame([{
        "n_classes": len(prior),
        "n_writers": int(ranking.n_writers),
        "n_examples": total_examples,
        "uniform_prior": 1.0 / 62.0,
        "empirical_prior_min": min_prior,
        "empirical_prior_max": max_prior,
        "prior_max_min_ratio": ratio,
        "prior_max_deviation_from_uniform": max_deviation,
        "js_global_vs_uniform_bits": float(ranking.js_global_vs_uniform),
        "spearman_cv_vs_js": float(ranking.spearman_cv_vs_js),
        "writers_tercile_changed": int(ranking.writers_band_changed),
        "writers_tercile_changed_pct": float(ranking.pct_band_changed),
        "writers_epochs_changed": int(ranking.writers_epochs_changed),
        "writers_epochs_changed_pct": float(ranking.pct_epochs_changed),
        "epoch_budget_cv": int(quantile.total_epochs_cv),
        "epoch_budget_js_quantile": int(quantile.total_epochs_variant),
    }])
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_DATA, index=False, float_format="%.10g")

    tex = "\n".join([
        r"% Generated from global_prior_analysis/tables/T3--T6 by generate_femnist_global_prior_table.py.",
        r"% Requires \usepackage{booktabs}.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Empirical global class prior and its operational implications on FEMNIST. The prior is aggregated over 2{,}095 distinct writers and compared with the uniform reference implicitly assumed by the normalised coefficient of variation used as $H_k$.}",
        r"\label{tab:femnist-global-prior}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        r"\multicolumn{2}{l}{\emph{Global class prior}} \\",
        f"Classes / examples & {len(prior)} / {total_examples:,}".replace(",", r"{,}") + r" \\",
        f"Uniform probability per class & {100 / 62:.3f}\\%" + r" \\",
        f"Empirical class probability, min--max & {100 * min_prior:.3f}--{100 * max_prior:.3f}\\%" + r" \\",
        f"Most/least frequent-class ratio & {ratio:.1f}$\\times$" + r" \\",
        f"Maximum deviation from uniform & {100 * max_deviation:.3f} pp" + r" \\",
        f"JS(global $\\parallel$ uniform), bits & {float(ranking.js_global_vs_uniform):.3f}" + r" \\",
        r"\addlinespace",
        r"\multicolumn{2}{l}{\emph{Metric disagreement over writers}} \\",
        f"Spearman $\\rho$, CV vs. JS ranking & {float(ranking.spearman_cv_vs_js):.3f}" + r" \\",
        f"Writers changing heterogeneity tercile & {int(ranking.writers_band_changed)} / {int(ranking.n_writers)} ({float(ranking.pct_band_changed):.1f}\\%)" + r" \\",
        f"Writers changing local-epoch allocation & {int(ranking.writers_epochs_changed)} / {int(ranking.n_writers)} ({float(ranking.pct_epochs_changed):.1f}\\%)" + r" \\",
        f"Total epoch budget, CV / remapped JS & {int(quantile.total_epochs_cv)} / {int(quantile.total_epochs_variant)}" + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"{\footnotesize\raggedright \textit{Note.} JS denotes Jensen--Shannon divergence from the empirical global prior. The JS scores were quantile matched to the CV scores, preserving the aggregate epoch budget; this is an offline counterfactual remapping rather than a trained FedHAD variant. FEMNIST has no client-drift telemetry, so these results establish a scope boundary for the uniform-prior assumption but do not compare the metrics' ability to predict drift or improve accuracy.\par}",
        r"\end{table}",
        "",
    ])
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text(tex, encoding="utf-8")

    print(audit.to_string(index=False))
    print(f"\nWrote {OUT_TABLE.relative_to(ROOT)}")
    print(f"Wrote {OUT_DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
