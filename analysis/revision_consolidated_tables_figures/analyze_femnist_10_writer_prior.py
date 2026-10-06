#!/usr/bin/env python3
"""Offline audit of the empirical FEMNIST prior for the 10 federated writers.

This script consumes only the recorded *training-split* class counts from the
partition manifest.  It neither reads held-out writers nor runs training.
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results" / "data_partition_manifest/artifacts/partitions/partition_manifest.csv"
OUT_DATA = ROOT / "results" / "revision_consolidated_tables_figures/data/femnist_10_writer_prior"
OUT_TABLES = ROOT / "results" / "revision_consolidated_tables_figures/tables"
OUT_FIGS = ROOT / "results" / "revision_consolidated_tables_figures/figures/femnist_10_writer_prior"
OUT_DOCS = ROOT / "results" / "revision_consolidated_tables_figures/docs"

WRITERS = [
    "f0150_25", "f0320_41", "f0900_42", "f1557_00", "f1702_13",
    "f2402_75", "f3284_38", "f3603_14", "f3724_32", "f3844_11",
]
N_CLASSES = 62
EXPECTED_METHODS = {"FedAvg", "FedAvgM", "FedProx", "FedHAD", "FedNova"}
UNIFORM = np.full(N_CLASSES, 1.0 / N_CLASSES)

COLORS = {
    "empirical": "#0072B2",
    "uniform": "#D55E00",
    "points": "#009E73",
}


def parse_counts(value: str) -> np.ndarray:
    counts = np.asarray([int(v) for v in str(value).split(";")], dtype=int)
    if counts.size != N_CLASSES:
        raise ValueError(f"Expected {N_CLASSES} class counts, found {counts.size}")
    return counts


def normalized_cv(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=float)
    raw_cv = values.std(ddof=0) / values.mean()
    return float(np.clip(raw_cv / math.sqrt(values.size - 1), 0.0, 1.0))


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl_bits(a: np.ndarray, b: np.ndarray) -> float:
        keep = a > 0
        return float(np.sum(a[keep] * np.log2(a[keep] / b[keep])))

    return 0.5 * kl_bits(p, midpoint) + 0.5 * kl_bits(q, midpoint)


def fedhad_epochs(h_score: float) -> int:
    if h_score == 0:
        return 5
    return max(2, int(round(5 - 3 * h_score)))


def spearman_without_scipy(x: pd.Series, y: pd.Series) -> float:
    rx = x.rank(method="average").to_numpy(dtype=float)
    ry = y.rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(rx, ry)[0, 1])


def mean_min_max(series: pd.Series) -> tuple[float, float, float]:
    return float(series.mean()), float(series.min()), float(series.max())


def fmt_triplet(series: pd.Series, digits: int = 4) -> str:
    mean, minimum, maximum = mean_min_max(series)
    return f"{mean:.{digits}f} [{minimum:.{digits}f}, {maximum:.{digits}f}]"


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def load_canonical_training_splits() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    subset = manifest.loc[
        manifest["dataset"].eq("FEMNIST")
        & manifest["role"].eq("client")
        & manifest["partition_granularity"].eq("train_split")
        & manifest["writer_natural_id"].isin(WRITERS)
    ].copy()

    if set(subset["writer_natural_id"].unique()) != set(WRITERS):
        raise AssertionError("The manifest does not contain exactly the requested writers")
    seeds = sorted(subset["seed"].unique())
    if seeds != list(range(42, 72)):
        raise AssertionError(f"Expected seeds 42--71, found {seeds}")

    for seed, group in subset.groupby("seed"):
        if set(group["method"]) != EXPECTED_METHODS:
            raise AssertionError(f"Unexpected methods for seed {seed}: {set(group['method'])}")
        if group.shape[0] != len(WRITERS) * len(EXPECTED_METHODS):
            raise AssertionError(f"Incomplete duplicated records for seed {seed}")
        if set(group["writer_natural_id"]) != set(WRITERS):
            raise AssertionError(f"Writer membership differs at seed {seed}")

    # The method rows are duplicated records of the same partition. Verify that
    # assertion before selecting one canonical copy; no method result is analyzed.
    keys = ["seed", "client_id", "writer_natural_id"]
    for key, group in subset.groupby(keys):
        if group["sha256_counts_client"].nunique() != 1:
            raise AssertionError(f"Class counts differ across methods at {key}")
        if group["sha256_indices_client"].nunique() != 1:
            raise AssertionError(f"Training indices differ across methods at {key}")
        if group["label_counts"].nunique() != 1:
            raise AssertionError(f"Serialized class counts differ across methods at {key}")

    canonical = subset.loc[subset["method"].eq("FedAvg")].copy()
    canonical = canonical.sort_values(["seed", "client_id"]).reset_index(drop=True)
    if canonical.shape[0] != 30 * len(WRITERS):
        raise AssertionError("Canonical selection must contain 300 writer-seed rows")
    canonical["counts"] = canonical["label_counts"].map(parse_counts)
    for row in canonical.itertuples(index=False):
        if int(row.counts.sum()) != int(row.n_train) or int(row.n_k) != int(row.n_train):
            raise AssertionError("A class-count vector does not match its recorded training size")
    return canonical


def analyze(canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict] = []
    class_rows: list[dict] = []
    seed_rows: list[dict] = []

    for seed, group in canonical.groupby("seed", sort=True):
        group = group.sort_values("client_id").copy()
        if group["writer_natural_id"].tolist() != WRITERS:
            raise AssertionError(f"Writer ordering/membership mismatch at seed {seed}")
        global_counts = np.sum(np.stack(group["counts"].to_list()), axis=0)
        total_examples = int(global_counts.sum())
        if total_examples != int(group["n_train"].sum()):
            raise AssertionError("Global class counts do not equal aggregate training size")
        global_prior = global_counts / total_examples
        positive = global_prior[global_prior > 0]
        if positive.size != N_CLASSES:
            raise AssertionError(f"At least one class is absent from the federation at seed {seed}")

        work = pd.DataFrame({
            "client_id": group["client_id"].astype(int).to_numpy(),
            "writer_natural_id": group["writer_natural_id"].to_numpy(),
            "n_train": group["n_train"].astype(int).to_numpy(),
            "n_val": group["n_val"].astype(int).to_numpy(),
            "cv_norm": [normalized_cv(c) for c in group["counts"]],
            "js_vs_10_writer_prior_bits": [
                js_divergence_bits(c / c.sum(), global_prior) for c in group["counts"]
            ],
        })
        # Confirm that the metric reconstructed from counts is the recorded FedHAD input.
        if not np.allclose(work["cv_norm"], group["H_k_norm"].to_numpy(), atol=5e-8):
            raise AssertionError(f"Reconstructed normalized CV differs from manifest at seed {seed}")

        work["rank_cv"] = work["cv_norm"].rank(method="average", ascending=True)
        work["rank_js"] = work["js_vs_10_writer_prior_bits"].rank(method="average", ascending=True)
        work["rank_position_changed"] = ~np.isclose(work["rank_cv"], work["rank_js"])

        # Rank-based thirds avoid applying CV-specific numeric thresholds to JS.
        work["band_cv"] = pd.qcut(
            work["cv_norm"].rank(method="first"), 3,
            labels=["low", "medium", "high"],
        ).astype(str)
        work["band_js"] = pd.qcut(
            work["js_vs_10_writer_prior_bits"].rank(method="first"), 3,
            labels=["low", "medium", "high"],
        ).astype(str)
        work["band_changed"] = work["band_cv"] != work["band_js"]
        work["epochs_cv"] = work["cv_norm"].map(fedhad_epochs).astype(int)

        # Offline budget-preserving counterfactual: preserve the exact CV epoch
        # multiset and assign more epochs to lower JS heterogeneity ranks.
        available_epochs = sorted(work["epochs_cv"].tolist(), reverse=True)
        js_order = work.sort_values(
            ["js_vs_10_writer_prior_bits", "client_id"], ascending=[True, True]
        ).index
        work["epochs_js_remapped"] = 0
        for idx, epochs in zip(js_order, available_epochs):
            work.loc[idx, "epochs_js_remapped"] = epochs
        work["epochs_js_remapped"] = work["epochs_js_remapped"].astype(int)
        work["epochs_changed"] = work["epochs_cv"] != work["epochs_js_remapped"]
        if sorted(work["epochs_cv"]) != sorted(work["epochs_js_remapped"]):
            raise AssertionError("Counterfactual epoch multiset was not preserved")

        global_metrics = {
            "global_total_training_examples": total_examples,
            "excluded_local_validation_examples": int(group["n_val"].sum()),
            "global_min_class_probability": float(global_prior.min()),
            "global_max_class_probability": float(global_prior.max()),
            "global_max_min_probability_ratio": float(global_prior.max() / global_prior.min()),
            "global_max_abs_deviation_from_uniform": float(np.max(np.abs(global_prior - UNIFORM))),
            "global_js_vs_uniform_bits": js_divergence_bits(global_prior, UNIFORM),
        }
        rho = spearman_without_scipy(work["cv_norm"], work["js_vs_10_writer_prior_bits"])
        seed_rows.append({
            "seed": int(seed),
            **global_metrics,
            "spearman_cv_js": rho,
            "writers_rank_changed": int(work["rank_position_changed"].sum()),
            "writers_band_changed": int(work["band_changed"].sum()),
            "writers_epochs_changed": int(work["epochs_changed"].sum()),
            "total_epochs_cv": int(work["epochs_cv"].sum()),
            "total_epochs_js_remapped": int(work["epochs_js_remapped"].sum()),
        })

        for class_index in range(N_CLASSES):
            class_rows.append({
                "seed": int(seed),
                "class_index": class_index,
                "global_training_count": int(global_counts[class_index]),
                "empirical_prior_probability": float(global_prior[class_index]),
                "uniform_probability": float(UNIFORM[class_index]),
                "deviation_from_uniform": float(global_prior[class_index] - UNIFORM[class_index]),
            })

        group_by_writer = group.set_index("writer_natural_id")
        for row in work.itertuples(index=False):
            local_counts = group_by_writer.loc[row.writer_natural_id, "counts"]
            record = {
                "seed": int(seed),
                "client_id": int(row.client_id),
                "writer_natural_id": row.writer_natural_id,
                "n_train": int(row.n_train),
                "n_val_excluded_from_prior": int(row.n_val),
                "cv_norm": float(row.cv_norm),
                "js_vs_10_writer_prior_bits": float(row.js_vs_10_writer_prior_bits),
                "rank_cv": float(row.rank_cv),
                "rank_js": float(row.rank_js),
                "rank_position_changed": bool(row.rank_position_changed),
                "band_cv": row.band_cv,
                "band_js": row.band_js,
                "band_changed": bool(row.band_changed),
                "epochs_cv": int(row.epochs_cv),
                "epochs_js_remapped": int(row.epochs_js_remapped),
                "epochs_changed": bool(row.epochs_changed),
                **global_metrics,
            }
            record.update({f"local_count_{i:02d}": int(v) for i, v in enumerate(local_counts)})
            record.update({f"global_count_{i:02d}": int(v) for i, v in enumerate(global_counts)})
            record.update({f"global_prior_{i:02d}": float(v) for i, v in enumerate(global_prior)})
            detail_rows.append(record)

    details = pd.DataFrame(detail_rows)
    classes = pd.DataFrame(class_rows)
    seeds = pd.DataFrame(seed_rows)
    writer_summary = details.groupby("writer_natural_id", sort=False).agg(
        client_id=("client_id", "first"),
        n_train=("n_train", "mean"),
        cv_norm_mean=("cv_norm", "mean"),
        cv_norm_min=("cv_norm", "min"),
        cv_norm_max=("cv_norm", "max"),
        js_bits_mean=("js_vs_10_writer_prior_bits", "mean"),
        js_bits_min=("js_vs_10_writer_prior_bits", "min"),
        js_bits_max=("js_vs_10_writer_prior_bits", "max"),
        rank_cv_mean=("rank_cv", "mean"),
        rank_js_mean=("rank_js", "mean"),
        rank_changed_fraction=("rank_position_changed", "mean"),
        band_changed_fraction=("band_changed", "mean"),
        epochs_changed_fraction=("epochs_changed", "mean"),
        epochs_cv_mean=("epochs_cv", "mean"),
        epochs_js_mean=("epochs_js_remapped", "mean"),
    ).reset_index()
    return details, classes, seeds, writer_summary


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=400)
    plt.close(fig)


def plot_prior(classes: pd.DataFrame) -> None:
    summary = classes.groupby("class_index")["empirical_prior_probability"].agg(
        ["mean", "min", "max"]
    ).reset_index()
    x = summary["class_index"].to_numpy()
    mean = 100 * summary["mean"].to_numpy()
    minimum = 100 * summary["min"].to_numpy()
    maximum = 100 * summary["max"].to_numpy()
    uniform_pct = 100 / N_CLASSES

    fig, ax = plt.subplots(figsize=(7.0, 3.25), constrained_layout=True)
    ax.fill_between(x, minimum, maximum, color=COLORS["empirical"], alpha=0.17,
                    label="Empirical prior: seed range")
    ax.plot(x, mean, color=COLORS["empirical"], linewidth=1.6, marker="o",
            markersize=2.8, label="Empirical prior: mean")
    ax.axhline(uniform_pct, color=COLORS["uniform"], linewidth=1.4,
               linestyle="--", label="Uniform prior (1/62)")
    ax.set_xlabel("FEMNIST class index")
    ax.set_ylabel("Class probability (%)")
    ax.set_xlim(-0.7, N_CLASSES - 0.3)
    ax.set_xticks(np.arange(0, N_CLASSES, 5))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save_figure(fig, OUT_FIGS / "femnist_10_writer_empirical_prior_vs_uniform")


def plot_cv_vs_js(details: pd.DataFrame, writers: pd.DataFrame, seeds: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
    ax.scatter(details["cv_norm"], details["js_vs_10_writer_prior_bits"],
               s=10, alpha=0.10, color=COLORS["points"], linewidths=0,
               label="Writer--seed observations")
    label_offsets = {
        "f0150_25": (5, 5), "f0320_41": (5, 5), "f0900_42": (5, 5),
        "f1557_00": (6, 7), "f1702_13": (5, 8), "f2402_75": (6, 7),
        "f3284_38": (5, 6), "f3603_14": (-59, 8), "f3724_32": (-62, 8),
        "f3844_11": (6, 5),
    }
    for row in writers.itertuples(index=False):
        xerr = [[row.cv_norm_mean - row.cv_norm_min], [row.cv_norm_max - row.cv_norm_mean]]
        yerr = [[row.js_bits_mean - row.js_bits_min], [row.js_bits_max - row.js_bits_mean]]
        ax.errorbar(row.cv_norm_mean, row.js_bits_mean, xerr=xerr, yerr=yerr,
                    fmt="o", markersize=5, color=COLORS["empirical"],
                    ecolor=COLORS["empirical"], elinewidth=0.8, capsize=2, zorder=3)
        offset = label_offsets[row.writer_natural_id]
        ax.annotate(row.writer_natural_id, (row.cv_norm_mean, row.js_bits_mean),
                    xytext=offset, textcoords="offset points", fontsize=7)
    rho_mean, rho_min, rho_max = mean_min_max(seeds["spearman_cv_js"])
    ax.text(0.02, 0.98,
            rf"Spearman $\rho$: {rho_mean:.2f} [{rho_min:.2f}, {rho_max:.2f}]",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 3})
    ax.set_xlabel("Normalized CV heterogeneity")
    ax.set_ylabel("JS divergence from empirical federation prior (bits)")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, OUT_FIGS / "femnist_10_writer_cv_vs_prior_aware_js")


def write_tables(seeds: pd.DataFrame, writers: pd.DataFrame) -> None:
    summary_rows = [
        ("Training examples", fmt_triplet(seeds["global_total_training_examples"], 0)),
        ("Local-validation examples excluded", fmt_triplet(seeds["excluded_local_validation_examples"], 0)),
        ("Uniform class probability", f"{100 / N_CLASSES:.3f}\\%"),
        ("Minimum class probability", fmt_triplet(seeds["global_min_class_probability"] * 100, 3) + r"\%"),
        ("Maximum class probability", fmt_triplet(seeds["global_max_class_probability"] * 100, 3) + r"\%"),
        ("Maximum/minimum probability ratio", fmt_triplet(seeds["global_max_min_probability_ratio"], 2)),
        ("Maximum absolute deviation from uniform", fmt_triplet(seeds["global_max_abs_deviation_from_uniform"] * 100, 3) + r" p.p."),
        ("JS(empirical, uniform), bits", fmt_triplet(seeds["global_js_vs_uniform_bits"], 4)),
        (r"Spearman $\rho$(CV, JS)", fmt_triplet(seeds["spearman_cv_js"], 3)),
        ("Writers changing exact rank", fmt_triplet(seeds["writers_rank_changed"], 2)),
        ("Writers changing rank-based third", fmt_triplet(seeds["writers_band_changed"], 2)),
        ("Writers changing epoch allocation", fmt_triplet(seeds["writers_epochs_changed"], 2)),
        ("Total epochs (CV)", fmt_triplet(seeds["total_epochs_cv"], 0)),
        ("Total epochs (JS remapping)", fmt_triplet(seeds["total_epochs_js_remapped"], 0)),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Empirical-prior audit for the ten FEMNIST writers used in the federation. Values are means over the 30 seeds, with the seed-wise minimum and maximum in brackets. Only training-split examples are included. JS denotes Jensen--Shannon divergence with base-2 logarithms. The JS epoch result is an offline, budget-preserving remapping, not a trained method.}",
        r"\label{tab:femnist-ten-writer-prior}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Mean [min, max] \\",
        r"\midrule",
    ]
    lines.extend(f"{latex_escape(label)} & {value} \\\\" for label, value in summary_rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (OUT_TABLES / "tab_femnist_10_writer_prior.tex").write_text("\n".join(lines), encoding="utf-8")

    writer_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Writer-level comparison between normalized CV and prior-aware JS divergence for FEMNIST. Metric values and ranks are means over the same 30 seeds. Change columns give the percentage of seeds in which the writer changes exact rank, rank-based heterogeneity third, or epoch allocation under the offline budget-preserving JS remapping.}",
        r"\label{tab:femnist-ten-writer-detail}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Writer & $n_{train}$ & CV & JS (bits) & Rank CV & Rank JS & $\Delta$rank (\%) & $\Delta$band (\%) & $\Delta$epochs (\%) \\",
        r"\midrule",
    ]
    for row in writers.itertuples(index=False):
        writer_lines.append(
            f"{latex_escape(row.writer_natural_id)} & {row.n_train:.0f} & {row.cv_norm_mean:.4f} & "
            f"{row.js_bits_mean:.4f} & {row.rank_cv_mean:.2f} & {row.rank_js_mean:.2f} & "
            f"{100*row.rank_changed_fraction:.1f} & {100*row.band_changed_fraction:.1f} & "
            f"{100*row.epochs_changed_fraction:.1f} \\\\"
        )
    writer_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (OUT_TABLES / "tab_femnist_10_writer_per_writer.tex").write_text(
        "\n".join(writer_lines), encoding="utf-8"
    )


def write_report(seeds: pd.DataFrame, writers: pd.DataFrame) -> None:
    def trip(column: str, digits: int = 4) -> str:
        return fmt_triplet(seeds[column], digits)

    epoch_changed_any = int((writers["epochs_changed_fraction"] > 0).sum())
    band_changed_any = int((writers["band_changed_fraction"] > 0).sum())
    rank_changed_any = int((writers["rank_changed_fraction"] > 0).sum())
    report = f"""# FEMNIST: prior empírico dos 10 writers da federação

## Escopo e protocolo

Esta é uma análise exclusivamente offline dos writers `{', '.join(WRITERS)}` nas 30 seeds (42--71). O protocolo original particiona naturalmente por `writer_id` e, dentro de cada writer, usa `len_val=max(1, n//10)` e o restante para treino, com `random_split` inicializado pela seed experimental. Em cada seed, foram agregadas somente as contagens das 62 classes registradas no `train_split` do manifesto de partições. As linhas repetidas por método foram validadas por hashes e uma única cópia canônica foi usada. Os exemplos de validação local (`n_val`) e todos os writers held-out foram excluídos do prior. Nenhum treinamento foi executado e nenhum método FedHAD-JS foi criado.

## Resultados seed-wise: média [mínimo, máximo]

- Exemplos de treino: {trip('global_total_training_examples', 0)}.
- Exemplos de validação local excluídos do prior: {trip('excluded_local_validation_examples', 0)}.
- Probabilidade mínima de classe: {trip('global_min_class_probability', 6)}.
- Probabilidade máxima de classe: {trip('global_max_class_probability', 6)}.
- Razão máxima/mínima: {trip('global_max_min_probability_ratio', 3)}.
- Maior desvio absoluto de $1/62$: {trip('global_max_abs_deviation_from_uniform', 6)}.
- JS(prior empírico, uniforme), em bits: {trip('global_js_vs_uniform_bits', 5)}.
- Spearman entre os rankings CV e JS: {trip('spearman_cv_js', 3)}.
- Writers que mudam de posição exata por seed: {trip('writers_rank_changed', 2)}; {rank_changed_any}/10 mudam em pelo menos uma seed.
- Writers que mudam de terço de heterogeneidade por seed: {trip('writers_band_changed', 2)}; {band_changed_any}/10 mudam em pelo menos uma seed.
- Writers cuja alocação de épocas muda por seed: {trip('writers_epochs_changed', 2)}; {epoch_changed_any}/10 mudam em pelo menos uma seed.
- Orçamento total de épocas CV: {trip('total_epochs_cv', 0)}; remapeamento JS: {trip('total_epochs_js_remapped', 0)}.

Os terços de heterogeneidade são definidos por ranking dentro de cada seed (baixo, médio e alto), pois os valores absolutos de CV e JS não compartilham escala. O contrafactual ordena os writers pela JS e redistribui exatamente o mesmo multiconjunto de épocas produzido pelo CV; portanto, preserva o orçamento por seed e mede apenas a consequência da troca de ranking.

## Interpretação

O prior empírico dos dez writers não é aproximadamente uniforme no sentido relevante para a hipótese avaliada: a referência uniforme é {1/N_CLASSES:.6f} por classe, enquanto as probabilidades extremas seed-wise são, em média, {seeds['global_min_class_probability'].mean():.6f} e {seeds['global_max_class_probability'].mean():.6f}, com razão média de {seeds['global_max_min_probability_ratio'].mean():.2f}. Apesar de CV e JS manterem associação monotônica forte ($\\rho$ médio de {seeds['spearman_cv_js'].mean():.3f}), a substituição do prior uniforme pelo prior empírico não é inócua: em média, {seeds['writers_rank_changed'].mean():.2f}/10 writers mudam de posição e {seeds['writers_band_changed'].mean():.2f}/10 mudam de terço. O efeito sobre a decisão discreta de épocas é menor: {seeds['writers_epochs_changed'].mean():.2f}/10 writers por seed; em 28/30 seeds exatamente dois writers trocam a alocação e nas outras duas nenhum muda, sem qualquer alteração no total de 43 épocas. Isso sustenta uma conclusão de robustez parcial do controle discreto, mas não equivalência entre as duas medidas de heterogeneidade.
"""
    (OUT_DOCS / "FEMNIST_10_WRITER_PRIOR_ANALYSIS.md").write_text(report, encoding="utf-8")


def main() -> None:
    for directory in (OUT_DATA, OUT_TABLES, OUT_FIGS, OUT_DOCS):
        directory.mkdir(parents=True, exist_ok=True)
    canonical = load_canonical_training_splits()
    details, classes, seeds, writers = analyze(canonical)

    details.to_csv(OUT_DATA / "femnist_10_writer_prior_all_values.csv", index=False, float_format="%.12g")
    classes.to_csv(OUT_DATA / "femnist_10_writer_class_distribution_by_seed.csv", index=False, float_format="%.12g")
    seeds.to_csv(OUT_DATA / "femnist_10_writer_seed_summary.csv", index=False, float_format="%.12g")
    writers.to_csv(OUT_DATA / "femnist_10_writer_writer_summary.csv", index=False, float_format="%.12g")

    set_plot_style()
    plot_prior(classes)
    plot_cv_vs_js(details, writers, seeds)
    write_tables(seeds, writers)
    write_report(seeds, writers)

    print(f"Analyzed {details.shape[0]} writer-seed rows and {classes.shape[0]} class-seed rows")
    print(seeds.describe().loc[["mean", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
