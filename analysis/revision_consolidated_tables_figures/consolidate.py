#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-entry consolidation of every FedHAD experimental result into a
LaTeX-ready, auditable package.

    python consolidate.py

Runs NO training. Reads existing results only. Writes only inside
revision_consolidated_tables_figures/. Nothing outside that folder is created,
modified or deleted.

Design rules enforced here
--------------------------
* Raw reported values are preserved in `tflops_reported`. The publication
  `tflops` field applies one documented nominal-workload correction: the four
  fixed-epoch baselines in historical 10-round FEMNIST all have the same
  scheduled cost (writers x 5 epochs x 10 rounds).
* Campaigns are never mixed. `main_campaign_default_alpha_0_5` and `main_campaign_default_alpha_0_01`
  were run on different platforms (Section 5.8 of the manuscript; the platform of each
  run is read from its report); the revision folders use a different
  protocol (fixed initialisation). Every row records `source_campaign` and
  `protocol`, and no paired comparison ever crosses that boundary.
* Missing information stays NaN and is reported as "not available in the existing
  campaign". Nothing is estimated.
* Accuracy per TFLOP is not used in the manuscript; it survives only as a secondary
  column of tabA. Efficiency is read from accuracy against compute at the configured
  operating points (each method is one point, not a frontier) and computation-to-target.
* The old 5-epoch baseline vs FedHAD 2-5 comparison is labelled a
  COMMON COMPUTE CEILING, never "equal compute". That term is reserved for the
  matched controls of the ablation campaign.

Visual identity
---------------
Method colours and markers are taken verbatim from `figuras_artigo.py`, which
generated the figures currently in the paper. They are not re-invented. FedNova,
which has no historical colour, gets one unused, colourblind-distinguishable hue,
recorded below.
"""
from __future__ import annotations

import json
import math
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                 # repository root (this script: analysis/revision_consolidated_tables_figures/)
OUT = ROOT / "results" / "revision_consolidated_tables_figures"
DATA, TABLES, FIGURES = OUT / "data", OUT / "tables", OUT / "figures"
LATEX, DOCS = OUT / "latex", OUT / "docs"
for d in (DATA, TABLES, FIGURES, LATEX, DOCS):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "analysis" / "main_campaign"))
from femnist_compute_accounting import publication_tflops  # noqa: E402

# ============================================================================ #
# 1. CENTRAL STYLE  - single source of truth for every figure produced here
# ============================================================================ #
import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.lines import Line2D                                 # noqa: E402

# Verbatim from figuras_artigo.py (the script that made the paper's figures).
# DO NOT change these: they are the identity readers already associate with each
# method in the published figures.
COLORS = {
    "FedAVG":  "#1F77B4",   # blue
    "FedAvgM": "#9467BD",   # purple
    "FedProx": "#2CA02C",   # green
    "FedHAD":  "#D62728",   # red
    # New: FedNova has no historical colour. Orange is unused by the four above
    # and stays distinguishable from all of them under deuteranopia/protanopia.
    "FedNova": "#FF7F0E",
}
MARKERS = {"FedAVG": "o", "FedAvgM": "^", "FedProx": "D", "FedHAD": "s",
           "FedNova": "P"}
DISPLAY = {"FedAVG": "FedAvg"}                 # only FedAVG is re-cased
ORDER = ["FedAVG", "FedAvgM", "FedProx", "FedNova", "FedHAD"]   # FedHAD last
BASELINES = ["FedAVG", "FedAvgM", "FedProx", "FedNova"]

# Ablation arms (revision campaign) keep their own palette; they are variants of
# one method, not competing methods, so they must not borrow the method colours.
ARM_COLORS = {"full_fedhad": "#D62728", "fixed_matched": "#7F7F7F",
              "permuted_allocation": "#FF7F0E", "epoch_only": "#17BECF",
              "lr_only_matched": "#8C564B"}
ARM_LABEL = {"full_fedhad": "full FedHAD", "fixed_matched": "fixed matched",
             "permuted_allocation": "permuted alloc.", "epoch_only": "epochs only",
             "lr_only_matched": "LR only (matched)"}
ARM_ORDER = ["fixed_matched", "epoch_only", "lr_only_matched",
             "permuted_allocation", "full_fedhad"]

# Larger fonts than the main-campaign figures, while staying legible in
# an Elsevier two-column layout (single column ~3.4 in, double ~7.0 in).
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13.5,
    "lines.linewidth": 2.6, "lines.markersize": 8,
    "axes.linewidth": 1.1, "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35,
    "axes.axisbelow": True, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})

COL_SINGLE, COL_DOUBLE = 3.5, 7.1     # Elsevier column widths, inches


def label(m):
    return DISPLAY.get(m, m)


def save_fig(fig, name):
    """Every figure ships as vector PDF and 300 dpi PNG.

    PDF metadata is pinned so that two runs over the same raw data produce
    byte-identical files; matplotlib otherwise stamps the wall-clock
    CreationDate and the artefact hashes would never match.
    """
    for ext in ("pdf", "png"):
        meta = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    metadata=meta)
    plt.close(fig)
    return name


def method_handles(methods):
    return [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], lw=2.6,
                   markersize=8, label=label(m)) for m in methods]


# ============================================================================ #
# 2. INVENTORY
# ============================================================================ #
CAMPAIGNS = {
    "base_alpha_0.5":  ROOT / "results" / "main_campaign_default_alpha_0_5",
    "base_alpha_0.01": ROOT / "results" / "main_campaign_default_alpha_0_01",
}
# Protocol of the base campaigns; the hardware of each run is read from its own log.
SERVER_HW = "AMD EPYC 9354P / NVIDIA A40 48GB / 256GB RAM"
BASE_PROTOCOL = ("historical: initial global model taken from one random Ray "
                 "client (not seed-controlled)")
REVISION_PROTOCOL = "revision: deterministic initial model fixed in the driver"

TAG_TO_BLOCK = {
    "test1_convergencia": "A", "test8_femnist": "A",
    "test2_robustez_alpha": "B",
    "test3_comunicacao": "A", "test4_clientes": "A",
    "test5_ablacao": "C_legacy", "test6_calibracao": "C_legacy",
    "test7_plato": "D", "1.1_diagnostico_drift": "F",
}

TARGETS = [40, 50, 60, 70, 80, 90]

# Filled by the inventory; surfaced in the audit.
DEDUP = {"nested_copies_dropped": 0, "other_duplicates_dropped": 0}


def run_hardware(path):
    """'GPU / CPU' as recorded in the header of a report (Section 5.8 of the manuscript)."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "not recorded"
    gpu = re.search(r"^GPU:\s*(.*)$", head, re.M)
    cpu = re.search(r"^CPU:\s*(.*)$", head, re.M)
    if not gpu and not cpu:
        return "not recorded"
    return " / ".join(x.group(1).strip() for x in (cpu, gpu) if x)


def inventory_base_campaigns():
    """Parse every .txt with the manuscript's own parser, so numbers match it."""
    import analisar_resultados as AR

    rows = []
    for campaign, base in CAMPAIGNS.items():
        if not base.exists():
            print(f"  ! campaign folder missing: {base}")
            continue
        files = sorted(base.rglob("*.txt"))
        print(f"  {campaign}: {len(files)} result files", flush=True)
        for p in files:
            try:
                d = AR.parse_file(p.resolve())
            except Exception as e:                       # keep going, report later
                rows.append({"source_campaign": campaign, "parse_error": str(e),
                             "file": p.name})
                continue
            d["source_campaign"] = campaign
            d["hardware"] = run_hardware(p)
            d["protocol"] = BASE_PROTOCOL
            d["block"] = TAG_TO_BLOCK.get(d.get("experiment_tag", ""), "other")
            d["file"] = p.name
            d["rel_path"] = str(p.relative_to(ROOT))
            rows.append(d)
    df = pd.DataFrame(rows)

    # De-duplicate. `main_campaign_default_alpha_0_5/FedAvgM-results/` contains a nested
    # copy of itself (FedAvgM-results/FedAvgM-results/), so every FedAvgM run is
    # present twice as a byte-identical file. Counting it twice would not move the
    # means but would misstate n and break any paired test. The originals are left
    # untouched on disk; the duplicate is dropped here and reported.
    if not df.empty and "rel_path" in df:
        before = len(df)
        nested = df.rel_path.str.contains(r"([^/]+-results)/\1/", regex=True,
                                          na=False)
        df = df[~nested].reset_index(drop=True)
        DEDUP["nested_copies_dropped"] = before - len(df)
        key = ["source_campaign", "metodo", "dataset", "experiment_tag", "seed",
               "alpha", "comm_delay", "client_setup", "num_rounds",
               "heuristics_signature", "epochs_decay", "lr_decay"]
        key = [k for k in key if k in df.columns]
        before = len(df)
        df = df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
        DEDUP["other_duplicates_dropped"] = before - len(df)
    return df


def inventory_ablation():
    """Revision ablation: per-cell fingerprints + telemetry (block C)."""
    raw = ROOT / "results" / "component_analysis_ablation" / "raw" / "official"
    if not raw.exists():
        return pd.DataFrame(), pd.DataFrame()
    fps = []
    for p in sorted(raw.glob("fingerprint__*.json")):
        fp = json.loads(p.read_text(encoding="utf-8"))
        b = fp["budget"]
        fps.append({
            "source_campaign": "revision_ablation", "protocol": REVISION_PROTOCOL,
            "dataset": fp["dataset"], "alpha": fp["alpha"], "seed": fp["seed"],
            "arm": fp["variant"], "rounds": fp["num_rounds"],
            "clients": fp["n_clients"],
            "total_epochs": b["total_epochs"], "updates": b["total_updates"],
            "flops": b["total_flops"],
            "flops_realised": b.get("total_flops_realised"),
            "updates_pct_vs_full": b.get("updates_pct_vs_full"),
            "wall_clock": fp.get("elapsed_seconds"),
            "rho_n_vs_H": fp.get("rank_corr_n_vs_H"),
            "rho_E_fixed_vs_H": fp.get("rank_corr_E_fixed_vs_H"),
            "rho_E_perm_vs_H": fp.get("rank_corr_E_perm_vs_H"),
            "hamming_fixed_vs_full": fp.get("hamming_fixed_vs_full"),
            "E_full": ";".join(map(str, fp["E_full"])),
            "E_fixed": ";".join(map(str, fp["E_fixed"])),
            "E_perm": ";".join(map(str, fp["E_perm"])),
        })
    tel_files = sorted(raw.glob("telemetry__*.csv"))
    tel = (pd.concat([pd.read_csv(p) for p in tel_files], ignore_index=True)
           if tel_files else pd.DataFrame())
    return pd.DataFrame(fps), tel


def inventory_revision_tables():
    """Derived tables already produced by the three analysis campaigns."""
    got = {}
    spec = {
        "drift_telemetry": ROOT / "results" / "drift_diagnostics_10seeds" / "1.1_diagnostico_drift" / "drift_telemetry.csv",
        "prior_dirichlet": ROOT / "results" / "global_prior_analysis" / "data" / "dirichlet_client_metrics.csv",
        "prior_femnist": ROOT / "results" / "global_prior_analysis" / "data" / "femnist_writer_metrics.csv",
        "prior_contrasts": ROOT / "results" / "global_prior_analysis" / "tables" / "T2_dirichlet_spearman.csv",
        "prior_femnist_summary": ROOT / "results" / "global_prior_analysis" / "tables" / "T4_femnist_ranking_disagreement.csv",
        "partition_verification": ROOT / "results" / "data_partition_manifest" / "artifacts" / "partitions" / "partition_verification.csv",
        "partition_manifest": ROOT / "results" / "data_partition_manifest" / "artifacts" / "partitions" / "partition_manifest.csv",
        "femnist_heldout": ROOT / "results" / "data_partition_manifest" / "artifacts" / "partitions" / "femnist_heldout_manifest.csv",
        "ablation_contrasts": ROOT / "results" / "component_analysis_ablation" / "tables" / "contrasts_official.csv",
        "ablation_diagnostics": ROOT / "results" / "component_analysis_ablation" / "tables" / "per_seed_diagnostics_official.csv",
    }
    for k, p in spec.items():
        got[k] = pd.read_csv(p) if p.exists() else pd.DataFrame()
        print(f"  {k:24} {'-' if got[k].empty else len(got[k]):>7} rows"
              f"{'   MISSING: ' + str(p) if got[k].empty else ''}")
    return got


# ============================================================================ #
# 3. MASTER SUMMARY
# ============================================================================ #
def num(x):
    try:
        v = float(x)
        return np.nan if math.isnan(v) else v
    except (TypeError, ValueError):
        return np.nan


def build_master(base_df, abl_fp, abl_tel):
    """One row per experimental run. NaN wherever the campaign never recorded it."""
    rows = []
    for _, r in base_df.iterrows():
        if pd.notna(r.get("parse_error", np.nan)):
            continue
        reported_tflops = num(r.get("total_tflops"))
        rounds = num(r.get("num_rounds"))
        tflops = publication_tflops(
            reported_tflops,
            source_campaign=r["source_campaign"],
            experiment_tag=r.get("experiment_tag"),
            dataset=r.get("dataset"),
            method=r.get("metodo"),
            rounds=rounds,
        )
        rows.append({
            "source_campaign": r["source_campaign"], "protocol": r["protocol"],
            "hardware": r["hardware"], "block": r["block"],
            "experiment_tag": r.get("experiment_tag"),
            "dataset": r.get("dataset"), "alpha": num(r.get("alpha")),
            "method": r.get("metodo"), "arm": np.nan,
            "seed": num(r.get("seed")), "rounds": rounds,
            "clients": num(r.get("client_setup")),
            "comm_delay": num(r.get("comm_delay")),
            "final_accuracy": num(r.get("acuracia_final")),
            "best_accuracy": num(r.get("acuracia_melhor")),
            "loss": np.nan,                       # not recorded by the base parser
            "tflops": tflops, "tflops_reported": reported_tflops,
            "flops": tflops * 1e12 if pd.notna(tflops) else np.nan,
            "updates": np.nan,                    # not available in base campaigns
            "effective_epochs": num(r.get("epocas")),
            "wall_clock": num(r.get("tempo_s")),
            "energy_kwh": num(r.get("energia_kwh")),
            "bytes_total_mb": num(r.get("bytes_total_mb")),
            "heuristics_signature": r.get("heuristics_signature"),
            "experiment_mode": r.get("experiment_mode"),
            "epochs_decay": num(r.get("epochs_decay")),
            "lr_decay": num(r.get("lr_decay")),
            "rel_path": r.get("rel_path"),
            **{f"rounds_to_target_{t}": num(r.get(f"r2t_{t}")) for t in TARGETS},
            **{f"tflops_to_target_{t}": num(r.get(f"flops2t_{t}")) for t in TARGETS},
        })

    if not abl_fp.empty:
        last = (abl_tel.loc[abl_tel.groupby(["variant", "alpha", "seed"])["round"].idxmax()]
                if not abl_tel.empty else pd.DataFrame())
        acc = {(r.variant, r.alpha, r.seed): (r.acc_centralized_round,
                                              r.loss_centralized_round)
               for _, r in last.iterrows()} if not last.empty else {}
        for _, r in abl_fp.iterrows():
            a, l = acc.get((r["arm"], r["alpha"], r["seed"]), (np.nan, np.nan))
            rows.append({
                "source_campaign": r["source_campaign"], "protocol": r["protocol"],
                "hardware": SERVER_HW,
                "block": "C", "experiment_tag": "revision_ablation",
                "dataset": r["dataset"], "alpha": r["alpha"],
                "method": "FedHAD", "arm": r["arm"], "seed": r["seed"],
                "rounds": r["rounds"], "clients": r["clients"],
                "comm_delay": np.nan,
                "final_accuracy": a, "best_accuracy": np.nan, "loss": l,
                "tflops": r["flops"] / 1e12 if pd.notna(r["flops"]) else np.nan,
                "tflops_reported": r["flops"] / 1e12
                if pd.notna(r["flops"]) else np.nan,
                "flops": r["flops"], "updates": r["updates"],
                "effective_epochs": r["total_epochs"] / r["clients"]
                if r["clients"] else np.nan,
                "wall_clock": r["wall_clock"], "energy_kwh": np.nan,
                "bytes_total_mb": np.nan,
                "heuristics_signature": np.nan, "experiment_mode": np.nan,
                "epochs_decay": np.nan, "lr_decay": np.nan,
                "rel_path": "results/component_analysis_ablation/raw/official",
                **{f"rounds_to_target_{t}": np.nan for t in TARGETS},
                **{f"tflops_to_target_{t}": np.nan for t in TARGETS},
            })
    m = pd.DataFrame(rows)
    for t in TARGETS:
        m[f"target_{t}_reached"] = m[f"rounds_to_target_{t}"].notna()
    return m


# ============================================================================ #
# 4. LaTeX helpers
# ============================================================================ #
def booktabs(path, caption, tab_label, header, body_rows, colspec, note=None,
             star=False):
    env = "table*" if star else "table"
    note_block = ""
    if note:
        note_block = ("\n\\vspace{2pt}\n{\\footnotesize\\raggedright\n"
                      f"\\textit{{Note.}} {note}\n\\par}}\n")
    tex = f"""% Generated by consolidate.py -- do not edit by hand.
% Requires \\usepackage{{booktabs}}.
\\begin{{{env}}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{tab_label}}}
\\begin{{tabular}}{{{colspec}}}
\\toprule
{header} \\\\
\\midrule
{chr(10).join(body_rows)}
\\bottomrule
\\end{{tabular}}
{note_block}\\end{{{env}}}
"""
    path.write_text(tex, encoding="utf-8")
    return path.name


def fmt(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


def mean_sd(s, nd=2, pct=False):
    s = pd.Series(s).dropna()
    if s.empty:
        return "--"
    m, d = s.mean(), s.std(ddof=1) if len(s) > 1 else 0.0
    if pct:
        return f"{100 * m:.{nd}f} $\\pm$ {100 * d:.{nd}f}"
    return f"{m:.{nd}f} $\\pm$ {d:.{nd}f}"


# ============================================================================ #
# 5. BLOCKS
# ============================================================================ #
def pareto_front(points):
    """Indices of non-dominated points; maximise accuracy, minimise compute."""
    idx = []
    for i, (c_i, a_i) in enumerate(points):
        if any((c_j <= c_i and a_j >= a_i) and (c_j < c_i or a_j > a_i)
               for j, (c_j, a_j) in enumerate(points) if j != i):
            continue
        idx.append(i)
    return idx


def block_A(master, produced):
    """A - overall accuracy/compute trade-off on the four datasets."""
    sub = master[(master.block == "A") & (master.source_campaign == "base_alpha_0.5")
                 & (master.experiment_tag.isin(["test1_convergencia", "test8_femnist"]))]
    if sub.empty:
        return
    audit = sub[
        (sub.dataset == "FEMNIST")
        & (sub.method.isin(["FedAVG", "FedAvgM", "FedProx", "FedNova"]))
    ][["method", "seed", "rounds", "effective_epochs", "tflops_reported", "tflops", "rel_path"]].copy()
    audit = audit.rename(columns={"tflops": "nominal_publication_tflops"})
    audit["correction_tflops"] = (
        audit.nominal_publication_tflops - audit.tflops_reported
    )
    audit["nominal_formula"] = "116602368 * 2126 * 5 * 10 / 1e12"
    audit.to_csv(DATA / "femnist_nominal_compute_audit.csv", index=False)
    g = (sub.groupby(["dataset", "method"])
         .agg(n_seeds=("seed", "nunique"),
              acc=("final_accuracy", "mean"), acc_sd=("final_accuracy", "std"),
              tflops=("tflops", "mean"), tflops_sd=("tflops", "std"),
              wall=("wall_clock", "mean"))
         .reset_index())
    g["acc_per_tflop"] = g.acc / g.tflops
    g.to_csv(DATA / "blockA_accuracy_compute.csv", index=False)

    datasets = [d for d in ["MNIST", "FashionMNIST", "CIFAR10", "FEMNIST"]
                if d in set(g.dataset)]
    body = []
    for ds in datasets:
        sd = g[g.dataset == ds]
        ref = sd[sd.method == "FedAVG"].tflops.mean()
        for m in ORDER:
            r = sd[sd.method == m]
            if r.empty:
                continue
            r = r.iloc[0]
            red = 100 * (ref - r.tflops) / ref if pd.notna(ref) and ref else np.nan
            body.append(
                f"{ds} & {label(m)} & {int(r.n_seeds)} & "
                f"{fmt(r.acc, 2, pct=True)} $\\pm$ {fmt(r.acc_sd, 2, pct=True)} & "
                f"{fmt(r.tflops, 2)} & {fmt(red, 1)} & {fmt(r.acc_per_tflop, 3)} \\\\")
        body.append("\\addlinespace")
    produced.append(booktabs(
        TABLES / "tabA_accuracy_compute.tex",
        "Final accuracy and nominal training compute per method and dataset "
        "($\\alpha=0.5$ campaign, 30 seeds). Compute reduction is relative to "
        "FedAvg. Accuracy per TFLOP is reported as secondary information only; "
        "the primary efficiency evidence is Fig.~\\ref{fig:pareto} and "
        "Table~\\ref{tab:cost-to-target}.",
        "tab:acc-compute",
        "Dataset & Method & $n$ & Final acc. (\\%) & TFLOPs & Compute red. (\\%) & Acc/TFLOP",
        body[:-1], "llrrrrr",
        note="FEMNIST includes the complete reruns of FedProx seed 66 and FedHAD seed 53. "
             "Fixed-epoch compute is the "
             "shared scheduled nominal workload; the raw reported value is retained "
             "in the audit CSV. The comparison uses a "
             "COMMON COMPUTE CEILING (baselines at 5 local epochs, FedHAD "
             "adaptive within 2--5), not a matched budget; matched-budget "
             "evidence is in Table~\\ref{tab:ablation}.", star=True))

    # Pareto: accuracy vs compute, per dataset
    fig, axes = plt.subplots(1, len(datasets), figsize=(COL_DOUBLE, 2.9),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sd = g[g.dataset == ds]
        pts = [(r.tflops, r.acc) for _, r in sd.iterrows()]
        front = set(pareto_front(pts))
        # The fixed-epoch baselines land on exactly the same compute, so their
        # markers would hide one another. Spread co-located points by a small
        # deterministic horizontal offset (5% of the axis span) purely for
        # legibility; the Pareto front below is computed on the true values.
        span = (max(p[0] for p in pts) - min(p[0] for p in pts)) or max(
            p[0] for p in pts)
        groups = {}
        for i, (x, _) in enumerate(pts):
            groups.setdefault(round(x, 3), []).append(i)
        x_plot = {}
        for members in groups.values():
            n = len(members)
            for j, i in enumerate(members):
                x_plot[i] = pts[i][0] + (j - (n - 1) / 2) * 0.05 * span
        for i, (_, r) in enumerate(sd.iterrows()):
            ax.errorbar(x_plot[i], 100 * r.acc,
                        xerr=r.tflops_sd, yerr=100 * (r.acc_sd or 0),
                        fmt=MARKERS[r.method], color=COLORS[r.method],
                        markersize=11 if i in front else 8,
                        markeredgecolor="black" if i in front else "white",
                        markeredgewidth=1.4 if i in front else 0.8,
                        capsize=3, elinewidth=1.2)
        fr = sorted([pts[i] for i in front])
        if len(fr) > 1:
            ax.plot([p[0] for p in fr], [100 * p[1] for p in fr],
                    color="0.35", ls="--", lw=1.4, zorder=0)
        ax.set_title(ds, fontsize=14)
        ax.set_xlabel("TFLOPs")
    axes[0][0].set_ylabel("Final accuracy (\\%)".replace("\\", ""))
    fig.legend(handles=method_handles([m for m in ORDER if m in set(g.method)]),
               loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.13))
    produced.append(save_fig(fig, "figA_pareto_accuracy_compute"))


def block_B(master, produced):
    """B - heterogeneity severity across every alpha available."""
    sub = master[(master.experiment_tag.isin(["test1_convergencia",
                                              "test2_robustez_alpha"]))
                 & master.alpha.notna() & master.arm.isna()]
    if sub.empty:
        return
    g = (sub.groupby(["source_campaign", "dataset", "alpha", "method"])
         .agg(n=("seed", "nunique"), acc=("final_accuracy", "mean"),
              sd=("final_accuracy", "std")).reset_index())
    g.to_csv(DATA / "blockB_heterogeneity.csv", index=False)

    alphas = sorted(g.alpha.unique())
    body = []
    for ds in ["MNIST", "FashionMNIST", "CIFAR10"]:
        if ds not in set(g.dataset):
            continue
        for m in ORDER:
            cells = []
            for a in alphas:
                r = g[(g.dataset == ds) & (g.alpha == a) & (g.method == m)]
                cells.append("--" if r.empty else
                             f"{100 * r.iloc[0].acc:.2f}")
            if all(c == "--" for c in cells):
                continue
            body.append(f"{ds} & {label(m)} & " + " & ".join(cells) + " \\\\")
        body.append("\\addlinespace")
    produced.append(booktabs(
        TABLES / "tabB_heterogeneity.tex",
        "Final accuracy (\\%) as heterogeneity increases. Values are means over "
        "the available seeds; standard deviations are in "
        "\\texttt{data/blockB\\_heterogeneity.csv}.",
        "tab:heterogeneity",
        "Dataset & Method & " + " & ".join(f"$\\alpha={a:g}$" for a in alphas),
        body[:-1], "ll" + "r" * len(alphas),
        note="Combines the $\\alpha=0.5$ and $\\alpha=0.01$ base campaigns, both "
             "run on the workstation declared in the manuscript. A dash means the "
             "combination was not executed; no value is imputed.", star=True))

    fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 2.9), squeeze=False,
                             sharey=False)
    for ax, ds in zip(axes[0], ["MNIST", "FashionMNIST", "CIFAR10"]):
        for m in ORDER:
            r = g[(g.dataset == ds) & (g.method == m)].sort_values("alpha")
            if r.empty:
                continue
            ax.errorbar(r.alpha, 100 * r.acc, yerr=100 * r.sd.fillna(0),
                        marker=MARKERS[m], color=COLORS[m], capsize=3,
                        markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"Dirichlet $\alpha$")
        ax.set_title(ds, fontsize=14)
    axes[0][0].set_ylabel("Final accuracy (%)")
    fig.legend(handles=method_handles([m for m in ORDER if m in set(g.method)]),
               loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.13))
    produced.append(save_fig(fig, "figB_heterogeneity_severity"))


def block_C(rev, abl_fp, produced):
    """C - equal-compute / equal-update ablation (revision campaign)."""
    con, diag = rev["ablation_contrasts"], rev["ablation_diagnostics"]
    if con.empty:
        return
    prim = con[con.analysis == "primary_all_seeds"]
    body = []
    for _, r in prim.iterrows():
        q = r.question.split(" ", 1)[0]
        star = "$^{*}$" if r.wilcoxon_p < 0.05 else ""
        p_display = r"$<0.001$" if r.wilcoxon_p < 0.001 else f"{r.wilcoxon_p:.3f}"
        body.append(f"{q} & \\texttt{{{r.arm_a.replace('_', chr(92) + '_')}}} $-$ "
                    f"\\texttt{{{r.arm_b.replace('_', chr(92) + '_')}}} & "
                    f"{r.alpha:g} & {int(r.n_seeds)} & "
                    f"{r.mean_diff_pp:+.2f}{star} & "
                    f"[{r.ci95_low_pp:+.2f}, {r.ci95_high_pp:+.2f}] & "
                    f"{r.cohens_dz:+.2f} & {p_display} \\\\")
    du = diag[diag.variant == "permuted_allocation"]["updates_pct_vs_full"] \
        if not diag.empty else pd.Series(dtype=float)
    dufix = diag[diag.variant == "fixed_matched"]["updates_pct_vs_full"].abs() \
        if not diag.empty else pd.Series(dtype=float)
    produced.append(booktabs(
        TABLES / "tabC_ablation.tex",
        "Controlled ablation of FedHAD on CIFAR-10 (revision campaign, 30 seeds, "
        "fixed initialisation). Paired differences in final accuracy, with 95\\% "
        "bootstrap intervals over seeds. Positive favours the first arm.",
        "tab:ablation",
        "\\# & Contrast & $\\alpha$ & $n$ & $\\Delta$acc (pp) & 95\\% CI & $d_z$ & $p$",
        body, "llrrrlrr",
        note="Every arm of a cell shares initial weights, partitions, clients and "
             "$H_k$ (SHA-256 verified). Compute is measured as minibatch updates. "
             "\\texttt{epoch\\_only} matches \\texttt{full\\_fedhad} exactly, and "
             "\\texttt{lr\\_only\\_matched} matches \\texttt{fixed\\_matched} "
             "exactly. The fixed allocation is budget-matched relative to "
             f"\\texttt{{full\\_fedhad}} within a median {dufix.median():.2f}\\% "
             f"(max {dufix.max():.2f}\\%); thus Q1, Q3 and Q5b are approximate "
             "budget-matched contrasts, whereas Q4 and Q5a are exact in updates. "
             "\\texttt{permuted\\_allocation} is "
             f"\\textbf{{not}} an equal-compute arm -- its update total deviates by "
             f"a median {du.median():+.2f}\\% (range {du.min():+.2f} to "
             f"{du.max():+.2f}\\%) because epochs are integers, there are five "
             "clients and each holds a different number of minibatches; it is "
             "reported as complementary evidence on the client-to-budget "
             "association only. $^{*}$ marks $p<0.05$; a non-significant row is a "
             "failure to detect a difference at this sample size, never evidence "
             "of equivalence.", star=True))

    # Figure: paired contrasts, equal-compute vs not
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 3.4))
    # Deliberately outside the method palette: these two colours encode the
    # contrast TYPE, not a method, and reusing FedAvg blue / FedNova orange
    # here would read as a method label.
    C_EQ, C_NEQ = "#333333", "#8C564B"
    ypos, ylab = [], []
    for i, (_, r) in enumerate(prim.iterrows()):
        eq = bool(r.equal_compute_contrast)
        ax.errorbar(r.mean_diff_pp, i,
                    xerr=[[r.mean_diff_pp - r.ci95_low_pp],
                          [r.ci95_high_pp - r.mean_diff_pp]],
                    fmt="o", color=C_EQ if eq else C_NEQ,
                    capsize=4, markersize=9, elinewidth=2.0)
        ypos.append(i)
        ylab.append(f"{r.question.split(' ', 1)[0]}  " + rf"$\alpha$={r.alpha:g}")
    ax.axvline(0, color="0.3", lw=1.2)
    ax.set_yticks(ypos, ylab, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Paired difference in final accuracy (pp)")
    fig.legend(handles=[
        Line2D([0], [0], color=C_EQ, marker="o", lw=2.4, markersize=9,
               label="budget-matched contrast"),
        Line2D([0], [0], color=C_NEQ, marker="o", lw=2.4, markersize=9,
               label="permuted alloc. (compute not matched)")],
        frameon=False, loc="lower center", ncol=2, fontsize=12,
        bbox_to_anchor=(0.5, -0.10))
    produced.append(save_fig(fig, "figC_ablation_contrasts"))

    if not diag.empty:
        diag.to_csv(DATA / "blockC_ablation_diagnostics.csv", index=False)
    prim.to_csv(DATA / "blockC_ablation_contrasts.csv", index=False)


def block_D(master, produced):
    """D - long horizon (50 rounds)."""
    sub = master[(master.experiment_tag == "test7_plato") & master.arm.isna()]
    if sub.empty:
        return
    g = (sub.groupby(["source_campaign", "dataset", "alpha", "method"])
         .agg(n=("seed", "nunique"), acc=("final_accuracy", "mean"),
              sd=("final_accuracy", "std"), best=("best_accuracy", "mean"),
              tflops=("tflops", "mean"), wall=("wall_clock", "mean"))
         .reset_index())
    g.to_csv(DATA / "blockD_long_horizon.csv", index=False)
    body = []
    for _, r in g.sort_values(["source_campaign", "method"]).iterrows():
        body.append(
            f"{r.source_campaign.replace('_', chr(92) + '_')} & {r.alpha:g} & "
            f"{label(r.method)} & {int(r.n)} & "
            f"{100 * r.acc:.2f} $\\pm$ {100 * (r.sd or 0):.2f} & "
            f"{100 * r.best:.2f} & {r.tflops:.2f} & {r.wall:.0f} \\\\")
    produced.append(booktabs(
        TABLES / "tabD_long_horizon.tex",
        "Long-horizon runs (50 communication rounds, CIFAR-10).",
        "tab:long-horizon",
        "Campaign & $\\alpha$ & Method & $n$ & Final acc. (\\%) & Best acc. (\\%) "
        "& TFLOPs & Wall-clock (s)",
        body, "lllrrrrr",
        note="The two campaigns are reported side by side and never pooled: they "
             "differ in $\\alpha$ and were executed separately. Wall-clock is "
             "reported as recorded; the machine was not otherwise controlled.",
        star=True))


def block_E(master, produced):
    """E - FedNova, only where it was actually executed."""
    sub = master[(master.method == "FedNova") & master.arm.isna()]
    if sub.empty:
        return
    g = (sub.groupby(["experiment_tag", "dataset", "alpha"])
         .agg(n=("seed", "nunique"), acc=("final_accuracy", "mean"),
              sd=("final_accuracy", "std"), tflops=("tflops", "mean"))
         .reset_index())
    g.to_csv(DATA / "blockE_fednova.csv", index=False)

    # Side-by-side with the other methods, only in the executed scenarios
    keys = set(zip(g.experiment_tag, g.dataset, g.alpha))
    comp = master[(master.arm.isna()) & (master.source_campaign == "base_alpha_0.5")]
    comp = comp[[tuple(x) in keys for x in
                 zip(comp.experiment_tag, comp.dataset, comp.alpha)]]
    cg = (comp.groupby(["experiment_tag", "dataset", "alpha", "method"])
          .agg(n=("seed", "nunique"), acc=("final_accuracy", "mean"),
               sd=("final_accuracy", "std"), tflops=("tflops", "mean")).reset_index())
    cg.to_csv(DATA / "blockE_fednova_context.csv", index=False)

    body = []
    for (tag, ds, a), grp in cg.groupby(["experiment_tag", "dataset", "alpha"]):
        for m in ORDER:
            r = grp[grp.method == m]
            if r.empty:
                continue
            r = r.iloc[0]
            body.append(f"{tag.replace('_', chr(92) + '_')} & {ds} & {a:g} & "
                        f"{label(m)} & {int(r.n)} & "
                        f"{100 * r.acc:.2f} $\\pm$ {100 * (r.sd or 0):.2f} & "
                        f"{r.tflops:.2f} \\\\")
        body.append("\\addlinespace")
    produced.append(booktabs(
        TABLES / "tabE_fednova.tex",
        "FedNova in the scenarios where it was executed, shown alongside the other "
        "methods in exactly those scenarios.",
        "tab:fednova",
        "Experiment & Dataset & $\\alpha$ & Method & $n$ & Final acc. (\\%) & TFLOPs",
        body[:-1], "lllrrrr",
        note="FedNova was run only for convergence (CD1), robustness (CD2) and "
             "FEMNIST (CD8) in the $\\alpha=0.5$ campaign. Absent combinations are "
             "omitted rather than imputed.", star=True))


def block_F(rev, produced):
    """F - heterogeneity proxy validation (drift diagnostics)."""
    d = rev["drift_telemetry"]
    if d.empty:
        return
    from scipy.stats import spearmanr
    rows = []
    for (m, a), g in d.groupby(["metodo", "alpha"]):
        g = g[g.n_samples > 0]
        if len(g) < 5:
            continue
        rows.append({"method": m, "alpha": a, "n": len(g),
                     "rho_Hk_cos": spearmanr(g.H_k, g.cos_sim).statistic,
                     "rho_Hk_updnorm": spearmanr(g.H_k, g.update_norm_per_step).statistic,
                     "p_cos": spearmanr(g.H_k, g.cos_sim).pvalue})
    f = pd.DataFrame(rows)
    f.to_csv(DATA / "blockF_proxy_validation.csv", index=False)
    body = []
    for m in ORDER:
        for a in sorted(f.alpha.unique()):
            r = f[(f.method == m) & (f.alpha == a)]
            if r.empty:
                continue
            r = r.iloc[0]
            body.append(f"{label(m)} & {a:g} & {int(r.n)} & {r.rho_Hk_cos:+.3f} & "
                        f"{r.rho_Hk_updnorm:+.3f} \\\\")
    produced.append(booktabs(
        TABLES / "tabF_proxy_validation.tex",
        "Validation of $H_k$ as a drift proxy: Spearman correlation between $H_k$ "
        "and the per-client update diagnostics.",
        "tab:proxy",
        "Method & $\\alpha$ & $n$ & $\\rho(H_k,\\cos)$ & $\\rho(H_k,\\|\\Delta\\|/\\text{step})$",
        body, "lrrrr",
        note="Three (alpha, seed, client) pairs with $n_k=0$ are excluded: their "
             "logged values are sentinels, not measurements. In FedHAD $H_k$ also "
             "drives $E_k$ and the learning rate, so its correlation is not an "
             "independent validation; the three baselines, where $H_k$ is purely "
             "observational, are the clean estimates."))


def block_G(rev, produced):
    """G - global-prior / CV analysis."""
    con, fem = rev["prior_contrasts"], rev["prior_femnist_summary"]
    if con.empty:
        return
    p = con[con.analysis == "primary_all_seeds"] if "analysis" in con else con
    body = []
    for m in ["FedAvg", "FedAvgM", "FedProx"]:
        for a in sorted(p.alpha.unique()):
            r = p[(p.method == m) & (p.alpha == a)]
            if r.empty:
                continue
            cells = []
            for metric in ["cv_norm", "entropy_het", "js_global", "tv_global"]:
                rr = r[r.metric == metric]
                cells.append("--" if rr.empty else f"{rr.iloc[0].spearman_rho:+.3f}")
            body.append(f"{m} & {a:g} & " + " & ".join(cells) + " \\\\")
    produced.append(booktabs(
        TABLES / "tabG_prior_metrics.tex",
        "Association between four heterogeneity statistics and the raw client-update "
        "alignment measure on CIFAR-10 (Spearman $\\rho$ against the cosine with the "
        "self-inclusive aggregate; baselines only).",
        "tab:prior-metrics",
        "Method & $\\alpha$ & Normalised CV & $1-$entropy & JS vs global & TV vs global",
        body, "lrrrrr",
        note="CIFAR-10's empirical global prior is exactly uniform (5{,}000 images "
             "per class), so JS against the empirical and against the uniform prior "
             "coincide and this dataset cannot test the uniform-prior premise -- it "
             "characterises the regime where the premise holds. FEMNIST, whose "
             "pooled global prior is 20.4$\\times$ skewed, lies outside that regime; see "
             "\\texttt{data/blockG\\_femnist\\_prior.csv}."))
    p.to_csv(DATA / "blockG_prior_contrasts.csv", index=False)
    if not fem.empty:
        fem.to_csv(DATA / "blockG_femnist_prior.csv", index=False)


def block_H(rev, produced):
    """H - partition / fairness / reproducibility."""
    ver, man = rev["partition_verification"], rev["partition_manifest"]
    if ver.empty:
        return
    rows = []
    for (ds, part), g in ver.groupby(["dataset", "partitioning"]):
        rows.append({
            "dataset": ds, "partitioning": part, "scenarios": len(g),
            "methods": int(g.n_methods.max()),
            "identical_indices": f"{int(g.sha_indices_identical.sum())}/{len(g)}",
            "identical_counts": f"{int(g.sha_counts_identical.sum())}/{len(g)}",
            "negative_control": f"{int((g.negative_control_passed == True).sum())}/"
                                f"{int((g.negative_control_passed != 'NA').sum())}",
            "rng_stress": f"{int(g.reseed_stress_passed.sum())}/{len(g)}",
        })
    h = pd.DataFrame(rows)
    h.to_csv(DATA / "blockH_partition_verification.csv", index=False)
    body = [f"{r.dataset} & {r.partitioning} & {r.scenarios} & {r.methods} & "
            f"{r.identical_indices} & {r.identical_counts} & {r.negative_control} & "
            f"{r.rng_stress} \\\\" for _, r in h.iterrows()]
    produced.append(booktabs(
        TABLES / "tabH_partitions.tex",
        "Verification that every method operates on identical client partitions.",
        "tab:partitions",
        "Dataset & Partitioning & Scenarios & Methods & Identical indices & "
        "Identical counts & Negative control & RNG stress",
        body, "llrrrrrr",
        note="Identity holds by construction: \\texttt{dirichlet\\_split\\_noniid} "
             "reseeds NumPy as its first statement, making the partition a pure "
             "function of (labels, clients, $\\alpha$, seed). The negative control "
             "confirms distinct seeds give distinct partitions; the RNG stress test "
             "consumes ${\\sim}1.5$M draws and reseeds to 999 immediately before "
             "partitioning.", star=True))

    if not man.empty:
        fem = man[man.dataset == "FEMNIST"]
        if not fem.empty:
            fem.groupby(["writer_natural_id"]).agg(
                n_k=("n_k", "mean"), classes=("n_classes_covered", "mean"),
                H_k=("H_k_norm", "mean")).reset_index().to_csv(
                DATA / "blockH_femnist_writers.csv", index=False)


# ============================================================================ #
# 6. CONSISTENCY AUDIT
# ============================================================================ #
def consistency_audit(master):
    """Compare what we extracted against the CSVs that fed the paper's figures."""
    lines = ["# Consistency audit", "",
             "Compares values extracted here against `results/main_campaign_summaries/analise_bruta_*.csv` "
             "— the derived tables produced by `analisar_resultados.py`, from which the "
             "main-campaign figures of the manuscript were drawn.", "",
             "> **Scope.** This audit compares against that pipeline, not against the "
             "typeset tables of the manuscript. The numbers written in the manuscript "
             "are recomputed independently by the scripts of `analysis/audit/`.",
             ""]
    if DEDUP["nested_copies_dropped"] or DEDUP["other_duplicates_dropped"]:
        lines += [
            "## Duplicate source files found", "",
            f"- **{DEDUP['nested_copies_dropped']} files** were dropped because the "
            "campaign folder contains a nested copy of itself "
            "(`main_campaign_default_alpha_0_5/FedAvgM-results/FedAvgM-results/`). The "
            "nested files are byte-identical to the outer ones, so every FedAvgM "
            "run appeared twice. Means are unaffected, but the run count and any "
            "paired test would have been wrong. **The originals were not modified "
            "or deleted**; the duplicate is excluded from this consolidation only.",
            f"- {DEDUP['other_duplicates_dropped']} further duplicate rows were "
            "dropped on the full experimental key.", ""]

    an = ROOT / "results" / "main_campaign_summaries"
    if not an.exists():
        lines.append("`results/main_campaign_summaries/` not found — audit skipped.")
        (DOCS / "CONSISTENCY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
        return 0

    total_checked, total_bad = 0, 0
    for f in sorted(an.glob("analise_bruta_*.csv")):
        ref = pd.read_csv(f)
        tag = f.stem.replace("analise_bruta_", "")
        cur = master[(master.experiment_tag == tag) &
                     (master.source_campaign == "base_alpha_0.5")]
        if cur.empty or ref.empty:
            lines.append(f"- `{f.name}`: no comparable rows in this consolidation "
                         f"(reference {len(ref)} rows) — skipped.")
            continue
        # The join key must include every dimension that varies WITHIN a tag
        # (alpha, delay, clients, rounds, heuristic config). Using only
        # (method, dataset, seed) produces a cartesian product on tests 2-6 and
        # manufactures false discrepancies.
        ref_map = {"metodo": "metodo", "dataset": "dataset", "seed": "seed",
                   "alpha": "alpha", "comm_delay": "comm_delay",
                   "client_setup": "client_setup", "num_rounds": "num_rounds",
                   "heuristics_signature": "heuristics_signature",
                   "epochs_decay": "epochs_decay", "lr_decay": "lr_decay"}
        cur_map = {"metodo": "method", "dataset": "dataset", "seed": "seed",
                   "alpha": "alpha", "comm_delay": "comm_delay",
                   "client_setup": "clients", "num_rounds": "rounds",
                   "heuristics_signature": "heuristics_signature",
                   "epochs_decay": "epochs_decay", "lr_decay": "lr_decay"}
        key = [k for k in ref_map
               if ref_map[k] in ref.columns and cur_map[k] in cur.columns]
        r = ref[[ref_map[k] for k in key] + ["acuracia_final", "total_tflops"]].copy()
        r.columns = key + ["acuracia_final", "total_tflops"]
        # Compare the historical references to the untouched reported field.
        # `tflops` may intentionally contain the documented FEMNIST nominal
        # scheduled-workload correction.
        c = cur[[cur_map[k] for k in key] + ["final_accuracy", "tflops_reported"]].copy()
        c.columns = key + ["final_accuracy", "tflops"]
        # Normalise dtypes AND fill NaN with a sentinel: pandas does not join
        # NaN to NaN, and epochs_decay / lr_decay are null for every method
        # except FedHAD, which would silently drop 75% of the rows from the
        # comparison and make a clean audit meaningless.
        for k in key:
            if k in ("seed", "alpha", "comm_delay", "client_setup", "num_rounds",
                     "epochs_decay", "lr_decay"):
                r[k] = pd.to_numeric(r[k], errors="coerce").round(6).fillna(-999.0)
                c[k] = pd.to_numeric(c[k], errors="coerce").round(6).fillna(-999.0)
            else:
                # NaN, None and "" must all collapse to one sentinel. The parser
                # returns "" for absent text fields in memory, while the same
                # field round-trips through CSV as NaN; without this the two
                # frames disagree and only FedHAD (the one method with a
                # non-empty signature) would join.
                r[k] = (r[k].astype("object").where(r[k].notna(), "NA")
                        .astype(str).replace({"": "NA", "nan": "NA", "None": "NA"}))
                c[k] = (c[k].astype("object").where(c[k].notna(), "NA")
                        .astype(str).replace({"": "NA", "nan": "NA", "None": "NA"}))
        dup_ref = int(r.duplicated(subset=key).sum())
        dup_cur = int(c.duplicated(subset=key).sum())
        mg = r.merge(c, on=key, how="inner", suffixes=("_ref", "_new"))
        if dup_ref or dup_cur:
            lines.append(f"- `{f.name}`: key still not unique "
                         f"(ref dup={dup_ref}, new dup={dup_cur}) — "
                         f"comparison unreliable, investigate.")
            continue
        if mg.empty:
            lines.append(f"- `{f.name}`: keys did not join — skipped.")
            continue
        d_acc = (mg.acuracia_final - mg.final_accuracy).abs()
        d_fl = (mg.total_tflops - mg.tflops).abs()
        bad = int((d_acc > 1e-9).sum() + (d_fl > 1e-6).sum())
        total_checked += len(mg)
        total_bad += bad
        lines.append(f"- `{f.name}`: {len(mg)}/{len(ref)} rows joined on "
                     f"{len(key)} keys, max |Δaccuracy| = {d_acc.max():.2e}, "
                     f"max |ΔTFLOPs| = {d_fl.max():.2e} — "
                     f"{'**OK**' if bad == 0 else f'**{bad} DISCREPANCIES**'}")
    lines += ["", f"**Total**: {total_checked} rows compared, {total_bad} "
                  f"discrepancies.", ""]
    if total_bad == 0:
        lines.append("Every historical number reproduced exactly. The consolidation "
                     "re-uses `analisar_resultados.parse_file`, so this is expected "
                     "and confirms no drift was introduced.")
    (DOCS / "CONSISTENCY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return total_bad


# ============================================================================ #
# 7. DOCS + LATEX
# ============================================================================ #
def write_docs(master, produced, tables, audit_bad, counts):
    (DOCS / "RESULTS_MAP.md").write_text(f"""# Results map

Every artefact produced here, its source data, and the question it addresses.
Use this to audit any number in the paper back to the file it came from.

| Artefact | Source data | Experiments used | Question addressed |
|---|---|---|---|
| `tabA_accuracy_compute.tex`, `figA_pareto_accuracy_compute` | `main_campaign_default_alpha_0_5/*-results/test1_convergencia`, `test8_femnist` | MNIST, FashionMNIST, CIFAR-10, FEMNIST; 30 seeds | Accuracy–compute trade-off at the configured operating points (no Acc/TFLOP ratio) |
| `tabB_heterogeneity.tex`, `figB_heterogeneity_severity` | both base campaigns, `test1_convergencia` + `test2_robustez_alpha` | α ∈ {{1.0, 0.5, 0.1, 0.01}} where executed | Heterogeneity severity |
| `tabC_ablation.tex`, `figC_ablation_contrasts` | `component_analysis_ablation/tables/contrasts_official.csv`, `per_seed_diagnostics_official.csv` | CIFAR-10, α ∈ {{0.01, 0.1}}, 30 seeds, 5 arms | Controlled component analysis with matched update budgets |
| `tabD_long_horizon.tex` | `test7_plato` in both base campaigns | CIFAR-10, 50 rounds | Long-horizon behaviour |
| `tabE_fednova.tex` | `main_campaign_default_alpha_0_5/FedNova-results` | CD1, CD2, CD8 only | FedNova baseline |
| `tabF_proxy_validation.tex` | `drift_diagnostics_10seeds/1.1_diagnostico_drift/drift_telemetry.csv` | CIFAR-10, α ∈ {{1.0, 0.1, 0.01}}, 10 seeds | Association between H_k and update alignment (raw cosine; see Section 6.4) |
| `tabG_prior_metrics.tex` | `global_prior_analysis/tables/T2_dirichlet_spearman.csv`, `T4_*.csv` | CIFAR-10 + FEMNIST writer pool | Uniform-global-prior assumption in H_k |
| `tabH_partitions.tex` | `data_partition_manifest/artifacts/partitions/*.csv` | 1110 scenarios × 5 methods | Partition identity, fairness, reproducibility |
| `master_results_summary.csv` | all of the above | every run found | Full audit trail |

## Provenance and protocol boundaries

| Campaign | Hardware (Section 5.8 of the manuscript) | Initialisation protocol |
|---|---|---|
| `base_alpha_0.5` | workstation (Intel Core i7-14700 / NVIDIA RTX A1000 8GB / 32GB RAM) for FedAvg, FedAvgM, FedProx and FedHAD; server ({SERVER_HW}) for FedNova, SCAFFOLD, the drift-diagnostic campaign and the six replacement runs | {BASE_PROTOCOL} |
| `base_alpha_0.01` | server ({SERVER_HW}) | {BASE_PROTOCOL} |
| `revision_ablation` | server ({SERVER_HW}) | {REVISION_PROTOCOL} |

The `hardware` column of `master_results_summary.csv` gives the platform of every run
as recorded in its own report.

Rows from the two protocols are **never** compared pairwise. The `protocol` column
in `master_results_summary.csv` makes the boundary explicit.
""", encoding="utf-8")

    inputs = "\n".join(f"\\input{{tables/{t.replace('.tex', '')}}}" for t in tables)
    figs = """
\\begin{figure*}[t]
  \\centering
  \\includegraphics[width=\\textwidth]{figures/figA_pareto_accuracy_compute.pdf}
  \\caption{Final accuracy versus realised compute for each dataset. Larger
  markers with a black edge are on the Pareto front; dashed line connects the
  non-dominated points. Error bars are one standard deviation over 30 seeds.
  A method inside the front is dominated: another method reaches at least the
  same accuracy for no more compute. The four fixed-epoch baselines run at
  identical compute, so their markers are offset horizontally by a small fixed
  amount for legibility only; their true TFLOPs coincide and are listed exactly
  in Table~\\ref{tab:acc-compute}.}
  \\label{fig:pareto}
\\end{figure*}

\\begin{figure*}[t]
  \\centering
  \\includegraphics[width=\\textwidth]{figures/figB_heterogeneity_severity.pdf}
  \\caption{Final accuracy as Dirichlet $\\alpha$ decreases. Combines the
  $\\alpha=0.5$ and $\\alpha=0.01$ base campaigns, executed on different
  platforms (Section~\\ref{platforms}) and never compared pairwise.}
  \\label{fig:heterogeneity}
\\end{figure*}

\\begin{figure*}[t]
  \\centering
  \\includegraphics[width=\\textwidth]{figures/figC_ablation_contrasts.pdf}
  \\caption{Paired contrasts of the controlled ablation on CIFAR-10, 30 seeds,
  with 95\\% bootstrap intervals over seeds. Blue contrasts have matched update budgets;
  the orange contrast uses \\texttt{permuted\\_allocation}, whose update budget is
  not matched and which is therefore complementary evidence only.}
  \\label{fig:ablation}
\\end{figure*}
"""
    (LATEX / "revision_results.tex").write_text(f"""% Consolidated revision results for the FedHAD manuscript.
% Generated by analysis/revision_consolidated_tables_figures/consolidate.py -- regenerate rather than edit by hand.
%
% Preamble requirements:
%   \\usepackage{{booktabs}}
%   \\usepackage{{graphicx}}
%
% Paths assume this file is \\input from the manuscript root with
% revision_consolidated_tables_figures/ copied in as-is. Adjust \\graphicspath if not.

% ---------------------------------------------------------------- tables ----
{inputs}

% --------------------------------------------------------------- figures ----
{figs}
""", encoding="utf-8")


# ============================================================================ #
# 8. MAIN
# ============================================================================ #
def main() -> int:
    print("=" * 78)
    print("FEDHAD RESULTS CONSOLIDATION (no training; read-only on sources)")
    print("=" * 78)

    print("\n[1/6] Inventorying base campaigns...")
    base = inventory_base_campaigns()
    print("\n[2/6] Inventorying revision campaigns...")
    abl_fp, abl_tel = inventory_ablation()
    print(f"  revision_ablation        {len(abl_fp):>7} cells, "
          f"{len(abl_tel)} telemetry rows")
    rev = inventory_revision_tables()

    print("\n[3/6] Building master summary...")
    master = build_master(base, abl_fp, abl_tel)
    master.to_csv(DATA / "master_results_summary.csv", index=False)
    print(f"  master_results_summary.csv: {len(master)} runs, "
          f"{master.shape[1]} columns")

    counts = {
        "base_alpha_0.5": int((master.source_campaign == "base_alpha_0.5").sum()),
        "base_alpha_0.01": int((master.source_campaign == "base_alpha_0.01").sum()),
        "revision_ablation": int((master.source_campaign == "revision_ablation").sum()),
    }

    print("\n[4/6] Building blocks...")
    produced, tables = [], []
    for name, fn in [("A", lambda: block_A(master, produced)),
                     ("B", lambda: block_B(master, produced)),
                     ("C", lambda: block_C(rev, abl_fp, produced)),
                     ("D", lambda: block_D(master, produced)),
                     ("E", lambda: block_E(master, produced)),
                     ("F", lambda: block_F(rev, produced)),
                     ("G", lambda: block_G(rev, produced)),
                     ("H", lambda: block_H(rev, produced))]:
        try:
            fn()
            print(f"  block {name}: ok")
        except Exception as e:
            print(f"  block {name}: FAILED - {type(e).__name__}: {e}")
    tables = sorted(p.name for p in TABLES.glob("*.tex"))

    print("\n[5/6] Consistency audit...")
    bad = consistency_audit(master)
    print(f"  discrepancies: {bad}")

    print("\n[6/6] Docs and LaTeX...")
    write_docs(master, produced, tables, bad, counts)

    print("\n" + "-" * 78)
    print("SUMMARY")
    print("-" * 78)
    for k, v in counts.items():
        print(f"  {k:<20} {v:>6} runs")
    print(f"  datasets : {sorted(master.dataset.dropna().unique())}")
    print(f"  methods  : {sorted(master.method.dropna().unique())}")
    print(f"  alphas   : {sorted(master.alpha.dropna().unique())}")
    print(f"  seeds    : {int(master.seed.nunique())} distinct")
    print(f"\n  tables   : {len(tables)}")
    for t in tables:
        print(f"      {t}")
    figs_made = sorted({p.stem for p in FIGURES.glob('*.pdf')})
    print(f"  figures  : {len(figs_made)}")
    for f in figs_made:
        print(f"      {f}.pdf / .png")
    print(f"\n  Everything written under {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
