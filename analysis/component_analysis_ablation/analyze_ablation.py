#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis for the FedHAD controlled ablation.

Reads only artifacts produced by this campaign (raw/<mode>/) and writes only into
component_analysis_ablation/. No training is executed and no file outside this
folder is touched.

Design commitments honoured here
--------------------------------
* All 30 seeds are the PRIMARY analysis. No seed is ever dropped.
* A PRE-SPECIFIED sensitivity analysis restricted to cells with |dU| <= 2% is
  reported alongside, never instead.
* Conclusions about equal budget rest on full_fedhad, fixed_matched, epoch_only
  and lr_only_matched, where compute is matched exactly or to ~0.2% median.
* permuted_allocation is complementary evidence about the client->budget
  association. It is NOT an equal-compute arm and is never described as one; its
  dU is reported per seed and treated as a covariate and a limitation.
* Accuracy and loss are never used to select allocations, seeds or subsets.
* Statistical significance and practical effect are reported separately, and
  p > 0.05 is never presented as evidence of equivalence.

Usage
-----
    python analysis/component_analysis_ablation/analyze_ablation.py --mode official
    python analysis/component_analysis_ablation/analyze_ablation.py --mode smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "algoritmos" / "component_analysis_ablation"))
import config_ablation as C  # noqa: E402

REFERENCE = "full_fedhad"
CONTROL = "fixed_matched"
EQUAL_COMPUTE_ARMS = ["full_fedhad", "fixed_matched", "epoch_only", "lr_only_matched"]
SENSITIVITY_DU = 2.0  # pre-specified, design-based, uses no outcome data

# The five questions, as paired contrasts (arm_a - arm_b).
CONTRASTS = [
    ("Q1 does adaptation help at equal compute?", "full_fedhad", "fixed_matched", True),
    ("Q2 does the client->budget association matter?", "full_fedhad", "permuted_allocation", False),
    ("Q3 contribution of adaptive epochs", "epoch_only", "fixed_matched", True),
    ("Q4 contribution of adaptive learning rate", "lr_only_matched", "fixed_matched", True),
    ("Q5a coordination vs adaptive epochs alone", "full_fedhad", "epoch_only", True),
    ("Q5b coordination vs adaptive LR alone", "full_fedhad", "lr_only_matched", True),
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(mode):
    raw = C.RAW / mode
    if not raw.exists():
        raise FileNotFoundError(f"no raw directory for mode={mode}: {raw}")

    fps = []
    for p in sorted(raw.glob("fingerprint__*.json")):
        fp = json.loads(p.read_text(encoding="utf-8"))
        b = fp["budget"]
        fps.append({
            "variant": fp["variant"], "alpha": fp["alpha"], "seed": fp["seed"],
            "total_epochs": b["total_epochs"], "total_updates": b["total_updates"],
            "total_flops": b["total_flops"],
            "total_flops_realised": b.get("total_flops_realised"),
            "updates_pct_vs_full": b.get("updates_pct_vs_full", 0.0),
            "elapsed_seconds": fp.get("elapsed_seconds"),
            "epochs_assigned": fp["epochs_assigned"],
            "E_full": fp["E_full"], "E_fixed": fp["E_fixed"], "E_perm": fp["E_perm"],
            "H": fp["H"], "n_samples": fp["n_samples"], "n_batches": fp["n_batches"],
            "rho_n_vs_H": fp.get("rank_corr_n_vs_H"),
            "rho_E_full_vs_H": fp.get("rank_corr_E_full_vs_H"),
            "rho_E_fixed_vs_H": fp.get("rank_corr_E_fixed_vs_H"),
            "rho_E_perm_vs_H": fp.get("rank_corr_E_perm_vs_H"),
            "perm_degenerate": fp.get("perm_degenerate", False),
        })
    fp_df = pd.DataFrame(fps)

    tel = pd.concat([pd.read_csv(p) for p in sorted(raw.glob("telemetry__*.csv"))],
                    ignore_index=True)
    return fp_df, tel


def final_accuracy(tel):
    """One row per (variant, alpha, seed): the last round's centralised metrics."""
    last = tel.loc[tel.groupby(["variant", "alpha", "seed"])["round"].idxmax()]
    return last[["variant", "alpha", "seed", "round", "acc_centralized_round",
                 "loss_centralized_round", "acc_distributed_round",
                 "loss_distributed_round"]].rename(
        columns={"acc_centralized_round": "acc_final",
                 "loss_centralized_round": "loss_final",
                 "acc_distributed_round": "acc_dist_final",
                 "loss_distributed_round": "loss_dist_final",
                 "round": "final_round"}).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def paired_stats(df, arm_a, arm_b, alpha, metric="acc_final"):
    """
    Paired comparison over seeds. Returns effect size, CI and Wilcoxon, keeping
    significance and practical effect separate.
    """
    from scipy.stats import wilcoxon

    a = df[(df.variant == arm_a) & (df.alpha == alpha)].set_index("seed")[metric]
    b = df[(df.variant == arm_b) & (df.alpha == alpha)].set_index("seed")[metric]
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 3:
        return None
    a, b = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    d = a - b

    # Bootstrap CI over seeds (the unit of independence).
    rng = np.random.default_rng(20260906)
    boot = np.array([rng.choice(d, size=len(d), replace=True).mean()
                     for _ in range(10000)])
    try:
        w = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        p, stat = float(w.pvalue), float(w.statistic)
    except ValueError:      # all differences zero
        p, stat = 1.0, 0.0

    sd = d.std(ddof=1)
    return {
        "contrast": f"{arm_a} - {arm_b}", "arm_a": arm_a, "arm_b": arm_b,
        "alpha": alpha, "n_seeds": len(common), "metric": metric,
        "mean_a": float(a.mean()), "sd_a": float(a.std(ddof=1)),
        "mean_b": float(b.mean()), "sd_b": float(b.std(ddof=1)),
        "mean_diff_pp": float(d.mean() * 100),
        "median_diff_pp": float(np.median(d) * 100),
        "ci95_low_pp": float(np.percentile(boot, 2.5) * 100),
        "ci95_high_pp": float(np.percentile(boot, 97.5) * 100),
        "cohens_dz": float(d.mean() / sd) if sd > 0 else 0.0,
        "wins_a": int((d > 0).sum()), "wins_b": int((d < 0).sum()),
        "ties": int((d == 0).sum()),
        "wilcoxon_stat": stat, "wilcoxon_p": p,
    }


def run_contrasts(df, cells, label):
    rows = []
    for name, a, b, equal_compute in CONTRASTS:
        for alpha in C.ALPHAS:
            sub = df.merge(cells, on=["alpha", "seed"], how="inner")
            r = paired_stats(sub, a, b, alpha)
            if r:
                r.update({"question": name, "analysis": label,
                          "equal_compute_contrast": equal_compute})
                rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def write_budget_diagnostics(fp_df, path):
    """Per-seed diagnostics: dU, FLOPs, the rhos, Hamming, near-degenerate flags."""
    rows = []
    for (alpha, seed), g in fp_df.groupby(["alpha", "seed"]):
        ref = g[g.variant == REFERENCE]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        e_full, e_fixed, e_perm = ref.E_full, ref.E_fixed, ref.E_perm
        ham_fixed = sum(1 for x, y in zip(e_fixed, e_full) if x != y)
        ham_perm = sum(1 for x, y in zip(e_perm, e_full) if x != y)
        for _, r in g.iterrows():
            rows.append({
                "alpha": alpha, "seed": seed, "variant": r.variant,
                "total_epochs": r.total_epochs, "total_updates": r.total_updates,
                "updates_pct_vs_full": round(r.updates_pct_vs_full, 4),
                "total_flops_nominal": r.total_flops,
                "total_flops_realised": r.total_flops_realised,
                "flops_pct_vs_full": round(
                    100 * (r.total_flops - ref.total_flops) / ref.total_flops, 4),
                "elapsed_seconds": r.elapsed_seconds,
                "rho_n_vs_H": ref.rho_n_vs_H,
                "rho_E_full_vs_H": ref.rho_E_full_vs_H,
                "rho_E_fixed_vs_H": ref.rho_E_fixed_vs_H,
                "rho_E_perm_vs_H": ref.rho_E_perm_vs_H,
                "hamming_fixed_vs_full": ham_fixed,
                "hamming_perm_vs_full": ham_perm,
                "fixed_near_full": ham_fixed <= 1,
                "perm_degenerate": ref.perm_degenerate,
                "E_full": ";".join(map(str, e_full)),
                "E_fixed": ";".join(map(str, e_fixed)),
                "E_perm": ";".join(map(str, e_perm)),
                "H": ";".join(f"{x:.6f}" for x in ref.H),
            })
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def write_tex(main, sens, diag, path):
    def fmt(r):
        sig = "$^{*}$" if r.wilcoxon_p < 0.05 else ""
        p_display = r"$<0.001$" if r.wilcoxon_p < 0.001 else f"{r.wilcoxon_p:.3f}"
        return (f"{r.question.split(' ', 1)[0]} & {r.alpha:g} & {r.n_seeds} & "
                f"{r.mean_diff_pp:+.2f}{sig} & "
                f"[{r.ci95_low_pp:+.2f}, {r.ci95_high_pp:+.2f}] & "
                f"{r.cohens_dz:+.2f} & {r.wins_a}/{r.wins_b} & {p_display} \\\\")

    lines = [fmt(r) for _, r in main.iterrows()]
    du = diag[diag.variant == "permuted_allocation"]["updates_pct_vs_full"]
    du_fix = diag[diag.variant == "fixed_matched"]["updates_pct_vs_full"].abs()
    tex = f"""% Generated by analyze_ablation.py -- do not edit by hand.
% Requires \\usepackage{{booktabs}}.
\\begin{{table}}[t]
\\centering
\\caption{{Controlled ablation of FedHAD on CIFAR-10. Paired differences in final
centralised accuracy over {int(main.n_seeds.max())} seeds, with 95\\% bootstrap intervals
resampled over seeds. Positive values favour the first arm. All seeds are included;
no cell is excluded.}}
\\label{{tab:fedhad-ablation}}
\\begin{{tabular}}{{llrrlrrr}}
\\toprule
Contrast & $\\alpha$ & $n$ & $\\Delta$acc (pp) & 95\\% CI & $d_z$ & W/L & $p$ \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}

\\vspace{{2pt}}
{{\\footnotesize\\raggedright
\\textit{{Note.}} Every arm of a given (\\(\\alpha\\), seed) cell shares the same initial
weights, the same client partitions, the same clients and the same $H_k$, verified by
SHA-256 fingerprints. Compute is measured as \\emph{{minibatch updates}}:
\\texttt{{epoch\\_only}} matches \\texttt{{full\\_fedhad}} exactly, and
\\texttt{{lr\\_only\\_matched}} matches \\texttt{{fixed\\_matched}} exactly. The fixed
allocation is budget-matched relative to \\texttt{{full\\_fedhad}} within a median of
{du_fix.median():.2f}\\% (max {du_fix.max():.2f}\\%); Q1, Q3 and Q5b are therefore
approximate budget-matched contrasts, while Q4 and Q5a are exact in updates.
\\texttt{{permuted\\_allocation}} preserves the exact epoch multiset but
\\textbf{{cannot}} be compute-matched -- integer epochs, only five clients and unequal
minibatch counts per client leave as few as four unique non-trivial arrangements, so its
update total deviates by a median of {du.median():+.2f}\\% (range {du.min():+.2f}\\% to
{du.max():+.2f}\\%). It is therefore reported as complementary evidence on whether the
client-to-budget association matters, never as an equal-compute contrast.
$^{{*}}$ marks $p<0.05$ (Wilcoxon signed-rank). A non-significant result is reported as a
failure to detect a difference at this sample size, never as evidence of equivalence;
the effect size and its interval are given in every row.
\\par}}
\\end{{table}}
"""
    path.write_text(tex, encoding="utf-8")


def make_figures(final, diag, main, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "figure.autolayout": True})
    C.FIGURES.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(C.FIGURES / f"{name}_{mode}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # F1 - final accuracy per arm
    fig, axes = plt.subplots(1, len(C.ALPHAS), figsize=(8.0, 3.2), sharey=False)
    for ax, alpha in zip(np.atleast_1d(axes), C.ALPHAS):
        data = [final[(final.variant == v) & (final.alpha == alpha)].acc_final.dropna()
                for v in C.VARIANTS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        tick_labels=[C.VARIANT_LABEL[v] for v in C.VARIANTS])
        for patch, v in zip(bp["boxes"], C.VARIANTS):
            patch.set_facecolor(C.VARIANT_COLOR[v])
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_title(rf"$\alpha = {alpha:g}$")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Final centralised accuracy")
    save(fig, "F1_final_accuracy_by_arm")

    # F2 - paired contrasts with CI
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ypos, labels = [], []
    for i, (_, r) in enumerate(main.iterrows()):
        col = "#0072B2" if r.equal_compute_contrast else "#D55E00"
        ax.errorbar(r.mean_diff_pp, i,
                    xerr=[[r.mean_diff_pp - r.ci95_low_pp],
                          [r.ci95_high_pp - r.mean_diff_pp]],
                    fmt="o", color=col, capsize=3, markersize=5)
        ypos.append(i)
        labels.append(f"{r.question.split(' ', 1)[0]}  α={r.alpha:g}")
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_yticks(ypos, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Paired difference in final accuracy (pp), 95% CI over seeds")
    ax.set_title("Blue: equal-compute contrast   Orange: permuted_allocation "
                 "(compute not matched)", fontsize=7.5)
    save(fig, "F2_paired_contrasts")

    # F3 - compute matching per arm
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i, v in enumerate(C.VARIANTS):
        d = diag[diag.variant == v]["updates_pct_vs_full"]
        ax.scatter(np.full(len(d), i) + np.random.default_rng(0).normal(0, .06, len(d)),
                   d, s=12, color=C.VARIANT_COLOR[v], alpha=0.7, edgecolors="none")
    ax.axhline(0, color="0.3", lw=0.9)
    ax.axhspan(-SENSITIVITY_DU, SENSITIVITY_DU, color="0.85", zorder=0,
               label=f"|ΔU| ≤ {SENSITIVITY_DU:g}% (sensitivity band)")
    ax.set_xticks(range(len(C.VARIANTS)),
                  [C.VARIANT_LABEL[v] for v in C.VARIANTS], rotation=25, ha="right")
    ax.set_ylabel("Minibatch updates vs full_fedhad (%)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "F3_compute_matching")

    # F4 - accuracy trajectory per round
    fig, axes = plt.subplots(1, len(C.ALPHAS), figsize=(8.0, 3.0), sharey=False)
    for ax, alpha in zip(np.atleast_1d(axes), C.ALPHAS):
        for v in C.VARIANTS:
            d = (final.attrs["telemetry"]
                 .query("variant == @v and alpha == @alpha")
                 .groupby("round").acc_centralized_round.mean())
            ax.plot(d.index, d.values, color=C.VARIANT_COLOR[v],
                    label=C.VARIANT_LABEL[v], lw=1.4)
        ax.set_title(rf"$\alpha = {alpha:g}$")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean centralised accuracy")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6.5)
    save(fig, "F4_accuracy_by_round")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--mode", choices=["smoke", "official"], default="official")
    args = ap.parse_args()

    C.TABLES.mkdir(parents=True, exist_ok=True)
    fp_df, tel = load(args.mode)
    final = final_accuracy(tel)
    final.attrs["telemetry"] = tel

    print("=" * 78)
    print(f"FEDHAD CONTROLLED ABLATION - ANALYSIS ({args.mode})")
    print("=" * 78)
    print(f"Runs: {len(fp_df)} | arms: {fp_df.variant.nunique()} | "
          f"seeds: {fp_df.seed.nunique()} | alphas: {sorted(fp_df.alpha.unique())}")

    diag = write_budget_diagnostics(fp_df, C.TABLES / f"per_seed_diagnostics_{args.mode}.csv")
    print(f"\nPer-seed diagnostics -> tables/per_seed_diagnostics_{args.mode}.csv")

    for v in C.VARIANTS:
        d = diag[diag.variant == v]["updates_pct_vs_full"]
        print(f"  {C.VARIANT_LABEL[v]:<22} dU median {d.median():+.2f}%  "
              f"range [{d.min():+.2f}%, {d.max():+.2f}%]  "
              f"|dU|<=1%: {(d.abs() <= 1).sum()}/{len(d)}")
    nd = diag[(diag.variant == REFERENCE) & (diag.fixed_near_full)]
    print(f"  cells where fixed_matched is within Hamming 1 of full_fedhad: "
          f"{len(nd)}/{diag.variant.eq(REFERENCE).sum()}")
    ref = diag[diag.variant == REFERENCE]
    for col, lab in [("rho_n_vs_H", "rho(n_k, H_k)"),
                     ("rho_E_fixed_vs_H", "rho(E_fixed, H_k)"),
                     ("rho_E_perm_vs_H", "rho(E_perm, H_k)")]:
        vals = pd.to_numeric(ref[col], errors="coerce").dropna()
        if len(vals):
            print(f"  {lab:<20} mean {vals.mean():+.3f}  median {vals.median():+.3f}")

    # ---- primary analysis: ALL seeds --------------------------------------- #
    all_cells = fp_df[["alpha", "seed"]].drop_duplicates()
    main_df = run_contrasts(final, all_cells, "primary_all_seeds")

    # ---- pre-specified sensitivity: |dU| <= 2% ----------------------------- #
    ok = (diag.groupby(["alpha", "seed"])["updates_pct_vs_full"]
              .apply(lambda s: s.abs().max() <= SENSITIVITY_DU).rename("keep").reset_index())
    sens_cells = ok[ok.keep][["alpha", "seed"]]
    sens_df = run_contrasts(final, sens_cells, f"sensitivity_dU_le_{SENSITIVITY_DU:g}pct")
    print(f"\nPre-specified sensitivity |dU| <= {SENSITIVITY_DU:g}%: "
          f"{len(sens_cells)}/{len(all_cells)} cells retained "
          f"(reported alongside, never instead of, the primary analysis)")

    out = pd.concat([main_df, sens_df], ignore_index=True)
    out.to_csv(C.TABLES / f"contrasts_{args.mode}.csv", index=False)
    final.to_csv(C.TABLES / f"final_metrics_{args.mode}.csv", index=False)

    print(f"\n{'contrast':<46} {'alpha':>5} {'d(pp)':>8} {'CI95':>18} {'p':>7}")
    for _, r in main_df.iterrows():
        print(f"  {r.question:<44} {r.alpha:>5} {r.mean_diff_pp:>+8.2f} "
              f"[{r.ci95_low_pp:>+6.2f},{r.ci95_high_pp:>+6.2f}] {r.wilcoxon_p:>7.3f}")

    write_tex(main_df, sens_df, diag, C.TABLES / f"ablation_{args.mode}.tex")
    make_figures(final, diag, main_df, args.mode)

    print(f"\nWritten under {C.ROOT}")
    for p in sorted(list(C.TABLES.glob(f"*{args.mode}*")) +
                    list(C.FIGURES.glob(f"*{args.mode}*"))):
        print(f"  {p.relative_to(C.ROOT)}")
    print("\nReminder: p > 0.05 is a failure to detect a difference at this sample "
          "size, not evidence of equivalence. Read every row with its effect size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
