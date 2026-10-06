"""A12 - Sensitivity grid (test6_calibracao, alpha=0.5 campaign) rebuilt with the four
complete re-executions substituted for the four incomplete runs. Read-only: no raw file is
moved, renamed or edited; the substitution happens only in this analysis.

Incomplete originals (fewer than 10 participations for some client):
  (lambda_E, lambda_eta, seed) = (1.0, 0.5, 54), (3.0, 0.5, 70), (3.0, 0.5, 71), (3.0, 3.0, 65)
Complete re-executions: results/seed_reexecutions_corrupted_runs/reexecutions/
  FedHAD/test6_calibracao/CIFAR10-Standard/<same file name>
The figure is regenerated with the original plotting routine of analysis/main_campaign/figuras_artigo.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, OUT, REV, CAMPAIGN_05, parse_report, seed_of

sys.path.insert(0, str(ROOT / "analysis" / "main_campaign"))

ORIG = CAMPAIGN_05 / "FedHAD-results/test6_calibracao/CIFAR10-Standard"
RERUN = REV / "seed_reexecutions_corrupted_runs/reexecutions/FedHAD/test6_calibracao/CIFAR10-Standard"
FIG_OUT = ROOT / "PaperJNCA_Revision/Elservier___Article_FedHAD__Revised_manuscript_v3/figs/test6/accuracy_by_decay_config_v3"


def complete(r):
    R = int(r["NUM_ROUNDS"])
    return len(r["acc_hist"]) == R + 1 and len(r["flops_round"]) == R and \
        min(c["rounds_participated"] for c in r["clients"].values()) == R


def main():
    rows = []
    for p in sorted(ORIG.glob("*.txt")):
        r = parse_report(p)
        src, hw = "original", r["gpu"]
        if not complete(r):
            q = RERUN / p.name
            assert q.exists(), f"no re-execution for {p.name}"
            r2 = parse_report(q)
            assert complete(r2)
            for k in ("SEED", "EPOCHS_DECAY_FACTOR", "LR_DECAY_FACTOR", "ALPHA", "NUM_ROUNDS", "CLIENT_SETUP", "BATCH_SIZE"):
                assert r2[k] == r[k], k
            r, src, hw = r2, "re-execution", r2["gpu"]
        rows.append({"epochs_decay": float(r["EPOCHS_DECAY_FACTOR"]), "lr_decay": float(r["LR_DECAY_FACTOR"]),
                     "seed": seed_of(p), "acc": r["acc_final"], "TF": r["total_flops"] / 1e12,
                     "source": src, "gpu": hw, "file": r["path"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a12_grid_per_run.csv", index=False)
    g = df.groupby(["epochs_decay", "lr_decay"]).agg(n=("seed", "size"), acc=("acc", "mean"), sd=("acc", "std"),
                                                    TF=("TF", "mean"), reexec=("source", lambda s: int((s == "re-execution").sum()))).reset_index()
    g.to_csv(OUT / "a12_grid_summary.csv", index=False)
    print(df[df.source == "re-execution"][["epochs_decay", "lr_decay", "seed", "acc", "TF", "gpu", "file"]].to_string(index=False))
    print(g.round(4).to_string(index=False))

    import figuras_artigo as fa
    labels, medias, stds = [], [], []
    for _, r in g.iterrows():
        lab = f"λ_E={r.epochs_decay:.1f}, λ_η={r.lr_decay:.1f}"
        if (r.epochs_decay, r.lr_decay) == (3.0, 1.5):
            lab += "  (default)"
        labels.append(lab); medias.append(r.acc); stds.append(r.sd)
    fa.fig_barras_horizontais(labels, medias, stds, "λ_E=3.0, λ_η=1.5  (default)", "Final accuracy", FIG_OUT,
                              legenda_destaque="Default configuration", legenda_outras="Other configurations")
    # Copy kept with the results: it supersedes figures/test6/accuracy_by_decay_config, which
    # figuras_artigo.py draws from the original grid, including the four incomplete runs.
    # (figuras_artigo.py rebuilds figures/ from scratch, so the copy lives beside it.)
    import shutil
    copy_dir = ROOT / "results/main_campaign_summaries/sensitivity_grid_reexecuted"
    copy_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        shutil.copy2(str(FIG_OUT) + ext, copy_dir / (FIG_OUT.name + ext))
    shutil.copy2(OUT / "a12_grid_summary.csv", copy_dir / "a12_grid_summary.csv")
    print("figure:", FIG_OUT, "(copy in", copy_dir.relative_to(ROOT), ")")


if __name__ == "__main__":
    main()
