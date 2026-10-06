"""A10 - Numbers for tables of the manuscript (read-only).

(1) Principal family F1 with accuracies + Holm (from a06 output + raw reports).
(2) FEMNIST 10-writer prior analysis (Table 12): epochs, optimizer
    steps and learning rates under the CV controller and three alternatives.
(3) Sensitivity grid restricted to complete runs (the four incomplete test6 reports
    are excluded, not replaced).
(4) Extended proxy table (per-seed, seed-bootstrap, LOO, partial given size).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT, REV, CAMPAIGN_05, parse_report, seed_of, alpha_of

E = lambda h: 5 if h == 0 else max(2, int(round(5 - 3.0 * h)))
LR = lambda h: 0.01 if h == 0 else max(0.01 / (1 + 1.5 * h), 0.001)


def acc_series(method, tag, ds, a):
    d = CAMPAIGN_05 / f"{method}-results" / tag / (f"{ds}-Standard" if method == "FedHAD" else ds)
    return pd.Series({seed_of(p): parse_report(p)["acc_final"] for p in d.glob("*.txt")
                      if a is None or abs(alpha_of(p) - a) < 1e-9})


def main():
    lines = []
    h = pd.read_csv(OUT / "a06_holm_families.csv")
    f1 = h[h.family == "F1_principal_CIFAR10_10r"].copy()
    tags = {1.0: "test2_robustez_alpha", 0.5: "test1_convergencia", 0.1: "test2_robustez_alpha", 0.01: "test2_robustez_alpha"}
    fh = {a: acc_series("FedHAD", t, "CIFAR10", a).mean() for a, t in tags.items()}
    f1["fedhad_acc"] = f1.alpha.map(fh)
    f1["base_acc"] = [acc_series(b, tags[a], "CIFAR10", a).mean() for a, b in zip(f1.alpha, f1.baseline)]
    f1.to_csv(OUT / "a10_F1_table.csv", index=False)
    lines.append(f1[["alpha", "baseline", "fedhad_acc", "base_acc", "mean_pp", "ci_lo", "ci_hi", "dz", "p_raw", "p_holm"]].round(4).to_string(index=False))

    # (2) 10-writer prior
    v = pd.read_csv(REV / "revision_consolidated_tables_figures/data/femnist_10_writer_prior/femnist_10_writer_prior_all_values.csv")
    rows = []
    for s, g in v.groupby("seed"):
        g = g.sort_values("client_id")
        cv, js = g.cv_norm.to_numpy(), g.js_vs_10_writer_prior_bits.to_numpy()
        b = (g.n_train // 32).to_numpy()
        hq = np.sort(cv)[np.argsort(np.argsort(js))]
        e0 = np.array([E(x) for x in cv]); l0 = np.array([LR(x) for x in cv])
        e1 = g.epochs_js_remapped.to_numpy()
        e2 = np.array([E(x) for x in hq]); l2 = np.array([LR(x) for x in hq])
        e3 = np.array([E(x) for x in js]); l3 = np.array([LR(x) for x in js])
        assert (e0 == g.epochs_cv.to_numpy()).all()
        rows.append({"seed": s, "E_cv": e0.sum(), "E_A1": e1.sum(), "E_A2": e2.sum(), "E_A3": e3.sum(),
                     "steps_cv": (e0 * b).sum(), "steps_A1": (e1 * b).sum(), "steps_A2": (e2 * b).sum(), "steps_A3": (e3 * b).sum(),
                     "chgE_A1": int((e1 != e0).sum()), "chgE_A2": int((e2 != e0).sum()), "chgE_A3": int((e3 != e0).sum()),
                     "LRrel_A2_mean": np.mean(np.abs(l2 / l0 - 1)) * 100, "LRrel_A2_max": np.max(np.abs(l2 / l0 - 1)) * 100,
                     "chgLR_A2_gt1pct": int((np.abs(l2 / l0 - 1) > 0.01).sum()),
                     "LR_cv_min": l0.min(), "LR_cv_max": l0.max(), "LR_A3_min": l3.min(), "LR_A3_max": l3.max()})
    p = pd.DataFrame(rows)
    for k in ("A1", "A2", "A3"):
        p[f"steps_{k}_pct"] = 100 * (p[f"steps_{k}"] / p.steps_cv - 1)
    p.to_csv(OUT / "a10_prior10_per_seed.csv", index=False)
    lines.append("\nFEMNIST 10-writer prior:\n" + p.describe().T[["mean", "min", "max"]].round(4).to_string())

    # (3) sensitivity grid, complete runs only
    a5 = pd.read_csv(OUT / "a05_all_reports.csv")
    g = a5[(a5.tag == "test6_calibracao") & (a5.campaign == "main_campaign_default_alpha_0_5") & ~a5.nested_copy]
    s_all = g.groupby(["epochs_decay", "lr_decay"]).agg(n=("seed", "size"), acc=("acc_final", "mean"), TF=("TF", "mean"))
    s_c = g[g.complete].groupby(["epochs_decay", "lr_decay"]).agg(n=("seed", "size"), acc=("acc_final", "mean"),
                                                                     sd=("acc_final", "std"), TF=("TF", "mean"))
    sens = s_all.join(s_c, lsuffix="_all", rsuffix="_complete").reset_index()
    sens.to_csv(OUT / "a10_sensitivity_complete.csv", index=False)
    lines.append("\nSensitivity:\n" + sens.round(4).to_string(index=False))

    # (4) proxy extended
    d = pd.read_csv(OUT / "a02_heterogeneity_dependence.csv")
    pr = pd.read_csv(OUT / "a02_partial_given_size.csv")
    d = d.merge(pr, on=["method", "alpha"])
    cols = ["method", "alpha", "records", "client_units", "rho_pooled_H_cos", "seedboot95_H_cos", "rho_perseed_mean",
            "rho_perseed_min", "rho_perseed_max", "rho_client_level_H_cos", "rho_pooled_H_cosLOO", "seedboot95_H_cosLOO",
            "partial_rho_H_cosLOO_given_n", "rho_client_n_cos", "rho_client_n_H", "rho_pooled_H_normstep"]
    lines.append("\nProxy:\n" + d[cols].round(3).to_string(index=False))
    d[cols].to_csv(OUT / "a10_proxy_extended.csv", index=False)
    txt = "\n".join(lines)
    open(OUT / "a10_summary.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
