"""A07 - Global-prior analysis: exact mapping, epochs AND learning rates
(Section 6.8, Table 12).

Inputs (read-only): global_prior_analysis/data/campaign_federation_remap.csv
(per seed x writer: n_k, cv_norm, js_global, epochs_cv, epochs_js_remapped) and the
executed FedHAD FEMNIST reports (per-client epochs and LR actually used).

Controllers compared, all using FedHAD's own maps
  E(h) = max(2, round(5 - 3h)) (h>0),  LR(h) = max(0.01 / (1 + 1.5h), 0.001):
  C0  CV controller (what FedHAD does)                          -> E(cv), LR(cv)
  A1  published counterfactual: rank remap of the CV epoch multiset by JS rank,
      learning rates left at LR(cv)                              -> epochs only
  A2  quantile-mapped JS plugged into the controller (the JS value at within-
      federation rank r replaced by the CV value at rank r)      -> E(h'), LR(h')
  A3  raw JS used as the score (no rescaling)                    -> E(js), LR(js)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT, REV, CAMPAIGN_05, parse_report, seed_of

E = lambda h: 5 if h == 0 else max(2, int(round(5 - 3.0 * h)))
LR = lambda h: 0.01 if h == 0 else max(0.01 / (1 + 1.5 * h), 0.001)


def main():
    d = pd.read_csv(REV / "global_prior_analysis/data/campaign_federation_remap.csv")
    rows = []
    for s, g in d.groupby("seed"):
        g = g.sort_values("client_id").copy()
        cv, js = g.cv_norm.to_numpy(), g.js_global.to_numpy()
        hq = np.sort(cv)[np.argsort(np.argsort(js))]
        g["E_C0"] = [E(h) for h in cv]
        g["LR_C0"] = [LR(h) for h in cv]
        g["E_A1"] = g.epochs_js_remapped
        g["LR_A1"] = g.LR_C0
        g["E_A2"] = [E(h) for h in hq]
        g["LR_A2"] = [LR(h) for h in hq]
        g["E_A3"] = [E(h) for h in js]
        g["LR_A3"] = [LR(h) for h in js]
        rows.append(g)
    x = pd.concat(rows)
    assert (x.E_C0 == x.epochs_cv).all(), "controller map does not reproduce epochs_cv"

    # executed FEMNIST FedHAD runs: per-client epochs/LR actually used
    ex = []
    for p in sorted((CAMPAIGN_05 / "FedHAD-results/test8_femnist/FEMNIST-Standard").glob("*.txt")):
        r = parse_report(p)
        for cid, c in r["clients"].items():
            ex.append({"seed": seed_of(p), "client_id": cid, "E_exec": c.get("epochs_avg"), "LR_exec": c.get("lr_avg"), "n_exec": c["n"]})
    ex = pd.DataFrame(ex)
    x = x.merge(ex, on=["seed", "client_id"], how="left")
    x["E_exec_matches_C0"] = np.isclose(x.E_exec, x.E_C0)
    x["LR_exec_matches_C0"] = np.isclose(x.LR_exec, x.LR_C0, atol=5e-7)
    x.to_csv(OUT / "a07_prior_per_writer.csv", index=False)

    per_seed = x.groupby("seed").agg(
        E_C0=("E_C0", "sum"), E_exec=("E_exec", "sum"), E_A1=("E_A1", "sum"), E_A2=("E_A2", "sum"), E_A3=("E_A3", "sum"),
        upd_C0=("E_C0", lambda s: 0), n_changed_E_A1=("E_A1", lambda s: 0)).reset_index()
    per_seed["n_changed_E_A1"] = x.assign(c=x.E_A1 != x.E_C0).groupby("seed").c.sum().values
    per_seed["n_changed_E_A2"] = x.assign(c=x.E_A2 != x.E_C0).groupby("seed").c.sum().values
    per_seed["n_changed_LR_A2_gt1pct"] = x.assign(c=(abs(x.LR_A2 / x.LR_C0 - 1) > 0.01)).groupby("seed").c.sum().values
    # budget in optimizer steps (drop_last, batch 32) and samples
    x["b"] = x.n_k // 32
    for k in ("C0", "A1", "A2", "A3"):
        x[f"steps_{k}"] = x[f"E_{k}"] * x.b
    st = x.groupby("seed")[[f"steps_{k}" for k in ("C0", "A1", "A2", "A3")]].sum()
    per_seed = per_seed.merge(st.reset_index(), on="seed")
    for k in ("A1", "A2", "A3"):
        per_seed[f"steps_{k}_pct_vs_C0"] = 100 * (per_seed[f"steps_{k}"] / per_seed["steps_C0"] - 1)
    per_seed.drop(columns=["upd_C0"]).to_csv(OUT / "a07_prior_per_seed.csv", index=False)

    lr_diff = pd.DataFrame({
        "A2_abs_mean": [(x.LR_A2 - x.LR_C0).abs().mean()],
        "A2_rel_mean_pct": [100 * (x.LR_A2 / x.LR_C0 - 1).abs().mean()],
        "A2_rel_max_pct": [100 * (x.LR_A2 / x.LR_C0 - 1).abs().max()],
        "A3_LR_range": [f"[{x.LR_A3.min():.5f},{x.LR_A3.max():.5f}]"],
        "C0_LR_range": [f"[{x.LR_C0.min():.5f},{x.LR_C0.max():.5f}]"],
        "A3_E_values": [str(sorted(x.E_A3.unique()))],
        "exec_E_matches_manifest_C0": [f"{int(x.E_exec_matches_C0.sum())}/{len(x)}"],
        "exec_LR_matches_manifest_C0": [f"{int(x.LR_exec_matches_C0.sum())}/{len(x)}"],
    })
    with open(OUT / "a07_summary.txt", "w") as f:
        f.write(per_seed.describe().T.to_string() + "\n\n")
        f.write(lr_diff.T.to_string() + "\n")
    print(per_seed.describe().T.to_string())
    print(lr_diff.T.to_string())


if __name__ == "__main__":
    main()
