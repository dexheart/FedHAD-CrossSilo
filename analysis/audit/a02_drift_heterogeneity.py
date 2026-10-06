"""A02 - Diagnostic record counts and dependence-aware heterogeneity analysis
(Section 6.4, Table 9).

Reads only results/drift_diagnostics_10seeds/1.1_diagnostico_drift/drift_telemetry.csv
and the driver script that produced it.

1. Reconstructs the 500 / 470 record counts from the raw telemetry.
2. Pooled Spearman (as reported) vs. per-seed Spearman, seed-cluster bootstrap,
   client-level (one record per client-seed) Spearman.
3. Client size vs H, E, LR, aggregate weight, cosine, update norm; size-stratified
   (within-seed rank of n_k) analysis.
4. Exact leave-one-client-out (LOO) reference cosine, reconstructed from the logged
   quantities. The reference is g = sum_k w_k u_k with w_k = n_k / sum n (trainable
   tensors only, drift_telemetry.record_round_drift). Hence
       |g| = sum_k w_k |u_k| cos_k,
       cos(u_k, g_-k) = (|g| cos_k - w_k |u_k|) / sqrt(|g|^2 - 2 w_k |u_k| |g| cos_k + w_k^2 |u_k|^2),
   which is exact up to float precision; no retraining is needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import REV, OUT

TEL = REV / "drift_diagnostics_10seeds" / "1.1_diagnostico_drift" / "drift_telemetry.csv"
RNG = np.random.default_rng(20260929)
NBOOT = 5000


def rho(x, y):
    if len(x) < 3 or np.all(x == x.iloc[0]) or np.all(y == y.iloc[0]):
        return np.nan
    return spearmanr(x, y).correlation


def seed_boot(df, xc, yc):
    seeds = df.seed.unique()
    groups = {s: g for s, g in df.groupby("seed")}
    vals = []
    for _ in range(NBOOT):
        pick = RNG.choice(seeds, size=len(seeds), replace=True)
        b = pd.concat([groups[s] for s in pick])
        vals.append(spearmanr(b[xc], b[yc]).correlation)
    return np.nanpercentile(vals, [2.5, 97.5])


def main():
    t = pd.read_csv(TEL)
    lines = []
    lines.append(f"rows={len(t)}  methods={sorted(t.metodo.unique())}  alphas={sorted(t.alpha.unique())}")
    lines.append(f"seeds={sorted(t.seed.unique())} (n={t.seed.nunique()})  rounds={sorted(t['round'].unique())}  clients={sorted(t.client_id.unique())}")
    cnt = t.groupby(["metodo", "alpha"]).agg(records=("seed", "size"), seeds=("seed", "nunique"),
                                             rounds=("round", "nunique"), clients=("client_id", "nunique"),
                                             ).reset_index()
    cnt["empty_records"] = t.assign(e=t.n_samples == 0).groupby(["metodo", "alpha"]).e.sum().values
    cnt["after_exclusion"] = cnt["records"] - cnt["empty_records"]
    lines.append(cnt.to_string(index=False))
    empty = t[t.n_samples == 0][["metodo", "alpha", "seed", "client_id"]].drop_duplicates(["alpha", "seed", "client_id"])
    lines.append("empty (alpha, seed, client): " + ", ".join(f"({r.alpha},{r.seed},{r.client_id})" for r in empty.itertuples()))
    ec = t[t.n_samples == 0].groupby(["alpha", "seed", "client_id"]).agg(rows=("round", "size"), H=("H_k", "first"), cos=("cos_sim", "first"), E=("E_k", "first"))
    lines.append(ec.to_string())
    nan = t.isna().sum()
    lines.append("NaN per column: " + nan[nan > 0].to_string() if nan.sum() else "NaN per column: none")

    # H constant across rounds per (method, alpha, seed, client)?
    var = t.groupby(["metodo", "alpha", "seed", "client_id"]).H_k.nunique()
    lines.append(f"H_k distinct values per client-run: max={var.max()} (1 means constant across the 10 rounds)")

    d = t[t.n_samples > 0].copy()
    N = d.groupby(["metodo", "alpha", "seed", "round"]).n_samples.transform("sum")
    d["w"] = d.n_samples / N
    # LOO reconstruction
    d["u"] = d.update_norm
    d["gnorm"] = (d.w * d.u * d.cos_sim).groupby([d.metodo, d.alpha, d.seed, d["round"]]).transform("sum")
    num = d.gnorm * d.cos_sim - d.w * d.u
    den = np.sqrt(np.maximum(d.gnorm ** 2 - 2 * d.w * d.u * d.gnorm * d.cos_sim + (d.w * d.u) ** 2, 1e-30))
    d["cos_loo"] = num / den
    d["size_rank"] = d.groupby(["metodo", "alpha", "seed", "round"]).n_samples.rank()

    rows = []
    for (m, a), g in d.groupby(["metodo", "alpha"]):
        per_seed = g.groupby("seed").apply(lambda s: rho(s.H_k, s.cos_sim))
        per_seed_loo = g.groupby("seed").apply(lambda s: rho(s.H_k, s.cos_loo))
        client = g.groupby(["seed", "client_id"]).agg(H=("H_k", "first"), cos=("cos_sim", "mean"),
                                                     cos_loo=("cos_loo", "mean"), n=("n_samples", "first"),
                                                     w=("w", "first"), E=("E_k", "first"), lr=("lr_k", "first"),
                                                     unorm=("update_norm", "mean"), ups=("update_norm_per_step", "mean")).reset_index()
        lo, hi = seed_boot(g, "H_k", "cos_sim")
        lo2, hi2 = seed_boot(g, "H_k", "cos_loo")
        rows.append({
            "method": m, "alpha": a, "records": len(g), "seeds": g.seed.nunique(), "client_units": len(client),
            "rho_pooled_H_cos": rho(g.H_k, g.cos_sim),
            "rho_pooled_H_normstep": rho(g.H_k, g.update_norm_per_step),
            "seedboot95_H_cos": f"[{lo:.3f},{hi:.3f}]",
            "rho_perseed_mean": per_seed.mean(), "rho_perseed_median": per_seed.median(),
            "rho_perseed_min": per_seed.min(), "rho_perseed_max": per_seed.max(),
            "seeds_rho_neg": int((per_seed < 0).sum()),
            "rho_client_level_H_cos": rho(client.H, client.cos),
            "rho_pooled_H_cosLOO": rho(g.H_k, g.cos_loo),
            "seedboot95_H_cosLOO": f"[{lo2:.3f},{hi2:.3f}]",
            "rho_perseed_mean_LOO": per_seed_loo.mean(),
            "rho_client_n_H": rho(client.n, client.H),
            "rho_client_n_w": rho(client.n, client.w),
            "rho_client_n_cos": rho(client.n, client.cos),
            "rho_client_n_cosLOO": rho(client.n, client.cos_loo),
            "rho_client_n_E": rho(client.n, client.E),
            "rho_client_n_lr": rho(client.n, client.lr),
            "rho_client_n_updnorm": rho(client.n, client.unorm),
            "rho_client_n_updnorm_step": rho(client.n, client.ups),
            "mean_weight_largest_client": client.groupby("seed").w.max().mean(),
            "mean_cos_minus_cosLOO": (g.cos_sim - g.cos_loo).mean(),
        })
        # size-stratified: within each size-rank stratum (1=smallest ... 5=largest), H vs cos across seeds
        for rk, s in client.assign(rk=client.groupby("seed").n.rank(method="first")).groupby("rk"):
            rows[-1][f"rho_H_cos_sizerank{int(rk)}"] = rho(s.H, s.cos)
            rows[-1][f"rho_H_cosLOO_sizerank{int(rk)}"] = rho(s.H, s.cos_loo)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "a02_heterogeneity_dependence.csv", index=False)
    d.to_csv(OUT / "a02_telemetry_with_loo.csv", index=False)

    # partial Spearman of H and cos controlling for log n (client level), via rank residuals
    pr = []
    for (m, a), g in d.groupby(["metodo", "alpha"]):
        c = g.groupby(["seed", "client_id"]).agg(H=("H_k", "first"), cos=("cos_sim", "mean"), cosl=("cos_loo", "mean"), n=("n_samples", "first")).reset_index()
        r = c[["H", "cos", "cosl", "n"]].rank()
        def resid(y, x):
            b = np.polyfit(x, y, 1)
            return y - np.polyval(b, x)
        pr.append({"method": m, "alpha": a,
                   "partial_rho_H_cos_given_n": np.corrcoef(resid(r.H, r.n), resid(r.cos, r.n))[0, 1],
                   "partial_rho_H_cosLOO_given_n": np.corrcoef(resid(r.H, r.n), resid(r.cosl, r.n))[0, 1]})
    pd.DataFrame(pr).to_csv(OUT / "a02_partial_given_size.csv", index=False)

    with open(OUT / "a02_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n\n")
        f.write(res.T.to_string() + "\n\n")
        f.write(pd.DataFrame(pr).to_string(index=False) + "\n")
    print("\n".join(lines))
    print(res.T.to_string())
    print(pd.DataFrame(pr).to_string(index=False))


if __name__ == "__main__":
    main()
