"""A05 - Report integrity, hardware provenance and initialisation evidence
(Sections 5.6 and 5.8).

For every raw report of both historical campaigns:
  * completeness: centralised evaluations == rounds + 1, every client participates
    in every round (rounds_participated), total FLOPs consistent;
  * hardware (GPU/CPU) and timestamp;
  * round-0 centralised accuracy/loss, which is a fingerprint of the initial global
    model on a fixed test set: if two methods started from identical weights on the
    same seed, their round-0 accuracy AND loss must coincide exactly.
Also reads the 300 ablation fingerprints (initial_weights_sha256).
"""
from __future__ import annotations

import json

import pandas as pd

from common import OUT, REV, CAMPAIGN_05, CAMPAIGN_001, parse_report, seed_of, alpha_of


def main():
    rows = []
    for camp in (CAMPAIGN_05, CAMPAIGN_001):
        for p in sorted(camp.rglob("*.txt")):
            rel = p.relative_to(camp).parts
            r = parse_report(p)
            R = int(r.get("NUM_ROUNDS", 0) or 0)
            cl = r["clients"]
            part = [c.get("rounds_participated", R) for c in cl.values()]
            rows.append({
                "campaign": camp.name, "method_dir": rel[0], "tag": rel[1], "subdir": rel[2] if len(rel) > 3 else "",
                "nested_copy": rel[1].endswith("-results"),
                "method": r["method"], "dataset": r["dataset"], "alpha": alpha_of(p), "seed": seed_of(p),
                "signature": r.get("HEURISTICS_SIGNATURE"), "epochs_decay": r.get("EPOCHS_DECAY_FACTOR"),
                "lr_decay": r.get("LR_DECAY_FACTOR"), "delay": r.get("COMMUNICATION_DELAY"),
                "clients_setup": r.get("CLIENT_SETUP"), "rounds": R,
                "n_central_evals": len(r["acc_hist"]), "n_fit_rounds": len(r["flops_round"]),
                "min_client_participation": min(part) if part else None,
                "complete": (len(r["acc_hist"]) == R + 1) and (len(r["flops_round"]) == R) and (not part or min(part) == R),
                "acc_r0": r["acc_hist"][0][1] if r["acc_hist"] else None,
                "loss_r0": r["loss_hist"][0][1] if r["loss_hist"] else None,
                "acc_final": r["acc_final"], "TF": (r["total_flops"] or 0) / 1e12,
                "gpu": r["gpu"], "cpu": r["cpu"], "datetime": r["datetime"], "path": r["path"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a05_all_reports.csv", index=False)

    inc = df[~df.complete & ~df.nested_copy]
    hw = df[~df.nested_copy].groupby(["campaign", "tag", "method", "gpu", "cpu"], dropna=False).size().reset_index(name="runs")

    # initialisation fingerprint: compare round-0 metrics across methods on same (tag, dataset, alpha, seed)
    base = df[(~df.nested_copy) & (df.tag.isin(["test1_convergencia", "test2_robustez_alpha", "test7_plato", "test8_femnist"]))
              & (df.signature.isna() | (df.signature == "E1L1F1"))]
    key = ["campaign", "tag", "dataset", "alpha", "seed"]
    piv = base.pivot_table(index=key, columns="method", values=["acc_r0", "loss_r0"], aggfunc="first")
    ident = []
    meths = sorted(base.method.dropna().unique())
    for i, a in enumerate(meths):
        for b in meths[i + 1:]:
            if ("loss_r0", a) not in piv or ("loss_r0", b) not in piv:
                continue
            both = piv[[("loss_r0", a), ("loss_r0", b), ("acc_r0", a), ("acc_r0", b)]].dropna()
            same = ((both[("loss_r0", a)] == both[("loss_r0", b)]) & (both[("acc_r0", a)] == both[("acc_r0", b)]))
            ident.append({"pair": f"{a} vs {b}", "cells": len(both), "identical_round0_acc_and_loss": int(same.sum())})
    ident = pd.DataFrame(ident)

    # within-method: does the same method give identical round-0 across tags for the same seed/dataset/alpha?
    wm = []
    for m, g in base.groupby("method"):
        gg = g.groupby(["campaign", "dataset", "alpha", "seed"]).loss_r0.nunique()
        wm.append({"method": m, "cells_with_>1_tag": int((g.groupby(["campaign", "dataset", "alpha", "seed"]).size() > 1).sum()),
                   "cells_where_round0_loss_differs": int((gg > 1).sum())})
    wm = pd.DataFrame(wm)

    # ablation fingerprints
    RAW = REV / "component_analysis_ablation" / "raw" / "official"
    fh = []
    for f in RAW.glob("fingerprint__*.json"):
        j = json.load(open(f))
        fh.append({"variant": j["variant"], "alpha": j["alpha"], "seed": j["seed"], "sha": j["initial_weights_sha256"]})
    fh = pd.DataFrame(fh)
    grp = fh.groupby(["alpha", "seed"]).sha.nunique()
    alpha_shared = fh.groupby("seed").sha.nunique()

    with open(OUT / "a05_summary.txt", "w") as o:
        o.write(f"reports scanned: {len(df)} (nested duplicate copies: {int(df.nested_copy.sum())})\n")
        o.write(f"incomplete canonical reports: {len(inc)}\n")
        o.write(inc[["campaign", "tag", "method", "dataset", "alpha", "seed", "signature", "epochs_decay", "lr_decay", "n_central_evals", "min_client_participation", "TF", "path"]].to_string(index=False) + "\n\n")
        o.write("Hardware per campaign/tag/method:\n" + hw.to_string(index=False) + "\n\n")
        o.write("Round-0 identity across methods (same campaign/tag/dataset/alpha/seed):\n" + ident.to_string(index=False) + "\n\n")
        o.write("Same method, same seed, different tags -> round-0 loss differs?\n" + wm.to_string(index=False) + "\n\n")
        o.write(f"Ablation fingerprints: {len(fh)}; distinct init hashes per (alpha,seed): max={grp.max()}\n")
        o.write(f"Distinct init hashes per seed across the two alphas: max={alpha_shared.max()} (1 = same init for both alphas)\n")
    print(open(OUT / "a05_summary.txt").read()[:12000])


if __name__ == "__main__":
    main()
