"""A08 - Long-horizon study inventory and numerical check of the loss panels
(Section 6.7, Table 11 and Figure 13).

Reads the canonical test7_plato (CIFAR-10, 50 rounds) and 50-round FEMNIST reports.
For every method x panel computes, on the seed-mean curve and per seed: initial
loss, minimum, round of minimum, final loss, final - min, and OLS slopes over the
last 10 / 20 rounds. Panel mapping assumed from the generator
(analysis/revision_consolidated_tables_figures/generate_long_horizon_alpha_comparison.py):
(a) accuracy alpha=0.5, (b) loss alpha=0.5, (c) accuracy alpha=0.01, (d) loss alpha=0.01.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT, CAMPAIGN_05, CAMPAIGN_001, parse_report, seed_of

M = ["FedAVG", "FedAvgM", "FedProx", "FedNova", "FedHAD"]


def slope(y):
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]


def main():
    panels = [("(b) loss, alpha=0.5", CAMPAIGN_05, "test7_plato", "CIFAR10"),
              ("(d) loss, alpha=0.01", CAMPAIGN_001, "test7_plato", "CIFAR10"),
              ("FEMNIST 50 rounds", CAMPAIGN_001, "test8_femnist", "FEMNIST")]
    inv, stats, seeds = [], [], []
    for lab, camp, tag, ds in panels:
        for m in M:
            d = camp / f"{m}-results" / tag / (f"{ds}-Standard" if m == "FedHAD" else ds)
            reps = [parse_report(p) for p in sorted(d.glob("*.txt"))] if d.exists() else []
            reps = [r for r in reps if int(r.get("NUM_ROUNDS", 0)) == 50]
            inv.append({"panel": lab, "method": m, "runs": len(reps),
                        "seeds": f"{min(seed_of(__import__('pathlib').Path(r['path'])) for r in reps)}-{max(seed_of(__import__('pathlib').Path(r['path'])) for r in reps)}" if reps else "-",
                        "alpha": reps[0].get("ALPHA") if reps else None,
                        "acc_final_mean": np.mean([r["acc_final"] for r in reps]) if reps else None,
                        "acc_final_sd": np.std([r["acc_final"] for r in reps], ddof=1) if len(reps) > 1 else None,
                        "TF_mean": np.mean([r["total_flops"] for r in reps]) / 1e12 if reps else None,
                        "gpu": reps[0]["gpu"] if reps else None})
            if not reps:
                continue
            L = np.array([[v for _, v in r["loss_hist"]] for r in reps])
            A = np.array([[v for _, v in r["acc_hist"]] for r in reps])
            mc = L.mean(0)
            stats.append({"panel": lab, "method": m, "loss_r0": mc[0], "loss_min": mc.min(),
                          "round_of_min": int(mc.argmin()), "loss_r10": mc[10], "loss_final": mc[-1],
                          "final_minus_min": mc[-1] - mc.min(),
                          "slope_last10_per_round": slope(mc[-10:]), "slope_last20_per_round": slope(mc[-20:]),
                          "acc_r10": A.mean(0)[10], "acc_max": A.mean(0).max(), "round_acc_max": int(A.mean(0).argmax()),
                          "acc_final": A.mean(0)[-1],
                          "seeds_final_gt_min_by_0.05": int(((L[:, -1] - L.min(1)) > 0.05).sum()),
                          "seeds_positive_slope_last20": int(sum(slope(l[-20:]) > 0 for l in L)),
                          "n": len(L)})
    inv, st = pd.DataFrame(inv), pd.DataFrame(stats)
    inv.to_csv(OUT / "a08_long_horizon_inventory.csv", index=False)
    st.to_csv(OUT / "a08_fig13_loss_panels.csv", index=False)
    pd.set_option("display.width", 250)
    txt = inv.to_string(index=False) + "\n\n" + st.round(4).to_string(index=False)
    open(OUT / "a08_summary.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
