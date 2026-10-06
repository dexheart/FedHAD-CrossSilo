"""A03b - Empty clients (n_k=0) inside the controlled ablation and cells whose
'control' executes exactly the same training as the treatment (read-only)."""
import json
import pandas as pd
from common import REV, OUT

RAW = REV / "component_analysis_ablation/raw/official"
rows = []
for a in ("0.01", "0.1"):
    for s in range(42, 72):
        f = json.load(open(RAW / f"fingerprint__full_fedhad__CIFAR10__alpha-{a}__seed-{s}.json"))
        n = f["n_samples"]; ef, ex, ep = f["E_full"], f["E_fixed"], f["E_perm"]
        emp = [k for k in range(5) if n[k] == 0]
        real = [k for k in range(5) if n[k] > 0]
        rows.append(dict(alpha=float(a), seed=s, empty_clients=emp, E_full=ef, E_fixed=ex, E_perm=ep, n=n,
                         H=[round(h, 3) for h in f["H"]], LR=[round(x, 5) for x in f["LR_full"]],
                         fixed_exec_identical_to_full=all(ef[k] == ex[k] for k in real),
                         phantom_epoch_moved_by_perm=any(ef[k] != ep[k] for k in emp),
                         perm_upd_pct=f["budget_perm"]["updates_pct_vs_full"],
                         fixed_upd_pct=f["budget_fixed"]["updates_pct_vs_full"]))
d = pd.DataFrame(rows)
d.to_csv(OUT / "a03b_empty_clients_ablation.csv", index=False)
fm = pd.read_csv(REV / "component_analysis_ablation/tables/final_metrics_official.csv")
out = [d[d.empty_clients.map(len) > 0].to_string()]
ident = d[d.fixed_exec_identical_to_full]
out.append("cells where E_fixed == E_full on every non-empty client: " + str(ident[["alpha", "seed"]].values.tolist()))
for a, s in ident[["alpha", "seed"]].values:
    g = fm[(fm.alpha == a) & (fm.seed == s)].set_index("variant").acc_final
    out.append(f"  alpha={a} seed={s}: full - lr_only = {100*(g.full_fedhad-g.lr_only_matched):+.2f} pp ; "
               f"epoch_only - fixed = {100*(g.epoch_only-g.fixed_matched):+.2f} pp  (identical executed allocation => pure run-to-run noise)")
open(OUT / "a03b_summary.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
