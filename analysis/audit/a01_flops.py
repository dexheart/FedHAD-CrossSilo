"""A01 - Independent reconstruction of the FLOP accounting (Sections 5.1 and 5.5, Table 2).

1. Extracts the three network classes from the audited source by AST (no module-level
   simulation code is executed), profiles them with thop exactly as the scripts do,
   and recomputes the MAC count analytically layer by layer.
2. Rebuilds, for every fixed-epoch and FedHAD report of the principal families, the
   nominal total (full local sample counts) and the executed total (drop_last=True,
   floor(n/32)*32 samples per epoch), and compares both with the reported total.
Outputs: out/a01_*.csv and out/a01_summary.txt
"""
from __future__ import annotations

import ast
import math
import sys
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ROOT, OUT, CAMPAIGN_05, CAMPAIGN_001, parse_report, seed_of, alpha_of

SRC = ROOT / "algoritmos" / "FedHAD_Final_2.0.py"
BATCH = 32


def load_models():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    ns = {"nn": nn, "F": F, "torch": torch}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Net_"):
            exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
    return {k: v for k, v in ns.items() if k.startswith("Net_")}


def analytic_macs(net, shape):
    """Per-layer MACs: conv = Cout*H*W*Cin*k*k (+bias ignored), linear = in*out,
    BN (thop convention, eval) = 2*numel of output. Pool/ReLU/Dropout = 0."""
    rows, x = [], torch.zeros(1, *shape)
    hooks = []

    def hook(name):
        def f(m, inp, out):
            if isinstance(m, nn.Conv2d):
                macs = out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            elif isinstance(m, nn.Linear):
                macs = m.in_features * m.out_features
            elif isinstance(m, nn.BatchNorm2d):
                macs = 2 * out.numel()
            else:
                macs = 0
            rows.append({"layer": name, "type": type(m).__name__,
                         "out_shape": "x".join(map(str, out.shape[1:])), "macs": macs})
        return f

    for n, m in net.named_modules():
        if n and not list(m.children()):
            hooks.append(m.register_forward_hook(hook(n)))
    net.eval()
    with torch.no_grad():
        net(x)
    for h in hooks:
        h.remove()
    return rows


def thop_flops(net, shape):
    from thop import profile
    macs, params = profile(net, inputs=(torch.randn(1, *shape),), verbose=False)
    return macs, params


def main():
    models = load_models()
    spec = {"MNIST/FashionMNIST": ("Net_MNIST_Fashion", (1, 28, 28)),
            "FEMNIST": ("Net_FEMNIST", (1, 28, 28)),
            "CIFAR10": ("Net_CIFAR10", (3, 32, 32))}
    per_model, layer_rows = [], []
    for label, (cls, shape) in spec.items():
        net = models[cls]()
        rows = analytic_macs(net, shape)
        for r in rows:
            layer_rows.append({"model": cls, **r})
        a_macs = sum(r["macs"] for r in rows)
        macs, params = thop_flops(models[cls](), shape)
        per_model.append({
            "dataset": label, "class": cls, "input_shape": "x".join(map(str, shape)),
            "params": int(sum(p.numel() for p in net.parameters())),
            "thop_macs": macs, "analytic_macs": a_macs,
            "fwd_flops(2*MAC)": 2 * macs, "full_flops(3*fwd)": 6 * macs,
            "conv_linear_macs_only": sum(r["macs"] for r in rows if r["type"] in ("Conv2d", "Linear")),
        })
    pm = pd.DataFrame(per_model)
    pm.to_csv(OUT / "a01_per_sample_flops.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(OUT / "a01_layer_macs.csv", index=False)

    # ---- per-report reconstruction ----
    fams = [
        (CAMPAIGN_05, "test1_convergencia", ["MNIST", "FashionMNIST", "CIFAR10"]),
        (CAMPAIGN_05, "test2_robustez_alpha", ["MNIST", "FashionMNIST", "CIFAR10"]),
        (CAMPAIGN_05, "test8_femnist", ["FEMNIST"]),
        (CAMPAIGN_05, "test7_plato", ["CIFAR10"]),
        (CAMPAIGN_001, "test7_plato", ["CIFAR10"]),
        (CAMPAIGN_001, "test8_femnist", ["FEMNIST"]),
    ]
    methods = ["FedAVG", "FedAvgM", "FedProx", "FedNova", "FedHAD"]
    rep = []
    for camp, tag, dss in fams:
        for ds in dss:
            for m in methods:
                d = camp / f"{m}-results" / tag / (f"{ds}-Standard" if m == "FedHAD" else ds)
                for p in sorted(d.glob("*.txt")):
                    r = parse_report(p)
                    R = int(r.get("NUM_ROUNDS", 0) or 0)
                    ff = r["flops_full"]
                    cl = r["clients"]
                    if not cl or ff is None:
                        continue
                    nominal = executed = 0.0
                    n_tot = ex_tot = 0
                    upd = 0
                    partial = False
                    for cid, c in cl.items():
                        e = c.get("epochs_avg", 5.0)
                        rr = c.get("rounds_participated", R)
                        partial |= rr != R
                        n = c["n"]
                        n_exec = (n // BATCH) * BATCH if n > 1 else n
                        nominal += ff * n * e * rr
                        executed += ff * n_exec * e * rr
                        n_tot += n
                        ex_tot += n_exec
                        upd += (n // BATCH) * e * rr
                    rep.append({
                        "campaign": camp.name, "tag": tag, "dataset": ds, "method": m,
                        "alpha": alpha_of(p), "seed": seed_of(p), "rounds": R,
                        "n_clients": len(cl), "train_samples": n_tot,
                        "samples_exec_per_epoch": ex_tot,
                        "flops_full_per_sample": ff,
                        "reported_TF": (r["total_flops"] or 0) / 1e12,
                        "nominal_recomputed_TF": nominal / 1e12,
                        "executed_recomputed_TF": executed / 1e12,
                        "optimizer_steps": upd,
                        "partial_participation": partial,
                        "gpu": r["gpu"], "cpu": r["cpu"],
                    })
    df = pd.DataFrame(rep)
    df["reported_minus_nominal_TF"] = df.reported_TF - df.nominal_recomputed_TF
    df["overcount_pct_vs_executed"] = 100 * (df.nominal_recomputed_TF / df.executed_recomputed_TF - 1)
    df.to_csv(OUT / "a01_per_run_flops.csv", index=False)

    g = df.groupby(["campaign", "tag", "dataset", "alpha", "method"], dropna=False)
    summ = g.agg(n=("seed", "size"), train_samples=("train_samples", "mean"),
                 reported_TF=("reported_TF", "mean"), nominal_TF=("nominal_recomputed_TF", "mean"),
                 executed_TF=("executed_recomputed_TF", "mean"),
                 max_abs_rep_minus_nom=("reported_minus_nominal_TF", lambda s: s.abs().max()),
                 overcount_pct=("overcount_pct_vs_executed", "mean"),
                 overcount_pct_max=("overcount_pct_vs_executed", "max"),
                 steps=("optimizer_steps", "mean"),
                 partial=("partial_participation", "sum")).reset_index()
    summ.to_csv(OUT / "a01_summary_by_cell.csv", index=False)

    # per-client overcount for FEMNIST (the small clients)
    fem = []
    for p in sorted((CAMPAIGN_05 / "FedAVG-results/test8_femnist/FEMNIST").glob("*.txt")):
        r = parse_report(p)
        for cid, c in r["clients"].items():
            n = c["n"]
            fem.append({"seed": seed_of(p), "client": cid, "n_train": n, "batches": n // BATCH,
                        "dropped_per_epoch": n - (n // BATCH) * BATCH,
                        "dropped_pct": 100 * (n - (n // BATCH) * BATCH) / n})
    pd.DataFrame(fem).to_csv(OUT / "a01_femnist_client_droplast.csv", index=False)

    with open(OUT / "a01_summary.txt", "w") as f:
        f.write(pm.to_string(index=False) + "\n\n")
        f.write(summ.to_string(index=False) + "\n")
    print(pm.to_string(index=False))
    print(summ.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
