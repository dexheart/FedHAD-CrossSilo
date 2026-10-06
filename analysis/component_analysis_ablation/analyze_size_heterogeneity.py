#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline check: is the anti-correlation between client size and heterogeneity
structural, or an artifact of the two seeds used in the smoke test?

No training is executed. Partitions are regenerated deterministically by
importing the campaign's own copied code, exactly as the runs do, and only the
allocation arithmetic is evaluated.

Why this matters
----------------
Matching minibatch updates requires favouring large clients. If client size is
strongly anti-correlated with H_k, then an H-blind allocation that matches
updates can drift towards the H-driven one, weakening `fixed_matched` as a
control. The smoke grid measured rho(n_k, H_k) = -1.0 in several cells; this
script establishes whether that holds across all 30 seeds.

H_k is used ONLY to report diagnostics after the fact. It is never used to select
or optimise any allocation: `fixed_matched` remains H-blind (argmin |U - U_full|
over the flat family, excluding E_full, with the documented tie-break) and
`permuted_allocation` remains a deterministic enumeration matched on updates.

Usage
-----
    python analysis/component_analysis_ablation/analyze_size_heterogeneity.py
    python analysis/component_analysis_ablation/analyze_size_heterogeneity.py --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "algoritmos" / "component_analysis_ablation"
sys.path.insert(0, str(ROOT))
import config_ablation as C  # noqa: E402


@contextlib.contextmanager
def quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def import_for_seed(seed):
    """
    One import per seed: `dirichlet_split_noniid(..., seed=SEED)` binds SEED as a
    default argument at import time and `load_data()` never passes it, so patching
    the module global afterwards would silently regenerate the seed-42 partition.
    ALPHA, by contrast, is read at call time and can be varied freely.
    """
    os.environ.update({
        "FL_DATASET": C.DATASET, "FL_USE_ENERGY": "0",
        "FL_CLIENT_SETUP": str(C.CLIENT_SETUP), "FL_SEED": str(seed),
        "FL_NUM_ROUNDS": str(C.NUM_ROUNDS), "FL_ABLATION_VARIANT": "full_fedhad",
        "FL_ALPHA": str(C.ALPHAS[0]),
    })
    name = f"_size_het_{seed}"
    spec = importlib.util.spec_from_file_location(name, C.ENTRY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    with quiet():
        spec.loader.exec_module(mod)
    return mod, name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seeds", type=int, nargs="*", default=C.SEEDS_OFFICIAL)
    args = ap.parse_args()

    import ablation_policy as AP

    prev_cwd = Path.cwd()
    os.chdir(C.CODE)  # the copied script resolves "./data" relatively
    rows = []
    try:
        for i, seed in enumerate(args.seeds, 1):
            mod, name = import_for_seed(seed)
            for alpha in C.ALPHAS:
                mod.ALPHA = float(alpha)
                with quiet():
                    trainloaders, _val, _test = mod.load_data()
                    h = [mod.normalizar_heterogeneidade(
                        mod.calcular_heterogeneidade(str(c), dl, mod.NUM_CLASSES),
                        mod.NUM_CLASSES) for c, dl in enumerate(trainloaders)]
                    st = AP.build_allocation(
                        trainloaders=trainloaders, h_values=h, seed=seed,
                        base_epochs=mod.BASE_EPOCHS, min_epochs=mod.MIN_EPOCHS,
                        base_lr=mod.BASE_LR, min_lr=mod.MIN_LR,
                        epochs_fn=mod.define_epochs_dinamicas,
                        lr_fn=mod.define_lr_dinamico,
                        flops_per_sample_full=mod.FLOPS_PER_SAMPLE_FULL)
                e_full, e_fixed, e_perm = st["E_full"], st["E_fixed"], st["E_perm"]
                bf, bx, bp = st["budget_full"], st["budget_fixed"], st["budget_perm"]
                rows.append({
                    "alpha": alpha, "seed": seed,
                    "n_samples": ";".join(map(str, st["n_samples"])),
                    "n_batches": ";".join(map(str, st["n_batches"])),
                    "H": ";".join(f"{x:.6f}" for x in h),
                    "E_full": ";".join(map(str, e_full)),
                    "E_fixed": ";".join(map(str, e_fixed)),
                    "E_perm": ";".join(map(str, e_perm)),
                    "rho_n_vs_H": AP._spearman(st["n_samples"], h),
                    "rho_E_full_vs_H": AP._spearman(e_full, h),
                    "rho_E_fixed_vs_H": AP._spearman(e_fixed, h),
                    "rho_E_perm_vs_H": AP._spearman(e_perm, h),
                    "hamming_fixed_vs_full": sum(1 for a, b in zip(e_fixed, e_full) if a != b),
                    "hamming_perm_vs_full": sum(1 for a, b in zip(e_perm, e_full) if a != b),
                    "U_full": bf["total_updates"],
                    "updates_pct_fixed": round(bx["updates_pct_vs_full"], 4),
                    "updates_pct_perm": round(bp["updates_pct_vs_full"], 4),
                    "perm_n_unique_nontrivial": st["perm_n_unique_nontrivial"],
                })
                del st
            del sys.modules[name], mod
            print(f"  seed {seed} done ({i}/{len(args.seeds)})", flush=True)
    finally:
        os.chdir(prev_cwd)

    C.TABLES.mkdir(parents=True, exist_ok=True)
    out = C.TABLES / "size_heterogeneity_offline.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 78)
    print("OFFLINE: client size vs heterogeneity, all seeds (no training)")
    print("=" * 78)
    for alpha in C.ALPHAS:
        sub = [r for r in rows if r["alpha"] == alpha]
        for field, label in [("rho_n_vs_H", "rho(n_k, H_k)"),
                             ("rho_E_full_vs_H", "rho(E_full, H_k)"),
                             ("rho_E_fixed_vs_H", "rho(E_fixed, H_k)"),
                             ("rho_E_perm_vs_H", "rho(E_perm, H_k)")]:
            vals = np.array([r[field] for r in sub if r[field] is not None], float)
            print(f"  alpha={alpha:<5} {label:<20} n={len(vals):2d}  "
                  f"mean {vals.mean():+.3f}  median {np.median(vals):+.3f}  "
                  f"[{vals.min():+.3f}, {vals.max():+.3f}]")
        upf = np.abs([r["updates_pct_fixed"] for r in sub])
        upp = np.abs([r["updates_pct_perm"] for r in sub])
        print(f"  alpha={alpha:<5} |dU| fixed  : max {upf.max():.2f}%  "
              f"median {np.median(upf):.2f}%  <=1%: {(upf <= 1).sum()}/{len(upf)}")
        print(f"  alpha={alpha:<5} |dU| perm   : max {upp.max():.2f}%  "
              f"median {np.median(upp):.2f}%  <=1%: {(upp <= 1).sum()}/{len(upp)}")
        hf = np.array([r["hamming_fixed_vs_full"] for r in sub])
        hp = np.array([r["hamming_perm_vs_full"] for r in sub])
        print(f"  alpha={alpha:<5} Hamming fixed vs full: min {hf.min()} "
              f"(cells with <=1: {(hf <= 1).sum()}/{len(hf)})")
        print(f"  alpha={alpha:<5} Hamming perm  vs full: min {hp.min()} "
              f"(cells with <=1: {(hp <= 1).sum()}/{len(hp)})")
        print()
    print(f"Written -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
