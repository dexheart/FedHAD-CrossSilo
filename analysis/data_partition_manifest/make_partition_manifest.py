#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partition manifest and cross-method identity verification.

This script is a reproducibility artifact for the paper's supplementary material.
It runs NO training and NO federated simulation. It only imports the partitioning
routines from the existing per-method scripts and regenerates the client
partitions for the full experimental grid, recording a cryptographic fingerprint
of every partition.

Claim under test
----------------
All five federated methods (FedAvg, FedAvgM, FedProx, FedHAD, FedNova) operate on
*identical* client partitions for a given (dataset, alpha, n_clients, seed). This
holds by construction: `dirichlet_split_noniid` calls `np.random.seed(seed)` as
its first executable statement, which makes it a pure function of
(labels, n_clients, alpha, seed) and immune to however much of the global RNG
stream each method consumed beforehand (model initialisation, sampling order,
scheduler warm-up, etc.).

Two controls are reported alongside the identity check:

  1. Negative control - distinct seeds must yield distinct partitions. Without
     this, the identity result would be vacuous (a degenerate partitioner that
     ignores the seed would trivially "pass").
  2. RNG stress test - roughly 1.5M draws are consumed from numpy, torch and the
     `random` module, and the global generators are reseeded to 999 immediately
     before partitioning. The resulting fingerprint must be unchanged.

Outputs (written to ./artifacts/partitions/ relative to this script)
-------------------------------------------------------------------
  partition_manifest.csv      one row per (scenario, method, client) - raw manifest
  partition_verification.csv  one row per scenario - aggregated verification
  partition_verification.tex  booktabs table for the supplementary material
  model_init_check.csv        only with --check-init

Exit status
-----------
Non-zero if any scenario fails any check, so the script doubles as a regression
test rather than being merely a table generator.

Usage
-----
    python analysis/data_partition_manifest/make_partition_manifest.py                # full grid
    python analysis/data_partition_manifest/make_partition_manifest.py --quick        # reduced grid (smoke test)
    python analysis/data_partition_manifest/make_partition_manifest.py --check-init   # also fingerprint model init
    python analysis/data_partition_manifest/make_partition_manifest.py --skip-femnist
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.util
import io
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ALGORITMOS_DIR = PROJECT_ROOT / "algoritmos"
OUT_DIR = PROJECT_ROOT / "results" / "data_partition_manifest" / "artifacts" / "partitions"

# The per-method scripts read their configuration from FL_* environment
# variables at import time. Importing them with the cheapest dataset keeps the
# import fast; the partitioning functions themselves take the dataset as an
# argument, so this choice does not affect any result below.
os.environ.setdefault("FL_DATASET", "MNIST")
os.environ.setdefault("FL_USE_ENERGY", "false")

import numpy as np  # noqa: E402  (imported after the env vars above are set)
import torch  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

# Display name -> script filename. Note the display name is "FedAvg", not the
# "FedAVG" spelling used internally by the script filename.
METHOD_FILES = {
    "FedAvg": "FedAVG_Final.py",
    "FedAvgM": "FedAvgM_Final.py",
    "FedProx": "FedProx_Final.py",
    "FedHAD": "FedHAD_Final_2.0.py",
    "FedNova": "FedNova_Final.py",
}
METHOD_ORDER = ["FedAvg", "FedAvgM", "FedProx", "FedHAD", "FedNova"]

# Experimental grid.
DATASETS = ["MNIST", "FASHION_MNIST", "CIFAR10"]
DATASET_LABEL = {
    "MNIST": "MNIST",
    "FASHION_MNIST": "FashionMNIST",
    "CIFAR10": "CIFAR10",
    "FEMNIST": "FEMNIST",
}
ALPHAS = [1.0, 0.5, 0.1, 0.01]
N_CLIENTS_GRID = [3, 5, 10]
SEEDS = list(range(42, 72))  # 30 seeds
# FEMNIST must mirror the `test8_femnist` block of the experimental campaign
# exactly: 10 clients (writers) and the same 30 seeds. Anything narrower would
# make the manifest describe a federation the campaign never runs.
FEMNIST_SETUPS = [10]
FEMNIST_SEEDS = SEEDS

N_CLASSES = {"MNIST": 10, "FASHION_MNIST": 10, "CIFAR10": 10, "FEMNIST": 62}

# Model architectures fingerprinted by --check-init. Every method script defines
# all three classes regardless of which dataset it was imported with.
ARCHITECTURES = ["Net_MNIST_Fashion", "Net_CIFAR10", "Net_FEMNIST"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def suppressed_output():
    """Silence the imported scripts, which print progress messages on import."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def sha256_of(obj) -> str:
    """Stable SHA-256 of a JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def import_methods() -> dict:
    """Import every per-method script once and return {display_name: module}."""
    sys.path.insert(0, str(ALGORITMOS_DIR))
    modules = {}
    for name in METHOD_ORDER:
        path = ALGORITMOS_DIR / METHOD_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"Method script not found: {path}")
        spec = importlib.util.spec_from_file_location(f"_method_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"_method_{name}"] = mod
        with suppressed_output():
            spec.loader.exec_module(mod)
        modules[name] = mod
    return modules


def load_trainsets(datasets) -> dict:
    """Load each torchvision training set once; the partitioners take it as an argument."""
    from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
    from torchvision.transforms import Compose, Normalize, ToTensor

    data_root = str(PROJECT_ROOT / "data")
    out = {}
    with suppressed_output():
        for ds in datasets:
            if ds == "MNIST":
                trf = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
                out[ds] = MNIST(data_root, train=True, download=True, transform=trf)
            elif ds == "FASHION_MNIST":
                trf = Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])
                out[ds] = FashionMNIST(data_root, train=True, download=True, transform=trf)
            elif ds == "CIFAR10":
                trf = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465),
                                                     (0.2023, 0.1994, 0.2010))])
                out[ds] = CIFAR10(data_root, train=True, download=True, transform=trf)
            else:
                raise ValueError(f"Unsupported dataset: {ds}")
    return out


def consume_rng_and_reseed(n_numpy=1_000_000, n_torch=500_000, n_py=10_000):
    """
    RNG stress test: burn ~1.5M draws across numpy/torch/random and then reseed
    every global generator to 999, immediately before partitioning. A partitioner
    that is a pure function of its arguments must be unaffected.
    """
    np.random.rand(n_numpy)
    np.random.randint(0, 10, 50_000)
    np.random.shuffle(np.arange(10_000))
    torch.randn(n_torch)
    [random.random() for _ in range(n_py)]
    np.random.seed(999)
    torch.manual_seed(999)
    random.seed(999)


def heterogeneity_from_counts(counts: np.ndarray, n_classes: int, mod) -> tuple:
    """
    H_k exactly as the method scripts compute it: the normalised standard
    deviation of the per-class sample counts (population std, i.e. unbiased=False),
    divided by the mean, then mapped to [0, 1] by the scripts' own
    `normalizar_heterogeneidade` (upper bound sqrt(K-1)).
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0 or counts.mean() == 0:
        return 0.0, 0.0
    h_raw = float(counts.std(ddof=0) / counts.mean())
    return h_raw, float(mod.normalizar_heterogeneidade(h_raw, n_classes))


def summarize_partition(indices, counts, n_classes, mod, extra=None):
    """
    Fingerprint a partition and summarise it per client. The raw index lists are
    hashed here and then dropped: retaining them for the whole grid would cost
    several GB (one list per client per method per scenario, over datasets with
    up to 50k samples), while the fingerprints are all the verification needs.
    """
    clients = []
    for cid, (idx, cnt) in enumerate(zip(indices, counts)):
        arr = np.asarray(cnt, dtype=np.int64)
        covered = int((arr > 0).sum()) if arr.size else 0
        h_raw, h_norm = heterogeneity_from_counts(arr, n_classes, mod)
        record = {
            "client_id": cid,
            "n_k": len(idx),
            "n_classes_covered": covered,
            "class_coverage_ratio": round(covered / n_classes, 6),
            "H_k_raw": round(h_raw, 8),
            "H_k_norm": round(h_norm, 8),
            "sha256_indices_client": sha256_of(idx),
            "sha256_counts_client": sha256_of(cnt),
            "label_counts": ";".join(str(c) for c in cnt),
            # Defaults for the Dirichlet path; overridden by `extra` for FEMNIST.
            "writer_id": "",
            "writer_natural_id": "",
            "n_train": "",
            "n_val": "",
            "role": "client",
            "excluded_reason": "",
        }
        if extra is not None:
            record.update(extra[cid])
        clients.append(record)
    return {
        "sha_indices": sha256_of(indices),
        "sha_counts": sha256_of(counts),
        "clients": clients,
    }


def client_records(subsets, labels_all, n_classes):
    """Per-client indices and label counts for one partition."""
    indices, counts = [], []
    for sub in subsets:
        idx = sorted(int(i) for i in sub.indices)
        indices.append(idx)
        counts.append(np.bincount(labels_all[idx], minlength=n_classes).astype(int).tolist())
    return indices, counts


# --------------------------------------------------------------------------- #
# Dirichlet scenarios
# --------------------------------------------------------------------------- #
def partition_dirichlet(mod, trainset, n_clients, alpha, seed):
    with suppressed_output():
        return mod.dirichlet_split_noniid(trainset, n_clients=n_clients, alpha=alpha, seed=seed)


def run_dirichlet_scenarios(modules, trainsets, datasets, alphas, n_clients_grid, seeds):
    """Yield one dict per scenario, holding per-method per-client records."""
    for ds in datasets:
        trainset = trainsets[ds]
        n_classes = N_CLASSES[ds]
        labels_all = np.asarray(trainset.targets)
        for alpha in alphas:
            for k in n_clients_grid:
                for seed in seeds:
                    scenario = {
                        "scenario_id": f"{DATASET_LABEL[ds]}_alpha-{alpha}_K-{k}_seed-{seed}",
                        "dataset": DATASET_LABEL[ds],
                        "partitioning": "dirichlet",
                        "alpha": alpha,
                        "n_clients": k,
                        "seed": seed,
                        "n_classes": n_classes,
                        "per_method": {},
                    }
                    for name in METHOD_ORDER:
                        mod = modules[name]
                        subs = partition_dirichlet(mod, trainset, k, alpha, seed)
                        idx, cnt = client_records(subs, labels_all, n_classes)
                        scenario["per_method"][name] = summarize_partition(
                            idx, cnt, n_classes, mod)

                    # RNG stress test: same scenario, hostile global RNG state.
                    consume_rng_and_reseed()
                    stress = {}
                    for name in METHOD_ORDER:
                        subs = partition_dirichlet(modules[name], trainset, k, alpha, seed)
                        stress[name] = sha256_of(
                            [sorted(int(i) for i in s.indices) for s in subs]
                        )
                    scenario["stress_hashes"] = stress
                    yield scenario


# --------------------------------------------------------------------------- #
# FEMNIST scenarios (natural partitioning by writer)
# --------------------------------------------------------------------------- #
def femnist_labels(hf_dataset):
    """
    Label column of a FEMNIST partition, respecting the dataset view.

    Do NOT reach for `hf_dataset.data` here. flwr-datasets partitions are *views*
    over the full ~814k-row table (their `_indices` mapping is set), so `.data`
    returns every row of the entire dataset instead of the writer's rows;
    indexing it with partition-local indices then yields labels belonging to
    unrelated samples, silently and plausibly. `with_format("numpy")` applies the
    view and also bypasses the image transform, so it is both correct and fast.
    """
    labels = np.asarray(hf_dataset.with_format("numpy")["character"], dtype=np.int64)
    # Guard against silently reading through the view. This exact bug produced
    # plausible, method-consistent but wrong label statistics, so it is checked
    # loudly rather than tolerated.
    if len(labels) != len(hf_dataset):
        raise RuntimeError(
            f"FEMNIST label column has {len(labels)} entries for a partition of "
            f"{len(hf_dataset)} samples: the dataset view was not applied."
        )
    return labels


def femnist_heldout_writers(testloader, n_classes):
    """
    Per-writer breakdown of the centralised held-out test set. Derived from the
    object the loader actually returned (its `writer_id` column) rather than by
    re-deriving the selection logic, so this cannot drift from the real code path.
    """
    ds = getattr(testloader, "dataset", None)
    if ds is None:
        return []
    try:
        # `with_format`, not `.data` — see the note in femnist_labels(): the
        # concatenated held-out set is a view, and `.data` would enumerate every
        # writer in the dataset instead of the held-out ones.
        formatted = ds.with_format("numpy")
        writer_ids = np.asarray(formatted["writer_id"])
        labels = np.asarray(formatted["character"], dtype=np.int64)
    except Exception:
        return []

    if len(writer_ids) != len(ds):
        raise RuntimeError(
            f"Held-out test set: read {len(writer_ids)} writer ids for {len(ds)} "
            f"samples: the dataset view was not applied."
        )
    n_writers = len(set(writer_ids.tolist()))
    # The loader caps the held-out set at min(100, remaining writers).
    if n_writers > 100:
        raise RuntimeError(
            f"Held-out test set spans {n_writers} writers, but the loader caps it "
            f"at 100: the wrong rows are being read."
        )

    rows = []
    for wid in sorted(set(writer_ids.tolist())):
        mask = writer_ids == wid
        cnt = np.bincount(labels[mask], minlength=n_classes).astype(int)
        covered = int((cnt > 0).sum())
        rows.append({
            "writer_natural_id": wid,
            "n_k": int(mask.sum()),
            "n_classes_covered": covered,
            "class_coverage_ratio": round(covered / n_classes, 6),
            "label_counts": ";".join(str(c) for c in cnt.tolist()),
            "counts": cnt.tolist(),
        })
    return rows


def partition_femnist(mod, setup, seed, with_heldout=True):
    """
    Call the script's own FEMNIST loader. It reads CLIENT_SETUP / NUM_CLIENTS /
    SEED from module globals rather than arguments, so they are set here first.

    This path was verified to reproduce the campaign's own loading path
    (FL_DATASET=FEMNIST) byte for byte: identical per-client index fingerprints
    and identical train/validation sizes. `load_data_femnist` reads neither
    DATASET nor NUM_CLASSES, so importing the module under a different dataset
    does not affect the partition.

    Returns (train indices, label counts, per-client metadata, held-out writers).
    """
    mod.CLIENT_SETUP = setup
    mod.NUM_CLIENTS = setup
    mod.SEED = seed
    with suppressed_output():
        trainloaders, valloaders, testloader = mod.load_data_femnist()

    writers = sorted(int(w) for w in mod.CLIENT_SETUP_CONFIG[setup]["femnist_writers"])
    n_classes = N_CLASSES["FEMNIST"]

    indices, counts, extra = [], [], []
    for cid, (tl, vl) in enumerate(zip(trainloaders, valloaders)):
        # A writer with no samples yields empty DataLoaders (a plain list, with
        # no `.indices`). The loader keeps the client slot rather than dropping
        # the writer, and so does this manifest: the row is emitted with n_k=0
        # and an explicit `excluded_reason` instead of being silently omitted.
        train_idx = sorted(int(i) for i in getattr(tl.dataset, "indices", []))
        val_idx = sorted(int(i) for i in getattr(vl.dataset, "indices", []))
        base = getattr(tl.dataset, "dataset", None)
        labels = femnist_labels(base) if base is not None else None

        if labels is not None and train_idx:
            cnt = np.bincount(labels[train_idx], minlength=n_classes).astype(int).tolist()
        else:
            cnt = [0] * n_classes

        # Two different identifiers are in play: the partition id (0..3596, what
        # CLIENT_SETUP_CONFIG lists) and the dataset's own writer_id string
        # (e.g. "f0320_41"). Both are recorded so the two manifests can be joined.
        natural_id = ""
        if base is not None:
            try:
                natural_id = str(base.with_format("numpy")["writer_id"][0])
            except Exception:
                natural_id = ""

        indices.append(train_idx)
        counts.append(cnt)
        extra.append({
            "writer_id": writers[cid] if cid < len(writers) else "",
            "writer_natural_id": natural_id,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "role": "client",
            "excluded_reason": "" if train_idx else "writer_has_no_training_samples",
        })

    # The RNG stress replica only needs the client fingerprints, so the (costly)
    # per-writer breakdown of the held-out test set is skipped there.
    heldout = femnist_heldout_writers(testloader, n_classes) if with_heldout else []

    # Each call rebuilds a FederatedDataset and concatenates ~100 held-out writer
    # partitions. Without dropping the references and collecting explicitly, RSS
    # grows by gigabytes across the grid.
    del trainloaders, valloaders, testloader
    gc.collect()

    return indices, counts, extra, heldout


def run_femnist_scenarios(modules, setups, seeds):
    scenario_index = -1
    for setup in setups:
        for seed in seeds:
            scenario_index += 1
            scenario = {
                "scenario_id": f"FEMNIST_natural_K-{setup}_seed-{seed}",
                "dataset": "FEMNIST",
                "partitioning": "natural",
                "alpha": "",
                "n_clients": setup,
                "seed": seed,
                "n_classes": N_CLASSES["FEMNIST"],
                "per_method": {},
            }
            heldout_by_method = {}
            for name in METHOD_ORDER:
                idx, cnt, extra, heldout = partition_femnist(modules[name], setup, seed)
                scenario["per_method"][name] = summarize_partition(
                    idx, cnt, N_CLASSES["FEMNIST"], modules[name], extra=extra)
                heldout_by_method[name] = heldout
            # The held-out test writers are method-independent; keep one copy and
            # verify the others agree.
            scenario["heldout"] = heldout_by_method[METHOD_ORDER[0]]
            ref_mod = modules[METHOD_ORDER[0]]
            scenario["heldout_h"] = {
                str(r["writer_natural_id"]): tuple(
                    round(x, 8) for x in heterogeneity_from_counts(
                        np.asarray(r["counts"]), N_CLASSES["FEMNIST"], ref_mod)
                )
                for r in scenario["heldout"]
            }
            scenario["heldout_identical"] = len({
                sha256_of([(str(r["writer_natural_id"]), r["counts"]) for r in h])
                for h in heldout_by_method.values()
            }) == 1

            # RNG stress replica. Reloading FEMNIST is expensive (each call
            # rebuilds the dataset and the ~100-writer held-out set), so instead
            # of replicating all five methods per scenario the method under test
            # is rotated across scenarios: every scenario still carries a stress
            # result, and every method is stress-tested on several scenarios.
            # The Dirichlet grid replicates all five methods in every scenario.
            stress_method = METHOD_ORDER[scenario_index % len(METHOD_ORDER)]
            consume_rng_and_reseed()
            idx, _cnt, _extra, _heldout = partition_femnist(
                modules[stress_method], setup, seed, with_heldout=False)
            scenario["stress_hashes"] = {stress_method: sha256_of(idx)}
            scenario["stress_method"] = stress_method
            yield scenario


# --------------------------------------------------------------------------- #
# Verification and output
# --------------------------------------------------------------------------- #
def verify(scenarios):
    """Compute per-scenario verification flags and the negative control."""
    # Negative control: within a (dataset, partitioning, alpha, n_clients) group,
    # every seed must produce a distinct fingerprint.
    groups = defaultdict(list)
    for sc in scenarios:
        key = (sc["dataset"], sc["partitioning"], sc["alpha"], sc["n_clients"])
        groups[key].append(sc)

    for sc in scenarios:
        per_method = sc["per_method"]
        idx_hashes = {m: v["sha_indices"] for m, v in per_method.items()}
        cnt_hashes = {m: v["sha_counts"] for m, v in per_method.items()}
        sc["sha256_indices"] = sorted(set(idx_hashes.values()))[0]
        sc["sha256_counts"] = sorted(set(cnt_hashes.values()))[0]
        sc["indices_identical"] = len(set(idx_hashes.values())) == 1
        sc["counts_identical"] = len(set(cnt_hashes.values())) == 1
        sc["method_hashes"] = idx_hashes
        sc["reseed_stress_passed"] = all(
            h == idx_hashes[m] for m, h in sc["stress_hashes"].items()
        )

    for key, group in groups.items():
        if len(group) < 2:
            # A single seed cannot support a negative control.
            for sc in group:
                sc["negative_control_passed"] = "NA"
            continue
        for sc in group:
            others = [o["sha256_indices"] for o in group if o["seed"] != sc["seed"]]
            sc["negative_control_passed"] = sc["sha256_indices"] not in others
    return scenarios


def write_manifest(scenarios, modules, path: Path):
    fields = [
        "scenario_id", "dataset", "partitioning", "alpha", "n_clients", "seed",
        "method", "client_id", "writer_id", "writer_natural_id", "role", "excluded_reason",
        "partition_granularity", "n_k", "n_train", "n_val", "n_classes_covered",
        "class_coverage_ratio", "H_k_raw", "H_k_norm",
        "sha256_indices_client", "sha256_counts_client",
        "sha256_indices_scenario", "sha256_counts_scenario", "label_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sc in scenarios:
            # Dirichlet fingerprints the full client partition (the 90/10 split
            # happens downstream in load_data); the FEMNIST loader already returns
            # the split, so there the fingerprint is over the training split.
            granularity = ("train_split" if sc["partitioning"] == "natural"
                           else "full_client_partition")
            for method in METHOD_ORDER:
                rec = sc["per_method"][method]
                for c in rec["clients"]:
                    row = {
                        "scenario_id": sc["scenario_id"],
                        "dataset": sc["dataset"],
                        "partitioning": sc["partitioning"],
                        "alpha": sc["alpha"],
                        "n_clients": sc["n_clients"],
                        "seed": sc["seed"],
                        "method": method,
                        "partition_granularity": granularity,
                        "sha256_indices_scenario": rec["sha_indices"],
                        "sha256_counts_scenario": rec["sha_counts"],
                    }
                    row.update(c)
                    w.writerow(row)


def write_femnist_heldout(scenarios, path: Path):
    """
    One row per (scenario, held-out writer). The held-out writers form the
    centralised test set and are disjoint from the client writers, so they are
    recorded separately rather than duplicated across the five methods (their
    agreement across methods is verified and reported instead).
    """
    fem = [sc for sc in scenarios if sc["partitioning"] == "natural"]
    fields = ["scenario_id", "dataset", "n_clients", "seed", "role", "writer_natural_id",
              "n_k", "n_classes_covered", "class_coverage_ratio",
              "H_k_raw", "H_k_norm", "label_counts"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sc in fem:
            for r in sc.get("heldout", []):
                h_raw, h_norm = sc["heldout_h"][str(r["writer_natural_id"])]
                w.writerow({
                    "scenario_id": sc["scenario_id"],
                    "dataset": sc["dataset"],
                    "n_clients": sc["n_clients"],
                    "seed": sc["seed"],
                    "role": "heldout_test",
                    "writer_natural_id": r["writer_natural_id"],
                    "n_k": r["n_k"],
                    "n_classes_covered": r["n_classes_covered"],
                    "class_coverage_ratio": r["class_coverage_ratio"],
                    "H_k_raw": h_raw,
                    "H_k_norm": h_norm,
                    "label_counts": r["label_counts"],
                })


def write_verification(scenarios, path: Path):
    fields = [
        "scenario_id", "dataset", "partitioning", "alpha", "n_clients", "seed",
        "n_methods", "sha_indices_identical", "sha_counts_identical",
        "negative_control_passed", "reseed_stress_passed",
        "sha256_indices", "sha256_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sc in scenarios:
            w.writerow({
                "scenario_id": sc["scenario_id"],
                "dataset": sc["dataset"],
                "partitioning": sc["partitioning"],
                "alpha": sc["alpha"],
                "n_clients": sc["n_clients"],
                "seed": sc["seed"],
                "n_methods": len(sc["per_method"]),
                "sha_indices_identical": sc["indices_identical"],
                "sha_counts_identical": sc["counts_identical"],
                "negative_control_passed": sc["negative_control_passed"],
                "reseed_stress_passed": sc["reseed_stress_passed"],
                "sha256_indices": sc["sha256_indices"],
                "sha256_counts": sc["sha256_counts"],
            })


def write_tex(scenarios, path: Path):
    """Condensed booktabs table, grouped by dataset and alpha."""
    groups = defaultdict(list)
    for sc in scenarios:
        groups[(sc["dataset"], sc["alpha"])].append(sc)

    def frac(group, key):
        applicable = [s for s in group if s[key] != "NA"]
        if not applicable:
            return "n/a"
        return f"{sum(1 for s in applicable if s[key] is True)}/{len(applicable)}"

    order = []
    for ds in ["MNIST", "FashionMNIST", "CIFAR10"]:
        for a in ALPHAS:
            if (ds, a) in groups:
                order.append((ds, a))
    for key in groups:
        if key not in order:
            order.append(key)

    rows = []
    for ds, alpha in order:
        g = groups[(ds, alpha)]
        alpha_txt = "---" if alpha == "" else f"{float(alpha):g}"
        rows.append(
            f"{ds} & {alpha_txt} & {len(g)} & {len(g[0]['per_method'])} & "
            f"{frac(g, 'indices_identical')} & {frac(g, 'counts_identical')} & "
            f"{frac(g, 'negative_control_passed')} & {frac(g, 'reseed_stress_passed')} \\\\"
        )

    total = len(scenarios)
    total_row = (
        f"\\textbf{{Total}} & & \\textbf{{{total}}} & & "
        f"\\textbf{{{frac(scenarios, 'indices_identical')}}} & "
        f"\\textbf{{{frac(scenarios, 'counts_identical')}}} & "
        f"\\textbf{{{frac(scenarios, 'negative_control_passed')}}} & "
        f"\\textbf{{{frac(scenarios, 'reseed_stress_passed')}}} \\\\"
    )
    tex = f"""% Generated by make_partition_manifest.py -- do not edit by hand.
% Requires \\usepackage{{booktabs}}.
\\begin{{table}}[t]
\\centering
\\caption{{Verification that all five federated methods operate on identical client
partitions. Each scenario is one (dataset, $\\alpha$, number of clients, seed)
combination; every scenario is regenerated independently for each method and
fingerprinted with SHA-256. Counts are reported as passed/applicable.}}
\\label{{tab:partition-verification}}
\\begin{{tabular}}{{llrrcccc}}
\\toprule
Dataset & $\\alpha$ & Scenarios & Methods & \\multicolumn{{2}}{{c}}{{Identical across methods}} & Negative & RNG \\\\
\\cmidrule(lr){{5-6}}
 & & & & Indices & Label counts & control & stress \\\\
\\midrule
{chr(10).join(rows)}
\\midrule
{total_row}
\\bottomrule
\\end{{tabular}}

\\vspace{{2pt}}
{{\\footnotesize\\raggedright
\\textit{{Note.}} The identity holds by construction rather than by coincidence:
\\texttt{{dirichlet\\_split\\_noniid}} calls \\texttt{{np.random.seed(seed)}} as its first
executable statement, which makes it a pure function of
(labels, \\texttt{{n\\_clients}}, $\\alpha$, \\texttt{{seed}}) and therefore independent of
how much of the global random stream each method consumed beforehand.
The \\emph{{negative control}} confirms that distinct seeds yield distinct partitions,
so the identity result is not vacuous. The \\emph{{RNG stress}} test consumes
approximately 1.5M draws from \\texttt{{numpy}}, \\texttt{{torch}} and \\texttt{{random}} and
reseeds every global generator to 999 immediately before partitioning, confirming the
fingerprint is unchanged. FEMNIST uses the dataset's natural per-writer partitioning,
for which $\\alpha$ does not apply.
\\par}}
\\end{{table}}
"""
    path.write_text(tex, encoding="utf-8")


# --------------------------------------------------------------------------- #
# --check-init
# --------------------------------------------------------------------------- #
def state_dict_sha256(state_dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(state_dict[key].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def check_model_init(modules, path: Path, seeds=(42, 43, 44)):
    """
    Fingerprint the initial model weights per (architecture, seed, method).

    Two distinct questions are answered here:
      - across methods, for a fixed seed: do all five scripts start from the same
        weights? (required for the paired comparison to isolate the algorithm)
      - across seeds, for a fixed method: does the initialisation actually depend
        on the seed, or is it fixed? (a seed-independent initialisation would mean
        the runs share a single starting point, reducing the variance the seeds
        are meant to sample)
    """
    rows = []
    for arch in ARCHITECTURES:
        for seed in seeds:
            for name in METHOD_ORDER:
                mod = modules[name]
                with suppressed_output():
                    mod.set_global_seed(seed)
                    net = getattr(mod, arch)()
                    sd = net.state_dict()
                rows.append({
                    "architecture": arch,
                    "seed": seed,
                    "method": name,
                    "n_tensors": len(sd),
                    "n_parameters": int(sum(v.numel() for v in sd.values())),
                    "sha256_state_dict": state_dict_sha256(sd),
                })
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n--- Initial model weights (--check-init, seeds={list(seeds)}) ---")
    print(f"{'architecture':22} {'seed':>5} {'sha256 (first 16)':20} "
          f"{'identical across methods'}")
    all_identical = True
    varies_with_seed = {}
    for arch in ARCHITECTURES:
        per_seed_hash = {}
        for seed in seeds:
            sub = [r for r in rows if r["architecture"] == arch and r["seed"] == seed]
            hashes = {r["sha256_state_dict"] for r in sub}
            identical = len(hashes) == 1
            all_identical &= identical
            per_seed_hash[seed] = sorted(hashes)[0]
            print(f"{arch:22} {seed:>5} {per_seed_hash[seed][:16]:20} {identical}")
        varies_with_seed[arch] = len(set(per_seed_hash.values())) == len(seeds)

    print(f"\nInitial weights identical across methods (every architecture, every seed): "
          f"{all_identical}")
    for arch, varies in varies_with_seed.items():
        print(f"  {arch:22} initialisation varies with the seed: {varies}")
    return all_identical


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate client partitions for the full grid and verify that "
                    "all federated methods share identical partitions."
    )
    ap.add_argument("--quick", action="store_true",
                    help="Reduced grid for a fast smoke test.")
    ap.add_argument("--skip-femnist", action="store_true",
                    help="Skip the FEMNIST scenarios (they are the slowest).")
    ap.add_argument("--check-init", action="store_true",
                    help="Also fingerprint the initial model weights of each method.")
    args = ap.parse_args()

    datasets, alphas, n_clients_grid, seeds = DATASETS, ALPHAS, N_CLIENTS_GRID, SEEDS
    femnist_setups, femnist_seeds = FEMNIST_SETUPS, FEMNIST_SEEDS
    if args.quick:
        datasets = ["MNIST"]
        alphas = [0.5, 0.01]
        n_clients_grid = [3]
        seeds = [42, 43, 44]
        # Keep the campaign's FEMNIST shape (10 clients), just fewer seeds.
        femnist_setups = [10]
        femnist_seeds = [42, 43]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The method scripts resolve their dataset directory relative to the current
    # working directory ("./data"), so run from the project root regardless of
    # where this script was invoked from. OUT_DIR is absolute and unaffected.
    os.chdir(PROJECT_ROOT)

    print("=" * 78)
    print("PARTITION MANIFEST AND CROSS-METHOD IDENTITY VERIFICATION")
    print("=" * 78)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output dir   : {OUT_DIR}")
    print(f"Methods      : {', '.join(METHOD_ORDER)}")
    print(f"Grid         : datasets={[DATASET_LABEL[d] for d in datasets]} "
          f"alphas={alphas} n_clients={n_clients_grid} seeds={len(seeds)}")
    if not args.skip_femnist:
        print(f"FEMNIST      : setups={femnist_setups} seeds={femnist_seeds} (natural partitioning)")
    print("Importing method scripts (no training is executed)...", flush=True)

    modules = import_methods()
    trainsets = load_trainsets(datasets)

    scenarios = []
    total_dirichlet = len(datasets) * len(alphas) * len(n_clients_grid) * len(seeds)
    print(f"Generating {total_dirichlet} Dirichlet scenarios "
          f"x {len(METHOD_ORDER)} methods (plus RNG stress replicas)...", flush=True)
    for i, sc in enumerate(run_dirichlet_scenarios(
            modules, trainsets, datasets, alphas, n_clients_grid, seeds), start=1):
        scenarios.append(sc)
        if i % 50 == 0 or i == total_dirichlet:
            print(f"  {i}/{total_dirichlet} scenarios", flush=True)

    if not args.skip_femnist:
        n_fem = len(femnist_setups) * len(femnist_seeds)
        print(f"Generating {n_fem} FEMNIST scenarios (this loads the HuggingFace "
              f"dataset repeatedly and is the slow part)...", flush=True)
        for i, sc in enumerate(run_femnist_scenarios(
                modules, femnist_setups, femnist_seeds), start=1):
            scenarios.append(sc)
            print(f"  {i}/{n_fem} FEMNIST scenarios", flush=True)

    verify(scenarios)

    print("Writing outputs...", flush=True)
    write_manifest(scenarios, modules, OUT_DIR / "partition_manifest.csv")
    write_verification(scenarios, OUT_DIR / "partition_verification.csv")
    write_tex(scenarios, OUT_DIR / "partition_verification.tex")
    femnist_scenarios = [sc for sc in scenarios if sc["partitioning"] == "natural"]
    if femnist_scenarios:
        write_femnist_heldout(scenarios, OUT_DIR / "femnist_heldout_manifest.csv")

    # ----------------------------- summary ---------------------------------- #
    failures = []
    for sc in scenarios:
        reasons = []
        if not sc["indices_identical"]:
            reasons.append("indices differ across methods")
        if not sc["counts_identical"]:
            reasons.append("label counts differ across methods")
        if sc["negative_control_passed"] is False:
            reasons.append("negative control failed (a different seed gave the same partition)")
        if not sc["reseed_stress_passed"]:
            reasons.append("RNG stress test changed the partition")
        if reasons:
            failures.append((sc["scenario_id"], reasons))

    n_neg = sum(1 for sc in scenarios if sc["negative_control_passed"] != "NA")
    print()
    print("-" * 78)
    print("SUMMARY")
    print("-" * 78)
    print(f"Scenarios verified                  : {len(scenarios)}")
    print(f"Methods per scenario                : {len(METHOD_ORDER)}")
    print(f"Client partitions fingerprinted     : "
          f"{sum(sc['n_clients'] * len(METHOD_ORDER) for sc in scenarios)}")
    print(f"Identical indices across methods    : "
          f"{sum(1 for sc in scenarios if sc['indices_identical'])}/{len(scenarios)}")
    print(f"Identical label counts across methods: "
          f"{sum(1 for sc in scenarios if sc['counts_identical'])}/{len(scenarios)}")
    print(f"Negative control passed             : "
          f"{sum(1 for sc in scenarios if sc['negative_control_passed'] is True)}/{n_neg} "
          f"({len(scenarios) - n_neg} not applicable)")
    print(f"RNG stress test passed              : "
          f"{sum(1 for sc in scenarios if sc['reseed_stress_passed'])}/{len(scenarios)}")

    if femnist_scenarios:
        excluded = sum(
            1 for sc in femnist_scenarios for m in METHOD_ORDER
            for c in sc["per_method"][m]["clients"] if c["excluded_reason"]
        )
        heldout_ok = sum(1 for sc in femnist_scenarios if sc["heldout_identical"])
        n_heldout = len(femnist_scenarios[0]["heldout"]) if femnist_scenarios else 0
        print(f"FEMNIST scenarios                   : {len(femnist_scenarios)} "
              f"(client writers per scenario: {femnist_scenarios[0]['n_clients']})")
        print(f"FEMNIST held-out writers per scenario: {n_heldout} "
              f"(identical across methods: {heldout_ok}/{len(femnist_scenarios)})")
        print(f"FEMNIST client rows with excluded_reason set: {excluded}")

    init_ok = True
    if args.check_init:
        init_ok = check_model_init(modules, OUT_DIR / "model_init_check.csv")

    print()
    if failures:
        print(f"DIVERGENCES FOUND: {len(failures)} scenario(s) failed")
        for sid, reasons in failures[:20]:
            print(f"  - {sid}: {'; '.join(reasons)}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more (see partition_verification.csv)")
    else:
        print("No divergences found. All scenarios passed every check.")

    if args.check_init and not init_ok:
        print("WARNING: initial model weights are NOT identical across methods for at "
              "least one architecture (reported above; does not affect the exit status).")

    print()
    print("Files written:")
    written = ["partition_manifest.csv", "partition_verification.csv",
               "partition_verification.tex"]
    if femnist_scenarios:
        written.append("femnist_heldout_manifest.csv")
    if args.check_init:
        written.append("model_init_check.csv")
    for f in written:
        p = OUT_DIR / f
        print(f"  {p}  ({p.stat().st_size:,} bytes)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
