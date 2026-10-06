# -*- coding: utf-8 -*-
"""
Partitions and per-client (epochs, learning rate) policy of the global-gradient
diagnostic battery (reviewer R2, Recommendation94, mechanism paragraph).

Partitions
  dirichlet   the partition of the paper (fedhad_r2.dirichlet_split_noniid, unmodified):
              client size and label skew are drawn together, and in these federations
              H_k and n_k are almost collinear (median per-seed Spearman -0.90).
  equal       equal-size label skew: every client holds exactly EQUAL_N samples whose
              label proportions are drawn from Dir(alpha * 1_C), sampled without
              replacement from the class pools (renormalised over non-exhausted classes).
              Client size, aggregation weight p_k and minibatches per epoch b_k are then
              identical across clients, so size cannot explain any association with H_k.

Arms
  full_fedhad     E_k = E(H_k), eta_k = eta(H_k)                         (the policy)
  inv_epochs      the multiset of E(H_k) re-assigned in INVERSE order of H_k among clients
                  with data (largest H_k -> largest E); each client keeps eta(H_k). On the
                  equal partition sum_k tau_k and sum_k p_k tau_k are preserved exactly.
  fedprox_default E = 5, eta = 1e-2 for every client (FedProx, mu as in FedHAD)
  fedprox_tuned   E, eta frozen by the R2 validation protocol for that alpha

The functions above build_plan() are pure (numpy only) and unit-tested.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ARMS = ("full_fedhad", "inv_epochs", "fedprox_default", "fedprox_tuned")
PARTITIONS = ("dirichlet", "equal")
ARM_ENV, PARTITION_ENV = "FL_GG_ARM", "FL_GG_PARTITION"
EQUAL_N = 5000                     # samples per client on the equal partition (CIFAR-10: 5,000 per class)
DEFAULT_STATIC = {"epochs": 5, "lr": 0.01}

_REPO = Path(__file__).resolve().parents[2]
_R2_CODE = _REPO / "reviewer_r2_controls" / "code"
FROZEN = _REPO / "results" / "tuned_controls_fedprox_validation" / "frozen_config.json"


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #
def equal_size_label_skew(labels, n_clients, alpha, n_per_client, seed):
    """Index lists, one per client, each of exactly n_per_client samples. Client order
    and draws come from np.random.default_rng(seed); no index is used twice."""
    labels = np.asarray(labels)
    n_classes = int(labels.max()) + 1
    if n_clients * n_per_client > len(labels):
        raise ValueError("not enough samples for the requested equal partition")
    rng = np.random.default_rng(int(seed))
    pools = [list(rng.permutation(np.where(labels == c)[0])) for c in range(n_classes)]
    out = [None] * n_clients
    for k in rng.permutation(n_clients):
        p = rng.dirichlet([float(alpha)] * n_classes)
        take_idx, need = [], int(n_per_client)
        while need > 0:
            avail = np.array([len(pl) for pl in pools], dtype=float)
            q = p * (avail > 0)
            if q.sum() <= 0:
                q = (avail > 0).astype(float)
            q = q / q.sum()
            draw = np.minimum(rng.multinomial(need, q), avail.astype(int))
            for c, m in enumerate(draw):
                if m:
                    take_idx.extend(pools[c][:m])
                    pools[c] = pools[c][m:]
            need -= int(draw.sum())
        out[int(k)] = sorted(int(i) for i in take_idx)
    return out


def inverse_epochs(e_full, h_values, n_samples):
    """Same multiset of epochs among clients WITH data, re-assigned so that the client
    with the largest H_k receives the largest E. Ties in H_k by client index."""
    ks = [k for k, n in enumerate(n_samples) if int(n) > 0]
    order_h = sorted(ks, key=lambda k: (h_values[k], k))      # ascending H
    epochs = sorted(int(e_full[k]) for k in ks)                # ascending E
    out = list(int(e) for e in e_full)
    for k, e in zip(order_h, epochs):
        out[k] = e
    return out


def step_budgets(epochs, n_batches):
    return [int(e) * int(b) for e, b in zip(epochs, n_batches)]


def weighted_steps(tau, n_samples):
    total = float(sum(n_samples))
    return sum(float(n) / total * float(t) for n, t in zip(n_samples, tau))


def frozen_static(alpha, frozen_path=FROZEN):
    fz = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    for c in fz["configs"]:
        if c["method"] == "FedProx" and abs(float(c["alpha"]) - float(alpha)) < 1e-12:
            return {"epochs": int(c["epochs"]), "lr": float(c["lr"])}
    raise ValueError(f"no frozen FedProx configuration for alpha={alpha}")


def arm_policy(arm, *, e_full, lr_full, h_values, n_samples, alpha, frozen_path=FROZEN):
    """(epochs per client, lr per client) of an arm. Pure apart from reading the frozen
    configuration file for fedprox_tuned."""
    K = len(e_full)
    if arm == "full_fedhad":
        return [int(e) for e in e_full], [float(x) for x in lr_full]
    if arm == "inv_epochs":
        return inverse_epochs(e_full, h_values, n_samples), [float(x) for x in lr_full]
    if arm == "fedprox_default":
        return [DEFAULT_STATIC["epochs"]] * K, [DEFAULT_STATIC["lr"]] * K
    if arm == "fedprox_tuned":
        st = frozen_static(alpha, frozen_path)
        return [st["epochs"]] * K, [st["lr"]] * K
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


# --------------------------------------------------------------------------- #
# Driver side
# --------------------------------------------------------------------------- #
def active_arm() -> str:
    a = os.environ.get(ARM_ENV, "")
    if a not in ARMS:
        raise ValueError(f"{ARM_ENV}={a!r} must be one of {ARMS}")
    return a


def active_partition() -> str:
    p = os.environ.get(PARTITION_ENV, "dirichlet")
    if p not in PARTITIONS:
        raise ValueError(f"{PARTITION_ENV}={p!r} must be one of {PARTITIONS}")
    return p


def build_plan(*, arm, partition, trainloaders, h_values, seed, dataset, alpha, epochs_fn,
               lr_fn, base_epochs, min_epochs, base_lr, min_lr, flops_per_sample_full):
    """Per-client table in the format read by fit(), and r2_policy._STATE filled so that
    r2_policy.write_run_artifacts records the run unchanged."""
    if str(_R2_CODE) not in sys.path:
        sys.path.insert(0, str(_R2_CODE))
    import r2_policy  # noqa: E402  (unmodified R2 infrastructure)

    n_samples = [len(dl.dataset) for dl in trainloaders]
    n_batches = [len(dl) for dl in trainloaders]
    e_full = [int(epochs_fn(h, base_epochs, min_epochs)) for h in h_values]
    lr_full = [float(lr_fn(h, base_lr, min_lr)) for h in h_values]
    epochs, lrs = arm_policy(arm, e_full=e_full, lr_full=lr_full, h_values=h_values,
                             n_samples=n_samples, alpha=alpha)
    tau = step_budgets(epochs, n_batches)
    tau_full = step_budgets(e_full, n_batches)
    table = [{"tau": int(tau[k]), "lr": float(lrs[k]), "n_batches": int(n_batches[k]),
              "E_full": int(e_full[k]), "E_arm": int(epochs[k]), "tau_original": int(tau_full[k]),
              "donor": k, "perm_id": ""} for k in range(len(trainloaders))]
    r2_policy._STATE.clear()
    r2_policy._STATE.update({
        "arm": arm, "partition": partition, "seed": int(seed), "dataset": dataset,
        "alpha": float(alpha), "n_clients": len(trainloaders), "n_samples": n_samples,
        "n_batches": n_batches, "H": [float(h) for h in h_values], "E_full": e_full,
        "LR_full": lr_full, "E_arm": [int(e) for e in epochs], "LR_arm": [float(x) for x in lrs],
        "tau_original": tau_full, "tau_received": [int(t) for t in tau],
        "lr_received": [float(x) for x in lrs], "permutation": None,
        "weighted_steps_arm": weighted_steps(tau, n_samples),
        "weighted_steps_full": weighted_steps(tau_full, n_samples),
        "partition_hashes": [r2_policy.sha256_of(r2_policy.resolve_indices(dl.dataset))
                             for dl in trainloaders],
        "flops_per_sample_full": flops_per_sample_full,
    })
    return table
