# -*- coding: utf-8 -*-
"""
Per-client (epochs, learning rate) policy for the LR-uniform control battery.

Question: does assigning the learning rate according to H_k differ from giving
every client ONE uniform learning rate with the same aggregate first-order
aggressiveness?

Arms (all share partition, H_k, initial checkpoint and per-(seed, client, round)
worker seeding through fedhad_lru.py, a marked copy of the R2 script):

  full_fedhad      E = E_full(H)   LR_k = LR_full(H_k)           reference
  lr_only_matched  E = E_fixed     LR_k = LR_full(H_k)           reference (Section 6.10)
  u2               E = E_full(H)   LR_k = eta_bar(LR_full, tau_full)       uniform
  u1               E = E_fixed     LR_k = eta_bar(LR_full, tau_fixed)      uniform
  u2_arith         E = E_full(H)   LR_k = arithmetic mean of LR_full (clients with data)
  inv              E = E_full(H)   LR_k = LR_full multiset re-assigned in INVERSE order of H_k

E_fixed is taken, unmodified, from
results/component_analysis_ablation/code/ablation_policy.build_allocation,
the construction used by the fixed_matched and lr_only_matched arms of Section 6.10.

Notation: tau_k = E_k * floor(n_k^tr / B) (optimizer steps per round; n_batches under
drop_last), p_k = n_k / N with n_k the training samples (the aggregation weight).
Clients without samples (tau_k = 0, n_k = 0) never enter eta_bar and keep their own
learning rate in INV; they do not train, exactly as in Section 6.10.

The functions above build_plan() are pure (numpy only) and unit-tested.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ARMS = ("full_fedhad", "lr_only_matched", "u2", "u1", "u2_arith", "inv")
ARM_ENV = "FL_LRU_ARM"

_REPO = Path(__file__).resolve().parents[2]
_ABL_CODE = _REPO / "results" / "component_analysis_ablation" / "code"
_R2_CODE = _REPO / "reviewer_r2_controls" / "code"


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #
def aggregation_weights(n_samples):
    """p_k = n_k / N over all clients (empty clients get weight 0)."""
    total = float(sum(n_samples))
    if total <= 0:
        raise ValueError("no client holds training samples")
    return [float(n) / total for n in n_samples]


def _data_clients(n_samples, tau=None):
    return [k for k, n in enumerate(n_samples)
            if n > 0 and (tau is None or tau[k] > 0)]


def eta_bar_weighted(lr, tau, n_samples):
    """Primary definition: sum_k p_k eta_k tau_k / sum_k p_k tau_k over clients with data.
    A uniform rate equal to this value reproduces the first-order aggregate displacement
    sum_k p_k eta_k tau_k of the reference policy exactly."""
    p = aggregation_weights(n_samples)
    ks = _data_clients(n_samples, tau)
    den = sum(p[k] * tau[k] for k in ks)
    if den <= 0:
        raise ValueError("sum_k p_k tau_k is zero")
    return sum(p[k] * lr[k] * tau[k] for k in ks) / den


def eta_bar_tau(lr, tau, n_samples):
    """Sensitivity definition: weighted by tau_k only."""
    ks = _data_clients(n_samples, tau)
    den = sum(tau[k] for k in ks)
    if den <= 0:
        raise ValueError("sum_k tau_k is zero")
    return sum(lr[k] * tau[k] for k in ks) / den


def eta_bar_arith(lr, n_samples):
    """Sensitivity definition: simple arithmetic mean over clients with data."""
    ks = _data_clients(n_samples)
    if not ks:
        raise ValueError("no client holds training samples")
    return sum(lr[k] for k in ks) / len(ks)


def aggregate_displacement(lr, tau, n_samples):
    """sum_k p_k eta_k tau_k (first-order aggregate displacement coefficient)."""
    p = aggregation_weights(n_samples)
    return sum(p[k] * lr[k] * tau[k] for k in range(len(lr)))


def inverted_lr(lr_full, h_values, n_samples):
    """Same multiset of learning rates among clients WITH data, re-assigned so that the
    client with the largest H_k receives the largest rate and the one with the smallest
    H_k the smallest. Ties in H_k are broken by client index (lower index first).
    Clients without data keep their own rate (they do not train)."""
    ks = _data_clients(n_samples)
    order_h = sorted(ks, key=lambda k: (h_values[k], k))            # ascending H, index
    rates = sorted(lr_full[k] for k in ks)                           # ascending rate
    out = list(lr_full)
    for k, r in zip(order_h, rates):
        out[k] = r
    return out


def step_budgets(epochs, n_batches):
    return [int(e) * int(b) for e, b in zip(epochs, n_batches)]


def arm_policy(arm, *, e_full, e_fixed, lr_full, h_values, n_samples, n_batches):
    """(epochs per client, lr per client) of an arm. Pure."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    tau_full = step_budgets(e_full, n_batches)
    tau_fixed = step_budgets(e_fixed, n_batches)
    K = len(e_full)
    if arm == "full_fedhad":
        return list(e_full), list(lr_full)
    if arm == "lr_only_matched":
        return list(e_fixed), list(lr_full)
    if arm == "u2":
        return list(e_full), [eta_bar_weighted(lr_full, tau_full, n_samples)] * K
    if arm == "u1":
        return list(e_fixed), [eta_bar_weighted(lr_full, tau_fixed, n_samples)] * K
    if arm == "u2_arith":
        return list(e_full), [eta_bar_arith(lr_full, n_samples)] * K
    return list(e_full), inverted_lr(lr_full, h_values, n_samples)          # inv


def eta_bar_record(*, arm, e_full, e_fixed, lr_full, n_samples, n_batches):
    """The three eta_bar values for the epoch allocation this arm uses, computed from
    FedHAD's eta_k, plus the reference displacement it must match."""
    uses_fixed = arm in ("lr_only_matched", "u1")
    tau = step_budgets(e_fixed if uses_fixed else e_full, n_batches)
    return {
        "eta_bar_allocation": "E_fixed" if uses_fixed else "E_full",
        "eta_bar_primary_p_tau": eta_bar_weighted(lr_full, tau, n_samples),
        "eta_bar_tau_only": eta_bar_tau(lr_full, tau, n_samples),
        "eta_bar_arithmetic": eta_bar_arith(lr_full, n_samples),
        "reference_arm": "lr_only_matched" if uses_fixed else "full_fedhad",
        "reference_sum_p_eta_tau": aggregate_displacement(lr_full, tau, n_samples),
    }


# --------------------------------------------------------------------------- #
# Driver-side plan (called from fedhad_lru.py; not pure: uses DataLoaders)
# --------------------------------------------------------------------------- #
def active_arm() -> str:
    a = os.environ.get(ARM_ENV, "")
    if a not in ARMS:
        raise ValueError(f"{ARM_ENV}={a!r} must be one of {ARMS}")
    return a


def _import_reused():
    for p in (str(_ABL_CODE), str(_R2_CODE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import ablation_policy  # noqa: E402  (Section 6.10, unmodified)
    import r2_policy        # noqa: E402  (R2 infrastructure, unmodified)
    return ablation_policy, r2_policy


def fixed_allocation(*, trainloaders, h_values, seed, base_epochs, min_epochs, base_lr,
                     epochs_fn, lr_fn, min_lr, flops_per_sample_full):
    """E_full, LR_full and E_fixed exactly as Section 6.10 builds them."""
    ablation_policy, _ = _import_reused()
    st = ablation_policy.build_allocation(
        trainloaders=trainloaders, h_values=h_values, seed=seed, base_epochs=base_epochs,
        min_epochs=min_epochs, base_lr=base_lr, epochs_fn=epochs_fn, lr_fn=lr_fn,
        min_lr=min_lr, flops_per_sample_full=flops_per_sample_full)
    return {"E_full": list(st["E_full"]), "LR_full": list(st["LR_full"]),
            "E_fixed": list(st["E_fixed"])}


def build_plan(*, arm, trainloaders, h_values, seed, dataset, alpha, epochs_fn, lr_fn,
               base_epochs, min_epochs, base_lr, min_lr, flops_per_sample_full):
    """Same table format as r2_policy.build_plan (read by fit() in fedhad_lru.py); also
    fills r2_policy._STATE so r2_policy.write_run_artifacts records the run unchanged,
    with the LRU fields (eta_bar values, planned displacement) added to it."""
    _, r2_policy = _import_reused()
    alloc = fixed_allocation(trainloaders=trainloaders, h_values=h_values, seed=seed,
                             base_epochs=base_epochs, min_epochs=min_epochs, base_lr=base_lr,
                             epochs_fn=epochs_fn, lr_fn=lr_fn, min_lr=min_lr,
                             flops_per_sample_full=flops_per_sample_full)
    n_samples = [len(dl.dataset) for dl in trainloaders]
    n_batches = [len(dl) for dl in trainloaders]
    e_full, e_fixed, lr_full = alloc["E_full"], alloc["E_fixed"], alloc["LR_full"]
    epochs, lrs = arm_policy(arm, e_full=e_full, e_fixed=e_fixed, lr_full=lr_full,
                             h_values=h_values, n_samples=n_samples, n_batches=n_batches)
    tau = step_budgets(epochs, n_batches)
    tau_full = step_budgets(e_full, n_batches)
    rec = eta_bar_record(arm=arm, e_full=e_full, e_fixed=e_fixed, lr_full=lr_full,
                         n_samples=n_samples, n_batches=n_batches)
    table = [{"tau": int(tau[k]), "lr": float(lrs[k]), "n_batches": int(n_batches[k]),
              "E_full": int(e_full[k]), "tau_original": int(tau_full[k]),
              "donor": k, "perm_id": ""} for k in range(len(trainloaders))]
    r2_policy._STATE.clear()
    r2_policy._STATE.update({
        "arm": arm, "seed": int(seed), "dataset": dataset, "alpha": float(alpha),
        "n_clients": len(trainloaders), "n_samples": n_samples, "n_batches": n_batches,
        "H": [float(h) for h in h_values], "E_full": e_full, "LR_full": lr_full,
        "E_fixed": e_fixed, "E_arm": [int(e) for e in epochs], "LR_arm": [float(x) for x in lrs],
        "tau_original": tau_full, "tau_received": [int(t) for t in tau],
        "lr_received": [float(x) for x in lrs], "permutation": None,
        "p_k": aggregation_weights(n_samples),
        "planned_sum_p_eta_tau": aggregate_displacement(lrs, tau, n_samples),
        **rec,
        "partition_hashes": [r2_policy.sha256_of(r2_policy.resolve_indices(dl.dataset))
                             for dl in trainloaders],
        "flops_per_sample_full": flops_per_sample_full,
    })
    return table


def plan_all_arms(*, trainloaders, h_values, seed, base_epochs, min_epochs, base_lr,
                  epochs_fn, lr_fn, min_lr, flops_per_sample_full):
    """Every arm's per-client plan for one (seed, alpha); used by the dry run."""
    alloc = fixed_allocation(trainloaders=trainloaders, h_values=h_values, seed=seed,
                             base_epochs=base_epochs, min_epochs=min_epochs, base_lr=base_lr,
                             epochs_fn=epochs_fn, lr_fn=lr_fn, min_lr=min_lr,
                             flops_per_sample_full=flops_per_sample_full)
    n_samples = [len(dl.dataset) for dl in trainloaders]
    n_batches = [len(dl) for dl in trainloaders]
    out = {"n_samples": n_samples, "n_batches": n_batches, "H": [float(h) for h in h_values],
           "E_full": alloc["E_full"], "E_fixed": alloc["E_fixed"], "LR_full": alloc["LR_full"],
           "arms": {}}
    for arm in ARMS:
        e, lr = arm_policy(arm, e_full=alloc["E_full"], e_fixed=alloc["E_fixed"],
                           lr_full=alloc["LR_full"], h_values=h_values,
                           n_samples=n_samples, n_batches=n_batches)
        tau = step_budgets(e, n_batches)
        out["arms"][arm] = {"epochs": e, "lr": lr, "tau": tau,
                            "sum_p_eta_tau": aggregate_displacement(lr, tau, n_samples),
                            **eta_bar_record(arm=arm, e_full=alloc["E_full"],
                                             e_fixed=alloc["E_fixed"], lr_full=alloc["LR_full"],
                                             n_samples=n_samples, n_batches=n_batches)}
    return out
