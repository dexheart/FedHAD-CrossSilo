# -*- coding: utf-8 -*-
"""
Policy, determinism and artifact helpers for the reviewer-R2 control experiments.

Everything NEW for these experiments lives here, so the diff applied to the copied
FedHAD source (fedhad_r2.py) stays small and auditable. This module never imports
the FedHAD script. The pure functions below are unit-tested in
reviewer_r2_controls/tests/test_r2_policy.py without any training.

Arms
----
  full_fedhad             FedHAD as published: E_k = E_full(H_k), eta_k = LR_full(H_k);
                          executed as tau_k = E_k * b_k optimizer steps.
  step_permuted_lrclient  optimizer-step budgets tau_k of full_fedhad are permuted
                          among the clients that can execute steps (a seeded
                          derangement); each client KEEPS its own eta_k = LR_full(H_k).
                          This is the control requested by the reviewer.
  step_permuted_lrfollow  optional: the pair (tau, eta) moves together, i.e. client k
                          receives (tau_{pi(k)}, eta_{pi(k)}).
  static                  tuned static control: every client gets the same epochs E
                          and the same learning rate eta; no H_k anywhere. The proximal
                          term is on for method=FedProx and off for method=FedAvg.

b_k is the number of minibatches the client's training DataLoader yields per epoch
(batch_size=32, drop_last=True as in the historical implementation), so tau_k is
the number of optimizer steps actually executed, not a nominal quantity.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

ARMS = ("full_fedhad", "step_permuted_lrclient", "step_permuted_lrfollow", "static")
STATIC_METHODS = ("FedProx", "FedAvg")

_STATE: dict = {}
_ROUND_CLOCK: list = []


# --------------------------------------------------------------------------- #
# Deterministic seed derivation (pure)
# --------------------------------------------------------------------------- #
def derive_seed(*parts) -> int:
    """Stable 31-bit seed from any tuple of printable parts (independent of
    PYTHONHASHSEED, process, host or Ray worker)."""
    key = ":".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16) & 0x7FFFFFFF


def client_round_seed(global_seed: int, cid, server_round: int) -> int:
    """Seed of the local training of client `cid` in round `server_round`.
    It depends only on (global seed, client, round), NOT on the arm, so two arms
    whose configuration of a client coincides share the same stochastic trajectory
    (minibatch order, dropout masks) for that client and round."""
    return derive_seed("r2-client", int(global_seed), int(cid), int(server_round))


def permutation_seed(dataset: str, alpha: float, global_seed: int) -> int:
    """Seed of the step-budget derangement. Uses only pre-training information."""
    return derive_seed("r2-step-perm", dataset, repr(float(alpha)), int(global_seed))


# --------------------------------------------------------------------------- #
# Step budgets and permutation (pure)
# --------------------------------------------------------------------------- #
def step_budgets(epochs, n_batches):
    """tau_k = E_k * b_k (executed optimizer steps per round)."""
    if len(epochs) != len(n_batches):
        raise ValueError("epochs and n_batches must have the same length")
    return [int(e) * int(b) for e, b in zip(epochs, n_batches)]


def permutation_domain(n_samples, n_batches):
    """Clients that can execute optimizer steps: n_k > 0 and b_k > 0. Empty clients
    (n_k = 0) and clients whose training split yields no full minibatch never
    receive nor transfer budget."""
    return [k for k, (n, b) in enumerate(zip(n_samples, n_batches)) if int(n) > 0 and int(b) > 0]


def random_derangement(domain, rng: np.random.Generator):
    """Uniform random permutation of `domain` with no fixed point (rejection
    sampling). Returns a dict receiver -> donor. With fewer than two clients no
    derangement exists and the identity is returned (flagged by the caller)."""
    domain = list(domain)
    if len(domain) < 2:
        return {k: k for k in domain}
    for _ in range(10000):
        perm = list(rng.permutation(domain))
        if all(int(a) != int(b) for a, b in zip(domain, perm)):
            return {int(r): int(d) for r, d in zip(domain, perm)}
    raise RuntimeError("could not draw a derangement")  # practically unreachable


def step_permuted_allocation(tau, n_samples, n_batches, lr_full, perm_seed, lr_follows=False):
    """
    Permute the executed step budgets tau of full_fedhad among the permutation
    domain with a seeded derangement. Returns per-client received steps and
    learning rates, the mapping and diagnostics. The total number of optimizer
    steps is preserved EXACTLY (a permutation of the same multiset); clients outside
    the domain keep tau = 0 (they have tau = 0 under full_fedhad as well).
    """
    k_all = len(tau)
    dom = permutation_domain(n_samples, n_batches)
    for k in range(k_all):
        if k not in dom and int(tau[k]) != 0:
            raise ValueError(f"client {k} is outside the domain but has tau={tau[k]}")
    rng = np.random.default_rng(int(perm_seed))
    mapping = random_derangement(dom, rng)          # receiver -> donor
    tau_recv = [0] * k_all
    lr_recv = [float(x) for x in lr_full]
    for r, d in mapping.items():
        tau_recv[r] = int(tau[d])
        if lr_follows:
            lr_recv[r] = float(lr_full[d])
    if sum(tau_recv) != sum(int(t) for t in tau):
        raise AssertionError("step budget not preserved")
    if sorted(tau_recv[k] for k in dom) != sorted(int(tau[k]) for k in dom):
        raise AssertionError("step-budget multiset not preserved on the domain")
    return {
        "tau_received": tau_recv, "lr_received": lr_recv,
        "mapping_receiver_to_donor": {str(r): int(d) for r, d in mapping.items()},
        "domain": dom, "perm_seed": int(perm_seed),
        "degenerate": len(dom) < 2,
        "n_value_unchanged": int(sum(1 for k in dom if tau_recv[k] == int(tau[k]))),
        "total_steps_original": int(sum(int(t) for t in tau)),
        "total_steps_received": int(sum(tau_recv)),
        "perm_id": hashlib.sha256(json.dumps(
            {str(r): int(d) for r, d in sorted(mapping.items())}, sort_keys=True).encode()
        ).hexdigest()[:12],
    }


def static_allocation(n_batches, epochs, lr):
    """Uniform static control: same E and eta for every client."""
    return {"tau_received": [int(epochs) * int(b) for b in n_batches],
            "lr_received": [float(lr)] * len(n_batches)}


# --------------------------------------------------------------------------- #
# Seed-set hygiene (pure)
# --------------------------------------------------------------------------- #
def parse_seed_list(text: str):
    """'42' | '42,43' | '42-71' | '42-45,50' -> sorted unique list of ints."""
    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                raise ValueError(f"bad seed range {part!r}")
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("empty seed list")
    return sorted(out)


def check_seed_separation(validation_seeds, evaluation_seeds):
    """Raise if any validation (tuning) seed is also a final-evaluation seed."""
    inter = sorted(set(validation_seeds) & set(evaluation_seeds))
    if inter:
        raise ValueError(
            "LEAKAGE GUARD: validation/tuning seeds overlap the final-evaluation seeds "
            f"{inter}. Tuning must use seeds disjoint from the evaluation seeds.")
    return True


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #
def sha256_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def state_dict_sha256(state_dict) -> str:
    """Identical to results_revision_ablation/code/ablation_policy.state_dict_sha256,
    so checkpoints can be compared with the initial_weights_sha256 of the historical
    component analysis."""
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(state_dict[key].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def resolve_indices(subset):
    idx = list(getattr(subset, "indices", []))
    cur = subset
    while hasattr(getattr(cur, "dataset", None), "indices"):
        cur = cur.dataset
        parent = list(cur.indices)
        idx = [parent[i] for i in idx]
    return sorted(int(i) for i in idx)


# --------------------------------------------------------------------------- #
# Driver-side plan (state lives in the driver only; workers receive plain lists)
# --------------------------------------------------------------------------- #
def active_arm() -> str:
    a = os.environ.get("FL_R2_ARM", "")
    if a not in ARMS:
        raise ValueError(f"FL_R2_ARM={a!r} must be one of {ARMS}")
    return a


def build_plan(*, trainloaders, h_values, seed, dataset, alpha, epochs_fn, lr_fn,
               base_epochs, min_epochs, base_lr, min_lr, flops_per_sample_full):
    arm = active_arm()
    n_samples = [len(dl.dataset) for dl in trainloaders]
    n_batches = [len(dl) for dl in trainloaders]
    e_full = [int(epochs_fn(h, base_epochs, min_epochs)) for h in h_values]
    lr_full = [float(lr_fn(h, base_lr, min_lr)) for h in h_values]
    tau_orig = step_budgets(e_full, n_batches)
    perm = None
    if arm == "full_fedhad":
        tau_recv, lr_recv = list(tau_orig), list(lr_full)
    elif arm in ("step_permuted_lrclient", "step_permuted_lrfollow"):
        perm = step_permuted_allocation(tau_orig, n_samples, n_batches, lr_full,
                                        permutation_seed(dataset, alpha, seed),
                                        lr_follows=(arm == "step_permuted_lrfollow"))
        if perm["degenerate"]:
            raise RuntimeError("step permutation undefined: fewer than two clients can train")
        tau_recv, lr_recv = perm["tau_received"], perm["lr_received"]
    else:
        e = int(os.environ["FL_R2_STATIC_EPOCHS"])
        lr = float(os.environ["FL_R2_STATIC_LR"])
        st = static_allocation(n_batches, e, lr)
        tau_recv, lr_recv = st["tau_received"], st["lr_received"]
    table = []
    for k in range(len(trainloaders)):
        table.append({
            "tau": int(tau_recv[k]), "lr": float(lr_recv[k]), "n_batches": int(n_batches[k]),
            "E_full": int(e_full[k]), "tau_original": int(tau_orig[k]),
            "donor": (int(perm["mapping_receiver_to_donor"].get(str(k), k)) if perm else k),
            "perm_id": (perm["perm_id"] if perm else ""),
        })
    _STATE.clear()
    _STATE.update({
        "arm": arm, "seed": int(seed), "dataset": dataset, "alpha": float(alpha),
        "n_clients": len(trainloaders), "n_samples": n_samples, "n_batches": n_batches,
        "H": [float(h) for h in h_values], "E_full": e_full, "LR_full": lr_full,
        "tau_original": tau_orig, "tau_received": [int(t) for t in tau_recv],
        "lr_received": [float(x) for x in lr_recv], "permutation": perm,
        "partition_hashes": [sha256_of(resolve_indices(dl.dataset)) for dl in trainloaders],
        "flops_per_sample_full": flops_per_sample_full,
    })
    return table


def mark_round():
    _ROUND_CLOCK.append(time.time())


def round_wall_clock():
    if len(_ROUND_CLOCK) < 2:
        return []
    return [round(b - a, 4) for a, b in zip(_ROUND_CLOCK[:-1], _ROUND_CLOCK[1:])]


def write_run_artifacts(out_dir, *, config_snapshot, init_sha256, init_path, history,
                        fit_metrics_history, elapsed, hardware):
    """fingerprint.json (paired-design proof + run totals) and telemetry.csv
    (one row per client and round)."""
    import csv
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mc = getattr(history, "metrics_centralized", {}) or {}
    acc_c = dict(mc.get("accuracy", []))
    loss_c = dict(getattr(history, "losses_centralized", []) or [])
    md = getattr(history, "metrics_distributed", {}) or {}
    acc_d = dict(md.get("accuracy", []))
    loss_d = dict(getattr(history, "losses_distributed", []) or [])
    wall = round_wall_clock()
    s = _STATE
    rows = []
    for r_idx, round_metrics in enumerate(fit_metrics_history, start=1):
        for num_examples, m in round_metrics:
            rows.append({
                "arm": s["arm"], "method": config_snapshot.get("method", ""),
                "dataset": s["dataset"], "alpha": s["alpha"], "seed": s["seed"], "round": r_idx,
                "client_id": m.get("cid", ""), "n_k": int(num_examples),
                "n_batches": m.get("n_batches", ""), "H_k": m.get("H_k", ""),
                "E_full": m.get("E_full", ""), "learning_rate": m.get("learning_rate", ""),
                "tau_original": m.get("tau_original", ""), "tau_received": m.get("tau_received", ""),
                "steps_executed": m.get("steps_executed", ""),
                "examples_processed": m.get("examples_processed", ""),
                "flops_nominal": m.get("flops_nominal", ""), "flops_executed": m.get("flops_executed", ""),
                "donor_client": m.get("donor", ""), "perm_id": m.get("perm_id", ""),
                "local_seed": m.get("local_seed", ""), "loss_after_fit": m.get("loss_after_fit", ""),
                "acc_centralized_round": acc_c.get(r_idx, ""), "loss_centralized_round": loss_c.get(r_idx, ""),
                "acc_distributed_round": acc_d.get(r_idx, ""), "loss_distributed_round": loss_d.get(r_idx, ""),
                "round_wall_clock_s": wall[r_idx - 1] if r_idx - 1 < len(wall) else "",
            })
    if rows:
        with (out / "telemetry.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def tot(key):
        return float(sum(float(r[key] or 0) for r in rows))

    last_round = max(acc_c) if acc_c else None
    fp = {
        **{k: v for k, v in s.items() if k not in ("flops_per_sample_full",)},
        "config": config_snapshot,
        "initial_checkpoint": {"path": str(init_path), "sha256": init_sha256},
        "partition_hash_all": sha256_of(s["partition_hashes"]),
        "H_hash": sha256_of([round(h, 12) for h in s["H"]]),
        "flops_per_sample_full": s["flops_per_sample_full"],
        "totals": {
            "rounds_recorded": len(fit_metrics_history),
            "client_round_rows": len(rows),
            "optimizer_steps": tot("steps_executed"),
            "examples_processed": tot("examples_processed"),
            "flops_nominal": tot("flops_nominal"),
            "flops_executed": tot("flops_executed"),
            "planned_steps_per_round": int(sum(s["tau_received"])),
        },
        "final": {
            "round": last_round,
            "acc_centralized": acc_c.get(last_round) if last_round is not None else None,
            "loss_centralized": loss_c.get(last_round) if last_round is not None else None,
            "acc_distributed_val": acc_d.get(max(acc_d)) if acc_d else None,
            "loss_distributed_val": loss_d.get(max(loss_d)) if loss_d else None,
        },
        "elapsed_seconds": round(float(elapsed), 3),
        "round_wall_clock": wall,
        "hardware": hardware,
        "completed": True,
    }
    (out / "fingerprint.json").write_text(json.dumps(fp, indent=2, sort_keys=True), encoding="utf-8")
    return fp
