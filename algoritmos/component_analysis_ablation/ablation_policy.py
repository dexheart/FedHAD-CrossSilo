# -*- coding: utf-8 -*-
"""
Ablation policy, fairness invariants and telemetry for the FedHAD ablation campaign.

All the logic that is NEW for the ablation lives here, so that the diff applied to
the copied FedHAD source stays small and auditable. This module never imports the
FedHAD script: everything it needs is passed in.

The five variants
-----------------
Let H_k be the client's normalised heterogeneity, E_full/LR_full the policies the
current FedHAD applies, and E_fixed a deterministic allocation that does not look
at H_k but sums to exactly the same total.

  full_fedhad       E = E_full(H_k)   LR = LR_full(H_k)    the method as it stands
  fixed_matched     E = E_fixed       LR = BASE_LR         neither depends on H_k
  permuted_allocation  E = perm(E_full)  LR = LR_full(H_k)    same epoch multiset, re-assigned
  epoch_only        E = E_full(H_k)   LR = BASE_LR         adaptive epochs alone
  lr_only_matched   E = E_fixed       LR = LR_full(H_k)    adaptive LR alone

Reading the contrasts:
  full vs fixed_matched      does adaptation help at equal budget at all?
  full vs permuted_allocation   does it matter WHO gets the compute, or only how much?
  epoch_only vs fixed        isolated contribution of adaptive epochs
  lr_only_matched vs fixed   isolated contribution of adaptive learning rate
  full vs both isolated arms is coordinated adaptation worth more than its parts?

Budget matching is done on MINIBATCH UPDATES
--------------------------------------------
Equal epochs is NOT equal compute when clients hold different amounts of data:
updates_k = E_k * batches_k. The matched quantity is therefore the total number
of minibatch updates, U = sum_k E_k * batches_k, and the epoch sum is reported
rather than matched.

  epoch_only        exact by construction (it reuses E_full)
  fixed_matched     argmin |U - U_full| over E_k in {e0, e0+1}, e0 = floor(U_full/sum(batches)),
                    excluding E_full itself; uses ONLY client sizes, never H_k
  permuted_allocation  argmin |U - U_full| over every unique non-trivial re-assignment
                    of the exact E_full multiset
  lr_only_matched   reuses fixed_matched's allocation exactly

Exact equality is not always reachable because epochs are integers; the residual
is recorded per cell and reported. Measured on the first smoke grid the residual
is <= 0.72% for fixed_matched and <= 3.12% for permuted_allocation, against 8.7% and
13.0% under the earlier epoch-sum matching.

A tension worth stating plainly: matching updates requires favouring large
clients, and in these Dirichlet partitions client size is almost perfectly
anti-correlated with H_k (measured rho(n_k, H_k) = -1.0 in several cells). An
H-blind allocation that matches updates can therefore end up resembling the
H-driven one. Two safeguards: E_fixed is forbidden from equalling E_full, and the
rank correlation of every allocation with H_k is recorded per cell so a reader can
judge how clean the control actually is.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import time

import numpy as np

VARIANTS = ["full_fedhad", "fixed_matched", "permuted_allocation",
            "epoch_only", "lr_only_matched"]

_STATE: dict = {}
_ROUND_CLOCK: list = []


def active_variant() -> str:
    v = os.environ.get("FL_ABLATION_VARIANT", "full_fedhad")
    if v not in VARIANTS:
        raise ValueError(f"Unknown FL_ABLATION_VARIANT={v!r}; expected one of {VARIANTS}")
    return v


def sha256_of(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def state_dict_sha256(state_dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(state_dict[key].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def _spearman(x, y):
    """Spearman rho without a scipy dependency; NaN-safe for constant inputs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _resolve_indices(subset):
    """Absolute indices of a (possibly nested) torch Subset into its base dataset."""
    idx = list(getattr(subset, "indices", []))
    cur = subset
    while hasattr(getattr(cur, "dataset", None), "indices"):
        cur = cur.dataset
        parent = list(cur.indices)
        idx = [parent[i] for i in idx]
    return sorted(int(i) for i in idx)


def build_allocation(trainloaders, h_values, seed, base_epochs, min_epochs,
                     base_lr, epochs_fn, lr_fn, min_lr, flops_per_sample_full):
    """
    Compute every variant's per-client policy once, deterministically, and verify
    the invariants that make the arms comparable. Raises (aborting the run) if any
    invariant is violated.
    """
    n_clients = len(h_values)
    e_full = [int(epochs_fn(h, base_epochs, min_epochs)) for h in h_values]
    lr_full = [float(lr_fn(h, base_lr, min_lr)) for h in h_values]

    total_epochs = sum(e_full)
    n_samples = [len(dl.dataset) for dl in trainloaders]
    n_batches = [len(dl) for dl in trainloaders]

    # ======================================================================== #
    # Budget matching is done on MINIBATCH UPDATES, not on the epoch sum.
    # Clients hold different amounts of data, so equal epochs != equal compute:
    # updates_k = E_k * batches_k. The epoch sum is still recorded, but it is a
    # reported quantity, not the matched one.
    # ======================================================================== #
    def updates_of(epochs):
        return int(sum(e * b for e, b in zip(epochs, n_batches)))

    u_full = updates_of(e_full)

    # ---- fixed_matched ----------------------------------------------------- #
    # Deterministic and H-INDEPENDENT: it reads only client sizes (batches per
    # client) and the target update total. It never reads H_k, accuracy or any
    # training output. Search space is the flattest family around the uniform
    # fractional epoch count, so the control stays as close to "everyone trains
    # the same amount" as exact-update-matching permits.
    e0 = max(int(min_epochs), int(u_full // max(sum(n_batches), 1)))
    candidates = []
    for mask in range(1 << n_clients):
        cand = [e0 + (1 if (mask >> i) & 1 else 0) for i in range(n_clients)]
        if min(cand) < min_epochs:
            continue
        # Non-degeneracy: the control must not BE the treatment. Matching updates
        # pushes the allocation towards FedHAD's (client size and H_k are strongly
        # anti-correlated here), and in at least one measured cell the unconstrained
        # optimum was bit-identical to E_full - which would have collapsed
        # fixed_matched into epoch_only and made contrast (3) zero by construction.
        # Measured cost of forbidding it: the update residual went from 0.00% to
        # 0.06% in that cell.
        if cand == e_full:
            continue
        # Tie-break, in order: closest update total; then flattest allocation
        # (fewest clients bumped); then lexicographically smallest.
        candidates.append((abs(updates_of(cand) - u_full), bin(mask).count("1"),
                           tuple(cand), cand))
    if not candidates:
        raise RuntimeError("no admissible fixed_matched allocation found")
    e_fixed = min(candidates)[3]

    # ---- permuted_allocation -------------------------------------------------- #
    # Every UNIQUE non-trivial re-assignment of the exact FedHAD epoch multiset,
    # enumerated offline (K is small). The one whose predicted update total is
    # closest to FedHAD's is selected, exact equality preferred, all before any
    # training and without reading accuracy or loss.
    # Tie-break, in order: closest update total; then MOST clients whose budget
    # changed (maximally destroying the H -> compute association); then
    # lexicographically smallest assignment.
    seen, perm_candidates = set(), []
    for arrangement in itertools.permutations(e_full):
        if arrangement in seen:
            continue
        seen.add(arrangement)
        if list(arrangement) == e_full:
            continue  # trivial: identical assignment, no association destroyed
        changed = sum(1 for a, b_ in zip(arrangement, e_full) if a != b_)
        perm_candidates.append((abs(updates_of(arrangement) - u_full), -changed,
                                arrangement, list(arrangement)))
    # Degenerate case: if FedHAD gives every client the same budget, the multiset
    # is uniform and NO non-trivial re-assignment exists - permuted_allocation is
    # undefined for that cell. Flagged rather than raised here, so that the other
    # four arms still run; policy_for() raises only if this arm is the active one.
    perm_degenerate = not perm_candidates
    e_perm = e_full if perm_degenerate else min(perm_candidates)[3]

    # ---- invariants that abort the run ------------------------------------- #
    assert sorted(e_perm) == sorted(e_full), (
        "permuted_allocation must reuse the exact multiset of FedHAD epoch budgets")
    if not perm_degenerate:
        assert e_perm != e_full, (
            "permuted_allocation must actually change the client -> epoch association")
    assert min(e_fixed) >= min_epochs, (
        f"fixed_matched would allocate {min(e_fixed)} epochs, below MIN_EPOCHS={min_epochs}")
    assert e_fixed == min(candidates)[3], "fixed_matched must be the argmin found"
    assert e_fixed != e_full, (
        "fixed_matched must not coincide with the FedHAD allocation, otherwise it "
        "is not a control")
    partition_hashes = [sha256_of(_resolve_indices(dl.dataset)) for dl in trainloaders]

    def budget(epochs, batch_size=32):
        """
        total_updates is the MATCHED quantity. Two FLOP figures are reported:
          flops_nominal  the campaign's own formula, n_k * E_k (counts every
                         sample, including the tail the DataLoader drops)
          flops_realised batches * batch_size * E_k, i.e. the samples actually
                         fed to the optimiser under drop_last
        """
        updates = int(sum(e * b for e, b in zip(epochs, n_batches)))
        flops_nominal = int(sum((flops_per_sample_full or 0) * n * e
                                for n, e in zip(n_samples, epochs)))
        flops_realised = int(sum((flops_per_sample_full or 0) * b * batch_size * e
                                 for b, e in zip(n_batches, epochs)))
        return {"total_epochs": int(sum(epochs)), "total_updates": updates,
                "total_flops": flops_nominal, "total_flops_realised": flops_realised,
                "updates_delta_vs_full": updates - u_full,
                "updates_pct_vs_full": (100.0 * (updates - u_full) / u_full
                                        if u_full else 0.0)}

    _STATE.update({
        "seed": int(seed), "n_clients": n_clients,
        "H": [float(h) for h in h_values],
        "E_full": e_full, "LR_full": lr_full,
        "E_fixed": e_fixed, "E_perm": e_perm,
        # The chosen arrangement is selected by enumeration + update matching, not
        # by a random permutation, so what is recorded for audit is the mapping it
        # implies: for each client, the budget it holds before and after.
        "perm_mapping": [{"client": i, "E_full": int(a), "E_perm": int(b)}
                         for i, (a, b) in enumerate(zip(e_full, e_perm))],
        "base_lr": float(base_lr),
        "n_samples": n_samples, "n_batches": n_batches,
        "partition_hashes": partition_hashes,
        "budget_full": budget(e_full), "budget_fixed": budget(e_fixed),
        "budget_perm": budget(e_perm),
        "u_full_target": u_full,
        "fixed_search_space": "E_k in {e0, e0+1}, e0 = floor(U_full / sum(batches))",
        "fixed_tiebreak": "min |dU|, then fewest clients bumped, then lexicographic",
        "perm_tiebreak": "min |dU|, then most clients changed, then lexicographic",
        "perm_n_unique_nontrivial": len(perm_candidates),
        "perm_degenerate": bool(perm_degenerate),
        "perm_clients_changed": int(sum(1 for a, b_ in zip(e_perm, e_full) if a != b_)),
        "fixed_clients_changed": int(sum(1 for a, b_ in zip(e_fixed, e_full) if a != b_)),
        # Contamination diagnostic: matching updates requires favouring large
        # clients, and client size is anti-correlated with H_k, so the H-blind
        # allocation can end up resembling the H-driven one. Reported per cell so
        # the reader can judge how clean the control actually is.
        "rank_corr_E_full_vs_H": _spearman(e_full, h_values),
        "rank_corr_E_fixed_vs_H": _spearman(e_fixed, h_values),
        "rank_corr_E_perm_vs_H": _spearman(e_perm, h_values),
        "rank_corr_n_vs_H": _spearman(n_samples, h_values),
    })
    return _STATE


def policy_for(cid: int):
    """(epochs, learning_rate) for this client under the active variant."""
    if not _STATE:
        raise RuntimeError("build_allocation() must run before policy_for()")
    v, cid = active_variant(), int(cid)
    if v == "full_fedhad":
        return _STATE["E_full"][cid], _STATE["LR_full"][cid]
    if v == "fixed_matched":
        return _STATE["E_fixed"][cid], _STATE["base_lr"]
    if v == "permuted_allocation":
        if _STATE.get("perm_degenerate"):
            raise RuntimeError(
                "permuted_allocation is undefined for this cell: FedHAD assigned every "
                "client the same epoch budget, so no non-trivial re-assignment "
                "exists. Exclude the cell for ALL arms rather than comparing an arm "
                "that silently equals full_fedhad.")
        return _STATE["E_perm"][cid], _STATE["LR_full"][cid]
    if v == "epoch_only":
        return _STATE["E_full"][cid], _STATE["base_lr"]
    if v == "lr_only_matched":
        return _STATE["E_fixed"][cid], _STATE["LR_full"][cid]
    raise ValueError(v)


def variant_epochs_and_lr():
    """The active variant's full per-client allocation, for the fingerprint."""
    return [policy_for(c) for c in range(_STATE["n_clients"])]


def mark_round():
    """Wall-clock stamp, called once per centralised evaluation (once per round)."""
    _ROUND_CLOCK.append(time.time())


def round_wall_clock():
    """Seconds elapsed per round, derived from the centralised-evaluation stamps."""
    if len(_ROUND_CLOCK) < 2:
        return []
    return [round(b - a, 4) for a, b in zip(_ROUND_CLOCK[:-1], _ROUND_CLOCK[1:])]


def fingerprint(initial_state_dict, dataset, alpha, num_rounds, batch_size,
                extra=None) -> dict:
    """
    Everything the runner needs to prove the arms are paired: identical initial
    weights, identical partitions, identical clients, identical H, and the budget
    each variant actually used.
    """
    alloc = variant_epochs_and_lr()
    v = active_variant()
    budget_key = {"full_fedhad": "budget_full", "epoch_only": "budget_full",
                  "fixed_matched": "budget_fixed", "lr_only_matched": "budget_fixed",
                  "permuted_allocation": "budget_perm"}[v]
    fp = {
        "variant": v, "dataset": dataset, "alpha": float(alpha),
        "seed": _STATE["seed"], "num_rounds": int(num_rounds),
        "batch_size": int(batch_size), "n_clients": _STATE["n_clients"],
        "initial_weights_sha256": state_dict_sha256(initial_state_dict),
        "partition_hashes": _STATE["partition_hashes"],
        "partition_hash_all": sha256_of(_STATE["partition_hashes"]),
        "H": _STATE["H"], "H_hash": sha256_of([round(h, 12) for h in _STATE["H"]]),
        "n_samples": _STATE["n_samples"], "n_batches": _STATE["n_batches"],
        "epochs_assigned": [int(e) for e, _ in alloc],
        "lr_assigned": [float(l) for _, l in alloc],
        "E_full": _STATE["E_full"], "E_fixed": _STATE["E_fixed"],
        "E_perm": _STATE["E_perm"], "perm_mapping": _STATE["perm_mapping"],
        # Per-seed diagnostics required by the analysis: how strongly each
        # allocation tracks H_k, how far the control sits from the treatment, and
        # whether the permuted arm was degenerate for this cell.
        "rank_corr_n_vs_H": _STATE["rank_corr_n_vs_H"],
        "rank_corr_E_full_vs_H": _STATE["rank_corr_E_full_vs_H"],
        "rank_corr_E_fixed_vs_H": _STATE["rank_corr_E_fixed_vs_H"],
        "rank_corr_E_perm_vs_H": _STATE["rank_corr_E_perm_vs_H"],
        "hamming_fixed_vs_full": _STATE["fixed_clients_changed"],
        "hamming_perm_vs_full": _STATE["perm_clients_changed"],
        "perm_degenerate": _STATE["perm_degenerate"],
        "perm_n_unique_nontrivial": _STATE["perm_n_unique_nontrivial"],
        "u_full_target": _STATE["u_full_target"],
        "LR_full": _STATE["LR_full"], "base_lr": _STATE["base_lr"],
        "budget": _STATE[budget_key],
        "budget_full": _STATE["budget_full"],
        "budget_fixed": _STATE["budget_fixed"],
        "budget_perm": _STATE["budget_perm"],
    }
    if extra:
        fp.update(extra)
    return fp


def write_run_artifacts(out_dir, fp, history, fit_metrics_history, elapsed):
    """Per-run raw artifacts: fingerprint JSON + per-round per-client telemetry."""
    import csv
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{fp['variant']}__{fp['dataset']}__alpha-{fp['alpha']}__seed-{fp['seed']}"

    fp = dict(fp)
    fp["elapsed_seconds"] = round(float(elapsed), 3)
    fp["round_wall_clock"] = round_wall_clock()
    (out_dir / f"fingerprint__{tag}.json").write_text(
        json.dumps(fp, indent=2, sort_keys=True), encoding="utf-8")

    mc = getattr(history, "metrics_centralized", {}) or {}
    acc_c = dict(mc.get("accuracy", []))
    loss_c = dict(getattr(history, "losses_centralized", []) or [])
    md = getattr(history, "metrics_distributed", {}) or {}
    acc_d = dict(md.get("accuracy", []))
    loss_d = dict(getattr(history, "losses_distributed", []) or [])
    wall = round_wall_clock()

    rows = []
    for r_idx, round_metrics in enumerate(fit_metrics_history, start=1):
        for num_examples, m in round_metrics:
            rows.append({
                "variant": fp["variant"], "dataset": fp["dataset"],
                "alpha": fp["alpha"], "seed": fp["seed"], "round": r_idx,
                "client_id": m.get("cid", ""), "n_k": int(num_examples),
                "H_k": m.get("H_k", ""), "epochs": m.get("epochs", ""),
                "learning_rate": m.get("learning_rate", ""),
                "minibatch_updates": m.get("minibatch_updates", ""),
                "flops_locais": m.get("flops_locais", ""),
                "loss_after_fit": m.get("loss_after_fit", ""),
                "acc_centralized_round": acc_c.get(r_idx, ""),
                "loss_centralized_round": loss_c.get(r_idx, ""),
                "acc_distributed_round": acc_d.get(r_idx, ""),
                "loss_distributed_round": loss_d.get(r_idx, ""),
                "round_wall_clock_s": wall[r_idx - 1] if r_idx - 1 < len(wall) else "",
            })
    if rows:
        with (out_dir / f"telemetry__{tag}.csv").open("w", newline="",
                                                      encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return tag
