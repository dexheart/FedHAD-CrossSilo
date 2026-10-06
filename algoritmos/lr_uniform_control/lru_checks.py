# -*- coding: utf-8 -*-
"""
Completeness validation and result assembly for the LR-uniform battery. Pure
functions over plain dicts/lists (unit-tested); the runner does the file I/O.

A run counts as COMPLETE only if:
  * the required fields are present;
  * it recorded every round;
  * every client WITH data reported in every round (the failure that forced
    re-executions in the past);
  * its initial-checkpoint SHA-256 equals the one expected for that seed;
  * arm, alpha and seed match the cell (and the code fingerprint, if given).
"""
from __future__ import annotations

REQUIRED_FIELDS = (
    "arm", "alpha", "seed", "num_rounds", "init_sha256", "code_fingerprint",
    "per_round", "final_accuracy", "totals", "clients", "eta_bar",
    "sum_p_eta_tau_executed", "platform", "elapsed_seconds", "finished_at",
)


def data_clients(n_samples):
    return [k for k, n in enumerate(n_samples) if int(n) > 0]


def participation_by_round(telemetry_rows):
    """{round: set(client ids with n_k > 0 that reported)} from telemetry rows."""
    out = {}
    for r in telemetry_rows:
        rnd = int(r["round"])
        out.setdefault(rnd, set())
        if int(float(r.get("n_k") or 0)) > 0:
            out[rnd].add(int(r["client_id"]))
    return out


def sum_p_eta_tau_executed(telemetry_rows, n_samples):
    """Per-round and mean executed sum_k p_k eta_k tau_k, with tau_k the steps actually
    executed and eta_k the rate actually used, from the telemetry."""
    total = float(sum(n_samples))
    per_round = {}
    for r in telemetry_rows:
        rnd = int(r["round"])
        k = int(r["client_id"])
        p = float(n_samples[k]) / total
        per_round[rnd] = per_round.get(rnd, 0.0) + p * float(r["learning_rate"] or 0) * float(r["steps_executed"] or 0)
    vals = [per_round[k] for k in sorted(per_round)]
    return {"per_round": vals, "mean": (sum(vals) / len(vals)) if vals else None}


def validate_result(res, *, arm, alpha, seed, num_rounds, expected_init_sha, code_fingerprint=None):
    """(True, "") if complete, else (False, reason)."""
    if not isinstance(res, dict):
        return False, "result missing or unreadable"
    missing = [f for f in REQUIRED_FIELDS if f not in res or res[f] is None]
    if missing:
        return False, f"missing fields: {missing}"
    if res["arm"] != arm or int(res["seed"]) != int(seed) or abs(float(res["alpha"]) - float(alpha)) > 1e-12:
        return False, "result does not match the cell (arm/alpha/seed)"
    if int(res["num_rounds"]) != int(num_rounds):
        return False, f"num_rounds {res['num_rounds']} != {num_rounds}"
    rounds = [int(x["round"]) for x in res["per_round"]]
    if sorted(rounds) != list(range(1, int(num_rounds) + 1)):
        return False, f"rounds recorded {sorted(rounds)} != 1..{num_rounds}"
    if any(x.get("acc_centralized") in (None, "") for x in res["per_round"]):
        return False, "a round has no centralized accuracy"
    need = set(data_clients(res["clients"]["n_samples"]))
    for x in res["per_round"]:
        got = set(int(c) for c in x.get("clients_with_data_reported", []))
        if not need <= got:
            return False, f"round {x['round']}: clients with data missing {sorted(need - got)}"
    if res["init_sha256"] != expected_init_sha:
        return False, "initial checkpoint checksum mismatch"
    if code_fingerprint is not None and res["code_fingerprint"] != code_fingerprint:
        return False, "produced by different code (code fingerprint)"
    return True, ""


def assemble_result(*, fingerprint, telemetry_rows, cell, code_fingerprint, platform_info,
                    started_at, finished_at):
    """Result record written (atomically) by the runner from the run's raw artifacts."""
    fp = fingerprint
    n_samples = fp["n_samples"]
    part = participation_by_round(telemetry_rows)
    rounds = {}
    for r in telemetry_rows:
        rnd = int(r["round"])
        rounds.setdefault(rnd, {"round": rnd, "acc_centralized": r.get("acc_centralized_round"),
                                "loss_centralized": r.get("loss_centralized_round")})
    per_round = []
    for rnd in sorted(rounds):
        x = rounds[rnd]
        per_round.append({
            "round": rnd,
            "acc_centralized": float(x["acc_centralized"]) if x["acc_centralized"] not in (None, "") else None,
            "loss_centralized": float(x["loss_centralized"]) if x["loss_centralized"] not in (None, "") else None,
            "clients_with_data_reported": sorted(part.get(rnd, set())),
        })
    ex = sum_p_eta_tau_executed(telemetry_rows, n_samples)
    return {
        "arm": cell["arm"], "alpha": cell["alpha"], "seed": cell["seed"],
        "num_rounds": cell["num_rounds"], "level": cell.get("level"),
        "init_sha256": fp.get("initial_checkpoint", {}).get("sha256"),
        "code_fingerprint": code_fingerprint,
        "per_round": per_round,
        "final_accuracy": fp.get("final", {}).get("acc_centralized"),
        "final_loss": fp.get("final", {}).get("loss_centralized"),
        "totals": fp.get("totals"),
        "clients": {"n_samples": n_samples, "n_batches": fp.get("n_batches"), "H": fp.get("H"),
                    "E": fp.get("E_arm"), "lr": fp.get("LR_arm"), "tau": fp.get("tau_received"),
                    "E_full": fp.get("E_full"), "E_fixed": fp.get("E_fixed"),
                    "LR_full": fp.get("LR_full"), "p_k": fp.get("p_k"),
                    "has_empty_client": any(int(n) == 0 for n in n_samples)},
        "eta_bar": {k: fp.get(k) for k in ("eta_bar_allocation", "eta_bar_primary_p_tau",
                                           "eta_bar_tau_only", "eta_bar_arithmetic",
                                           "reference_arm", "reference_sum_p_eta_tau")},
        "sum_p_eta_tau_planned": fp.get("planned_sum_p_eta_tau"),
        "sum_p_eta_tau_executed": ex["mean"],
        "sum_p_eta_tau_executed_per_round": ex["per_round"],
        "partition_hash_all": fp.get("partition_hash_all"), "H_hash": fp.get("H_hash"),
        "platform": {**(fp.get("hardware") or {}), **platform_info},
        "elapsed_seconds": fp.get("elapsed_seconds"), "round_wall_clock": fp.get("round_wall_clock"),
        "started_at": started_at, "finished_at": finished_at,
    }
