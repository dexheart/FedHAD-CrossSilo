# -*- coding: utf-8 -*-
"""
Completeness validation and result assembly for the global-gradient battery. Pure
functions over plain dicts/lists (unit-tested); the runner does the file I/O.

A run counts as COMPLETE only if every round was recorded, every client with data
reported in every round (telemetry and server diagnostics), the class-sensitive
evaluation exists for every round, the epoch diagnostics exist whenever a client
trained, the initial-checkpoint checksum is the expected one, and the cell and code
fingerprint match.
"""
from __future__ import annotations

REQUIRED_FIELDS = ("arm", "partition", "alpha", "seed", "num_rounds", "init_sha256",
                   "code_fingerprint", "final_accuracy", "final_class", "totals", "clients",
                   "diag_rows", "finished_at")


def _data_clients(n_samples):
    return {k for k, n in enumerate(n_samples) if int(n) > 0}


def check_diagnostics(*, n_samples, num_rounds, telemetry_rows, server_rows, epoch_rows, class_rows):
    """(True, "") or (False, reason)."""
    need = _data_clients(n_samples)
    rounds = list(range(1, int(num_rounds) + 1))
    by_round = {}
    for r in telemetry_rows:
        if int(float(r.get("n_k") or 0)) > 0:
            by_round.setdefault(int(r["round"]), set()).add(int(r["client_id"]))
    for t in rounds:
        if not need <= by_round.get(t, set()):
            return False, f"telemetry: round {t} lacks clients {sorted(need - by_round.get(t, set()))}"
    srv = {}
    for r in server_rows:
        srv.setdefault(int(r["round"]), set()).add(int(r["client_id"]))
    for t in rounds:
        if not need <= srv.get(t, set()):
            return False, f"server diagnostics: round {t} lacks clients {sorted(need - srv.get(t, set()))}"
    if sorted(int(r["round"]) for r in class_rows if int(r["round"]) >= 1) != rounds:
        return False, "class-sensitive evaluation missing for some round"
    trained = {(int(r["round"]), int(r["client_id"])) for r in telemetry_rows
               if int(float(r.get("steps_executed") or 0)) > 0}
    have = {(int(r["round"]), int(r["client_id"])) for r in epoch_rows}
    if not trained <= have:
        return False, f"epoch diagnostics missing for {len(trained - have)} client-rounds that trained"
    return True, ""


def validate_result(res, *, arm, partition, alpha, seed, num_rounds, expected_init_sha,
                    code_fingerprint=None):
    if not isinstance(res, dict):
        return False, "result missing or unreadable"
    missing = [f for f in REQUIRED_FIELDS if f not in res or res[f] is None]
    if missing:
        return False, f"missing fields: {missing}"
    if (res["arm"] != arm or res["partition"] != partition or int(res["seed"]) != int(seed)
            or abs(float(res["alpha"]) - float(alpha)) > 1e-12):
        return False, "result does not match the cell"
    if int(res["num_rounds"]) != int(num_rounds):
        return False, f"num_rounds {res['num_rounds']} != {num_rounds}"
    if res["init_sha256"] != expected_init_sha:
        return False, "initial checkpoint checksum mismatch"
    if code_fingerprint is not None and res["code_fingerprint"] != code_fingerprint:
        return False, "produced by different code (code fingerprint)"
    return True, ""


def assemble_result(*, fingerprint, cell, class_rows, server_rows, epoch_rows, code_fingerprint,
                    platform_info, started_at, finished_at):
    fp = fingerprint
    last = max(class_rows, key=lambda r: int(r["round"]))
    return {
        "arm": cell["arm"], "partition": cell["partition"], "alpha": cell["alpha"],
        "seed": cell["seed"], "num_rounds": cell["num_rounds"], "level": cell["level"],
        "init_sha256": fp.get("initial_checkpoint", {}).get("sha256"),
        "code_fingerprint": code_fingerprint,
        "final_accuracy": fp.get("final", {}).get("acc_centralized"),
        "final_class": {k: float(v) for k, v in last.items() if k != "round" and v not in (None, "")},
        "totals": fp.get("totals"),
        "clients": {"n_samples": fp.get("n_samples"), "n_batches": fp.get("n_batches"), "H": fp.get("H"),
                    "E_full": fp.get("E_full"), "E_arm": fp.get("E_arm"), "LR_arm": fp.get("LR_arm"),
                    "tau": fp.get("tau_received"), "weighted_steps_arm": fp.get("weighted_steps_arm"),
                    "weighted_steps_full": fp.get("weighted_steps_full")},
        "partition_hash_all": fp.get("partition_hash_all"),
        "diag_rows": {"server": len(server_rows), "epochs": len(epoch_rows), "class": len(class_rows)},
        "platform": {**(fp.get("hardware") or {}), **platform_info},
        "elapsed_seconds": fp.get("elapsed_seconds"),
        "started_at": started_at, "finished_at": finished_at,
    }
