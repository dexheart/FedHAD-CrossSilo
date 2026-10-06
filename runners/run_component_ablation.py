#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner for the FedHAD controlled-ablation campaign, with checkpoint/resume.

Launches one isolated subprocess per (dataset, alpha, seed, variant) cell, records
persistent state for every cell, and verifies that the five arms of each cell were
genuinely paired: identical initial weights, identical partitions, identical
clients, identical H_k, identical round count, and the compute budget each arm was
supposed to spend.

Resume model
------------
The campaign is designed to be interrupted. Re-running the SAME command resumes:
completed and still-valid cells are skipped, everything else is re-executed.

A cell counts as completed only if its ARTIFACTS are present and valid - the state
ledger is metadata, not the source of truth. A cell left as `running` by a killed
job has no valid artifacts and is therefore re-executed, never mistaken for done.

Validity requires all of:
  * fingerprint JSON exists, parses, and its variant / alpha / seed / dataset /
    num_rounds / n_clients match the campaign being run
  * telemetry CSV exists, parses, and holds num_rounds x n_clients rows
  * the campaign fingerprint recorded with the cell matches the current one
    (dataset, alphas, seeds, rounds, hyper-parameters, and the SHA-256 of the two
    source files that determine results)

That last check is what prevents silently reusing a result produced by a different
configuration or a different version of the code.

Commands
--------
    python runners/run_component_ablation.py                  # start OR resume the 300-run battery
    python runners/run_component_ablation.py --official       # identical, explicit
    python runners/run_component_ablation.py --resume         # identical, explicit
    python runners/run_component_ablation.py --status         # how many cells are done, no execution
    python runners/run_component_ablation.py --rerun-failed   # retry only the cells that failed
    python runners/run_component_ablation.py --smoke          # 2-seed validation grid
    python runners/run_component_ablation.py --self-test      # synthetic check of the resume logic
    python runners/run_component_ablation.py --official --dry-run
    python runners/run_component_ablation.py --verify-only --mode official
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algoritmos" / "component_analysis_ablation"))
import config_ablation as C  # noqa: E402

STATUS_PENDING, STATUS_RUNNING = "pending", "running"
STATUS_COMPLETED, STATUS_FAILED = "completed", "failed"


# --------------------------------------------------------------------------- #
# Paths and identity
# --------------------------------------------------------------------------- #
def python_executable() -> str:
    venv = C.ROOT.parent.parent / "env_flwr_pt" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def run_dir(mode: str) -> Path:
    return C.RAW / ("smoke" if mode == "smoke" else "official")


def state_dir(mode: str) -> Path:
    return run_dir(mode) / "_state"


def cell_id(variant, alpha, seed):
    return f"{variant}__{C.DATASET}__alpha-{alpha}__seed-{seed}"


def fingerprint_path(raw_dir, variant, alpha, seed):
    return raw_dir / f"fingerprint__{cell_id(variant, alpha, seed)}.json"


def telemetry_path(raw_dir, variant, alpha, seed):
    return raw_dir / f"telemetry__{cell_id(variant, alpha, seed)}.csv"


def state_path(mode, variant, alpha, seed):
    return state_dir(mode) / f"{cell_id(variant, alpha, seed)}.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def campaign_fingerprint(mode) -> dict:
    """
    Identity of THIS campaign. Any change here invalidates previously completed
    cells, so a resumed run can never mix results from different configurations.
    Only files that can change a result are hashed; the runner itself is not.
    """
    payload = {
        "dataset": C.DATASET, "alphas": C.ALPHAS, "client_setup": C.CLIENT_SETUP,
        "num_rounds": C.NUM_ROUNDS, "comm_delay": C.COMM_DELAY,
        "use_energy": C.USE_ENERGY, "variants": C.VARIANTS,
        "seeds": C.SEEDS_SMOKE if mode == "smoke" else C.SEEDS_OFFICIAL,
        "base_epochs": C.BASE_EPOCHS, "min_epochs": C.MIN_EPOCHS,
        "epochs_decay": C.EPOCHS_DECAY, "base_lr": C.BASE_LR, "min_lr": C.MIN_LR,
        "lr_decay": C.LR_DECAY, "fedprox_mu": C.FEDPROX_MU,
        "sha256_fedhad_ablation": _sha256_file(C.CODE / "fedhad_ablation.py"),
        "sha256_ablation_policy": _sha256_file(C.CODE / "ablation_policy.py"),
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


# --------------------------------------------------------------------------- #
# Atomic state ledger
# --------------------------------------------------------------------------- #
def write_state_atomic(path: Path, payload: dict) -> None:
    """Temporary file + os.replace, so an interrupted write never leaves a partial
    state file that a later run could misread."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_state(mode, variant, alpha, seed):
    p = state_path(mode, variant, alpha, seed)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Integrity validation
# --------------------------------------------------------------------------- #
def validate_artifacts(mode, variant, alpha, seed, camp):
    """
    Artifacts are the source of truth. Returns (is_valid, reason_if_not).
    A state file alone is never enough to skip a cell.
    """
    raw = run_dir(mode)
    fp_p = fingerprint_path(raw, variant, alpha, seed)
    tel_p = telemetry_path(raw, variant, alpha, seed)
    if not fp_p.exists():
        return False, "fingerprint missing"
    if not tel_p.exists():
        return False, "telemetry missing"
    try:
        fp = json.loads(fp_p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "fingerprint unreadable"

    expected = {"variant": variant, "alpha": float(alpha), "seed": int(seed),
                "dataset": C.DATASET, "num_rounds": int(C.NUM_ROUNDS),
                "n_clients": int(C.CLIENT_SETUP)}
    for k, want in expected.items():
        got = fp.get(k)
        if isinstance(want, float):
            if got is None or abs(float(got) - want) > 1e-12:
                return False, f"{k} mismatch ({got!r} != {want!r})"
        elif got != want:
            return False, f"{k} mismatch ({got!r} != {want!r})"

    try:
        with tel_p.open(encoding="utf-8") as fh:
            n_rows = sum(1 for _ in fh) - 1  # minus header
    except OSError:
        return False, "telemetry unreadable"
    want_rows = int(C.NUM_ROUNDS) * int(C.CLIENT_SETUP)
    if n_rows != want_rows:
        return False, f"telemetry has {n_rows} rows, expected {want_rows}"

    st = read_state(mode, variant, alpha, seed)
    if st is None:
        # Artifacts without a state entry: provenance cannot be verified, so the
        # cell is NOT silently reused. `--adopt-orphans` validates and adopts such
        # cells explicitly, which is the only way they ever count as completed.
        return False, "orphan artifacts (no state entry; provenance unverifiable)"
    recorded = st.get("campaign_fingerprint")
    if recorded != camp["fingerprint"]:
        return False, "produced by a different campaign configuration"
    return True, ""


def cell_status(mode, variant, alpha, seed, camp):
    """
    Effective status. A `running` state with no valid artifacts (a job killed
    mid-cell) resolves to pending, so the cell is re-executed rather than skipped.
    """
    ok, why = validate_artifacts(mode, variant, alpha, seed, camp)
    st = read_state(mode, variant, alpha, seed)
    if ok:
        return STATUS_COMPLETED, ""
    if st is None:
        return STATUS_PENDING, why
    recorded = st.get("status")
    if recorded == STATUS_FAILED:
        return STATUS_FAILED, st.get("error_message") or why
    if recorded == STATUS_RUNNING:
        return STATUS_PENDING, "interrupted while running; artifacts incomplete"
    return STATUS_PENDING, why


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def build_env(variant, alpha, seed, raw_dir, repeat_tag=""):
    env = os.environ.copy()
    env.update({
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "FL_DATASET": C.DATASET, "FL_ALPHA": str(alpha), "FL_SEED": str(seed),
        "FL_NUM_ROUNDS": str(C.NUM_ROUNDS),
        "FL_CLIENT_SETUP": str(C.CLIENT_SETUP),
        "FL_COMM_DELAY": str(C.COMM_DELAY),
        "FL_USE_ENERGY": "1" if C.USE_ENERGY else "0",
        "FL_ABLATION_VARIANT": variant,
        "FL_ABLATION_RAW_DIR": str(raw_dir),
        "FL_RUN_TAG": f"ablation{repeat_tag}",
    })
    return env


def execute(mode, variant, alpha, seed, camp, logs_dir, repeat_tag="",
            raw_dir_override=None):
    """
    Run one cell. `running` is recorded before launching and the final state is
    written atomically afterwards, with timestamps, duration, exit status and any
    error message.
    """
    raw_dir = raw_dir_override or run_dir(mode)
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    cid = cell_id(variant, alpha, seed)
    log = logs_dir / f"{cid}{repeat_tag}.log"
    sp = state_path(mode, variant, alpha, seed)

    prev = read_state(mode, variant, alpha, seed) or {}
    base = {
        "cell_id": cid, "dataset": C.DATASET, "alpha": float(alpha),
        "seed": int(seed), "variant": variant,
        "campaign_fingerprint": camp["fingerprint"],
        "attempts": int(prev.get("attempts", 0)) + 1,
        "hostname": socket.gethostname(), "python": platform.python_version(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {
            "fingerprint": fingerprint_path(raw_dir, variant, alpha, seed).name,
            "telemetry": telemetry_path(raw_dir, variant, alpha, seed).name,
            "log": str(log.relative_to(C.ROOT)),
        },
    }
    if not repeat_tag:
        write_state_atomic(sp, {**base, "status": STATUS_RUNNING,
                                "finished_at": None, "duration_s": None,
                                "exit_status": None, "error_message": ""})

    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [python_executable(), "-u", str(C.ENTRY_SCRIPT)],
            cwd=str(C.REPO),  # datasets under <repo>/data
            env=build_env(variant, alpha, seed, raw_dir, repeat_tag),
            stdout=fh, stderr=subprocess.STDOUT,
        )
    dur = time.time() - t0

    if raw_dir_override is not None:
        ok, why = proc.returncode == 0, ""
    else:
        ok, why = validate_artifacts(mode, variant, alpha, seed, camp)
    err = ""
    if proc.returncode != 0:
        err = f"exit status {proc.returncode}; see {log.name}"
    elif not ok:
        err = f"exit 0 but artifacts invalid: {why}"

    if not repeat_tag:
        write_state_atomic(sp, {
            **base,
            "status": STATUS_COMPLETED if (proc.returncode == 0 and ok) else STATUS_FAILED,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(dur, 2), "exit_status": proc.returncode,
            "error_message": err,
        })
    return (proc.returncode == 0 and ok), dur, err


# --------------------------------------------------------------------------- #
# Planning and status
# --------------------------------------------------------------------------- #
def all_cells(mode):
    seeds = C.SEEDS_SMOKE if mode == "smoke" else C.SEEDS_OFFICIAL
    return [(v, a, s) for a in C.ALPHAS for s in seeds for v in C.VARIANTS]


def survey(mode, camp):
    counts, detail = defaultdict(int), []
    for v, a, s in all_cells(mode):
        st, why = cell_status(mode, v, a, s, camp)
        counts[st] += 1
        detail.append((v, a, s, st, why))
    return counts, detail


def print_status(mode, camp):
    counts, detail = survey(mode, camp)
    total = sum(counts.values())
    done = counts[STATUS_COMPLETED]
    seeds = C.SEEDS_SMOKE if mode == "smoke" else C.SEEDS_OFFICIAL
    print("=" * 78)
    print(f"CAMPAIGN STATUS - {mode}")
    print("=" * 78)
    print(f"Campaign fingerprint : {camp['fingerprint'][:16]}")
    print(f"Grid                 : {C.DATASET} | alphas {C.ALPHAS} | "
          f"{len(seeds)} seeds | {len(C.VARIANTS)} arms")
    print(f"Cells                : {total}")
    print(f"  completed          : {done}  ({100 * done / total:.1f}%)")
    print(f"  pending            : {counts[STATUS_PENDING]}")
    print(f"  failed             : {counts[STATUS_FAILED]}")

    per_arm = defaultdict(lambda: [0, 0])
    for v, a, s, st, _ in detail:
        per_arm[v][1] += 1
        if st == STATUS_COMPLETED:
            per_arm[v][0] += 1
    print("\n  by arm:")
    for v in C.VARIANTS:
        d, t = per_arm[v]
        print(f"    {C.VARIANT_LABEL[v]:<22} {d:>3}/{t}")

    print("\n  by alpha:")
    for alpha in C.ALPHAS:
        d = sum(1 for v, a, s, st, _ in detail if a == alpha and st == STATUS_COMPLETED)
        t = sum(1 for v, a, s, _, _ in detail if a == alpha)
        print(f"    alpha={alpha:<6} {d:>3}/{t}")

    fails = [(v, a, s, why) for v, a, s, st, why in detail if st == STATUS_FAILED]
    if fails:
        print(f"\n  failed cells ({len(fails)}):")
        for v, a, s, why in fails[:10]:
            print(f"    {cell_id(v, a, s)}: {why}")
        if len(fails) > 10:
            print(f"    ... and {len(fails) - 10} more")

    durs = [st["duration_s"] for v, a, s in all_cells(mode)
            if (st := read_state(mode, v, a, s)) and st.get("duration_s")]
    if durs:
        mean = sum(durs) / len(durs)
        left = counts[STATUS_PENDING] + counts[STATUS_FAILED]
        print(f"\n  mean cell duration : {mean:.0f}s over {len(durs)} recorded runs")
        print(f"  estimated time left: {left * mean / 3600:.1f} h for {left} cells")
    if done == total:
        print(f"\n  Complete. Analyse with:  python analysis/component_analysis_ablation/analyze_ablation.py --mode {mode}")
    else:
        print(f"\n  Resume with:  python runners/run_component_ablation.py "
              f"{'--smoke' if mode == 'smoke' else '--resume'}")
    return counts


# --------------------------------------------------------------------------- #
# Fairness verification
# --------------------------------------------------------------------------- #
def load_fingerprints(raw_dir):
    fps = defaultdict(dict)
    for p in sorted(raw_dir.glob("fingerprint__*.json")):
        try:
            fp = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        fps[(fp["alpha"], fp["seed"])][fp["variant"]] = fp
    return fps


def verify_cell(alpha, seed, arms):
    failures = []
    missing = [v for v in C.VARIANTS if v not in arms]
    if missing:
        return [f"missing arms: {missing}"]

    for field in C.IDENTICAL_ACROSS_VARIANTS:
        values = {v: json.dumps(arms[v][field], sort_keys=True) for v in C.VARIANTS}
        if len(set(values.values())) != 1:
            failures.append(f"{field} differs across arms")

    u_full = arms["full_fedhad"]["budget"]["total_updates"]
    for v in C.BUDGET_EXACT_ARMS:
        if arms[v]["budget"]["total_updates"] != u_full:
            failures.append(
                f"{v} must match full_fedhad's update total exactly: "
                f"{arms[v]['budget']['total_updates']} != {u_full}")

    ref = arms["full_fedhad"]
    e_full, e_fixed, lr_full = ref["E_full"], ref["E_fixed"], ref["LR_full"]
    base_lr = ref["base_lr"]

    def eq(a, b, tol=1e-12):
        return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))

    checks = {
        "full_fedhad": lambda a: (a["epochs_assigned"] == e_full and
                                  eq(a["lr_assigned"], lr_full)),
        "fixed_matched": lambda a: (a["epochs_assigned"] == e_fixed and
                                    all(abs(l - base_lr) <= 1e-12 for l in a["lr_assigned"])),
        "permuted_allocation": lambda a: (sorted(a["epochs_assigned"]) == sorted(e_full) and
                                          eq(a["lr_assigned"], lr_full)),
        "epoch_only": lambda a: (a["epochs_assigned"] == e_full and
                                 all(abs(l - base_lr) <= 1e-12 for l in a["lr_assigned"])),
        "lr_only_matched": lambda a: (a["epochs_assigned"] == e_fixed and
                                      eq(a["lr_assigned"], lr_full)),
    }
    for v, check in checks.items():
        if not check(arms[v]):
            failures.append(f"{v} violates its invariant ({C.VARIANT_INVARIANTS[v]})")

    if arms["epoch_only"]["epochs_assigned"] != arms["full_fedhad"]["epochs_assigned"]:
        failures.append("epoch_only must reuse full_fedhad's epoch allocation exactly")
    if arms["lr_only_matched"]["epochs_assigned"] != arms["fixed_matched"]["epochs_assigned"]:
        failures.append("lr_only_matched must reuse fixed_matched's epoch allocation exactly")
    return failures


def budget_table(fps):
    rows = []
    for (alpha, seed), arms in sorted(fps.items()):
        if "full_fedhad" not in arms:
            continue
        ref = arms["full_fedhad"]
        for v in C.VARIANTS:
            if v not in arms:
                continue
            b = arms[v]["budget"]
            rows.append({
                "dataset": C.DATASET, "alpha": alpha, "seed": seed, "variant": v,
                "total_epochs": b["total_epochs"],
                "total_updates": b["total_updates"],
                "updates_pct_vs_full": round(b.get("updates_pct_vs_full", 0.0), 4),
                "total_flops": b["total_flops"],
                "total_flops_realised": b.get("total_flops_realised", ""),
                "epochs_match_full": b["total_epochs"] == ref["budget"]["total_epochs"],
                "rho_n_vs_H": arms[v].get("rank_corr_n_vs_H"),
                "rho_E_full_vs_H": arms[v].get("rank_corr_E_full_vs_H"),
                "rho_E_fixed_vs_H": arms[v].get("rank_corr_E_fixed_vs_H"),
                "rho_E_perm_vs_H": arms[v].get("rank_corr_E_perm_vs_H"),
                "hamming_fixed_vs_full": arms[v].get("hamming_fixed_vs_full"),
                "hamming_perm_vs_full": arms[v].get("hamming_perm_vs_full"),
                "perm_degenerate": arms[v].get("perm_degenerate"),
            })
    return rows


def verify(mode, expected_seeds, verbose=True):
    fps = load_fingerprints(run_dir(mode))
    failures, complete = {}, 0
    for key in sorted({(a, s) for a in C.ALPHAS for s in expected_seeds}):
        arms = fps.get(key, {})
        if len(arms) < len(C.VARIANTS):
            continue   # not finished yet: not a fairness failure
        complete += 1
        f = verify_cell(key[0], key[1], arms)
        if f:
            failures[key] = f
    if verbose:
        print(f"\nFairness verification over {complete} fully-populated "
              f"(alpha, seed) cells x {len(C.VARIANTS)} arms")
        if failures:
            print(f"  FAILED in {len(failures)} cell(s):")
            for key, f in list(failures.items())[:10]:
                print(f"    alpha={key[0]} seed={key[1]}")
                for msg in f:
                    print(f"      - {msg}")
        elif complete:
            print("  PASS - every complete cell shares initial weights, partitions, "
                  "clients, H_k, round count, and respects its budget invariant")
        else:
            print("  (no complete cells yet)")
    return fps, failures


def write_budget_table(fps, mode):
    import csv
    rows = budget_table(fps)
    if not rows:
        return None
    C.TABLES.mkdir(parents=True, exist_ok=True)
    out = C.TABLES / f"budget_verification_{mode}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out, rows


# --------------------------------------------------------------------------- #
# Reproducibility check (smoke only)
# --------------------------------------------------------------------------- #
def check_reproducibility(mode, camp, logs_dir):
    v, seed, alpha = C.REPRODUCIBILITY_VARIANT, C.REPRODUCIBILITY_SEED, C.ALPHAS[0]
    print(f"\nReproducibility check: re-running {v} alpha={alpha} seed={seed}")
    first_path = fingerprint_path(run_dir(mode), v, alpha, seed)
    if not first_path.exists():
        print("  SKIPPED: the first run's fingerprint is not present")
        return True
    first = json.loads(first_path.read_text(encoding="utf-8"))

    repeat_dir = run_dir(mode) / "repeat"
    ok, secs, err = execute(mode, v, alpha, seed, camp, logs_dir,
                            repeat_tag="_repeat", raw_dir_override=repeat_dir)
    if not ok:
        print(f"  FAILED: the repeat run did not complete ({err})")
        return False
    second = json.loads(
        fingerprint_path(repeat_dir, v, alpha, seed).read_text(encoding="utf-8"))
    print(f"  repeat completed in {secs:.0f}s")

    deterministic = ["initial_weights_sha256", "partition_hash_all", "H_hash",
                     "epochs_assigned", "lr_assigned", "n_samples", "n_batches",
                     "E_full", "E_fixed", "E_perm", "perm_mapping", "budget"]
    bad = [f for f in deterministic
           if json.dumps(first.get(f), sort_keys=True) !=
           json.dumps(second.get(f), sort_keys=True)]
    if bad:
        print(f"  FAILED: fields differ between two runs of the same seed: {bad}")
        return False
    print(f"  PASS - all {len(deterministic)} deterministic fields identical "
          f"across two runs of seed {seed}")
    return True


def adopt_orphans(mode, camp) -> int:
    """
    Adopt cells whose artifacts exist but which carry no state entry (for example,
    produced before the ledger existed). Every structural check except the
    campaign-fingerprint comparison is applied first; adoption is explicit,
    logged, and never happens as a side effect of a normal run.
    """
    adopted, rejected = [], []
    for v, a, s in all_cells(mode):
        if read_state(mode, v, a, s) is not None:
            continue
        raw = run_dir(mode)
        if not fingerprint_path(raw, v, a, s).exists():
            continue
        # Temporarily accept the missing state entry to run every other check.
        write_state_atomic(state_path(mode, v, a, s), {
            "cell_id": cell_id(v, a, s), "status": STATUS_RUNNING,
            "campaign_fingerprint": camp["fingerprint"]})
        ok, why = validate_artifacts(mode, v, a, s, camp)
        if ok:
            write_state_atomic(state_path(mode, v, a, s), {
                "cell_id": cell_id(v, a, s), "dataset": C.DATASET,
                "alpha": float(a), "seed": int(s), "variant": v,
                "status": STATUS_COMPLETED,
                "campaign_fingerprint": camp["fingerprint"],
                "adopted": True,
                "adopted_at": datetime.now().isoformat(timespec="seconds"),
                "note": "artifacts pre-dated the state ledger; adopted after "
                        "passing every structural validity check",
                "duration_s": None, "exit_status": 0, "error_message": ""})
            adopted.append(cell_id(v, a, s))
        else:
            state_path(mode, v, a, s).unlink(missing_ok=True)
            rejected.append((cell_id(v, a, s), why))

    print("=" * 78)
    print(f"ADOPTING ORPHAN ARTIFACTS - {mode}")
    print("=" * 78)
    print(f"Campaign fingerprint : {camp['fingerprint'][:16]}")
    print(f"Adopted   : {len(adopted)} cell(s)")
    for cid in adopted[:10]:
        print(f"    {cid}")
    if len(adopted) > 10:
        print(f"    ... and {len(adopted) - 10} more")
    print(f"Rejected  : {len(rejected)} cell(s)")
    for cid, why in rejected[:10]:
        print(f"    {cid}: {why}")
    print("\nAdopted cells are marked `adopted: true` in their state file so the "
          "provenance stays visible.")
    return 0


# --------------------------------------------------------------------------- #
# Synthetic self-test of the resume logic (no training)
# --------------------------------------------------------------------------- #
def self_test() -> int:
    """
    Exercises the state machine end to end against a temporary directory with
    fabricated artifacts. Runs no training and touches no campaign data.
    """
    print("=" * 78)
    print("SELF-TEST OF THE CHECKPOINT / RESUME LOGIC (synthetic, no training)")
    print("=" * 78)
    tmp = Path(tempfile.mkdtemp(prefix="ablation_selftest_"))
    original_raw = C.RAW
    passed = []
    try:
        C.RAW = tmp
        camp = campaign_fingerprint("smoke")
        mode = "smoke"
        cells = all_cells(mode)
        raw = run_dir(mode)
        raw.mkdir(parents=True, exist_ok=True)

        def fake_complete(v, a, s, camp_fp=None, rows=None, seed_override=None):
            """Fabricate the artifacts a finished cell would leave behind."""
            fp = {"variant": v, "alpha": float(a),
                  "seed": int(seed_override if seed_override is not None else s),
                  "dataset": C.DATASET, "num_rounds": C.NUM_ROUNDS,
                  "batch_size": 32, "n_clients": C.CLIENT_SETUP}
            fingerprint_path(raw, v, a, s).write_text(json.dumps(fp), encoding="utf-8")
            n = rows if rows is not None else C.NUM_ROUNDS * C.CLIENT_SETUP
            with telemetry_path(raw, v, a, s).open("w", encoding="utf-8") as fh:
                fh.write("h1,h2\n")
                for i in range(n):
                    fh.write(f"{i},{i}\n")
            write_state_atomic(state_path(mode, v, a, s), {
                "cell_id": cell_id(v, a, s), "status": STATUS_COMPLETED,
                "campaign_fingerprint": camp_fp or camp["fingerprint"],
                "duration_s": 1.0, "exit_status": 0, "error_message": ""})

        def check(name, cond):
            print(f"  [{'OK  ' if cond else 'FAIL'}] {name}")
            passed.append(bool(cond))

        counts, _ = survey(mode, camp)
        check("a fresh campaign reports every cell pending",
              counts[STATUS_PENDING] == len(cells) and counts[STATUS_COMPLETED] == 0)

        # interruption after 3 cells
        for v, a, s in cells[:3]:
            fake_complete(v, a, s)
        counts, _ = survey(mode, camp)
        check("after 3 cells complete, exactly those 3 are skipped on resume",
              counts[STATUS_COMPLETED] == 3 and
              counts[STATUS_PENDING] == len(cells) - 3)

        # a cell left `running` with no artifacts must be re-executed
        v, a, s = cells[3]
        write_state_atomic(state_path(mode, v, a, s), {
            "cell_id": cell_id(v, a, s), "status": STATUS_RUNNING,
            "campaign_fingerprint": camp["fingerprint"]})
        st, why = cell_status(mode, v, a, s, camp)
        check("a cell killed while `running` is incomplete, never counted as done",
              st == STATUS_PENDING and "interrupted" in why)

        # missing telemetry invalidates a supposedly completed cell
        v, a, s = cells[0]
        telemetry_path(raw, v, a, s).unlink()
        st, why = cell_status(mode, v, a, s, camp)
        check("a completed cell with missing telemetry is re-executed",
              st == STATUS_PENDING and "telemetry missing" in why)
        fake_complete(v, a, s)

        # truncated telemetry is rejected
        v, a, s = cells[1]
        fake_complete(v, a, s, rows=3)
        st, why = cell_status(mode, v, a, s, camp)
        check("truncated telemetry is rejected",
              st == STATUS_PENDING and "rows" in why)
        fake_complete(v, a, s)

        # a result from a different campaign configuration is never reused
        v, a, s = cells[2]
        fake_complete(v, a, s, camp_fp="a-different-campaign")
        st, why = cell_status(mode, v, a, s, camp)
        check("a cell from a different campaign fingerprint is not reused",
              st == STATUS_PENDING and "different campaign" in why)
        fake_complete(v, a, s)

        # a fingerprint whose seed does not match is rejected
        v, a, s = cells[4]
        fake_complete(v, a, s, seed_override=int(s) + 999)
        st, why = cell_status(mode, v, a, s, camp)
        check("a fingerprint whose seed does not match is rejected",
              st == STATUS_PENDING and "seed mismatch" in why)

        # a failed cell is reported as failed (so --rerun-failed can pick it up)
        v, a, s = cells[5]
        write_state_atomic(state_path(mode, v, a, s), {
            "cell_id": cell_id(v, a, s), "status": STATUS_FAILED,
            "campaign_fingerprint": camp["fingerprint"],
            "error_message": "synthetic failure"})
        st, _ = cell_status(mode, v, a, s, camp)
        check("a failed cell is reported as failed", st == STATUS_FAILED)

        # changing the code invalidates completed cells
        other = dict(camp)
        other["fingerprint"] = "changed-code-fingerprint"
        st, _ = cell_status(mode, cells[0][0], cells[0][1], cells[0][2], other)
        check("changing the campaign fingerprint invalidates completed cells",
              st == STATUS_PENDING)

        # artifacts with no state entry are NOT silently reused
        v, a, s = cells[6]
        fp = {"variant": v, "alpha": float(a), "seed": int(s),
              "dataset": C.DATASET, "num_rounds": C.NUM_ROUNDS,
              "batch_size": 32, "n_clients": C.CLIENT_SETUP}
        fingerprint_path(raw, v, a, s).write_text(json.dumps(fp), encoding="utf-8")
        with telemetry_path(raw, v, a, s).open("w", encoding="utf-8") as fh:
            fh.write("h1,h2\n")
            for i in range(C.NUM_ROUNDS * C.CLIENT_SETUP):
                fh.write(f"{i},{i}\n")
        st, why = cell_status(mode, v, a, s, camp)
        check("artifacts without a state entry are not silently reused",
              st == STATUS_PENDING and "orphan" in why)

        # ...but --adopt-orphans takes them after validation
        adopt_orphans(mode, camp)
        st, _ = cell_status(mode, v, a, s, camp)
        check("--adopt-orphans adopts them once validated", st == STATUS_COMPLETED)

        # atomic writes leave nothing partial behind
        check("atomic writes leave no .tmp files behind",
              not list(state_dir(mode).glob("*.tmp")))
    finally:
        C.RAW = original_raw
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    return 0 if all(passed) else 1


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="FedHAD controlled ablation runner (checkpoint/resume).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="With no flag at all this resumes the official battery, which is "
               "the same as --official / --resume. Re-running any of them skips "
               "completed, valid cells automatically.")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--official", action="store_true",
                   help="start or resume the full 30-seed battery (300 cells); "
                        "this is also what running with NO flag does")
    g.add_argument("--resume", action="store_true",
                   help="alias for --official; resumes an interrupted battery")
    g.add_argument("--smoke", action="store_true", help="2-seed validation grid")
    g.add_argument("--status", action="store_true",
                   help="report progress and exit without running anything")
    g.add_argument("--rerun-failed", action="store_true",
                   help="re-execute only the cells recorded as failed")
    g.add_argument("--verify-only", action="store_true",
                   help="re-check fairness invariants on existing artifacts")
    g.add_argument("--self-test", action="store_true",
                   help="synthetic check of the resume logic; runs no training")
    g.add_argument("--adopt-orphans", action="store_true",
                   help="validate artifacts that have no state entry and adopt them "
                        "as completed; use only when you know they came from this "
                        "exact code and configuration")
    ap.add_argument("--mode", choices=["smoke", "official"], default=None,
                    help="grid inspected by --status / --verify-only / --rerun-failed")
    ap.add_argument("--dry-run", action="store_true", help="plan only, execute nothing")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.smoke:
        mode = "smoke"
    elif args.official or args.resume:
        mode = "official"
    else:
        # No action flag: resume the official battery. Resuming is idempotent -
        # it only ever runs cells that are not already complete and valid.
        mode = args.mode or "official"
        if not any([args.status, args.rerun_failed, args.verify_only]):
            args.official = True
    seeds = C.SEEDS_SMOKE if mode == "smoke" else C.SEEDS_OFFICIAL
    camp = campaign_fingerprint(mode)
    logs_dir = C.LOGS / mode
    state_dir(mode).mkdir(parents=True, exist_ok=True)
    write_state_atomic(state_dir(mode) / "campaign.json", camp)

    if args.adopt_orphans:
        return adopt_orphans(mode, camp)
    if args.status:
        print_status(mode, camp)
        return 0
    if args.verify_only:
        _, failures = verify(mode, seeds)
        return 1 if failures else 0

    counts, detail = survey(mode, camp)
    if args.rerun_failed:
        todo = [(v, a, s) for v, a, s, st, _ in detail if st == STATUS_FAILED]
        header = f"RE-RUNNING FAILED CELLS - {mode.upper()}"
    else:
        todo = [(v, a, s) for v, a, s, st, _ in detail
                if st in (STATUS_PENDING, STATUS_FAILED)]
        header = f"FEDHAD CONTROLLED ABLATION - {mode.upper()}"

    print("=" * 78)
    print(header)
    print("=" * 78)
    print(f"Entry script  : {C.ENTRY_SCRIPT}")
    print(f"Interpreter   : {python_executable()}")
    print(f"Campaign hash : {camp['fingerprint'][:16]}")
    print(f"Grid          : {C.DATASET} | alphas {C.ALPHAS} | {C.CLIENT_SETUP} clients "
          f"| {C.NUM_ROUNDS} rounds | {len(seeds)} seeds | {len(C.VARIANTS)} arms")
    print(f"Cells total   : {len(all_cells(mode))}")
    print(f"  completed   : {counts[STATUS_COMPLETED]} (skipped)")
    print(f"  to run now  : {len(todo)}")
    if mode == "smoke":
        print("NOTE: smoke output lives in raw/smoke and is never mixed with official.")
    if args.dry_run:
        for i, (v, a, s) in enumerate(todo, 1):
            print(f"  [{i}/{len(todo)}] {cell_id(v, a, s)}")
        return 0
    if not todo:
        print("\nNothing to do: every cell is complete and valid.")
        verify(mode, seeds)
        return 0

    started = datetime.now()
    failed = []
    for i, (v, a, s) in enumerate(todo, 1):
        ok, secs, err = execute(mode, v, a, s, camp, logs_dir)
        print(f"  [{i}/{len(todo)}] {cell_id(v, a, s)}  {secs:6.0f}s  "
              f"{'ok' if ok else 'FAILED - ' + err}", flush=True)
        if not ok:
            failed.append(cell_id(v, a, s))
    print(f"\nFinished in {(datetime.now() - started).total_seconds() / 60:.1f} min")

    counts, _ = survey(mode, camp)
    print(f"Campaign now: {counts[STATUS_COMPLETED]}/{len(all_cells(mode))} complete, "
          f"{counts[STATUS_FAILED]} failed, {counts[STATUS_PENDING]} pending")
    if failed:
        print(f"\n{len(failed)} cell(s) failed. Retry with:  "
              f"python runners/run_component_ablation.py --rerun-failed --mode {mode}")

    fps, failures = verify(mode, seeds)
    res = write_budget_table(fps, mode)
    if res:
        out, rows = res
        print(f"Budget verification table -> {out}")
        pct = [abs(r["updates_pct_vs_full"]) for r in rows]
        print(f"  |updates - full| : max {max(pct):.2f}%, mean {sum(pct) / len(pct):.2f}%")

    if failures:
        print("\nFairness verification FAILED; do not analyse until this is explained.")
        return 1
    if mode == "smoke" and counts[STATUS_PENDING] == 0 and not failed:
        if not check_reproducibility(mode, camp, logs_dir):
            return 1
        print("\nSmoke test passed. Start the battery with:  "
              "python runners/run_component_ablation.py --official")
    elif counts[STATUS_COMPLETED] == len(all_cells(mode)):
        print("\nAll cells complete. Analyse with:  "
              f"python analyze_ablation.py --mode {mode}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
