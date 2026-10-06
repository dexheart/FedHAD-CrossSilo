# -*- coding: utf-8 -*-
"""
Fixed configuration of the LR-uniform control battery. Identical to the controlled
component analysis of Section 6.10 (algoritmos/component_analysis_ablation/config_ablation.py);
nothing is tuned and nothing favours any arm. Only margins.json is meant to be edited,
and only before the first official run.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CODE = HERE
ENTRY_SCRIPT = CODE / "fedhad_lru.py"
MARGINS_FILE = HERE / "margins.json"

OUT = REPO / "results" / "lr_uniform_control"          # new, never shared
OUT_SMOKE = OUT / "smoke"                               # smoke runs, never analysed
CKPT_DIR = REPO / "results" / "initial_checkpoints_common"    # common initial checkpoints

# ---- substrate (Section 6.10) ---------------------------------------------- #
DATASET = "CIFAR10"
ALPHAS = [0.1, 0.01]
SEEDS = list(range(42, 72))
CLIENT_SETUP = 5
NUM_ROUNDS = 10
BATCH_SIZE = 32                 # fixed in the training script
COMM_DELAY = 0.05
USE_ENERGY = False
FEDPROX_MU = 0.01
BASE_EPOCHS, MIN_EPOCHS, EPOCHS_DECAY = 5, 2, 3.0
BASE_LR, MIN_LR, LR_DECAY = 0.01, 0.001, 1.5

# ---- smoke ----------------------------------------------------------------- #
SMOKE_SEEDS = [42]
SMOKE_ROUNDS = 2

# ---- arms and execution order ---------------------------------------------- #
LEVEL1_ARMS = ["full_fedhad", "u2", "lr_only_matched", "u1"]   # answer the main question
LEVEL2_ARMS = ["u2_arith", "inv"]
REFERENCE_ARMS = ["full_fedhad", "lr_only_matched"]            # re-executed by default

# ---- contrasts (A - B, percentage points of final centralized accuracy) ---- #
PRIMARY = ("full_fedhad", "u2")
SECONDARY = [("lr_only_matched", "u1"), ("full_fedhad", "u2_arith"), ("full_fedhad", "inv")]

# Rough estimate for the plan printout: mean duration of the revision control runs executed with
# the same code base on the server (205.6 s over 120 runs).
EST_SECONDS_PER_RUN = 206


def load_margins():
    d = json.loads(MARGINS_FILE.read_text(encoding="utf-8"))["delta_pp"]
    out = {float(k): float(v) for k, v in d.items()}
    missing = [a for a in ALPHAS if a not in out]
    if missing:
        raise ValueError(f"margins.json lacks delta for alpha {missing}")
    return out
