# -*- coding: utf-8 -*-
"""
Fixed configuration of the global-gradient diagnostic battery (Section 6.11 of the
manuscript): a limited offline comparison with the gradient of the global training objective,
together with class-sensitive accuracy measures. Same substrate as the other revision control batteries;
nothing is tuned for FedHAD. Only margins.json is meant to be edited, and only before
the first official run.

Levels (each a separate question; see the pre-registration for the hypotheses):
  1  premise on the paper's partitions   dirichlet, alpha {0.1, 0.01}, FedHAD
  2  equal-size label skew               equal, alpha {0.1, 0.2}, FedHAD and inverse epochs
  3  baselines on the paper's partitions dirichlet, alpha {0.1, 0.01}, FedProx default and tuned
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CODE = HERE
ENTRY_SCRIPT = CODE / "fedhad_gg.py"
MARGINS_FILE = HERE / "margins.json"

OUT = REPO / "results" / "global_gradient_diagnostic"
OUT_SMOKE = OUT / "smoke"
CKPT_DIR = REPO / "results" / "initial_checkpoints_common"

# ---- substrate (identical to the other revision control batteries) ------------------------------ #
DATASET = "CIFAR10"
SEEDS = list(range(42, 72))
CLIENT_SETUP = 5
NUM_ROUNDS = 10
BATCH_SIZE = 32
COMM_DELAY = 0.05
USE_ENERGY = False
FEDPROX_MU = 0.01
BASE_EPOCHS, MIN_EPOCHS, EPOCHS_DECAY = 5, 2, 3.0
BASE_LR, MIN_LR, LR_DECAY = 0.01, 0.001, 1.5
EQUAL_N = 5000

LEVELS = {
    1: {"name": "premise_paper_partitions", "partition": "dirichlet", "alphas": [0.1, 0.01],
        "arms": ["full_fedhad"]},
    2: {"name": "equal_size_label_skew", "partition": "equal", "alphas": [0.1, 0.2],
        "arms": ["full_fedhad", "inv_epochs"]},
    3: {"name": "baselines_paper_partitions", "partition": "dirichlet", "alphas": [0.1, 0.01],
        "arms": ["fedprox_default", "fedprox_tuned"]},
}

SMOKE_SEEDS = [42]
SMOKE_ROUNDS = 2

# Rough per-run durations for the plan printout (A40, revision control batteries; equal partition
# trains on half the data); the gradient pass adds a few seconds per round.
EST_SECONDS = {("dirichlet", "full_fedhad"): 230, ("dirichlet", "fedprox_default"): 290,
               ("dirichlet", "fedprox_tuned"): 180, ("equal", "full_fedhad"): 130,
               ("equal", "inv_epochs"): 130}


def load_margins():
    d = json.loads(MARGINS_FILE.read_text(encoding="utf-8"))["delta_pp"]
    out = {float(k): float(v) for k, v in d.items()}
    missing = [a for a in LEVELS[2]["alphas"] if a not in out]
    if missing:
        raise ValueError(f"margins.json lacks delta for alpha {missing}")
    return out
