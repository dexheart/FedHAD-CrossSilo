# -*- coding: utf-8 -*-
"""
Explicit configuration of the revision control experiments (step-preserving
permutation and tuned FedProx controls).
Every protocol decision lives here so it is visible in review and hashed into the
campaign fingerprint. Nothing here is read by the historical campaigns.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
ENTRY_SCRIPT = CODE / "fedhad_r2.py"

# Output roots (new, never shared with historical result trees)
OUT_STEP = REPO / "results" / "step_preserving_permutation"
OUT_TUNED = REPO / "results" / "tuned_controls_fedprox_validation"
OUT_CKPT = REPO / "results" / "initial_checkpoints_common"

# --------------------------------------------------------------------------- #
# Common substrate (identical to the controlled component analysis)
# --------------------------------------------------------------------------- #
DATASET = "CIFAR10"
ALPHAS = [0.01, 0.1]
CLIENT_SETUP = 5
NUM_ROUNDS = 10
COMM_DELAY = 0.05
USE_ENERGY = False
BATCH_SIZE = 32          # fixed in the training script; recorded for the manifest
FEDPROX_MU = 0.01        # shared by FedHAD arms and the static FedProx control

# FedHAD hyper-parameters (defaults of the paper; not tuned here)
BASE_EPOCHS, MIN_EPOCHS, EPOCHS_DECAY = 5, 2, 3.0
BASE_LR, MIN_LR, LR_DECAY = 0.01, 0.001, 1.5

EVAL_SEEDS = list(range(42, 72))          # the 30 evaluation seeds of the paper

# --------------------------------------------------------------------------- #
# Experiment 1: step-preserving permuted allocation
# --------------------------------------------------------------------------- #
STEP_ARMS_DEFAULT = ["full_fedhad", "step_permuted_lrclient"]
STEP_ARM_OPTIONAL = "step_permuted_lrfollow"   # enabled with --include-lr-follow

# --------------------------------------------------------------------------- #
# Experiment 2: tuned static controls
# --------------------------------------------------------------------------- #
# Validation (tuning) seeds: a block disjoint from 42-71. 72-76 are the next five
# integers; they were not used by any historical campaign of this repository
# (main campaigns and the component analysis used 42-71; the drift diagnostics
# used 42-51). Configurable with --val-seeds; overlap with the evaluation seeds is
# rejected by r2_policy.check_seed_separation.
VAL_SEEDS = [72, 73, 74, 75, 76]

# Grid justified by the learning rates and epochs FedHAD itself produces at
# alpha in {0.1, 0.01} (analysis/audit/a04_lr_survey.py): per-client LR 0.0040-0.0072
# (sample-weighted mean 0.0055-0.0058), epochs 2-4 (weighted mean 3.2-3.6). The
# grid spans that range and includes the untuned baseline point (0.01, 5 epochs).
# No accuracy of the 30 evaluation seeds was used to choose it.
LR_GRID = [0.01, 0.0075, 0.005, 0.004]
EPOCHS_GRID = [5, 4, 3, 2]
STATIC_METHODS_DEFAULT = ["FedProx"]       # FedAvg available via --methods
STATIC_METHODS_ALLOWED = ["FedProx", "FedAvg"]

# Default (untuned) baseline point, re-run under the new protocol in the evaluate
# phase so tuned vs untuned vs FedHAD are paired on checkpoint and seeding.
DEFAULT_BASELINE = {"lr": 0.01, "epochs": 5}

# Selection rules (reported by --summarize-tune; the final choice must be frozen
# explicitly with --freeze or by writing frozen_config.json by hand):
#   budget_matched : among static configs whose mean executed optimizer steps on the
#                    validation seeds are <= (1 + BUDGET_TOL) x FedHAD's mean executed
#                    steps on the same seeds, the highest mean VALIDATION accuracy
#                    (ties -> fewer steps).
#   operating_points: the non-dominated set in (validation accuracy up, executed
#                    steps down); reported, never auto-selected.
# Validation accuracy = sample-weighted accuracy of the clients' local validation
# splits (Flower distributed evaluation) at the last round. The centralized test
# accuracy is recorded but NOT used for selection (the CIFAR-10 test set is the same
# for every seed, so selecting on it would leak).
BUDGET_TOL = 0.02
SELECTION_METRIC = "acc_distributed_val"

# Rough duration for dry-run estimates: mean cell duration of the component
# analysis on the A40 server (component_analysis_ablation/README.md, 182 s).
EST_SECONDS_PER_RUN = 182
