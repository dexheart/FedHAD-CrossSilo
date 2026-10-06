# -*- coding: utf-8 -*-
"""
Explicit configuration for the FedHAD controlled-ablation campaign.

Everything that defines the campaign lives here, so a reader never has to infer
the design from the runner (runners/run_component_ablation.py). Results go to
results/component_analysis_ablation/. Nothing else is read or
written by this campaign, except the three read-only source files that were
copied into this folder (see ORIGINAL_HASHES.txt).
"""
from pathlib import Path

# Code (this folder) and results (results/component_analysis_ablation/) are kept apart.
CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]
ROOT = REPO / "results" / "component_analysis_ablation"
RAW = ROOT / "raw"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"

ENTRY_SCRIPT = CODE / "fedhad_ablation.py"

# ---------------------------------------------------------------------------- #
# The five arms
# ---------------------------------------------------------------------------- #
VARIANTS = [
    "full_fedhad",       # E=E_full(H), LR=LR_full(H)   - FedHAD as it stands
    "fixed_matched",     # E=E_fixed,   LR=BASE_LR      - no H anywhere, same total epochs
    "permuted_allocation",  # E=perm(E_full), LR=LR_full(H)- same epoch multiset, re-assigned
    "epoch_only",        # E=E_full(H), LR=BASE_LR      - adaptive epochs alone
    "lr_only_matched",   # E=E_fixed,   LR=LR_full(H)   - adaptive LR alone
]

# ---------------------------------------------------------------------------- #
# Experimental grid
# ---------------------------------------------------------------------------- #
DATASET = "CIFAR10"
ALPHAS = [0.01, 0.1]          # the two most heterogeneous settings of the campaign
CLIENT_SETUP = 5              # 5 clients, matching the main campaign
NUM_ROUNDS = 10
COMM_DELAY = 0.05
USE_ENERGY = False            # energy tracking off: it does not affect the contrasts

SEEDS_OFFICIAL = list(range(42, 72))   # the same 30 seeds as the main campaign
SEEDS_SMOKE = [42, 43]                 # smoke test only - never mixed with official runs

# Repeat one seed twice to verify run-to-run reproducibility under the fixed
# initialisation. Both repeats must agree exactly on the deterministic telemetry.
REPRODUCIBILITY_SEED = 42
REPRODUCIBILITY_VARIANT = "full_fedhad"

# ---------------------------------------------------------------------------- #
# FedHAD hyper-parameters. Left at the campaign's defaults; listed explicitly so
# the ablation is self-describing and so a change here is visible in review.
# ---------------------------------------------------------------------------- #
BASE_EPOCHS = 5
MIN_EPOCHS = 2
EPOCHS_DECAY = 3.0
BASE_LR = 0.01
MIN_LR = 0.001
LR_DECAY = 1.5
FEDPROX_MU = 0.01

# ---------------------------------------------------------------------------- #
# Fairness invariants. Any violation aborts the campaign rather than being
# reported as a caveat: an unpaired comparison is not worth running.
# ---------------------------------------------------------------------------- #
# Fields that MUST be byte-identical across all five arms of a (alpha, seed) cell.
IDENTICAL_ACROSS_VARIANTS = [
    "initial_weights_sha256",   # same starting model
    "partition_hash_all",       # same client partitions
    "H_hash",                   # same heterogeneity scores
    "n_samples",                # same clients, same sizes
    "n_batches",
    "num_rounds",
    "batch_size",
    "n_clients",
]

# Per-variant structural invariants, checked against the fingerprint.
VARIANT_INVARIANTS = {
    "full_fedhad": "epochs == E_full and lr == LR_full",
    "fixed_matched": "epochs == E_fixed and lr == BASE_LR for every client",
    "permuted_allocation": "multiset(epochs) == multiset(E_full) and lr == LR_full",
    "epoch_only": "epochs == E_full and lr == BASE_LR for every client",
    "lr_only_matched": "epochs == E_fixed and lr == LR_full",
}

# The MATCHED quantity is the total number of minibatch updates. Exact equality
# is not always reachable (epochs are integers), so each arm's residual against
# full_fedhad is recorded and reported instead of being asserted to zero.
# epoch_only is exact by construction.
BUDGET_MATCHED_FIELD = "total_updates"
BUDGET_REPORTED_FIELDS = ["total_epochs", "total_updates", "total_flops",
                          "total_flops_realised"]
# Arms whose update total must equal full_fedhad's exactly.
BUDGET_EXACT_ARMS = ["full_fedhad", "epoch_only"]

# Okabe-Ito colourblind-safe palette, one colour per variant.
VARIANT_COLOR = {
    "full_fedhad": "#0072B2",
    "fixed_matched": "#999999",
    "permuted_allocation": "#D55E00",
    "epoch_only": "#009E73",
    "lr_only_matched": "#CC79A7",
}
VARIANT_LABEL = {
    "full_fedhad": "full FedHAD",
    "fixed_matched": "fixed matched",
    "permuted_allocation": "permuted matched",
    "epoch_only": "epochs only",
    "lr_only_matched": "LR only (matched)",
}
