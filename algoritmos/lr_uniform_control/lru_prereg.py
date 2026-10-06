# -*- coding: utf-8 -*-
"""
Pre-registration of the LR-uniform control battery. The document is generated from
config_lru.py + margins.json, written once before the first official training run,
and hashed. The runner refuses to resume, and the analysis refuses to run, if the
document, the margins or the reference-source choice change afterwards.
"""
from __future__ import annotations

import hashlib
import json

from lru_stats import BOOT_RESAMPLES, BOOT_SEED, CATEGORY


def canonical_json(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prereg(*, C, margins, reference_source, code_fingerprint):
    """C is the config_lru module; margins {alpha: delta_pp}."""
    return {
        "title": "LR-uniform control battery for FedHAD (Section 6.10 substrate)",
        "question": ("Does assigning the local learning rate according to the heterogeneity "
                     "score H_k produce a different final accuracy from giving every client one "
                     "uniform learning rate with the same aggregate first-order aggressiveness?"),
        "hypotheses": {
            "outcome_1": "heterogeneity-based assignment superior (Full FedHAD > U2)",
            "outcome_2": "practically equivalent (|Full - U2| within +/- delta)",
            "outcome_3": "uniform learning rate superior (U2 > Full FedHAD)",
            "outcome_4": "inconclusive (none of the above by the rule below)",
        },
        "substrate": {
            "dataset": C.DATASET, "alphas": C.ALPHAS, "seeds": C.SEEDS, "clients": C.CLIENT_SETUP,
            "rounds": C.NUM_ROUNDS, "participation": "full", "batch_size": C.BATCH_SIZE,
            "optimizer": "SGD, momentum 0.9", "proximal_mu": C.FEDPROX_MU,
            "epoch_ceiling": C.BASE_EPOCHS, "comm_delay_s": C.COMM_DELAY,
            "fedhad": {"base_epochs": C.BASE_EPOCHS, "min_epochs": C.MIN_EPOCHS,
                       "epochs_decay": C.EPOCHS_DECAY, "base_lr": C.BASE_LR,
                       "min_lr": C.MIN_LR, "lr_decay": C.LR_DECAY},
            "initialization": "common checkpoint per seed, SHA-256 verified (results_reviewer_r2_checkpoints)",
            "stochastic_trajectory": "worker seeding per (seed, client, round), identical across arms",
        },
        "notation": {
            "tau_k": "E_k * floor(n_k_train / 32), optimizer steps per round",
            "p_k": "n_k_train / N, aggregation weight",
        },
        "arms": {
            "full_fedhad": "E_full(H_k), eta_k = max(eta_min, eta_base / (1 + lambda_eta H_k))",
            "lr_only_matched": "E_fixed of Section 6.10 (ablation_policy.build_allocation), eta_k of FedHAD",
            "u2": "E_full(H_k), uniform eta_bar computed with tau_k from E_full",
            "u1": "E_fixed, uniform eta_bar computed with tau_k from E_fixed",
            "u2_arith": "E_full(H_k), uniform eta = arithmetic mean of eta_k over clients with data",
            "inv": ("E_full(H_k), FedHAD multiset of eta_k re-assigned in inverse order of H_k among "
                    "clients with data (largest H_k -> largest eta); ties by client index; "
                    "clients without data keep their own rate"),
        },
        "eta_bar_definitions": {
            "primary": "sum_k p_k eta_k tau_k / sum_k p_k tau_k (clients with data)",
            "tau_weighted": "sum_k eta_k tau_k / sum_k tau_k (recorded, sensitivity)",
            "arithmetic": "mean_k eta_k over clients with data (arm u2_arith)",
            "eta_k_source": "the rates FedHAD produces on that partition",
        },
        "empty_clients": ("clients without samples do not enter eta_bar and do not train, as in "
                          "Section 6.10; primary analysis includes all 30 seeds; sensitivity "
                          "analysis excludes seeds with an empty client"),
        "outcome": "final centralized test accuracy after the last round, in percentage points",
        "contrasts": {"primary": "Full FedHAD - U2",
                      "secondary": ["LR-only matched - U1", "Full FedHAD - U2-arith",
                                    "Full FedHAD - INV"]},
        "test": "paired two-sided Wilcoxon signed-rank over seeds",
        "holm_families": {
            "primary": "Full - U2 alone, separately at each alpha (Holm-adjusted p equals raw p)",
            "secondary": "the three secondary contrasts, one family at each alpha",
        },
        "bootstrap": {"type": "percentile over seeds", "resamples": BOOT_RESAMPLES, "seed": BOOT_SEED,
                      "levels": [0.95, 0.90]},
        "equivalence_margins_pp": {str(a): d for a, d in sorted(margins.items())},
        "classification_rule": {
            "applies_to": "each contrast, separately at each alpha, mechanically",
            "1": "Holm p < 0.05 and the whole 95% bootstrap CI above 0",
            "3": "Holm p < 0.05 and the whole 95% bootstrap CI below 0",
            "2": ("neither 1 nor 3, and the whole 90% bootstrap CI inside (-delta, +delta) "
                  "(TOST at 5%); the two one-sided Wilcoxon p-values against -delta and +delta "
                  "are reported"),
            "4": ("any other case: inconclusive; report the 90% CI half-width and the number of "
                  "seeds that would fit it in the margin by simple extrapolation from the SD "
                  "of the paired differences"),
            "categories": CATEGORY,
            "alphas_not_combined": True,
            "secondary_role": "supporting evidence only, never a substitute for the primary",
        },
        "reference_source": reference_source,
        "code_fingerprint": code_fingerprint,
    }
