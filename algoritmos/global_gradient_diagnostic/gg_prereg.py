# -*- coding: utf-8 -*-
"""
Pre-registration of the global-gradient diagnostic battery. Generated from config_gg.py
and margins.json, written once before the first official run and hashed; the runner
refuses to resume and the analysis refuses to run if it changes afterwards.

The hypotheses are directional and state, before any data, what would support and
what would contradict the premise of FedHAD.
"""
from __future__ import annotations

import hashlib
import json


def canonical_json(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prereg(*, C, margins, code_fingerprint):
    return {
        "title": "Global-gradient and class-sensitive diagnostic battery for FedHAD",
        "questions": {
            "Q1_premise": ("Is a client's label skew H_k associated with lower alignment between its "
                           "gradient (and its update) and the gradient of the REST of the federation, "
                           "when client size cannot explain the association?"),
            "Q2_removed_epochs": ("Are the local epochs that FedHAD removes from a client less aligned "
                                  "with the descent direction of the rest of the federation than the "
                                  "epochs it keeps, and is the decline over epochs steeper for more "
                                  "skewed clients?"),
            "Q3_allocation": ("With equal client sizes, does assigning more epochs to less skewed clients "
                              "(FedHAD) give a different final accuracy from the inverse assignment at "
                              "an identical number of updates?"),
            "Q4_classes": ("Does FedHAD sacrifice the worst class or macro-F1 relative to FedProx at the "
                           "default and at the tuned configuration?"),
        },
        "substrate": {
            "dataset": C.DATASET, "seeds": C.SEEDS, "clients": C.CLIENT_SETUP, "rounds": C.NUM_ROUNDS,
            "batch_size": C.BATCH_SIZE, "optimizer": "SGD, momentum 0.9", "proximal_mu": C.FEDPROX_MU,
            "fedhad": {"base_epochs": C.BASE_EPOCHS, "min_epochs": C.MIN_EPOCHS, "epochs_decay": C.EPOCHS_DECAY,
                       "base_lr": C.BASE_LR, "min_lr": C.MIN_LR, "lr_decay": C.LR_DECAY},
            "initialization": "common checkpoint per seed, SHA-256 verified (results_reviewer_r2_checkpoints)",
            "stochastic_trajectory": "worker seeding per (seed, client, round), identical across arms",
            "levels": {str(k): v for k, v in C.LEVELS.items()},
            "equal_partition": (f"every client holds exactly {C.EQUAL_N} samples; label proportions "
                                "~ Dir(alpha 1_C) sampled without replacement from the class pools, "
                                "renormalised over non-exhausted classes; np.random.default_rng(seed)"),
        },
        "measurements": {
            "g_k": "mean cross-entropy gradient over client k's training split at w^t, eval mode",
            "g": "sum_k p_k g_k (global training objective), p_k = n_k / N",
            "g_minus_k": "(N g - n_k g_k) / (N - n_k): global gradient without client k",
            "grad_cos_loo": "cos(g_k, g_-k)",
            "upd_cos_loo": "cos(u_k, -g_-k), u_k = w_k^{t+1} - w^t (trainable parameters)",
            "marg_cos_loo_e": "cos(w_{k,e} - w_{k,e-1}, -g_-k) for local epoch e",
            "class": "per-class recall, worst-class recall and macro-F1 of the global model on the test set",
        },
        "unit_of_analysis": ("the seed: every client-round statistic is first summarised within a seed, "
                             "so within-run dependence is preserved; 30 values per (level, alpha)"),
        "hypotheses_and_tests": {
            "H1_primary": ("Level 2 (equal sizes), arm full_fedhad, each alpha: rho_s = Spearman(H_k, "
                           "grad_cos_loo) over the client-rounds of seed s. Supports the premise if the "
                           "median of rho_s is below 0: two-sided Wilcoxon signed-rank on the 30 values, "
                           "Holm over the two alphas; mean with 95% bootstrap CI reported."),
            "H1_update": "As H1_primary with upd_cos_loo (Holm within the same family of two).",
            "H1_paper_partitions": ("Level 1, each alpha: (i) rho_s as above, per seed; (ii) size-adjusted: "
                                    "pooled OLS over all client-rounds of all seeds, grad_cos_loo ~ H_k + "
                                    "log n_k + round fixed effects, coefficient of H_k with a 95% cluster "
                                    "bootstrap CI resampling seeds (2,000 resamples) and a two-sided "
                                    "bootstrap p. Within a seed H_k and n_k are often perfectly rank-"
                                    "collinear, so the adjustment uses the variation across seeds. "
                                    "Secondary; Holm over the four tests."),
            "H2_removed_epochs": ("Level 3, arm fedprox_default (every client trains 5 epochs), each alpha: "
                                  "for each client-round with E_full(H_k) < 5, d = mean marg_cos_loo over the "
                                  "removed epochs E_full+1..5 minus mean over the kept epochs 1..E_full; d_s "
                                  "= mean over the client-rounds of seed s. Supports the premise if median "
                                  "d_s < 0 (Wilcoxon, Holm over the two alphas). The fraction of removed "
                                  "epochs with marg_lin_loo > 0 (first-order increase of the others' loss) "
                                  "is reported descriptively."),
            "H2_decline_by_skew": ("Level 3, fedprox_default: per client-round, OLS slope of marg_cos_loo "
                                   "over epochs 1..5; rho_s = Spearman(H_k, slope) within seed. Supports the "
                                   "premise if median rho_s < 0 (steeper decline for skewed clients). "
                                   "Secondary; Holm with H2 over four tests."),
            "H3_allocation": ("Level 2, each alpha: paired difference in final accuracy, full_fedhad - "
                              "inv_epochs, over seeds; four-category rule of the R2 batteries with the "
                              "margin delta of margins.json; Holm over the two alphas. Primary analysis on all 30 "
                              "seeds; sensitivity analysis on the seeds in which the two allocations "
                              "differ (in the others the arms execute the same plan)."),
            "H4_classes": ("Levels 1 and 3, each alpha: full_fedhad - fedprox_default and full_fedhad - "
                           "fedprox_tuned for worst-class recall and for macro-F1 (Holm over the eight "
                           "tests); Level 2: full_fedhad - inv_epochs for the same two metrics (Holm over "
                           "four). Categories 1, 3 or 'not detected'; no equivalence claim (no margin)."),
        },
        "what_would_contradict": {
            "H1": "median rho_s >= 0 or not detected on the equal partition",
            "H2": "removed epochs not less aligned than kept ones",
            "H3": "inverse epochs not less accurate (categories 2, 3 or 4)",
        },
        "decision_rule_accuracy": {
            "1": "Holm p < 0.05 and the whole 95% bootstrap CI above 0",
            "3": "Holm p < 0.05 and the whole 95% bootstrap CI below 0",
            "2": "otherwise, the whole 90% CI inside (-delta, +delta)",
            "4": "otherwise: inconclusive",
        },
        "equivalence_margins_pp": {str(a): d for a, d in margins.items()},
        "all_results_reported": "every hypothesis is reported whatever its outcome",
        "code_fingerprint": code_fingerprint,
    }
