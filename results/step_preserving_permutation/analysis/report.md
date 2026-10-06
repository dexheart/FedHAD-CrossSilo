# Step-preserving permutation and tuning phase — consolidated results

Substrate: CIFAR-10, 5 clients, 10 rounds, FedProx mu=0.01 inside FedHAD arms, common initial checkpoint per seed (SHA-256 recorded), seeds 42-71.

## Experiment 1 — step-preserving permuted allocation

`step_permuted_lrclient`: the executed optimizer-step budgets tau_k of full FedHAD are permuted among clients (derangement); each client KEEPS its own FedHAD learning rate.

| alpha | arm | n | acc (%) | sd | steps | examples | TFLOPs exec | TFLOPs nominal |
|---|---|---|---|---|---|---|---|---|
| 0.01 | full_fedhad | 30 | 55.44 | 4.17 | 45332 | 1450572 | 262.4 | 262.8 |
| 0.01 | step_permuted_lrclient | 30 | 51.48 | 5.64 | 45332 | 1422538 | 257.3 | 258.3 |
| 0.1 | full_fedhad | 30 | 68.69 | 3.32 | 49795 | 1593451 | 288.2 | 288.7 |
| 0.1 | step_permuted_lrclient | 30 | 66.83 | 4.23 | 49795 | 1593451 | 288.2 | 289.0 |

| alpha | mean diff (pp) | 95% CI boot | 95% CI t | dz | W/L | p Wilcoxon | p Holm | max abs mismatch steps / examples / exec FLOPs | nominal FLOPs mismatch mean (max) |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | +3.96 | [+2.58, +5.77] | [+2.22, +5.70] | 0.85 | 28/2 | 0.0000 | 0.0000 | 0.00% / 40.83% / 40.83% | -1.54% (40.80%) |
| 0.1 | +1.86 | [+0.99, +2.76] | [+0.93, +2.79] | 0.75 | 22/8 | 0.0005 | 0.0005 | 0.00% / 0.00% / 0.00% | +0.11% (0.90%) |

### Aggregation-weighted work (p_k = n_k / N)

| alpha | perm/full sum p*tau median [min, max] | perm/full sum p*eta*tau median | seeds with ratio < 1 | Spearman(diff, ratio p*eta*tau) | median Spearman(H_k, n_k) in Full |
|---|---|---|---|---|---|
| 0.01 | 0.70 [0.20, 0.92] | 0.65 | 30/30 | -0.40 (p=0.0265) | -0.90 |
| 0.1 | 0.74 [0.46, 0.93] | 0.70 | 30/30 | -0.63 (p=0.0002) | -0.90 |

## Experiment 2 — tuned static FedProx (validation phase only, seeds 72-76)

Evaluate phase NOT executed (no frozen_config.json). Numbers below are 5 validation seeds; they select a configuration, they are not the evaluation.

| alpha | config | val acc (%) | val - FedHAD (pp) | steps / FedHAD |
|---|---|---|---|---|
| 0.01 | FedProx lr=0.004 E=2 | 54.88 | +0.73 | 0.65 |
| 0.01 | FedProx lr=0.005 E=2 | 55.64 | +1.49 | 0.65 |
| 0.01 | FedProx lr=0.0075 E=2 | 55.43 | +1.28 | 0.65 |
| 0.01 | FedProx lr=0.01 E=2 | 52.80 | -1.35 | 0.65 |
| 0.01 | FedProx lr=0.004 E=3 | 55.48 | +1.34 | 0.97 |
| 0.01 | FedProx lr=0.005 E=3 | 54.80 | +0.66 | 0.97 |
| 0.01 | FedProx lr=0.0075 E=3 | 54.62 | +0.47 | 0.97 |
| 0.01 | FedProx lr=0.01 E=3 | 53.38 | -0.76 | 0.97 |
| 0.01 | FedHAD | 54.15 | +0.00 | 1.00 |
| 0.01 | FedProx lr=0.004 E=4 | 54.44 | +0.29 | 1.29 |
| 0.01 | FedProx lr=0.005 E=4 | 55.54 | +1.39 | 1.29 |
| 0.01 | FedProx lr=0.0075 E=4 | 53.42 | -0.73 | 1.29 |
| 0.01 | FedProx lr=0.01 E=4 | 52.24 | -1.91 | 1.29 |
| 0.01 | FedProx lr=0.004 E=5 | 54.10 | -0.04 | 1.62 |
| 0.01 | FedProx lr=0.005 E=5 | 53.99 | -0.16 | 1.62 |
| 0.01 | FedProx lr=0.0075 E=5 | 53.38 | -0.77 | 1.62 |
| 0.01 | FedProx lr=0.01 E=5 | 51.76 | -2.38 | 1.62 |
| 0.1 | FedProx lr=0.004 E=2 | 67.21 | -1.08 | 0.57 |
| 0.1 | FedProx lr=0.005 E=2 | 66.03 | -2.26 | 0.57 |
| 0.1 | FedProx lr=0.0075 E=2 | 65.93 | -2.36 | 0.57 |
| 0.1 | FedProx lr=0.01 E=2 | 63.79 | -4.50 | 0.57 |
| 0.1 | FedProx lr=0.004 E=3 | 68.31 | +0.02 | 0.85 |
| 0.1 | FedProx lr=0.005 E=3 | 68.51 | +0.22 | 0.85 |
| 0.1 | FedProx lr=0.0075 E=3 | 68.94 | +0.65 | 0.85 |
| 0.1 | FedProx lr=0.01 E=3 | 67.56 | -0.73 | 0.85 |
| 0.1 | FedHAD | 68.29 | +0.00 | 1.00 |
| 0.1 | FedProx lr=0.004 E=4 | 68.44 | +0.15 | 1.14 |
| 0.1 | FedProx lr=0.005 E=4 | 68.35 | +0.06 | 1.14 |
| 0.1 | FedProx lr=0.0075 E=4 | 68.22 | -0.06 | 1.14 |
| 0.1 | FedProx lr=0.01 E=4 | 67.52 | -0.77 | 1.14 |
| 0.1 | FedProx lr=0.004 E=5 | 68.48 | +0.20 | 1.42 |
| 0.1 | FedProx lr=0.005 E=5 | 69.12 | +0.83 | 1.42 |
| 0.1 | FedProx lr=0.0075 E=5 | 68.16 | -0.13 | 1.42 |
| 0.1 | FedProx lr=0.01 E=5 | 67.46 | -0.83 | 1.42 |

Pre-specified `budget_matched` proposal (steps <= 1.02 x FedHAD, best validation accuracy):
- alpha=0.01: FedProx lr=0.005, E=2 (val 55.64% vs FedHAD 54.15%; steps 28084 vs 43402)
- alpha=0.1: FedProx lr=0.0075, E=3 (val 68.94% vs FedHAD 68.29%; steps 42102 vs 49324)
