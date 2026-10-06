# FedHAD: Federated Heterogeneity-Aware Dynamics

> Artifact of the paper **"FedHAD: Enhancing Federated Learning under Non-IID Data Distributions"**,
> submitted to the *Journal of Network and Computer Applications* (JNCA), part of a doctoral thesis.
> It contains the training code of the six compared methods, the runner of each experiment, the
> raw results of every run, the scripts that produce each table and figure of the paper, and the
> pre-registrations of the control batteries.

The complete campaign comprises almost **10,000 federated runs** (more than 600,000 rounds of local
training at the client level), all with their raw report and telemetry preserved in
[`results/`](results/).

> **Language note.** Some folder names (e.g. `algoritmos/`, `test2_robustez_alpha`), code comments,
> CSV column names and internal reports are in Portuguese. All documentation needed to trace the
> reported results back to the raw runs is in English.

---

## Contents

1. [The method on one page](#1-the-method-on-one-page)
2. [Compared methods](#2-compared-methods)
3. [Datasets, partitions and models](#3-datasets-partitions-and-models)
4. [Experimental protocol](#4-experimental-protocol)
5. [Executed experiments](#5-executed-experiments)
6. [Main results](#6-main-results)
7. [What the work concludes](#7-what-the-work-concludes)
8. [Repository organization](#8-repository-organization)
9. [How to reproduce](#9-how-to-reproduce)
10. [Limitations](#10-limitations)

---

## 1. The method on one page

In federated learning with non-IID data, each client sees a class distribution that differs from
the global one, and local training pulls the model away from the global objective (*client drift*).
**FedHAD** derives the number of local epochs and the learning rate of each client from a single,
locally computed statistic. The reference implementation is
[`algoritmos/FedHAD_Final_2.0.py`](algoritmos/FedHAD_Final_2.0.py).

**Heterogeneity score.** Each client computes the coefficient of variation (CV) of its class counts
and normalizes it by the theoretical maximum `sqrt(C − 1)`, where `C` is the number of classes:

```
H_k = clamp( CV(counts_k) / sqrt(C − 1), 0, 1 )
```

`H_k = 0` denotes a balanced client and `H_k = 1` a single-class client. The score uses only the
local counts, which are not transmitted. The CV measures dispersion around the uniform distribution,
which coincides with the global prior only when that prior is uniform. Local entropy normalized by
`log C` has the same operational properties; the CV is a design choice, not a demonstrated advantage
over it.

**Epochs and learning rate.** Both decrease with `H_k`:

```
E_k   = max(E_min, round(E_base − λ_E · H_k))        # E_base = 5, E_min = 2, λ_E = 3.0
η_k   = max(η_min, η_base / (1 + λ_η · H_k))         # η_base = 1e-2, η_min = 1e-3, λ_η = 1.5
```

With these values, `E_k` ranges from 5 to 2 epochs and `η_k` from `1e-2` to `4e-3` (the floor
`η_min` is never reached). The factors were fixed at the start of the campaign and were not tuned.

**Proximal term.** The local loss includes `(μ/2)·‖w − w_global‖²`, with `μ = 1e-2` fixed for all
clients, the same coefficient as FedProx. The difference between FedHAD and FedProx therefore
isolates the two adaptations.

**What does not change.** Aggregation is that of FedAvg (average weighted by sample count), and the
exchanged messages are those of standard FedAvg. The whole mechanism runs at the client. Because the
partitions are static, `H_k`, `E_k` and `η_k` are constant across rounds for a given client: the
adaptation is across clients, not over time.

---

## 2. Compared methods

All methods were reimplemented on the same substrate: same networks, same partitioning, same seeding
and same cost accounting. In the main campaign the baselines use the same common configuration,
without per-method tuning: **5 local epochs**, SGD with momentum 0.9, `lr = 1e-2`, batch size 32.

| Method | File | What it does |
|---|---|---|
| FedAvg | [`FedAVG_Final.py`](algoritmos/FedAVG_Final.py) | Average of the weights, weighted by sample count |
| FedAvgM | [`FedAvgM_Final.py`](algoritmos/FedAvgM_Final.py) | Server momentum (0.9), applied only to trainable parameters |
| FedProx | [`FedProx_Final.py`](algoritmos/FedProx_Final.py) | Proximal term in the local loss, `μ = 1e-2` |
| FedNova | [`FedNova_Final.py`](algoritmos/FedNova_Final.py) | Normalizes updates by the effective number of local steps |
| SCAFFOLD | [`SCAFFOLD_Final.py`](algoritmos/SCAFFOLD_Final.py) | Control variates (option II), effective steps under momentum; doubles the state exchanged per round |
| **FedHAD** | [`FedHAD_Final_2.0.py`](algoritmos/FedHAD_Final_2.0.py) | Proposed method (Section 1) |

The protocol equalizes the **local-epoch ceiling**, not the realized computation: FedHAD never
trains more than the baselines and, in aggregate, performs fewer updates by construction.
Comparisons at a matched update budget are in the component analysis and the control batteries
(Section 5).

---

## 3. Datasets, partitions and models

| Dataset | Classes | Network | Training FLOPs per sample | Partition |
|---|:-:|---|:-:|---|
| MNIST, FashionMNIST | 10 | LeNet-5 (44,426 parameters) | 1.69 × 10⁶ | Dirichlet(α) |
| CIFAR-10 | 10 | 5-convolution CNN with BatchNorm (667,178 parameters) | 180.89 × 10⁶ | Dirichlet(α) |
| FEMNIST | 62 | 4-convolution CNN with BatchNorm (909,342 parameters) | 116.60 × 10⁶ | Natural, by writer |

- **Dirichlet:** high α yields nearly IID partitions and low α strongly imbalanced ones. The levels
  used are α ∈ {1.0, 0.5, 0.1, 0.01}. At α = 0.01, five of the 30 CIFAR-10 seeds leave one client
  with no samples.
- **FEMNIST:** the same 10 writers form the federation in all 30 seeds, which are replicates of a
  single federation, not 30 independent federations. Evaluation uses 100 writers unseen in training,
  drawn per seed.
- Each client splits its data 90/10 into training and local validation. In the main campaign the
  validation split is not used to select anything; its only use for selection is the tuning of
  FedProx in the control batteries, on separate seeds.

---

## 4. Experimental protocol

- **Simulation:** Flower 1.31 on Ray, with full participation of all clients in every round
  (*cross-silo* regime), 5 clients and 10 rounds by default.
- **Repetitions:** 30 seeds per configuration (42 to 71), paired by seed. The seed fixes the data
  partition, so every method sees the same partitions.
- **Initialization:** in the main campaign, initialization is not common to all methods. The
  component analysis and the control batteries start from a common checkpoint per seed, verified by
  SHA-256 (seeds 42 to 166). No run is bitwise reproducible.
- **Statistics:** paired Wilcoxon test, bootstrap intervals over seeds and effect size `d_z`. Holm
  within defined families: F1 (CIFAR-10), F2 (MNIST, FashionMNIST and FEMNIST), F3 (50 rounds) and
  F4 (component analysis), plus two families for SCAFFOLD. An undetected difference is not treated
  as equivalence; equivalence and non-inferiority are claimed only in the pre-registered batteries,
  with margins fixed in advance (0.75 points at α = 0.1 and 1.0 at α = 0.01).
- **Cost:** nominal network FLOPs (forward + backward = 3 × forward, counted with `thop`) and executed
  optimizer steps. Executed totals discount the last incomplete minibatch, which matters only on
  FEMNIST (about 9%). Wall-clock time is reported separately, in the communication family.
- **Platforms:** a workstation (RTX A1000) and a server (A40). The hardware changes neither the
  configuration nor the FLOPs; wall-clock times are compared only within the same machine.

Details in [`results/README.md`](results/README.md) and in Section 5 of the paper.

---

## 5. Executed experiments

| Experiment | Runs | Runner | Results |
|---|:-:|---|---|
| Main campaign, default α 0.5: tests 1 to 8, FedNova and SCAFFOLD in tests 1, 2 and 8, drift-diagnostic campaign | 4,530 | `run_main_campaign.py`, `run_drift_diagnostics.py` | `results/main_campaign_default_alpha_0_5/` |
| Main campaign, default α 0.01 (includes 50 rounds at α = 0.01 and on FEMNIST) | 3,630 | `run_main_campaign.py` | `results/main_campaign_default_alpha_0_01/` |
| Component analysis at a matched budget | 300 | `run_component_ablation.py` | `results/component_analysis_ablation/` |
| Permutation preserving executed steps | 210 | `run_tuned_controls_and_step_permutation.py` | `results/step_preserving_permutation/` |
| Tuned FedProx: validation grid and pre-registered evaluation | 170 + 300 | `run_tuned_controls_and_step_permutation.py`, `run_tuned_controls_evaluation.sh` | `results/tuned_controls_fedprox_validation/` |
| Uniform learning-rate control (pre-registered) | 360 | `run_lr_uniform_control.py` | `results/lr_uniform_control/` |
| Global-gradient diagnostic (pre-registered) | 300 | `run_global_gradient_diagnostic.py` | `results/global_gradient_diagnostic/` |
| Re-executions of incomplete runs | 6 | no dedicated runner | `results/seed_reexecutions_corrupted_runs/` |

The tests of the main campaign keep the (Portuguese) names under which they were executed:

| Folder | Family | Paper section |
|---|---|---|
| `test1_convergencia` | Base point: α = 0.5, 5 clients, 10 rounds, three datasets | 6.2 |
| `test2_robustez_alpha` | Heterogeneity: α ∈ {1.0, 0.1, 0.01} | 6.3 |
| `test3_comunicacao` | Artificial communication delay (0.05, 0.1 and 0.5 s) | 6.6 |
| `test4_clientes` | 3, 5 and 10 clients | 6.6 |
| `test5_ablacao` | Historical on/off ablation, superseded by the component analysis | not used |
| `test6_calibracao` | 3 × 3 grid of the decay factors | 6.12 |
| `test7_plato` | Extended horizon, 50 rounds | 6.7 |
| `test8_femnist` | FEMNIST natural federation | 6.8 |
| `1.1_diagnostico_drift` | Diagnostic campaign, 10 seeds | 6.4 |

FedNova and SCAFFOLD were executed at the base point, in the heterogeneity family and on FEMNIST,
but not in the communication, client-count and extended-horizon families.

---

## 6. Main results

Final accuracy values: mean ± standard deviation over 30 seeds. "Best baseline" is the best of
FedAvg, FedAvgM and FedProx.

### 6.1. Summary (CIFAR-10 and FEMNIST)

| Configuration | FedHAD | Best baseline | Δ (points) | TFLOPs (FedHAD / base) | ΔFLOPs |
|---|:-:|:-:|:-:|:-:|:-:|
| CIFAR-10, α = 1.0 | 82.04 | 82.01 (FedProx) | +0.03 | 330.0 / 407.0 | −19% |
| CIFAR-10, α = 0.5 | 80.21 | 80.41 (FedProx) | −0.20 | 325.3 / 407.0 | −20% |
| CIFAR-10, α = 0.1 | 68.53 | 69.03 (FedAvg) | −0.50 | 288.7 / 407.0 | −29% |
| CIFAR-10, α = 0.01 | 55.78 | 54.16 (FedAvg) | +1.62 | 262.8 / 407.0 | −35% |
| CIFAR-10, 50 rounds, α = 0.5 | 81.80 | 81.77 (FedProx) | +0.02 | 1626.3 / 2035.2 | −20% |
| CIFAR-10, 50 rounds, α = 0.01 | 65.81 | 64.97 (FedProx) | +0.84 | 1313.9 / 2035.2 | −35% |
| FEMNIST, 10 rounds | 65.11 | 65.94 (FedProx) | −0.83 | 11.13 / 12.39 | −10% |

- **Computation:** FedHAD performs 19% to 35% less local training on CIFAR-10 (about 10% on
  FEMNIST), and the reduction grows with the imbalance. This is a reduction in network operations,
  not in time: on CIFAR-10, FedHAD takes 458.1 s against 563.7 s for FedProx and 387.6 s for FedAvg.
- **Extreme skew (α = 0.01):** FedHAD has the highest mean accuracy and the smallest dispersion. The
  advantage over FedProx, FedNova and FedAvgM survives the Holm adjustment; the advantage over FedAvg
  does not (p = 0.026, p_H = 0.157).
- **Moderate skew:** at α = 1.0 and 0.5, FedHAD stays within 0.2 points of the best baseline other
  than SCAFFOLD; at α = 0.1 it is 0.50 points behind FedAvg.
- **MNIST:** FedAvg is more accurate at every level; there is little drift to mitigate in a task that
  every method solves almost equally well.
- **FEMNIST:** at 10 rounds FedHAD is 0.83 points behind FedProx (a detected difference). At 50
  rounds the differences against the three baselines are below 0.15 points, none is detected, and
  the reduction of about 10% in computation persists.
- **Number of clients:** with 3, 5 and 10 clients, FedProx has the highest mean; with 10 clients the
  difference is detected. The relative position of FedHAD worsens as clients become individually
  smaller.

### 6.2. FedNova and SCAFFOLD

- **FedNova:** competitive in most configurations. Under extreme skew on CIFAR-10, it is 9.17 points
  behind FedHAD (46.60% against 55.78%).
- **SCAFFOLD:** the most accurate method wherever it is stable. On CIFAR-10 it leads FedHAD by 0.83,
  1.81 and 3.75 points at α = 1.0, 0.5 and 0.1. The exception is MNIST at α = 0.1, where FedNova is
  marginally ahead.
- **SCAFFOLD collapse:** under extreme skew it collapses to chance-level accuracy in **18 of its 90
  runs**; no FedHAD or FedNova run collapses. The cause is a tiny client (85 samples) whose control
  variate dominates the unweighted server average. The per-client diagnosis is reproducible with
  [`sanity_tests/scaffold_collapse_diagnostic.py`](sanity_tests/scaffold_collapse_diagnostic.py).

### 6.3. Component analysis (CIFAR-10, matched budget)

Five arms start from the same checksum-verified initialization: complete policy, epochs only,
learning rate only, fixed allocation at a matched budget, and permuted allocation.

- The **learning rate** accounts for most of the accuracy effect: +1.81 points over the fixed control
  at α = 0.01.
- The **epochs** select the budget: 35% fewer updates than the baselines at α = 0.01 and 29% at
  α = 0.1.
- No additional gain from combining the two is detected (+0.14 and +0.04 points, intervals crossing
  zero).

### 6.4. Control batteries (CIFAR-10, common checkpoint)

- **Uniform learning rate (pre-registered):** a single rate with the same aggregate step reproduces
  the effect (equivalent at α = 0.1) and exceeds it at α = 0.01 (+1.13 points). The gain comes from
  the average step size, not from giving each client a different rate.
- **Step-preserving permutation:** redistributing the budgets while keeping the exact total costs
  3.96 and 1.86 points. The allocation matters, but in the partitions used it coincides with an
  allocation by client size.
- **FedProx tuned on validation seeds (pre-registered):** FedHAD is practically equivalent at
  α = 0.01 (120 seeds) and non-inferior at α = 0.1, while the tuned FedProx uses fewer updates.
  Against the default FedProx, FedHAD is more accurate at both levels.
- **Global gradient (pre-registered):** with equal client sizes, more imbalanced clients return
  updates less aligned with the rest of the federation. On the real partitions this cannot be
  separated from size. The epochs FedHAD removes are neutral, not harmful. The accuracy of FedHAD's
  allocation against the inverted one is inconclusive or equivalent under the pre-registered rule.

### 6.5. Sensitivity to the decay factors

On the 3 × 3 grid (CIFAR-10, α = 0.5), accuracy varies between 79.55% and 80.55%, less than the
seed-to-seed deviation. Cost is governed almost entirely by `λ_E` (404.0, 325.3 and 274.6 TFLOPs).

---

## 7. What the work concludes

- **Against baselines in the common configuration**, FedHAD offers an accuracy–computation
  trade-off: comparable accuracy with 19% to 35% less local training on CIFAR-10 (about 10% on
  FEMNIST), and the highest accuracy under extreme skew on CIFAR-10.
- **Against a tuned baseline**, it is neither more accurate nor cheaper. What it delivers is an
  operating point comparable to that of a validation search, obtained **in a single run and without
  the search**, from local statistics and without extra communication. In this study the search cost
  80 runs per regime, and the selected configuration changed between regimes.
- **The mechanism** acts through the average step size and by directing training towards the clients
  that weigh most in the aggregation, not by correcting the drift of each client individually.

---

## 8. Repository organization

The repository separates **training code** (`algoritmos/`), **runners** (`runners/`), **analysis
code** (`analysis/`) and **data** (`results/`). Every runner writes only to `results/`.

```
FedHAD-CrossSilo/
├── runners/                       # one runner per experiment (run from the repository root)
├── algoritmos/                    # training code
│   ├── *_Final.py                 # the six methods of the main campaign
│   ├── drift_telemetry.py         # update-alignment telemetry
│   ├── component_analysis_ablation/
│   ├── tuned_controls_and_step_permutation/
│   ├── lr_uniform_control/
│   ├── global_gradient_diagnostic/
│   └── seed_reexecution_probes/
├── analysis/                      # results/ -> tables, figures and numbers of the paper
│   ├── README.md                  # map: paper item -> script -> output file
│   ├── main_campaign/             # consolidation and figures of tests 1 to 8
│   ├── audit/                     # independent recomputation of the paper's numbers (a01 to a15)
│   ├── revision_consolidated_tables_figures/
│   └── <one folder per controlled experiment>
├── sanity_tests/                  # sanity tests (FedNova, SCAFFOLD) and unit tests
├── results/                       # raw and derived data (see results/README.md)
└── requirements.txt               # pinned dependencies (Python 3.12)
```

- [`runners/README.md`](runners/README.md): what each runner executes.
- [`analysis/README.md`](analysis/README.md): which script produces each table and figure of the paper.
- [`results/README.md`](results/README.md): what each results folder contains, what can be verified
  and where, and the history of the folder reorganizations.

**Frozen code.** The training scripts of the controlled batteries (`fedhad_r2.py`, `r2_policy.py`,
`fedhad_lru.py`, `lru_policy.py`, `fedhad_gg.py`, `gg_policy.py`, `gg_diag.py`, `fedhad_ablation.py`,
`ablation_policy.py`, `drift_telemetry.py`) are, byte for byte, the ones that produced the data. The
SHA-256 of each is recorded in the run fingerprints and in the pre-registrations, so neither their
names nor their contents were changed. Some comments and internal paths refer to the old folder
layout; the runners supply the current paths through `PYTHONPATH`. The suffix `r2` is only the
internal name of this code.

---

## 9. How to reproduce

### 9.1. Environment

Tested with Python 3.12, PyTorch 2.5.1 + CUDA 12.1, Flower 1.31 and Ray 2.55.

```bash
python -m venv env_flwr_pt
source env_flwr_pt/bin/activate          # Windows: .\env_flwr_pt\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

For CPU, remove the `+cu121` suffixes and the `--extra-index-url` line from `requirements.txt`.
FEMNIST requires `flwr-datasets[vision]` and downloads the dataset from the HuggingFace Hub on the
first run.

### 9.2. Regenerate tables, figures and numbers (without training)

No analysis script trains models; all of them read `results/`. The complete order is in
[`analysis/README.md`](analysis/README.md). Examples:

```bash
python analysis/main_campaign/analisar_resultados.py      # consolidated CSVs per test
python analysis/main_campaign/figuras_artigo.py            # figures of tests 1 to 8
python analysis/audit/a06_stats_holm.py                    # paired tests and Holm per family
python analysis/lr_uniform_control/analyze_lr_uniform_control.py   # checks the pre-registration first
```

The three pre-registered analyses check the SHA-256 of the pre-registration before computing
anything and refuse to run if it, the frozen configuration or the code changed.

### 9.3. Run experiments

Every runner resumes automatically from where it stopped, and `--dry-run` shows the plan without
training or writing any file:

```bash
python runners/run_main_campaign.py --dry-run
python runners/run_global_gradient_diagnostic.py --dry-run
```

In the main campaign, `ALGORITMOS_ATIVOS` (active methods) and `EXPERIMENTOS_ATIVOS` (active
experiments), at the top of `runners/run_main_campaign.py`, select what to run. Each `*_Final.py`
also runs on its own, with the values at the top of the file or with `FL_*` environment variables.

### 9.4. Tests

```bash
python -m pytest sanity_tests/test_tuned_controls_policy.py sanity_tests/test_lr_uniform_control.py \
                 sanity_tests/test_global_gradient_diagnostic.py      # unit tests (fast)
python sanity_tests/test_fednova_sanity.py                           # FedNova sanity checks
python sanity_tests/test_scaffold_sanity.py                          # SCAFFOLD sanity checks
```

---

## 10. Limitations

- **Evaluated regime:** *cross-silo* with full participation and at most 10 clients. Client sampling,
  failures and large populations were not evaluated.
- **Hyperparameter tuning:** only FedProx was tuned, on CIFAR-10, at two levels of α and 10 rounds.
  FedAvg, FedAvgM, FedNova and SCAFFOLD remained at the common configuration; FedDyn was not
  evaluated.
- **Coverage:** FedNova and SCAFFOLD were not executed in the communication, client-count and
  extended-horizon families. The extended horizon covers CIFAR-10 at two levels of α and FEMNIST.
- **Initialization:** the comparisons of the main campaign do not start from a common checkpoint.
- **Imbalance and size:** in the Dirichlet partitions used, the score `H_k` and client size move
  almost together, which prevents separating allocation by heterogeneity from allocation by size.
- **Cost:** FLOPs are a nominal estimate of the network computation. They exclude the proximal term,
  the optimizer and evaluation, and are not equivalent to execution time. The main-campaign reports
  record energy via CodeCarbon, but these values are not used in the paper: they depend on the
  hardware and, in a large share of the runs, GPU energy is recorded as zero.
- **Theory:** the motivation is a first-order argument for plain SGD; there is no convergence
  analysis, and no experiment was run to a stationary state.

---

*Doctoral thesis repository. Code, data and scripts intended for the verification and
reproducibility of the paper.*
