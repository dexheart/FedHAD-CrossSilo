# FedHAD controlled component analysis

Controlled experiment of Section 6.10 of the manuscript (Tables 14 and 15). This folder holds
its data: raw fingerprints and telemetry, the training script's own `.txt` reports, tables,
figures and logs. The training code and configuration are in
`algoritmos/component_analysis_ablation/`, the runner is `runners/run_component_ablation.py`
and the analysis scripts are in `analysis/component_analysis_ablation/`. No result from any
other campaign is overwritten, moved or deleted.

> **How this analysis is read in the manuscript.** It is the first of the controlled analyses.
> Two of its readings were refined by the control batteries of Section 6.11, executed later
> from the same initial checkpoints: the learning-rate effect is reproduced by a single
> uniform rate with the same aggregate step (`results/lr_uniform_control/`), and the permuted
> arm (Q2), whose budget is not matched, is superseded by a permutation that preserves executed
> updates exactly (`results/step_preserving_permutation/`). See §11.

## 1. Objective

The campaign addresses four questions that a single controlled design can answer:

1. **Equal budget.** Does FedHAD beat a non-adaptive baseline when both spend the same
   amount of local computation? Any gain that disappears at matched budget is a gain
   from training more, not from adapting.
2. **Isolated contribution of adaptive epochs.**
3. **Isolated contribution of the adaptive learning rate.**
4. **Does heterogeneity-guided allocation matter**, or is only the *amount* of
   computation that matters, irrespective of which client receives it?

## 2. Hypothesis and pre-declared interpretations

Written **before any result was produced**, so the conclusion cannot be
reverse-engineered from the numbers. All five outcomes below are considered live.

- **H1 — coordinated adaptation helps at matched budget.** `full_fedhad` beats
  `fixed_matched`, and beats `permuted_allocation` too, i.e. *who* receives the compute
  matters, not only how much.
- **H2 — the gain is entirely attributable to one component.** One isolated arm
  (`epoch_only` or `lr_only_matched`) already matches `full_fedhad`. Then FedHAD should
  be presented as that single mechanism plus an unnecessary second one.
- **H3 — coordination adds nothing beyond the parts.** Both isolated arms improve on
  `fixed_matched`, but `full_fedhad` does not exceed the better of them. Then the claim
  of a *coordinated* benefit is not supported and must be dropped.
- **H4 — the budget explains everything.** `fixed_matched` matches `full_fedhad`. Then
  the reported advantage of FedHAD is a budget artifact and the paper must say so.
- **H5 — only the multiset matters.** `permuted_allocation` matches `full_fedhad`. Then the
  amount of computation is what matters and the heterogeneity-guided *assignment* is
  decorative.

**Reporting rule.** Statistical significance and practical effect are reported
separately, and `p > 0.05` is never treated as proof of equivalence — only as failure to
detect a difference at this sample size, always accompanied by the effect size and its
confidence interval.

## 3. Experimental design

| Axis | Value |
|---|---|
| Dataset | CIFAR-10 |
| α (Dirichlet) | 0.01 and 0.1 — the two most heterogeneous settings of the campaign |
| Clients | 5 |
| Rounds | 10 |
| Seeds | 42–71 (the same 30 as the main campaign) |
| Arms | 5 |
| Runs | 5 × 2 × 30 = **300** |

### The five arms

Let `H_k` be the client's normalised heterogeneity, `E_full(H_k)` / `LR_full(H_k)` the
policies current FedHAD applies, and `E_fixed` a deterministic allocation that never
looks at `H_k` but sums to exactly the same total.

| Arm | Epochs | Learning rate | Isolates |
|---|---|---|---|
| `full_fedhad` | `E_full(H_k)` | `LR_full(H_k)` | FedHAD as it stands |
| `fixed_matched` | `E_fixed` | `BASE_LR` | **the matched-budget control** |
| `permuted_allocation` | `perm(E_full)` | `LR_full(H_k)` | *who* gets the compute (compute **not** matched) |
| `epoch_only` | `E_full(H_k)` | `BASE_LR` | adaptive epochs alone |
| `lr_only_matched` | `E_fixed` | `LR_full(H_k)` | adaptive learning rate alone |

Contrasts that answer the five questions:

1. `full_fedhad` vs `fixed_matched` — does adaptation help at equal budget?
2. `full_fedhad` vs `permuted_allocation` — does the *assignment* matter?
3. `epoch_only` vs `fixed_matched` — contribution of adaptive epochs.
4. `lr_only_matched` vs `fixed_matched` — contribution of adaptive LR.
5. `full_fedhad` vs both isolated arms — is coordination worth more than its parts?

**`E_fixed`** is the argmin of `|U - U_full|` over the flat family
`E_k ∈ {e0, e0+1}`, `e0 = ⌊U_full / Σ batches⌋`, excluding `E_full` itself. It reads
**only client sizes** and the target update total — never `H_k`, accuracy or any training
output. Tie-break, in order: closest update total; flattest allocation (fewest clients
bumped); lexicographically smallest.

**`perm`** is chosen by enumerating every unique non-trivial re-assignment of the exact
`E_full` multiset (9–29 per cell for K=5) and taking the argmin of `|U - U_full|`, again
before any training and without reading accuracy or loss. Tie-break: closest update total;
then *most* clients whose budget changed (maximally destroying the H → compute
association); then lexicographically smallest. Both allocations are stored in every
fingerprint for audit.

`fixed_matched` is the **primary matched-budget control**; `permuted_allocation` is
complementary evidence about the client→budget association (see §4).

## 4. Fairness controls

Every `(alpha, seed)` cell is checked across its five arms. **Any violation aborts the
campaign** rather than being reported as a caveat — an unpaired ablation is not worth
publishing.

| Invariant | Enforced by |
|---|---|
| Identical initial weights | SHA-256 of the initial `state_dict` |
| Identical client partitions | SHA-256 of each client's absolute indices |
| Identical clients and sizes | `n_samples`, `n_batches` per client |
| Identical heterogeneity | SHA-256 of the `H_k` vector |
| Identical round count and batch size | direct comparison |
| Exact update budget for the arms that reuse `E_full` | `full_fedhad` and `epoch_only` |
| Per-arm structural invariants | see below |

Per-arm invariants verified from the fingerprints:

- `full_fedhad`: `epochs == E_full` and `lr == LR_full`
- `fixed_matched`: `epochs == E_fixed` and `lr == BASE_LR` for every client
- `permuted_allocation`: `multiset(epochs) == multiset(E_full)` and `lr == LR_full`
- `epoch_only`: `epochs == E_full` **and** identical to `full_fedhad`'s allocation — only
  the learning rate differs
- `lr_only_matched`: `epochs == E_fixed` **and** identical to `fixed_matched`'s
  allocation — only the learning rate differs

### What is matched, and what is not

**Compute is matched on minibatch updates**, `U = Σ_k E_k · batches_k`, not on the epoch
sum. Equal epochs is not equal compute when clients hold different amounts of data.

| Arm | Update matching |
|---|---|
| `full_fedhad` | reference |
| `epoch_only` | **exact** by construction (reuses `E_full`) |
| `fixed_matched` | argmin over the flat family — **≤1% in 52/60 cells**, median 0.23–0.27%, max 3.44% |
| `lr_only_matched` | identical to `fixed_matched` |
| `permuted_allocation` | **not matched** — ≤1% in only 25/60 cells, median 2.75% (α=0.01), max 15.5% |

**Why `permuted_allocation` cannot be budget-matched.** Three facts combine: epochs are
integers, there are only five clients, and each client has a different number of
minibatches. A permutation has no free parameters — it can only reuse the multiset it was
given — so with as few as 4 unique non-trivial arrangements in some cells, none lands near
`U_full`. The deviation is also **biased downward** (26/30 cells at α=0.01 spend *fewer*
updates than `full_fedhad`).

Consequently:

- **`permuted_allocation` is NOT claimed to be a matched-budget arm.** Its ΔU is reported
  per seed and treated as a limitation and a covariate.
- Conclusions about equal budget rest on **`full_fedhad`, `fixed_matched`, `epoch_only`
  and `lr_only_matched`**, where compute is matched exactly or to within ~0.2% median.
- `permuted_allocation` is **complementary evidence** on whether the client→budget
  association matters, read together with its ΔU.

### Offline diagnostics of the control (all 30 seeds, no training)

`tables/size_heterogeneity_offline.csv`, produced by
`analysis/component_analysis_ablation/analyze_size_heterogeneity.py`:

| Quantity | α=0.01 | α=0.1 |
|---|---|---|
| ρ(n_k, H_k) | mean −0.757, median −0.900 | mean −0.800, median −0.900 |
| ρ(E_full, H_k) | mean −0.753 | mean −0.777 |
| **ρ(E_fixed, H_k)** | **mean +0.158** | **mean +0.000** |
| ρ(E_perm, H_k) | mean −0.393 | mean −0.327 |

Client size and heterogeneity are strongly anti-correlated, and that is **structural**
rather than an artifact of the smoke seeds. Because matching updates requires favouring
large clients, an H-blind allocation could in principle drift towards FedHAD's. It does
not: ρ(E_fixed, H_k) averages ≈0 against ≈−0.77 for `E_full`, and in 4 of 60 cells
`E_fixed` is perfectly uniform. In 10 of 60 cells `E_fixed` differs from `E_full` in only
one client (Hamming ≤1) — those cells are flagged per seed rather than removed.

## 5. Code provenance and the one behavioural change

The original FedHAD sources were **copied**, not modified in place. The copies are in
`algoritmos/component_analysis_ablation/`; hashes of the originals and of the copies are in
`ORIGINAL_HASHES.txt` there, and the full diff is in `fedhad_ablation.diff`.

| File | Origin | Modified? |
|---|---|---|
| `fedhad_ablation.py` | copy of `FedHAD_Final_2.0.py` | **yes**, at 4 marked sites |
| `drift_telemetry.py` | copy of `drift_telemetry.py` | no, verified byte-identical |
| `ablation_policy.py` | new | new |

The four modifications, each marked `=== ABLATION MODIFICATION (n of 4) ===`:

1. Import `ablation_policy`.
2. In `fit()`, read `(E_k, LR_k)` from a precomputed per-client table instead of deriving
   them inline from `H_k`. For `full_fedhad` the values are exactly
   `define_epochs_dinamicas(H_k)` and `define_lr_dinamico(H_k)` — the original behaviour.
3. After `load_data()`, precompute every arm's allocation and assert the invariants.
4. Pass deterministic `initial_parameters`, and write the per-run fingerprint and
   telemetry.

### The initialisation difference from the historical implementation

The original script passes **no** `initial_parameters`, so Flower logs *"Requesting
initial parameters from one random client"* and the initial global model comes from a
`get_net()` executed inside a Ray worker, whose torch RNG is not seeded by the driver's
`set_global_seed(SEED)`.

Measured on the unmodified script, CIFAR-10 α=0.01 seed 42, three runs:

| Run | Round-0 loss | Round-0 accuracy | Round-1 accuracy |
|---|---|---|---|
| 1 | 2.3031 | 0.1000 | 0.1740 |
| 2 | 2.3036 | 0.1001 | 0.2059 |
| 3 | 2.302957 | 0.0983 | — |

**The seed controls the partition, the split and the data order, but not the initial
model.** Same-seed runs differ by 3.2 accuracy points at round 1 — comparable to or larger
than the effects this ablation is meant to resolve.

This copy therefore fixes the initial model deterministically in the driver, exactly as
`FedAvgM_Final.py` and `FedNova_Final.py` already do. **This is the only behavioural
difference from the historical implementation.** It applies identically to all five arms,
including `full_fedhad`, which is why:

- the ablation re-runs `full_fedhad` under this protocol instead of reusing historical
  FedHAD results, and
- **all comparisons in this folder use only runs from this campaign.**

No original file and no previous result was altered. Whether the historical campaign
should adopt the same fix is a separate decision, deliberately left untouched here.

### Sanity check against the original

`full_fedhad` from the copy was verified to reproduce the original's **deterministic**
telemetry exactly, for CIFAR-10 α=0.01 seed 42 (all five clients, `diff`-identical):

```
Cliente 0: HetNorm=0.5140 | Epochs=3 | LR=0.005646
Cliente 1: HetNorm=0.4573 | Epochs=4 | LR=0.005931
Cliente 2: HetNorm=0.6719 | Epochs=3 | LR=0.004980
Cliente 3: HetNorm=0.9995 | Epochs=2 | LR=0.004001
Cliente 4: HetNorm=0.6506 | Epochs=3 | LR=0.005061
```

Accuracy and loss are *not* required to match, because the original does not reproduce
itself (see above). The check therefore covers everything that is deterministic —
`H_k`, `E_k`, `LR_k`, partitions, client sizes — and treats the trajectory as
distributional.

> **Implementation note.** Ray does not re-execute the entry script's module-level code
> inside worker processes: imported modules arrive re-imported and empty. The first
> implementation stored the allocation in `ablation_policy` and every client raised
> `build_allocation() must run before policy_for()`, which Flower absorbed as a client
> failure — the run "completed" in 4.9 s with round 0 and round 1 identical. The
> allocation is now held in a plain list, which cloudpickle serialises **by value**. Any
> future change here must keep that property.

## 6. Inclusion / exclusion criteria

- Every `(arm, alpha, seed)` cell must complete with exit status 0. A failed run aborts
  the campaign; results are never reported from a partial grid.
- No client is excluded. The protocol declared that a cell in which a client ends a round
  with `n_k = 0` would be excluded for all five arms simultaneously, to preserve pairing.
  **That rule was not applied.** Five seeds at α = 0.01 (46, 48, 51, 52 and 57) contain a client
  without samples; they were retained in the primary analysis, and the manuscript declares the
  deviation (Section 6.14). A sensitivity analysis without them (n = 25, same twelve-test Holm
  family) leaves the conclusions unchanged except for a weaker epoch-only contrast (Section 6.10;
  `analysis/audit/a11_ablation_sensitivity_empty.py`). In two of these seeds the fixed allocation
  coincides with that of the complete policy on every client with data, so the LR-only and
  complete arms execute identical training.
- Smoke-test output lives in `raw/smoke/` and is **never** merged with `raw/official/`.

## 7. Running the campaign

Every command is run from the **repository root**. The runner itself only uses the standard
library and finds the project environment on its own for the training subprocesses, so a plain
`python` is enough. Training subprocesses run with the repository root as working directory, so
the datasets are read from `<repo>/data`; results are written to this folder.

### How to run

```bash
python runners/run_component_ablation.py
```

That is the whole command. With no flag it starts the battery, and if one is
already under way it resumes it — the two are the same operation.
`--official` and `--resume` are explicit aliases for it.

Starts (or continues) the 300-cell battery: CIFAR-10, α ∈ {0.01, 0.1}, 30 seeds,
5 arms. Roughly 15 h end to end at ~180 s per cell. On a cluster, run it under
`nohup`/`tmux`/`sbatch` so it survives a lost session:

```bash
nohup python runners/run_component_ablation.py --official > results/component_analysis_ablation/logs/official_run.log 2>&1 &
```

### How to resume

**Run exactly the same command again.** Resume is the default behaviour; there is
nothing extra to pass and nothing to clean up first.

```bash
python runners/run_component_ablation.py              # resumes
```

Completed, still-valid cells are skipped; pending, failed and interrupted cells
are re-executed. A cell that was mid-flight when the job died has no valid
artifacts, so it is re-run rather than counted as done — losing at most the one
cell that was in progress.

### How to check status

```bash
python runners/run_component_ablation.py --status                    # official grid
python runners/run_component_ablation.py --status --mode smoke       # smoke grid
```

Reports how many of the 300 cells are complete, the breakdown by arm and by α,
any failed cells with their error message, the mean cell duration and an estimate
of the time remaining. It executes nothing.

Other useful commands:

```bash
python runners/run_component_ablation.py --official --dry-run    # list what would run, run nothing
python runners/run_component_ablation.py --rerun-failed --mode official
python runners/run_component_ablation.py --verify-only --mode official
python runners/run_component_ablation.py --self-test             # synthetic check of the resume logic
python runners/run_component_ablation.py --smoke                 # 2-seed validation grid
```

### How to analyse

Once `--status` reports 300/300:

```bash
python analysis/component_analysis_ablation/analyze_ablation.py --mode official
```

Writes `tables/`, `figures/` and the LaTeX table. The primary analysis uses all
30 seeds; the pre-specified |ΔU| ≤ 2% sensitivity analysis is reported alongside
it, never in its place.

### How the checkpoint works

| Concern | Mechanism |
|---|---|
| Per-cell state | `raw/official/_state/<cell_id>.json` — `pending` / `running` / `completed` / `failed`, with timestamps, duration, exit status, error message and attempt count |
| Atomicity | state files are written to a temporary file, `fsync`-ed, then `os.replace`-d, so an interrupted write never leaves a half-written state |
| Source of truth | the **artifacts**, not the ledger: a `running` entry with no valid artifacts resolves to pending and is re-executed |
| Integrity before skipping | fingerprint JSON parses and its variant / α / seed / dataset / rounds / clients match this campaign; telemetry CSV holds exactly `rounds × clients` rows |
| Wrong-configuration reuse | each cell records a **campaign fingerprint** — the grid, the hyper-parameters and the SHA-256 of `fedhad_ablation.py` and `ablation_policy.py`. Change any of them and previously completed cells stop counting as done |
| Unverifiable provenance | artifacts with no state entry are **not** silently reused; `--adopt-orphans` validates and adopts them explicitly, marking them `adopted: true` |

The resume logic is covered by `--self-test`, which fabricates artifacts in a
temporary directory and checks all twelve behaviours (interrupted `running` cell,
missing telemetry, truncated telemetry, mismatched seed, foreign campaign
fingerprint, orphan artifacts, failed cells, atomic writes). It runs no training.

> **Note on the 47 pre-existing cells.** A first launch of the battery was stopped
> at 47/300 while the checkpoint mechanism was being written. Those cells were
> produced by exactly this code and configuration; they were validated and adopted
> with `--adopt-orphans`, and their state files carry `adopted: true`. Resume
> therefore starts from 253 remaining cells. To discard them and start from
> scratch, delete `raw/official/`.

## 8. Limitations

- Two α values and one dataset (CIFAR-10). The conclusions are not claimed to transfer to
  FEMNIST or to balanced settings.
- Ten rounds. An arm that is slower to start but better at convergence would be
  mis-ranked; the per-round curves are provided so this is visible.
- Updates and FLOPs are matched only approximately (§4).
- The initial model is fixed across arms, which *removes* a variance source the
  historical campaign contains. Effect sizes here are therefore not directly comparable in
  magnitude to those in the main campaign, only within this folder.
- Wall-clock is recorded but is not a controlled variable: the machine is shared.

## 9. Outputs

```
raw/FedHAD-results-txt-reports/  the training script's own .txt reports of the campaign
                 (formerly code/FedHAD-results/; the code itself is now in
                 algoritmos/component_analysis_ablation/)
raw/smoke/       fingerprints + per-round telemetry (validation only)
raw/official/    fingerprints + per-round telemetry (the campaign)
tables/          budget verification, aggregates, paired tests, .tex
figures/         PDF + PNG at 300 dpi, Okabe-Ito palette
logs/            one log per run, plus the runner log
```

## 10. Results

300/300 cells completed, 0 failed. Fairness verification passed on all 60
`(α, seed)` cells: identical initial weights, partitions, clients, `H_k` and round
count across the five arms. Mean cell duration 182 s.

### Final centralised accuracy (mean ± sd over 30 seeds)

| Arm | α = 0.01 | α = 0.1 |
|---|---|---|
| `full_fedhad` | **0.5599 ± 0.0380** | **0.6872 ± 0.0323** |
| `lr_only_matched` | 0.5585 ± 0.0408 | 0.6868 ± 0.0324 |
| `permuted_allocation` | 0.5519 ± 0.0388 | 0.6842 ± 0.0320 |
| `epoch_only` | 0.5505 ± 0.0478 | 0.6754 ± 0.0342 |
| `fixed_matched` | 0.5404 ± 0.0423 | 0.6796 ± 0.0329 |

### Paired contrasts, all 30 seeds (primary analysis)

Positive favours the first arm. CIs are bootstrap over seeds.

`p` is the raw two-sided Wilcoxon value; `p_H` is Holm-adjusted over the twelve contrasts
(family F4 of the manuscript), and `p_H'` the hierarchical variant in which Q1 forms its own
family. The manuscript reports these values in Table 15, and a contrast is described as
established only when it survives adjustment.

| # | Contrast | α | Δ (pp) | 95% CI | p | p_H | p_H' |
|---|---|---|---|---|---|---|---|
| Q1 | full − fixed | 0.01 | **+1.95** | [+1.23, +2.72] | <0.001 | 0.001 | <0.001 |
| Q1 | full − fixed | 0.1 | **+0.76** | [+0.16, +1.32] | 0.014 | 0.095 | 0.014 |
| Q2 | full − permuted | 0.01 | +0.80 | [+0.30, +1.30] | 0.006 | 0.054 | 0.048 |
| Q2 | full − permuted | 0.1 | +0.30 | [−0.23, +0.83] | 0.328 | 0.985 | 0.985 |
| Q3 | epoch_only − fixed | 0.01 | +1.01 | [+0.22, +1.83] | 0.038 | 0.231 | 0.231 |
| Q3 | epoch_only − fixed | 0.1 | **−0.42** | [−1.08, +0.17] | 0.229 | 0.914 | 0.914 |
| Q4 | lr_only − fixed | 0.01 | **+1.81** | [+1.09, +2.58] | <0.001 | <0.001 | <0.001 |
| Q4 | lr_only − fixed | 0.1 | **+0.71** | [+0.25, +1.21] | 0.010 | 0.084 | 0.073 |
| Q5a | full − epoch_only | 0.01 | +0.94 | [+0.13, +1.84] | 0.096 | 0.481 | 0.481 |
| Q5a | full − epoch_only | 0.1 | +1.18 | [+0.64, +1.71] | <0.001 | 0.003 | 0.003 |
| Q5b | full − lr_only | 0.01 | **+0.14** | [−0.64, +0.93] | 0.926 | 1.000 | 1.000 |
| Q5b | full − lr_only | 0.1 | **+0.04** | [−0.49, +0.55] | 0.655 | 1.000 | 1.000 |

### Answers to the five questions

**Q1. Does FedHAD beat a non-adaptive baseline at a matched update budget? Yes under extreme skew.**
+1.95 pp at α=0.01 (p_H = 0.001), against a control whose update budget matches to a median of
0.00% (52/60 cells within 1%). At α=0.1 the difference is +0.76 pp with a CI clear of zero, but
it survives adjustment only if Q1 is treated as its own pre-declared family (p_H' = 0.014, p_H =
0.095). **H4 is rejected under extreme skew**: there the advantage is not a budget artifact.

**Q2. Does the client→budget association matter? Weak, and confounded.**
+0.80 pp at α=0.01 (raw p=0.006, borderline after adjustment: p_H = 0.054, p_H' = 0.048) but
+0.30 pp at α=0.1 (p=0.328). `permuted_allocation`
also spends a median 1.53% fewer updates than `full_fedhad`, so part of the α=0.01
gap may be compute rather than assignment. In the pre-specified |ΔU| ≤ 2% subset
neither α reaches significance (+0.64, p=0.175; +0.38, p=0.276). **This contrast
does not support a firm claim either way.** The step-preserving permutation of Section 6.11
removes the budget confound (see §11).

**Q3. Isolated contribution of adaptive epochs? Not established.**
+1.01 pp at α=0.01 (raw p=0.038, not established after adjustment: p_H = 0.23), and
**−0.42 pp** at α=0.1 — redistributing epochs by `H_k`
performs slightly *worse* than a flat allocation in the milder heterogeneity
setting, though the CI [−1.08, +0.17] includes zero.

**Q4. Isolated contribution of the adaptive learning rate? Large under extreme skew.**
+1.81 pp at α=0.01 (p_H < 0.001) and +0.71 pp at α=0.1 (raw p = 0.010, not established after
adjustment: p_H = 0.084). This single component recovers **93% and 94%** of FedHAD's total gain
over the control.

**Q5. Is coordinated adaptation worth more than its parts? Not beyond the LR.**
`full_fedhad` clearly beats `epoch_only` (+0.94, +1.18), but does **not** beat
`lr_only_matched`: +0.14 pp [−0.64, +0.93] and +0.04 pp [−0.49, +0.55]. The point
estimates are essentially zero and the intervals do not exclude a benefit of up to about 1 pp;
the null result is not evidence of equivalence. **H2/H3 are supported**: the gain is attributable
to the adaptive learning rate, and adding the adaptive epoch policy on top of it buys nothing
measurable here. The LR-only arm is not an alternative policy, since its budget is the one the
complete policy selected.

### Pre-specified sensitivity, |ΔU| ≤ 2% (33/60 cells)

Reported alongside the primary analysis, never in its place. Effect sizes stay in
the same direction for Q1, Q4 and Q5a, but almost every contrast loses
significance — expected, since n falls to 11 (α=0.01) and 22 (α=0.1). Two cautions:
the subset is **selected on a design property** (how well a permutation happened to
match compute), so it is not a random sample of seeds; and non-significance at
n=11 is a failure to detect, never evidence of equivalence. The one contrast that
strengthens is Q5a at α=0.1 (+1.21, p=0.005).

### Compute matching, as realised

| Arm | ΔU median | ΔU range | \|ΔU\| ≤ 1% |
|---|---|---|---|
| `full_fedhad` | 0.00% | — | 60/60 |
| `epoch_only` | **0.00%** | [0.00%, 0.00%] | 60/60 |
| `fixed_matched` | 0.00% | [−2.63%, +3.44%] | 52/60 |
| `lr_only_matched` | 0.00% | [−2.63%, +3.44%] | 52/60 |
| `permuted_allocation` | −1.53% | [−15.51%, +13.11%] | 25/60 |

The matched-budget contrasts (Q1, Q3, Q4, Q5) rest on the first four arms.
`permuted_allocation` is reported as complementary evidence only.

### Structural note on the control

Across all 60 cells, ρ(n_k, H_k) = −0.778 on average (median −0.900): client size
and heterogeneity are strongly anti-correlated, and that is **structural** to
Dirichlet partitioning with five clients, not an artifact of particular seeds.
Because matching updates requires favouring large clients, an H-blind allocation
could in principle drift towards FedHAD's. **It does not**: ρ(E_fixed, H_k)
averages **+0.073** (median +0.100) against ρ(E_full, H_k) ≈ −0.78. In 10 of 60
cells `E_fixed` differs from `E_full` in only one client; those cells are flagged
in `tables/per_seed_diagnostics_official.csv` rather than removed.

### What this means for the paper

Under extreme skew FedHAD beats a budget-matched, heterogeneity-blind baseline, and the gain
is carried almost entirely by the **learning-rate mapping**; the two mappings together are no
better than the learning-rate mapping alone at the budget the complete policy selected. The
manuscript therefore does not claim a synergistic benefit from coordination: coordination means
only that one signal sets both decisions. The epoch mapping is what selects the training budget
(35% fewer updates than the five-epoch baselines at α = 0.01 and 29% at α = 0.1), and its
accuracy contribution at a fixed budget is not established. §11 summarizes how the later
batteries refined this reading.

### Limitations of these results

- CIFAR-10 only, two α values, ten rounds, five clients. An arm that converges
  more slowly but better would be mis-ranked at round 10; `F4` shows the curves.
- The initial model is fixed across arms, removing a variance source the historical
  campaign contains. Effect sizes here are comparable **within this folder** only.
- Q2 is inconclusive by construction, for the compute reason given above.
- Wall-clock was recorded but the machine is shared, so it is not a controlled
  variable.
- Five seeds at α = 0.01 contain a client without samples and were retained (see §6).

## 11. How the later control batteries refine this analysis

The control batteries of Section 6.11, executed later from the same initial checkpoints
(`results/initial_checkpoints_common/`), sharpened two readings of this folder:

- **The learning-rate effect is a matter of aggregate step size.** A single uniform rate that
  reproduces the aggregate first-order step of the policy is practically equivalent to the
  heterogeneity-dependent rates at α = 0.1 and more accurate by 1.13 pp at α = 0.01
  (pre-registered; `results/lr_uniform_control/`, Table 18). The gain identified in Q4 comes from
  the overall step size the score induces, not from assigning different rates to different
  clients.
- **The allocation matters at a fixed number of updates, but coincides with client size.** A
  permutation of the executed step budgets that preserves total updates exactly, replacing the
  budget-confounded Q2, costs 3.96 pp at α = 0.01 and 1.86 pp at α = 0.1
  (`results/step_preserving_permutation/`, Table 17). In these partitions H_k and client size are
  almost collinear, so this does not separate allocation by heterogeneity from allocation by size.
- **Against a tuned baseline.** A FedProx tuned on separate validation seeds is practically
  equivalent to FedHAD under extreme skew and FedHAD is non-inferior to it under high skew, while
  the tuned configuration uses fewer updates (`results/tuned_controls_fedprox_validation/`,
  Table 19).
