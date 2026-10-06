# Testing the uniform-global-prior assumption in $H_k$

Offline analysis artifact. **No training and no federated simulation is executed.** The script
is `analysis/global_prior_analysis/analyze_prior_assumption.py`; it writes only to this folder.

> **What the manuscript uses from this folder, and what it does not.**
> - **Part 1 (CIFAR-10)** is reported in Section 6.4 as Table 10 (association of four
>   heterogeneity statistics with the raw alignment measure, baselines only).
> - **Part 2 (FEMNIST)** in this folder estimates the global prior from a pool of 2,095 writers,
>   including the held-out evaluation writers. The manuscript does **not** report these numbers.
>   Section 6.8 and Table 12 use a later analysis that reconstructs the prior from the ten
>   participating writers alone, using only their training splits, and also quantifies the
>   learning-rate changes: `analysis/revision_consolidated_tables_figures/analyze_femnist_10_writer_prior.py`,
>   output in `results/revision_consolidated_tables_figures/` (`tables/tab_femnist_10_writer_prior.tex`
>   and `docs/FEMNIST_10_WRITER_PRIOR_ANALYSIS.md`). The two FEMNIST analyses answer the same
>   question with different reference priors, so their numbers differ (for example, Spearman between
>   the CV and JS orderings inside the federation is 0.533 here and 0.841 in Table 12, and the writers
>   whose epoch allocation changes are 3.6 per seed here and 1.87 in Table 12).

## 1. The question

The heterogeneity score used throughout the paper is a normalised coefficient of
variation of a client's per-class sample counts:

$$H_k = \frac{\mathrm{std}(c_k)}{\mathrm{mean}(c_k)} \Big/ \sqrt{K-1}$$

The assumption embedded in the score: it is zero exactly when a client's classes are
*equally* represented, so it measures departure from a **uniform** class prior. In a
naturally federated dataset the global prior is not uniform, and a client that perfectly
mirrors the federation can still be scored as highly heterogeneous. The assumption is not, in
general, true; the manuscript states it explicitly (Sections 1, 2 and 4.2).

## 2. Interpretations declared in advance

Written **before the numbers were produced**, so the conclusion could not be
reverse-engineered from the results.

**Reading A — the CV holds up.** If no alternative metric predicts client drift
materially better than the CV, and re-ranking under a realistic prior does not change
how the federation is treated, the assumption is a *documented simplification*: state it
explicitly, justify it, change nothing in the method.

**Reading B — the CV degrades.** If an alternative predicts drift better, or if
switching the reference distribution reorders clients enough to change the operational
decisions the score drives, the paper must **delimit the scope of applicability** of
$H_k$.

**Decision rule.** Part 1 (CIFAR-10, where drift telemetry exists) decides on
*predictive* grounds; Part 2 (FEMNIST, where it does not) on *operational* grounds. The
two parts are allowed to disagree; if they do, the honest conclusion is a scope
boundary, not a verdict.

## 3. Method

### Data sources (read-only)

| Source | Used for |
|---|---|
| `data_partition_manifest/.../partition_manifest.csv` | class counts; cross-check basis; FEMNIST client writers |
| `data_partition_manifest/.../femnist_heldout_manifest.csv` | FEMNIST held-out writer class counts |
| `drift_diagnostics_10seeds/1.1_diagnostico_drift/drift_telemetry.csv` | logged `H_k`, `cos_sim` |

### The four metrics

All oriented so **higher = more heterogeneous**, matching the drift analysis.

| Key | Definition |
|---|---|
| `cv_norm` | normalised coefficient of variation — the current $H_k$ |
| `entropy_het` | $1 - H(p_k)/\log K$ |
| `js_global` | Jensen–Shannon divergence (bits) vs the **empirical** global prior |
| `tv_global` | total variation ($L_1/2$) vs the same empirical prior |
| `js_uniform` | JS vs the **uniform** prior — isolates the effect of changing the reference |

### (1) Formal validation of the offline reconstruction

The recomputed CV must reproduce the logged `H_k`. This is what guarantees the analysis
measures the same quantity the campaign logged.

| Quantity | Value |
|---|---|
| Clients compared | 150 |
| **Informative cases** | **147** — the 3 M2 clients compare sentinel `0.0` to `0.0` and are *not* evidence |
| Max absolute error | **7.379e-08** |
| Median absolute error | 1.139e-08 |
| Max **relative** error | 9.733e-08 = **0.82× float32 machine epsilon** (1.192e-07) |
| Cases above 2× eps32 | **0 / 147** |

**The residual is float32 representation error, not reconstruction error.** The campaign
computes counts via `torch.bincount(...).float()` — single precision — so the *logged*
value itself carries ~1e-7 of representation error. Every observed disagreement sits
below one float32 ulp.

**Why the guard could not use the manifest.** The manifest records each Dirichlet
client's *full partition*; the logged `H_k` is computed over the *90% training split*
(the telemetry's `n_samples` is exactly `0.9 × n_k`). Recomputing from manifest counts
reproduces `H_k` only to ~2e-3. The Dirichlet counts are therefore regenerated offline
and deterministically by importing the campaign's own partitioning code — partitioning
only, no training.

### (2) $H_k$ is constant when the partition is static

Verified: the within-client span of `H_k` across the 10 rounds and 4 methods is
**exactly 0.000e+00** for all 150 clients. The Dirichlet partition does not change during
training, and `H_k` is a pure function of it. Validating one round therefore validates
**all 6,000 telemetry rows**, not 150.

### (3) The manifest as an independent cross-check — not an identity

The two bases measure *different things*: the manifest the full client partition, the
telemetry the 90% training split. They are compared to show the choice of basis does not
drive conclusions, **not** to assert equality.

| Comparison (excluding the 3 M2 clients) | Value |
|---|---|
| Max abs difference CV(train split) vs CV(full partition) | 5.245e-03 |
| Spearman between the two bases | **0.999846** |

The three M2 clients are excluded from this comparison because their single sample falls
entirely into the validation split, giving CV = 1 on one basis and the sentinel 0 on the
other. Outside them the bases are rank-equivalent to four decimals — a consistency
result, not an identity.

### Statistics

Spearman $\rho$ between each metric and `cos_sim`, per method and per $\alpha$.
Confidence intervals are percentile bootstrap (2000 resamples) **over the ten seeds**,
not over client–round observations, which are not independent within a seed. The three
`(alpha, seed, client_id)` pairs with $n_k = 0$ documented as **M2** —
`(0.01, 46, 1)`, `(0.01, 48, 4)`, `(0.01, 51, 2)` — are excluded from the analysis (not
from the experiment): their logged values are sentinels, not measurements.

## 4. Results

### (4) Part 1 — CIFAR-10 characterises the regime where the premise *holds*

**CIFAR-10's empirical global prior is uniform by construction** — 5,000 images per
class, maximum deviation 0.0. Consequently `js_global` and `js_uniform` are identical to
three decimals in every cell.

> **This dataset cannot test the critique.** The failure mode it describes is undefined
> where the premise is true. Part 1 characterises the regime in which the uniform
> hypothesis is satisfied and asks a narrower question: does a different *functional
> form* predict drift better? It is not evidence that the premise is harmless in general.

### (10) Baselines first, FedHAD separately

Spearman $\rho$ against `cos_sim`. The three baselines come first because there $H_k$ is
purely observational — it does not steer the optimisation.

| Metric | FedAvg $\alpha$=1.0 / 0.1 / 0.01 | FedAvgM | FedProx |
|---|---|---|---|
| `cv_norm` | −0.445 / **−0.908** / −0.961 | −0.469 / −0.923 / −0.967 | −0.455 / −0.911 / −0.970 |
| `entropy_het` | **−0.447** / −0.868 / **−0.963** | −0.469 / −0.883 / −0.968 | −0.456 / −0.872 / −0.972 |
| `js_global` | −0.418 / −0.793 / −0.958 | −0.438 / −0.807 / −0.964 | −0.428 / −0.799 / −0.968 |
| `tv_global` | −0.394 / −0.783 / −0.939 | −0.416 / −0.795 / −0.943 | −0.406 / −0.790 / −0.952 |
| `js_uniform` | −0.418 / −0.793 / −0.958 | −0.438 / −0.807 / −0.964 | −0.428 / −0.799 / −0.968 |

The CV is among the most strongly associated statistics throughout, but it is not uniquely so.
The entropy complement is marginally ahead at $\alpha=0.01$ for every baseline (for FedAvg,
−0.963 against −0.961), by a margin far inside the confidence intervals, while at $\alpha=0.1$
the CV is ahead of the three alternatives. The prior-referenced divergences are consistently
weaker, most visibly at $\alpha=0.1$. **No alternative is consistently more strongly associated
than the CV in the regime where the premise holds, and the CV is not measurably superior to the
entropy complement.** The two share the operational properties that matter (local
computability and a closed-form bound); the manuscript presents the CV as a design choice, not as
a demonstrated improvement over local entropy. All four statistics inherit the client-size and
self-inclusion effect of the raw cosine described in Section 6.4.

**FedHAD is reported separately** (−0.527 / −0.928 / −0.970 for `cv_norm`) because it has
a causal path the baselines lack:

$$H_k \;\rightarrow\; E_k,\ \mathrm{LR} \;\rightarrow\; \text{local update} \;\rightarrow\; \cos\text{-}\mathrm{sim}$$

$H_k$ controls the optimisation it is being correlated against, so the within-FedHAD
correlation is **not a fully independent validation**. Notably FedHAD shows the strongest
$\alpha=1.0$ correlation of all four methods, which is consistent with that feedback.

### (5)(6) Part 2 — FEMNIST, aggregated writer pool

> **Scope.** These numbers describe an **aggregated pool of 2,095 distinct writers**
> (10 campaign clients + 2,085 held-out). They are **not** the campaign's federation,
> which has **10** clients. The federation is analysed separately in §4.4.

| Quantity | Value |
|---|---|
| Empirical global prior, min / max class probability | 0.00271 / 0.05511 (uniform = 0.01613) |
| Ratio, most to least frequent class | **20.4×** |
| Max deviation from uniform | 0.03898 |
| JS(global ‖ uniform) | **0.156** |
| Spearman, CV ranking vs JS ranking | **0.691** |
| Pool writers changing $H_k$ tercile band | 756 / 2095 (36.1%) |
| Pool writers whose $E_k$ would differ | 504 / 2095 (24.1%) |

**The pooled global prior is far from uniform: here the uniform-prior premise does not hold.**

**What this does not say.** It does **not** show that JS predicts drift, nor that it
improves accuracy. There is no FEMNIST drift telemetry and no model was trained with JS.
It shows the two metrics *disagree*, and that the disagreement is operationally material.

### (8) The pooled figure is contingent on the normalisation

JS and CV live on different scales, so putting them on a common footing is a choice, and
the headline moves with it:

| Normalisation | Pool writers with different $E_k$ | Total epochs | Budget |
|---|---:|---:|---|
| **Quantile matching (used)** | **24.1%** | 9226 vs 9226 | **preserved** |
| Min–max | 54.7% | 10218 vs 9226 | inflated |
| Raw JS | 57.6% | 10400 vs 9226 | inflated |

Only quantile matching isolates re-ranking; the other two inflate the budget, conflating
"reorder the clients" with "simply train more". **The 24.1% is meaningful only with the
qualifier "under budget-preserving quantile matching over the pool".**

Note also that this pooled preservation is **global, not per scenario**: preserving a
multiset over 2,095 writers does not preserve it within any subset. That is precisely why
the federation needs its own analysis.

### (7) Part 2b — counterfactual rank remapping *within the campaign federation*

Over the **10 writers the campaign actually uses**, on all **30 seeds**. The exact
multiset of $E_k$ values assigned by the CV is preserved and redistributed by the JS
ranking (least divergent writer receives the largest budget, mirroring the direction of
FedHAD's rule).

> Called **within-federation budget-preserving rank remapping** (equivalently,
> **counterfactual rank remapping**). It is **not** "FedHAD-JS". Nothing was trained; no
> claim is made about drift or accuracy.

| Quantity | Value |
|---|---|
| Seeds evaluated | 30 |
| Budget preserved | **30 / 30 seeds** — 43 epochs before, 43 after |
| Writers reallocated per seed | mean **3.6 of 10** (min 2, max 4) |
| Spearman CV vs JS *inside the federation* | mean **0.533** (range 0.224–0.709) |
| Writers reallocated in ≥ 1 seed | 8 of 10 |

Transition matrix over the 300 (writer, seed) pairs:

| $E_k$ (CV) → $E_k$ (remapped) | Pairs | Outcome |
|---|---:|---|
| 4 → 4 | 156 | unchanged |
| 4 → 5 | **54** | reallocated |
| 5 → 4 | **54** | reallocated |
| 5 → 5 | 36 | unchanged |

The gains and losses balance exactly, as budget preservation requires. One writer
(`f0900_42`) is reallocated in **all 30 seeds**.

**The disagreement is sharper on the real federation than on the pool** (ρ = 0.533 vs
0.691), so the pooled figure is, if anything, conservative.

### (12) Basis, deduplication and case counts

- **Mixed basis in the FEMNIST pool.** Client writers enter with their 90% training split;
  held-out writers with 100% of their data. That is 10 of 2,095 rows on a different basis
  — negligible for the prior estimate (0.48% of the sample), but it means those 10 rows
  are not strictly comparable to the other 2,085 in the pooled ranking. Part 2b avoids
  this entirely: it uses only client writers, all on the training-split basis.
- **Deduplication of held-out writers.** Verified lossless: a held-out writer's class
  counts are identical across all 30 scenarios (0 divergences), so keeping the first
  occurrence discards nothing.
- **Guard case count.** 147 informative of 150; the 3 M2 clients pass trivially.
- **Pool vs federation.** 2,095 writers = aggregated pool for population-level questions;
  10 writers = the campaign's federation for deployment-level questions. Never conflated.

### (13) Reproducibility note (implementation, not a result)

`dirichlet_split_noniid(dataset, n_clients, alpha, seed=SEED)` binds `seed` as a **default
argument evaluated at import time**, and `load_data()` never passes it. Setting
`module.SEED` after import therefore silently regenerates the *seed-42* partition. The
campaign is unaffected — it sets `FL_SEED` before importing — but offline reuse must
import once per seed, which is what this script does. This is an implementation and
reproducibility note about reusing the code, **not a scientific finding** and not a defect
in the campaign's results.

## 5. (14) Risk register

Full text in `tables/T9_risk_register.csv`.

| Risk | Status |
|---|---|
| Leakage (predictive) | **resolved** — no held-out target predicted from itself; CIFAR-10 prior is definitional, FEMNIST prior pooled over 2,095 writers (clients = 0.48%) |
| Circularity (FedHAD) | **residual** — inherent to the method; mitigated by reporting the three baselines first and labelling FedHAD as non-independent |
| Scale dependence of the pooled headline | **resolved** — quantile matching is the only budget-preserving option; all three sensitivities reported |
| Budget invariance claimed per scenario | **resolved** — pooled mapping is global-only; the federation gets its own exactly-preserving remapping |
| Scope: pool vs campaign federation | **resolved** — both reported and labelled |
| Mixed basis in the FEMNIST pool | **documentation** — 10 of 2,095 rows; Part 2b is unaffected |
| Guard passing on sentinel values | **documentation** — informative *n* is 147, not 150 |
| Deduplication of held-out writers | **resolved** — verified lossless |
| SEED as a default argument | **documentation** — reproducibility note |
| Overinterpretation of Part 2 | **resolved** — applicability boundary, not JS superiority |

Six resolved, one residual (structural, disclosed), three documentation-only.

## 6. (15) Conclusion

### What this analysis establishes

1. The offline reconstruction reproduces the campaign's $H_k$ to within one float32 ulp
   over 147 informative clients, and — because $H_k$ is exactly constant across rounds and
   methods — that validation extends to all 6,000 telemetry rows.
2. **CIFAR-10's global prior is uniform by construction.** The uniform-reference premise is
   *satisfied* on the Dirichlet benchmarks; in that regime no alternative is consistently more
   strongly associated with the raw alignment measure than the CV, the entropy complement is
   marginally ahead under extreme skew, and the prior-referenced divergences are weaker.
3. **FEMNIST's global prior is not uniform** (20.4× skew over the pool, JS vs uniform = 0.156).
   With the pooled prior, the CV and a prior-referenced divergence rank writers differently
   (ρ = 0.691 over the pool, **0.533 inside the campaign federation**).
4. With the pooled prior, the disagreement is **operationally material**: under a
   budget-preserving remapping, **3.6 of the 10 campaign client writers** change their
   local-epoch budget per seed, with the total held constant by construction. With the prior of
   the ten participating writers, which is what the manuscript reports, the effect is smaller:
   1.87 writers per seed change epochs, and the learning rate changes by 0.96% on average
   (Table 12).

### What it cannot establish without new training

- Whether JS (or any prior-referenced metric) **predicts drift better** on a naturally
  federated dataset. There is no FEMNIST drift telemetry.
- Whether reallocating epochs by JS would **improve accuracy, convergence or cost**. No
  model was trained with it; Part 2b is a counterfactual arithmetic exercise.
- Whether the effect generalises beyond FEMNIST to other natural federations.

Answering any of these requires running the drift instrumentation on FEMNIST and, for the
accuracy question, a full training campaign with a JS-based score.

### How the manuscript uses this analysis

As a scoping and disclosure matter. The manuscript (a) declares the uniform-reference
assumption, (b) shows that it is satisfied by construction on the Dirichlet benchmarks, where
the CIFAR-10 comparison of Table 10 is made, and (c) bounds its operational consequence on the
FEMNIST federation with the ten-writer analysis of Table 12. It does not claim that JS is
better, that the CV is wrong, or that accuracy would change: no alternative controller was
trained, and the analysis does not allow one to say whether a prior-aware controller would
improve or degrade accuracy. Prior-aware formulations are presented only as a well-motivated
extension.

### Can this folder be considered closed?

**Yes.** Both verifications pass, every identified risk is resolved, residual-and-disclosed,
or documentation-only, and the remaining open questions are explicitly out of scope for an
offline analysis. No further offline work is required. Reopening is warranted only if the
FEMNIST drift experiment is run, which would turn Part 2 from a disagreement result into a
predictive comparison.

## 7. Reproducing

```bash
./env_flwr_pt/bin/python analysis/global_prior_analysis/analyze_prior_assumption.py --bootstrap 2000
```

Run from the repository root.

Flags: `--bootstrap N`, `--skip-regen` (manifest basis only; the 1e-6 guard cannot pass in
that mode, by construction). Exit status is non-zero if the guard fails.

## 8. Outputs

```
tables/
  T1_guard_validation.csv               per-client recomputed CV vs logged H_k, abs/rel error, M2 flag
  T1b_basis_cross_check.csv             training split vs full partition, per client
  T2_dirichlet_spearman.csv             rho + 95% CI, per method x alpha x metric
  T3_femnist_prior.csv                  FEMNIST empirical prior, per class
  T4_femnist_ranking_disagreement.csv   pooled Part 2 headline numbers
  T5_femnist_epoch_impact.csv           pooled E_k(CV) x E_k(JS) contingency
  T6_normalisation_sensitivity.csv      quantile / min-max / raw JS
  T7_campaign_federation_remap.csv      per-seed within-federation remapping
  T8_federation_transition_matrix.csv   E_k transitions over 300 (writer, seed) pairs
  T9_risk_register.csv                  every risk, classified
  prior_analysis.tex                    main booktabs table
  federation_remap.tex                  Part 2b booktabs table
data/
  dirichlet_client_metrics.csv          four metrics per (alpha, seed, client)
  femnist_writer_metrics.csv            four metrics per pooled writer
  campaign_federation_remap.csv         per (writer, seed) for the 10 campaign clients
figures/                                PDF + PNG at 300 dpi, Okabe-Ito palette
  F1_global_prior_vs_uniform            CIFAR-10 vs FEMNIST prior against uniform
  F2_dirichlet_spearman_by_metric       rho with bootstrap CIs
  F3_femnist_cv_vs_js                   per-writer CV vs JS, extreme case marked
  F4_femnist_epoch_allocation           pooled epoch contingency
  F5_within_federation_remap            reallocation inside the campaign federation
```
