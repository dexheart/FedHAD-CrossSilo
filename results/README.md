# Results of the FedHAD experiments

Every raw result and every derived table behind the manuscript is under this folder; it holds
data only. The code that produced it is in `algoritmos/` (training) and `analysis/` (tables,
figures, numbers), and the runners are in `runners/`. Each sub-folder is named after
the experiment it holds. Section, table and figure numbers refer to
the revised manuscript (`PaperJNCA_Revision/.../Revised manuscript.tex`).

All runs use CIFAR-10, MNIST, FashionMNIST or FEMNIST with 5 clients (unless stated otherwise) and
full participation. The main campaign and the component analysis use seeds 42–71; the drift
diagnostic campaign uses seeds 42–51; the tuned controls use validation seeds 72–76 and, at
α = 0.01, evaluation seeds up to 166. The datasets themselves are not included; they are downloaded
by torchvision / flwr-datasets into `data/`.

## Map: folder → experiment → manuscript

| Folder | What it holds | Manuscript |
|---|---|---|
| `main_campaign_default_alpha_0_5/` | Main campaign, every method (FedAvg, FedAvgM, FedProx, FedNova, SCAFFOLD, FedHAD), default Dirichlet α = 0.5 for the tests that do not vary α. FedAvg, FedAvgM, FedProx and FedHAD ran on a workstation (NVIDIA RTX A1000), except the drift-diagnostic runs and the replacement runs listed in Section 5.8 of the manuscript; FedNova and SCAFFOLD ran on the server (NVIDIA A40). One `.txt` report per run. Destination of `runners/run_main_campaign.py`. | §6.2–6.9, Tables 7, 8, 11, 13; Figs. 4–15 |
| `main_campaign_default_alpha_0_01/` | Same tests repeated with default α = 0.01 (extreme skew), FedAvg/FedAvgM/FedProx/FedHAD, including the 50-round runs at α = 0.01 and on FEMNIST. Server (NVIDIA A40). `resultados_cluster.zip` is a byte-identical archive of the 3,630 reports of this folder (folders named `<Method>-results_Cluster` inside the archive); `analysis/revision_consolidated_tables_figures/generate_femnist_main_figures.py` reads it. The manuscript uses the 50-round blocks of this campaign. | §6.7 (α = 0.01 and FEMNIST blocks), Table 11, Fig. 13 (c, d) |
| `main_campaign_summaries/` | Per-test CSVs consolidated from the main campaign (`analise_bruta_*`, `analise_resumo_*`), the paired Wilcoxon table `wilcoxon_results.csv` and the figures of tests 1–8 (`figures/`), all written by `analysis/main_campaign/`; `sensitivity_grid_reexecuted/` holds Figure 16 and its grid summary with the four re-executed runs, written by `analysis/audit/a12_sensitivity_grid_reinstalled.py` (it supersedes `figures/test6/`, which uses the original grid). | inputs of the figures |
| `data_partition_manifest/` | Reconstruction of every client partition (index sets, label counts, hashes) and the FEMNIST held-out writers; proof that all methods saw identical partitions per seed. | §5.3, §5.6 |
| `seed_reexecutions_corrupted_runs/` | Complete re-executions, on the server, of six runs left incomplete by failures of the workstation (four of the sensitivity grid, FedProx seed 66 and FedHAD seed 53 of the ten-round FEMNIST federation), with the same configuration and seed. The incomplete originals were discarded. The instrumentation is in `algoritmos/seed_reexecution_probes/`. | §5.8, §6.12 |
| `drift_diagnostics_10seeds/` | Separate 10-seed campaign logging client-update alignment (telemetry, analysis, tables). | §6.4, Table 9 |
| `global_prior_analysis/` | Offline comparison of the heterogeneity statistic with a global-prior-aware alternative and with entropy, Jensen–Shannon and total-variation statistics. | §6.4 (Table 10), §6.8 |
| `component_analysis_ablation/` | Controlled component analysis: Full, fixed matched, epoch-only, LR-only, permuted; common checksummed initialization; raw fingerprints and telemetry, the training script's own `.txt` reports (`raw/FedHAD-results-txt-reports/`), tables and figures. | §6.10, Tables 14–15 |
| `revision_consolidated_tables_figures/` | Consolidation of all campaigns into tables and figures (master summary, FedNova comparison, long horizon, FEMNIST, ten-writer global prior). | Tables 3, 7, 11 (CIFAR-10 blocks), 12, 13 (FedNova columns); Figs. 13 (c, d), 15 |
| `initial_checkpoints_common/` | One initial model per seed (42–166) with SHA-256 manifest, shared by every arm of the control batteries below. | §5.6, §5.7 |
| `step_preserving_permutation/` | Control battery: optimizer-step budgets of FedHAD permuted among clients, total updates preserved exactly. | §6.11, Table 17 |
| `tuned_controls_fedprox_validation/` | Control battery: FedProx tuned on validation seeds 72–76 (`tune/`), frozen choice (`frozen_config.json`), pre-registered evaluation against FedHAD (`evaluate/`, `evaluate_analysis/`). | §6.11, Tables 16, 19 |
| `lr_uniform_control/` | Pre-registered control battery: heterogeneity-dependent learning rates vs one uniform rate with the same aggregate step. | §6.11, Table 18 |
| `global_gradient_diagnostic/` | Pre-registered diagnostic against the gradient of the rest of the federation, equal-size partitions, removed epochs, class-sensitive metrics. | §6.11, Table 20 |
| `execution_logs/` | Coordinator logs of the main campaign and the progress log of `runners/run_tuned_controls_and_step_permutation.py`. | — |

## Test folders inside the two main campaigns

Inside `main_campaign_default_alpha_*/<Method>-results/` the test folders keep the names used
when they were executed (read by the parsers, so they are not renamed):

| Folder | Experiment | Manuscript |
|---|---|---|
| `test1_convergencia` | Base operating point: accuracy and loss over rounds | §6.2, Figs. 4–5 |
| `test2_robustez_alpha` | Heterogeneity sweep, α ∈ {1.0, 0.5, 0.1, 0.01} | §6.3, Table 8, Figs. 6–8 |
| `test3_comunicacao` | Artificial communication delay | §6.6, Fig. 11 |
| `test4_clientes` | Client scaling (3, 5, 10 clients) | §6.6, Fig. 12 |
| `test5_ablacao` | Historical 8-arm ablation (exploratory; superseded by `component_analysis_ablation/`) | — |
| `test6_calibracao` | 3 × 3 grid of the decay factors | §6.12, Table 21, Fig. 16 |
| `test7_plato` | Extended horizon, 50 rounds (α = 0.5 in the α = 0.5 campaign, α = 0.01 in the α = 0.01 campaign) | §6.7, Table 11, Fig. 13 |
| `test8_femnist` | FEMNIST natural federation (10 writers): 10 rounds in the α = 0.5 campaign, 50 rounds in the α = 0.01 campaign | §6.8 (10 rounds), Figs. 14–15; §6.7, Table 11 (50 rounds) |
| `1.1_diagnostico_drift` | Reports of the 10-seed drift diagnostic campaign, FedAvg, FedAvgM, FedProx and FedHAD (α = 0.5 campaign only; byte-identical copies are in `drift_diagnostics_10seeds/1.1_diagnostico_drift/raw/`) | §6.4 |

Report file names state every parameter of the run, e.g.
`FedHAD__CIFAR10__Mode-STANDARD__Cfg-E1L1F1__Setup-5__Clients-5__Rounds-10__Delay-0_05__Alpha-0_5__Seed-42.txt`.

## What can be verified, and where

| Item | Location |
|---|---|
| Data partitions identical across methods | `data_partition_manifest/artifacts/partitions/` |
| Initialization checksums | `initial_checkpoints_common/checkpoints_manifest.json`; `fingerprint.json` of every control run (`initial_checkpoint.sha256`); `component_analysis_ablation/raw/official/fingerprint__*.json` |
| Raw per-seed results | `.txt` reports of the main campaigns; `fingerprint.json` + `telemetry.csv` per run of the control batteries |
| Client-level telemetry (steps, examples, FLOPs per client and round) | `telemetry.csv` in every control run; `drift_diagnostics_10seeds/` |
| Pre-registrations (hashed before execution) | `lr_uniform_control/preregistration.json`, `tuned_controls_fedprox_validation/evaluate_preregistration.json`, `global_gradient_diagnostic/preregistration.json` (each with its `.sha256`) |
| Analysis scripts | `analysis/` (one folder per experiment, plus `analysis/audit/`); map of every table and figure in `analysis/README.md` |
| Number in the paper → source file | `PaperJNCA_Revision/CHANGES_V3.md`, section 4 |

## Reproducing the analyses (no training)

See `analysis/README.md`: it lists, for every table and figure of the manuscript, the script that
produces it from this folder and the file it writes, and the order in which to run them.

## Notes on the reorganisation (for provenance)

The folders were renamed after the experiments, only to make the archive readable. No raw file
was edited. Old → new names:

| Old | New |
|---|---|
| `resultados/Results-base-alpha-0_5` | `main_campaign_default_alpha_0_5` |
| `resultados/Results-base-alpha-0_01` | `main_campaign_default_alpha_0_01` |
| `analises/` | `main_campaign_summaries` |
| `logs/` | `execution_logs` |
| `results_revision/results_revision_ablation` | `component_analysis_ablation` |
| `results_revision/results_revision_consolidated` | `revision_consolidated_tables_figures` |
| `results_revision/results_revision_drift` | `drift_diagnostics_10seeds` |
| `results_revision/results_revision_package` | `revision_supplementary_package` |
| `results_revision/results_revision_partition_manifest` | `data_partition_manifest` |
| `results_revision/results_revision_prior` | `global_prior_analysis` |
| `results_revision/results_revision_seed_reexecutions` | `seed_reexecutions_corrupted_runs` |
| `results_reviewer_r2_checkpoints` | `initial_checkpoints_common` |
| `results_reviewer_r2_step_permutation` | `step_preserving_permutation` |
| `results_reviewer_r2_tuned_controls` | `tuned_controls_fedprox_validation` |
| `results/global_gradient_diag` | `global_gradient_diagnostic` |

Consequences, stated so that nothing looks inconsistent:

- Raw artifacts written before the move (pre-registrations, `fingerprint.json`, the `path` field of
  `checkpoints_manifest.json`, run logs) still mention the old folder names; they are historical
  records and were not rewritten.
- `main_campaign_default_alpha_0_5/FedAvgM-results/` contained a nested copy of itself
  (`FedAvgM-results/FedAvgM-results/`, 810 reports). Every one of the 810 files was verified to be
  byte-identical (SHA-256) to the file at the same relative path one level up, with no file
  unique to the copy, and the copy was removed (it remains in the git history). The analysis
  and audit scripts already discarded it; re-running them after the removal gives the same
  numbers, and only their counts of skipped duplicates changed (810 and 270 to 0).
- All scripts were updated to the new paths. Re-running them reproduces the committed numbers;
  the only differences are folder names inside path columns and column widths.
- `tuned_controls_fedprox_validation/code_relocation.json` declares the path-only edit of
  the tuned-control analysis script after its pre-registration (registered and
  current hashes, git commit of the registered version, exact diff); the analysis accepts the
  edit only if it matches that record.
- The training scripts whose hashes enter a campaign fingerprint (`fedhad_r2.py`, `r2_policy.py`,
  `drift_telemetry.py`, `fedhad_lru.py`, `fedhad_gg.py`, the component-analysis code) were not
  edited, so the runner of the step-permutation and tuned-control batteries still recognises every
completed run. The runners of the LR-uniform and
  global-gradient batteries, whose policy modules had to be pointed to the new paths, refuse to
  resume (their pre-registrations no longer match the code); both batteries are complete and are
  not meant to be resumed.

### Second reorganisation: code out of the root and out of `results/`

Code was moved so that `results/` holds data only and every runner sits in `runners/` under the
name of its experiment. No raw file was edited, and every analysis was re-run after the move (see
`analysis/README.md`). Old → new:

| Old | New |
|---|---|
| `run_experiments.py` | `runners/run_main_campaign.py` |
| `run_experiments_revision_drift.py` | `runners/run_drift_diagnostics.py` |
| `run_experiments_reviewer_r2_controls.py` | `runners/run_tuned_controls_and_step_permutation.py` |
| `run_r2_experiment2_definitivo.sh` | `runners/run_tuned_controls_evaluation.sh` |
| `run_global_gradient_diag.py` | `runners/run_global_gradient_diagnostic.py` |
| `results/component_analysis_ablation/run_ablation.py` | `runners/run_component_ablation.py` |
| `reviewer_r2_controls/code/`, `config_r2.py` | `algoritmos/tuned_controls_and_step_permutation/` (`config_tuned_controls.py`) |
| `lr_uniform_control/code/`, `config_lru.py`, `margins.json` | `algoritmos/lr_uniform_control/` |
| `global_gradient_diag/code/`, `config_gg.py`, `margins.json` | `algoritmos/global_gradient_diagnostic/` |
| `results/component_analysis_ablation/code/`, `config.py` | `algoritmos/component_analysis_ablation/` (`config_ablation.py`) |
| `results/component_analysis_ablation/code/FedHAD-results/` | `results/component_analysis_ablation/raw/FedHAD-results-txt-reports/` |
| `results/seed_reexecutions_corrupted_runs/instrumentation/` | `algoritmos/seed_reexecution_probes/` |
| `reviewer_r2_controls/analyze_r2_controls.py` | `analysis/tuned_controls_and_step_permutation/analyze_step_permutation_and_tuning.py` |
| `reviewer_r2_controls/analyze_tuned_evaluate.py` | `analysis/tuned_controls_and_step_permutation/analyze_tuned_evaluate.py` |
| `lr_uniform_control/analyze_lr_uniform_control.py` | `analysis/lr_uniform_control/` |
| `global_gradient_diag/analyze_global_gradient.py` | `analysis/global_gradient_diagnostic/` |
| `results/*/analyze_*.py`, `make_partition_manifest.py`, `analise/analise_1_1.py` | `analysis/<same folder name>/` |
| `results/revision_consolidated_tables_figures/scripts/` | `analysis/revision_consolidated_tables_figures/` |
| `audit/` | `analysis/audit/` |
| `scripts/analisar_resultados.py`, `figuras_artigo.py`, `limpar_corrompidos.py` | `analysis/main_campaign/` |
| `*/tests/test_*.py` | `sanity_tests/test_<experiment>.py` |
| `results/execution_logs/r2_controls_progress.jsonl` | `results/execution_logs/tuned_controls_and_step_permutation_progress.jsonl` |

Removed as obsolete (still in the git history): `scripts/dashboard.py`, `scripts/tabelas_artigo.py`
and their HTML outputs (summaries that no table of the paper uses), `scripts/figuras_test3_hardware.py`
(figures of an earlier submission), the root `figs/` folder (an earlier rendering of the main-campaign
figures; the figures are now regenerated in `main_campaign_summaries/figures/`) and
`revision_supplementary_package/` (the figure/table package of the first revision round, not used by
the current manuscript).

Raw records keep the names they were written with: run fingerprints still name the runner
`run_experiments_reviewer_r2_controls.py`, the step-permutation and tuned-control runs write their
reports under `report/r2/`, and the frozen training code keeps its file names (`fedhad_r2.py`,
`r2_policy.py`), because those names and their SHA-256 are part of the recorded fingerprints.
