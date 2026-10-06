# Analysis code: from `results/` to the tables, figures and numbers of the paper

Every script in this folder only **reads** `results/` and writes derived files (CSV, LaTeX,
figures, reports). None of them trains a model. Run them from the repository root with the
project environment (`env_flwr_pt/bin/python ...`). Folder names mirror the folders of
`results/`.

| Folder | What it does | Writes to |
|---|---|---|
| `main_campaign/` | Parses every `.txt` report of the main campaign (`analisar_resultados.py`), draws the figures of tests 1–8 and the paired Wilcoxon table (`figuras_artigo.py`); `limpar_corrompidos.py` lists/removes crashed runs (FLOPs = 0) before a re-run | `results/main_campaign_summaries/` |
| `audit/` | Independent recomputation of the numbers of the manuscript, one script per question (`a01`–`a15`; `common.py` holds its own report parser) | `analysis/audit/out/` |
| `revision_consolidated_tables_figures/` | Consolidated tables and figures: summary across campaigns, FedNova comparison, 50-round study, FEMNIST | `results/revision_consolidated_tables_figures/` |
| `component_analysis_ablation/` | Contrasts of the controlled component analysis; size/heterogeneity analysis of the partitions | `results/component_analysis_ablation/` |
| `tuned_controls_and_step_permutation/` | Step-preserving permutation and tuning phase (`analyze_step_permutation_and_tuning.py`); pre-registered evaluation of the tuned FedProx (`analyze_tuned_evaluate.py`) | `results/step_preserving_permutation/analysis/`, `results/tuned_controls_fedprox_validation/evaluate_analysis/` |
| `lr_uniform_control/` | Pre-registered analysis of the uniform-learning-rate control | `results/lr_uniform_control/analysis/` |
| `global_gradient_diagnostic/` | Pre-registered analysis of the global-gradient diagnostic | `results/global_gradient_diagnostic/analysis/` |
| `drift_diagnostics_10seeds/` | Tables and figures of the 10-seed drift campaign | `results/drift_diagnostics_10seeds/` |
| `global_prior_analysis/` | Heterogeneity statistic vs global-prior-aware, entropy, Jensen–Shannon and total-variation alternatives | `results/global_prior_analysis/` |
| `data_partition_manifest/` | Rebuilds every client partition (index sets, label counts, hashes) and the FEMNIST held-out writers | `results/data_partition_manifest/` |

The three pre-registered analyses check their pre-registration (SHA-256) before computing
anything and refuse to run if it, the frozen configuration or the analysis code changed. The
tuned-control analysis was moved twice after registration (paths only); both moves are declared,
with diffs, in `results/tuned_controls_fedprox_validation/code_relocation.json`, and the check
accepts exactly those.

## Manuscript item → script → file

Labels are the `\label{}` keys of `PaperJNCA_Revision/.../Revised manuscript.tex`.

| Manuscript item | Script | Output used |
|---|---|---|
| `tab:architectures` (MACs, parameters, FLOPs per sample) | `audit/a01_flops.py` | `audit/out/a01_per_sample_flops.csv` |
| `tab:femnist_composition` | `revision_consolidated_tables_figures/consolidate.py` (from the partition manifest built by `data_partition_manifest/make_partition_manifest.py`) | `results/revision_consolidated_tables_figures/data/blockH_femnist_writers.csv` |
| `tab:headline` | `revision_consolidated_tables_figures/consolidate.py` | `results/revision_consolidated_tables_figures/data/master_results_summary.csv`, `blockB_heterogeneity.csv`, `blockD_long_horizon.csv` |
| `fig:convergence`, `fig:loss` (test 1) | `main_campaign/figuras_artigo.py` | `results/main_campaign_summaries/figures/test1/` |
| `fig:alpha`, `fig:conv_extreme` (test 2) | `main_campaign/figuras_artigo.py` | `results/main_campaign_summaries/figures/test2/` |
| `tab:wilcoxon_cifar` (family F1, Holm) | `audit/a06_stats_holm.py` → `audit/a10_v3_numbers.py` | `audit/out/a06_holm_families.csv`, `audit/out/a10_F1_table.csv` |
| `fig:std_alpha`, `fig:scatter`, `fig:flops2target` | `main_campaign/figuras_artigo.py` | `results/main_campaign_summaries/figures/efficiency/` |
| `tab:proxy` | `audit/a02_drift_heterogeneity.py` → `audit/a10_v3_numbers.py` | `audit/out/a02_*.csv`, `audit/out/a10_proxy_extended.csv` |
| `tab:prior-metrics` (Table 10) | `global_prior_analysis/analyze_prior_assumption.py` (Part 1, CIFAR-10; its FEMNIST part uses a pooled prior and is not reported in the manuscript) | `results/global_prior_analysis/tables/` |
| `fig:time` (test 3) | `main_campaign/figuras_artigo.py`; check in `audit/a13_wallclock_delay.py` | `results/main_campaign_summaries/figures/test3/`, `audit/out/a13_*` |
| `fig:clients` (test 4) | `main_campaign/figuras_artigo.py` | `results/main_campaign_summaries/figures/test4/` |
| `tab:long_horizon_alpha_comparison`, CIFAR-10 blocks | `revision_consolidated_tables_figures/generate_long_horizon_alpha_comparison.py`; Holm p-values from `audit/a06_stats_holm.py` | `results/revision_consolidated_tables_figures/tab_long_horizon_alpha_comparison.tex` |
| `tab:long_horizon_alpha_comparison`, FEMNIST block (50 rounds) | `audit/a06_stats_holm.py` (paired tests, family F3) and `audit/a08_long_horizon_fig13.py` (means, standard deviations and TFLOPs); the generator of the CIFAR-10 blocks does not produce these rows | `audit/out/a06_holm_families.csv`, `audit/out/a08_long_horizon_inventory.csv` |
| `fig:long_horizon*` (Figure 13) | all four panels: `revision_consolidated_tables_figures/generate_long_horizon_alpha_comparison.py` (the α = 0.5 curves are also drawn by `main_campaign/figuras_artigo.py`, test7); loss panels checked by `audit/a08_long_horizon_fig13.py` | `results/revision_consolidated_tables_figures/figures/long_horizon/`, `results/main_campaign_summaries/figures/test7/` |
| `fig:femnist` (Figure 14, test 8) | `main_campaign/figuras_artigo.py` | `results/main_campaign_summaries/figures/test8/accuracy_femnist.*` |
| `fig:femnist_compute` (Figure 15, 10 rounds) | `revision_consolidated_tables_figures/generate_femnist_alpha_0_5_figures.py`; compute accounting in `revision_consolidated_tables_figures/femnist_compute_accounting.py` | `results/revision_consolidated_tables_figures/figures/femnist_main_base_alpha_0_5/01_pareto_final_accuracy_vs_tflops.*` |
| `tab:femnist_10_writer_prior` | `revision_consolidated_tables_figures/analyze_femnist_10_writer_prior.py`, `audit/a07_global_prior.py`, `audit/a10_v3_numbers.py` | `results/revision_consolidated_tables_figures/tables/tab_femnist_10_writer_prior.tex`, `audit/out/a10_prior10_per_seed.csv` |
| `tab:fednova_comparison` (FedNova columns) | `revision_consolidated_tables_figures/generate_fedhad_vs_fednova.py`, checked by `audit_fedhad_vs_fednova.py` | `results/revision_consolidated_tables_figures/tables/fedhad_vs_fednova_*` |
| `tab:fednova_comparison` (SCAFFOLD columns) and the SCAFFOLD numbers in Section 6.9 | `audit/a15_scaffold_comparison.py` | `audit/out/a15_*` |
| `tab:ablation_budget` | `audit/a03_ablation.py` (budgets), `audit/a01_flops.py` | `audit/out/a03_arms_absolute.csv`, `a03_per_run_budget.csv` |
| `tab:ablation` | `component_analysis_ablation/analyze_ablation.py`; Holm from `audit/a06_stats_holm.py` | `results/component_analysis_ablation/tables/contrasts_official.csv` |
| Sensitivity of the component analysis without empty clients (Section 6.10) | `audit/a03b_empty_clients.py`, `audit/a11_ablation_sensitivity_empty.py` | `audit/out/a03b_*`, `audit/out/a11_*` |
| `tab:revision_validation` | `tuned_controls_and_step_permutation/analyze_step_permutation_and_tuning.py` → `audit/a14_revision_controls_tables.py` | `audit/out/a14_validation.tex` |
| `tab:step_permutation` | `tuned_controls_and_step_permutation/analyze_step_permutation_and_tuning.py` → `audit/a14_revision_controls_tables.py` | `audit/out/a14_step_permutation.tex` |
| `tab:lr_uniform` | `lr_uniform_control/analyze_lr_uniform_control.py` → `audit/a14_revision_controls_tables.py` | `audit/out/a14_lr_uniform.tex` |
| `tab:tuned_controls` | `tuned_controls_and_step_permutation/analyze_tuned_evaluate.py` → `audit/a14_revision_controls_tables.py` | `audit/out/a14_tuned.tex` |
| `tab:global_gradient` | `global_gradient_diagnostic/analyze_global_gradient.py` → `audit/a14_revision_controls_tables.py` | `audit/out/a14_global_gradient.tex` |
| `fig:sensitivity`, `tab:sensitivity` | `audit/a12_sensitivity_grid_reinstalled.py` (draws with the routine of `main_campaign/figuras_artigo.py`) | `audit/out/a12_grid_*.csv`; the figure is written into the manuscript's `figs/test6/`, with a copy in `results/main_campaign_summaries/sensitivity_grid_reexecuted/` |
| Numbers of Section 3 (objective reweighting by local steps) | `audit/a09_epochs_vs_steps.py` | `audit/out/a09_summary.txt` |
| Learning rates and epochs of every method (Section 5) | `audit/a04_lr_survey.py` | `audit/out/a04_*` |
| Report integrity, hardware, initialisation evidence (Sections 5.6, 5.8) | `audit/a05_integrity_init.py` | `audit/out/a05_*` |

`tab:related_work`, `tab:base_config`, `tab:factors` and `tab:bibliotecas` describe the method and
the setup and are not computed from results. `fig:sensitivity` is written directly into
the manuscript folder by `audit/a12_sensitivity_grid_reinstalled.py`, which also keeps a copy in
`results/main_campaign_summaries/sensitivity_grid_reexecuted/`. That copy supersedes
`results/main_campaign_summaries/figures/test6/accuracy_by_decay_config.*`, which
`main_campaign/figuras_artigo.py` draws from the original grid, including the four incomplete runs.
The other figures were copied into the manuscript folder from the files listed above, which are
regenerated with the same data and plotting routines. The copies of the main-campaign figures of
tests 1 to 4 and of the efficiency figures in the manuscript are a rendering with larger fonts,
higher resolution and no subtitle line that no script in this repository reproduces exactly; their
data are those of the listed outputs.

## Order to rebuild everything

```
python analysis/main_campaign/analisar_resultados.py
python analysis/tuned_controls_and_step_permutation/analyze_step_permutation_and_tuning.py
python analysis/tuned_controls_and_step_permutation/analyze_tuned_evaluate.py
python analysis/lr_uniform_control/analyze_lr_uniform_control.py
python analysis/global_gradient_diagnostic/analyze_global_gradient.py
python analysis/component_analysis_ablation/analyze_ablation.py --mode official
for f in analysis/audit/a*.py; do python $f; done             # a01 ... a15, in name order
python analysis/revision_consolidated_tables_figures/consolidate.py   # then the other scripts of that folder
python analysis/main_campaign/figuras_artigo.py
python analysis/drift_diagnostics_10seeds/analise_1_1.py
python analysis/global_prior_analysis/analyze_prior_assumption.py
python analysis/component_analysis_ablation/analyze_size_heterogeneity.py   # ~15 min (rebuilds partitions)
python analysis/data_partition_manifest/make_partition_manifest.py          # rebuilds every partition
```
