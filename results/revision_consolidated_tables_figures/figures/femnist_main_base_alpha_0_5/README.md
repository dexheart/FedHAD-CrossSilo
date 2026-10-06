# FEMNIST figures of the base-alpha-0_5 campaign (10 rounds)

## Scope and audit

These artifacts use only `main_campaign_default_alpha_0_5/test8_femnist`: 10 rounds, 10 clients, natural partitioning by writer, delay 0.05 and seeds 42--71. The folder name is historical; FEMNIST does not use a Dirichlet alpha in this campaign. There are exactly 30 runs for each approved method. No training was run. The terminal values were checked against `data/master_results_summary.csv`. For the methods with 5 fixed epochs, the publication cost is the nominal scheduled workload: 116602368 FLOPs/sample x 2126 samples x 5 epochs x 10 rounds = 12.3948317184 TFLOPs. The values actually summed from the logs remain in the `reported_*` columns.
The canonical logs of FedProx/seed 66 and FedHAD/seed 53 are the complete re-executions, run on the server with the same configuration and seed because the original executions on the workstation were left incomplete by failures of that machine (Section 5.8 of the manuscript); the partial logs were preserved with the suffix `discarded_incomplete_2026-09-10` and fall outside the `*.txt` pattern used in this analysis.

All error bars and bands show the mean +/- one sample standard deviation over 30 seeds. Accuracy and loss are centralized. No Accuracy/TFLOP ratio is used.

## Purpose of each figure

1. `01_pareto_final_accuracy_vs_tflops`: main trade-off between final accuracy and cost, with uncertainty on both axes and FedHAD highlighted, without offsetting the points.
2. `02_seed_scatter_accuracy_vs_tflops`: full distribution of the 30 runs.
3. `03_final_accuracy_errorbars`: predictive performance alone.
4. `04_total_training_tflops_errorbars`: computational cost alone.
5. `05_accuracy_vs_accumulated_tflops`: learning trajectory in the domain of the measured cost.
6. `06_cost_to_target`: TFLOPs to reach 50%, 55%, 60%, with reached/30 stated explicitly.
7. `07_accuracy_over_communication_rounds`: convergence per communication round.
8. `08_loss_over_communication_rounds`: dynamics of the centralized loss.
9. `09_target_attainment_probability_vs_tflops`: empirical probability of reaching each target under a computational budget.

## Use in the manuscript

`01_pareto_final_accuracy_vs_tflops` is Figure 15 of the manuscript (Section 6.8). The other figures in this folder do not appear in the manuscript. At 10 rounds FedHAD is 0.83 percentage points behind FedProx, a difference that survives the Holm adjustment, with 10.2% less nominal computation; the executed totals are 10.30 and 11.38 TFLOPs. The Accuracy/TFLOP ratio is not used in the manuscript.

## Suggested caption

Final accuracy against total training computation on FEMNIST after 10 communication rounds. Points show means over 30 seeds and error bars denote one standard deviation. Points toward the upper-left region represent a more favorable accuracy--computation trade-off.

## Target rule

The candidates were 50%, 55%, 60% and 65%. A target is plotted only when every method reaches it in at least 80% of the seeds. Retained: 50%, 55%, 60%; the full audit is in `femnist_cost_to_target.csv`.

## Aggregate values used

| Method | n | Final accuracy, mean +/- SD (%) | Total TFLOPs, mean +/- SD |
|---|---:|---:|---:|
| FedAvg | 30 | 65.649 +/- 1.072 | 12.395 +/- 0.000 |
| FedAvgM | 30 | 64.927 +/- 1.266 | 12.395 +/- 0.000 |
| FedProx | 30 | 65.938 +/- 1.364 | 12.395 +/- 0.000 |
| FedHAD | 30 | 65.109 +/- 1.184 | 11.133 +/- 0.000 |
