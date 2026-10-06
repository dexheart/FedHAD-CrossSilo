# FEMNIST figures of the main campaign (50 rounds)

## Scope and audit

These artifacts use only `main_campaign_default_alpha_0_01/test8_femnist` (read from `resultados_cluster.zip`, a byte-for-byte copy of the folder): 50 rounds, 10 clients, natural partitioning by writer, communication delay 0.05 and seeds 42--71. There are exactly 30 runs for each of FedAvg, FedAvgM, FedProx and FedHAD. No training was run. The terminal values were checked against `data/master_results_summary.csv`; the accumulated cost was reconstructed by summing the values recorded in `total_flops_round`.

All error bars and shaded bands show the mean +/- one sample standard deviation over 30 seeds. Accuracy and loss are centralized evaluation metrics. No Accuracy/TFLOP ratio is used.

## Purpose of each figure

1. `01_pareto_final_accuracy_vs_tflops`: main view of the accuracy--cost trade-off; shows both uncertainties and highlights FedHAD without artificially offsetting the points. The reading of the upper-left region is left to the caption.
2. `02_seed_scatter_accuracy_vs_tflops`: shows the whole distribution across runs, not only the aggregated means.
3. `03_final_accuracy_errorbars`: compares predictive performance alone.
4. `04_total_training_tflops_errorbars`: compares training cost alone.
5. `05_accuracy_vs_accumulated_tflops`: shows learning as the measured cost accumulates; the line uses the mean accumulated cost and the mean accuracy per round, the vertical band is the SD of accuracy and the sparse horizontal bars are the SD of cost.
6. `06_cost_to_target`: compares the cost to first reach 60%, 65%, 70%. Means and SDs are conditional on the seeds that reach the target, and each bar reports reached/30.
7. `07_accuracy_over_communication_rounds`: compares convergence in the communication domain.
8. `08_loss_over_communication_rounds`: checks whether the accuracy evidence is consistent with the dynamics of the centralized loss.
9. `09_target_attainment_probability_vs_tflops`: complements the conditional cost-to-target means with the empirical probability of reaching each target under a cost budget.

## Use in the manuscript

None of these figures appears in the current manuscript. The 50-round FEMNIST block is reported in Table 11 (Section 6.7): differences between FedHAD and the three baselines below 0.15 percentage points, none detected after the Holm adjustment, and a reduction of about 10% in nominal computation (55.67 against 61.97 TFLOPs). An undetected difference is not treated as equivalence. The Accuracy/TFLOP ratio is not used in the manuscript.

## Suggested caption

Final accuracy against total training computation on FEMNIST after 50 communication rounds. Points show means over 30 seeds and error bars denote one standard deviation. Points toward the upper-left region represent a more favorable accuracy--computation trade-off.

## Target rule

The candidate targets were 60%, 65%, 70% and 75%. A target appears only when every method reaches it in at least 80% of the seeds. This retains 60%, 65%, 70%; the full audit, including omitted targets, is in `femnist_cost_to_target.csv`.

## Aggregate values used

| Method | n | Final accuracy, mean +/- SD (%) | Total TFLOPs, mean +/- SD |
|---|---:|---:|---:|
| FedAvg | 30 | 70.909 +/- 1.055 | 61.974 +/- 0.000 |
| FedAvgM | 30 | 70.687 +/- 0.955 | 61.974 +/- 0.000 |
| FedProx | 30 | 70.924 +/- 1.108 | 61.974 +/- 0.000 |
| FedHAD | 30 | 70.804 +/- 1.020 | 55.667 +/- 0.003 |
