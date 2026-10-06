# Results map

Every artefact produced here, its source data, and the question it addresses.
Use this to audit any number in the paper back to the file it came from.

| Artefact | Source data | Experiments used | Question addressed |
|---|---|---|---|
| `tabA_accuracy_compute.tex`, `figA_pareto_accuracy_compute` | `main_campaign_default_alpha_0_5/*-results/test1_convergencia`, `test8_femnist` | MNIST, FashionMNIST, CIFAR-10, FEMNIST; 30 seeds | Accuracy–compute trade-off at the configured operating points (no Acc/TFLOP ratio) |
| `tabB_heterogeneity.tex`, `figB_heterogeneity_severity` | both base campaigns, `test1_convergencia` + `test2_robustez_alpha` | α ∈ {1.0, 0.5, 0.1, 0.01} where executed | Heterogeneity severity |
| `tabC_ablation.tex`, `figC_ablation_contrasts` | `component_analysis_ablation/tables/contrasts_official.csv`, `per_seed_diagnostics_official.csv` | CIFAR-10, α ∈ {0.01, 0.1}, 30 seeds, 5 arms | Controlled component analysis with matched update budgets |
| `tabD_long_horizon.tex` | `test7_plato` in both base campaigns | CIFAR-10, 50 rounds | Long-horizon behaviour |
| `tabE_fednova.tex` | `main_campaign_default_alpha_0_5/FedNova-results` | CD1, CD2, CD8 only | FedNova baseline |
| `tabF_proxy_validation.tex` | `drift_diagnostics_10seeds/1.1_diagnostico_drift/drift_telemetry.csv` | CIFAR-10, α ∈ {1.0, 0.1, 0.01}, 10 seeds | Association between H_k and update alignment (raw cosine; see Section 6.4) |
| `tabG_prior_metrics.tex` | `global_prior_analysis/tables/T2_dirichlet_spearman.csv`, `T4_*.csv` | CIFAR-10 + FEMNIST writer pool | Uniform-global-prior assumption in H_k |
| `tabH_partitions.tex` | `data_partition_manifest/artifacts/partitions/*.csv` | 1110 scenarios × 5 methods | Partition identity, fairness, reproducibility |
| `master_results_summary.csv` | all of the above | every run found | Full audit trail |

## Provenance and protocol boundaries

| Campaign | Hardware (Section 5.8 of the manuscript) | Initialisation protocol |
|---|---|---|
| `base_alpha_0.5` | workstation (Intel Core i7-14700 / NVIDIA RTX A1000 8GB / 32GB RAM) for FedAvg, FedAvgM, FedProx and FedHAD; server (AMD EPYC 9354P / NVIDIA A40 48GB / 256GB RAM) for FedNova, SCAFFOLD, the drift-diagnostic campaign and the six replacement runs | historical: initial global model taken from one random Ray client (not seed-controlled) |
| `base_alpha_0.01` | server (AMD EPYC 9354P / NVIDIA A40 48GB / 256GB RAM) | historical: initial global model taken from one random Ray client (not seed-controlled) |
| `revision_ablation` | server (AMD EPYC 9354P / NVIDIA A40 48GB / 256GB RAM) | revision: deterministic initial model fixed in the driver |

The `hardware` column of `master_results_summary.csv` gives the platform of every run
as recorded in its own report.

Rows from the two protocols are **never** compared pairwise. The `protocol` column
in `master_results_summary.csv` makes the boundary explicit.
