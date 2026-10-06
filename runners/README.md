# Runners

One runner per experiment. Run them from the repository root with the project environment, e.g.
`env_flwr_pt/bin/python runners/run_main_campaign.py --dry-run`. Every runner writes only under
`results/`, resumes automatically when re-run, and shows its plan without training or
writing any file with `--dry-run`.

| Runner | Experiment (manuscript section) | Training code | Results |
|---|---|---|---|
| `run_main_campaign.py` | Main campaign, tests 1–8, six methods (§5–6, §6.9) | `algoritmos/*_Final.py` | `results/main_campaign_default_alpha_0_5/` |
| `run_drift_diagnostics.py` | 10-seed drift diagnostic (§6.4) | `algoritmos/*_Final.py` | `results/drift_diagnostics_10seeds/` |
| `run_component_ablation.py` | Controlled component analysis (§6.10) | `algoritmos/component_analysis_ablation/` | `results/component_analysis_ablation/` |
| `run_tuned_controls_and_step_permutation.py` | Step-preserving permutation; FedProx tuned on validation seeds (§5.7, §6.11) | `algoritmos/tuned_controls_and_step_permutation/` | `results/step_preserving_permutation/`, `results/tuned_controls_fedprox_validation/`, `results/initial_checkpoints_common/` |
| `run_tuned_controls_evaluation.sh` | Pre-registration + evaluation + analysis of the tuned FedProx | (calls the runner above) | `results/tuned_controls_fedprox_validation/` |
| `run_lr_uniform_control.py` | Uniform-learning-rate control, pre-registered (§6.11) | `algoritmos/lr_uniform_control/` | `results/lr_uniform_control/` |
| `run_global_gradient_diagnostic.py` | Global-gradient diagnostic, pre-registered (§5.7, §6.11) | `algoritmos/global_gradient_diagnostic/` | `results/global_gradient_diagnostic/` |

The analyses that turn these results into the tables and figures of the paper are in `analysis/`.

## Notes on `run_main_campaign.py`

- **Two campaigns, one runner.** The same runner and the same `*_Final.py` scripts produced both
  main campaigns, which share the folder structure (`<Method>-results/<test>/<dataset>/`).
  `results/main_campaign_default_alpha_0_5/` is the configuration written in the file
  (`RESULTS_DIR`). `results/main_campaign_default_alpha_0_01/` was produced on the server with
  `RESULTS_DIR` pointing to that folder, the blocks set to α = 0.01 (test 2 keeps its α grid) and
  the FEMNIST block (`test8_femnist`) set to 50 rounds; the parameters of every run are recorded
  in its file name and in its report. The paper uses the 50-round blocks of this campaign
  (Section 6.7, Table 11, Figure 13 c, d).
- **FedNova and SCAFFOLD coverage.** Both were executed only in tests 1, 2 and 8 (base point,
  heterogeneity family and natural federation), as stated in Section 5.2 of the paper. The file
  is left as configured for the last batch (`ALGORITMOS_ATIVOS = ["SCAFFOLD"]` and
  `EXPERIMENTOS_ATIVOS` = tests 1, 2 and 8). Setting `ALGORITMOS_ATIVOS` to every method with
  all tests active would also schedule FedNova in tests 3, 4 and 7, which were not executed.
- Test 5 (`test5_ablacao`) is the historical on/off ablation and is not used in the paper; with a
  heuristic switched off, `FedHAD_Final_2.0.py` trains one epoch instead of five, which is why it
  was superseded by the controlled component analysis (`run_component_ablation.py`).
