# Global-gradient diagnostic battery: result of the pre-registered tests

Seed-level statistics (one value per seed); Wilcoxon two-sided on the 30 values; Holm within the pre-registered family. 'Supports the premise' requires the Holm-adjusted test and the whole 95% CI on the predicted side.

| Hypothesis | alpha | statistic | n | mean | 95% CI | p | p_H | verdict |
|---|---|---|---|---|---|---|---|---|
| H1_primary | 0.1 | Spearman(H_k, grad_cos_loo) per seed | 30 | -0.082 | [-0.192, +0.030] | +0.2667 | +0.2667 | not detected |
| H1_primary | 0.2 | Spearman(H_k, grad_cos_loo) per seed | 30 | -0.225 | [-0.353, -0.093] | +0.0022 | +0.0044 | supports the premise |
| H1_update | 0.1 | Spearman(H_k, upd_cos_loo) per seed | 30 | -0.261 | [-0.370, -0.149] | +0.0003 | +0.0003 | supports the premise |
| H1_update | 0.2 | Spearman(H_k, upd_cos_loo) per seed | 30 | -0.307 | [-0.398, -0.212] | +0.0000 | +0.0001 | supports the premise |
| H1_paper_partitions (raw) | 0.1 | Spearman(H_k, grad_cos_loo) per seed | 30 | +0.139 | [+0.029, +0.245] | +0.0293 | +0.0587 | not detected |
| H1_paper_partitions (size-adjusted) | 0.1 | pooled OLS coefficient of H_k given log n_k and round (seed bootstrap) | 30 | -0.200 | [-0.529, +0.156] | +0.2140 | +0.2140 | not detected |
| H1_paper_partitions (raw) | 0.01 | Spearman(H_k, grad_cos_loo) per seed | 30 | +0.444 | [+0.381, +0.502] | +0.0000 | +0.0000 | contradicts the premise |
| H1_paper_partitions (size-adjusted) | 0.01 | pooled OLS coefficient of H_k given log n_k and round (seed bootstrap) | 30 | +0.590 | [+0.269, +0.777] | +0.0030 | +0.0090 | contradicts the premise |
| H2_removed_epochs | 0.1 | mean marg_cos_loo removed - kept epochs, per seed | 30 | +0.007 | [+0.005, +0.009] | +0.0000 | +0.0000 | contradicts the premise |
| ↳ share of removed epochs that increase the others' loss (first order) | 0.1 | | | 0.525 | | | | descriptive |
| H2_decline_by_skew | 0.1 | Spearman(H_k, slope of marg_cos_loo over epochs) per seed | 30 | -0.291 | [-0.383, -0.190] | +0.0000 | +0.0000 | supports the premise |
| H2_removed_epochs | 0.01 | mean marg_cos_loo removed - kept epochs, per seed | 30 | +0.017 | [+0.009, +0.025] | +0.0002 | +0.0002 | contradicts the premise |
| ↳ share of removed epochs that increase the others' loss (first order) | 0.01 | | | 0.464 | | | | descriptive |
| H2_decline_by_skew | 0.01 | Spearman(H_k, slope of marg_cos_loo over epochs) per seed | 30 | -0.302 | [-0.399, -0.197] | +0.0000 | +0.0000 | supports the premise |
| H3_allocation (all seeds) | 0.1 | final accuracy Full - inverse epochs (pp) | 30 | +0.647 | [-0.005, +1.362] | +0.1180 | +0.1681 | inconclusive |
| H3_allocation (all seeds) | 0.2 | final accuracy Full - inverse epochs (pp) | 30 | +0.377 | [-0.047, +0.798] | +0.0841 | +0.1681 | equivalent |
| H3_allocation (seeds where allocations differ) | 0.1 | final accuracy Full - inverse epochs (pp) | 20 | +1.141 | [+0.278, +2.083] | +0.0169 | +0.0337 | A superior |
| H3_allocation (seeds where allocations differ) | 0.2 | final accuracy Full - inverse epochs (pp) | 29 | +0.417 | [-0.026, +0.849] | +0.0562 | +0.0562 | inconclusive |
| H4_classes (paper partitions) / worst_class_recall | 0.1 | worst_class_recall full_fedhad - fedprox_default (pp) | 30 | +1.907 | [-0.403, +4.357] | +0.3085 | +1.0000 | not detected |
| H4_classes (paper partitions) / worst_class_recall | 0.01 | worst_class_recall full_fedhad - fedprox_default (pp) | 30 | +4.173 | [+2.650, +5.947] | +0.0000 | +0.0001 | A superior |
| H4_classes (paper partitions) / worst_class_recall | 0.1 | worst_class_recall full_fedhad - fedprox_tuned (pp) | 30 | -1.473 | [-3.663, +0.553] | +0.3493 | +1.0000 | not detected |
| H4_classes (paper partitions) / worst_class_recall | 0.01 | worst_class_recall full_fedhad - fedprox_tuned (pp) | 30 | -2.380 | [-3.923, -0.910] | +0.0010 | +0.0062 | B superior |
| H4_classes (paper partitions) / macro_f1 | 0.1 | macro_f1 full_fedhad - fedprox_default (pp) | 30 | +0.936 | [+0.283, +1.572] | +0.0062 | +0.0310 | A superior |
| H4_classes (paper partitions) / macro_f1 | 0.01 | macro_f1 full_fedhad - fedprox_default (pp) | 30 | +4.243 | [+3.285, +5.205] | +0.0000 | +0.0000 | A superior |
| H4_classes (paper partitions) / macro_f1 | 0.1 | macro_f1 full_fedhad - fedprox_tuned (pp) | 30 | +0.417 | [-0.162, +1.051] | +0.3818 | +1.0000 | not detected |
| H4_classes (paper partitions) / macro_f1 | 0.01 | macro_f1 full_fedhad - fedprox_tuned (pp) | 30 | -0.229 | [-1.131, +0.697] | +0.3285 | +1.0000 | not detected |
| H4_classes (equal partition) / worst_class_recall | 0.1 | worst_class_recall full_fedhad - inv_epochs (pp) | 30 | +0.480 | [-0.523, +1.470] | +0.3103 | +0.8130 | not detected |
| H4_classes (equal partition) / worst_class_recall | 0.2 | worst_class_recall full_fedhad - inv_epochs (pp) | 30 | +0.820 | [-1.173, +2.697] | +0.2804 | +0.8130 | not detected |
| H4_classes (equal partition) / macro_f1 | 0.1 | macro_f1 full_fedhad - inv_epochs (pp) | 30 | +0.662 | [-0.122, +1.513] | +0.2710 | +0.8130 | not detected |
| H4_classes (equal partition) / macro_f1 | 0.2 | macro_f1 full_fedhad - inv_epochs (pp) | 30 | +0.415 | [-0.078, +0.899] | +0.0699 | +0.2796 | not detected |

## Accuracy, worst-class recall and macro-F1 (mean over seeds, %)

| partition | arm | alpha | accuracy | worst class | macro-F1 | updates |
|---|---|---|---|---|---|---|
| dirichlet | fedprox_default | 0.01 | 52.68 | 2.56 | 47.39 | 70202 |
| dirichlet | fedprox_default | 0.1 | 68.14 | 17.86 | 65.74 | 70197 |
| dirichlet | fedprox_tuned | 0.01 | 55.15 | 9.11 | 51.86 | 28081 |
| dirichlet | fedprox_tuned | 0.1 | 68.35 | 21.24 | 66.26 | 42118 |
| dirichlet | full_fedhad | 0.01 | 55.35 | 6.73 | 51.63 | 45332 |
| dirichlet | full_fedhad | 0.1 | 68.85 | 19.76 | 66.68 | 49795 |
| equal | full_fedhad | 0.1 | 53.42 | 2.98 | 47.98 | 20720 |
| equal | full_fedhad | 0.2 | 63.12 | 15.35 | 60.18 | 23567 |
| equal | inv_epochs | 0.1 | 52.77 | 2.50 | 47.32 | 20720 |
| equal | inv_epochs | 0.2 | 62.75 | 14.53 | 59.77 | 23567 |
