# Global-gradient diagnostic battery: result of the pre-registered tests

Seed-level statistics (one value per seed); Wilcoxon two-sided on the 30 values; Holm within the pre-registered family. 'Supports the premise' requires the Holm-adjusted test and the whole 95% CI on the predicted side.

| Hypothesis | alpha | statistic | n | mean | 95% CI | p | p_H | verdict |
|---|---|---|---|---|---|---|---|---|
| H1_primary | 0.1 | Spearman(H_k, grad_cos_loo) per seed | 1 | +0.098 | [—, —] | — | +1.0000 | insufficient data |
| H1_primary | 0.2 | Spearman(H_k, grad_cos_loo) per seed | 1 | -0.615 | [—, —] | — | +1.0000 | insufficient data |
| H1_update | 0.1 | Spearman(H_k, upd_cos_loo) per seed | 1 | +0.566 | [—, —] | — | +1.0000 | insufficient data |
| H1_update | 0.2 | Spearman(H_k, upd_cos_loo) per seed | 1 | -0.468 | [—, —] | — | +1.0000 | insufficient data |
| H1_paper_partitions (raw) | 0.1 | Spearman(H_k, grad_cos_loo) per seed | 1 | +0.394 | [—, —] | — | +1.0000 | insufficient data |
| H1_paper_partitions (size-adjusted) | 0.1 | pooled OLS coefficient of H_k given log n_k and round (seed bootstrap) | 1 | +1.046 | [—, —] | — | +1.0000 | insufficient data |
| H1_paper_partitions (raw) | 0.01 | Spearman(H_k, grad_cos_loo) per seed | 1 | +0.468 | [—, —] | — | +1.0000 | insufficient data |
| H1_paper_partitions (size-adjusted) | 0.01 | pooled OLS coefficient of H_k given log n_k and round (seed bootstrap) | 1 | +0.585 | [—, —] | — | +1.0000 | insufficient data |
| H2_removed_epochs | 0.1 | mean marg_cos_loo removed - kept epochs, per seed | 1 | +0.027 | [—, —] | — | +1.0000 | insufficient data |
| ↳ share of removed epochs that increase the others' loss (first order) | 0.1 | | | 0.688 | | | | descriptive |
| H2_decline_by_skew | 0.1 | Spearman(H_k, slope of marg_cos_loo over epochs) per seed | 1 | -0.025 | [—, —] | — | +1.0000 | insufficient data |
| H2_removed_epochs | 0.01 | mean marg_cos_loo removed - kept epochs, per seed | 1 | +0.059 | [—, —] | — | +1.0000 | insufficient data |
| ↳ share of removed epochs that increase the others' loss (first order) | 0.01 | | | 0.200 | | | | descriptive |
| H2_decline_by_skew | 0.01 | Spearman(H_k, slope of marg_cos_loo over epochs) per seed | 1 | +0.517 | [—, —] | — | +1.0000 | insufficient data |

## Accuracy, worst-class recall and macro-F1 (mean over seeds, %)

| partition | arm | alpha | accuracy | worst class | macro-F1 | updates |
|---|---|---|---|---|---|---|
| dirichlet | fedprox_default | 0.01 | 31.10 | 0.00 | 26.87 | 14040 |
| dirichlet | fedprox_default | 0.1 | 41.09 | 0.00 | 31.82 | 14040 |
| dirichlet | fedprox_tuned | 0.01 | 34.70 | 0.00 | 30.33 | 5616 |
| dirichlet | fedprox_tuned | 0.1 | 42.38 | 0.00 | 34.74 | 8424 |
| dirichlet | full_fedhad | 0.01 | 40.47 | 0.00 | 34.34 | 9208 |
| dirichlet | full_fedhad | 0.1 | 45.47 | 0.00 | 39.17 | 10030 |
| equal | full_fedhad | 0.1 | 29.24 | 0.00 | 21.08 | 4200 |
| equal | full_fedhad | 0.2 | 36.72 | 0.00 | 28.70 | 5040 |
| equal | inv_epochs | 0.1 | 29.55 | 0.00 | 20.58 | 4200 |
| equal | inv_epochs | 0.2 | 37.71 | 0.00 | 31.43 | 5040 |
