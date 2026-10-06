# FedHAD versus FedNova raw-data audit

Primary raw campaign: main_campaign_default_alpha_0_5. Every raw seed was verified against the consolidated master CSV before output generation.

## MNIST — $\alpha=1.0$

- FedHAD path:  results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/MNIST-Standard
- FedNova path: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/MNIST
- FedHAD seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedNova seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- paired seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedHAD accuracy: 99.060333 +/- 0.097609%
- FedNova accuracy: 99.091000 +/- 0.073689%
- FedHAD TFLOPs: 3.69930000 +/- 0.10461461
- FedNova TFLOPs: 4.56300000 +/- 0.00000000
- partition: confirmed: 30/30 manifest seeds, identical indices/counts across 5 methods

## MNIST — $\alpha=0.1$

- FedHAD path:  results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/MNIST-Standard
- FedNova path: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/MNIST
- FedHAD seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedNova seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- paired seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedHAD accuracy: 96.759667 +/- 1.221072%
- FedNova accuracy: 98.031000 +/- 0.435291%
- FedHAD TFLOPs: 3.22946667 +/- 0.15338924
- FedNova TFLOPs: 4.56300000 +/- 0.00000000
- partition: confirmed: 30/30 manifest seeds, identical indices/counts across 5 methods

## MNIST — $\alpha=0.01$

- FedHAD path:  results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/MNIST-Standard
- FedNova path: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/MNIST
- FedHAD seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedNova seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- paired seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedHAD accuracy: 91.137333 +/- 2.611311%
- FedNova accuracy: 91.822333 +/- 4.579132%
- FedHAD TFLOPs: 2.94423333 +/- 0.17633758
- FedNova TFLOPs: 4.56300000 +/- 0.00000000
- partition: confirmed: 30/30 manifest seeds, identical indices/counts across 5 methods

## FashionMNIST — $\alpha=1.0$

- FedHAD path:  results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/FashionMNIST-Standard
- FedNova path: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/FashionMNIST
- FedHAD seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedNova seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- paired seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedHAD accuracy: 89.086000 +/- 0.330429%
- FedNova accuracy: 89.099000 +/- 0.244884%
- FedHAD TFLOPs: 3.69913333 +/- 0.10427078
- FedNova TFLOPs: 4.56300000 +/- 0.00000000
- partition: confirmed: 30/30 manifest seeds, identical indices/counts across 5 methods

## FashionMNIST — $\alpha=0.1$

- FedHAD path:  results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/FashionMNIST-Standard
- FedNova path: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/FashionMNIST
- FedHAD seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedNova seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- paired seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedHAD accuracy: 81.010667 +/- 2.924844%
- FedNova accuracy: 82.240000 +/- 2.556922%
- FedHAD TFLOPs: 3.23633333 +/- 0.15245380
- FedNova TFLOPs: 4.56300000 +/- 0.00000000
- partition: confirmed: 30/30 manifest seeds, identical indices/counts across 5 methods

## FashionMNIST — $\alpha=0.01$

- FedHAD path:  results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/FashionMNIST-Standard
- FedNova path: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/FashionMNIST
- FedHAD seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedNova seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- paired seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedHAD accuracy: 70.715000 +/- 3.139881%
- FedNova accuracy: 67.334000 +/- 9.375090%
- FedHAD TFLOPs: 2.94576667 +/- 0.17333338
- FedNova TFLOPs: 4.56300000 +/- 0.00000000
- partition: confirmed: 30/30 manifest seeds, identical indices/counts across 5 methods

## CIFAR-10 — $\alpha=1.0$

- FedHAD path:  results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/CIFAR10-Standard
- FedNova path: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/CIFAR10
- FedHAD seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedNova seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- paired seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedHAD accuracy: 82.040667 +/- 0.502799%
- FedNova accuracy: 81.611000 +/- 0.551064%
- FedHAD TFLOPs: 330.00726667 +/- 9.29780370
- FedNova TFLOPs: 407.03280000 +/- 0.00549231
- partition: confirmed: 30/30 manifest seeds, identical indices/counts across 5 methods

## CIFAR-10 — $\alpha=0.1$

- FedHAD path:  results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/CIFAR10-Standard
- FedNova path: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/CIFAR10
- FedHAD seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedNova seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- paired seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedHAD accuracy: 68.531667 +/- 3.220017%
- FedNova accuracy: 67.549000 +/- 3.630273%
- FedHAD TFLOPs: 288.71210000 +/- 13.60076426
- FedNova TFLOPs: 407.03460000 +/- 0.00559310
- partition: confirmed: 30/30 manifest seeds, identical indices/counts across 5 methods

## CIFAR-10 — $\alpha=0.01$

- FedHAD path:  results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/CIFAR10-Standard
- FedNova path: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/CIFAR10
- FedHAD seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedNova seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- paired seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedHAD accuracy: 55.775333 +/- 4.053285%
- FedNova accuracy: 46.602333 +/- 7.122871%
- FedHAD TFLOPs: 262.78740000 +/- 15.46495882
- FedNova TFLOPs: 407.03010000 +/- 0.00760376
- partition: confirmed: 30/30 manifest seeds, identical indices/counts across 5 methods

## FEMNIST — natural federation

- FedHAD path:  results/main_campaign_default_alpha_0_5/FedHAD-results/test8_femnist/FEMNIST-Standard
- FedNova path: results/main_campaign_default_alpha_0_5/FedNova-results/test8_femnist/FEMNIST
- FedHAD seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedNova seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- paired seeds (30): [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
- FedHAD accuracy: 65.108503 +/- 1.184330%
- FedNova accuracy: 65.001336 +/- 1.276352%
- FedHAD TFLOPs: 11.13300000 +/- 0.00000000
- FedNova TFLOPs: 12.39483172 +/- 0.00000000
- partition: confirmed: 30/30 manifest seeds, identical indices/counts across 5 methods

## Separate alpha=0.5 base configuration

- MNIST, alpha=0.5: exists separately in test1_convergencia; 30 paired seeds; excluded from the main table.
- FashionMNIST, alpha=0.5: exists separately in test1_convergencia; 30 paired seeds; excluded from the main table.
- CIFAR-10, alpha=0.5: exists separately in test1_convergencia; 30 paired seeds; excluded from the main table.

## Statistical methods

Differences are FedHAD minus FedNova in percentage points. The 95% interval is a 10,000-resample percentile bootstrap of the paired mean, NumPy RNG seed 20260906, matching the ablation pipeline. Wilcoxon is two-sided paired signed-rank with zero_method='wilcox'; paired effect size is dz = mean paired difference / sample standard deviation.
