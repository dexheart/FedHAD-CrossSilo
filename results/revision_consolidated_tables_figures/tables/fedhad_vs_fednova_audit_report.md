# FedHAD versus FedNova audit

This independent audit reads raw per-seed reports and does not modify raw data, consolidation code, or the LaTeX table.

## Methods

Paired differences use d_i = 100 * (FedHAD_i - FedNova_i), in percentage points.
Bootstrap is a paired percentile bootstrap of d_i: 10,000 resamples, NumPy RNG seed 20260906, matching the paired_stats convention in component_analysis_ablation/analyze_ablation.py.
Wilcoxon uses SciPy 1.18.1: two-sided, zero_method='wilcox', correction=False, method='auto' (the current pipeline default).

## Per-scenario results

### MNIST — $\alpha=1.0$

- Raw sources: FedHAD: results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/MNIST-Standard; FedNova: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/MNIST.
- Seeds, FedHAD/FedNova/paired: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71].
- Duplicates, FedHAD/FedNova: 0/0; complete pairing: True; one experimental family: True.
- Partitions: 30/30 matching manifest rows; indices/counts identical across >=5 methods.
- Accuracy (%), FedHAD/FedNova: 99.060333333 +/- 0.097608766 / 99.091000000 +/- 0.073688534.
- d (pp): mean -0.030666667; median -0.035000000; SD 0.100204389; min/max [-0.330000000, +0.150000000]; positive/negative/zero 10/18/2.
- Wilcoxon: statistic 126.5; p 0.0812788096032; zero_method=wilcox; correction=False; method=auto.
- Paired bootstrap CI95 (pp): [-0.067000000, +0.003666667]; 10,000 resamples; seed 20260906.
- FLOPs, FedHAD/FedNova/Delta: 3.699300000 / 4.563000000 / +18.928336621%.
- Consolidated summary/statistics/LaTeX match: True/True/True.

### MNIST — $\alpha=0.1$

- Raw sources: FedHAD: results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/MNIST-Standard; FedNova: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/MNIST.
- Seeds, FedHAD/FedNova/paired: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71].
- Duplicates, FedHAD/FedNova: 0/0; complete pairing: True; one experimental family: True.
- Partitions: 30/30 matching manifest rows; indices/counts identical across >=5 methods.
- Accuracy (%), FedHAD/FedNova: 96.759666667 +/- 1.221072122 / 98.031000000 +/- 0.435291024.
- d (pp): mean -1.271333333; median -0.930000000; SD 0.970651162; min/max [-3.530000000, +0.060000000]; positive/negative/zero 1/29/0.
- Wilcoxon: statistic 2; p 2.12532037346e-06; zero_method=wilcox; correction=False; method=auto.
- Paired bootstrap CI95 (pp): [-1.621008333, -0.942991667]; 10,000 resamples; seed 20260906.
- FLOPs, FedHAD/FedNova/Delta: 3.229466667 / 4.563000000 / +29.224925122%.
- Consolidated summary/statistics/LaTeX match: True/True/True.

### MNIST — $\alpha=0.01$

- Raw sources: FedHAD: results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/MNIST-Standard; FedNova: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/MNIST.
- Seeds, FedHAD/FedNova/paired: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71].
- Duplicates, FedHAD/FedNova: 0/0; complete pairing: True; one experimental family: True.
- Partitions: 30/30 matching manifest rows; indices/counts identical across >=5 methods.
- Accuracy (%), FedHAD/FedNova: 91.137333333 +/- 2.611310852 / 91.822333333 +/- 4.579131587.
- d (pp): mean -0.685000000; median -1.660000000; SD 4.169400643; min/max [-8.190000000, +12.570000000]; positive/negative/zero 11/19/0.
- Wilcoxon: statistic 161; p 0.145999491215; zero_method=wilcox; correction=False; method=auto.
- Paired bootstrap CI95 (pp): [-2.061666667, +0.855350000]; 10,000 resamples; seed 20260906.
- FLOPs, FedHAD/FedNova/Delta: 2.944233333 / 4.563000000 / +35.475929578%.
- Consolidated summary/statistics/LaTeX match: True/True/True.

### FashionMNIST — $\alpha=1.0$

- Raw sources: FedHAD: results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/FashionMNIST-Standard; FedNova: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/FashionMNIST.
- Seeds, FedHAD/FedNova/paired: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71].
- Duplicates, FedHAD/FedNova: 0/0; complete pairing: True; one experimental family: True.
- Partitions: 30/30 matching manifest rows; indices/counts identical across >=5 methods.
- Accuracy (%), FedHAD/FedNova: 89.086000000 +/- 0.330429188 / 89.099000000 +/- 0.244883505.
- d (pp): mean -0.013000000; median +0.020000000; SD 0.389147167; min/max [-0.840000000, +0.590000000]; positive/negative/zero 15/15/0.
- Wilcoxon: statistic 231.5; p 0.983589562802; zero_method=wilcox; correction=False; method=auto.
- Paired bootstrap CI95 (pp): [-0.148000000, +0.119341667]; 10,000 resamples; seed 20260906.
- FLOPs, FedHAD/FedNova/Delta: 3.699133333 / 4.563000000 / +18.931989188%.
- Consolidated summary/statistics/LaTeX match: True/True/True.

### FashionMNIST — $\alpha=0.1$

- Raw sources: FedHAD: results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/FashionMNIST-Standard; FedNova: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/FashionMNIST.
- Seeds, FedHAD/FedNova/paired: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71].
- Duplicates, FedHAD/FedNova: 0/0; complete pairing: True; one experimental family: True.
- Partitions: 30/30 matching manifest rows; indices/counts identical across >=5 methods.
- Accuracy (%), FedHAD/FedNova: 81.010666667 +/- 2.924843566 / 82.240000000 +/- 2.556921641.
- d (pp): mean -1.229333333; median -0.945000000; SD 2.712525142; min/max [-7.380000000, +4.070000000]; positive/negative/zero 9/21/0.
- Wilcoxon: statistic 123; p 0.0234101433307; zero_method=wilcox; correction=False; method=auto.
- Paired bootstrap CI95 (pp): [-2.188758333, -0.297933333]; 10,000 resamples; seed 20260906.
- FLOPs, FedHAD/FedNova/Delta: 3.236333333 / 4.563000000 / +29.074439331%.
- Consolidated summary/statistics/LaTeX match: True/True/True.

### FashionMNIST — $\alpha=0.01$

- Raw sources: FedHAD: results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/FashionMNIST-Standard; FedNova: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/FashionMNIST.
- Seeds, FedHAD/FedNova/paired: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71].
- Duplicates, FedHAD/FedNova: 0/0; complete pairing: True; one experimental family: True.
- Partitions: 30/30 matching manifest rows; indices/counts identical across >=5 methods.
- Accuracy (%), FedHAD/FedNova: 70.715000000 +/- 3.139881120 / 67.334000000 +/- 9.375089738.
- d (pp): mean +3.381000000; median +1.975000000; SD 9.767470060; min/max [-11.660000000, +22.120000000]; positive/negative/zero 16/14/0.
- Wilcoxon: statistic 161; p 0.145999491215; zero_method=wilcox; correction=False; method=auto.
- Paired bootstrap CI95 (pp): [+0.103225000, +6.866091667]; 10,000 resamples; seed 20260906.
- FLOPs, FedHAD/FedNova/Delta: 2.945766667 / 4.563000000 / +35.442325955%.
- Consolidated summary/statistics/LaTeX match: True/True/True.

#### FashionMNIST alpha=0.01 paired differences

Ordered by seed; values are unrounded raw-pair calculations in pp.

     seed  paired_difference_pp
       42       -0.740000000000
       43      +17.610000000000
       44       -6.760000000000
       45       -1.300000000000
       46      -10.280000000000
       47       -8.270000000000
       48       -2.570000000000
       49       -2.200000000000
       50       +6.350000000000
       51      +19.370000000000
       52      -11.660000000000
       53       -2.270000000000
       54       +2.120000000000
       55       -4.610000000000
       56       -4.080000000000
       57      +18.730000000000
       58      +22.120000000000
       59       +6.220000000000
       60       +1.830000000000
       61      +16.730000000000
       62      +14.320000000000
       63      +11.820000000000
       64       -9.010000000000
       65       -1.990000000000
       66       +4.920000000000
       67       -7.240000000000
       68       +8.590000000000
       69       +6.930000000000
       70      +12.640000000000
       71       +4.110000000000

The bootstrap interval targets the arithmetic paired mean, while Wilcoxon ranks signed absolute differences and tests a location shift. Mixed signs and rank mass can yield a positive mean CI while the two-sided Wilcoxon p-value remains above 0.05. This is a legitimate difference in inferential targets, not a pairing or arithmetic error.

### CIFAR-10 — $\alpha=1.0$

- Raw sources: FedHAD: results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/CIFAR10-Standard; FedNova: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/CIFAR10.
- Seeds, FedHAD/FedNova/paired: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71].
- Duplicates, FedHAD/FedNova: 0/0; complete pairing: True; one experimental family: True.
- Partitions: 30/30 matching manifest rows; indices/counts identical across >=5 methods.
- Accuracy (%), FedHAD/FedNova: 82.040666667 +/- 0.502798605 / 81.611000000 +/- 0.551063861.
- d (pp): mean +0.429666667; median +0.515000000; SD 0.494901476; min/max [-0.380000000, +1.710000000]; positive/negative/zero 22/8/0.
- Wilcoxon: statistic 54; p 0.000240813315105; zero_method=wilcox; correction=False; method=auto.
- Paired bootstrap CI95 (pp): [+0.258333333, +0.605666667]; 10,000 resamples; seed 20260906.
- FLOPs, FedHAD/FedNova/Delta: 330.007266667 / 407.032800000 / +18.923667413%.
- Consolidated summary/statistics/LaTeX match: True/True/True.

### CIFAR-10 — $\alpha=0.1$

- Raw sources: FedHAD: results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/CIFAR10-Standard; FedNova: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/CIFAR10.
- Seeds, FedHAD/FedNova/paired: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71].
- Duplicates, FedHAD/FedNova: 0/0; complete pairing: True; one experimental family: True.
- Partitions: 30/30 matching manifest rows; indices/counts identical across >=5 methods.
- Accuracy (%), FedHAD/FedNova: 68.531666667 +/- 3.220017223 / 67.549000000 +/- 3.630273429.
- d (pp): mean +0.982666667; median +0.935000000; SD 3.521754004; min/max [-3.820000000, +7.500000000]; positive/negative/zero 16/14/0.
- Wilcoxon: statistic 174.5; p 0.232871242333; zero_method=wilcox; correction=False; method=auto.
- Paired bootstrap CI95 (pp): [-0.251333333, +2.224366667]; 10,000 resamples; seed 20260906.
- FLOPs, FedHAD/FedNova/Delta: 288.712100000 / 407.034600000 / +29.069396066%.
- Consolidated summary/statistics/LaTeX match: True/True/True.

### CIFAR-10 — $\alpha=0.01$

- Raw sources: FedHAD: results/main_campaign_default_alpha_0_5/FedHAD-results/test2_robustez_alpha/CIFAR10-Standard; FedNova: results/main_campaign_default_alpha_0_5/FedNova-results/test2_robustez_alpha/CIFAR10.
- Seeds, FedHAD/FedNova/paired: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71].
- Duplicates, FedHAD/FedNova: 0/0; complete pairing: True; one experimental family: True.
- Partitions: 30/30 matching manifest rows; indices/counts identical across >=5 methods.
- Accuracy (%), FedHAD/FedNova: 55.775333333 +/- 4.053284520 / 46.602333333 +/- 7.122870732.
- d (pp): mean +9.173000000; median +6.615000000; SD 8.297686071; min/max [-3.280000000, +27.600000000]; positive/negative/zero 26/4/0.
- Wilcoxon: statistic 14; p 2.04890966415e-07; zero_method=wilcox; correction=False; method=auto.
- Paired bootstrap CI95 (pp): [+6.291991667, +12.135425000]; 10,000 resamples; seed 20260906.
- FLOPs, FedHAD/FedNova/Delta: 262.787400000 / 407.030100000 / +35.437845997%.
- Consolidated summary/statistics/LaTeX match: True/True/True.

### FEMNIST — natural federation

- Raw sources: FedHAD: results/main_campaign_default_alpha_0_5/FedHAD-results/test8_femnist/FEMNIST-Standard; FedNova: results/main_campaign_default_alpha_0_5/FedNova-results/test8_femnist/FEMNIST.
- Seeds, FedHAD/FedNova/paired: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71] / [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71].
- Duplicates, FedHAD/FedNova: 0/0; complete pairing: True; one experimental family: True.
- Partitions: 30/30 matching manifest rows; indices/counts identical across >=5 methods.
- Accuracy (%), FedHAD/FedNova: 65.108503289 +/- 1.184330080 / 65.001335655 +/- 1.276352021.
- d (pp): mean +0.107167635; median +0.059950095; SD 1.370143439; min/max [-3.025688885, +3.584318606]; positive/negative/zero 16/14/0.
- Wilcoxon: statistic 226; p 0.903226472437; zero_method=wilcox; correction=False; method=auto.
- Paired bootstrap CI95 (pp): [-0.373005651, +0.593084410]; 10,000 resamples; seed 20260906.
- FLOPs, FedHAD/FedNova/Delta: 11.133000000 / 12.394831718 / +10.180305365%.
- Consolidated summary/statistics/LaTeX match: True/True/True.

## Conclusion

Overall audit status: PASS.
No values were rounded before mean/SD, paired differences, bootstrap, Wilcoxon, or FLOP calculations.
The same 30 seed pairs were used for every calculation within each scenario.
