# LR-uniform control battery: result of the pre-registered rule

Pre-registration sha256: see preregistration.sha256. Reference source: re-executed_in_this_battery.
Valid runs: 360. Invalid or incomplete runs ignored: 0.

## alpha = 0.1 (delta = 0.75 pp)
- Primary, Full - U2: n = 30, mean paired difference -0.19 pp, 95% CI [-0.56, +0.19], 90% CI [-0.50, +0.13], d_z -0.18, Wilcoxon p 0.217, Holm p 0.217, one-sided p against -delta 0.005 and +delta <0.001, seeds A/B/ties 13/17/0. Category assigned: 2: practically equivalent (within +/- delta).
- Secondary (supporting evidence), LR-only - U1: n = 30, mean paired difference +0.05 pp, 95% CI [-0.38, +0.49], 90% CI [-0.31, +0.43], d_z +0.04, Wilcoxon p 0.926, Holm p 1.000, one-sided p against -delta <0.001 and +delta 0.002, seeds A/B/ties 14/16/0. Category assigned: 2: practically equivalent (within +/- delta).
- Secondary (supporting evidence), Full - U2-arith: n = 30, mean paired difference -0.09 pp, 95% CI [-0.55, +0.38], 90% CI [-0.48, +0.30], d_z -0.07, Wilcoxon p 0.789, Holm p 1.000, one-sided p against -delta 0.004 and +delta <0.001, seeds A/B/ties 15/15/0. Category assigned: 2: practically equivalent (within +/- delta).
- Secondary (supporting evidence), Full - INV: n = 30, mean paired difference -0.12 pp, 95% CI [-0.69, +0.47], 90% CI [-0.60, +0.38], d_z -0.07, Wilcoxon p 0.612, Holm p 1.000, one-sided p against -delta 0.037 and +delta 0.001, seeds A/B/ties 15/15/0. Category assigned: 2: practically equivalent (within +/- delta).

## alpha = 0.01 (delta = 1 pp)
- Primary, Full - U2: n = 30, mean paired difference -1.13 pp, 95% CI [-1.82, -0.50], 90% CI [-1.70, -0.61], d_z -0.60, Wilcoxon p 0.001, Holm p 0.001, one-sided p against -delta 0.572 and +delta <0.001, seeds A/B/ties 9/21/0. Category assigned: 3: uniform LR superior.
- Secondary (supporting evidence), LR-only - U1: n = 30, mean paired difference -0.23 pp, 95% CI [-0.78, +0.28], 90% CI [-0.68, +0.21], d_z -0.15, Wilcoxon p 0.704, Holm p 0.704, one-sided p against -delta 0.003 and +delta <0.001, seeds A/B/ties 14/16/0. Category assigned: 2: practically equivalent (within +/- delta).
- Secondary (supporting evidence), Full - U2-arith: n = 30, mean paired difference -1.04 pp, 95% CI [-1.79, -0.33], 90% CI [-1.67, -0.43], d_z -0.50, Wilcoxon p 0.019, Holm p 0.047, one-sided p against -delta 0.496 and +delta <0.001, seeds A/B/ties 9/21/0. Category assigned: 3: uniform LR superior.
- Secondary (supporting evidence), Full - INV: n = 30, mean paired difference -0.90 pp, 95% CI [-1.65, -0.24], 90% CI [-1.53, -0.33], d_z -0.44, Wilcoxon p 0.016, Holm p 0.047, one-sided p against -delta 0.123 and +delta <0.001, seeds A/B/ties 9/21/0. Category assigned: 3: uniform LR superior.

## Sensitivity analysis (seeds with an empty client excluded)
- alpha 0.1, Full - U2: n = 30, -0.19 pp, 95% CI [-0.56, +0.19], 90% CI [-0.50, +0.13], Holm p 0.217, category 2.
- alpha 0.1, LR-only - U1: n = 30, +0.05 pp, 95% CI [-0.38, +0.49], 90% CI [-0.31, +0.43], Holm p 1.000, category 2.
- alpha 0.1, Full - U2-arith: n = 30, -0.09 pp, 95% CI [-0.55, +0.38], 90% CI [-0.48, +0.30], Holm p 1.000, category 2.
- alpha 0.1, Full - INV: n = 30, -0.12 pp, 95% CI [-0.69, +0.47], 90% CI [-0.60, +0.38], Holm p 1.000, category 2.
- alpha 0.01, Full - U2: n = 25, -1.02 pp, 95% CI [-1.82, -0.35], 90% CI [-1.66, -0.44], Holm p 0.007, category 3.
- alpha 0.01, LR-only - U1: n = 25, -0.35 pp, 95% CI [-0.97, +0.22], 90% CI [-0.86, +0.13], Holm p 0.508, category 2.
- alpha 0.01, Full - U2-arith: n = 25, -0.94 pp, 95% CI [-1.78, -0.14], 90% CI [-1.64, -0.28], Holm p 0.109, category 4.
- alpha 0.01, Full - INV: n = 25, -0.83 pp, 95% CI [-1.66, -0.08], 90% CI [-1.52, -0.20], Holm p 0.079, category 4.

## Matching of sum_k p_k eta_k tau_k (executed), relative to the reference arm
- U2 vs Full, alpha 0.1: n = 30, mean relative difference +3.57e-17, max |relative difference| 3.98e-16.
- U2 vs Full, alpha 0.01: n = 30, mean relative difference -2.39e-17, max |relative difference| 3.64e-16.
- U1 vs LR-only, alpha 0.1: n = 30, mean relative difference +3.30e-17, max |relative difference| 2.90e-16.
- U1 vs LR-only, alpha 0.01: n = 30, mean relative difference +2.97e-17, max |relative difference| 4.42e-16.

Levels of alpha are reported separately and not combined. Secondary contrasts are supporting evidence and do not replace the primary contrast.
