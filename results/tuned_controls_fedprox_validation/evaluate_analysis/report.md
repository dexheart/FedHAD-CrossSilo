# Experimento 2 — avaliação definitiva (pré-registrada)

Pré-registro: `results/tuned_controls_fedprox_validation/evaluate_preregistration.json` (registrado em 2026-10-01T19:41:42). Regra aplicada mecanicamente; nada abaixo foi escolhido depois de ver os dados.

## Acurácia e custo por braço

| α | braço | n | acurácia (%) | DP | passos | passos/FedHAD | TFLOPs exec. | TFLOPs/FedHAD |
|---|---|---|---|---|---|---|---|---|
| 0.1 | FedHAD | 30 | 68.69 | 3.32 | 49795 | 1.00 | 288.2 | 1.00 |
| 0.1 | FedProx tuned (budget_matched) | 30 | 68.26 | 3.27 | 42118 | 0.85 | 243.8 | 0.85 |
| 0.1 | FedProx default (0.01, E=5) | 30 | 67.97 | 3.77 | 70197 | 1.41 | 406.3 | 1.41 |
| 0.01 | FedHAD | 120 | 55.81 | 3.73 | 45168 | 1.00 | 261.5 | 1.00 |
| 0.01 | FedProx tuned (budget_matched) | 120 | 55.61 | 4.06 | 28079 | 0.62 | 162.5 | 0.62 |
| 0.01 | FedProx default (0.01, E=5) | 120 | 52.84 | 4.16 | 70198 | 1.55 | 406.3 | 1.55 |

## Contraste primário (Holm sobre os 2 alphas)

| α | contraste (A − B) | n | média (pp) | IC95% | IC90% | δ | A/B vitórias | p | p Holm | não-inferior A | categoria |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | FedHAD - FedProx tuned (budget_matched) | 30 | +0.43 | [-0.01, +0.93] | [+0.06, +0.84] | 0.75 | 17/13 | 0.1280 | 0.2560 | sim | **4: inconclusivo** |
| 0.01 | FedHAD - FedProx tuned (budget_matched) | 120 | +0.21 | [-0.14, +0.56] | [-0.09, +0.50] | 1 | 56/64 | 0.5297 | 0.5297 | sim | **2: equivalente (±δ)** |

## Contrastes secundários (Holm sobre os 4)

| α | contraste (A − B) | n | média (pp) | IC95% | IC90% | δ | A/B vitórias | p | p Holm | não-inferior A | categoria |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | FedHAD - FedProx default (0.01, E=5) | 30 | +0.72 | [+0.24, +1.21] | [+0.32, +1.13] | 0.75 | 23/7 | 0.0016 | 0.0032 | sim | **1: A superior** |
| 0.1 | FedProx tuned (budget_matched) - FedProx default (0.01, E=5) | 30 | +0.29 | [-0.28, +0.84] | [-0.20, +0.75] | 0.75 | 18/12 | 0.2802 | 0.2802 | sim | **4: inconclusivo** |
| 0.01 | FedHAD - FedProx default (0.01, E=5) | 120 | +2.97 | [+2.58, +3.36] | [+2.64, +3.30] | 1 | 109/11 | 0.0000 | 0.0000 | sim | **1: A superior** |
| 0.01 | FedProx tuned (budget_matched) - FedProx default (0.01, E=5) | 120 | +2.76 | [+2.31, +3.20] | [+2.39, +3.13] | 1 | 101/19 | 0.0000 | 0.0000 | sim | **1: A superior** |

## Sensibilidade: α=0,01 só com as 30 sementes do artigo (42–71)

| α | contraste (A − B) | n | média (pp) | IC95% | IC90% | δ | A/B vitórias | p | p Holm | não-inferior A | categoria |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | FedHAD - FedProx tuned (budget_matched) | 30 | +0.23 | [-0.46, +0.94] | [-0.35, +0.83] | 1 | 14/16 | 0.5928 | 0.5928 | sim | **2: equivalente (±δ)** |

## Custo do ajuste (fase de validação)

- α=0.1: 80 execuções de grade, 3929520 passos no total = 80× uma execução do FedHAD.
- α=0.01: 80 execuções de grade, 3931760 passos no total = 91× uma execução do FedHAD.

## Leitura pré-registrada

- α=0.1: categoria 4 → Inconclusive: report the CI and non-inferiority result; no accuracy claim against tuned baselines may be made.
  Não-inferioridade (margem 0.75 pp): satisfeita.
- α=0.01: categoria 2 → Equivalent within +/- delta: FedHAD's accuracy is matched by tuning; the contribution must be framed as automatic, tuning-free budget selection (it reaches the tuned operating point without the validation grid). Compare costs: if the tuned FedProx also uses fewer steps, FedHAD does not dominate it.
  Não-inferioridade (margem 1 pp): satisfeita.
