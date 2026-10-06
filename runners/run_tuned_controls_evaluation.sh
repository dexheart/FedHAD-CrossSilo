#!/usr/bin/env bash
# Controles FedProx ajustados — fase de avaliação DEFINITIVA, pré-registrada.
#
#   FedHAD  vs  FedProx ajustado (budget_matched, congelado em frozen_config.json)
#           vs  FedProx padrão (lr=0.01, E=5)
#
#   alpha 0.1 : sementes 42-71           (30)  -> 60 execuções novas
#   alpha 0.01: sementes 42-71 e 77-166 (120) -> 240 novas + 90 FedHAD nas sementes novas
#   total ~390 execuções, ~19-20 h a ~180 s cada
#
# USO:  bash runners/run_tuned_controls_evaluation.sh
# Retomada: rode o mesmo comando de novo (execuções completas são puladas).
# Não rode junto com run_lr_uniform_control.py (GPU compartilhada altera tempos).
set -euo pipefail
cd "$(dirname "$0")/.."   # raiz do repositório
PY=env_flwr_pt/bin/python
RUN=runners/run_tuned_controls_and_step_permutation.py

if pgrep -af "fedhad_lru.py|run_lr_uniform_control.py|fedhad_ablation.py|fedhad_r2.py|run_tuned_controls_and_step_permutation.py" >/dev/null; then
    echo "ERRO: há outra bateria de FL rodando:"; pgrep -af "fedhad_lru.py|run_lr_uniform_control.py|fedhad_ablation.py|fedhad_r2.py|run_tuned_controls_and_step_permutation.py"
    echo "Espere terminar antes de iniciar esta avaliação."; exit 2
fi

# 1) pré-registro: gravado uma única vez, antes de qualquer execução do evaluate
if [ ! -f results/tuned_controls_fedprox_validation/evaluate_preregistration.json ]; then
    $PY analysis/tuned_controls_and_step_permutation/analyze_tuned_evaluate.py --preregister
fi

# 2) execuções (FedHAD das sementes novas vai para results/step_preserving_permutation/raw)
$PY $RUN --experiment tuned_controls --phase evaluate --alpha 0.1  --seeds 42-71
$PY $RUN --experiment tuned_controls --phase evaluate --alpha 0.01 --seeds 42-71,77-166

# 3) análise mecânica pela regra pré-registrada
$PY analysis/tuned_controls_and_step_permutation/analyze_tuned_evaluate.py
echo
echo "Relatório: results/tuned_controls_fedprox_validation/evaluate_analysis/report.md"
