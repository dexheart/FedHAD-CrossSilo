# === GG MODIFICATION (header) ==============================================
# Global-gradient diagnostic battery. This file is fedhad_r2.py (reviewer_r2_controls/code)
# with ONLY the blocks marked "GG MODIFICATION" changed (see fedhad_gg.diff):
#   (a) import: unmodified r2_policy / drift_telemetry from reviewer_r2_controls/code,
#       gg_policy / gg_diag from this directory;
#   (b) partition: FL_GG_PARTITION=dirichlet (the paper's, unmodified) or equal
#       (equal-size label skew, gg_policy.equal_size_label_skew);
#   (c) plan: per-client (steps, lr) from gg_policy.build_plan (arms full_fedhad,
#       inv_epochs, fedprox_default, fedprox_tuned); FL_GG_PLAN_ONLY=<path> writes every
#       arm's plan and exits before any training (runner dry run);
#   (d) fit: training identical to train_steps, with a read-only callback at every
#       epoch boundary that records the alignment of that epoch with -g_-k;
#   (e) server: before each round the strategy computes g_k, g and g_-k at w^t and, after
#       it, the alignment of each returned update; the centralized evaluation also
#       records the confusion matrix. None of these change the aggregation or training.
# Everything else (H_k, checkpoint, worker seeding, step-budget training, FLOP
# accounting, telemetry, fail-fast) is the R2 code, byte for byte.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "reviewer_r2_controls" / "code"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
# === R2 MODIFICATION (header) ==============================================
# Copy of results_revision/results_revision_ablation/code/fedhad_ablation.py for the
# reviewer-R2 control experiments (step-preserving permutation; tuned static
# controls). The historical scripts are NOT modified. Every change is marked
# "R2 MODIFICATION" and listed in fedhad_r2.diff. Behavioural changes relative to
# the ablation copy:
#   (a) initial model loaded from a common checkpoint file, SHA-256 verified;
#   (b) deterministic seeding INSIDE each client fit (Python/NumPy/Torch/CUDA,
#       DataLoader generator, cuDNN flags), seed = f(global seed, client, round);
#   (c) local training executes an explicit number of optimizer steps (tau_k),
#       counted directly; epoch-based arms use tau_k = E_k * b_k, which is the same
#       computation as the epoch loop;
#   (d) per-client policy taken from r2_policy.build_plan (arms full_fedhad,
#       step_permuted_lrclient, step_permuted_lrfollow, static).
# -*- coding: utf-8 -*-
"""
Este script implementa uma simulação de Aprendizado Federado utilizando a biblioteca Flower.
O objetivo é demonstrar e avaliar o impacto de várias heurísticas e estratégias para lidar
com dados Não-IID (Não Independentes e Identicamente Distribuídos) e para melhorar a 
estabilidade e eficiência do treinamento federado.

Heurísticas e Estratégias Aplicadas:
-------------------------------------
1. Épocas de Treinamento Dinâmicas (por Cliente):
   - O número de épocas de treinamento local é ajustado inversamente à heterogeneidade dos dados
     do cliente. Clientes com dados mais desbalanceados (maior heterogeneidade) treinam por menos
     épocas para mitigar o "client drift".

2. Taxa de Aprendizagem Dinâmica (por Cliente):
   - A taxa de aprendizagem (learning rate) também é ajustada com base na heterogeneidade dos dados.
     Clientes com dados mais desbalanceados usam uma taxa de aprendizagem menor para um treinamento
     mais cauteloso.

3. FedProx (Termo Proximal):
   - Adiciona um termo de regularização à função de perda que penaliza os modelos locais
     quando se afastam do modelo global. Isso reduz a variância e estabiliza o
     treinamento, agindo diretamente sobre o "client drift".

Métricas de Custo Computacional (FLOPs):
-----------------------------------------
A medição de custo usa a biblioteca `thop`. IMPORTANTE para a redação da tese:
  - O `thop.profile` reporta MACs (multiply-accumulate operations), NÃO FLOPs.
  - Convenção adotada: 1 MAC = 1 multiplicação + 1 adição = 2 FLOPs.
    Logo, FLOPs(forward) = 2 × MACs (constante MACS_TO_FLOPS = 2).
  - Custo do backward ≈ 2 × forward (convenção da literatura), portanto o passe
    completo (forward + backward) = 3 × forward.
  - Combinando: FLOPs(passe completo) = 3 × (2 × MACs) = 6 × MACs por amostra.
Os valores reportados (FLOPS_PER_SAMPLE_FORWARD / _FULL) são, portanto, FLOPs
"verdadeiros" e não MACs. A mesma convenção é aplicada no FedAVG_Final.py para
garantir comparabilidade direta entre as abordagens.

Dependências para telemetria de custo/energia (instalar se necessário):
    pip install codecarbon nvidia-ml-py
O pacote `nvidia-ml-py` (módulo pynvml) é necessário para que o CodeCarbon
consiga medir a energia da GPU NVIDIA via NVML. Sem ele, a medição reflete
apenas CPU + RAM.
"""

# ==============================================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÕES GLOBAIS
# ==============================================================================

# IMPORTANTE: flwr-datasets (usado pelo FEMNIST) DEVE ser importado ANTES do torch.
# No Windows, importar flwr_datasets DEPOIS do torch causa um conflito de runtime
# OpenMP/MKL (libiomp5md.dll duplicada, puxada por datasets/pyarrow) que derruba o
# processo com ACCESS VIOLATION (0xC0000005) já nos imports — antes de qualquer print.
# Importar primeiro evita o conflito. Importação defensiva: sem a lib, os demais
# datasets continuam funcionando; só o FEMNIST fica indisponível.
try:
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import NaturalIdPartitioner
    from datasets import concatenate_datasets
    FLWR_DATASETS_AVAILABLE = True
except ImportError:
    FederatedDataset = None
    NaturalIdPartitioner = None
    concatenate_datasets = None
    FLWR_DATASETS_AVAILABLE = False
    print("Aviso: biblioteca 'flwr-datasets' não encontrada. FEMNIST indisponível. "
          "Instale com: pip install flwr-datasets[vision]")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
import flwr as fl
from collections import OrderedDict
import time
import random
import threading
from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path
from datetime import datetime
from pprint import pformat
import drift_telemetry

# thop é usado para medir FLOPs do forward pass dos modelos. Importação defensiva:
# se não estiver instalado, a simulação continua e os FLOPs ficam indisponíveis.
try:
    from thop import profile as thop_profile
    THOP_AVAILABLE = True
except ImportError:
    thop_profile = None
    THOP_AVAILABLE = False
    print("Aviso: biblioteca 'thop' não encontrada. FLOPs não serão medidos. "
          "Instale com: pip install thop")

# CodeCarbon (opcional) mede energia consumida e emissões de CO2. Importação defensiva:
# se não estiver instalado, a medição é simplesmente desabilitada (opt-in via flag).
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    print("Aviso: biblioteca 'codecarbon' não encontrada. Medição de energia indisponível. "
          "Instale com: pip install codecarbon nvidia-ml-py")


def diagnosticar_medicao_gpu() -> dict:
    """
    Diagnostica se o CodeCarbon conseguirá medir a GPU (via pynvml/NVML).
    Não quebra se pynvml estiver ausente: apenas reporta a indisponibilidade.
    """
    diagnostico = {}

    # Verifica pynvml (fornecido pelo pacote nvidia-ml-py)
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        diagnostico["pynvml_disponivel"] = True
        ver = pynvml.nvmlSystemGetDriverVersion()
        diagnostico["pynvml_versao"] = ver.decode() if isinstance(ver, bytes) else ver
        diagnostico["gpus_detectadas"] = device_count

        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                diagnostico["leitura_potencia_ok"] = True
                diagnostico["potencia_atual_W"] = power_mw / 1000.0
            except Exception as e:
                diagnostico["leitura_potencia_ok"] = False
                diagnostico["erro_potencia"] = str(e)

        pynvml.nvmlShutdown()
    except ImportError:
        diagnostico["pynvml_disponivel"] = False
        diagnostico["instrucao"] = "pip install nvidia-ml-py"
    except Exception as e:
        diagnostico["pynvml_disponivel"] = False
        diagnostico["erro"] = str(e)

    return diagnostico


# Resultado do diagnóstico de GPU, preenchido no início da execução (main).
GPU_DIAGNOSTICO = None


class _GpuEnergyEstimator:
    """
    Estima a energia da GPU integrando TDP × utilização ao longo do tempo, amostrando
    nvmlDeviceGetUtilizationRates periodicamente numa thread daemon.

    Usado como FALLBACK quando o NVML não expõe a potência real da GPU
    (ex.: RTX A1000 Laptop -> nvmlDeviceGetPowerUsage lança NVMLError_NotSupported).
    O valor resultante é uma ESTIMATIVA, não uma medição: energia ≈ Σ (TDP × util × dt).
    """
    def __init__(self, tdp_watts: float, gpu_index: int = 0, sample_secs: float = 5.0):
        self.tdp_watts = tdp_watts
        self.gpu_index = gpu_index
        self.sample_secs = sample_secs
        self._thread = None
        self._stop = threading.Event()
        self.energy_kwh = 0.0
        self.util_samples = []
        self.available = False

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            last = time.time()
            while not self._stop.is_set():
                self._stop.wait(self.sample_secs)
                now = time.time()
                dt = now - last
                last = now
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu / 100.0
                except Exception:
                    continue
                self.util_samples.append(util)
                # energia(kWh) = potencia(W) * tempo(h) / 1000; potencia ≈ TDP * util
                self.energy_kwh += (self.tdp_watts * util) * (dt / 3600.0) / 1000.0
            pynvml.nvmlShutdown()
        except Exception:
            # Falha silenciosa: o estimador apenas não contribui.
            pass

    def start(self) -> bool:
        """Inicia a amostragem. Retorna False se a utilização da GPU não puder ser lida."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            pynvml.nvmlDeviceGetUtilizationRates(handle)  # testa o suporte
            pynvml.nvmlShutdown()
            self.available = True
        except Exception:
            self.available = False
            return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> float:
        """Para a amostragem e retorna a energia estimada acumulada (kWh)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.sample_secs)
        return self.energy_kwh

# --- Configurações da Simulação ---
# << ESCOLHA O DATASET AQUI >> Opções: "MNIST", "FASHION_MNIST", "CIFAR10", "FEMNIST"
DATASET = "FEMNIST"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Setup de clientes (escolha 3, 5 ou 10) ---
CLIENT_SETUP = 5
# "femnist_writers": lista FIXA de partition IDs (cada um = 1 writer do FEMNIST) usados
# como clientes naquele setup. Devem ter o mesmo tamanho de "num_clients". São índices
# 0..3596 do NaturalIdPartitioner (ex.: o id 320 mapeia para o writer 'f0320_41').
# Usado SOMENTE quando DATASET == "FEMNIST"; ignorado pelos demais datasets.
# Se uma lista for None/ausente, o FEMNIST volta a selecionar writers aleatoriamente (seed).
CLIENT_SETUP_CONFIG = {
    3:  {"num_clients": 3,  "num_gpus": 0.3, "femnist_writers": [320, 1557, 2781]},
    5:  {"num_clients": 5,  "num_gpus": 0.2, "femnist_writers": [320, 1557, 2781, 3221, 3341]},
    10: {"num_clients": 10, "num_gpus": 0.1, "femnist_writers": [320, 1557, 2781, 3221, 3341,
                                                                 150, 900, 1700, 2400, 3100]},
}

if CLIENT_SETUP not in CLIENT_SETUP_CONFIG:
    raise ValueError("CLIENT_SETUP inválido. Use 3, 5 ou 10.")

NUM_CLIENTS = CLIENT_SETUP_CONFIG[CLIENT_SETUP]["num_clients"]
CLIENT_GPU_FRACTION = CLIENT_SETUP_CONFIG[CLIENT_SETUP]["num_gpus"]

NUM_ROUNDS = 10
BATCH_SIZE = 32
NUM_CLASSES = 10  # Para MNIST, FashionMNIST e CIFAR10

# --- Configurações das Heurísticas ---
# Particionamento Não-IID (Dirichlet)
ALPHA = 0.5  # Valores menores de alpha geram maior heterogeneidade.

# Épocas e Taxa de Aprendizagem Dinâmicas
BASE_EPOCHS = 5       # Número de épocas base
MIN_EPOCHS = 2        # Número mínimo de épocas que um cliente pode treinar
#MAX_EPOCHS = 10       # Número máximo de épocas que um cliente pode treinar
EPOCHS_DECAY_FACTOR = 3.0  # Sensibilidade das épocas à heterogeneidade normalizada
LR_DECAY_FACTOR = 1.5      # Sensibilidade do learning rate à heterogeneidade normalizada

# --- Ablação das 3 heurísticas ---
# True = habilita heurística | False = desabilita heurística
USE_DYNAMIC_EPOCHS = True
USE_DYNAMIC_LR = True
USE_FEDPROX = True

BASE_LR = 0.01       # Taxa de aprendizagem base (para Adam)
MIN_LR = 0.001       # Taxa de aprendizagem mínima

# FedProx
FEDPROX_MU = 0.01     # Hiperparâmetro de regularização (penalidade de divergência)

# --- Outras Configurações ---
COMMUNICATION_DELAY = 0.05  # Simula 20ms de delay de comunicação
TEST_NAME = "FedHAD"

# --- Reprodutibilidade ---
SEED = 42  # Semente global para todas as fontes de aleatoriedade

# --- Métricas de convergência (rounds/FLOPs/bytes até atingir cada acurácia) ---
ACCURACY_TARGETS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]  # Acurácias centrais-alvo para Rounds-to-Target e FLOPs-to-Target

# --- Medição de energia (opt-in; requer codecarbon instalado) ---
USE_ENERGY_TRACKING = True

# TDP (Thermal Design Power) da GPU em watts. Usado APENAS para ESTIMAR a energia da
# GPU quando o hardware/driver não expõe a potência real via NVML (caso de muitas GPUs
# de notebook, ex.: RTX A1000 Laptop ~ 35-50 W). Ajuste conforme a sua GPU. Não afeta a
# medição quando a leitura de potência via NVML está disponível.
GPU_TDP_WATTS = 35.0


def set_global_seed(seed: int) -> None:
    """
    Fixa TODAS as fontes de aleatoriedade para garantir reprodutibilidade.
    Deve ser chamada antes de qualquer operação que envolva aleatoriedade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# === Overrides via variáveis de ambiente (usados pelo coordenador run_experiments.py) ===
# Se a env não estiver definida, mantém-se o valor padrão acima — rodar o script
# isoladamente continua funcionando exatamente como antes.
# === R2 MODIFICATION (import) ================================================
# New module holding every change the R2 controls need (see r2_policy.py), so the
# diff applied to this copied source stays small and auditable.
import importlib
import json
# === GG MODIFICATION (import) =============================================
import r2_policy
import gg_policy
import gg_diag

import os as _os

def _env_int(name, default):
    v = _os.environ.get(name)
    return int(v) if v is not None else default

def _env_float(name, default):
    v = _os.environ.get(name)
    return float(v) if v is not None else default

def _env_bool(name, default):
    v = _os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "sim")

DATASET = _os.environ.get("FL_DATASET", DATASET)
SEED = _env_int("FL_SEED", SEED)
NUM_ROUNDS = _env_int("FL_NUM_ROUNDS", NUM_ROUNDS)
ALPHA = _env_float("FL_ALPHA", ALPHA)
COMMUNICATION_DELAY = _env_float("FL_COMM_DELAY", COMMUNICATION_DELAY)
USE_ENERGY_TRACKING = _env_bool("FL_USE_ENERGY", USE_ENERGY_TRACKING)

# Número de clientes via preset (3, 5 ou 10): re-deriva num_clients e fração de GPU.
CLIENT_SETUP = _env_int("FL_CLIENT_SETUP", CLIENT_SETUP)
if CLIENT_SETUP not in CLIENT_SETUP_CONFIG:
    raise ValueError("FL_CLIENT_SETUP inválido. Use 3, 5 ou 10.")
NUM_CLIENTS = CLIENT_SETUP_CONFIG[CLIENT_SETUP]["num_clients"]
CLIENT_GPU_FRACTION = CLIENT_SETUP_CONFIG[CLIENT_SETUP]["num_gpus"]

# FEMNIST tem 62 classes (10 dígitos + 26 maiúsculas + 26 minúsculas); os demais, 10.
# Recalculado aqui pois DATASET pode ter sido sobrescrito por env (FL_DATASET). Afeta o
# cálculo de heterogeneidade (normalização por nº de classes).
NUM_CLASSES = 62 if DATASET == "FEMNIST" else 10

# Ablação do FedHAD (cada heurística liga/desliga independentemente).
USE_DYNAMIC_EPOCHS = _env_bool("FL_USE_DYNAMIC_EPOCHS", USE_DYNAMIC_EPOCHS)
USE_DYNAMIC_LR = _env_bool("FL_USE_DYNAMIC_LR", USE_DYNAMIC_LR)
USE_FEDPROX = _env_bool("FL_USE_FEDPROX", USE_FEDPROX)

# Variáveis de CALIBRAÇÃO das heurísticas do FedHAD (teste de sensibilidade). Permitem
# variar os hiperparâmetros das heurísticas sem editar o código (via coordenador).
BASE_EPOCHS = _env_int("FL_BASE_EPOCHS", BASE_EPOCHS)
MIN_EPOCHS = _env_int("FL_MIN_EPOCHS", MIN_EPOCHS)
EPOCHS_DECAY_FACTOR = _env_float("FL_EPOCHS_DECAY", EPOCHS_DECAY_FACTOR)
LR_DECAY_FACTOR = _env_float("FL_LR_DECAY", LR_DECAY_FACTOR)
BASE_LR = _env_float("FL_BASE_LR", BASE_LR)
MIN_LR = _env_float("FL_MIN_LR", MIN_LR)
FEDPROX_MU = _env_float("FL_FEDPROX_MU", FEDPROX_MU)
# === Fim dos overrides ===


# Fixa as sementes logo no início, antes de criar modelos e carregar/particionar os dados.
set_global_seed(SEED)

print(
    f"Treinando no dispositivo: {DEVICE} com o dataset: {DATASET} | "
    f"Setup clientes: {CLIENT_SETUP} (num_gpus por cliente={CLIENT_GPU_FRACTION}) | SEED={SEED}"
)

def _format_value_for_filename(value) -> str:
    """Formata valores para uso seguro no nome do arquivo."""
    return str(value).replace(".", "_")


def _dataset_folder_name(dataset_name: str) -> str:
    mapping = {
        "MNIST": "MNIST",
        "FASHION_MNIST": "FashionMNIST",
        "CIFAR10": "CIFAR10",
        "FEMNIST": "FEMNIST",
    }
    return mapping.get(dataset_name, dataset_name)


def _get_experiment_mode() -> str:
    """Define o modo automaticamente: Standard se 3 heurísticas ativas, senão Ablation."""
    if USE_DYNAMIC_EPOCHS and USE_DYNAMIC_LR and USE_FEDPROX:
        return "STANDARD"
    return "ABLATION"


def _mode_title() -> str:
    """Retorna modo com capitalização para nomes de pasta (Standard/Ablation)."""
    return _get_experiment_mode().capitalize()


def _build_base_result_filename() -> str:
    # FEMNIST usa particionamento NATURAL (por writer); o ALPHA do Dirichlet não se
    # aplica, então o nome do arquivo recebe "Partitioning-natural" em vez de "Alpha-...".
    partitioning_tag = (
        "Partitioning-natural" if DATASET == "FEMNIST"
        else f"Alpha-{_format_value_for_filename(ALPHA)}"
    )
    return (
        f"{TEST_NAME}"
        f"__{_dataset_folder_name(DATASET)}"
        f"__Mode-{_get_experiment_mode()}"
        f"__Cfg-{_heuristics_signature()}"
        f"{_calibration_tag()}"
        f"__Setup-{CLIENT_SETUP}"
        f"__Clients-{NUM_CLIENTS}"
        f"__Rounds-{NUM_ROUNDS}"
        f"__Delay-{_format_value_for_filename(COMMUNICATION_DELAY)}"
        f"__{partitioning_tag}"
        f"__Seed-{SEED}"
    )


def _heuristics_signature() -> str:
    """Assinatura compacta das heurísticas: E=epochs, L=lr, F=fedprox."""
    return (
        f"E{int(USE_DYNAMIC_EPOCHS)}"
        f"L{int(USE_DYNAMIC_LR)}"
        f"F{int(USE_FEDPROX)}"
    )


def _calibration_tag() -> str:
    """Sufixo com os parâmetros de CALIBRAÇÃO que diferem dos defaults do FedHAD.

    Nos testes normais (tudo no default) devolve "" -> nomes de arquivo inalterados.
    Quando a calibração varia (ex.: test6 varia εd/λd), gera um sufixo único por
    combinação. Sem isto, combinações com a MESMA assinatura de heurísticas (E1L1F1)
    e mesma seed produziriam o MESMO nome de arquivo, sobrescrevendo-se umas às outras.
    """
    calib = [
        ("BE", BASE_EPOCHS, 5),          ("ME", MIN_EPOCHS, 2),
        ("ED", EPOCHS_DECAY_FACTOR, 3.0), ("LD", LR_DECAY_FACTOR, 1.5),
        ("BL", BASE_LR, 0.01),           ("ML", MIN_LR, 0.001),
        ("MU", FEDPROX_MU, 0.01),
    ]
    partes = [f"{sig}{_format_value_for_filename(val)}"
              for sig, val, default in calib if abs(float(val) - float(default)) > 1e-9]
    return ("__Cal-" + "-".join(partes)) if partes else ""


def _get_unique_result_path(results_dir: Path, base_name: str, extension: str = ".txt") -> Path:
    increment = 1
    candidate = results_dir / f"{base_name}__{increment}{extension}"

    while candidate.exists():
        increment += 1
        candidate = results_dir / f"{base_name}__{increment}{extension}"

    return candidate


# Coletor global das métricas brutas de fit, preenchido por fit_metrics_aggregation_fn.
# Cada elemento corresponde a uma rodada e contém a lista de observações cliente-rodada
# no formato (num_examples, métricas). Necessário porque history.metrics_distributed_fit
# guarda apenas os valores já agregados por rodada, não as observações individuais.
fit_metrics_history: List[Tuple[int, Dict]] = []


def _format_lr(value: float) -> str:
    """Formata um learning rate com 6 casas decimais (ex: '0.004321')."""
    return f"{value:.6f}"


def _build_dynamic_heuristics_report_lines(history) -> List[str]:
    """
    Constrói as seções de telemetria das heurísticas dinâmicas (épocas e learning
    rate adaptativos) a partir das métricas de fit coletadas durante a simulação.

    Usa o coletor global `fit_metrics_history` (observações cliente-rodada) para as
    estatísticas detalhadas/por cliente e `history.metrics_distributed_fit` para o
    histórico por rodada. Trata o caso de coleta vazia sem quebrar o relatório.
    """
    lines: List[str] = ["", "--- Estatísticas de Heurísticas Dinâmicas ---"]

    # Agrega todas as observações cliente-rodada a partir do coletor global.
    all_epochs: List[float] = []
    all_lrs: List[float] = []
    all_weights: List[float] = []
    per_client: Dict[str, Dict[str, list]] = {}
    per_round_epoch_means: List[float] = []

    for round_metrics in fit_metrics_history:
        round_epochs: List[float] = []
        for num_examples, m in round_metrics:
            # Ignora observações sem os campos esperados (robustez).
            if "epochs" not in m or "learning_rate" not in m:
                continue
            ep = float(m["epochs"])
            lr = float(m["learning_rate"])
            all_epochs.append(ep)
            all_lrs.append(lr)
            all_weights.append(float(num_examples))
            round_epochs.append(ep)

            cid = str(m.get("cid", "?"))
            bucket = per_client.setdefault(cid, {"epochs": [], "lrs": [], "examples": []})
            bucket["epochs"].append(ep)
            bucket["lrs"].append(lr)
            bucket["examples"].append(float(num_examples))
        if round_epochs:
            per_round_epoch_means.append(sum(round_epochs) / len(round_epochs))

    if not all_epochs:
        # Coleta vazia: não quebra o relatório.
        lines.append("Nenhuma métrica de fit coletada (metrics_distributed_fit vazio).")
        return lines

    epochs_arr = np.array(all_epochs, dtype=float)
    lrs_arr = np.array(all_lrs, dtype=float)
    weights_arr = np.array(all_weights, dtype=float)
    total_w = float(weights_arr.sum())

    # --- Estatísticas de épocas ---
    epochs_weighted_mean = float((epochs_arr * weights_arr).sum() / total_w) if total_w > 0 else 0.0
    epochs_simple_mean = float(np.mean(per_round_epoch_means)) if per_round_epoch_means else 0.0
    epochs_min = int(epochs_arr.min())
    epochs_max = int(epochs_arr.max())
    epochs_median = float(np.median(epochs_arr))
    epochs_std = float(epochs_arr.std())

    # Distribuição: quantas vezes cada valor inteiro de épocas foi usado.
    epochs_distribution: Dict[int, int] = {}
    for ep in all_epochs:
        key = int(round(ep))
        epochs_distribution[key] = epochs_distribution.get(key, 0) + 1
    epochs_distribution = dict(sorted(epochs_distribution.items()))

    lines.append("Épocas (todas as observações cliente-rodada):")
    lines.append(f"  Média ponderada por amostras: {epochs_weighted_mean:.2f}")
    lines.append(f"  Média aritmética simples: {epochs_simple_mean:.2f}")
    lines.append(
        f"  Mínimo: {epochs_min} | Máximo: {epochs_max} | "
        f"Mediana: {epochs_median:.2f} | Desvio padrão: {epochs_std:.2f}"
    )
    lines.append(f"  Distribuição: {epochs_distribution}")
    lines.append("")

    # --- Estatísticas de learning rate ---
    lr_weighted_mean = float((lrs_arr * weights_arr).sum() / total_w) if total_w > 0 else 0.0
    lr_min = float(lrs_arr.min())
    lr_max = float(lrs_arr.max())
    lr_median = float(np.median(lrs_arr))
    lr_std = float(lrs_arr.std())

    lines.append("Learning Rate (todas as observações cliente-rodada):")
    lines.append(f"  Média ponderada por amostras: {_format_lr(lr_weighted_mean)}")
    lines.append(f"  Mínimo: {_format_lr(lr_min)} | Máximo: {_format_lr(lr_max)}")
    lines.append(f"  Mediana: {_format_lr(lr_median)} | Desvio padrão: {_format_lr(lr_std)}")

    # --- Histórico adaptativo por rodada (a partir de metrics_distributed_fit) ---
    lines.append("")
    lines.append("--- Histórico Adaptativo por Rodada ---")
    mdf = getattr(history, "metrics_distributed_fit", {}) or {}
    epochs_per_round = dict(mdf.get("avg_epochs_weighted", []))
    lr_per_round = dict(mdf.get("avg_lr_weighted", []))
    rounds = sorted(set(epochs_per_round) | set(lr_per_round))
    if rounds:
        for r in rounds:
            avg_ep = float(epochs_per_round.get(r, 0.0))
            avg_lr = float(lr_per_round.get(r, 0.0))
            lines.append(f"Rodada {r}: avg_epochs={avg_ep:.2f}, avg_lr={_format_lr(avg_lr)}")
    else:
        lines.append("Sem histórico por rodada disponível.")

    # --- Estatísticas por cliente (médias ao longo de todas as rodadas) ---
    lines.append("")
    lines.append("--- Estatísticas por Cliente ---")

    def _cid_sort_key(c: str):
        # Ordena numericamente quando o cid é um inteiro; senão, alfabeticamente.
        return (0, int(c)) if c.isdigit() else (1, c)

    for cid in sorted(per_client.keys(), key=_cid_sort_key):
        bucket = per_client[cid]
        n_obs = len(bucket["epochs"])
        if n_obs == 0:
            continue
        avg_epochs = sum(bucket["epochs"]) / n_obs
        avg_lr = sum(bucket["lrs"]) / n_obs
        # num_examples por cliente é constante entre rodadas; soma/num_rodadas
        # recupera o tamanho do dataset local do cliente.
        amostras = int(round(sum(bucket["examples"]) / n_obs))
        lines.append(
            f"Cliente {cid}: amostras={amostras}, "
            f"avg_epochs={avg_epochs:.2f}, avg_lr={_format_lr(avg_lr)}"
        )

    return lines


def _format_flops(flops) -> str:
    """Formata uma contagem de FLOPs em unidade legível (KFLOPs..TFLOPs)."""
    if flops is None:
        return "N/A"
    for factor, name in ((1e12, "TFLOPs"), (1e9, "GFLOPs"), (1e6, "MFLOPs"), (1e3, "KFLOPs")):
        if abs(flops) >= factor:
            return f"{flops / factor:.3f} {name}"
    return f"{flops:.0f} FLOPs"


def _build_flops_report_lines(history) -> List[str]:
    """
    Constrói a seção de telemetria de custo computacional (FLOPs) do relatório.

    Usa as constantes globais FLOPS_PER_SAMPLE_* (auditoria), o coletor global
    `fit_metrics_history` (FLOPs por cliente) e `history.metrics_distributed_fit`
    (FLOPs por rodada). Degrada graciosamente se thop não estiver disponível.
    """
    lines: List[str] = ["", "--- Estatísticas de Custo Computacional (FLOPs) ---"]

    if FLOPS_PER_SAMPLE_FORWARD is None:
        lines.append("FLOPs não disponíveis (biblioteca 'thop' ausente ou falha na medição).")
        lines.append("Instale com: pip install thop")
        return lines

    # Constantes por amostra (auditoria).
    lines.append(
        f"FLOPS por amostra (forward apenas): {FLOPS_PER_SAMPLE_FORWARD:.3e} "
        f"({FLOPS_PER_SAMPLE_FORWARD / 1e6:.1f} MFLOPs)"
    )
    lines.append(
        f"FLOPS por amostra (forward + backward, fator 3x): {FLOPS_PER_SAMPLE_FULL:.3e} "
        f"({FLOPS_PER_SAMPLE_FULL / 1e6:.1f} MFLOPs)"
    )
    lines.append("(Convenção: thop mede MACs; FLOPs = 2 × MACs; passe completo = 3 × forward.)")
    lines.append("")

    # FLOPs por rodada (a partir de metrics_distributed_fit).
    mdf = getattr(history, "metrics_distributed_fit", {}) or {}
    flops_per_round = [float(v) for _, v in mdf.get("total_flops_round", [])]

    # FLOPs acumulados por cliente (a partir do coletor global de observações brutas).
    per_client_flops: Dict[str, float] = {}
    per_client_info: Dict[str, Dict] = {}
    for round_metrics in fit_metrics_history:
        for num_examples, m in round_metrics:
            cid = str(m.get("cid", "?"))
            fl_local = float(m.get("flops_locais", 0) or 0)
            per_client_flops[cid] = per_client_flops.get(cid, 0.0) + fl_local
            info = per_client_info.setdefault(cid, {"examples": int(num_examples), "epochs": [], "rounds": 0})
            info["epochs"].append(int(m.get("epochs", 0)))
            info["rounds"] += 1

    total_flops = sum(flops_per_round) if flops_per_round else sum(per_client_flops.values())

    lines.append(f"Total acumulado da execução: {_format_flops(total_flops)}")
    if flops_per_round:
        arr = np.array(flops_per_round, dtype=float)
        lines.append(f"Média por rodada: {_format_flops(float(arr.mean()))}")
        lines.append(
            f"Min/Max por rodada: {_format_flops(float(arr.min()))} / {_format_flops(float(arr.max()))} "
            f"| Desvio padrão: {_format_flops(float(arr.std()))}"
        )
    else:
        lines.append("Sem histórico de FLOPs por rodada disponível.")
    lines.append("")

    # FLOPs por cliente (acumulado ao longo das rodadas).
    lines.append(f"FLOPs por cliente (acumulado em {NUM_ROUNDS} rodadas):")

    def _cid_sort_key(c: str):
        return (0, int(c)) if c.isdigit() else (1, c)

    for cid in sorted(per_client_flops.keys(), key=_cid_sort_key):
        info = per_client_info.get(cid, {})
        amostras = int(info.get("examples", 0))
        eps = info.get("epochs", [])
        avg_ep = (sum(eps) / len(eps)) if eps else 0.0
        lines.append(
            f"  Cliente {cid}: {_format_flops(per_client_flops[cid])} "
            f"({amostras} amostras × ~{avg_ep:.0f} épocas × {info.get('rounds', 0)} rodadas)"
        )
    lines.append("")

    # Eficiência: acurácia centralizada final / total de TFLOPs.
    mc = getattr(history, "metrics_centralized", {}) or {}
    acc_hist = mc.get("accuracy", [])
    final_acc = float(acc_hist[-1][1]) if acc_hist else None
    total_tflops = total_flops / 1e12 if total_flops else 0.0

    if final_acc is not None and total_tflops > 0:
        eff = final_acc / total_tflops
        lines.append(
            f"Eficiência: {final_acc:.4f} acurácia / {total_tflops:.3f} TFLOPs = "
            f"{eff:.4f} (acc/TFLOP)"
        )
        lines.append(f"           = {eff * 100:.2f} pontos percentuais por TFLOP")
    else:
        lines.append("Eficiência: não disponível (acurácia centralizada ou FLOPs ausentes).")

    return lines


def _get_results_dir() -> Path:
    """
    Retorna (criando, se preciso) a pasta de resultados desta execução.
    Se FL_RUN_TAG estiver definido (pelo coordenador run_experiments.py), os resultados
    vão para uma subpasta com esse nome — ex.:
    FedHAD-results/test1_convergencia/<dataset>-<Mode>/ —, mantendo cada experimento da
    tese separado e organizado.
    """
    # R2 MODIFICATION: the runner points the standard text report to the run's own
    # output directory (FL_RESULTS_DIR); never into the historical result trees.
    base = Path(_os.environ.get("FL_RESULTS_DIR", "").strip() or
                (Path(__file__).resolve().parent / "FedHAD-results"))
    tag = _os.environ.get("FL_RUN_TAG", "").strip()
    if tag:
        base = base / tag
    dataset_dir = base / f"{_dataset_folder_name(DATASET)}-{_mode_title()}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def _get_final_accuracy(history):
    """Extrai a acurácia centralizada final do history (ou None)."""
    mc = getattr(history, "metrics_centralized", {}) or {}
    acc_hist = mc.get("accuracy", [])
    return float(acc_hist[-1][1]) if acc_hist else None


def _get_flops_per_round_dict(history) -> Dict[int, float]:
    """Retorna {round: total_flops_round} a partir de history.metrics_distributed_fit."""
    mdf = getattr(history, "metrics_distributed_fit", {}) or {}
    return {int(r): float(v) for r, v in mdf.get("total_flops_round", [])}


def _format_bytes(num_bytes) -> str:
    """Formata uma contagem de bytes em unidade legível (B..GB, base decimal)."""
    if num_bytes is None:
        return "N/A"
    for factor, name in ((1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if abs(num_bytes) >= factor:
            return f"{num_bytes / factor:.3f} {name}"
    return f"{num_bytes:.0f} B"


def calcular_rounds_to_target(accuracy_history, targets) -> Dict[float, int]:
    """
    Recebe o histórico de acurácia centralizada [(round, acc), ...] e uma lista de
    alvos. Para cada alvo, retorna o primeiro round em que a acurácia o atinge ou
    ultrapassa, ou None se nunca atingido durante a execução.
    """
    sorted_hist = sorted(accuracy_history, key=lambda t: t[0])
    result: Dict[float, int] = {}
    for target in targets:
        hit = None
        for rnd, acc in sorted_hist:
            if acc >= target:
                hit = int(rnd)
                break
        result[target] = hit
    return result


def calcular_bytes_por_round(net, num_clients) -> int:
    """
    Calcula os bytes transmitidos por rodada: tamanho do modelo (params × 4 bytes,
    float32) multiplicado por num_clients × 2 (upload + download por cliente).
    """
    total_params = sum(p.numel() for p in net.parameters())
    model_bytes = total_params * 4  # float32 = 4 bytes
    return model_bytes * num_clients * 2


def _build_rounds_to_target_report_lines(history) -> List[str]:
    """Constrói as seções Rounds-to-Target e FLOPs-to-Target do relatório."""
    lines: List[str] = ["", "--- Rounds-to-Target (Acurácia Centralizada) ---"]
    mc = getattr(history, "metrics_centralized", {}) or {}
    acc_hist = mc.get("accuracy", [])
    if not acc_hist:
        lines.append("Histórico de acurácia centralizada não disponível.")
        return lines

    r2t = calcular_rounds_to_target(acc_hist, ACCURACY_TARGETS)
    for target in ACCURACY_TARGETS:
        rnd = r2t.get(target)
        if rnd is not None:
            lines.append(f"Target {target * 100:.0f}%: round {rnd}")
        else:
            lines.append(f"Target {target * 100:.0f}%: não atingido em {NUM_ROUNDS} rounds")

    # FLOPs acumulados até atingir cada target.
    flops_per_round = _get_flops_per_round_dict(history)
    if flops_per_round:
        lines.append("")
        lines.append("--- FLOPs-to-Target ---")
        for target in ACCURACY_TARGETS:
            rnd = r2t.get(target)
            if rnd is not None:
                cum_flops = sum(v for r, v in flops_per_round.items() if r <= rnd)
                lines.append(f"Target {target * 100:.0f}%: {_format_flops(cum_flops)} (round {rnd})")
            else:
                lines.append(f"Target {target * 100:.0f}%: não atingido em {NUM_ROUNDS} rounds")
    return lines


def _build_communication_report_lines(history, net) -> List[str]:
    """Constrói a seção Custo de Comunicação (bytes transmitidos)."""
    lines: List[str] = ["", "--- Custo de Comunicação ---"]
    total_params = sum(p.numel() for p in net.parameters())
    model_bytes = total_params * 4
    bytes_per_round = calcular_bytes_por_round(net, NUM_CLIENTS)

    # Número de rodadas de fit efetivas (a partir do histórico de FLOPs por rodada).
    flops_per_round = _get_flops_per_round_dict(history)
    n_fit_rounds = len(flops_per_round) if flops_per_round else NUM_ROUNDS
    total_bytes = bytes_per_round * n_fit_rounds

    lines.append(f"Tamanho do modelo: {_format_bytes(model_bytes)} ({total_params} parâmetros)")
    lines.append(
        f"Bytes por rodada: {_format_bytes(bytes_per_round)} "
        f"({NUM_CLIENTS} clientes × 2 × {_format_bytes(model_bytes)})"
    )
    lines.append(f"Bytes acumulados na execução: {_format_bytes(total_bytes)} ({n_fit_rounds} rounds)")

    # Bytes acumulados até atingir cada target de acurácia.
    mc = getattr(history, "metrics_centralized", {}) or {}
    acc_hist = mc.get("accuracy", [])
    if acc_hist:
        r2t = calcular_rounds_to_target(acc_hist, ACCURACY_TARGETS)
        for target in ACCURACY_TARGETS:
            rnd = r2t.get(target)
            if rnd is not None:
                lines.append(
                    f"Bytes até atingir target {target * 100:.0f}%: "
                    f"{_format_bytes(bytes_per_round * rnd)} (round {rnd})"
                )
            else:
                lines.append(
                    f"Bytes até atingir target {target * 100:.0f}%: não atingido em {NUM_ROUNDS} rounds"
                )
    return lines


def _criar_emissions_tracker(dataset_dir):
    """
    Cria o EmissionsTracker passando apenas os parâmetros suportados pela versão
    instalada do CodeCarbon (parâmetros não reconhecidos são ignorados com aviso).
    Retorna o tracker, ou None em caso de falha.
    """
    desired = {
        "project_name": f"{TEST_NAME}_{DATASET}",
        "output_dir": str(dataset_dir),
        # log_level="error": evita o spam de WARNINGs quando a GPU não expõe potência
        # via NVML (a cada ciclo de medição). O diagnóstico de GPU já informa a causa.
        # Use "warning" temporariamente se precisar depurar o CodeCarbon.
        "log_level": "error",
        # save_to_file=False: os dados de energia já são extraídos para o
        # relatório .txt. Mude para True se desejar auditoria via CSV.
        "save_to_file": False,
        "measure_power_secs": 5,        # medir a cada 5s (captura melhor picos de consumo)
        "tracking_mode": "machine",     # mede toda a máquina, não só o processo
        "gpu_ids": [0],                 # explicitamente medir a GPU 0
        "allow_multiple_runs": True,    # importante para execuções repetidas
    }
    try:
        import inspect
        params = inspect.signature(EmissionsTracker.__init__).parameters
        accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
        if accepts_var_kw:
            supported = desired
        else:
            supported = {k: v for k, v in desired.items() if k in params}
            dropped = [k for k in desired if k not in supported]
            if dropped:
                print(f"Aviso: parâmetros do EmissionsTracker ignorados (não suportados "
                      f"nesta versão do CodeCarbon): {dropped}")
        return EmissionsTracker(**supported)
    except Exception as e:
        print(f"Aviso: falha ao criar EmissionsTracker com config completa ({e}). "
              f"Tentando configuração mínima.")
        try:
            return EmissionsTracker(
                project_name=f"{TEST_NAME}_{DATASET}",
                output_dir=str(dataset_dir),
                log_level="error",
                save_to_file=False,
            )
        except Exception as e2:
            print(f"Aviso: falha ao criar EmissionsTracker ({e2}).")
            return None


def _collect_energy_info(tracker):
    """
    Para a medição do CodeCarbon e extrai os dados de forma defensiva.
    Retorna um dict com os campos disponíveis, ou None em caso de falha.
    """
    if tracker is None:
        return None
    try:
        emissions_kg = tracker.stop()
    except Exception as e:
        print(f"Aviso: falha ao parar o CodeCarbon ({e}).")
        return None

    info = {"emissions_kg": emissions_kg}
    data = getattr(tracker, "final_emissions_data", None)
    if data is not None:
        for attr in (
            "energy_consumed", "cpu_energy", "gpu_energy", "ram_energy", "emissions",
            "country_name", "country_iso_code", "cpu_model", "gpu_model",
            "cpu_count", "gpu_count",
        ):
            info[attr] = getattr(data, attr, None)
    return info


def _build_energy_report_lines(energy_info, history) -> List[str]:
    """Constrói a seção Consumo de Energia (CodeCarbon), degradando graciosamente."""
    lines: List[str] = ["", "--- Consumo de Energia (CodeCarbon) ---"]

    if not USE_ENERGY_TRACKING:
        lines.append("Medição de energia desabilitada (USE_ENERGY_TRACKING = False).")
        return lines
    if not CODECARBON_AVAILABLE:
        lines.append("Medição de energia não disponível (biblioteca 'codecarbon' ausente).")
        lines.append("Instale com: pip install codecarbon")
        return lines
    if energy_info is None:
        lines.append("Medição de energia não disponível (falha no CodeCarbon).")
        return lines

    # País e hardware detectados.
    country = energy_info.get("country_name")
    iso = energy_info.get("country_iso_code")
    if country:
        lines.append(f"País detectado: {country}" + (f" ({iso})" if iso else ""))
    else:
        lines.append(
            "País detectado: não disponível (CodeCarbon pode ter usado valor padrão/cache; "
            "recomenda-se conexão à internet na primeira execução)"
        )

    energy_kwh = energy_info.get("energy_consumed")
    emissions_kg = energy_info.get("emissions_kg")
    if emissions_kg is None:
        emissions_kg = energy_info.get("emissions")

    # Intensidade de carbono efetiva da rede (emissões / energia).
    if energy_kwh and emissions_kg is not None and energy_kwh > 0:
        lines.append(f"Intensidade de carbono da rede: {emissions_kg / energy_kwh:.4f} kg CO2eq/kWh")

    gpu_model = energy_info.get("gpu_model")
    cpu_model = energy_info.get("cpu_model")
    if gpu_model:
        lines.append(f"GPU: {gpu_model}")
    if cpu_model:
        lines.append(f"CPU: {cpu_model}")
    lines.append("")

    # Energia total e breakdown por componente.
    if energy_kwh is not None:
        lines.append(f"Energia total consumida: {energy_kwh:.6f} kWh")
        if energy_kwh > 0:
            for label, key in (("GPU", "gpu_energy"), ("CPU", "cpu_energy"), ("RAM", "ram_energy")):
                comp = energy_info.get(key)
                if comp is not None:
                    lines.append(f"  - {label}: {comp:.6f} kWh ({comp / energy_kwh * 100:.1f}%)")

    if emissions_kg is not None:
        lines.append(f"Emissões de CO2 equivalente: {emissions_kg:.6e} kg CO2eq")

    # --- Estimativa de fallback da energia da GPU (TDP × utilização) ---
    gpu_est = energy_info.get("gpu_energy_estimada_kwh")
    if gpu_est is not None:
        lines.append("")
        lines.append("GPU (ESTIMATIVA de fallback — potência real indisponível via NVML):")
        util_med = energy_info.get("gpu_util_media")
        tdp = energy_info.get("gpu_tdp_watts")
        detalhe = f"TDP={tdp} W"
        if util_med is not None:
            detalhe += f" × utilização média {util_med * 100:.1f}%"
        lines.append(f"  Energia GPU estimada: {gpu_est:.6f} kWh ({detalhe})")
        if energy_kwh is not None:
            lines.append(
                f"  Energia total AJUSTADA (CPU+RAM medidos + GPU estimada): "
                f"{energy_kwh + gpu_est:.6f} kWh"
            )
        lines.append(
            "  AVISO: valor ESTIMADO (não medido). Mantenha o MESMO GPU_TDP_WATTS ao "
            "comparar métodos; ajuste a constante conforme a sua GPU."
        )

    # Eficiência energética: energia por TFLOP e por ponto percentual de acurácia.
    total_flops = sum(_get_flops_per_round_dict(history).values())
    total_tflops = total_flops / 1e12 if total_flops else 0.0
    if energy_kwh is not None and total_tflops > 0:
        lines.append(f"Energia por TFLOP processado: {energy_kwh / total_tflops:.3e} kWh/TFLOP")
        if emissions_kg is not None:
            lines.append(f"Emissões por TFLOP processado: {emissions_kg / total_tflops:.3e} kg CO2eq/TFLOP")

    final_acc = _get_final_accuracy(history)
    if energy_kwh is not None and final_acc is not None and final_acc > 0:
        pp = final_acc * 100  # pontos percentuais
        lines.append(f"Energia por ponto percentual de acurácia final: {energy_kwh / pp:.3e} kWh/pp")
        if emissions_kg is not None:
            lines.append(f"Emissões por ponto percentual de acurácia final: {emissions_kg / pp:.3e} kg CO2eq/pp")

    # --- Sub-seção de diagnóstico da medição (pynvml/GPU) ---
    lines.append("")
    lines.append("--- Diagnóstico de Medição (CodeCarbon) ---")
    diag = GPU_DIAGNOSTICO or {}
    pynvml_ok = diag.get("pynvml_disponivel")
    lines.append(f"pynvml disponível: {pynvml_ok}")
    if pynvml_ok:
        lines.append(f"GPUs detectadas: {diag.get('gpus_detectadas', 'N/A')}")
        lines.append("Modo de tracking: machine")
        lines.append("Intervalo de medição: 5s")
        if diag.get("leitura_potencia_ok"):
            lines.append(
                f"Status da medição de GPU: OK "
                f"({diag.get('potencia_atual_W', 0.0):.2f} W na leitura de diagnóstico)"
            )
        else:
            lines.append(
                f"Status da medição de GPU: leitura de potência falhou "
                f"({diag.get('erro_potencia', 'erro desconhecido')})"
            )
    else:
        lines.append("ATENÇÃO: A medição de GPU não está funcionando.")
        lines.append("Instale com: pip install nvidia-ml-py")
        if diag.get("erro"):
            lines.append(f"Erro: {diag.get('erro')}")
        lines.append("A medição atual reflete apenas CPU + RAM.")

    # --- Aviso automático se a energia atribuída à GPU for 0 ---
    gpu_e = energy_info.get("gpu_energy")
    if gpu_e is not None and gpu_e == 0:
        lines.append("")
        lines.append("ATENÇÃO: GPU com 0.0 kWh na medição direta do CodeCarbon.")
        if energy_info.get("gpu_energy_estimada_kwh") is not None:
            lines.append("  -> Uma ESTIMATIVA de energia da GPU (TDP×utilização) foi fornecida acima, "
                         "pois o NVML desta GPU não expõe a potência real (NVMLError_NotSupported).")
        else:
            lines.append("Os números de energia refletem apenas CPU e RAM. Para corrigir:")
            lines.append("  1. Instale pynvml: pip install nvidia-ml-py")
            lines.append("  2. Verifique se nvidia-smi funciona no terminal")
            lines.append("  3. Se a GPU não suportar leitura de potência (comum em GPUs de "
                         "notebook), ative a estimativa via TDP (GPU_TDP_WATTS)")

    lines.append("")
    lines.append(
        "Nota: o consumo absoluto de energia é dependente do hardware. A comparação "
        "válida é entre métodos executados na MESMA máquina."
    )
    return lines


def save_simulation_results(history, elapsed_seconds: float, energy_info=None) -> Path:
    """Salva os resultados da execução em FedHAD-results/<dataset>."""
    dataset_dir = _get_results_dir()

    file_base_name = _build_base_result_filename()
    # Nome do arquivo = parâmetros (já únicos por seed); re-rodar a mesma config sobrescreve.
    result_file_path = dataset_dir / f"{file_base_name}.txt"

    report_lines = [
        "=== Resultado da Simulação Federada ===",
        f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Teste: {TEST_NAME}",
        f"Dataset: {_dataset_folder_name(DATASET)}",
        f"Device: {DEVICE}",
        f"EXPERIMENT_MODE: {_get_experiment_mode()}",
        f"HEURISTICS_SIGNATURE: {_heuristics_signature()}",
        "",
        "--- Parâmetros ---",
        f"EXPERIMENT_TAG: {_os.environ.get('FL_RUN_TAG', '')}",
        f"SEED: {SEED}",
        f"CLIENT_SETUP: {CLIENT_SETUP}",
        f"NUM_CLIENTS: {NUM_CLIENTS}",
        f"CLIENT_GPU_FRACTION: {CLIENT_GPU_FRACTION}",
        f"NUM_ROUNDS: {NUM_ROUNDS}",
        f"BATCH_SIZE: {BATCH_SIZE}",
        f"COMMUNICATION_DELAY: {COMMUNICATION_DELAY}",
        f"PARTITIONING: {'NATURAL (per-writer)' if DATASET == 'FEMNIST' else f'DIRICHLET (alpha={ALPHA})'}",
        f"ALPHA: {'N/A (FEMNIST usa particionamento natural)' if DATASET == 'FEMNIST' else ALPHA}",
        f"NUM_CLASSES: {NUM_CLASSES}",
        f"BASE_EPOCHS: {BASE_EPOCHS}",
        f"MIN_EPOCHS: {MIN_EPOCHS}",
        f"EPOCHS_DECAY_FACTOR: {EPOCHS_DECAY_FACTOR}",
        f"LR_DECAY_FACTOR: {LR_DECAY_FACTOR}",
        f"USE_DYNAMIC_EPOCHS: {USE_DYNAMIC_EPOCHS}",
        f"USE_DYNAMIC_LR: {USE_DYNAMIC_LR}",
        f"USE_FEDPROX: {USE_FEDPROX}",
        f"BASE_LR: {BASE_LR}",
        f"MIN_LR: {MIN_LR}",
        f"FEDPROX_MU: {FEDPROX_MU}",
        f"Tempo total (s): {elapsed_seconds:.2f}",
        "",
        "--- History (loss, distributed) ---",
        pformat(history.losses_distributed),
        "",
        "--- History (loss, centralized) ---",
        pformat(history.losses_centralized),
        "",
        "--- History (metrics, distributed, evaluate) ---",
        pformat(history.metrics_distributed),
        "",
        "--- History (metrics, centralized) ---",
        pformat(history.metrics_centralized),
        "",
        "--- History (metrics, distributed, fit) ---",
        pformat(history.metrics_distributed_fit),
    ]

    # Anexa as seções de telemetria das heurísticas dinâmicas.
    report_lines.extend(_build_dynamic_heuristics_report_lines(history))

    # Anexa a seção de custo computacional (FLOPs).
    report_lines.extend(_build_flops_report_lines(history))

    # Anexa as seções de convergência (rounds/FLOPs-to-target).
    report_lines.extend(_build_rounds_to_target_report_lines(history))

    # Anexa a seção de custo de comunicação (bytes transmitidos).
    report_lines.extend(_build_communication_report_lines(history, get_net()))

    # Anexa a seção de consumo de energia (CodeCarbon).
    report_lines.extend(_build_energy_report_lines(energy_info, history))

    result_file_path.write_text("\n".join(report_lines), encoding="utf-8")
    return result_file_path

# ==============================================================================
# 2. DEFINIÇÃO DOS MODELOS DE REDE NEURAL
# ==============================================================================

class Net_MNIST_Fashion(nn.Module):
    """CNN para datasets com imagens 1x28x28 (MNIST, FashionMNIST)."""
    def __init__(self) -> None:
        super(Net_MNIST_Fashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# CNN para CIFAR-10 (imagens 3x32x32)
class Net_CIFAR10(nn.Module):
    def __init__(self):
        super(Net_CIFAR10, self).__init__()
        # --- BLOCO 1 ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # -> 16x16

        # --- BLOCO 2 ---
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # -> 8x8

        # --- BLOCO 3 ---
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # -> 4x4

        # --- CAMADAS LINEARES ---
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # BLOCO 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # BLOCO 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # BLOCO 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)

        # CLASSIFICADOR
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class Net_FEMNIST(nn.Module):
    """
    CNN dedicada para FEMNIST (imagens 1x28x28, 62 classes:
    10 dígitos + 26 maiúsculas + 26 minúsculas).

    Mais robusta que a LeNet de Net_MNIST_Fashion para lidar com o maior número de
    classes (62 vs 10), a variabilidade entre writers e as ambiguidades visuais.
    Usa BatchNorm para estabilizar o treino federado e Dropout contra overfitting.
    """
    def __init__(self) -> None:
        super(Net_FEMNIST, self).__init__()
        # Bloco 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # Bloco 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        # Classificador
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 62)  # 62 classes do FEMNIST

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def get_net():
    """Função fábrica para selecionar a arquitetura de rede correta com base no dataset."""
    if DATASET == "CIFAR10":
        return Net_CIFAR10()
    if DATASET == "FEMNIST":
        return Net_FEMNIST()
    return Net_MNIST_Fashion()


def _get_model_input_shape() -> Tuple[int, int, int]:
    """Retorna o shape (C, H, W) de uma única amostra conforme o dataset selecionado."""
    if DATASET == "CIFAR10":
        return (3, 32, 32)
    return (1, 28, 28)  # MNIST, FASHION_MNIST e FEMNIST


# Fator de conversão MAC -> FLOP. thop reporta MACs (multiply-accumulate);
# 1 MAC = 1 multiplicação + 1 adição = 2 FLOPs. Ver nota no cabeçalho do arquivo.
MACS_TO_FLOPS = 2


def calcular_flops_por_amostra(net, input_shape):
    """
    Calcula os FLOPs do forward pass para uma única amostra (batch=1) usando thop.
    O thop reporta MACs, então convertemos para FLOPs verdadeiros multiplicando por
    MACS_TO_FLOPS (=2). Retorna o número de FLOPs como int, ou None se thop não
    estiver disponível ou se a medição falhar (a simulação continua normalmente).
    """
    if not THOP_AVAILABLE:
        return None
    try:
        # Coloca o tensor dummy no mesmo device dos parâmetros do modelo.
        try:
            device = next(net.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        dummy = torch.randn(1, *input_shape, device=device)
        macs, _ = thop_profile(net, inputs=(dummy,), verbose=False)
        # Converte MACs -> FLOPs verdadeiros (forward).
        return int(macs * MACS_TO_FLOPS)
    except Exception as e:
        print(f"Aviso: falha ao calcular FLOPs com thop ({e}). Prosseguindo com flops=None.")
        return None


# Calcula os FLOPs por amostra (forward) uma única vez, num modelo limpo na CPU.
FLOPS_PER_SAMPLE_FORWARD = calcular_flops_por_amostra(get_net(), _get_model_input_shape())
# Convenção da literatura: backward custa ~2x o forward => passe completo ~3x forward.
FLOPS_PER_SAMPLE_FULL = (
    FLOPS_PER_SAMPLE_FORWARD * 3 if FLOPS_PER_SAMPLE_FORWARD is not None else None
)

if FLOPS_PER_SAMPLE_FORWARD is not None:
    print(
        f"FLOPS por amostra (forward): {FLOPS_PER_SAMPLE_FORWARD:.3e} | "
        f"(forward+backward, 3x): {FLOPS_PER_SAMPLE_FULL:.3e}"
    )

# ==============================================================================
# 3. FUNÇÕES DAS HEURÍSTICAS
# ==============================================================================

def calcular_heterogeneidade(cid: str, dataloader: DataLoader, num_classes: int) -> float:
    """
    Calcula uma métrica de heterogeneidade para um cliente com base na distribuição de classes
    de seus dados locais. A métrica é o desvio padrão normalizado da contagem de cada classe.
    Um valor mais alto indica maior desbalanceamento (maior heterogeneidade).
    """
    if not hasattr(dataloader.dataset, 'indices') or not dataloader.dataset.indices:
        return 0.0

    # Navega pela estrutura de Subsets para encontrar os índices e o dataset original
    current_subset = dataloader.dataset
    final_indices = list(current_subset.indices)
    while isinstance(current_subset.dataset, Subset):
        current_subset = current_subset.dataset
        parent_indices = list(current_subset.indices)
        final_indices = [parent_indices[i] for i in final_indices]

    original_dataset = current_subset.dataset
    
    try:
        all_labels = torch.as_tensor(original_dataset.targets)
        labels = all_labels[final_indices]
    except Exception as e:
        print(f"Aviso (Cliente {cid}): Falha ao acessar rótulos. Iterando sobre o dataloader (lento). Erro: {e}")
        labels = torch.cat([labels for _, labels in dataloader], dim=0)

    if len(labels) == 0:
        return 0.0

    # Calcula o desvio padrão normalizado das contagens de classe
    counts = torch.bincount(labels, minlength=num_classes).float()
    mean_count = counts.mean()
    
    if mean_count == 0:
        return 0.0  # Evita divisão por zero

    # std populacional (unbiased=False) para manter limite teórico consistente na normalização
    normalized_std_dev = counts.std(unbiased=False) / mean_count
    return normalized_std_dev.item()


def normalizar_heterogeneidade(heterogeneidade_raw: float, num_classes: int) -> float:
    """
    Normaliza a heterogeneidade para o intervalo [0, 1].
    Para H_raw = std(counts)/mean(counts), um limite superior útil é sqrt(K-1).
    """
    if num_classes <= 1:
        return 0.0
    h_max = (num_classes - 1) ** 0.5
    if h_max == 0:
        return 0.0
    h_norm = heterogeneidade_raw / h_max
    return float(max(0.0, min(1.0, h_norm)))

def define_epochs_dinamicas(heterogeneidade: float, base_epochs: int, min_epochs: int) -> int:
    """
    HEURÍSTICA: Define o número de épocas de treinamento local.
    O número de épocas é inversamente proporcional à heterogeneidade dos dados.
    """
    if heterogeneidade == 0.0:
        return base_epochs

    decay_factor = EPOCHS_DECAY_FACTOR
    epochs = base_epochs - (decay_factor * heterogeneidade)

    # Arredonda e garante que o valor esteja dentro dos limites definidos
    num_epochs = int(round(epochs))
    num_epochs = max(min_epochs, num_epochs)
    return num_epochs

def define_lr_dinamico(heterogeneidade: float, base_lr: float, min_lr: float) -> float:
    """
    HEURÍSTICA: Define a taxa de aprendizagem (learning rate).
    O LR é reduzido com base na heterogeneidade para um aprendizado mais estável.
    """
    if heterogeneidade == 0.0:
        return base_lr

    decay_factor = LR_DECAY_FACTOR
    lr = base_lr / (1 + decay_factor * heterogeneidade)
    
    return max(lr, min_lr)

# ==============================================================================
# 4. FUNÇÕES DE TREINAMENTO E AVALIAÇÃO
# ==============================================================================

def train(net, trainloader, epochs, learning_rate, use_fedprox=False, mu=0.0, global_params=None):
    """Função de treinamento local para um cliente."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            # Cálculo da perda padrão
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            # --- HEURÍSTICA: Aplica o termo proximal (FedProx) ---
            if use_fedprox and global_params is not None:
                proximal_term = 0.0
                for local_param, global_param in zip(net.parameters(), global_params):
                    proximal_term += (local_param - global_param).norm(2) ** 2
                loss += (mu / 2) * proximal_term
                
            loss.backward()
            optimizer.step()
    return epochs

# === R2 MODIFICATION: deterministic worker seeding and step-budget training =====
def seed_worker_process(local_seed: int) -> None:
    """Seed every RNG of the process that executes this client's fit (Ray worker)."""
    random.seed(local_seed)
    np.random.seed(local_seed % (2 ** 32))
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    # torch.backends.cudnn is a ModuleType subclass (CudnnModule) that Ray's cloudpickle
    # cannot serialize; naming it in a function shipped to the workers breaks the
    # client job. Resolve it by name at run time instead (same effect).
    _cudnn = importlib.import_module("torch.backends." + "cudnn")
    _cudnn.deterministic = True
    _cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def train_steps(net, trainloader, steps, learning_rate, use_fedprox=False, mu=0.0,
                global_params=None):
    """Run exactly `steps` minibatch optimizer steps, re-iterating the DataLoader
    (a new shuffle from the same seeded generator) when the budget exceeds one
    epoch. Per-step computation is identical to train(): SGD momentum 0.9, the same
    loss and the same proximal term. Returns (steps_executed, examples_processed),
    both counted directly."""
    steps = int(steps)
    if steps <= 0:
        return 0, 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    done, examples = 0, 0
    while done < steps:
        progressed = False
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            if use_fedprox and global_params is not None:
                proximal_term = 0.0
                for local_param, global_param in zip(net.parameters(), global_params):
                    proximal_term += (local_param - global_param).norm(2) ** 2
                loss += (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            done += 1
            examples += int(labels.shape[0])
            progressed = True
            if done >= steps:
                break
        if not progressed:
            raise RuntimeError("step budget > 0 but the training loader yields no batch")
    return done, examples


# === GG MODIFICATION (epoch callback) =======================================
def train_steps_gg(net, trainloader, steps, learning_rate, use_fedprox=False, mu=0.0,
                   global_params=None, epoch_len=0, on_epoch=None):
    """train_steps with a read-only callback after every `epoch_len` steps. The
    optimizer, loss, proximal term and data order are those of train_steps; the
    callback only reads the parameters."""
    steps = int(steps)
    if steps <= 0:
        return 0, 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    done, examples = 0, 0
    while done < steps:
        progressed = False
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            if use_fedprox and global_params is not None:
                proximal_term = 0.0
                for local_param, global_param in zip(net.parameters(), global_params):
                    proximal_term += (local_param - global_param).norm(2) ** 2
                loss += (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            done += 1
            examples += int(labels.shape[0])
            progressed = True
            if on_epoch is not None and epoch_len > 0 and done % epoch_len == 0:
                on_epoch(done // epoch_len)
                net.train()
            if done >= steps:
                break
        if not progressed:
            raise RuntimeError("step budget > 0 but the training loader yields no batch")
    return done, examples


def test(net, testloader):
    """Função de avaliação para um cliente ou para o servidor."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            # criterion usa reduction='mean', então multiplicamos pelo tamanho
            # do batch para acumular a soma das perdas por amostra.
            loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0, 0.0 # Evita divisão por zero se o testloader estiver vazio

    accuracy = correct / total
    avg_loss = loss / total
    return avg_loss, accuracy

# ==============================================================================
# 5. CARREGAMENTO E PARTICIONAMENTO DE DADOS (NÃO-IID)
# ==============================================================================

def dirichlet_split_noniid(dataset, n_clients, alpha, seed=SEED):
    """
    HEURÍSTICA: Divide o dataset em partições Não-IID usando uma distribuição de Dirichlet.
    Garante que cada cliente receba pelo menos uma amostra, se possível.
    """
    # Proteção: este particionamento NÃO se aplica ao FEMNIST, que tem particionamento
    # natural por writer. Nunca deve ser chamado com DATASET == "FEMNIST".
    assert DATASET != "FEMNIST", (
        "dirichlet_split_noniid não deve ser chamado com DATASET=='FEMNIST'. "
        "FEMNIST usa particionamento natural por writer_id."
    )

    np.random.seed(seed)
    
    if not hasattr(dataset, 'targets'):
        raise ValueError("O dataset precisa ter um atributo 'targets' para a divisão de Dirichlet.")
        
    labels = np.array(dataset.targets)
    n_classes = int(labels.max() + 1)
    
    # Gera a distribuição de rótulos para cada cliente
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    
    class_idcs = [np.where(labels == y)[0] for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    
    # Aloca os índices de dados para cada cliente
    for class_idx, distribution in zip(class_idcs, label_distribution):
        np.random.shuffle(class_idx)
        proportions = (distribution * len(class_idx)).astype(int)
        proportions_sum = proportions.sum()
        if proportions_sum < len(class_idx):
            proportions[np.random.choice(n_clients)] += len(class_idx) - proportions_sum
        
        start = 0
        for i in range(n_clients):
            end = start + proportions[i]
            client_idcs[i].extend(class_idx[start:end])
            start = end

    # Garante que cada cliente tenha pelo menos uma amostra
    for i in range(n_clients):
        if not client_idcs[i]:
            for j in range(n_clients):
                if len(client_idcs[j]) > 1:
                    client_idcs[i].append(client_idcs[j].pop())
                    break

    return [Subset(dataset, idcs) for idcs in client_idcs]

def _femnist_collate(batch):
    """
    Collate customizado para converter o formato de dicts do HuggingFace (chaves
    "image" e "character") no par (tensor_imagens, tensor_labels) esperado por
    train()/test().
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["character"] for item in batch], dtype=torch.long)
    return images, labels


def load_data_femnist():
    """
    Carrega FEMNIST de forma nativa-federada: cada cliente corresponde a um writer do
    dataset original. NÃO usa Dirichlet. Os writers são fixos por setup
    (CLIENT_SETUP_CONFIG[...]["femnist_writers"]) ou sorteados com seed se ausentes.
    O teste centralizado usa writers HELD-OUT (não vistos no treino).
    """
    if not FLWR_DATASETS_AVAILABLE:
        raise RuntimeError(
            "FEMNIST requer flwr-datasets. Instale com: "
            "pip install flwr-datasets[vision]"
        )

    print(f"Carregando FEMNIST nativo-federado com {NUM_CLIENTS} writers (clientes) | seed={SEED}.")
    print("Particionamento: NATURAL (por writer_id). Dirichlet NÃO é aplicado.")

    # Particionador natural: agrupa por writer_id já existente no dataset.
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={
            "train": NaturalIdPartitioner(partition_by="writer_id"),
        },
    )

    num_total_partitions = fds.partitioners["train"].num_partitions
    rng = np.random.default_rng(SEED)

    # Seleção dos clientes (writers). Se o setup atual define uma lista FIXA de writers,
    # usa-a; senão, sorteia com seed fixa.
    writers_fixos = CLIENT_SETUP_CONFIG[CLIENT_SETUP].get("femnist_writers")
    if writers_fixos is not None:
        if len(writers_fixos) != NUM_CLIENTS:
            raise ValueError(
                f"femnist_writers do setup {CLIENT_SETUP} tem {len(writers_fixos)} ids, "
                f"mas num_clients={NUM_CLIENTS}. Eles precisam ter o mesmo tamanho."
            )
        fora = [w for w in writers_fixos if not (0 <= int(w) < num_total_partitions)]
        if fora:
            raise ValueError(
                f"femnist_writers fora do intervalo válido [0, {num_total_partitions}): {fora}"
            )
        selected_partition_ids = sorted(int(w) for w in writers_fixos)
        print(f"Writers FIXOS do setup {CLIENT_SETUP} (partition ids): {selected_partition_ids}")
    else:
        selected = rng.choice(num_total_partitions, size=NUM_CLIENTS, replace=False)
        selected_partition_ids = sorted(int(p) for p in selected)
        print(f"Writers selecionados aleatoriamente (seed={SEED}, partition ids): {selected_partition_ids}")

    # Transformação para tensores PyTorch (FEMNIST vem com chaves "image" e "character").
    trf_image = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    def _apply_transforms(batch):
        batch["image"] = [trf_image(img) for img in batch["image"]]
        return batch

    trainloaders, valloaders = [], []
    for cid, pid in enumerate(selected_partition_ids):
        partition = fds.load_partition(pid, split="train")
        partition = partition.with_transform(_apply_transforms)

        num_samples = len(partition)
        if num_samples == 0:
            trainloaders.append(DataLoader([], batch_size=BATCH_SIZE))
            valloaders.append(DataLoader([], batch_size=BATCH_SIZE))
            print(f"Aviso: writer {pid} (cid={cid}) sem amostras.")
            continue

        # Split train/val 90/10 (mesmo padrão dos outros datasets).
        len_val = max(1, num_samples // 10)
        len_train = num_samples - len_val
        ds_train, ds_val = random_split(
            partition,
            [len_train, len_val],
            torch.Generator().manual_seed(SEED),
        )

        trainloaders.append(DataLoader(
            ds_train, batch_size=BATCH_SIZE,
            shuffle=True, drop_last=len_train > 1,
            collate_fn=_femnist_collate,
        ))
        valloaders.append(DataLoader(
            ds_val, batch_size=BATCH_SIZE,
            collate_fn=_femnist_collate,
        ))

        print(f"  Cliente {cid} (writer {pid}): {len_train} train + {len_val} val")

    # Teste centralizado com writers HELD-OUT (o flwrlabs/femnist só tem split "train").
    remaining_ids = [p for p in range(num_total_partitions)
                     if p not in set(selected_partition_ids)]
    num_test_writers = min(100, len(remaining_ids))
    test_writer_ids = sorted(
        int(p) for p in rng.choice(remaining_ids, size=num_test_writers, replace=False)
    )
    test_parts = [fds.load_partition(pid, split="train") for pid in test_writer_ids]
    testset = concatenate_datasets(test_parts).with_transform(_apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, collate_fn=_femnist_collate)
    print(f"Teste centralizado: {len(test_writer_ids)} writers held-out -> {len(testset)} amostras.")

    return trainloaders, valloaders, testloader


def load_data():
    """Carrega o dataset selecionado e o particiona entre os clientes."""
    # FEMNIST usa particionamento NATURAL por writer_id — NÃO aplica Dirichlet.
    if DATASET == "FEMNIST":
        return load_data_femnist()

    transform_map = {
        "MNIST": Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
        "FASHION_MNIST": Compose([ToTensor(), Normalize((0.2860,), (0.3530,))]),
        "CIFAR10": Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }
    dataset_map = {
        "MNIST": MNIST, "FASHION_MNIST": FashionMNIST, "CIFAR10": CIFAR10
    }

    if DATASET not in dataset_map:
        raise ValueError(f"Dataset {DATASET} não suportado.")

    trf = transform_map[DATASET]
    dataset_class = dataset_map[DATASET]
    trainset = dataset_class("./data", train=True, download=True, transform=trf)
    testset = dataset_class("./data", train=False, download=True, transform=trf)

    print(f"Dividindo o dataset usando a distribuição de Dirichlet com alpha={ALPHA}.")
    # === GG MODIFICATION (partition) ========================================
    if gg_policy.active_partition() == "equal":
        print(f"[gg] equal-size label-skew partition: {gg_policy.EQUAL_N} samples per client")
        datasets = [Subset(trainset, idx) for idx in gg_policy.equal_size_label_skew(
            trainset.targets, NUM_CLIENTS, ALPHA, gg_policy.EQUAL_N, SEED)]
    else:
        datasets = dirichlet_split_noniid(trainset, n_clients=NUM_CLIENTS, alpha=ALPHA)

    trainloaders, valloaders = [], []
    for i, ds in enumerate(datasets):
        num_samples = len(ds)
        if num_samples == 0:
            trainloaders.append(DataLoader([], batch_size=BATCH_SIZE))
            valloaders.append(DataLoader([], batch_size=BATCH_SIZE))
            continue
            
        len_val = max(1, num_samples // 10) # 10% para validação
        len_train = num_samples - len_val

        if len_train == 0:
            # Sem amostras de treino: usa tudo para validação e deixa o treino vazio.
            # Um DataLoader com shuffle=True sobre um dataset vazio cria um RandomSampler
            # que rebenta com "num_samples=0".
            print(f"Aviso: Cliente {i} tem apenas {num_samples} amostra(s), que serão usadas para validação.")
            _, ds_val = random_split(ds, [0, num_samples], torch.Generator().manual_seed(SEED))
            trainloaders.append(DataLoader([], batch_size=BATCH_SIZE))
            valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
            continue

        ds_train, ds_val = random_split(ds, [len_train, len_val], torch.Generator().manual_seed(SEED))

        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=len_train > 1))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
        
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader

# ==============================================================================
# 6. IMPLEMENTAÇÃO DO CLIENTE FLOWER (FLOWERCLIENT)
# ==============================================================================

class FlowerClient(fl.client.NumPyClient):
    """Define o comportamento do cliente de Aprendizado Federado."""
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        time.sleep(COMMUNICATION_DELAY)
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Treina o modelo localmente aplicando as heurísticas definidas."""
        self.set_parameters(parameters)
        
        # Armazena os parâmetros globais para a heurística FedProx
        global_params = None
        if USE_FEDPROX:
             # Clona os parâmetros para garantir que não sejam alterados pelo otimizador
             global_params = [param.clone().detach() for param in self.net.parameters()]
        
        # --- Aplicação das Heurísticas Dinâmicas ---
        # 1. Calcula a heterogeneidade dos dados do cliente
        heterogeneidade_raw = calcular_heterogeneidade(self.cid, self.trainloader, NUM_CLASSES)
        heterogeneidade = normalizar_heterogeneidade(heterogeneidade_raw, NUM_CLASSES)

        # === R2 MODIFICATION (fit) =========================================
        # Per-client policy precomputed in the driver (plain dicts, serialised by
        # value). Seeding is done HERE, in the process that trains, from
        # (global seed, client id, server round) only, so arms that assign the same
        # configuration to a client share its stochastic trajectory.
        _pol = R2_POLICY_TABLE[int(self.cid)]
        num_steps, learning_rate = int(_pol["tau"]), float(_pol["lr"])
        _round = int(config.get("server_round", 0))
        _local_seed = r2_policy.client_round_seed(SEED, self.cid, _round)
        seed_worker_process(_local_seed)
        _ds = self.trainloader.dataset
        if num_steps > 0 and len(_ds) > 0:
            _loader = DataLoader(_ds, batch_size=BATCH_SIZE, shuffle=True,
                                 drop_last=self.trainloader.drop_last,
                                 generator=torch.Generator().manual_seed(_local_seed))
        else:
            _loader = None

        print(
            f"Cliente {self.cid}: HetNorm={heterogeneidade:.4f} | arm={R2_ARM_NAME} | "
            f"steps={num_steps} (orig={_pol['tau_original']}, donor={_pol['donor']}) | "
            f"LR={learning_rate:.6f} | FedProx={USE_FEDPROX} | seed={_local_seed}"
        )

        # === GG MODIFICATION (fit) ==========================================
        _gg_tracker = None
        _gfile = str(config.get("gg_grad_file", ""))
        if _loader is not None and _gfile:
            _gg = torch.load(_gfile, map_location=DEVICE, weights_only=True)
            _gg_tracker = gg_diag.EpochTracker(self.net, _gg["loo"].get(int(self.cid)), _gg["all"])
        if _loader is not None:
            steps_executed, examples_processed = train_steps_gg(
                self.net, _loader, num_steps, learning_rate,
                use_fedprox=USE_FEDPROX, mu=FEDPROX_MU, global_params=global_params,
                epoch_len=int(_pol["n_batches"]), on_epoch=_gg_tracker)
        else:
            steps_executed, examples_processed = 0, 0

        loss, _ = test(self.net, self.valloader)
        final_parameters = [val.cpu().numpy() for _, val in self.net.state_dict().items()]

        num_amostras = len(_ds)
        n_batches = int(_pol["n_batches"])
        epochs_equiv = (steps_executed / n_batches) if n_batches > 0 else 0.0
        fps = FLOPS_PER_SAMPLE_FULL or 0
        # nominal = campaign convention (n_k x epochs); executed = samples actually fed
        flops_nominal = int(fps * num_amostras * epochs_equiv)
        flops_executed = int(fps * examples_processed)

        time.sleep(COMMUNICATION_DELAY)
        return final_parameters, num_amostras, {
            "loss_after_fit": loss,
            "epochs": epochs_equiv,
            "learning_rate": learning_rate,
            "cid": str(self.cid),
            "flops_locais": flops_nominal,
            "H_k": heterogeneidade,
            "minibatch_updates": steps_executed,
            "n_batches": n_batches,
            "E_full": int(_pol["E_full"]),
            "tau_original": int(_pol["tau_original"]),
            "tau_received": num_steps,
            "steps_executed": steps_executed,
            "examples_processed": examples_processed,
            "flops_nominal": flops_nominal,
            "flops_executed": flops_executed,
            "donor": int(_pol["donor"]),
            "perm_id": str(_pol["perm_id"]),
            "local_seed": _local_seed,
            "E_arm": int(_pol["E_arm"]),
            **(_gg_tracker.as_metrics() if _gg_tracker is not None else {}),
        }

    def evaluate(self, parameters, config):
        """Avalia o modelo recebido do servidor nos dados de validação locais."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader)
        time.sleep(COMMUNICATION_DELAY)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy), "loss": float(loss)}

def client_fn(cid: str):
    """Função fábrica para criar instâncias de FlowerClient."""
    net = get_net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader).to_client()

# ==============================================================================
# 7. ESTRATÉGIA DO SERVIDOR E AGREGAÇÃO
# ==============================================================================

GG_CLASS_ROWS, GG_CONFUSION = [], {}   # GG MODIFICATION: filled by the centralized evaluation


def get_evaluate_fn(test_loader):
    """Retorna uma função para avaliação centralizada no servidor."""
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        net = get_net().to(DEVICE)
        net.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), parameters)}))
        loss, accuracy = test(net, test_loader)
        # === GG MODIFICATION (class-sensitive evaluation) =====================
        _cm = gg_diag.confusion(net, test_loader, NUM_CLASSES, DEVICE)
        GG_CLASS_ROWS.append({"round": int(server_round), **{
            k: v for k, v in gg_diag.class_metrics(_cm).items() if k != "per_class_recall"},
            **{f"recall_c{c}": r for c, r in enumerate(gg_diag.class_metrics(_cm)["per_class_recall"])}})
        GG_CONFUSION[int(server_round)] = _cm.tolist()
        r2_policy.mark_round()  # R2: wall-clock stamp, one per round
        print(f"\n--- Rodada {server_round} - Avaliação Centralizada (Servidor) ---")
        print(f"Loss (Servidor): {loss:.4f} | Acurácia (Servidor): {accuracy:.4f}")
        return loss, {"accuracy": accuracy, "loss": loss}
    return evaluate

def weighted_average(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    """Função de agregação para as métricas de avaliação dos clientes."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    avg_accuracy = sum(accuracies) / sum(examples)
    avg_loss = sum(losses) / sum(examples)
    
    print(f"--- Avaliação Agregada (Clientes) ---")
    print(f"Loss Média (Clientes): {avg_loss:.4f} | Acurácia Média (Clientes): {avg_accuracy:.4f}")

    return {"accuracy": avg_accuracy, "loss": avg_loss}

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    """
    Agrega as métricas de treinamento (fit) de uma rodada, ponderando "epochs" e
    "learning_rate" por num_examples. Também armazena as observações brutas da rodada
    no coletor global `fit_metrics_history` para a telemetria detalhada do relatório.
    """
    # Armazena as observações cliente-rodada brutas (para estatísticas por observação/cliente).
    fit_metrics_history.append(metrics)

    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {
            "avg_epochs_weighted": 0.0,
            "avg_lr_weighted": 0.0,
            "total_flops_round": 0,
            "avg_flops_per_client_round": 0.0,
        }

    avg_epochs = sum(num_examples * m["epochs"] for num_examples, m in metrics) / total_examples
    avg_lr = sum(num_examples * m["learning_rate"] for num_examples, m in metrics) / total_examples

    # Custo computacional agregado da rodada (soma simples e média entre clientes).
    flops_list = [int(m.get("flops_locais", 0)) for _, m in metrics]
    total_flops_round = sum(flops_list)
    avg_flops_per_client_round = total_flops_round / len(metrics) if metrics else 0.0

    return {
        "avg_epochs_weighted": avg_epochs,
        "avg_lr_weighted": avg_lr,
        "total_flops_round": total_flops_round,
        "avg_flops_per_client_round": avg_flops_per_client_round,
    }

# Carrega os dados antes de definir a estratégia
trainloaders, valloaders, testloader = load_data()

# === R2 MODIFICATION (plan) ==================================================
# H_k is a pure function of the client's partition (static), so evaluating it once
# here equals evaluating it per round. The plan depends only on pre-training
# information (partition, H_k, the arm and a recorded permutation seed).
_R2_H = [
    normalizar_heterogeneidade(
        calcular_heterogeneidade(str(_c), _dl, NUM_CLASSES), NUM_CLASSES)
    for _c, _dl in enumerate(trainloaders)
]
# === GG MODIFICATION (plan) ===============================================
GG_PARTITION = gg_policy.active_partition()
if _os.environ.get("FL_GG_PLAN_ONLY", "").strip():
    _gg_plans = {}
    for _arm in gg_policy.ARMS:
        if _arm == "fedprox_tuned" and GG_PARTITION != "dirichlet":
            continue
        _t = gg_policy.build_plan(
            arm=_arm, partition=GG_PARTITION, trainloaders=trainloaders, h_values=_R2_H, seed=SEED,
            dataset=DATASET, alpha=ALPHA, epochs_fn=define_epochs_dinamicas, lr_fn=define_lr_dinamico,
            base_epochs=BASE_EPOCHS, min_epochs=MIN_EPOCHS, base_lr=BASE_LR, min_lr=MIN_LR,
            flops_per_sample_full=FLOPS_PER_SAMPLE_FULL)
        _st = dict(r2_policy._STATE)
        _gg_plans[_arm] = {k: _st[k] for k in ("E_arm", "LR_arm", "tau_received",
                                               "weighted_steps_arm", "weighted_steps_full")}
    _st = dict(r2_policy._STATE)
    _Path(_os.environ["FL_GG_PLAN_ONLY"]).write_text(json.dumps({
        "seed": SEED, "alpha": ALPHA, "partition": GG_PARTITION, "n_samples": _st["n_samples"],
        "n_batches": _st["n_batches"], "H": _st["H"], "E_full": _st["E_full"], "LR_full": _st["LR_full"],
        "arms": _gg_plans}, indent=1), encoding="utf-8")
    print(f"[gg] plan-only: written {_os.environ['FL_GG_PLAN_ONLY']}")
    raise SystemExit(0)
R2_ARM_NAME = gg_policy.active_arm()
R2_POLICY_TABLE = gg_policy.build_plan(
    arm=R2_ARM_NAME, partition=GG_PARTITION,
    trainloaders=trainloaders, h_values=_R2_H, seed=SEED, dataset=DATASET, alpha=ALPHA,
    epochs_fn=define_epochs_dinamicas, lr_fn=define_lr_dinamico,
    base_epochs=BASE_EPOCHS, min_epochs=MIN_EPOCHS, base_lr=BASE_LR, min_lr=MIN_LR,
    flops_per_sample_full=FLOPS_PER_SAMPLE_FULL,
)
print(f"[gg] partition={GG_PARTITION} | arm={R2_ARM_NAME} | per-client plan={R2_POLICY_TABLE}")

# Estratégia instrumentada: hereda de FedAvg normalmente, mas grava a telemetria de
# diagnóstico de drift (ver drift_telemetry.py) ao agregar cada rodada, sem alterar
# em nada o resultado da agregação (no-op quando FL_DRIFT_CSV não está definida).
class FedAvgDrift(drift_telemetry.DriftAwareMixin, fl.server.strategy.FedAvg):
    # === R2 MODIFICATION (fail fast) ===========================================
    # A round with failed clients would silently continue with fewer (or zero)
    # updates; abort instead so the runner records the cell as failed immediately.
    def aggregate_fit(self, server_round, results, failures):
        if failures:
            first = failures[0]
            raise RuntimeError(f"[r2] round {server_round}: {len(failures)} client fit failure(s); "
                               f"aborting the run. First failure: {first!r}")
        return super().aggregate_fit(server_round, results, failures)


# === GG MODIFICATION (server diagnostics) ===================================
GG_RUN_DIR = _Path(_os.environ.get("FL_R2_RUN_DIR", "").strip() or ".")
GG_SERVER_ROWS = []
_GG_N = [len(dl.dataset) for dl in trainloaders]
_GG_EVAL_LOADERS = [DataLoader(dl.dataset, batch_size=256, shuffle=False) if len(dl.dataset) else None
                    for dl in trainloaders]


def _gg_grad_file(rnd):
    return str(GG_RUN_DIR / f"gg_grads_round_{int(rnd)}.pt")


class FedAvgGG(FedAvgDrift):
    """Reads w^t before a round and the returned updates after it; aggregation is FedAvg's."""

    def configure_fit(self, server_round, parameters, client_manager):
        net = get_net().to(DEVICE)
        nd = fl.common.parameters_to_ndarrays(parameters)
        net.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), nd)}))
        local = []
        for k, ld in enumerate(_GG_EVAL_LOADERS):
            local.append(None if ld is None else gg_diag.full_gradient(net, ld, DEVICE)[0])
        g, loo = gg_diag.loo_gradients(local, _GG_N)
        self._gg = {"w": gg_diag.flat_params(net).clone(), "g": g, "loo": loo, "local": local,
                    "round": int(server_round)}
        GG_RUN_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"all": g.detach().cpu(),
                    "loo": {k: v.detach().cpu() for k, v in enumerate(loo) if v is not None}},
                   _gg_grad_file(server_round))
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        out = super().aggregate_fit(server_round, results, failures)
        st = getattr(self, "_gg", None)
        if st is not None and st["round"] == int(server_round):
            net = get_net()
            keys = list(net.state_dict().keys())
            order = [n for n, _ in net.named_parameters()]
            N = float(sum(_GG_N))
            agg = None
            per = {}
            for _proxy, fit_res in results:
                cid = int(fit_res.metrics.get("cid"))
                nd = dict(zip(keys, fl.common.parameters_to_ndarrays(fit_res.parameters)))
                wk = torch.cat([torch.as_tensor(nd[n]).reshape(-1) for n in order]).to(st["w"].device)
                u = wk - st["w"]
                per[cid] = u
                agg = (_GG_N[cid] / N) * u if agg is None else agg + (_GG_N[cid] / N) * u
            for cid, u in sorted(per.items()):
                gk, gl = st["local"][cid], st["loo"][cid]
                GG_SERVER_ROWS.append({
                    "round": int(server_round), "client_id": cid, "n_k": _GG_N[cid],
                    "H_k": float(_R2_H[cid]), "E_arm": int(R2_POLICY_TABLE[cid]["E_arm"]),
                    "lr": float(R2_POLICY_TABLE[cid]["lr"]), "tau": int(R2_POLICY_TABLE[cid]["tau"]),
                    "grad_cos_loo": gg_diag.cosine(gk, gl) if gk is not None and gl is not None else "",
                    "grad_cos_all": gg_diag.cosine(gk, st["g"]) if gk is not None else "",
                    "grad_dissim": float((gk - st["g"]).norm()) if gk is not None else "",
                    "grad_norm": float(gk.norm()) if gk is not None else "",
                    "upd_cos_loo": gg_diag.cosine(u, -gl) if gl is not None else "",
                    "upd_cos_all": gg_diag.cosine(u, -st["g"]),
                    "upd_cos_agg": gg_diag.cosine(u, agg),
                    "upd_lin_loo": float(torch.dot(u, gl)) if gl is not None else "",
                    "upd_norm": float(u.norm()),
                    "agg_cos_all": gg_diag.cosine(agg, -st["g"]),
                    "global_grad_norm": float(st["g"].norm()),
                })
        try:
            _Path(_gg_grad_file(server_round)).unlink()
        except OSError:
            pass
        return out


_ref_net_drift = get_net()
_DRIFT_TRAINABLE_KEYS = drift_telemetry.get_trainable_keys(_ref_net_drift)
_DRIFT_STATE_DICT_KEYS = list(_ref_net_drift.state_dict().keys())

# === R2 MODIFICATION (initial checkpoint) ====================================
# The initial global model is loaded from a common checkpoint file shared by every
# arm and method of a (dataset, seed); its SHA-256 is verified before training.
# There is no fallback: a missing or mismatching checkpoint aborts the run.
set_global_seed(SEED)
_R2_INIT_PATH = _os.environ["FL_R2_INIT_CKPT"]
_R2_INIT_SHA = _os.environ["FL_R2_INIT_SHA256"]
_R2_INITIAL_NET = get_net()
_R2_INITIAL_NET.load_state_dict(torch.load(_R2_INIT_PATH, map_location="cpu", weights_only=True), strict=True)
_R2_INITIAL_STATE = _R2_INITIAL_NET.state_dict()
_r2_sha = r2_policy.state_dict_sha256(_R2_INITIAL_STATE)
if _r2_sha != _R2_INIT_SHA:
    raise RuntimeError(f"initial checkpoint SHA-256 mismatch: {_r2_sha} != {_R2_INIT_SHA}")
initial_parameters = fl.common.ndarrays_to_parameters(
    [val.cpu().numpy() for _, val in _R2_INITIAL_STATE.items()]
)

# Define a estratégia de agregação do servidor (FedAvg)
strategy = FedAvgGG(  # GG MODIFICATION (server diagnostics)
    initial_parameters=initial_parameters,
    fraction_fit=1.0,           # Usa 100% dos clientes para treinamento em cada rodada
    fraction_evaluate=1.0,      # Usa 100% dos clientes para avaliação
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=get_evaluate_fn(testloader),  # Avaliação centralizada no servidor
    evaluate_metrics_aggregation_fn=weighted_average, # Agregação de métricas dos clientes
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, # Telemetria das heurísticas dinâmicas
    on_fit_config_fn=lambda rnd: {"server_round": int(rnd), "gg_grad_file": _gg_grad_file(rnd)},  # GG
    drift_trainable_keys=_DRIFT_TRAINABLE_KEYS,
    drift_state_dict_keys=_DRIFT_STATE_DICT_KEYS,
    drift_method="FedHAD",
    drift_dataset=DATASET,
    drift_alpha=ALPHA,
    drift_seed=SEED,
)

# ==============================================================================
# 8. INÍCIO DA SIMULAÇÃO FEDERADA
# ==============================================================================

if __name__ == "__main__":
    print("Iniciando a simulação de Aprendizado Federado...")

    # Garante reprodutibilidade também no ponto de entrada da execução.
    set_global_seed(SEED)

    # --- Diagnóstico de medição de GPU (executado uma vez) ---
    print("=== Diagnóstico de Medição de GPU ===")
    GPU_DIAGNOSTICO = diagnosticar_medicao_gpu()
    for chave, valor in GPU_DIAGNOSTICO.items():
        print(f"  {chave}: {valor}")

    # Define os recursos de hardware para os clientes (GPU, se disponível)
    client_resources = {"num_gpus": CLIENT_GPU_FRACTION} if DEVICE.type == "cuda" else None

    # --- Inicia a medição de energia (opt-in via USE_ENERGY_TRACKING) ---
    tracker = None
    if USE_ENERGY_TRACKING and CODECARBON_AVAILABLE:
        tracker = _criar_emissions_tracker(_get_results_dir())
        if tracker is not None:
            try:
                tracker.start()
            except Exception as e:
                print(f"Aviso: falha ao iniciar o CodeCarbon ({e}). Prosseguindo sem medição.")
                tracker = None
    elif USE_ENERGY_TRACKING and not CODECARBON_AVAILABLE:
        print("Aviso: USE_ENERGY_TRACKING=True mas 'codecarbon' não está instalado. "
              "Instale com: pip install codecarbon nvidia-ml-py")

    # --- Fallback de energia da GPU via TDP×utilização ---
    # Acionado quando a GPU é usada mas o NVML não expõe a potência real
    # (leitura_potencia_ok=False), evitando GPU = 0 kWh na medição.
    gpu_estimator = None
    if (USE_ENERGY_TRACKING and DEVICE.type == "cuda"
            and GPU_DIAGNOSTICO.get("pynvml_disponivel")
            and not GPU_DIAGNOSTICO.get("leitura_potencia_ok", False)):
        gpu_estimator = _GpuEnergyEstimator(GPU_TDP_WATTS, gpu_index=0, sample_secs=5.0)
        if gpu_estimator.start():
            print(f"Info: leitura de potência da GPU indisponível via NVML; usando estimativa "
                  f"TDP×utilização (TDP={GPU_TDP_WATTS} W).")
        else:
            gpu_estimator = None

    start_time = time.time()

    # Inicia a simulação
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        # Propaga UTF-8 aos workers do Ray para evitar o crash do raylet no Windows
        # (SIGABRT em WorkerTableData.exit_detail por 'invalid UTF-8' quando há acentos
        # nos logs). Apenas codificação dos logs: NÃO altera lógica, resultados nem a
        # concorrência (num_gpus), preservando a comparabilidade entre os métodos.
        ray_init_args={"runtime_env": {"env_vars": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                                                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # R2
                                                    "PYTHONHASHSEED": "0"}}},
    )
    elapsed = time.time() - start_time

    # --- Encerra a medição de energia e coleta os dados ---
    energy_info = _collect_energy_info(tracker)

    # Anexa a estimativa de energia da GPU (fallback), se ativa.
    if gpu_estimator is not None:
        gpu_energy_estimada = gpu_estimator.stop()
        if energy_info is None:
            energy_info = {}
        energy_info["gpu_energy_estimada_kwh"] = gpu_energy_estimada
        energy_info["gpu_tdp_watts"] = GPU_TDP_WATTS
        if gpu_estimator.util_samples:
            energy_info["gpu_util_media"] = sum(gpu_estimator.util_samples) / len(gpu_estimator.util_samples)

    # R2 MODIFICATION: raw per-run artifacts (fingerprint.json + telemetry.csv).
    _r2_dir = _os.environ.get("FL_R2_RUN_DIR", "").strip()
    if _r2_dir:
        import platform as _pf
        _hw = {"device": str(DEVICE), "hostname": _pf.node(), "python": _pf.python_version(),
               "torch": torch.__version__, "flwr": getattr(fl, "__version__", ""),
               "cuda": torch.version.cuda,
               "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")}
        _cfg = json.loads(_os.environ.get("FL_R2_CONFIG_JSON", "{}"))
        _cfg.update({"use_fedprox": USE_FEDPROX, "fedprox_mu": FEDPROX_MU, "batch_size": BATCH_SIZE,
                     "num_rounds": NUM_ROUNDS, "client_setup": CLIENT_SETUP, "alpha": ALPHA,
                     "dataset": DATASET, "base_epochs": BASE_EPOCHS, "min_epochs": MIN_EPOCHS,
                     "epochs_decay": EPOCHS_DECAY_FACTOR, "lr_decay": LR_DECAY_FACTOR,
                     "base_lr": BASE_LR, "min_lr": MIN_LR, "comm_delay": COMMUNICATION_DELAY})
        r2_policy.write_run_artifacts(
            _r2_dir, config_snapshot=_cfg, init_sha256=_R2_INIT_SHA, init_path=_R2_INIT_PATH,
            history=history, fit_metrics_history=fit_metrics_history, elapsed=elapsed,
            hardware=_hw)
        print(f"[r2] artifacts written to {_r2_dir}")
        # === GG MODIFICATION (artifacts) ==================================
        gg_diag.write_rows(_Path(_r2_dir) / "gg_server.csv", GG_SERVER_ROWS)
        _ep_rows = []
        for _rnd, _rm in enumerate(fit_metrics_history, start=1):
            for _n, _m in _rm:
                for _e in range(1, 6):
                    if f"gg_step_norm_e{_e}" not in _m:
                        continue
                    _ep_rows.append({"round": _rnd, "client_id": int(_m.get("cid")), "n_k": int(_n),
                                     "H_k": _m.get("H_k"), "E_arm": _m.get("E_arm"),
                                     "E_full": _m.get("E_full"), "lr": _m.get("learning_rate"),
                                     "epoch": _e, **{q: _m.get(f"gg_{q}_e{_e}", "") for q in (
                                         "step_norm", "marg_cos_loo", "cum_cos_loo", "marg_lin_loo",
                                         "marg_cos_all", "marg_lin_all")}})
        gg_diag.write_rows(_Path(_r2_dir) / "gg_epochs.csv", _ep_rows)
        gg_diag.write_rows(_Path(_r2_dir) / "gg_class.csv", GG_CLASS_ROWS)
        (_Path(_r2_dir) / "gg_confusion.json").write_text(json.dumps(GG_CONFUSION), encoding="utf-8")
        print(f"[gg] diagnostics written: {len(GG_SERVER_ROWS)} server rows, {len(_ep_rows)} epoch rows")

    saved_path = save_simulation_results(history, elapsed, energy_info=energy_info)
    print(f"\nResultados salvos em: {saved_path}")