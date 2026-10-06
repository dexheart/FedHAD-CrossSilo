# import warnings
# warnings.filterwarnings("ignore")
# import logging
# logging.getLogger("flwr").setLevel(logging.ERROR)

# =============================================================================
# Métricas de Custo Computacional (FLOPs) — IMPORTANTE para a redação da tese:
#   - O `thop.profile` reporta MACs (multiply-accumulate operations), NÃO FLOPs.
#   - Convenção adotada: 1 MAC = 1 multiplicação + 1 adição = 2 FLOPs.
#     Logo, FLOPs(forward) = 2 × MACs (constante MACS_TO_FLOPS = 2).
#   - Custo do backward ≈ 2 × forward (convenção da literatura) => passe completo
#     (forward + backward) = 3 × forward. Combinando: FLOPs(completo) = 6 × MACs.
#   - Os valores reportados (FLOPS_PER_SAMPLE_FORWARD / _FULL) são FLOPs verdadeiros,
#     não MACs. Mesma convenção do FedHAD_Final_2.0.py para garantir comparabilidade.
#
# Dependências para telemetria de custo/energia (instalar se necessário):
#     pip install codecarbon nvidia-ml-py
# O pacote `nvidia-ml-py` (módulo pynvml) é necessário para que o CodeCarbon meça a
# energia da GPU NVIDIA via NVML. Sem ele, a medição reflete apenas CPU + RAM.
# =============================================================================

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

# 1. Definições e Configurações
# -----------------------------------
# << ESCOLHA O DATASET AQUI >>
# Opções: "MNIST", "FASHION_MNIST", "CIFAR10", "FEMNIST" (nativamente federado, particionado por writer_id)
DATASET = "FEMNIST"
# -----------------------------------

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Setup de clientes (escolha 3, 5 ou 10) ---
# Presets que ajustam automaticamente a fração de GPU por cliente conforme o número
# de clientes, evitando ter de mexer manualmente em num_gpus a cada experimento.
CLIENT_SETUP = 5
# "femnist_writers": lista FIXA de partition IDs (cada um = 1 writer do FEMNIST) usados
# como clientes naquele setup. Devem ter o mesmo tamanho de "num_clients". São índices
# 0..3596 do NaturalIdPartitioner (ex.: o id 320 mapeia para o writer 'f0320_41'); são
# exatamente os números impressos como "Writers selecionados (partition ids)".
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
COMMUNICATION_DELAY = 0.05  # 20ms de delay
ALPHA = 0.5 # Valor de alpha para a distribuição de Dirichlet
TEST_NAME = "FedAVG"

# Hiperparâmetros do treino local (extraídos do train(), que os usava hardcoded, para
# poderem ser reportados na telemetria de diagnóstico de drift — ver drift_telemetry.py).
LOCAL_LR = 0.01
LOCAL_MOMENTUM = 0.9

# --- Reprodutibilidade ---
SEED = 42  # Semente global para todas as fontes de aleatoriedade

# --- Métricas de convergência (rounds/FLOPs/bytes até atingir cada acurácia) ---
ACCURACY_TARGETS = [0.50, 0.60, 0.70, 0.80]

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
# === Fim dos overrides ===

# Nº de classes do dataset (usado só para normalizar o escore de heterogeneidade H_k
# na telemetria de diagnóstico de drift; não afeta o treino do FedAVG).
NUM_CLASSES = 62 if DATASET == "FEMNIST" else 10


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
        f"__Setup-{CLIENT_SETUP}"
        f"__Clients-{NUM_CLIENTS}"
        f"__Rounds-{NUM_ROUNDS}"
        f"__Delay-{_format_value_for_filename(COMMUNICATION_DELAY)}"
        f"__{partitioning_tag}"
        f"__Seed-{SEED}"
    )


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


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    """
    Agrega as métricas de treinamento (fit) de uma rodada. Em FedAVG as épocas são
    fixas, mas o custo (FLOPs) varia por cliente conforme o tamanho do dataset local.
    Também armazena as observações brutas no coletor global `fit_metrics_history`.
    """
    fit_metrics_history.append(metrics)

    flops_list = [int(m.get("flops_locais", 0)) for _, m in metrics]
    total_flops_round = sum(flops_list)
    avg_flops_per_client_round = total_flops_round / len(metrics) if metrics else 0.0

    return {
        "total_flops_round": total_flops_round,
        "avg_flops_per_client_round": avg_flops_per_client_round,
    }


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
    vão para uma subpasta com esse nome — ex.: FedAVG-results/test1_convergencia/<dataset>/
    —, mantendo cada experimento da tese separado e organizado.
    """
    # Raiz dos resultados: FL_RESULTS_DIR (definido pelo run_experiments.py) ou, se o
    # script for rodado direto, a pasta results/main_campaign_default_alpha_0_5/ na raiz do projeto.
    results_root = Path(_os.environ.get("FL_RESULTS_DIR", "").strip()
                        or Path(__file__).resolve().parent.parent / "results" / "main_campaign_default_alpha_0_5")
    base = results_root / "FedAVG-results"
    tag = _os.environ.get("FL_RUN_TAG", "").strip()
    if tag:
        base = base / tag
    dataset_dir = base / _dataset_folder_name(DATASET)
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
    """Salva os resultados da execução em FedAVG-results/<dataset>."""
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

# 2. Definição das Redes Neurais

# CNN para MNIST e FashionMNIST (imagens 1x28x28)
class Net_MNIST_Fashion(nn.Module):
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

# CNN dedicada para FEMNIST (imagens 1x28x28, 62 classes)
class Net_FEMNIST(nn.Module):
    """
    CNN dedicada para FEMNIST (imagens 1x28x28, 62 classes:
    10 dígitos + 26 maiúsculas + 26 minúsculas).

    Arquitetura mais robusta que a LeNet de Net_MNIST_Fashion para lidar com:
    - Maior número de classes (62 vs 10)
    - Variabilidade alta entre writers (estilos pessoais de escrita)
    - Ambiguidades visuais entre maiúsculas/minúsculas/dígitos
      (ex.: '0' vs 'O' vs 'o', '1' vs 'l' vs 'I')

    Usa BatchNorm para estabilizar treinamento em cenário federado e Dropout para
    reduzir overfitting em clientes com poucos dados.
    """
    def __init__(self) -> None:
        super(Net_FEMNIST, self).__init__()
        # Bloco 1: extração de features locais
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # Bloco 2: features mais abstratas
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
        # Bloco 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Bloco 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Classificador
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Função para selecionar a rede correta
def get_net():
    if DATASET == "CIFAR10":
        return Net_CIFAR10().to(DEVICE)
    if DATASET == "FEMNIST":
        return Net_FEMNIST().to(DEVICE)
    return Net_MNIST_Fashion().to(DEVICE)


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


# Calcula os FLOPs por amostra (forward) uma única vez.
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

# 3. Funções de Treinamento e Teste
def train(net, trainloader, epochs=5):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LOCAL_LR, momentum=LOCAL_MOMENTUM)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
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
        return 0.0, 0.0
    accuracy = correct / total
    return loss / total, accuracy

# 4. Carregamento e Particionamento dos Dados
def dirichlet_split_noniid(dataset, n_clients, alpha, seed=SEED):
    """
    Divide um conjunto de dados em partições não-IID usando uma distribuição de Dirichlet.
    Garante que cada cliente receba pelo menos uma amostra.
    """
    # Proteção: este particionamento NÃO se aplica ao FEMNIST, que tem particionamento
    # natural por writer. Nunca deve ser chamado com DATASET == "FEMNIST".
    assert DATASET != "FEMNIST", (
        "dirichlet_split_noniid não deve ser chamado com DATASET=='FEMNIST'. "
        "FEMNIST usa particionamento natural por writer_id."
    )

    np.random.seed(seed)
    
    # Garante que o dataset tenha o atributo 'targets'
    if not hasattr(dataset, 'targets'):
        raise ValueError("O dataset precisa ter um atributo 'targets' para a divisão de Dirichlet.")
        
    try:
        # Tenta converter para numpy array, funciona para listas e tensores
        labels = np.array(dataset.targets)
    except Exception as e:
        raise TypeError(f"Não foi possível converter 'dataset.targets' para um array numpy: {e}")

    n_classes = int(labels.max() + 1)
    
    # Gera a distribuição de rótulos para cada cliente
    # (n_classes, n_clients)
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    
    # Índices de dados para cada classe
    class_idcs = [np.where(labels == y)[0] for y in range(n_classes)]
    
    client_idcs = [[] for _ in range(n_clients)]
    
    # Aloca os índices de dados para cada cliente
    for class_idx, distribution in zip(class_idcs, label_distribution):
        # Embaralha os índices da classe para garantir aleatoriedade na alocação
        np.random.shuffle(class_idx)
        
        # Calcula o número de amostras por cliente para a classe atual
        proportions = (distribution * len(class_idx)).astype(int)
        
        # Garante que a soma das proporções seja igual ao total de amostras da classe
        proportions = proportions[:n_clients]
        proportions_sum = proportions.sum()
        if proportions_sum < len(class_idx):
            # Adiciona o restante a um cliente aleatório para não perder amostras
            proportions[np.random.choice(n_clients)] += len(class_idx) - proportions_sum
        
        # Divide os índices e aloca para os clientes
        start = 0
        for i in range(n_clients):
            end = start + proportions[i]
            client_idcs[i].extend(class_idx[start:end])
            start = end

    # Garante que cada cliente tenha pelo menos uma amostra, se possível
    # Se um cliente ficou sem amostras, move uma de outro cliente
    for i in range(n_clients):
        if not client_idcs[i]:
            # Encontra um cliente com mais de uma amostra para "doar"
            for j in range(n_clients):
                if len(client_idcs[j]) > 1:
                    # Move uma amostra
                    moved_idx = client_idcs[j].pop()
                    client_idcs[i].append(moved_idx)
                    print(f"Aviso: Cliente {i} não tinha amostras. Moveu uma amostra do cliente {j}.")
                    break

    # Cria os Subsets para cada cliente
    client_datasets = [Subset(dataset, idcs) for idcs in client_idcs]
    
    return client_datasets


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
    Carrega FEMNIST de forma nativa-federada: cada cliente corresponde a um writer
    aleatório do dataset original. NÃO usa Dirichlet.

    Seleciona NUM_CLIENTS writers aleatoriamente (seed=SEED) do conjunto total
    disponível. O conjunto de teste usado para avaliação centralizada é o split de
    teste oficial do FEMNIST.
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

    # Seleção dos clientes (writers). Se o setup atual define uma lista FIXA de writers
    # (CLIENT_SETUP_CONFIG[...]["femnist_writers"]), usa-a; senão, sorteia com seed fixa.
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

    # Transformação para tensores PyTorch.
    # FEMNIST do flwr-datasets vem com chaves "image" e "character".
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

    # Conjunto de teste centralizado.
    # OBS: o dataset flwrlabs/femnist NÃO possui split de teste oficial (só "train").
    # Em vez disso, montamos o teste com writers HELD-OUT (não usados como clientes):
    # avaliação em escritores NÃO vistos no treino — protocolo usual para FEMNIST e
    # sem vazamento de dados dos clientes.
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
    """Carrega o dataset e o particiona entre os clientes."""
    # FEMNIST usa particionamento NATURAL por writer_id — NÃO aplica Dirichlet.
    if DATASET == "FEMNIST":
        return load_data_femnist()

    if DATASET == "MNIST":
        trf = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        trainset = MNIST("./data", train=True, download=True, transform=trf)
        testset = MNIST("./data", train=False, download=True, transform=trf)
    elif DATASET == "FASHION_MNIST":
        trf = Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])
        trainset = FashionMNIST("./data", train=True, download=True, transform=trf)
        testset = FashionMNIST("./data", train=False, download=True, transform=trf)
    elif DATASET == "CIFAR10":
        trf = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = CIFAR10("./data", train=True, download=True, transform=trf)
        testset = CIFAR10("./data", train=False, download=True, transform=trf)
    else:
        raise ValueError(f"Dataset {DATASET} não suportado.")

    # Particionamento Non-IID com distribuição de Dirichlet
    print(f"Dividindo o dataset usando a distribuição de Dirichlet com alpha={ALPHA}.")
    datasets = dirichlet_split_noniid(trainset, n_clients=NUM_CLIENTS, alpha=ALPHA)

    trainloaders, valloaders = [], []
    for i, ds in enumerate(datasets):
        num_samples = len(ds)
        if num_samples == 0:
            print(f"Aviso: Cliente {i} não recebeu dados.")
            # Adiciona DataLoaders vazios para clientes sem dados
            trainloaders.append(DataLoader([], batch_size=BATCH_SIZE))
            valloaders.append(DataLoader([], batch_size=BATCH_SIZE))
            continue
            
        # Define 10% dos dados para validação, com um mínimo de 1, se possível
        len_val = max(1, num_samples // 10)
        len_train = num_samples - len_val
        
        if len_train == 0:
            # Se não houver amostras de treino, usa tudo para validação e deixa treino vazio.
            # Um DataLoader com shuffle=True sobre um dataset vazio cria um RandomSampler
            # que rebenta com "num_samples=0"; por isso o trainloader fica explicitamente vazio.
            print(f"Aviso: Cliente {i} tem apenas {num_samples} amostra(s), que serão usadas para validação.")
            _, ds_val = random_split(ds, [0, num_samples], torch.Generator().manual_seed(SEED))
            trainloaders.append(DataLoader([], batch_size=BATCH_SIZE))
            valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
            continue

        ds_train, ds_val = random_split(ds, [len_train, len_val], torch.Generator().manual_seed(SEED))

        # drop_last=True pode ser importante se o último batch tiver tamanho 1 e o batch norm não o suportar
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=len_train > 1))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
        
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader

trainloaders, valloaders, testloader = load_data()


def calcular_heterogeneidade(cid: str, dataloader: DataLoader, num_classes: int) -> float:
    """
    Calcula uma métrica de heterogeneidade para um cliente com base na distribuição de classes
    de seus dados locais. A métrica é o desvio padrão normalizado da contagem de cada classe.
    Um valor mais alto indica maior desbalanceamento (maior heterogeneidade).

    Usada AQUI apenas para diagnóstico (telemetria de drift, ver drift_telemetry.py); o
    FedAVG não usa H_k para adaptar nada — épocas e LR permanecem fixos.
    """
    if not hasattr(dataloader.dataset, 'indices') or not dataloader.dataset.indices:
        return 0.0

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

    counts = torch.bincount(labels, minlength=num_classes).float()
    mean_count = counts.mean()

    if mean_count == 0:
        return 0.0

    normalized_std_dev = counts.std(unbiased=False) / mean_count
    return normalized_std_dev.item()


def normalizar_heterogeneidade(heterogeneidade_raw: float, num_classes: int) -> float:
    """Normaliza a heterogeneidade para o intervalo [0, 1] (limite superior sqrt(K-1))."""
    if num_classes <= 1:
        return 0.0
    h_max = (num_classes - 1) ** 0.5
    if h_max == 0:
        return 0.0
    h_norm = heterogeneidade_raw / h_max
    return float(max(0.0, min(1.0, h_norm)))


# 5. Implementação do Cliente Flower
class FlowerClient(fl.client.NumPyClient):
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
        self.set_parameters(parameters)
        num_epochs = 5  # FedAVG usa épocas fixas
        train(self.net, self.trainloader, epochs=num_epochs)

        # --- Diagnóstico de drift (só custa algo se FL_DRIFT_CSV estiver definida) ---
        # H_k é calculado em TODOS os métodos (mesmo aqui, onde não adapta nada) para
        # servir de proxy comparável na análise de correlação entre H_k e desalinhamento.
        heterogeneidade_raw = calcular_heterogeneidade(self.cid, self.trainloader, NUM_CLASSES)
        heterogeneidade = normalizar_heterogeneidade(heterogeneidade_raw, NUM_CLASSES)
        minibatch_updates = num_epochs * len(self.trainloader)

        # --- Custo computacional: FLOPs locais (forward+backward) deste cliente nesta rodada ---
        num_amostras = len(self.trainloader.dataset)
        if FLOPS_PER_SAMPLE_FULL is not None:
            flops_locais = int(FLOPS_PER_SAMPLE_FULL * num_amostras * num_epochs)
        else:
            flops_locais = 0  # thop indisponível: sinalizado como 0 (relatório indica ausência)

        time.sleep(COMMUNICATION_DELAY)
        return self.get_parameters(config={}), num_amostras, {
            "epochs": num_epochs,
            "learning_rate": LOCAL_LR,
            "cid": str(self.cid),
            "flops_locais": flops_locais,
            "H_k": heterogeneidade,
            "minibatch_updates": minibatch_updates,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader)
        time.sleep(COMMUNICATION_DELAY)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy), "loss": float(loss)}

def client_fn(cid: str):
    net = get_net()
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader).to_client()

# 6. Estratégia de Agregação e Validação Centralizada
def get_evaluate_fn(test_loader):
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        net = get_net()
        net.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), parameters)}))
        loss, accuracy = test(net, test_loader)
        print(f"\n--- Rodada {server_round} - Avaliação Centralizada (Servidor) ---")
        print(f"Loss (Servidor): {loss:.4f} | Acurácia (Servidor): {accuracy:.4f}")
        return loss, {"accuracy": accuracy, "loss": loss}
    return evaluate

def weighted_average(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    avg_accuracy = sum(accuracies) / sum(examples)
    avg_loss = sum(losses) / sum(examples)
    
    print(f"--- Avaliação Agregada (Clientes) ---")
    print(f"Loss Média (Clientes): {avg_loss:.4f} | Acurácia Média (Clientes): {avg_accuracy:.4f}")
    
    return {"accuracy": avg_accuracy, "loss": avg_loss}

# Estratégia instrumentada: hereda de FedAvg normalmente, mas grava a telemetria de
# diagnóstico de drift (ver drift_telemetry.py) ao agregar cada rodada, sem alterar
# em nada o resultado da agregação (no-op quando FL_DRIFT_CSV não está definida).
class FedAvgDrift(drift_telemetry.DriftAwareMixin, fl.server.strategy.FedAvg):
    pass


_ref_net_drift = get_net()
_DRIFT_TRAINABLE_KEYS = drift_telemetry.get_trainable_keys(_ref_net_drift)
_DRIFT_STATE_DICT_KEYS = list(_ref_net_drift.state_dict().keys())

strategy = FedAvgDrift(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=get_evaluate_fn(testloader),
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, # Telemetria de custo (FLOPs)
    drift_trainable_keys=_DRIFT_TRAINABLE_KEYS,
    drift_state_dict_keys=_DRIFT_STATE_DICT_KEYS,
    drift_method="FedAVG",
    drift_dataset=DATASET,
    drift_alpha=ALPHA,
    drift_seed=SEED,
)

# 7. Início da Simulação
if __name__ == "__main__":
    print("Iniciando a simulação de Aprendizado Federado...")

    # Garante reprodutibilidade também no ponto de entrada da execução.
    set_global_seed(SEED)

    # --- Diagnóstico de medição de GPU (executado uma vez) ---
    print("=== Diagnóstico de Medição de GPU ===")
    GPU_DIAGNOSTICO = diagnosticar_medicao_gpu()
    for chave, valor in GPU_DIAGNOSTICO.items():
        print(f"  {chave}: {valor}")

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
        ray_init_args={"runtime_env": {"env_vars": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}}},
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

    saved_path = save_simulation_results(history, elapsed, energy_info=energy_info)
    print(f"\nResultados salvos em: {saved_path}")
