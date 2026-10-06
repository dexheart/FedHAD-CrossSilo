# -*- coding: utf-8 -*-
"""
Instrumentação de diagnóstico de drift (experimento 1.1 do checklist de revisão).

Compartilhada pelos quatro *_Final.py (FedAVG, FedAvgM, FedProx, FedHAD). Ativada
via variável de ambiente FL_DRIFT_CSV (caminho do CSV de saída); se ausente, todas
as funções abaixo são no-ops e o comportamento normal do treino não muda em nada.

Para cada cliente, em cada rodada, registra: o escore de heterogeneidade (H_k) e os
hiperparâmetros efetivamente usados (E_k, lr_k), e duas medidas de quanto o update
local do cliente diverge da direção de agregação do grupo (update_norm[_per_step] e
cos_sim). Três cuidados importantes, já validados numa rodada anterior:

  1. update_norm/cos_sim usam SOMENTE os tensores treináveis do modelo (filtram os
     buffers de BatchNorm — running_mean/running_var/num_batches_tracked — que têm
     escala muito maior e, se incluídos, dominam a norma e destroem a similaridade).
  2. cos_sim é calculado contra a agregação FedAvg PURA (Σ (n_k/N)·delta_k) dos deltas
     da própria rodada, nunca contra w_global^{t+1} - w_global^t — isso importa em
     especial no FedAvgM, cujo servidor aplica momento (o que inflaria artificialmente
     a similaridade).
  3. update_norm_per_step normaliza pela quantidade de passos de otimizador do
     cliente na rodada (E_k x nº de batches), pois a norma bruta do delta está
     fortemente correlacionada com n_samples.
"""
import os
import csv
import threading
from typing import List, Optional, Sequence

import numpy as np

try:
    from flwr.common import parameters_to_ndarrays
except ImportError:  # pragma: no cover - flwr sempre deve estar presente no projeto
    parameters_to_ndarrays = None


DRIFT_CSV_PATH = os.environ.get("FL_DRIFT_CSV", "").strip()

_HEADER = [
    "metodo", "dataset", "alpha", "seed", "round", "client_id", "H_k", "E_k", "lr_k",
    "update_norm", "update_norm_per_step", "cos_sim", "n_samples", "minibatch_updates",
]

_lock = threading.Lock()
_header_written = False


def is_enabled() -> bool:
    """True se a instrumentação estiver ligada (FL_DRIFT_CSV definida e não vazia)."""
    return bool(DRIFT_CSV_PATH)


def _ensure_header() -> None:
    global _header_written
    if _header_written or not DRIFT_CSV_PATH:
        return
    with _lock:
        if _header_written:
            return
        exists_nonempty = os.path.exists(DRIFT_CSV_PATH) and os.path.getsize(DRIFT_CSV_PATH) > 0
        if not exists_nonempty:
            parent = os.path.dirname(DRIFT_CSV_PATH)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(DRIFT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_HEADER)
        _header_written = True


def get_trainable_keys(net) -> List[str]:
    """
    Chaves do state_dict que correspondem a parâmetros TREINÁVEIS (requires_grad=True),
    na mesma ordem em que aparecem no state_dict. Filtra implicitamente os buffers de
    BatchNorm (running_mean, running_var, num_batches_tracked), que não têm gradiente.
    """
    trainable_names = {name for name, p in net.named_parameters() if p.requires_grad}
    return [k for k in net.state_dict().keys() if k in trainable_names]


def _flatten_trainable_delta(
    new_ndarrays: Sequence[np.ndarray],
    old_ndarrays: Sequence[np.ndarray],
    state_dict_keys: Sequence[str],
    trainable_keys: Sequence[str],
) -> np.ndarray:
    """Vetor achatado de (novo - antigo) considerando SOMENTE os índices treináveis."""
    trainable_set = set(trainable_keys)
    parts = []
    for k, new_v, old_v in zip(state_dict_keys, new_ndarrays, old_ndarrays):
        if k in trainable_set:
            parts.append((new_v.astype(np.float64) - old_v.astype(np.float64)).ravel())
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _append_rows(rows: List[dict]) -> None:
    if not DRIFT_CSV_PATH or not rows:
        return
    _ensure_header()
    with _lock:
        with open(DRIFT_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_HEADER)
            for row in rows:
                writer.writerow(row)


def record_round_drift(
    server_round: int,
    results,
    old_ndarrays: Optional[Sequence[np.ndarray]],
    state_dict_keys: Sequence[str],
    trainable_keys: Sequence[str],
    method_name: str,
    dataset_name: str,
    alpha_value,
    seed_value,
) -> None:
    """
    Calcula e grava, para todos os clientes de UMA rodada, as métricas de drift.

    `results` é a lista (ClientProxy, FitRes) recebida em aggregate_fit. `old_ndarrays`
    são os parâmetros globais (ndarrays, na ordem de state_dict_keys) que foram enviados
    aos clientes NO INÍCIO desta rodada (capturados em configure_fit) — nunca os
    parâmetros globais já agregados/atualizados.
    """
    if not is_enabled() or not results or old_ndarrays is None:
        return
    if parameters_to_ndarrays is None:
        return

    deltas = []
    weights = []
    metrics_list = []
    for _client_proxy, fit_res in results:
        new_ndarrays = parameters_to_ndarrays(fit_res.parameters)
        delta = _flatten_trainable_delta(new_ndarrays, old_ndarrays, state_dict_keys, trainable_keys)
        deltas.append(delta)
        weights.append(fit_res.num_examples)
        metrics_list.append((fit_res.metrics or {}, fit_res.num_examples))

    total_n = sum(weights)
    if total_n <= 0 or not deltas or deltas[0].size == 0:
        return

    agg_delta = np.zeros_like(deltas[0])
    for delta, n_k in zip(deltas, weights):
        agg_delta += (n_k / total_n) * delta

    rows = []
    for delta, (metrics, n_k) in zip(deltas, metrics_list):
        update_norm = float(np.linalg.norm(delta))
        minibatch_updates = int(metrics.get("minibatch_updates", 0) or 0)
        update_norm_per_step = update_norm / minibatch_updates if minibatch_updates > 0 else 0.0
        rows.append({
            "metodo": method_name,
            "dataset": dataset_name,
            "alpha": alpha_value,
            "seed": seed_value,
            "round": server_round,
            "client_id": metrics.get("cid", ""),
            "H_k": metrics.get("H_k", ""),
            "E_k": metrics.get("epochs", ""),
            "lr_k": metrics.get("learning_rate", ""),
            "update_norm": update_norm,
            "update_norm_per_step": update_norm_per_step,
            "cos_sim": _cosine_similarity(delta, agg_delta),
            "n_samples": n_k,
            "minibatch_updates": minibatch_updates,
        })
    _append_rows(rows)


class DriftAwareMixin:
    """
    Mixin de estratégia Flower: captura os parâmetros globais no início de cada rodada
    (configure_fit) e, ao agregar (aggregate_fit), grava a telemetria de drift ANTES de
    delegar para a agregação real da classe-base (super()) — não altera o resultado da
    agregação, só observa os deltas dos clientes.

    Uso: `class FedAvgDrift(DriftAwareMixin, fl.server.strategy.FedAvg): pass`, passando
    os kwargs drift_* ao construir a estratégia (ver *_Final.py).
    """

    def __init__(
        self,
        *args,
        drift_trainable_keys: Optional[Sequence[str]] = None,
        drift_state_dict_keys: Optional[Sequence[str]] = None,
        drift_method: str = "",
        drift_dataset: str = "",
        drift_alpha=None,
        drift_seed=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._drift_trainable_keys = list(drift_trainable_keys or [])
        self._drift_state_dict_keys = list(drift_state_dict_keys or [])
        self._drift_method = drift_method
        self._drift_dataset = drift_dataset
        self._drift_alpha = drift_alpha
        self._drift_seed = drift_seed
        self._drift_round_start_ndarrays = None

    def configure_fit(self, server_round, parameters, client_manager):
        if is_enabled():
            self._drift_round_start_ndarrays = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        if is_enabled() and results and self._drift_round_start_ndarrays is not None:
            record_round_drift(
                server_round,
                results,
                self._drift_round_start_ndarrays,
                self._drift_state_dict_keys,
                self._drift_trainable_keys,
                self._drift_method,
                self._drift_dataset,
                self._drift_alpha,
                self._drift_seed,
            )
        return super().aggregate_fit(server_round, results, failures)
