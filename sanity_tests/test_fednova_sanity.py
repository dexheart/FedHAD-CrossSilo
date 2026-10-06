# -*- coding: utf-8 -*-
"""
Testes de sanidade do núcleo matemático do FedNova: effective_local_steps() e
fednova_aggregate(), definidos em FedNova_Final.py.

Essas duas funções são PURAS (só numpy — não usam flwr/torch/net/dataloader),
mas moram dentro de FedNova_Final.py junto com o resto do baseline (mesmo
padrão dos outros *_Final.py: um único arquivo por método). Isso significa que
importar este teste executa, como efeito colateral, todo o código de nível de
módulo do FedNova_Final.py — carregamento do dataset (CIFAR10 por padrão),
criação do modelo e profiling de FLOPs (thop). Portanto ESTE teste depende de
torch/flwr/thop instalados e do dataset (baixado automaticamente na 1ª vez) —
mais lento que um teste puramente unitário, mas ainda assim não roda nenhuma
rodada de simulação federada de fato (isso só acontece em __main__).

Rodar com:  python sanity_tests/test_fednova_sanity.py
(ou, se pytest estiver instalado, a partir da raiz do projeto:
    pytest sanity_tests/test_fednova_sanity.py -v)
"""
import sys
from pathlib import Path

import numpy as np

# FedNova_Final.py mora em algoritmos/ — adiciona essa pasta ao sys.path para
# que este teste rode independentemente do diretório de onde é chamado.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "algoritmos"))

from FedNova_Final import effective_local_steps, fednova_aggregate

FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def allclose(a, b, tol=1e-8):
    return np.allclose(a, b, atol=tol, rtol=1e-6)


# ---------------------------------------------------------------------------
# 1. effective_local_steps: casos-base da fórmula de normalização
# ---------------------------------------------------------------------------

def test_tau_eff_sem_momento_igual_a_tau():
    """rho=0 (SGD puro): o nº efetivo de passos é exatamente tau_i, para vários tau_i."""
    for tau in (1, 2, 5, 10, 100):
        a = effective_local_steps(tau, rho=0.0)
        check(f"tau_eff(tau={tau}, rho=0) == {tau}", allclose(a, tau))


def test_tau_eff_um_passo_ignora_momento():
    """Com um único passo local, o momento (v_0=0) não teve chance de agir: a_i deve ser 1
    independentemente de rho."""
    for rho in (0.1, 0.5, 0.9, 0.99):
        a = effective_local_steps(1, rho=rho)
        check(f"tau_eff(tau=1, rho={rho}) == 1", allclose(a, 1.0), f"got {a}")


def test_tau_eff_maior_que_tau_quando_ha_momento():
    """Com momento > 0, o nº efetivo de passos é SEMPRE >= tau_i (cada passo pondera também
    os gradientes acumulados anteriores)."""
    for tau in (2, 5, 10, 50):
        for rho in (0.1, 0.5, 0.9):
            a = effective_local_steps(tau, rho)
            check(f"tau_eff(tau={tau}, rho={rho}) >= tau", a >= tau - 1e-9, f"got {a}")


def test_tau_eff_continuo_no_limite_rho_para_zero():
    """A fórmula com rho>0 deve convergir para tau_i conforme rho -> 0 (checa que não há
    descontinuidade/bug de sinal perto da divisão por (1-rho) e rho)."""
    for tau in (5, 20):
        a_small_rho = effective_local_steps(tau, rho=1e-6)
        check(
            f"tau_eff(tau={tau}, rho->0) ~ {tau}",
            allclose(a_small_rho, tau, tol=1e-3),
            f"got {a_small_rho}",
        )


def test_tau_eff_formula_manual_tau3_rho09():
    """Confere a fórmula fechada contra a soma explícita (dupla) para um caso concreto."""
    tau, rho = 3, 0.9
    manual = sum(sum(rho ** j for j in range(k + 1)) for k in range(tau))
    a = effective_local_steps(tau, rho)
    check(f"tau_eff(tau=3, rho=0.9) == soma manual ({manual:.6f})", allclose(a, manual), f"got {a}")


# ---------------------------------------------------------------------------
# 2. fednova_aggregate: casos degenerados e propriedades de consistência
# ---------------------------------------------------------------------------

def _make_params(seed, shapes):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=s).astype(np.float32) for s in shapes]


SHAPES = [(4, 3), (5,), (2, 2, 2)]  # simula pesos de conv/linear + bias + buffer BN


def test_reduz_a_fedavg_quando_tau_igual_sem_momento():
    """Sanidade central do FedNova: se todos os clientes fizeram o MESMO nº de passos
    locais (tau_i igual) e rho=0, a agregação deve ser IDÊNTICA à média ponderada
    simples do FedAvg (x_new = sum p_i * x_i_local), com server_lr=1."""
    old = _make_params(0, SHAPES)
    is_trainable = [True, True, True]
    tau = 20  # mesmo número de passos para todos os clientes
    n = [10, 30, 60]  # pesos (nº de amostras) DIFERENTES entre clientes

    clients_new = [_make_params(i + 1, SHAPES) for i in range(3)]
    updates = [(n[i], clients_new[i], tau) for i in range(3)]

    fednova_result = fednova_aggregate(old, updates, is_trainable, server_lr=1.0, rho=0.0)

    total_n = sum(n)
    fedavg_result = [
        sum((n[i] / total_n) * clients_new[i][k].astype(np.float64) for i in range(3))
        for k in range(len(SHAPES))
    ]

    for k in range(len(SHAPES)):
        check(
            f"tensor[{k}]: FedNova(tau homogêneo, rho=0) == FedAvg",
            allclose(fednova_result[k], fedavg_result[k], tol=1e-5),
        )


def test_reduz_a_fedavg_quando_tau_igual_com_momento():
    """Mesma propriedade acima, mas agora com rho>0: tau_eff_i é igual (e maior que tau)
    para todos os clientes, mas ainda cancela na fórmula e recupera o FedAvg puro."""
    old = _make_params(10, SHAPES)
    is_trainable = [True, True, True]
    tau = 15
    rho = 0.9
    n = [5, 15, 20]

    clients_new = [_make_params(i + 11, SHAPES) for i in range(3)]
    updates = [(n[i], clients_new[i], tau) for i in range(3)]

    fednova_result = fednova_aggregate(old, updates, is_trainable, server_lr=1.0, rho=rho)

    total_n = sum(n)
    fedavg_result = [
        sum((n[i] / total_n) * clients_new[i][k].astype(np.float64) for i in range(3))
        for k in range(len(SHAPES))
    ]

    for k in range(len(SHAPES)):
        check(
            f"tensor[{k}]: FedNova(tau homogêneo, rho={rho}) == FedAvg",
            allclose(fednova_result[k], fedavg_result[k], tol=1e-4),
        )


def test_um_unico_cliente_reduz_ao_proprio_modelo():
    """Com um único cliente participando da rodada, o resultado (server_lr=1) deve ser
    EXATAMENTE o modelo local dele, para qualquer tau_i/rho — a normalização e o
    tau_eff se cancelam perfeitamente no caso trivial de 1 cliente."""
    old = _make_params(20, SHAPES)
    is_trainable = [True, False, True]
    for tau, rho in ((7, 0.0), (7, 0.9), (1, 0.5)):
        client_new = _make_params(21, SHAPES)
        result = fednova_aggregate(old, [(42, client_new, tau)], is_trainable, server_lr=1.0, rho=rho)
        for k in range(len(SHAPES)):
            check(
                f"1 cliente (tau={tau}, rho={rho}): tensor[{k}] == modelo local",
                allclose(result[k], client_new[k], tol=1e-4),
            )


def test_buffers_nao_treinaveis_usam_media_simples():
    """Tensores marcados como NÃO treináveis (ex.: buffers de BatchNorm) devem receber
    sempre a média ponderada simples do valor NOVO, independente de tau_i/rho — nunca a
    normalização do FedNova (que só faz sentido para gradientes)."""
    old = _make_params(30, SHAPES)
    is_trainable = [True, False, True]  # índice 1 é o "buffer"
    n = [10, 90]
    taus = [3, 50]  # bem diferentes, para expor qualquer contaminação da normalização
    clients_new = [_make_params(31, SHAPES), _make_params(32, SHAPES)]
    updates = [(n[i], clients_new[i], taus[i]) for i in range(2)]

    result = fednova_aggregate(old, updates, is_trainable, server_lr=1.0, rho=0.9)

    total_n = sum(n)
    expected_buffer = sum((n[i] / total_n) * clients_new[i][1].astype(np.float64) for i in range(2))
    check(
        "buffer (índice 1) == média ponderada simples dos valores novos",
        allclose(result[1], expected_buffer, tol=1e-5),
    )


def test_tau_heterogeneo_diverge_de_fedavg_ingenuo():
    """Quando os clientes têm tau_i DIFERENTES, o FedNova deve produzir um resultado
    DIFERENTE da média ponderada ingênua por num_examples (essa é justamente a correção
    que o FedNova introduz para o "objective inconsistency problem"). Não checamos um
    valor específico (dependeria de recalcular a fórmula à mão) — só que a normalização
    tem efeito real quando tau_i varia."""
    old = _make_params(40, SHAPES)
    is_trainable = [True, True, True]
    n = [50, 50]  # mesmo peso por nº de amostras...
    taus = [5, 40]  # ...mas número de passos bem diferente
    clients_new = [_make_params(41, SHAPES), _make_params(42, SHAPES)]
    updates = [(n[i], clients_new[i], taus[i]) for i in range(2)]

    fednova_result = fednova_aggregate(old, updates, is_trainable, server_lr=1.0, rho=0.9)

    total_n = sum(n)
    naive_fedavg = [
        sum((n[i] / total_n) * clients_new[i][k].astype(np.float64) for i in range(2))
        for k in range(len(SHAPES))
    ]

    diverge = any(not allclose(fednova_result[k], naive_fedavg[k], tol=1e-6) for k in range(len(SHAPES)))
    check("FedNova(tau heterogêneo) != média ponderada ingênua (FedAvg)", diverge)


def test_server_lr_escala_o_passo():
    """Dobrar server_lr deve dobrar o DESLOCAMENTO em relação a old_params (para os
    tensores treináveis), já que x_new = old - server_lr * tau_eff * d_bar é linear em
    server_lr."""
    old = _make_params(50, SHAPES)
    is_trainable = [True, True, True]
    n = [10, 20]
    taus = [8, 12]
    clients_new = [_make_params(51, SHAPES), _make_params(52, SHAPES)]
    updates = [(n[i], clients_new[i], taus[i]) for i in range(2)]

    result_lr1 = fednova_aggregate(old, updates, is_trainable, server_lr=1.0, rho=0.5)
    result_lr2 = fednova_aggregate(old, updates, is_trainable, server_lr=2.0, rho=0.5)

    for k in range(len(SHAPES)):
        delta1 = old[k].astype(np.float64) - result_lr1[k].astype(np.float64)
        delta2 = old[k].astype(np.float64) - result_lr2[k].astype(np.float64)
        check(
            f"tensor[{k}]: deslocamento com server_lr=2 == 2x deslocamento com server_lr=1",
            allclose(delta2, 2.0 * delta1, tol=1e-4),
        )


def test_soma_examples_zero_levanta_erro():
    """Proteção contra divisão por zero: se todos os num_examples forem 0, deve levantar
    ValueError em vez de propagar NaN silenciosamente para o modelo global."""
    old = _make_params(60, SHAPES)
    is_trainable = [True, True, True]
    updates = [(0, _make_params(61, SHAPES), 5)]
    try:
        fednova_aggregate(old, updates, is_trainable)
        check("num_examples=0 total levanta ValueError", False, "não levantou exceção")
    except ValueError:
        check("num_examples=0 total levanta ValueError", True)


def test_tau_zero_nao_quebra_por_divisao_por_zero():
    """Um cliente que (por algum bug de telemetria) reporte tau_i=0 não deve derrubar a
    agregação com ZeroDivisionError — effective_local_steps blinda com max(tau_i, 1)."""
    old = _make_params(70, SHAPES)
    is_trainable = [True, True, True]
    updates = [(10, _make_params(71, SHAPES), 0)]
    try:
        result = fednova_aggregate(old, updates, is_trainable, rho=0.9)
        check("tau_i=0 não levanta exceção (tratado como 1)", all(np.all(np.isfinite(r)) for r in result))
    except ZeroDivisionError:
        check("tau_i=0 não levanta exceção (tratado como 1)", False, "ZeroDivisionError")


if __name__ == "__main__":
    testes = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    print(f"Rodando {len(testes)} testes de sanidade do FedNova...\n")
    for t in testes:
        t()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} teste(s) FALHARAM: {FAILURES}")
        raise SystemExit(1)
    else:
        print("Todos os testes de sanidade passaram.")
