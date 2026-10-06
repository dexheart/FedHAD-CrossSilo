# -*- coding: utf-8 -*-
"""
Coordenador de experimentos da campanha principal (FedAVG, FedAvgM, FedProx, FedNova, SCAFFOLD, FedHAD).

Roda execuções SEQUENCIAIS, controlando: seed global, nº de clientes (preset 3/5/10),
nº de rounds, communication delay, alpha do Dirichlet, dataset, ablação do FedHAD e
medição de energia. Cada execução é um SUBPROCESSO isolado: o respectivo *_Final.py
roda a própria simulação e salva o relatório .txt em results/main_campaign_default_alpha_0_5/<Metodo>-results/<dataset>/.

Como funciona:
  - Os parâmetros são passados por variáveis de ambiente FL_* (ver _build_env).
  - Cada *_Final.py lê essas envs como OVERRIDE no início; se a env não existir, usa
    o valor padrão hardcoded no próprio script (rodar o script sozinho continua igual).
  - ALGORITMOS_ATIVOS (no topo) escolhe quais métodos entram na rodada — útil para
    comparar apenas um subconjunto (ex.: só FedHAD e FedProx).
  - GRADE (produto cartesiano): qualquer parâmetro de um bloco pode ser uma LISTA. O
    coordenador expande TODAS as combinações. Ex.: alpha=[1, 0.5, 0.1, 0.01] com
    dataset=["MNIST","FASHION_MNIST","CIFAR10"] e 10 seeds -> 4 x 3 x 10 = 120 execuções
    por algoritmo. Vale para dataset, client_setup, num_rounds, comm_delay, alpha,
    energy e as flags de ablação do FedHAD.
  - Em cada bloco, a QUANTIDADE de seeds = nº de repetições de CADA combinação. Os
    seeds devem ser DIFERENTES (fonte de variância; mesmo seed = relatórios idênticos).
  - As flags de ablação (fedhad_*) entram no produto SÓ para o FedHAD; os baselines
    não são multiplicados por elas.

Uso:
    1) Ajuste ALGORITMOS_ATIVOS e edite a lista EXPERIMENTS abaixo.
    2) Rode (de preferência com o python da venv):  python runners/run_main_campaign.py

Retomada (continuar de onde parou):
    Se RETOMAR = True (padrão), cada execução concluída com sucesso é registrada em
    run_experiments_progress.jsonl. Se a rodada for interrompida (erro, Ctrl+C, queda),
    basta relançar  python runners/run_main_campaign.py  — ele PULA as já concluídas e recomeça
    da execução seguinte. A identificação é por conteúdo (algoritmo + seed + parâmetros),
    não por ordem. Flags de linha de comando:
      --reset      apaga o progresso e recomeça tudo do zero
      --no-resume  ignora o progresso nesta chamada (não pula nada; útil p/ re-rodar tudo)
"""
import os
import sys
import csv
import json
import time
import itertools
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]  # raiz do repositório (este runner fica em runners/)

# Os experimentos são lançados com sys.executable. Se este coordenador for chamado com um
# Python sem PyTorch/Flower (ex.: o "python" do sistema/JupyterHub), todos falham no
# import em ~0 s. Por isso ele se reexecuta com o Python do ambiente do projeto
# (env_flwr_pt), como os demais runners do repositório.
_VENV = REPO_DIR / "env_flwr_pt"
if (__name__ == "__main__" and (_VENV / "bin" / "python").exists()
        and Path(sys.prefix).resolve() != _VENV.resolve() and not os.environ.get("RUNEXP_NO_REEXEC")):
    os.environ["RUNEXP_NO_REEXEC"] = "1"
    os.execv(str(_VENV / "bin" / "python"), [str(_VENV / "bin" / "python"), str(Path(__file__).resolve()), *sys.argv[1:]])

# Pastas do projeto: scripts dos algoritmos, resultados (<Metodo>-results/) e logs.
ALGORITMOS_DIR = REPO_DIR / "algoritmos"
RESULTS_DIR = REPO_DIR / "results" / "main_campaign_default_alpha_0_5"
LOGS_DIR = REPO_DIR / "results" / "execution_logs"

# Nome do método -> arquivo do script
ALGORITMOS = {
    "FedAVG":  "FedAVG_Final.py",
    "FedAvgM": "FedAvgM_Final.py",
    "FedProx": "FedProx_Final.py",
    "FedNova": "FedNova_Final.py",
    "FedHAD":  "FedHAD_Final_2.0.py",
    "SCAFFOLD": "SCAFFOLD_Final.py",  # correção explícita de drift (Karimireddy et al., 2020)
}

# ===========================================================================
# SELEÇÃO GLOBAL DE ALGORITMOS — controle central de quais métodos entram na rodada.
# Edite só esta lista para restringir os experimentos a um subconjunto.
#   Ex.: comparar apenas FedHAD e FedProx -> ALGORITMOS_ATIVOS = ["FedHAD", "FedProx"]
# Um algoritmo citado num bloco de EXPERIMENTS só roda se também estiver AQUI.
# (Para rodar todos, deixe os 4 na lista.)
# ===========================================================================
ALGORITMOS_ATIVOS = ["SCAFFOLD"]
# (Cada bloco já declara seus "algorithms"; esta lista é um filtro global adicional —
#  útil para, p.ex., re-rodar só o FedHAD de um teste: ALGORITMOS_ATIVOS = ["FedHAD"].)
# CONFIGURAÇÃO ATUAL: rodando SÓ o SCAFFOLD, na MESMA bateria documentada para o FedNova
# no artigo (ponto base, família de heterogeneidade e federação natural; 10 rodadas),
# sem re-executar os outros métodos. (Rodada anterior: ["FedNova"], mesmos 3 blocos.)
# Para voltar à comparação completa, use: ALGORITMOS_ATIVOS = _TODOS.

# Presets de clientes VÁLIDOS — devem coincidir com as chaves de CLIENT_SETUP_CONFIG nos
# *_Final.py. Atenção: no FEMNIST, o client_setup também define QUAIS writers são usados
# (CLIENT_SETUP_CONFIG[...]["femnist_writers"]); usar um valor fora desta lista quebraria
# a amarração writer<->cliente e o script abortaria. Para criar um novo nº de clientes
# (ex.: 7), adicione o preset (num_gpus + femnist_writers) em TODOS os *_Final.py E aqui.
VALID_CLIENT_SETUPS = (3, 5, 10)

# Datasets suportados pelos scripts.
VALID_DATASETS = ("MNIST", "FASHION_MNIST", "CIFAR10", "FEMNIST")

# Seeds usados em TODOS os testes (30 repetições, conforme a metodologia da tese). Mais
# seeds = mais robustez estatística e MAIS tempo. Para um teste rápido, reduza esta lista.
SEEDS_30 = list(range(42, 72))  # 42, 43, ..., 71  -> 30 seeds

# ===========================================================================
# SELEÇÃO DE EXPERIMENTOS — quais testes (por "name") rodar NESTA chamada.
# A grade completa dos 8 testes tem MILHARES de execuções; rode UM teste por vez.
#   Ex.: EXPERIMENTOS_ATIVOS = ["test2_robustez_alpha"]
#   []  (lista vazia) = roda TODOS os blocos de EXPERIMENTS.
# ===========================================================================
# CONFIGURAÇÃO ATUAL (SCAFFOLD, mesma bateria do FedNova): os 3 blocos desta rodada.
# Fora desta lista (podem ser reativados depois acrescentando o nome aqui):
#   - test3_comunicacao, test4_clientes, test7_plato: aplicáveis ao FedNova, só não
#     entram nesta rodada.
#   - test5_ablacao e test6_calibracao: exclusivos do FedHAD (ligam/desligam e calibram
#     as heurísticas dele); não existe equivalente no FedNova.
EXPERIMENTOS_ATIVOS = [
    "test1_convergencia",
    "test2_robustez_alpha",
    "test8_femnist",
]  # [] = todos os blocos de EXPERIMENTS

# ===========================================================================
# RETOMADA — continuar de onde parou.
# Se True, o coordenador registra cada execução concluída COM SUCESSO (rc==0) no arquivo
# de progresso (ARQUIVO_PROGRESSO) e, ao ser relançado, PULA as que já terminaram. Assim,
# se a rodada for interrompida (erro, Ctrl+C, queda de energia), basta rodar de novo:
#       python runners/run_main_campaign.py
# que ele recomeça exatamente da execução seguinte.
#   - A identificação é por CONTEÚDO (algoritmo + seed + parâmetros concretos), não por
#     ordem: reordenar/adicionar blocos NÃO re-roda o que já foi feito, e o que já foi
#     feito com os MESMOS parâmetros nunca roda duas vezes.
#   - A execução que foi interrompida no meio NÃO conta como concluída (rc != 0), então
#     ela roda de novo — sem relatórios pela metade.
#   - RETROATIVO: além do .jsonl, o coordenador também lê os LOGS CSV existentes
#     (run_experiments_log_*.csv) e trata cada linha com returncode==0 como concluída.
#     Isso permite retomar rodadas feitas ANTES de existir o arquivo de progresso. O CSV
#     não registra 'energy' nem os 'fedhad_*'; por isso a retomada por CSV só pula uma
#     execução quando ela é INEQUÍVOCA no plano atual (nenhuma outra execução planejada
#     tem os mesmos campos do CSV) — nos testes de ablação/calibração isso evita pular a
#     combinação errada (essas contam só com o .jsonl daqui pra frente).
#   - Para COMEÇAR DO ZERO (re-rodar tudo), apague o .jsonl (e/ou os CSV) ou use --reset.
#   - Para IGNORAR o progresso só nesta chamada (sem apagar), rode com  --no-resume.
#   - Para só VER o plano de retomada sem executar nada, rode com  --dry-run.
# ===========================================================================
RETOMAR = True
ARQUIVO_PROGRESSO = LOGS_DIR / "run_experiments_progress.jsonl"
# Padrão dos logs CSV do coordenador, usados na retomada retroativa (ver acima).
LOG_CSV_GLOB = "run_experiments_log_*.csv"

# ===========================================================================
# PLANO DE EXPERIMENTOS — edite aqui.
#
# Qualquer campo de grade pode ser um VALOR ÚNICO ou uma LISTA. O bloco expande o
# produto cartesiano de todas as listas, e cada combinação roda para cada seed.
# Campos por bloco:
#   algorithms   : lista de métodos (de ALGORITMOS); cada um só roda se estiver em ALGORITMOS_ATIVOS
#   dataset      : "MNIST" | "FASHION_MNIST" | "CIFAR10" | "FEMNIST"  (ou lista deles)
#   client_setup : 3, 5 ou 10                              (ou lista)
#                  -> no FEMNIST, também escolhe QUAIS writers (CLIENT_SETUP_CONFIG[...]
#                     ["femnist_writers"] em cada script); varie client_setup para variar clientes.
#   num_rounds   : nº de rounds federados                 (ou lista)
#   comm_delay   : delay de comunicação simulado (s)       (ou lista)
#   alpha        : alpha do Dirichlet (menor = mais não-IID) (ou lista)
#                  -> IGNORADO no FEMNIST (particionamento natural por writer). NÃO use
#                     'alpha' como lista em blocos de FEMNIST: geraria execuções idênticas.
#   energy       : True/False (medição de energia CodeCarbon) (ou lista)
#   seeds        : lista de seeds; a QUANTIDADE = nº de repetições de CADA combinação
#   Ablação do FedHAD (cada uma True, False ou [True, False]); entra no produto SÓ p/ FedHAD:
#     fedhad_dynamic_epochs, fedhad_dynamic_lr, fedhad_fedprox
#
# Campos omitidos usam o valor padrão do próprio script.
# Nº de execuções de um bloco = (produto das listas de grade) x nº de seeds x nº de algoritmos.
#
# IMPORTANTE (estatística): os seeds devem ser DIFERENTES entre si — são a fonte de
# variância das repetições (mesmo seed = relatórios idênticos, desvio padrão = 0).
# ===========================================================================
_TODOS = ["FedAVG", "FedAvgM", "FedProx", "FedNova", "FedHAD"]  # atalho: 5 algoritmos

EXPERIMENTS = [
    # ===================== TESTE 1 — Convergência =====================
    # Varia só o dataset; congela alpha=0.5, clientes=5, delay=0.05, 10 rounds.
    {
        "name": "test1_convergencia",
        "algorithms": _TODOS + ["SCAFFOLD"],  # SCAFFOLD: mesma bateria do FedNova
        "dataset": ["MNIST", "FASHION_MNIST", "CIFAR10"],
        "client_setup": 5, "num_rounds": 10, "comm_delay": 0.05, "alpha": 0.5,
        "energy": True, "seeds": SEEDS_30,
    },

    # ===================== TESTE 2 — Robustez (alpha) =====================
    # Varia alpha; 3 datasets; clientes=5, delay=0.05, 10 rounds.
    {
        "name": "test2_robustez_alpha",
        "algorithms": _TODOS + ["SCAFFOLD"],  # SCAFFOLD: mesma bateria do FedNova
        "dataset": ["MNIST", "FASHION_MNIST", "CIFAR10"],
        "client_setup": 5, "num_rounds": 10, "comm_delay": 0.05,
        "alpha": [1.0, 0.1, 0.01],
        "energy": True, "seeds": SEEDS_30,
    },

    # ===================== TESTE 3 — Comunicação (delay) =====================
    # Varia o communication delay; 3 datasets; clientes=5, alpha=0.5, 10 rounds.
    {
        "name": "test3_comunicacao",
        "algorithms": _TODOS,
        "dataset": ["MNIST", "FASHION_MNIST", "CIFAR10"],
        "client_setup": 5, "num_rounds": 10, "alpha": 0.5,
        "comm_delay": [0.05, 0.1, 0.5],
        "energy": True, "seeds": SEEDS_30,
    },

    # ===================== TESTE 4 — Participação de clientes =====================
    # Varia client_setup; só CIFAR10; alpha=0.5, delay=0.05, 10 rounds.
    {
        "name": "test4_clientes",
        "algorithms": _TODOS,
        "dataset": "CIFAR10",
        "client_setup": [3, 5, 10], "num_rounds": 10, "comm_delay": 0.05, "alpha": 0.5,
        "energy": True, "seeds": SEEDS_30,
    },

    # ===================== TESTE 5 — Ablação (só FedHAD) =====================
    # Liga/desliga cada heurística (8 combos). CIFAR10, 5 clientes, 10 rounds, delay=0.05,
    # alpha=0.01 (bem heterogêneo, para estressar o client drift).
    {
        "name": "test5_ablacao",
        "algorithms": ["FedHAD"],
        "dataset": "CIFAR10",
        "client_setup": 5, "num_rounds": 10, "comm_delay": 0.05, "alpha": 0.01,
        "energy": True, "seeds": SEEDS_30,
        "fedhad_dynamic_epochs": [True, False],
        "fedhad_dynamic_lr": [True, False],
        "fedhad_fedprox": [True, False],
    },

    # ===================== TESTE 6 — Calibração do FedHAD =====================
    # Varia os hiperparâmetros das heurísticas (sensibilidade). Padrão: CIFAR10, 5 clientes,
    # alpha=0.5, delay=0.05, 10 rounds. Exemplo varia os 2 fatores de decaimento; outras
    # variáveis disponíveis (cuidado: o produto cartesiano multiplica as execuções):
    #   fedhad_base_epochs, fedhad_min_epochs, fedhad_base_lr, fedhad_min_lr, fedhad_mu
    {
        "name": "test6_calibracao",
        "algorithms": ["FedHAD"],
        "dataset": "CIFAR10",
        "client_setup": 5, "num_rounds": 10, "comm_delay": 0.05, "alpha": 0.5,
        "energy": True, "seeds": SEEDS_30,
        "fedhad_epochs_decay": [1.0, 3.0, 5.0],
        "fedhad_lr_decay": [0.5, 1.5, 3.0],
    },

    # ===================== TESTE 7 — Platô (muitos rounds) =====================
    # 50 rounds, parâmetros padrão, só CIFAR10.
    {
        "name": "test7_plato",
        "algorithms": _TODOS,
        "dataset": "CIFAR10",
        "client_setup": 5, "num_rounds": 50, "comm_delay": 0.05, "alpha": 0.5,
        "energy": True, "seeds": SEEDS_30,
    },

    # ===================== TESTE 8 — FEMNIST =====================
    # Dataset nativo-federado; 10 clientes (writers fixos); resto padrão. 'alpha' NÃO se aplica.
    {
        "name": "test8_femnist",
        "algorithms": _TODOS + ["SCAFFOLD"],  # SCAFFOLD: mesma bateria do FedNova
        "dataset": "FEMNIST",
        "client_setup": 10, "num_rounds": 10, "comm_delay": 0.05,
        "energy": True, "seeds": SEEDS_30,
    },
]
# ===========================================================================


def _build_env(exp: dict, seed: int) -> dict:
    """Monta o ambiente do subprocesso com as variáveis FL_* do experimento."""
    env = os.environ.copy()
    # Evita o crash de UTF-8 do Ray no Windows também no processo driver.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    env["FL_SEED"] = str(seed)
    if "dataset" in exp:
        env["FL_DATASET"] = str(exp["dataset"])
    if "client_setup" in exp:
        env["FL_CLIENT_SETUP"] = str(exp["client_setup"])
    if "num_rounds" in exp:
        env["FL_NUM_ROUNDS"] = str(exp["num_rounds"])
    if "comm_delay" in exp:
        env["FL_COMM_DELAY"] = str(exp["comm_delay"])
    if "alpha" in exp:
        env["FL_ALPHA"] = str(exp["alpha"])
    if "energy" in exp:
        env["FL_USE_ENERGY"] = "1" if exp["energy"] else "0"

    # Ablação do FedHAD (booleans -> "1"/"0").
    if "fedhad_dynamic_epochs" in exp:
        env["FL_USE_DYNAMIC_EPOCHS"] = "1" if exp["fedhad_dynamic_epochs"] else "0"
    if "fedhad_dynamic_lr" in exp:
        env["FL_USE_DYNAMIC_LR"] = "1" if exp["fedhad_dynamic_lr"] else "0"
    if "fedhad_fedprox" in exp:
        env["FL_USE_FEDPROX"] = "1" if exp["fedhad_fedprox"] else "0"

    # Variáveis de calibração das heurísticas do FedHAD (teste de sensibilidade).
    for chave, envname in FEDHAD_CALIB_ENV.items():
        if chave in exp:
            env[envname] = str(exp[chave])

    # Etiqueta do experimento -> subpasta de resultados (FL_RUN_TAG) nos *_Final.py.
    if exp.get("_name"):
        env["FL_RUN_TAG"] = str(exp["_name"])

    # Raiz dos resultados (<Metodo>-results/ fica dentro dela) nos *_Final.py.
    env["FL_RESULTS_DIR"] = str(RESULTS_DIR)

    return env


def _seeds_for(exp) -> list:
    """Seeds do bloco = a lista 'seeds' (a quantidade define o nº de execuções)."""
    seeds = exp.get("seeds")
    if not seeds:
        raise ValueError("Cada bloco de EXPERIMENTS precisa de uma lista 'seeds' não vazia.")
    return [int(s) for s in seeds]


# Campos que podem virar GRADE (valor único ou lista -> produto cartesiano).
COMMON_GRID_KEYS = ["dataset", "client_setup", "num_rounds", "comm_delay", "alpha", "energy"]
# Campos exclusivos do FedHAD (ablação + calibração das heurísticas); entram no produto
# SÓ para o FedHAD (os baselines não são multiplicados por eles).
FEDHAD_GRID_KEYS = [
    "fedhad_dynamic_epochs", "fedhad_dynamic_lr", "fedhad_fedprox",
    "fedhad_base_epochs", "fedhad_min_epochs", "fedhad_epochs_decay",
    "fedhad_base_lr", "fedhad_min_lr", "fedhad_lr_decay", "fedhad_mu",
]

# Mapeamento das variáveis de calibração do FedHAD -> variáveis de ambiente FL_*.
FEDHAD_CALIB_ENV = {
    "fedhad_base_epochs": "FL_BASE_EPOCHS",
    "fedhad_min_epochs": "FL_MIN_EPOCHS",
    "fedhad_epochs_decay": "FL_EPOCHS_DECAY",
    "fedhad_base_lr": "FL_BASE_LR",
    "fedhad_min_lr": "FL_MIN_LR",
    "fedhad_lr_decay": "FL_LR_DECAY",
    "fedhad_mu": "FL_FEDPROX_MU",
}


def _as_list(value) -> list:
    """Normaliza um campo de grade para lista (valor único -> lista de 1 elemento)."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _combos_for_algo(exp, algo):
    """
    Gera as combinações concretas (produto cartesiano) dos campos de grade do bloco,
    para um dado algoritmo. As flags de ablação (fedhad_*) só entram no produto quando
    algo == 'FedHAD' (os baselines não são multiplicados por elas).
    Cada combinação é um dict {chave: valor_concreto} com apenas as chaves presentes.
    """
    keys = [k for k in COMMON_GRID_KEYS if k in exp]
    if algo == "FedHAD":
        keys += [k for k in FEDHAD_GRID_KEYS if k in exp]
    if not keys:
        yield {}
        return
    value_lists = [_as_list(exp[k]) for k in keys]
    for combo in itertools.product(*value_lists):
        yield dict(zip(keys, combo))


def _run_key(algo, seed, params) -> str:
    """
    Assinatura estável de uma execução (algoritmo + seed + parâmetros concretos), usada
    para marcar/detectar execuções já concluídas na RETOMADA. É por conteúdo (não por
    ordem): a MESMA combinação sempre gera a MESMA chave. sort_keys garante estabilidade.
    """
    payload = {"algo": algo, "seed": int(seed), "params": params}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _carregar_concluidas(path) -> set:
    """Lê o arquivo de progresso e devolve o conjunto de chaves (_run_key) já concluídas."""
    concluidas = set()
    if not path.exists():
        return concluidas
    with path.open("r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                reg = json.loads(linha)
            except json.JSONDecodeError:
                continue  # linha corrompida (ex.: escrita interrompida) — ignora
            chave = reg.get("key")
            if chave:
                concluidas.add(chave)
    return concluidas


def _marcar_concluida(path, algo, seed, params):
    """Anexa (append + flush) uma execução concluída ao arquivo de progresso. O append
    incremental garante que, mesmo com queda no meio do lote, o já feito fica registrado."""
    reg = {
        "key": _run_key(algo, seed, params),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experimento": params.get("_name", ""),
        "algoritmo": algo,
        "seed": int(seed),
        "dataset": params.get("dataset", ""),
    }
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        f.flush()


# --- Retomada retroativa a partir dos logs CSV -----------------------------------------
# O CSV registra: experimento, algoritmo, seed, dataset, client_setup, num_rounds, alpha,
# comm_delay, returncode. NÃO registra 'energy' nem os 'fedhad_*'. Por isso a chave CSV é
# feita só desses campos; a ambiguidade que isso possa gerar é tratada em main() (só pula
# execuções cuja chave CSV seja única no plano atual).

def _norm(v) -> str:
    """Normaliza um valor para comparação de chaves CSV (str; '' quando ausente)."""
    return "" if v is None else str(v)


def _csv_key_from_run(algo, seed, params) -> tuple:
    """Chave CSV de uma execução PLANEJADA (mesmos campos e formatação do log CSV)."""
    return (
        _norm(params.get("_name", "")),
        _norm(algo),
        _norm(int(seed)),
        _norm(params.get("dataset", "")),
        _norm(params.get("client_setup", "")),
        _norm(params.get("num_rounds", "")),
        _norm(params.get("alpha", "")),
        _norm(params.get("comm_delay", "")),
    )


def _csv_key_from_row(row: dict) -> tuple:
    """Chave CSV a partir de uma linha do log (mesma ordem de _csv_key_from_run)."""
    return (
        _norm(row.get("experimento", "")),
        _norm(row.get("algoritmo", "")),
        _norm(row.get("seed", "")),
        _norm(row.get("dataset", "")),
        _norm(row.get("client_setup", "")),
        _norm(row.get("num_rounds", "")),
        _norm(row.get("alpha", "")),
        _norm(row.get("comm_delay", "")),
    )


def _carregar_concluidas_csv(paths) -> set:
    """Lê os logs CSV e devolve o conjunto de chaves CSV com returncode == 0 (sucesso)."""
    done = set()
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if _norm(row.get("returncode")).strip() == "0":
                        done.add(_csv_key_from_row(row))
        except (OSError, csv.Error):
            continue
    return done


def _plan_runs():
    """
    Expande EXPERIMENTS numa lista linear de execuções (algo, seed, params), onde params
    é uma combinação concreta da grade. Aplica o filtro global ALGORITMOS_ATIVOS.
    """
    for a in ALGORITMOS_ATIVOS:
        if a not in ALGORITMOS:
            raise ValueError(f"ALGORITMOS_ATIVOS contém método desconhecido: {a}. Use: {list(ALGORITMOS)}")

    # Filtra os blocos ATIVOS por nome (EXPERIMENTOS_ATIVOS). Lista vazia = todos.
    nomes = [exp.get("name") for exp in EXPERIMENTS if exp.get("name")]
    if EXPERIMENTOS_ATIVOS:
        for nome in EXPERIMENTOS_ATIVOS:
            if nome not in nomes:
                raise ValueError(
                    f"EXPERIMENTOS_ATIVOS contém nome inexistente: {nome!r}. Disponíveis: {nomes}"
                )
        blocos = [exp for exp in EXPERIMENTS if exp.get("name") in EXPERIMENTOS_ATIVOS]
    else:
        blocos = list(EXPERIMENTS)

    # Validação fail-fast de cada bloco ATIVO ANTES de lançar qualquer subprocesso. Pega
    # erros como um client_setup fora dos presets (quebraria a amarração de writers no
    # FEMNIST) ou um dataset inexistente — em vez de descobrir só quando o subprocesso falha.
    for exp in blocos:
        nome = exp.get("name", "?")
        for cs in _as_list(exp.get("client_setup", VALID_CLIENT_SETUPS[0])):
            if int(cs) not in VALID_CLIENT_SETUPS:
                raise ValueError(
                    f"Experimento {nome!r}: client_setup={cs} inválido. "
                    f"Use um de {list(VALID_CLIENT_SETUPS)} — presets de CLIENT_SETUP_CONFIG nos "
                    f"*_Final.py (no FEMNIST também definem os writers). Para um novo nº de "
                    f"clientes, adicione o preset em TODOS os scripts e em VALID_CLIENT_SETUPS."
                )
        for ds in _as_list(exp.get("dataset", VALID_DATASETS[0])):
            if ds not in VALID_DATASETS:
                raise ValueError(
                    f"Experimento {nome!r}: dataset={ds!r} inválido. Use um de {list(VALID_DATASETS)}."
                )

    runs = []
    for exp in blocos:
        nome = exp.get("name")
        seeds = _seeds_for(exp)
        for algo in exp["algorithms"]:
            if algo not in ALGORITMOS:
                raise ValueError(f"Algoritmo desconhecido: {algo}. Use: {list(ALGORITMOS)}")
            if algo not in ALGORITMOS_ATIVOS:
                continue  # filtrado pela seleção global
            for params in _combos_for_algo(exp, algo):
                params = dict(params)
                if nome:
                    params["_name"] = nome  # vira FL_RUN_TAG (subpasta de resultados)
                for seed in seeds:
                    runs.append((algo, seed, params))
    return runs


def _resumo_run(i, total, algo, seed, exp):
    return (f"[{i}/{total}] {exp.get('_name', '-')} | {algo} | seed={seed} "
            f"| dataset={exp.get('dataset', 'padrão')} | setup={exp.get('client_setup', 'padrão')} "
            f"| rounds={exp.get('num_rounds', 'padrão')} | alpha={exp.get('alpha', 'padrão')} "
            f"| delay={exp.get('comm_delay', 'padrão')}")


def main():
    # Flags de retomada (ver bloco RETOMADA no topo).
    resume = RETOMAR
    usar_csv = True  # retomada retroativa a partir dos logs CSV (desligada no --reset)
    if "--reset" in sys.argv:
        if ARQUIVO_PROGRESSO.exists():
            ARQUIVO_PROGRESSO.unlink()
            print(f"[--reset] Progresso apagado ({ARQUIVO_PROGRESSO.name}) — recomeçando do zero.")
        resume = True       # começa do zero, mas volta a registrar o progresso desta rodada
        usar_csv = False    # ignora também o histórico dos logs CSV (reset de verdade)
    if "--no-resume" in sys.argv:
        resume = False  # ignora o progresso nesta chamada (não pula nada)

    dry_run = "--dry-run" in sys.argv

    runs = _plan_runs()
    total = len(runs)

    # --- RETOMADA: monta o conjunto de execuções já concluídas -------------------------
    # Duas fontes: (1) o arquivo de progresso .jsonl (identidade completa, por _run_key);
    # (2) os logs CSV existentes (retroativo, para rodadas feitas antes do .jsonl).
    concluidas_jsonl = _carregar_concluidas(ARQUIVO_PROGRESSO) if resume else set()
    csv_paths = sorted(LOGS_DIR.glob(LOG_CSV_GLOB)) if (resume and usar_csv) else []
    concluidas_csv = _carregar_concluidas_csv(csv_paths)

    # O CSV não distingue 'energy'/'fedhad_*'; só podemos pular por CSV as execuções cuja
    # chave CSV seja ÚNICA no plano atual (senão pularíamos a combinação errada).
    contagem_csv = Counter(_csv_key_from_run(a, s, p) for (a, s, p) in runs)
    csv_keys_unicas = {k for k, n in contagem_csv.items() if n == 1}

    def _ja_concluida(algo, seed, params) -> bool:
        if _run_key(algo, seed, params) in concluidas_jsonl:
            return True
        ck = _csv_key_from_run(algo, seed, params)
        return ck in csv_keys_unicas and ck in concluidas_csv

    if resume:
        pendentes = [r for r in runs if not _ja_concluida(r[0], r[1], r[2])]
    else:
        pendentes = list(runs)
    n_puladas = total - len(pendentes)
    n_pend = len(pendentes)

    print("=" * 78)
    print(f"COORDENADOR DE EXPERIMENTOS — {total} execução(ões) planejada(s)")
    if resume:
        print(f"RETOMADA ATIVA — {n_puladas} já concluída(s) (puladas), {n_pend} pendente(s).")
        print(f"  Fontes: {ARQUIVO_PROGRESSO.name} ({len(concluidas_jsonl)} reg.) "
              f"+ {len(csv_paths)} log(s) CSV ({len(concluidas_csv)} execução(ões) rc=0).")
    else:
        print("RETOMADA DESATIVADA (--no-resume) — nenhuma execução será pulada.")
    if dry_run:
        print("MODO --dry-run: apenas listando o plano; NADA será executado.")
    print(f"Interpretador: {sys.executable}")
    print("=" * 78)
    for i, (algo, seed, exp) in enumerate(pendentes, 1):
        print("  " + _resumo_run(i, n_pend, algo, seed, exp))
    print("=" * 78)

    if n_pend == 0:
        print("Nada a fazer: todas as execuções planejadas já foram concluídas.")
        print(f"(Para re-rodar tudo: apague {ARQUIVO_PROGRESSO.name} e os logs CSV, ou use --reset.)")
        return

    if dry_run:
        print(f"[--dry-run] {n_pend} execução(ões) seriam executadas. Remova --dry-run para rodar.")
        return

    # Log CSV desta execução do coordenador. O timestamp no nome evita sobrescrever
    # logs de execuções anteriores (cada chamada gera um arquivo novo).
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"run_experiments_log_{ts}.csv"
    log_file = log_path.open("w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "timestamp", "experimento", "algoritmo", "seed", "dataset", "client_setup",
        "num_rounds", "alpha", "comm_delay", "returncode", "duracao_s",
    ])

    resultados = []
    inicio_total = time.time()
    try:
        for i, (algo, seed, exp) in enumerate(pendentes, 1):
            script = ALGORITMOS[algo]
            env = _build_env(exp, seed)
            cabecalho = _resumo_run(i, n_pend, algo, seed, exp)
            print(f"\n>>> INICIANDO  {cabecalho}   ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

            t0 = time.time()
            rc = -1
            try:
                proc = subprocess.run(
                    [sys.executable, str(ALGORITMOS_DIR / script)],
                    env=env, cwd=str(REPO_DIR),
                )
                rc = proc.returncode
            except KeyboardInterrupt:
                print("\n[Ctrl+C] Interrompido pelo usuário. Abortando o lote.")
                raise
            except Exception as e:
                print(f"!!! Falha ao lançar {script}: {e}")
                rc = -1
            dur = time.time() - t0

            status = "OK" if rc == 0 else f"FALHOU (rc={rc})"
            print(f"<<< {status}  {cabecalho} — {dur:.1f}s")

            # RETOMADA: só marca como concluída se terminou com sucesso (rc==0). Assim,
            # uma execução interrompida/falhada roda de novo na próxima chamada.
            if resume and rc == 0:
                _marcar_concluida(ARQUIVO_PROGRESSO, algo, seed, exp)

            resultados.append((i, algo, seed, exp, rc, dur))
            log_writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), exp.get("_name", ""), algo, seed,
                exp.get("dataset", ""), exp.get("client_setup", ""),
                exp.get("num_rounds", ""), exp.get("alpha", ""),
                exp.get("comm_delay", ""), rc, f"{dur:.1f}",
            ])
            log_file.flush()
    except KeyboardInterrupt:
        pass
    finally:
        log_file.close()

    # Resumo final.
    print("\n" + "=" * 78)
    print("RESUMO DAS EXECUÇÕES")
    ok = sum(1 for r in resultados if r[4] == 0)
    print(f"Sucessos: {ok}/{len(resultados)} | Tempo total: {(time.time() - inicio_total) / 60:.1f} min")
    for i, algo, seed, exp, rc, dur in resultados:
        marca = "OK  " if rc == 0 else "ERRO"
        print(f"  [{marca}] {i:>3} {algo:8s} seed={seed} "
              f"dataset={exp.get('dataset', '-')} alpha={exp.get('alpha', '-')} "
              f"setup={exp.get('client_setup', '-')} — {dur:.1f}s")
    print(f"\nLog salvo em: {log_path}")
    if resume:
        print(f"Progresso registrado em: {ARQUIVO_PROGRESSO}")
    falhas = [r for r in resultados if r[4] != 0]
    if falhas:
        print(f"ATENÇÃO: {len(falhas)} execução(ões) falharam — verifique o output de cada subprocesso acima.")
        if resume:
            print("As que falharam NÃO foram marcadas como concluídas: rode "
                  "'python runners/run_main_campaign.py' de novo para retomar só as pendentes.")


if __name__ == "__main__":
    main()
