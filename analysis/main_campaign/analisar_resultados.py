# -*- coding: utf-8 -*-
"""
Consolidação estatística dos resultados (.txt) gerados pelos *_Final.py.

Lê o CONTEÚDO de cada relatório .txt (não depende do nome do arquivo): extrai os
parâmetros (seção "--- Parâmetros ---", incl. EXPERIMENT_TAG) e as métricas
(acurácia final/melhor, rounds-to-target, FLOPs totais, energia, emissões, tempo).
Agrupa por CONFIGURAÇÃO (tudo, menos a seed) e calcula média ± desvio sobre as seeds.

Para CADA teste encontrado (agrupado por EXPERIMENT_TAG) gera, dentro da pasta
'results/main_campaign_summaries/', um par de CSVs com o nome do teste no arquivo:
  - analise_bruta_<tag>.csv   : 1 linha por execução (todos os params + métricas)
  - analise_resumo_<tag>.csv  : 1 linha por configuração
                                (média/desvio-padrão/variância/n sobre as seeds)

Uso:
    python analisar_resultados.py                 # gera um par de CSVs para CADA teste encontrado
    python analisar_resultados.py test2_robustez_alpha   # gera só o par do teste indicado

Só usa a biblioteca padrão (sem pandas).
"""
import re
import ast
import csv
import sys
import statistics
from pathlib import Path
from collections import defaultdict

# Raiz do projeto (este script fica em analysis/main_campaign/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Pasta onde os *_Final.py gravam os relatórios (<Metodo>-results/).
RESULTS_DIR = PROJECT_ROOT / "results" / "main_campaign_default_alpha_0_5"

# Pasta onde todos os CSVs de análise são gravados.
OUTPUT_DIR = PROJECT_ROOT / "results" / "main_campaign_summaries"

# Colunas que IDENTIFICAM uma configuração (a agregação é feita sobre as seeds destas).
# Inclui os parâmetros de calibração do FedHAD (epochs_decay/lr_decay/...); nos testes que
# não os variam eles são constantes (não criam grupos extra), mas no test6_calibracao
# distinguem corretamente as 9 combinações — sem isto, colapsariam num único grupo n=270.
GROUP_COLS = [
    "experiment_tag", "metodo", "dataset", "alpha_key", "client_setup",
    "num_rounds", "comm_delay", "heuristics_signature",
    "epochs_decay", "lr_decay", "base_lr", "min_lr", "fedprox_mu",
]

# Métricas numéricas a agregar (média/desvio). r2t_* (rounds-to-target) são detectadas
# dinamicamente e tratadas à parte.
METRIC_COLS = [
    "acuracia_final", "acuracia_melhor", "total_tflops",
    "energia_kwh", "energia_ajustada_kwh", "emissoes_kg", "tempo_s",
    "epocas", "tflops_por_epoca",  # FedHAD=dinâmicas; baselines=fixas (5). Preenchido p/ todos.
    "modelo_mb", "bytes_total_mb",  # custo de comunicação (secção "Custo de Comunicação")
]

# Prefixos das métricas "até-atingir-target" (dinâmicas por %), agregadas com um "n_atingiu"
# à parte: r2t_=rodadas, flops2t_=TFLOPs, bytes2t_=MB de comunicação.
TARGET_PREFIXOS = ("r2t_", "flops2t_", "bytes2t_")

# Colunas de FLOPs em que o valor 0 indica execução corrompida/crashada e deve
# ser DESCONSIDERADO no cálculo das estatísticas (média/desvio/variância).
FLOP_COLS = {"total_tflops", "tflops_por_epoca"}

_FLOP_UNIT = {"TFLOPs": 1.0, "GFLOPs": 1e-3, "MFLOPs": 1e-6, "KFLOPs": 1e-9, "FLOPs": 1e-12}


def _f(txt):
    """Converte string para float, ou None se falhar."""
    try:
        return float(txt)
    except (TypeError, ValueError):
        return None


def _param(text, key):
    """Extrai 'KEY: valor' da seção de parâmetros."""
    m = re.search(rf'^{re.escape(key)}:\s*(.+)$', text, re.M)
    return m.group(1).strip() if m else None


def parse_file(path: Path) -> dict:
    """Lê um relatório .txt e devolve um dict com params + métricas."""
    text = path.read_text(encoding="utf-8", errors="replace")
    row = {"arquivo": path.name, "caminho": str(path.relative_to(PROJECT_ROOT))}

    # --- Identificação / parâmetros ---
    row["metodo"] = _param(text, "Teste") or ""
    row["dataset"] = _param(text, "Dataset") or ""
    row["experiment_tag"] = _param(text, "EXPERIMENT_TAG") or ""
    row["seed"] = _param(text, "SEED") or ""
    row["client_setup"] = _param(text, "CLIENT_SETUP") or ""
    row["num_rounds"] = _param(text, "NUM_ROUNDS") or ""
    row["comm_delay"] = _param(text, "COMMUNICATION_DELAY") or ""
    row["alpha"] = _param(text, "ALPHA") or ""
    row["partitioning"] = _param(text, "PARTITIONING") or ""
    row["experiment_mode"] = _param(text, "EXPERIMENT_MODE") or ""          # só FedHAD
    row["heuristics_signature"] = _param(text, "HEURISTICS_SIGNATURE") or ""  # só FedHAD
    # Parâmetros de calibração do FedHAD (variados no test6). Vazios nos baselines.
    row["epochs_decay"] = _param(text, "EPOCHS_DECAY_FACTOR") or ""
    row["lr_decay"] = _param(text, "LR_DECAY_FACTOR") or ""
    row["base_lr"] = _param(text, "BASE_LR") or ""
    row["min_lr"] = _param(text, "MIN_LR") or ""
    row["fedprox_mu"] = _param(text, "FEDPROX_MU") or ""

    # alpha_key: chave de configuração para o particionamento (alpha numérico ou "natural").
    if "NATURAL" in row["partitioning"].upper() or row["dataset"].upper() == "FEMNIST":
        row["alpha_key"] = "natural"
    else:
        # ALPHA pode vir "0.5" ou "N/A (...)"; usa o valor numérico quando existir.
        row["alpha_key"] = row["alpha"] if re.match(r'^[\d.]+$', row["alpha"]) else row["alpha"]

    # --- Tempo ---
    m = re.search(r'^Tempo total \(s\):\s*([\d.]+)', text, re.M)
    row["tempo_s"] = _f(m.group(1)) if m else None

    # --- Acurácia centralizada (final e melhor) via metrics_centralized ---
    row["acuracia_final"] = None
    row["acuracia_melhor"] = None
    m = re.search(r'--- History \(metrics, centralized\) ---\s*(\{.*?\})', text, re.S)
    if m:
        try:
            d = ast.literal_eval(m.group(1))
            acc = d.get("accuracy") or []
            if acc:
                row["acuracia_final"] = float(acc[-1][1])
                row["acuracia_melhor"] = max(float(v) for _, v in acc)
        except Exception:
            pass

    # --- FLOPs totais (normalizados para TFLOPs) ---
    row["total_tflops"] = None
    m = re.search(r'Total acumulado da execução:\s*([\d.]+)\s*(TFLOPs|GFLOPs|MFLOPs|KFLOPs|FLOPs)', text)
    if m:
        val = _f(m.group(1))
        if val is not None:
            row["total_tflops"] = val * _FLOP_UNIT.get(m.group(2), 1.0)

    # --- Energia / emissões ---
    m = re.search(r'Energia total consumida:\s*([\d.eE+\-]+)\s*kWh', text)
    row["energia_kwh"] = _f(m.group(1)) if m else None
    m = re.search(r'Energia total AJUSTADA[^:]*:\s*([\d.eE+\-]+)\s*kWh', text)
    row["energia_ajustada_kwh"] = _f(m.group(1)) if m else None
    m = re.search(r'Emissões de CO2 equivalente:\s*([\d.eE+\-]+)\s*kg', text)
    row["emissoes_kg"] = _f(m.group(1)) if m else None

    # --- Épocas médias e FLOPs por época ---
    # FedHAD (épocas DINÂMICAS): lê a seção "History (metrics, distributed, fit)" com as
    # listas por rodada (avg_epochs_weighted / avg_flops_per_client_round).
    row["epocas"] = None
    row["tflops_por_epoca"] = None
    m = re.search(r'--- History \(metrics, distributed, fit\) ---\s*(\{.*?\})', text, re.S)
    if m:
        try:
            d = ast.literal_eval(m.group(1))
            epocas = [v for _, v in d.get("avg_epochs_weighted", [])]
            flops_cr = [v for _, v in d.get("avg_flops_per_client_round", [])]
            if epocas:
                row["epocas"] = statistics.mean(epocas)
            if epocas and flops_cr:
                media_ep = statistics.mean(epocas)
                if media_ep:
                    # FLOPs por época (por cliente-rodada), normalizado para TFLOPs.
                    row["tflops_por_epoca"] = (statistics.mean(flops_cr) / media_ep) * 1e-12
        except Exception:
            pass

    # Baselines de época FIXA (FedAVG/FedAvgM/FedProx): FedAvgM/FedProx escrevem
    # "LOCAL_EPOCHS: N"; o FedAVG usa 5 fixas hardcoded (não escreve linha) -> default.
    # A definição de tflops_por_epoca é a MESMA do FedHAD (TFLOPs por época por
    # cliente-rodada): total_tflops / (rodadas x clientes x epocas). Assim a coluna fica
    # preenchida para TODOS os métodos e a comparação passa a ser direta.
    if row["epocas"] is None:
        epocas_fixas = _f(_param(text, "LOCAL_EPOCHS"))
        if epocas_fixas is None and row["metodo"].upper() == "FEDAVG":
            epocas_fixas = 5.0  # FedAVG_Final.py: num_epochs = 5 (fixo no fit)
        if epocas_fixas:
            row["epocas"] = epocas_fixas
            nr = _f(row["num_rounds"])
            nc = _f(row["client_setup"])
            if row["total_tflops"] and nr and nc:
                row["tflops_por_epoca"] = row["total_tflops"] / (nr * nc * epocas_fixas)

    # --- Rounds-to-target (seção "--- Rounds-to-Target ---") ---
    # Linhas "Target 50%: round 3"; "não atingido" simplesmente não casa (fica ausente).
    for mm in re.finditer(r'Target (\d+)%:\s*round (\d+)', text):
        row[f"r2t_{mm.group(1)}"] = int(mm.group(2))

    # --- FLOPs-to-target (seção "--- FLOPs-to-Target ---") ---
    # Linhas "Target 50%: 122.107 TFLOPs (round 3)"; unidade normalizada para TFLOPs.
    for mm in re.finditer(r'Target (\d+)%:\s*([\d.]+)\s*(TFLOPs|GFLOPs|MFLOPs|KFLOPs|FLOPs)', text):
        val = _f(mm.group(2))
        if val is not None:
            row[f"flops2t_{mm.group(1)}"] = val * _FLOP_UNIT.get(mm.group(3), 1.0)

    # --- Custo de comunicação (seção "--- Custo de Comunicação ---") ---
    m = re.search(r'Tamanho do modelo:\s*([\d.]+)\s*MB', text)
    row["modelo_mb"] = _f(m.group(1)) if m else None
    m = re.search(r'Bytes acumulados na execução:\s*([\d.]+)\s*MB', text)
    row["bytes_total_mb"] = _f(m.group(1)) if m else None
    # Bytes-to-target: "Bytes até atingir target 50%: 80.061 MB (round 3)".
    for mm in re.finditer(r'Bytes até atingir target (\d+)%:\s*([\d.]+)\s*MB', text):
        row[f"bytes2t_{mm.group(1)}"] = _f(mm.group(2))

    return row


def _agg(values, drop_zeros=False):
    """(média, desvio-padrão amostral, variância amostral, n) ignorando None.
    Se drop_zeros=True, também ignora valores 0 (FLOPs corrompidos).
    desvio e variância = None se n<2."""
    vals = [v for v in values if v is not None]
    if drop_zeros:
        vals = [v for v in vals if v != 0]
    if not vals:
        return (None, None, None, 0)
    media = statistics.mean(vals)
    if len(vals) >= 2:
        variancia = statistics.variance(vals)
        desvio = statistics.stdev(vals)
    else:
        variancia = None
        desvio = None
    return (media, desvio, variancia, len(vals))


def _gerar_saidas(linhas, tag_suffix):
    """Gera o par de CSVs (bruto + resumo) para um conjunto de relatórios."""
    # Detecta dinamicamente as colunas "até-target" (r2t_/flops2t_/bytes2t_), agrupadas
    # por prefixo e ordenadas pelo alvo (%). Cada prefixo mantém a sua própria ordem.
    def _cols(prefixo):
        return sorted({k for row in linhas for k in row if k.startswith(prefixo)},
                      key=lambda c: int(c.split("_")[1]))
    target_cols = [c for prefixo in TARGET_PREFIXOS for c in _cols(prefixo)]

    # --- CSV bruto (1 linha por execução) ---
    bruta_cols = (["arquivo", "caminho", "metodo", "dataset", "experiment_tag", "seed",
                   "client_setup", "num_rounds", "comm_delay", "alpha", "alpha_key",
                   "partitioning", "experiment_mode", "heuristics_signature",
                   "epochs_decay", "lr_decay", "base_lr", "min_lr", "fedprox_mu"]
                  + METRIC_COLS + target_cols)
    bruta_path = OUTPUT_DIR / f"analise_bruta_{tag_suffix}.csv"
    with bruta_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=bruta_cols, extrasaction="ignore")
        w.writeheader()
        for row in sorted(linhas, key=lambda r: [str(r.get(c, "")) for c in GROUP_COLS]):
            w.writerow(row)

    # --- Agregação por configuração (sobre as seeds) ---
    grupos = defaultdict(list)
    for row in linhas:
        chave = tuple(row.get(c, "") for c in GROUP_COLS)
        grupos[chave].append(row)

    resumo_cols = list(GROUP_COLS) + ["n_seeds"]
    for mcol in METRIC_COLS:
        resumo_cols += [f"{mcol}_media", f"{mcol}_desvio", f"{mcol}_variancia"]
    for tc in target_cols:
        resumo_cols += [f"{tc}_media", f"{tc}_desvio", f"{tc}_variancia", f"{tc}_n_atingiu"]

    resumo_path = OUTPUT_DIR / f"analise_resumo_{tag_suffix}.csv"
    with resumo_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(resumo_cols)
        for chave in sorted(grupos):
            rows = grupos[chave]
            out = list(chave) + [len(rows)]
            for mcol in METRIC_COLS:
                media, desvio, variancia, _ = _agg([r.get(mcol) for r in rows],
                                                   drop_zeros=(mcol in FLOP_COLS))
                out += [_fmt(media), _fmt(desvio), _fmt(variancia)]
            for tc in target_cols:
                media, desvio, variancia, n = _agg([r.get(tc) for r in rows])
                out += [_fmt(media), _fmt(desvio), _fmt(variancia), n]
            w.writerow(out)

    print(f"[{tag_suffix}] {len(linhas)} relatórios em {len(grupos)} configuração(ões)"
          f" -> {bruta_path.name}, {resumo_path.name}")
    # Pequena prévia no console.
    for chave in sorted(grupos):
        rows = grupos[chave]
        media, desvio, variancia, n = _agg([r.get("acuracia_final") for r in rows])
        if media is None:
            continue
        rotulo = " | ".join(f"{c}={v}" for c, v in zip(GROUP_COLS, chave) if v not in ("", None))
        dp = f"±{desvio:.4f}" if desvio is not None else ""
        print(f"    acc={media:.4f}{dp} (n={n})  [{rotulo}]")


def main():
    filtro_tag = sys.argv[1] if len(sys.argv) > 1 else None

    # Varre todas as pastas results/main_campaign_default_alpha_0_5/*-results/ (FedAVG-results, FedHAD-results, ...).
    arquivos = []
    for results_dir in RESULTS_DIR.glob("*-results"):
        arquivos.extend(results_dir.rglob("*.txt"))

    if not arquivos:
        print("Nenhum .txt encontrado em results/main_campaign_default_alpha_0_5/*-results/. Rode os experimentos primeiro.")
        return

    linhas = []
    for path in arquivos:
        try:
            row = parse_file(path)
        except Exception as e:
            print(f"Aviso: falha ao ler {path.name}: {e}")
            continue
        if filtro_tag and row.get("experiment_tag") != filtro_tag:
            continue
        linhas.append(row)

    if not linhas:
        alvo = f" (tag={filtro_tag})" if filtro_tag else ""
        print(f"Nenhum resultado correspondente{alvo}.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Avisa sobre execuções com FLOPs corrompidos (total_tflops == 0): são mantidas
    # no CSV bruto, mas DESCONSIDERADAS no cálculo das médias de FLOPs.
    corrompidos = [r for r in linhas if r.get("total_tflops") == 0]
    if corrompidos:
        print(f"\n[AVISO] {len(corrompidos)} execução(ões) com total_tflops=0 "
              f"(corrompido/crashado) — desconsideradas nas médias de FLOPs:")
        for r in sorted(corrompidos, key=lambda x: x.get("arquivo", "")):
            print(f"        - {r.get('arquivo')}")
        print()

    # Agrupa os relatórios por EXPERIMENT_TAG e gera um par de CSVs por teste.
    # Sem argumento -> consolida TODOS os testes encontrados (um arquivo por tag,
    # sem redundância). Com argumento -> só a tag pedida (já filtrada acima).
    por_tag = defaultdict(list)
    for row in linhas:
        por_tag[row.get("experiment_tag") or "sem_tag"].append(row)

    print(f"Encontrados {len(linhas)} relatórios em {len(por_tag)} teste(s): "
          f"{', '.join(sorted(por_tag))}")
    print(f"Pasta de saída: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/\n")
    for tag in sorted(por_tag):
        _gerar_saidas(por_tag[tag], _slug(tag))


def _fmt(x):
    """Formata número para o CSV (vazio se None)."""
    if x is None:
        return ""
    return f"{x:.6g}"


def _slug(txt):
    """Torna um texto seguro para nome de arquivo (mantém letras/dígitos/_-.)."""
    return re.sub(r'[^\w.\-]+', '_', txt).strip('_') or "todos"


if __name__ == "__main__":
    main()
