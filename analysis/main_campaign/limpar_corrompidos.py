# -*- coding: utf-8 -*-
"""
Identifica e remove experimentos CORROMPIDOS/CRASHADOS dos resultados.

Um experimento é considerado corrompido quando o seu relatório .txt indica que o
treino NÃO aconteceu de facto — o sinal principal é o total de FLOPs == 0
("Total acumulado da execução: 0 FLOPs"), que acompanha as execuções que
crasharam no arranque (tipicamente com acurácia presa no "chute aleatório").
Opcionalmente, também trata como corrompido um relatório SEM a secção de FLOPs
(execução interrompida a meio) — ver CRITERIOS abaixo.

Além de apagar o relatório .txt, o script LIMPA a marcação de "concluído" que o
run_main_campaign.py usa na retomada, senão a execução nunca voltaria a correr:
  - run_experiments_progress.jsonl  (1 linha JSON por execução concluída)
  - run_experiments_log_*.csv       (linhas com returncode==0)

Segurança:
  - DRY-RUN por defeito: só mostra o que faria. Use --apply para executar.
  - Faz BACKUP de tudo o que remove/edita numa pasta 'backups_corrompidos/<timestamp>/'.

Uso:
    python limpar_corrompidos.py                      # analisa TODOS os testes (dry-run)
    python limpar_corrompidos.py test3_comunicacao    # filtra por EXPERIMENT_TAG (dry-run)
    python limpar_corrompidos.py --apply              # aplica (todos os testes)
    python limpar_corrompidos.py test3_comunicacao --apply
    python limpar_corrompidos.py --incluir-incompletos   # também remove relatórios sem FLOPs

Só usa a biblioteca padrão (sem pandas).
"""
import re
import csv
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Raiz do projeto (este script fica em analysis/main_campaign/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "main_campaign_default_alpha_0_5"
LOGS_DIR = PROJECT_ROOT / "results" / "execution_logs"
PROGRESSO = LOGS_DIR / "run_experiments_progress.jsonl"
CSV_GLOB = "run_experiments_log_*.csv"

# --- CRITÉRIOS de corrupção -------------------------------------------------------------
# FLOP == 0 é sempre corrompido. "Incompleto" (sem secção de FLOPs) só conta se pedido
# via --incluir-incompletos (por defeito NÃO, para não apagar formatos de relatório
# antigos/legítimos que porventura não tenham essa secção).


# ========================================================================================
# 1) PARSING dos relatórios .txt
# ========================================================================================
def _param(text, key):
    m = re.search(rf'^{re.escape(key)}:\s*(.+)$', text, re.M)
    return m.group(1).strip() if m else None


def _ds(v):
    """Normaliza o nome do dataset p/ comparação: FASHION_MNIST == FashionMNIST."""
    return "".join(str(v or "").split("_")).upper()


def _norm_delay(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def identidade(tag, algo, seed, dataset, client_setup, num_rounds, alpha, comm_delay):
    """Assinatura canónica (normalizada) de uma execução, comum às 3 fontes
    (.txt, .jsonl, .csv). É por ela que se casa o que remover."""
    return (
        str(tag or ""),
        str(algo or ""),
        _norm_int(seed),
        _ds(dataset),
        _norm_int(client_setup),
        _norm_int(num_rounds),
        str(alpha if alpha not in (None, "") else ""),
        _norm_delay(comm_delay),
    )


def analisar_relatorio(path: Path):
    """Lê um .txt e devolve (identidade, info) ou None se não for um relatório válido.
    info: dict com 'motivo' (None se OK), 'flops', 'acuracia', 'caminho'."""
    text = path.read_text(encoding="utf-8", errors="replace")
    tag = _param(text, "EXPERIMENT_TAG")
    algo = _param(text, "Teste")
    seed = _param(text, "SEED")
    dataset = _param(text, "Dataset")
    if not (algo and seed and dataset):
        return None  # não parece um relatório dos *_Final.py

    ident = identidade(
        tag, algo, seed, dataset,
        _param(text, "CLIENT_SETUP"), _param(text, "NUM_ROUNDS"),
        _param(text, "ALPHA"), _param(text, "COMMUNICATION_DELAY"),
    )

    # --- FLOPs totais ---
    m = re.search(r'Total acumulado da execução:\s*([\d.]+)\s*(\w*FLOPs)', text)
    flops = float(m.group(1)) if m else None

    # --- Acurácia final (contexto no relatório) ---
    acc = None
    ma = re.search(r'--- History \(metrics, centralized\) ---\s*(\{.*?\})', text, re.S)
    if ma:
        try:
            import ast
            d = ast.literal_eval(ma.group(1))
            lst = d.get("accuracy") or []
            if lst:
                acc = float(lst[-1][1])
        except Exception:
            pass

    motivo = None
    if flops is not None and flops == 0:
        motivo = "FLOPs=0 (crash no arranque)"
    elif m is None:
        motivo = "sem secção de FLOPs (incompleto)"

    return ident, {"motivo": motivo, "flops": flops, "acuracia": acc, "caminho": path}


# ========================================================================================
# 2) IDENTIFICAÇÃO dos corrompidos
# ========================================================================================
def identificar(filtro_tag=None, incluir_incompletos=False):
    """Varre *-results/ e devolve dois dicts identidade->info:
    corrompidos (a remover) e todos (para diagnóstico)."""
    corrompidos, arquivos = {}, []
    for results_dir in RESULTS_DIR.glob("*-results"):
        arquivos.extend(results_dir.rglob("*.txt"))

    for path in arquivos:
        try:
            res = analisar_relatorio(path)
        except Exception as e:
            print(f"Aviso: falha ao ler {path.name}: {e}")
            continue
        if res is None:
            continue
        ident, info = res
        tag = ident[0]
        if filtro_tag and tag != filtro_tag:
            continue
        motivo = info["motivo"]
        if motivo == "sem secção de FLOPs (incompleto)" and not incluir_incompletos:
            continue
        if motivo:
            corrompidos[ident] = info
    return corrompidos


# ========================================================================================
# 3) REMOÇÃO (com backup)
# ========================================================================================
def _jsonl_identidade(linha):
    """Extrai a identidade de uma linha do progress .jsonl (ou None)."""
    try:
        reg = json.loads(linha)
        payload = json.loads(reg["key"])
        p = payload.get("params", {})
        return identidade(
            p.get("_name"), payload.get("algo"), payload.get("seed"),
            p.get("dataset"), p.get("client_setup"), p.get("num_rounds"),
            p.get("alpha"), p.get("comm_delay"),
        )
    except Exception:
        return None


def _csv_identidade(row):
    return identidade(
        row.get("experimento"), row.get("algoritmo"), row.get("seed"),
        row.get("dataset"), row.get("client_setup"), row.get("num_rounds"),
        row.get("alpha"), row.get("comm_delay"),
    )


def remover(corrompidos, apply, backup_dir):
    alvos = set(corrompidos)

    # --- 3.1 relatórios .txt ---
    print("\n--- Relatórios .txt ---")
    n_txt = 0
    for ident, info in sorted(corrompidos.items(), key=lambda kv: str(kv[0])):
        p = info["caminho"]
        rel = p.relative_to(PROJECT_ROOT)
        acc = f"{info['acuracia']:.4f}" if info["acuracia"] is not None else "?"
        print(f"  [{info['motivo']}] acc={acc}  {rel}")
        if apply and p.exists():
            dest = backup_dir / "txt" / str(rel).replace("\\", "__").replace("/", "__")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            p.unlink()
        n_txt += 1

    # --- 3.2 run_experiments_progress.jsonl ---
    print("\n--- run_experiments_progress.jsonl ---")
    n_jsonl = 0
    if PROGRESSO.exists():
        kept = []
        for linha in PROGRESSO.read_text(encoding="utf-8").splitlines():
            if linha.strip() and _jsonl_identidade(linha) in alvos:
                n_jsonl += 1
            else:
                kept.append(linha)
        print(f"  linhas a remover: {n_jsonl} | a manter: {len(kept)}")
        if apply and n_jsonl:
            shutil.copy2(PROGRESSO, backup_dir / (PROGRESSO.name + ".bak"))
            PROGRESSO.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    else:
        print("  (não existe)")

    # --- 3.3 run_experiments_log_*.csv ---
    print("\n--- run_experiments_log_*.csv ---")
    n_csv = 0
    for csvp in sorted(LOGS_DIR.glob(CSV_GLOB)):
        with open(csvp, encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            fields = rd.fieldnames
            rows = list(rd)
        keep = [r for r in rows if _csv_identidade(r) not in alvos]
        rm = len(rows) - len(keep)
        if rm:
            print(f"  {csvp.name}: remover {rm}")
            n_csv += rm
            if apply:
                shutil.copy2(csvp, backup_dir / (csvp.name + ".bak"))
                with open(csvp, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(keep)
    if n_csv == 0:
        print("  (nenhuma linha correspondente)")

    return n_txt, n_jsonl, n_csv


# ========================================================================================
# 4) main
# ========================================================================================
def main():
    args = sys.argv[1:]
    apply = "--apply" in args
    incluir_incompletos = "--incluir-incompletos" in args
    tags = [a for a in args if not a.startswith("--")]
    filtro_tag = tags[0] if tags else None

    print("=" * 78)
    print("LIMPEZA DE EXPERIMENTOS CORROMPIDOS/CRASHADOS")
    print(f"  Filtro de teste : {filtro_tag or '(todos)'}")
    print(f"  Incompletos     : {'incluídos' if incluir_incompletos else 'ignorados (só FLOPs=0)'}")
    print(f"  Modo            : {'APLICAR (remove de vez)' if apply else 'DRY-RUN (não altera nada)'}")
    print("=" * 78)

    corrompidos = identificar(filtro_tag, incluir_incompletos)
    if not corrompidos:
        print("\nNenhum experimento corrompido encontrado. Nada a fazer.")
        return

    print(f"\nCorrompidos encontrados: {len(corrompidos)}")

    backup_dir = PROJECT_ROOT / "backups_corrompidos" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if apply:
        backup_dir.mkdir(parents=True, exist_ok=True)

    n_txt, n_jsonl, n_csv = remover(corrompidos, apply, backup_dir)

    print("\n" + "-" * 78)
    print(f"Resumo: {n_txt} relatório(s) .txt | {n_jsonl} entrada(s) no progresso | "
          f"{n_csv} linha(s) nos CSV")
    if apply:
        print(f"APLICADO. Backup em: {backup_dir.relative_to(PROJECT_ROOT)}/")
        print("Agora pode relançar:  python runners/run_main_campaign.py   (re-executa só os removidos)")
        print("E depois:             python analisar_resultados.py")
    else:
        print("DRY-RUN: nada foi alterado. Para remover de vez, repita com  --apply")
    print("-" * 78)


if __name__ == "__main__":
    main()
