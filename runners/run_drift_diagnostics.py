# -*- coding: utf-8 -*-
"""
Orquestrador da campanha de DIAGNÓSTICO DE DRIFT (10 sementes; Seção 6.4 do artigo).

Cuida SOMENTE deste experimento (não é o run_main_campaign.py genérico): para cada
cliente, em cada rodada, mede o escore de heterogeneidade (H_k), os hiperparâmetros
que ele recebeu (E_k, lr_k) e duas medidas de quanto seu update local diverge da
direção de agregação do grupo (update_norm[_per_step], cos_sim) — ver instrumentação
em drift_telemetry.py e nos quatro *_Final.py.

Escopo (grade fixa, 120 execuções):
    CIFAR-10; alpha em {1.0, 0.1, 0.01}; métodos FedAVG/FedAvgM/FedProx/FedHAD;
    seeds 42..51 (10 seeds); 10 rounds; client_setup=5 (5 clientes).
    4 métodos x 3 alphas x 10 seeds = 120 execuções.

Saída:
    results/drift_diagnostics_10seeds/
      _manifest.csv                    <- plano + status de cada execução (retomada)
      _run.log                         <- espelho de tudo que foi ao console
      1.1_diagnostico_drift/
        raw/<run_id>.txt               <- relatório padrão do método (cópia)
        drift_telemetry.csv            <- telemetria consolidada (todas as execuções)
        summary.csv                    <- agregado por método/alpha
        logs/<run_id>.log              <- stdout/stderr do subprocesso
        logs/<run_id>__drift.csv       <- telemetria bruta desta execução (antes de consolidar)

Retomada: cada execução tem um run_id determinístico
(f"{metodo}__{dataset}__a{alpha}__s{seed}"). Ao (re)iniciar, o manifesto existente é
relido e tudo que já está 'done' é pulado. Gravação sempre via arquivo temporário +
os.replace (atômica), para não corromper o manifesto se o processo for interrompido.

CLI:
    --dry-run       imprime o plano completo (120 execuções) sem rodar nada
    --retry-failed  reexecuta SOMENTE as execuções marcadas como 'failed'
    --status        mostra o progresso atual sem rodar nada
"""
import csv
import itertools
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]  # raiz do repositório (este runner fica em runners/)
ALGORITMOS_DIR = REPO_DIR / "algoritmos"

ALGORITMOS = {
    "FedAVG": "FedAVG_Final.py",
    "FedAvgM": "FedAvgM_Final.py",
    "FedProx": "FedProx_Final.py",
    "FedHAD": "FedHAD_Final_2.0.py",
}

# ===========================================================================
# ESCOPO DO EXPERIMENTO 1.1 — grade fixa (ver docstring). Não editar para incluir
# outros objetivos; este orquestrador cuida SÓ do diagnóstico de drift.
# ===========================================================================
EXP_NAME = "1.1_diagnostico_drift"
DATASET = "CIFAR10"
ALPHAS = [1.0, 0.1, 0.01]
METHODS = ["FedAVG", "FedAvgM", "FedProx", "FedHAD"]
SEEDS = list(range(42, 52))  # 42..51 -> 10 seeds
NUM_ROUNDS = 10
CLIENT_SETUP = 5  # client_gpu_fraction já calibrado nos *_Final.py p/ este preset

RESULTS_ROOT = REPO_DIR / "results" / "drift_diagnostics_10seeds"
EXP_DIR = RESULTS_ROOT / EXP_NAME
RAW_DIR = EXP_DIR / "raw"
LOGS_DIR = EXP_DIR / "logs"
DRIFT_TELEMETRY_CSV = EXP_DIR / "drift_telemetry.csv"
SUMMARY_CSV = EXP_DIR / "summary.csv"
MANIFEST_CSV = RESULTS_ROOT / "_manifest.csv"
RUN_LOG = RESULTS_ROOT / "_run.log"

MANIFEST_FIELDS = [
    "run_id", "metodo", "dataset", "alpha", "seed", "num_rounds", "client_setup",
    "status", "timestamp", "output_path", "duration_s", "error",
]

DRIFT_HEADER = [
    "metodo", "dataset", "alpha", "seed", "round", "client_id", "H_k", "E_k", "lr_k",
    "update_norm", "update_norm_per_step", "cos_sim", "n_samples", "minibatch_updates",
]


# ===========================================================================
# Log — tudo que vai ao console também vai a _run.log.
# ===========================================================================
def log(msg: str = "") -> None:
    print(msg)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ===========================================================================
# Plano de execuções (grade fixa) e manifesto (retomada).
# ===========================================================================
def build_plan() -> list:
    plan = []
    for metodo, alpha, seed in itertools.product(METHODS, ALPHAS, SEEDS):
        run_id = f"{metodo}__{DATASET}__a{alpha}__s{seed}"
        plan.append({
            "run_id": run_id,
            "metodo": metodo,
            "dataset": DATASET,
            "alpha": str(alpha),
            "seed": str(seed),
            "num_rounds": str(NUM_ROUNDS),
            "client_setup": str(CLIENT_SETUP),
            "status": "pending",
            "timestamp": "",
            "output_path": "",
            "duration_s": "",
            "error": "",
        })
    return plan


def _read_manifest() -> list:
    if not MANIFEST_CSV.exists():
        return []
    with MANIFEST_CSV.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_manifest(rows: list) -> None:
    """Gravação atômica: escreve em arquivo temporário e faz rename (os.replace)."""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_CSV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
    os.replace(tmp, MANIFEST_CSV)


def load_or_init_manifest() -> list:
    """
    Gera o plano completo (120 execuções) e o funde com o manifesto já existente em
    disco (se houver): execuções já presentes preservam seu status ('done'/'failed'/
    'pending'); novas entradas do plano entram como 'pending'. Sempre regrava o
    manifesto (atômico) para refletir o plano atual.
    """
    plan = build_plan()
    existing_by_id = {r["run_id"]: r for r in _read_manifest()}
    merged = []
    for spec in plan:
        if spec["run_id"] in existing_by_id:
            merged.append(existing_by_id[spec["run_id"]])
        else:
            merged.append(spec)
    save_manifest(merged)
    return merged


def _resumo_spec(spec: dict) -> str:
    return f"{spec['metodo']} {spec['dataset']} a={spec['alpha']} seed={spec['seed']}"


# ===========================================================================
# Consolidação da telemetria de drift — append incremental (não só no final).
# ===========================================================================
def consolidate_drift(run_drift_csv: Path) -> int:
    if not run_drift_csv.exists():
        return 0
    with run_drift_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return 0
    data_rows = rows[1:]
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not DRIFT_TELEMETRY_CSV.exists() or DRIFT_TELEMETRY_CSV.stat().st_size == 0
    with DRIFT_TELEMETRY_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(DRIFT_HEADER)
        writer.writerows(data_rows)
    return len(data_rows)


def rebuild_summary() -> None:
    """Agrega drift_telemetry.csv por método/alpha. Regravado a cada execução concluída."""
    if not DRIFT_TELEMETRY_CSV.exists():
        return
    try:
        import pandas as pd
    except ImportError:
        return  # pandas ausente: summary.csv fica só para o final, se disponível

    try:
        df = pd.read_csv(DRIFT_TELEMETRY_CSV)
    except Exception:
        return
    if df.empty:
        return

    for col in ["H_k", "E_k", "lr_k", "update_norm", "update_norm_per_step", "cos_sim"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def _corr_h_e(g):
        # NaN (não 0.0) quando E_k não varia dentro do grupo: nos baselines (FedAVG/
        # FedAvgM/FedProx) E_k é constante por desenho (não adaptam épocas), então a
        # correlação de Pearson é matematicamente indefinida (variância zero), não
        # "próxima de zero" — reportar 0.0 ali seria menos honesto que reportar NaN.
        # Ver coluna 'corr_Hk_Ek_nota'.
        if g["H_k"].nunique() < 2 or g["E_k"].nunique() < 2:
            return float("nan")
        return g["H_k"].corr(g["E_k"])

    def _corr_h_e_nota(g):
        if g["E_k"].nunique() < 2:
            return "NaN: sem variância de E_k para correlacionar (E_k constante no grupo)"
        if g["H_k"].nunique() < 2:
            return "NaN: sem variância de H_k para correlacionar (H_k constante no grupo)"
        return ""

    grouped = df.groupby(["metodo", "alpha"])
    summary = grouped.agg(
        n_observacoes=("H_k", "size"),
        n_clientes_rodadas=("client_id", "count"),
        H_k_medio=("H_k", "mean"),
        E_k_medio=("E_k", "mean"),
        lr_k_medio=("lr_k", "mean"),
        update_norm_medio=("update_norm", "mean"),
        update_norm_per_step_medio=("update_norm_per_step", "mean"),
        cos_sim_medio=("cos_sim", "mean"),
        cos_sim_min=("cos_sim", "min"),
        cos_sim_max=("cos_sim", "max"),
    ).reset_index()
    corr = grouped.apply(_corr_h_e, include_groups=False).reset_index(name="corr_Hk_Ek")
    nota = grouped.apply(_corr_h_e_nota, include_groups=False).reset_index(name="corr_Hk_Ek_nota")
    summary = summary.merge(corr, on=["metodo", "alpha"], how="left")
    summary = summary.merge(nota, on=["metodo", "alpha"], how="left")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)


# ===========================================================================
# Execução de uma corrida (subprocesso isolado, no padrão FL_* do run_main_campaign.py).
# ===========================================================================
def _build_env(spec: dict, run_drift_csv: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["FL_DATASET"] = spec["dataset"]
    env["FL_ALPHA"] = str(spec["alpha"])
    env["FL_SEED"] = str(spec["seed"])
    env["FL_CLIENT_SETUP"] = str(spec["client_setup"])
    env["FL_NUM_ROUNDS"] = str(spec["num_rounds"])
    env["FL_RUN_TAG"] = EXP_NAME
    env["FL_DRIFT_CSV"] = str(run_drift_csv)
    return env


class RunFailed(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def run_one(spec: dict) -> dict:
    """
    Roda UMA execução em subprocesso isolado. Retorna dict com status/duração/erro.
    Deixa KeyboardInterrupt se propagar (após finalizar o subprocesso de forma limpa)
    para que o chamador trate a interrupção do lote.
    """
    run_id = spec["run_id"]
    metodo = spec["metodo"]
    script = ALGORITMOS[metodo]

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    run_drift_csv = LOGS_DIR / f"{run_id}__drift.csv"
    stdout_log_path = LOGS_DIR / f"{run_id}.log"
    env = _build_env(spec, run_drift_csv)

    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(ALGORITMOS_DIR / script)],
        env=env, cwd=str(REPO_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    output_lines = []
    try:
        with stdout_log_path.open("w", encoding="utf-8") as lf:
            for line in proc.stdout:
                lf.write(line)
                output_lines.append(line)
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    dur = time.time() - t0

    if rc != 0:
        tail = "".join(output_lines[-25:]).strip()
        raise RunFailed(f"rc={rc}. Últimas linhas do log:\n{tail}")

    saved_path = None
    for line in output_lines:
        if "Resultados salvos em:" in line:
            saved_path = line.split("Resultados salvos em:", 1)[1].strip()
            break

    output_path_str = ""
    if saved_path:
        src = Path(saved_path)
        if src.exists():
            dst = RAW_DIR / f"{run_id}{src.suffix}"
            shutil.copy2(src, dst)
            output_path_str = str(dst)

    n_rows = consolidate_drift(run_drift_csv)

    return {
        "duration_s": dur,
        "output_path": output_path_str,
        "drift_rows": n_rows,
    }


# ===========================================================================
# Loop principal.
# ===========================================================================
def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _print_progress(i, total, spec, dur, elapsed, eta):
    pct = int(round(100 * i / total)) if total else 100
    log(
        f"[1.1] {i}/{total} ({pct}%) | {_resumo_spec(spec)} | {_fmt_hms(dur)} "
        f"| decorrido {_fmt_hms(elapsed)} | ETA {_fmt_hms(eta)}"
    )


def main() -> None:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    retry_failed = "--retry-failed" in args
    status_only = "--status" in args

    manifest = load_or_init_manifest()
    total_plan = len(manifest)

    if status_only:
        counts = {"pending": 0, "done": 0, "failed": 0}
        for r in manifest:
            counts[r.get("status", "pending")] = counts.get(r.get("status", "pending"), 0) + 1
        print("=" * 78)
        print(f"STATUS — experimento 1.1 (diagnóstico de drift) — {total_plan} execuções no plano")
        print(f"  done:    {counts.get('done', 0)}")
        print(f"  pending: {counts.get('pending', 0)}")
        print(f"  failed:  {counts.get('failed', 0)}")
        if counts.get("failed", 0):
            print("\nExecuções com falha:")
            for r in manifest:
                if r.get("status") == "failed":
                    print(f"  - {r['run_id']}: {r.get('error', '')[:120]}")
        print("=" * 78)
        return

    if retry_failed:
        pendentes = [r for r in manifest if r.get("status") == "failed"]
    else:
        pendentes = [r for r in manifest if r.get("status") != "done"]

    print("=" * 78)
    print(f"EXPERIMENTO 1.1 — diagnóstico de drift — {total_plan} execução(ões) no plano total")
    print(f"  {'--retry-failed: reexecutando apenas as falhas' if retry_failed else 'retomada ativa'} "
          f"— {len(pendentes)} pendente(s) nesta chamada.")
    print(f"Interpretador: {sys.executable}")
    print("=" * 78)

    if dry_run:
        for i, spec in enumerate(pendentes, 1):
            print(f"  [{i}/{len(pendentes)}] {spec['run_id']} | {_resumo_spec(spec)}")
        print("=" * 78)
        print(f"[--dry-run] {len(pendentes)} execução(ões) seriam executadas. "
              f"Remova --dry-run para rodar.")
        return

    if not pendentes:
        print("Nada a fazer: todas as execuções planejadas já foram concluídas.")
        return

    manifest_by_id = {r["run_id"]: r for r in manifest}
    n_pend = len(pendentes)
    inicio_total = time.time()
    duracoes = []
    interrompido = False

    try:
        for i, spec in enumerate(pendentes, 1):
            run_id = spec["run_id"]
            log(f"\n>>> INICIANDO  [{i}/{n_pend}] {run_id}   "
                f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            try:
                result = run_one(spec)
            except RunFailed as e:
                spec["status"] = "failed"
                spec["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                spec["error"] = str(e).replace("\n", " | ")[:2000]
                manifest_by_id[run_id] = spec
                save_manifest(list(manifest_by_id.values()))
                log(f"<<< FALHOU  {run_id} — {e}")
                dur = 0.0
            else:
                spec["status"] = "done"
                spec["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                spec["output_path"] = result["output_path"]
                spec["duration_s"] = f"{result['duration_s']:.1f}"
                spec["error"] = ""
                manifest_by_id[run_id] = spec
                save_manifest(list(manifest_by_id.values()))
                rebuild_summary()
                dur = result["duration_s"]
                duracoes.append(dur)

            elapsed = time.time() - inicio_total
            media = sum(duracoes) / len(duracoes) if duracoes else dur
            restantes = n_pend - i
            eta = media * restantes
            _print_progress(i, n_pend, spec, dur, elapsed, eta)
    except KeyboardInterrupt:
        interrompido = True

    # --- Resumo final ---
    log("\n" + "=" * 78)
    manifest_final = list(manifest_by_id.values()) if manifest_by_id else manifest
    counts = {"pending": 0, "done": 0, "failed": 0}
    for r in manifest_final:
        counts[r.get("status", "pending")] = counts.get(r.get("status", "pending"), 0) + 1

    if interrompido:
        restantes_apos = counts.get("pending", 0) + counts.get("failed", 0)
        log(f"[Ctrl+C] Interrompido pelo usuário. {restantes_apos} execução(ões) restante(s) "
            f"(pendentes+falhas). Manifesto salvo em: {MANIFEST_CSV}")
    else:
        log("Lote concluído.")

    log(f"RESUMO — done: {counts.get('done', 0)} | pending: {counts.get('pending', 0)} "
        f"| failed: {counts.get('failed', 0)} | total no plano: {total_plan}")
    if counts.get("failed", 0):
        log("Execuções com falha:")
        for r in manifest_final:
            if r.get("status") == "failed":
                log(f"  - {r['run_id']}: {r.get('error', '')[:200]}")
        log("Rode com --retry-failed para reexecutar só as falhas.")
    log(f"Manifesto: {MANIFEST_CSV}")
    log(f"Telemetria consolidada: {DRIFT_TELEMETRY_CSV}")
    log(f"Resumo por método/alpha: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
