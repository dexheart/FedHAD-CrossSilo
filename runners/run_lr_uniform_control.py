#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bateria de controle "LR uniforme" do FedHAD (substrato da Seção 6.10).

Pergunta: atribuir o LR segundo H_k produz resultado diferente de dar a todos os
clientes um único LR uniforme com a mesma agressividade agregada de primeira ordem?

USO (igual ao run_main_campaign.py):

    python runners/run_lr_uniform_control.py --dry-run     # plano completo, sem treinar
    python runners/run_lr_uniform_control.py --smoke       # 1 semente, 2 rodadas (valida o pipeline)
    python runners/run_lr_uniform_control.py --level 1     # só o nível 1 (Full, U2, LR-only, U1)
    python runners/run_lr_uniform_control.py               # nível 1 e depois nível 2 (U2-arith, INV)

Antes da primeira execução oficial grava results/lr_uniform_control/preregistration.json
e o hash dele; depois disso recusa retomar se o pré-registro, as margens
(algoritmos/lr_uniform_control/margins.json), a fonte das referências ou o código mudarem.

RETOMADA: rode o mesmo comando de novo. Execuções completas (todas as rodadas, todos
os clientes com dados em todas as rodadas, checksum da inicialização, campos
obrigatórios, mesmo código) são puladas; parciais/inválidas são movidas para
_discarded/ (nunca apagadas) e refeitas.

Filtros: --arms, --alphas, --seeds. Outras opções: --timeout, --retries, --gpu,
--skip-reference-reruns (usa os resultados antigos da Seção 6.10 para Full e LR-only;
desligada por padrão), --allow-concurrent.
"""
from __future__ import annotations

# =============================================================================== #
# CONFIGURAÇÃO — edite aqui
# =============================================================================== #
TIMEOUT_POR_EXECUCAO_S = 3600     # uma execução que passar disso é encerrada e conta como falha
NOVAS_TENTATIVAS = 2              # tentativas extras por execução antes de desistir dela
GPU = None                        # ex.: "0" para CUDA_VISIBLE_DEVICES=0; None = não mexe
MOSTRAR_SAIDA_TREINO = True       # False = só progresso no terminal (saída completa vai p/ o log)
# =============================================================================== #

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repository root (this runner lives in runners/)
_VENV = REPO / "env_flwr_pt"
if (__name__ == "__main__" and (_VENV / "bin" / "python").exists()
        and Path(sys.prefix).resolve() != _VENV.resolve() and not os.environ.get("LRU_NO_REEXEC")):
    os.environ["LRU_NO_REEXEC"] = "1"
    os.execv(str(_VENV / "bin" / "python"), [str(_VENV / "bin" / "python"), str(Path(__file__).resolve()), *sys.argv[1:]])

sys.path.insert(0, str(REPO / "algoritmos" / "lr_uniform_control"))
import config_lru as C          # noqa: E402
import lru_checks as K          # noqa: E402
import lru_prereg as PR         # noqa: E402

ABL_OFFICIAL = REPO / "results" / "component_analysis_ablation" / "raw" / "official"


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_fingerprint() -> str:
    files = [C.ENTRY_SCRIPT, C.CODE / "lru_policy.py",
             REPO / "algoritmos" / "tuned_controls_and_step_permutation" / "r2_policy.py",
             REPO / "algoritmos" / "tuned_controls_and_step_permutation" / "drift_telemetry.py",
             REPO / "algoritmos" / "component_analysis_ablation" / "ablation_policy.py"]
    payload = {f.name: sha_file(f) for f in files}
    payload["substrate"] = [C.DATASET, C.CLIENT_SETUP, C.NUM_ROUNDS, C.COMM_DELAY, C.FEDPROX_MU,
                            C.BASE_EPOCHS, C.MIN_EPOCHS, C.EPOCHS_DECAY, C.BASE_LR, C.MIN_LR, C.LR_DECAY]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def python_executable() -> str:
    venv = _VENV / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def checkpoint_for(seed: int):
    man = read_json(C.CKPT_DIR / "checkpoints_manifest.json") or {}
    rec = man.get(str(seed))
    p = C.CKPT_DIR / C.DATASET / f"seed-{seed}.pt"
    if rec is None or not p.exists():
        return None, None
    return p, rec["sha256"]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def fmt_dur(sec: float) -> str:
    sec = int(max(sec, 0)); h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}min" if h else f"{m}min{sec % 60:02d}s"


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
def cell_id(arm, alpha, seed):
    return f"{arm}__alpha-{alpha:g}__seed-{seed}"


def plan_cells(args):
    arms_l1 = [a for a in C.LEVEL1_ARMS if a in args.arms]
    arms_l2 = [a for a in C.LEVEL2_ARMS if a in args.arms]
    if args.skip_reference_reruns:
        arms_l1 = [a for a in arms_l1 if a not in C.REFERENCE_ARMS]
    levels = {"1": [(1, arms_l1)], "2": [(2, arms_l2)], "all": [(1, arms_l1), (2, arms_l2)]}[args.level]
    cells = []
    for lvl, arms in levels:
        # order: all seeds and both alphas of a level before the next level; within a
        # (alpha, seed) the arms run back to back so partial results stay paired
        for a in args.alphas:
            for s in args.seeds:
                for arm in arms:
                    cells.append({"arm": arm, "alpha": a, "seed": s, "level": lvl,
                                  "num_rounds": args.rounds, "id": cell_id(arm, a, s)})
    return cells


# --------------------------------------------------------------------------- #
# Pre-registration and state
# --------------------------------------------------------------------------- #
class Campaign:
    def __init__(self, out: Path, rounds: int, official: bool, args):
        self.out, self.rounds, self.official, self.args = out, rounds, official, args
        self.runs = out / "runs"
        self.manifest_path = out / "manifest.json"
        self.failures = out / "failures.jsonl"
        self.discarded = out / "_discarded"
        self.prereg_path = out / "preregistration.json"
        self.prereg_hash_path = out / "preregistration.sha256"
        self.code_fp = code_fingerprint()
        self.reference_source = ("section_6_10_results" if args.skip_reference_reruns
                                 else "re-executed_in_this_battery")

    def ensure_prereg(self):
        margins = C.load_margins()
        doc = PR.build_prereg(C=C, margins=margins, reference_source=self.reference_source,
                              code_fingerprint=self.code_fp)
        text = PR.canonical_json(doc)
        if not self.prereg_path.exists():
            self.out.mkdir(parents=True, exist_ok=True)
            self.prereg_path.write_text(text, encoding="utf-8")
            (self.prereg_hash_path).write_text(PR.sha256_text(text) + "\n", encoding="utf-8")
            print(f">>> Pré-registro gravado: {self.prereg_path}  sha256={PR.sha256_text(text)[:16]}…")
            return
        stored_hash = self.prereg_hash_path.read_text().strip() if self.prereg_hash_path.exists() else ""
        on_disk = self.prereg_path.read_text(encoding="utf-8")
        if PR.sha256_text(on_disk) != stored_hash:
            raise SystemExit("ERRO: preregistration.json foi alterado depois do início da bateria "
                             "(hash não confere). Recusando retomar.")
        if PR.sha256_text(text) != stored_hash:
            raise SystemExit("ERRO: a configuração atual (margens, fonte das referências ou código) "
                             "difere do pré-registro gravado no início. Recusando retomar.\n"
                             f"  pré-registro: {self.prereg_path}")

    def manifest(self):
        return read_json(self.manifest_path) or {}

    def set_state(self, cid, **kw):
        m = self.manifest()
        rec = m.get(cid, {})
        rec.update(kw); rec["updated_at"] = datetime.now().isoformat(timespec="seconds")
        m[cid] = rec
        write_json_atomic(self.manifest_path, m)

    def run_dir(self, cell):
        return self.runs / cell["id"]

    def is_complete(self, cell):
        res = read_json(self.run_dir(cell) / "result.json")
        _, sha = checkpoint_for(cell["seed"])
        return K.validate_result(res, arm=cell["arm"], alpha=cell["alpha"], seed=cell["seed"],
                                 num_rounds=self.rounds, expected_init_sha=sha,
                                 code_fingerprint=self.code_fp)

    def discard(self, cell, why):
        d = self.run_dir(cell)
        if d.exists() and any(d.iterdir()):
            dst = self.discarded / f"{cell['id']}__{int(time.time())}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(dst))
            print(f"    (execução anterior inválida movida para {dst.relative_to(REPO)}: {why})")


# --------------------------------------------------------------------------- #
# Execution of one cell
# --------------------------------------------------------------------------- #
def cell_env(cell, raw_dir: Path, ckpt: Path, sha: str, args) -> dict:
    env = os.environ.copy()
    cfg = {"experiment": "lr_uniform_control", "arm": cell["arm"], "method": "FedHAD-LRU",
           "level": cell["level"], "cell_id": cell["id"], "runner": Path(__file__).name}
    env.update({
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "FL_DATASET": C.DATASET, "FL_ALPHA": repr(cell["alpha"]), "FL_SEED": str(cell["seed"]),
        "FL_NUM_ROUNDS": str(cell["num_rounds"]), "FL_CLIENT_SETUP": str(C.CLIENT_SETUP),
        "FL_COMM_DELAY": str(C.COMM_DELAY), "FL_USE_ENERGY": "1" if C.USE_ENERGY else "0",
        "FL_USE_FEDPROX": "1", "FL_FEDPROX_MU": str(C.FEDPROX_MU),
        "FL_BASE_EPOCHS": str(C.BASE_EPOCHS), "FL_MIN_EPOCHS": str(C.MIN_EPOCHS),
        "FL_EPOCHS_DECAY": str(C.EPOCHS_DECAY), "FL_BASE_LR": str(C.BASE_LR),
        "FL_MIN_LR": str(C.MIN_LR), "FL_LR_DECAY": str(C.LR_DECAY),
        "FL_LRU_ARM": cell["arm"], "FL_R2_INIT_CKPT": str(ckpt), "FL_R2_INIT_SHA256": sha,
        "FL_R2_RUN_DIR": str(raw_dir), "FL_RESULTS_DIR": str(raw_dir / "report"),
        "FL_R2_CONFIG_JSON": json.dumps(cfg), "FL_RUN_TAG": "lru",
        # Ray workers do not inherit the driver's sys.path edits; without this they
        # fail to unpickle client_fn with "No module named 'r2_policy'"
        "PYTHONPATH": os.pathsep.join(filter(None, [
            str(REPO / "algoritmos" / "tuned_controls_and_step_permutation"), str(C.CODE),
            str(REPO / "algoritmos" / "component_analysis_ablation"),
            env.get("PYTHONPATH", "")])),
    })
    env.pop("FL_LRU_PLAN_ONLY", None)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return env


class Interrupted(Exception):
    pass


def run_subprocess(cmd, env, log_path: Path, timeout_s: int, show: bool):
    """Streams output to the log (and terminal); kills the whole process group on
    timeout or Ctrl+C. Returns (returncode, timed_out)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = {"v": False}
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} =====\n"); fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1, start_new_session=True)

        def kill_group(sig):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass

        def on_timeout():
            timed_out["v"] = True
            kill_group(signal.SIGINT); time.sleep(20); kill_group(signal.SIGKILL)

        timer = threading.Timer(timeout_s, on_timeout); timer.daemon = True; timer.start()
        try:
            for line in proc.stdout:
                fh.write(line)
                if show:
                    sys.stdout.write(line); sys.stdout.flush()
            rc = proc.wait()
        except KeyboardInterrupt:
            kill_group(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                kill_group(signal.SIGKILL); proc.wait()
            fh.write("\n[runner] interrupted by the user (Ctrl+C)\n")
            raise Interrupted()
        finally:
            timer.cancel()
    return rc, timed_out["v"]


def read_telemetry(path: Path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def execute_cell(camp: Campaign, cell, args):
    ckpt, sha = checkpoint_for(cell["seed"])
    if ckpt is None:
        return False, f"missing initial checkpoint for seed {cell['seed']} in {C.CKPT_DIR}"
    rd = camp.run_dir(cell)
    attempt_dir = rd / "attempt"
    if attempt_dir.exists():          # leftovers of an interrupted/failed attempt: keep, never delete
        camp.discarded.mkdir(parents=True, exist_ok=True)
        shutil.move(str(attempt_dir), str(camp.discarded / f"{cell['id']}__attempt__{int(time.time())}"))
    attempt_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    rc, to = run_subprocess([python_executable(), "-u", str(C.ENTRY_SCRIPT)],
                            cell_env(cell, attempt_dir, ckpt, sha, args),
                            rd / "run.log", args.timeout, args.show_output)
    if to:
        return False, f"timeout after {args.timeout} s"
    if rc != 0:
        return False, f"exit status {rc}"
    fp = read_json(attempt_dir / "fingerprint.json")
    tel_path = attempt_dir / "telemetry.csv"
    if fp is None or not tel_path.exists():
        return False, "raw artifacts missing"
    res = K.assemble_result(
        fingerprint=fp, telemetry_rows=read_telemetry(tel_path), cell=cell,
        code_fingerprint=camp.code_fp,
        platform_info={"cpu": cpu_model(), "hostname": socket.gethostname(),
                       "platform": platform.platform(), "numpy": __import__("numpy").__version__},
        started_at=started, finished_at=datetime.now().isoformat(timespec="seconds"))
    ok, why = K.validate_result(res, arm=cell["arm"], alpha=cell["alpha"], seed=cell["seed"],
                                num_rounds=camp.rounds, expected_init_sha=sha,
                                code_fingerprint=camp.code_fp)
    if not ok:
        return False, f"result invalid: {why}"
    write_json_atomic(rd / "result.json", res)       # the record the analysis reads
    return True, ""


# --------------------------------------------------------------------------- #
# Dry run: every arm's plan per (seed, alpha), without training
# --------------------------------------------------------------------------- #
def dry_run(args, cells):
    from collections import Counter
    n = len(cells)
    print("=" * 100)
    print(f"LR-UNIFORM CONTROL — plano: {n} execuções "
          f"(~{fmt_dur(n * C.EST_SECONDS_PER_RUN)} a {C.EST_SECONDS_PER_RUN} s/execução)")
    for (lvl, arm), k in sorted(Counter((c["level"], c["arm"]) for c in cells).items()):
        print(f"  nível {lvl}  {arm:16s} {k:4d} execuções")
    print(f"  margens (pp): {C.load_margins()}  | fonte das referências: "
          f"{'resultados antigos da Seção 6.10' if args.skip_reference_reruns else 'reexecutadas nesta bateria'}")
    print("=" * 100)
    pairs = sorted({(c["alpha"], c["seed"]) for c in cells})
    tmp = Path(tempfile.mkdtemp(prefix="lru_plan_"))
    for a, s in pairs:
        out = tmp / f"plan_{a:g}_{s}.json"
        env = os.environ.copy()
        env.update({"FL_DATASET": C.DATASET, "FL_ALPHA": repr(a), "FL_SEED": str(s),
                    "FL_NUM_ROUNDS": str(C.NUM_ROUNDS), "FL_CLIENT_SETUP": str(C.CLIENT_SETUP),
                    "FL_BASE_EPOCHS": str(C.BASE_EPOCHS), "FL_MIN_EPOCHS": str(C.MIN_EPOCHS),
                    "FL_EPOCHS_DECAY": str(C.EPOCHS_DECAY), "FL_BASE_LR": str(C.BASE_LR),
                    "FL_MIN_LR": str(C.MIN_LR), "FL_LR_DECAY": str(C.LR_DECAY),
                    "FL_USE_ENERGY": "0", "FL_LRU_PLAN_ONLY": str(out),
                    "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0",
                    "FL_RESULTS_DIR": str(tmp / "report"),
                    # the frozen fedhad_lru.py imports r2_policy / drift_telemetry / ablation_policy
                    "PYTHONPATH": os.pathsep.join([
                        str(REPO / "algoritmos" / "tuned_controls_and_step_permutation"), str(C.CODE),
                        str(REPO / "algoritmos" / "component_analysis_ablation")])})
        r = subprocess.run([python_executable(), "-u", str(C.ENTRY_SCRIPT)], cwd=str(REPO), env=env,
                           capture_output=True, text=True)
        plan = read_json(out)
        if r.returncode != 0 or plan is None:
            print(f"\n[alpha={a:g} seed={s}] FALHOU ao montar o plano (sem treino):\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
            continue
        print(f"\n[alpha={a:g} | seed={s}]  n_k={plan['n_samples']}  H_k={[round(h, 3) for h in plan['H']]}")
        print(f"   E_full={plan['E_full']}  E_fixed={plan['E_fixed']}  "
              f"eta_k(FedHAD)={[round(x, 5) for x in plan['LR_full']]}")
        for arm, p in plan["arms"].items():
            print(f"   {arm:16s} E={p['epochs']}  lr={[round(x, 5) for x in p['lr']]}  "
                  f"sum_p_eta_tau={p['sum_p_eta_tau']:.4f}  eta_bar[p,tau / tau / arith]="
                  f"{p['eta_bar_primary_p_tau']:.5f} / {p['eta_bar_tau_only']:.5f} / {p['eta_bar_arithmetic']:.5f}")
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nNada foi treinado; nenhum arquivo de resultado foi escrito.")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def other_fl_processes():
    try:
        out = subprocess.run(["pgrep", "-af", "fedhad_r2.py|fedhad_ablation.py|fedhad_lru.py|_Final"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return []
    return [l for l in out.splitlines() if l.strip() and str(os.getpid()) not in l.split()[0]]


def run(args):
    official = not args.smoke
    out = C.OUT if official else C.OUT_SMOKE
    camp = Campaign(out, args.rounds, official, args)
    cells = plan_cells(args)
    if args.dry_run:
        dry_run(args, cells)
        return 0
    others = other_fl_processes()
    if others and not args.allow_concurrent:
        print("ERRO: há outra execução de FL usando a máquina (a GPU seria compartilhada e os\n"
              "tempos e a ordem de execução deixariam de ser comparáveis):")
        for l in others:
            print("   ", l[:160])
        print("Espere terminar ou use --allow-concurrent.")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    lock_fh = (out / ".lock").open("w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"ERRO: outra instância já está rodando nesta bateria ({out / '.lock'}).")
        return 2
    if official:
        camp.ensure_prereg()
    missing = sorted({c["seed"] for c in cells if checkpoint_for(c["seed"])[0] is None})
    if missing:
        print(f"ERRO: faltam checkpoints iniciais para as sementes {missing} em {C.CKPT_DIR}.\n"
              "Crie com: python runners/run_tuned_controls_and_step_permutation.py --make-checkpoints "
              f"--seeds {','.join(map(str, missing))} --val-seeds {','.join(map(str, missing))}")
        return 2

    todo, done0 = [], 0
    for c in cells:
        ok, why = camp.is_complete(c)
        if ok:
            done0 += 1; continue
        if (camp.run_dir(c) / "result.json").exists() or (camp.run_dir(c) / "attempt").exists():
            camp.discard(c, why)
        todo.append(c)
    print("=" * 100)
    print(f"LR-UNIFORM CONTROL ({'oficial' if official else 'SMOKE'}) → {out.relative_to(REPO)}")
    print(f"  total {len(cells)} | já completas {done0} | a executar {len(todo)} | "
          f"rodadas {args.rounds} | timeout {args.timeout}s | novas tentativas {args.retries}")
    print("=" * 100, flush=True)

    ok_n, failed, durs = 0, [], []
    t0 = time.time()
    try:
        for i, c in enumerate(todo, 1):
            head = f"[{i}/{len(todo)}] nível {c['level']} | {c['arm']} | alpha={c['alpha']:g} | seed={c['seed']}"
            success, err = False, ""
            prev = camp.manifest().get(c["id"], {})
            for attempt in range(1, args.retries + 2):
                print(f"\n>>> INICIANDO {head} | tentativa {attempt} ({datetime.now():%Y-%m-%d %H:%M:%S})", flush=True)
                camp.set_state(c["id"], status="running", attempt=int(prev.get("attempt", 0)) + attempt)
                ts = time.time()
                success, err = execute_cell(camp, c, args)
                dur = time.time() - ts
                if success:
                    durs.append(dur)
                    camp.set_state(c["id"], status="completed", duration_s=round(dur, 1), error="")
                    break
                camp.set_state(c["id"], status="failed", duration_s=round(dur, 1), error=err)
                with camp.failures.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                                         "cell": c["id"], "attempt": attempt, "error": err}) + "\n")
                print(f"<<< FALHOU {head} em {dur:.1f}s — {err} (log: {(camp.run_dir(c) / 'run.log').relative_to(REPO)})")
            if success:
                ok_n += 1
                rem = len(todo) - i
                per = sum(durs) / len(durs)
                print(f"<<< OK {head} em {dur:.1f}s | concluídas {done0 + ok_n}/{len(cells)} | "
                      f"faltam {rem} (~{fmt_dur(per * rem)})", flush=True)
            else:
                failed.append((c["id"], err))
    except Interrupted:
        camp.set_state(c["id"], status="interrupted", error="Ctrl+C")
        print("\nInterrompido (Ctrl+C). O manifesto está consistente; rode o mesmo comando para continuar.")
        return 130
    finally:
        print("\n" + "=" * 100)
        print(f"RESUMO: concluídas nesta sessão {ok_n} | falhas persistentes {len(failed)} | tempo {fmt_dur(time.time() - t0)}")
        for cid, e in failed:
            print(f"  FALHOU  {cid}: {e}")
        if failed:
            print(f"  (detalhes em {camp.failures.relative_to(REPO)}; rode o mesmo comando para tentar de novo)")
        print("=" * 100)
    return 0 if not failed else 1


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="plano completo por semente/alpha; não treina")
    ap.add_argument("--smoke", action="store_true", help=f"sementes {C.SMOKE_SEEDS}, {C.SMOKE_ROUNDS} rodadas, em {C.OUT_SMOKE.relative_to(REPO)}")
    ap.add_argument("--level", choices=["1", "2", "all"], default="all")
    ap.add_argument("--arms", default=",".join(C.LEVEL1_ARMS + C.LEVEL2_ARMS))
    ap.add_argument("--alphas", default=",".join(f"{a:g}" for a in C.ALPHAS))
    ap.add_argument("--seeds", default=None, help="ex.: 42-45,50 (padrão: 42-71; no smoke: 42)")
    ap.add_argument("--skip-reference-reruns", action="store_true",
                    help="não reexecuta Full e LR-only; a análise usa os resultados da Seção 6.10")
    ap.add_argument("--timeout", type=int, default=TIMEOUT_POR_EXECUCAO_S)
    ap.add_argument("--retries", type=int, default=NOVAS_TENTATIVAS)
    ap.add_argument("--gpu", default=GPU)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--allow-concurrent", action="store_true")
    a = ap.parse_args(argv)
    a.arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in a.arms if x not in C.LEVEL1_ARMS + C.LEVEL2_ARMS]
    if bad:
        ap.error(f"braços desconhecidos: {bad}")
    a.alphas = [float(x) for x in a.alphas.split(",")]
    if any(x not in C.ALPHAS for x in a.alphas):
        ap.error(f"alpha fora de {C.ALPHAS}")
    if a.seeds is None:
        a.seeds = C.SMOKE_SEEDS if a.smoke else C.SEEDS
    else:
        seeds = []
        for part in a.seeds.split(","):
            if "-" in part:
                lo, hi = part.split("-"); seeds += list(range(int(lo), int(hi) + 1))
            elif part.strip():
                seeds.append(int(part))
        a.seeds = seeds
    a.rounds = C.SMOKE_ROUNDS if a.smoke else C.NUM_ROUNDS
    a.show_output = MOSTRAR_SAIDA_TREINO and not a.quiet
    if a.smoke and a.skip_reference_reruns:
        ap.error("--skip-reference-reruns não se aplica ao smoke")
    return a


if __name__ == "__main__":
    try:
        sys.exit(run(parse_args()))
    except BrokenPipeError:
        sys.exit(0)
