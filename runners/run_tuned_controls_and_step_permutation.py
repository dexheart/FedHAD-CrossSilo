#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordenador de dois experimentos de controle (Seções 5.7 e 6.11 do artigo): a alocação permutada que
preserva passos (step_permutation) e os controles FedProx ajustados em sementes de
validação separadas (tuned_controls).

USO (igual ao run_main_campaign.py):

    python runners/run_tuned_controls_and_step_permutation.py

Sem argumentos, roda o pipeline inteiro, na ordem:
  0) cria os checkpoints iniciais comuns que faltarem (1 por seed, com SHA-256);
  1) step_permutation: full_fedhad + step_permuted_lrclient, 2 alphas x 30 seeds;
  2) tuned_controls / tune: grade LR x E do FedProx nos seeds de VALIDAÇÃO (72-76);
  3) resumo da validação (validation_summary.csv + selection_proposal.json);
  4) tuned_controls / evaluate: config(s) congelada(s) nos 30 seeds de avaliação.
     Só roda se existir results/tuned_controls_fedprox_validation/frozen_config.json (ou se
     CONGELAR_AUTOMATICAMENTE = True). Caso contrário, o pipeline para após o passo 3
     e diz como congelar.

Cada execução mostra ">>> INICIANDO [i/N] ..." e "<<< OK/FALHOU ... tempo | ETA".
A saída do treino aparece no terminal e fica também no log de cada execução.

RETOMADA: se cair (erro, Ctrl+C, queda do servidor), rode O MESMO comando de novo.
Execuções concluídas (artefatos completos + checksum do checkpoint + fingerprint do
código) são puladas; as interrompidas ou que falharam são refeitas.

Flags opcionais (nenhuma é necessária):
  --dry-run / --status     mostra o plano e o estado; não executa nada
  --experiment X           restringe a step_permutation | tuned_controls | all
  --phase tune|evaluate    (com --experiment tuned_controls)
  --summarize-tune         só gera o resumo da validação
  --freeze budget_matched  congela a regra pré-especificada (escreve frozen_config.json)
  --force                  refaz execuções concluídas (renomeia as antigas, nunca apaga)
  --rerun-failed           executa só as que falharam
  --max-runs N             para depois de N execuções
  --self-test              testes unitários (sem treino)
  --print-counts           contagem esperada de execuções

Desenho herdado de run_component_ablation.py: um
subprocesso isolado por execução, ledger de estado JSON atômico, artefatos como fonte
da verdade para a retomada, fingerprint da campanha. O script de treino é
algoritmos/tuned_controls_and_step_permutation/fedhad_r2.py (ver fedhad_r2.diff).
Os nomes fedhad_r2.py e r2_policy.py são os nomes com que o código foi executado e
registrado (SHA-256 nos fingerprints de cada execução); por isso foram mantidos.
"""
from __future__ import annotations

# =============================================================================== #
# CONFIGURAÇÃO — edite aqui (equivalente às listas no topo do run_main_campaign.py)
# =============================================================================== #
EXECUTAR_STEP_PERMUTATION = True    # experimento 1
EXECUTAR_TUNE = True                # experimento 2, fase de ajuste (seeds de validação)
EXECUTAR_EVALUATE = True            # experimento 2, fase de avaliação (precisa de frozen_config.json)

# Após o tune completo, congelar automaticamente a regra pré-especificada
# "budget_matched" (melhor acurácia de VALIDAÇÃO com passos <= 1.02 x FedHAD)?
# False = o pipeline para após o tune e você decide (--freeze budget_matched ou
# frozen_config.json escrito à mão); é o protocolo documentado.
CONGELAR_AUTOMATICAMENTE = False

INCLUIR_LR_FOLLOW = False           # braço opcional step_permuted_lrfollow (+60 execuções)
METODOS_ESTATICOS = ["FedProx"]     # ["FedProx"] ou ["FedAvg", "FedProx"]
ALPHAS_ATIVOS = [0.01, 0.1]         # subconjunto de [0.01, 0.1]
SEEDS_AVALIACAO = "42-71"           # step_permutation e evaluate
SEEDS_VALIDACAO = "72-76"           # tune (não pode sobrepor 42-71)
GPU = None                          # ex.: "0" para CUDA_VISIBLE_DEVICES=0; None = não mexe
MOSTRAR_SAIDA_TREINO = True         # False = só as linhas de progresso (saída vai só p/ o log)
MAX_FALHAS_SEGUIDAS = 3             # aborta se N execuções seguidas falharem (ex.: GPU caiu)
# =============================================================================== #

import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repository root (this runner lives in runners/)

# `python runners/run_tuned_controls_and_step_permutation.py` with any interpreter: re-launch
# itself with the project venv (the one used by the component analysis).
_VENV = REPO / "env_flwr_pt"
if (__name__ == "__main__" and (_VENV / "bin" / "python").exists()
        and Path(sys.prefix).resolve() != _VENV.resolve() and not os.environ.get("CONTROLS_NO_REEXEC")):
    os.environ["CONTROLS_NO_REEXEC"] = "1"
    os.execv(str(_VENV / "bin" / "python"), [str(_VENV / "bin" / "python"), str(Path(__file__).resolve()), *sys.argv[1:]])

sys.path.insert(0, str(REPO / "algoritmos" / "tuned_controls_and_step_permutation"))
import config_tuned_controls as C  # noqa: E402
import r2_policy as P  # noqa: E402  (pure functions; numpy only)

STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED = "pending", "running", "completed", "failed"


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    experiment: str          # step_permutation | tuned_tune | tuned_evaluate
    arm: str                 # full_fedhad | step_permuted_lrclient | step_permuted_lrfollow | static
    method: str              # FedHAD | FedProx | FedAvg
    alpha: float
    seed: int
    lr: float | None = None
    epochs: int | None = None
    label: str = ""
    out_root: Path = field(default=C.OUT_STEP)

    @property
    def cell_id(self) -> str:
        if self.arm == "static":
            arm = f"static_{self.method}_lr{self.lr:g}_E{self.epochs}"
        else:
            arm = self.arm
        return f"{arm}__{C.DATASET}__alpha-{self.alpha:g}__seed-{self.seed}"

    @property
    def run_dir(self) -> Path:
        return self.out_root / "raw" / self.cell_id

    @property
    def state_path(self) -> Path:
        return self.out_root / "_state" / f"{self.cell_id}.json"

    @property
    def log_path(self) -> Path:
        return self.out_root / "logs" / f"{self.cell_id}.log"


def out_root_for(experiment: str, base_step: Path, base_tuned: Path) -> Path:
    return {"step_permutation": base_step, "tuned_tune": base_tuned / "tune",
            "tuned_evaluate": base_tuned / "evaluate"}[experiment]


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def _rel(p) -> str:
    p = Path(p)
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def campaign_fingerprint() -> dict:
    payload = {
        "dataset": C.DATASET, "client_setup": C.CLIENT_SETUP, "num_rounds": C.NUM_ROUNDS,
        "comm_delay": C.COMM_DELAY, "use_energy": C.USE_ENERGY, "fedprox_mu": C.FEDPROX_MU,
        "fedhad": [C.BASE_EPOCHS, C.MIN_EPOCHS, C.EPOCHS_DECAY, C.BASE_LR, C.MIN_LR, C.LR_DECAY],
        "sha256_fedhad_r2": _sha_file(C.ENTRY_SCRIPT),
        "sha256_r2_policy": _sha_file(C.CODE / "r2_policy.py"),
        "sha256_drift_telemetry": _sha_file(C.CODE / "drift_telemetry.py"),
    }
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def git_info() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip() != ""
        return {"commit": head, "dirty": dirty}
    except Exception:
        return {"commit": "", "dirty": None}


def python_executable(explicit: str | None) -> str:
    if explicit:
        return explicit
    venv = REPO / "env_flwr_pt" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def ckpt_path(seed: int, root: Path) -> Path:
    return root / C.DATASET / f"seed-{seed}.pt"


def ckpt_manifest_path(root: Path) -> Path:
    return root / "checkpoints_manifest.json"


def load_ckpt_manifest(root: Path) -> dict:
    p = ckpt_manifest_path(root)
    return json.loads(p.read_text()) if p.exists() else {}


def _net_class_from_source(name: str):
    """Extract a network class from fedhad_r2.py by AST, without executing the
    module (which would start a simulation)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    tree = ast.parse(C.ENTRY_SCRIPT.read_text(encoding="utf-8"))
    ns = {"nn": nn, "F": F, "torch": torch}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            exec(compile(ast.Module([node], []), str(C.ENTRY_SCRIPT), "exec"), ns)
            return ns[name]
    raise KeyError(name)


def make_checkpoints(seeds, root: Path, force=False, verify_ablation=True) -> list:
    """Deterministic initial checkpoint per (dataset, seed): the global generators
    are seeded exactly as set_global_seed(seed) does and Net_CIFAR10() is built on
    CPU, which is how the component analysis built its common initial model. Where
    a historical ablation fingerprint exists for that seed, its initial_weights_sha256
    is compared (a match proves the checkpoint reproduces that initialization)."""
    import random
    import numpy as np
    import torch
    net_cls = _net_class_from_source({"CIFAR10": "Net_CIFAR10"}[C.DATASET])
    man = load_ckpt_manifest(root)
    abl = REPO / "results" / "component_analysis_ablation" / "raw" / "official"
    out = []
    for s in seeds:
        p = ckpt_path(s, root)
        if p.exists() and not force:
            state = torch.load(p, map_location="cpu")
        else:
            random.seed(s); np.random.seed(s); torch.manual_seed(s)
            torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
            state = net_cls().state_dict()
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, p)
        sha = P.state_dict_sha256(state)
        match = None
        fp = abl / f"fingerprint__full_fedhad__CIFAR10__alpha-0.01__seed-{s}.json"
        if verify_ablation and fp.exists():
            match = json.loads(fp.read_text()).get("initial_weights_sha256") == sha
        rec = {"seed": s, "dataset": C.DATASET, "path": str(_rel(p)) if p.is_relative_to(REPO) else str(p),
               "sha256": sha, "torch": torch.__version__, "created_or_verified": datetime.now().isoformat(timespec="seconds"),
               "matches_component_analysis_init": match}
        if str(s) in man and man[str(s)]["sha256"] != sha:
            raise RuntimeError(f"checkpoint for seed {s} changed ({man[str(s)]['sha256']} -> {sha})")
        man[str(s)] = rec
        out.append(rec)
    ckpt_manifest_path(root).parent.mkdir(parents=True, exist_ok=True)
    ckpt_manifest_path(root).write_text(json.dumps(man, indent=2, sort_keys=True))
    return out


def checkpoint_for(seed: int, root: Path):
    man = load_ckpt_manifest(root)
    rec = man.get(str(seed))
    p = ckpt_path(seed, root)
    if rec is None or not p.exists():
        return None, None
    return p, rec["sha256"]


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def plan_step(args) -> list[Cell]:
    arms = list(C.STEP_ARMS_DEFAULT) + ([C.STEP_ARM_OPTIONAL] if args.include_lr_follow else [])
    return [Cell("step_permutation", arm, "FedHAD", a, s, out_root=args.out_step)
            for a in args.alphas for s in args.seeds for arm in arms]


def plan_tune(args) -> list[Cell]:
    P.check_seed_separation(args.val_seeds, C.EVAL_SEEDS)
    P.check_seed_separation(args.val_seeds, args.seeds)
    root = out_root_for("tuned_tune", args.out_step, args.out_tuned)
    cells = []
    for a in args.alphas:
        for s in args.val_seeds:
            cells.append(Cell("tuned_tune", "full_fedhad", "FedHAD", a, s, label="fedhad_reference", out_root=root))
            for m in args.methods:
                for lr in args.lr_grid:
                    for e in args.epochs_grid:
                        cells.append(Cell("tuned_tune", "static", m, a, s, lr, e, "grid", out_root=root))
    return cells


def read_frozen(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"ERROR: frozen configuration not found: {path}\n"
            "The evaluate phase needs an explicit frozen configuration. Run the tune phase, "
            "inspect --summarize-tune, then --freeze <rule> or write the file by hand "
            "(format: see the recorded results/tuned_controls_fedprox_validation/frozen_config.json).")
    fz = json.loads(path.read_text())
    for c in fz.get("configs", []):
        for k in ("method", "alpha", "lr", "epochs", "label"):
            if k not in c:
                raise SystemExit(f"ERROR: frozen config entry lacks {k!r}: {c}")
        if c["method"] not in C.STATIC_METHODS_ALLOWED:
            raise SystemExit(f"ERROR: unsupported method in frozen config: {c['method']}")
    if not fz.get("configs"):
        raise SystemExit("ERROR: frozen configuration has no 'configs'")
    return fz


def plan_evaluate(args, frozen: dict) -> list[Cell]:
    P.check_seed_separation(frozen.get("validation_seeds", args.val_seeds), args.seeds)
    root = out_root_for("tuned_evaluate", args.out_step, args.out_tuned)
    cells, seen = [], set()

    def add(c):
        if c.cell_id + str(c.out_root) not in seen:
            seen.add(c.cell_id + str(c.out_root))
            cells.append(c)

    for a in args.alphas:
        for s in args.seeds:
            # FedHAD reference: the SAME cell as the step_permutation experiment
            # (same code, checkpoint and seeding), so it is shared, never duplicated.
            add(Cell("step_permutation", "full_fedhad", "FedHAD", a, s, label="fedhad_reference", out_root=args.out_step))
            for c in frozen["configs"]:
                if abs(float(c["alpha"]) - a) < 1e-12 and c["method"] in args.methods:
                    add(Cell("tuned_evaluate", "static", c["method"], a, s, float(c["lr"]), int(c["epochs"]), c["label"], out_root=root))
            if not args.no_default_baseline:
                for m in args.methods:
                    add(Cell("tuned_evaluate", "static", m, a, s, C.DEFAULT_BASELINE["lr"],
                             C.DEFAULT_BASELINE["epochs"], "default_baseline", out_root=root))
    return cells


# --------------------------------------------------------------------------- #
# State and validation
# --------------------------------------------------------------------------- #
def write_json_atomic(path: Path, payload: dict) -> None:
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
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def validate(cell: Cell, camp: dict, ckpt_root: Path):
    fp = read_json(cell.run_dir / "fingerprint.json")
    if fp is None:
        return False, "fingerprint missing/unreadable"
    tel = cell.run_dir / "telemetry.csv"
    if not tel.exists():
        return False, "telemetry missing"
    with tel.open(encoding="utf-8") as fh:
        n_rows = sum(1 for _ in fh) - 1
    if n_rows != C.NUM_ROUNDS * C.CLIENT_SETUP:
        return False, f"telemetry has {n_rows} rows, expected {C.NUM_ROUNDS * C.CLIENT_SETUP}"
    if not fp.get("completed") or fp.get("arm") != cell.arm or int(fp.get("seed", -1)) != cell.seed \
            or abs(float(fp.get("alpha", -1)) - cell.alpha) > 1e-12:
        return False, "fingerprint does not match the cell"
    if cell.arm == "static":
        cfg = fp.get("config", {})
        if cfg.get("method") != cell.method or abs(float(cfg.get("static_lr", -1)) - cell.lr) > 1e-12 \
                or int(cfg.get("static_epochs", -1)) != cell.epochs:
            return False, "static configuration mismatch"
    _, sha = checkpoint_for(cell.seed, ckpt_root)
    if fp.get("initial_checkpoint", {}).get("sha256") != sha:
        return False, "initial checkpoint checksum mismatch"
    st = read_json(cell.state_path)
    if st is None:
        return False, "orphan artifacts (no state entry)"
    if st.get("campaign_fingerprint") != camp["fingerprint"]:
        return False, "produced by a different code/configuration (campaign fingerprint)"
    return True, ""


def status_of(cell: Cell, camp: dict, ckpt_root: Path):
    ok, why = validate(cell, camp, ckpt_root)
    if ok:
        return STATUS_COMPLETED, ""
    st = read_json(cell.state_path)
    if st and st.get("status") == STATUS_FAILED:
        return STATUS_FAILED, st.get("error_message") or why
    if st and st.get("status") == STATUS_RUNNING:
        return STATUS_PENDING, "interrupted while running"
    return STATUS_PENDING, why


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def cell_env(cell: Cell, args, ckpt: Path, sha: str, git: dict) -> dict:
    env = os.environ.copy()
    cfg = {"experiment": cell.experiment, "arm": cell.arm, "method": cell.method,
           "label": cell.label, "static_lr": cell.lr, "static_epochs": cell.epochs,
           "cell_id": cell.cell_id, "git": git, "runner": Path(__file__).name,
           "validation_seeds": args.val_seeds if cell.experiment == "tuned_tune" else None}
    env.update({
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "FL_DATASET": C.DATASET, "FL_ALPHA": repr(cell.alpha), "FL_SEED": str(cell.seed),
        "FL_NUM_ROUNDS": str(C.NUM_ROUNDS), "FL_CLIENT_SETUP": str(C.CLIENT_SETUP),
        "FL_COMM_DELAY": str(C.COMM_DELAY), "FL_USE_ENERGY": "1" if C.USE_ENERGY else "0",
        "FL_USE_FEDPROX": "0" if cell.method == "FedAvg" else "1",
        "FL_FEDPROX_MU": str(C.FEDPROX_MU),
        "FL_BASE_EPOCHS": str(C.BASE_EPOCHS), "FL_MIN_EPOCHS": str(C.MIN_EPOCHS),
        "FL_EPOCHS_DECAY": str(C.EPOCHS_DECAY), "FL_BASE_LR": str(C.BASE_LR),
        "FL_MIN_LR": str(C.MIN_LR), "FL_LR_DECAY": str(C.LR_DECAY),
        "FL_R2_ARM": cell.arm, "FL_R2_INIT_CKPT": str(ckpt), "FL_R2_INIT_SHA256": sha,
        "FL_R2_RUN_DIR": str(cell.run_dir), "FL_RESULTS_DIR": str(cell.run_dir / "report"),
        "FL_R2_CONFIG_JSON": json.dumps(cfg, default=str), "FL_RUN_TAG": "r2",
    })
    if cell.arm == "static":
        env["FL_R2_STATIC_LR"] = repr(cell.lr)
        env["FL_R2_STATIC_EPOCHS"] = str(cell.epochs)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def execute(cell: Cell, args, camp, ckpt_root, git) -> tuple[bool, str]:
    """Run one cell in its own subprocess. Output goes to the per-cell log and, if
    MOSTRAR_SAIDA_TREINO, also to the terminal. Ctrl+C marks the cell as failed
    (it is re-run on the next launch) and re-raises KeyboardInterrupt."""
    ckpt, sha = checkpoint_for(cell.seed, ckpt_root)
    if ckpt is None:
        return False, f"missing initial checkpoint for seed {cell.seed}"
    if cell.run_dir.exists() and (args.force or not validate(cell, camp, ckpt_root)[0]):
        # partial/stale artifacts of an interrupted attempt, or --force: keep them aside, never delete
        if any(cell.run_dir.iterdir()):
            cell.run_dir.rename(cell.run_dir.with_name(cell.run_dir.name + f".bak_{int(time.time())}"))
    cell.run_dir.mkdir(parents=True, exist_ok=True)
    cell.log_path.parent.mkdir(parents=True, exist_ok=True)
    prev = read_json(cell.state_path) or {}
    base = {"cell": asdict(cell) | {"out_root": str(cell.out_root)}, "cell_id": cell.cell_id,
            "campaign_fingerprint": camp["fingerprint"], "campaign": camp, "git": git,
            "initial_checkpoint": {"path": str(ckpt), "sha256": sha},
            "attempts": int(prev.get("attempts", 0)) + 1, "hostname": socket.gethostname(),
            "python": platform.python_version(), "started_at": datetime.now().isoformat(timespec="seconds"),
            "gpu_selector": args.gpu, "log": str(cell.log_path)}
    write_json_atomic(cell.state_path, {**base, "status": STATUS_RUNNING})
    t0 = time.time()
    interrupted = False
    rc = None
    with cell.log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} {cell.cell_id} =====\n")
        fh.flush()
        proc = subprocess.Popen([python_executable(args.python), "-u", str(C.ENTRY_SCRIPT)],
                                cwd=str(REPO), env=cell_env(cell, args, ckpt, sha, git),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in proc.stdout:
                fh.write(line)
                if args.show_output:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            rc = proc.wait()
        except KeyboardInterrupt:
            interrupted = True
            try:
                proc.wait(timeout=30)          # the child received the same SIGINT
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            rc = proc.returncode
            fh.write("\n[runner] interrupted by the user (Ctrl+C)\n")
    dur = time.time() - t0
    if interrupted:
        ok, err = False, "interrupted by the user (Ctrl+C)"
    else:
        ok, why = validate(cell, camp, ckpt_root) if rc == 0 else (False, "")
        err = "" if ok else (f"exit status {rc}" if rc else f"artifacts invalid: {why}")
    write_json_atomic(cell.state_path, {**base, "status": STATUS_COMPLETED if ok else STATUS_FAILED,
                                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                                        "duration_s": round(dur, 1), "exit_status": rc,
                                        "error_message": err})
    if interrupted:
        raise KeyboardInterrupt
    return ok, err


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_manifest(cells, camp, ckpt_root):
    by_root = {}
    for c in cells:
        by_root.setdefault(c.out_root, []).append(c)
    for root, cs in by_root.items():
        rows = []
        for c in cs:
            st, why = status_of(c, camp, ckpt_root)
            fp = read_json(c.run_dir / "fingerprint.json") or {}
            perm = fp.get("permutation") or {}
            tot, fin = fp.get("totals", {}), fp.get("final", {})
            rows.append({"cell_id": c.cell_id, "experiment": c.experiment, "arm": c.arm, "method": c.method,
                         "label": c.label, "alpha": c.alpha, "seed": c.seed, "lr": c.lr, "epochs": c.epochs,
                         "status": st, "reason": why,
                         "init_sha256": fp.get("initial_checkpoint", {}).get("sha256", ""),
                         "perm_id": perm.get("perm_id", ""),
                         "perm_mapping_receiver_to_donor": json.dumps(perm.get("mapping_receiver_to_donor", "")),
                         "perm_seed": perm.get("perm_seed", ""),
                         "total_steps": tot.get("optimizer_steps", ""), "total_examples": tot.get("examples_processed", ""),
                         "flops_nominal": tot.get("flops_nominal", ""), "flops_executed": tot.get("flops_executed", ""),
                         "acc_centralized": fin.get("acc_centralized", ""), "acc_distributed_val": fin.get("acc_distributed_val", ""),
                         "elapsed_s": fp.get("elapsed_seconds", "")})
        root.mkdir(parents=True, exist_ok=True)
        # an out_root can be shared by stages (the evaluate phase reuses the FedHAD cells of
        # step_permutation): keep rows of cells outside this plan instead of dropping them
        mpath = root / "manifest.csv"
        if mpath.exists():
            planned = {r["cell_id"] for r in rows}
            with mpath.open(newline="", encoding="utf-8") as fh:
                rows = [r for r in csv.DictReader(fh) if r["cell_id"] not in planned] + rows
            rows.sort(key=lambda r: (r["experiment"], r["arm"], str(r["label"]), str(r["lr"]),
                                     str(r["epochs"]), float(r["alpha"]), int(r["seed"])))
        with mpath.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)


def print_plan(cells, camp, ckpt_root, show_all=True):
    from collections import Counter
    n = len(cells)
    counts = Counter((c.experiment, c.arm if c.arm != "static" else f"static:{c.method}") for c in cells)
    status = Counter(status_of(c, camp, ckpt_root)[0] for c in cells)
    missing_ckpt = sorted({c.seed for c in cells if checkpoint_for(c.seed, ckpt_root)[0] is None})
    print("=" * 96)
    print(f"CONTROLS PLAN  runs={n}  rounds={n * C.NUM_ROUNDS} "
          f"(~{n * C.EST_SECONDS_PER_RUN / 3600:.1f} h at {C.EST_SECONDS_PER_RUN} s/run, rough)")
    for (e, a), k in sorted(counts.items()):
        print(f"  {e:18s} {a:26s} {k:4d} runs")
    print(f"  status: " + ", ".join(f"{k}={v}" for k, v in sorted(status.items())))
    print(f"  seeds without initial checkpoint: {missing_ckpt if missing_ckpt else 'none'}")
    print(f"  outputs: " + ", ".join(sorted({str(_rel(c.out_root)) for c in cells})))
    print("=" * 96)
    if show_all:
        print(f"{'#':>4} {'experiment':16s} {'arm':24s} {'method':8s} {'alpha':>6} {'seed':>5} {'lr':>7} {'E':>3} {'label':18s} status  -> run dir")
        for i, c in enumerate(cells, 1):
            st, _ = status_of(c, camp, ckpt_root)
            print(f"{i:4d} {c.experiment:16s} {c.arm:24s} {c.method:8s} {c.alpha:6g} {c.seed:5d} "
                  f"{'' if c.lr is None else f'{c.lr:g}':>7} {'' if c.epochs is None else c.epochs:>3} "
                  f"{c.label:18s} {st:9s} {_rel(c.run_dir)}")


def summarize_tune(args):
    root = out_root_for("tuned_tune", args.out_step, args.out_tuned)
    cells = plan_tune(args)
    recs = []
    for c in cells:
        fp = read_json(c.run_dir / "fingerprint.json")
        if not fp or not fp.get("completed"):
            continue
        recs.append({"alpha": c.alpha, "method": c.method, "arm": c.arm, "lr": c.lr, "epochs": c.epochs,
                     "seed": c.seed, "val_acc": fp["final"].get(C.SELECTION_METRIC),
                     "test_acc_not_for_selection": fp["final"].get("acc_centralized"),
                     "steps": fp["totals"]["optimizer_steps"], "examples": fp["totals"]["examples_processed"],
                     "tflops_nominal": fp["totals"]["flops_nominal"] / 1e12,
                     "tflops_executed": fp["totals"]["flops_executed"] / 1e12})
    print(f"completed tune runs: {len(recs)} / {len(cells)}")
    if not recs:
        return
    import statistics as st
    groups = {}
    for r in recs:
        groups.setdefault((r["alpha"], r["method"], r["arm"], r["lr"], r["epochs"]), []).append(r)
    rows = []
    for (a, m, arm, lr, e), g in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], str(x[0][3]), str(x[0][4]))):
        va = [x["val_acc"] for x in g if x["val_acc"] is not None]
        rows.append({"alpha": a, "method": m, "arm": arm, "lr": lr, "epochs": e, "n_seeds": len(g),
                     "val_acc_mean": st.mean(va) if va else None, "val_acc_sd": st.stdev(va) if len(va) > 1 else None,
                     "steps_mean": st.mean(x["steps"] for x in g), "examples_mean": st.mean(x["examples"] for x in g),
                     "tflops_nominal_mean": st.mean(x["tflops_nominal"] for x in g),
                     "tflops_executed_mean": st.mean(x["tflops_executed"] for x in g),
                     "test_acc_mean_NOT_FOR_SELECTION": st.mean(x["test_acc_not_for_selection"] for x in g)})
    root.mkdir(parents=True, exist_ok=True)
    with (root / "validation_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    proposal = {"selection_metric": C.SELECTION_METRIC, "budget_tol": C.BUDGET_TOL,
                "validation_seeds": args.val_seeds, "generated_at": datetime.now().isoformat(timespec="seconds"),
                "summary_sha256": _sha_file(root / "validation_summary.csv"), "by_alpha_method": []}
    for a in args.alphas:
        ref = [r for r in rows if r["alpha"] == a and r["arm"] == "full_fedhad"]
        for m in args.methods:
            grid = [r for r in rows if r["alpha"] == a and r["method"] == m and r["arm"] == "static"
                    and r["n_seeds"] == len(args.val_seeds) and r["val_acc_mean"] is not None]
            entry = {"alpha": a, "method": m, "fedhad_reference": ref[0] if ref else None,
                     "complete_grid_points": len(grid)}
            if ref and grid:
                cap = ref[0]["steps_mean"] * (1 + C.BUDGET_TOL)
                elig = [r for r in grid if r["steps_mean"] <= cap]
                best = sorted(elig, key=lambda r: (-r["val_acc_mean"], r["steps_mean"]))
                entry["budget_matched"] = best[0] if best else None
                entry["budget_cap_steps"] = cap
            nd = [r for r in grid if not any((o["val_acc_mean"] >= r["val_acc_mean"] and o["steps_mean"] <= r["steps_mean"]
                                              and (o["val_acc_mean"] > r["val_acc_mean"] or o["steps_mean"] < r["steps_mean"]))
                                             for o in grid)]
            entry["operating_points_nondominated"] = sorted(nd, key=lambda r: r["steps_mean"])
            proposal["by_alpha_method"].append(entry)
    write_json_atomic(root / "selection_proposal.json", proposal)
    hdr = f"{'alpha':>6} {'method':8} {'arm':12} {'lr':>7} {'E':>3} {'n':>2} {'val_acc':>15} {'steps':>9} {'TF_nom':>8} {'TF_exe':>8}"
    print(hdr)
    for r in rows:
        va = "" if r["val_acc_mean"] is None else f"{r['val_acc_mean']:.4f}±{(r['val_acc_sd'] or 0):.4f}"
        print(f"{r['alpha']:6g} {r['method']:8} {r['arm']:12} {'' if r['lr'] is None else r['lr']:>7} "
              f"{'' if r['epochs'] is None else r['epochs']:>3} {r['n_seeds']:2d} {va:>15} {r['steps_mean']:9.0f} "
              f"{r['tflops_nominal_mean']:8.2f} {r['tflops_executed_mean']:8.2f}")
    for e in proposal["by_alpha_method"]:
        bm = e.get("budget_matched")
        print(f"\nalpha={e['alpha']} {e['method']}: budget_matched -> "
              f"{'' if not bm else (bm['lr'], bm['epochs'], round(bm['val_acc_mean'], 4))}; "
              f"non-dominated: {[(r['lr'], r['epochs']) for r in e['operating_points_nondominated']]}")
    print(f"\nwritten: {root / 'validation_summary.csv'} and {root / 'selection_proposal.json'}")
    print("Selection is NOT applied automatically: use --freeze budget_matched, or write frozen_config.json by hand.")


def freeze(args, rule: str):
    root = out_root_for("tuned_tune", args.out_step, args.out_tuned)
    prop = read_json(root / "selection_proposal.json")
    if prop is None:
        raise SystemExit("ERROR: no selection_proposal.json; run --summarize-tune first")
    if rule != "budget_matched":
        raise SystemExit("ERROR: only 'budget_matched' can be frozen automatically; operating points "
                         "must be chosen by hand in frozen_config.json")
    configs = []
    for e in prop["by_alpha_method"]:
        bm = e.get("budget_matched")
        if not bm:
            raise SystemExit(f"ERROR: no budget_matched choice for alpha={e['alpha']} {e['method']} "
                             "(incomplete grid or FedHAD reference missing)")
        configs.append({"method": e["method"], "alpha": e["alpha"], "lr": bm["lr"], "epochs": bm["epochs"],
                        "label": "tuned_budget_matched"})
    out = args.frozen_config
    if out.exists() and not args.force:
        raise SystemExit(f"ERROR: {out} exists; refusing to overwrite without --force")
    write_json_atomic(out, {"selection_rule": rule, "validation_seeds": prop["validation_seeds"],
                            "selection_metric": prop["selection_metric"],
                            "source_proposal_sha256": _sha_file(root / "selection_proposal.json"),
                            "frozen_at": datetime.now().isoformat(timespec="seconds"), "configs": configs})
    print(f"frozen configuration written to {out}:\n" + json.dumps(configs, indent=2))


def expected_counts_text(args) -> str:
    n_step = len(args.alphas) * len(C.EVAL_SEEDS) * len(C.STEP_ARMS_DEFAULT)
    n_step_opt = len(args.alphas) * len(C.EVAL_SEEDS)
    n_grid = len(C.LR_GRID) * len(C.EPOCHS_GRID)
    lines = [
        f"step_permutation (default arms {C.STEP_ARMS_DEFAULT}): {n_step} runs = {len(C.STEP_ARMS_DEFAULT)} arms x {len(args.alphas)} alphas x {len(C.EVAL_SEEDS)} seeds",
        f"  + optional {C.STEP_ARM_OPTIONAL} (--include-lr-follow): +{n_step_opt} runs",
        f"tuned_controls tune (FedProx only): {n_grid * len(args.alphas) * len(C.VAL_SEEDS) + len(args.alphas) * len(C.VAL_SEEDS)} runs = "
        f"{n_grid} grid points x {len(args.alphas)} alphas x {len(C.VAL_SEEDS)} val seeds + {len(args.alphas) * len(C.VAL_SEEDS)} FedHAD reference runs",
        f"  with --methods FedAvg,FedProx: {2 * n_grid * len(args.alphas) * len(C.VAL_SEEDS) + len(args.alphas) * len(C.VAL_SEEDS)} runs",
        f"tuned_controls evaluate (FedProx, 1 frozen config per alpha, default baseline on): "
        f"{2 * len(args.alphas) * len(C.EVAL_SEEDS)} new runs + {len(args.alphas) * len(C.EVAL_SEEDS)} FedHAD reference runs "
        f"shared with step_permutation (0 extra if that experiment is complete)",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["pipeline", "step_permutation", "tuned_controls", "all"], default="pipeline",
                    help="default: pipeline (the EXECUTAR_* switches at the top of this file)")
    ap.add_argument("--phase", choices=["tune", "evaluate"], default=None,
                    help="with --experiment tuned_controls; default: tune then evaluate")
    ap.add_argument("--alpha", default=",".join(str(a) for a in ALPHAS_ATIVOS), help="e.g. 0.01 or 0.01,0.1")
    ap.add_argument("--seeds", default=SEEDS_AVALIACAO, help="evaluation seeds (step_permutation, evaluate); e.g. 42 or 42-71")
    ap.add_argument("--val-seeds", default=SEEDS_VALIDACAO, help="tuning seeds; must not overlap 42-71")
    ap.add_argument("--methods", default=",".join(METODOS_ESTATICOS), help="static-control methods: FedProx and/or FedAvg")
    ap.add_argument("--lr-grid", default=",".join(map(str, C.LR_GRID)))
    ap.add_argument("--epochs-grid", default=",".join(map(str, C.EPOCHS_GRID)))
    ap.add_argument("--include-lr-follow", action="store_true", default=INCLUIR_LR_FOLLOW,
                    help="also run step_permuted_lrfollow")
    ap.add_argument("--no-default-baseline", action="store_true", help="evaluate: skip the untuned (0.01, 5) baseline")
    ap.add_argument("--frozen-config", type=Path, default=C.OUT_TUNED / "frozen_config.json")
    ap.add_argument("--out-step", type=Path, default=C.OUT_STEP)
    ap.add_argument("--out-tuned", type=Path, default=C.OUT_TUNED)
    ap.add_argument("--checkpoint-dir", type=Path, default=C.OUT_CKPT)
    ap.add_argument("--gpu", default=GPU, help="value for CUDA_VISIBLE_DEVICES of each run (e.g. 0)")
    ap.add_argument("--cpu", action="store_true", help="hide GPUs from the runs")
    ap.add_argument("--python", default=None, help="interpreter for the runs (default: env_flwr_pt)")
    ap.add_argument("--quiet", action="store_true", help="do not echo training output (it still goes to the logs)")
    ap.add_argument("--max-runs", type=int, default=None, help="stop after N executed runs")
    ap.add_argument("--dry-run", action="store_true", help="list the plan; execute nothing")
    ap.add_argument("--status", action="store_true", help="plan + status + manifest; execute nothing")
    ap.add_argument("--force", action="store_true", help="re-run completed cells (old artifacts are renamed, never deleted)")
    ap.add_argument("--rerun-failed", action="store_true", help="execute only cells whose last attempt failed")
    ap.add_argument("--make-checkpoints", action="store_true", help="only create/verify initial checkpoints for --seeds and --val-seeds")
    ap.add_argument("--summarize-tune", action="store_true", help="validation table + selection proposal")
    ap.add_argument("--freeze", default=None, metavar="RULE", help="freeze a rule from the proposal (budget_matched)")
    ap.add_argument("--print-counts", action="store_true", help="expected run counts of the prepared configuration")
    ap.add_argument("--self-test", action="store_true", help="run the unit tests of the pure functions (no training)")
    a = ap.parse_args(argv)
    a.show_output = MOSTRAR_SAIDA_TREINO and not a.quiet
    a.alphas = [float(x) for x in str(a.alpha).split(",")]
    a.seeds = P.parse_seed_list(a.seeds)
    a.val_seeds = P.parse_seed_list(a.val_seeds)
    a.methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    for m in a.methods:
        if m not in C.STATIC_METHODS_ALLOWED:
            ap.error(f"--methods: {m} not in {C.STATIC_METHODS_ALLOWED}")
    a.lr_grid = [float(x) for x in a.lr_grid.split(",")]
    a.epochs_grid = [int(x) for x in a.epochs_grid.split(",")]
    for al in a.alphas:
        if al not in C.ALPHAS:
            ap.error(f"--alpha {al} is outside the prepared scope {C.ALPHAS}")
    if a.experiment == "pipeline":
        a.stages = [st for st, on in (("step_permutation", EXECUTAR_STEP_PERMUTATION), ("tune", EXECUTAR_TUNE),
                                      ("evaluate", EXECUTAR_EVALUATE)) if on]
    elif a.experiment == "step_permutation":
        a.stages = ["step_permutation"]
    elif a.experiment == "tuned_controls":
        a.stages = [a.phase] if a.phase else ["tune", "evaluate"]
    else:  # all
        a.stages = ["step_permutation"] + ([a.phase] if a.phase else ["tune", "evaluate"])
    return a


# Kept for the tests and for external callers: the plan of the requested stages
# that can be planned now (evaluate only when the frozen configuration exists).
def build_plan(args) -> list[Cell]:
    cells = []
    for st in args.stages:
        cells += stage_cells(args, st, strict=True)
    return dedup(cells)


def dedup(cells):
    uniq, seen = [], set()
    for c in cells:
        k = (str(c.out_root), c.cell_id)
        if k not in seen:
            seen.add(k); uniq.append(c)
    return uniq


def stage_cells(args, stage: str, strict=False) -> list[Cell]:
    if stage == "step_permutation":
        return plan_step(args)
    if stage == "tune":
        return plan_tune(args)
    if stage == "evaluate":
        if not args.frozen_config.exists() and not strict:
            return []
        return plan_evaluate(args, read_frozen(args.frozen_config))
    raise ValueError(stage)


STAGE_TITLE = {"step_permutation": "EXPERIMENTO 1 — step-preserving permuted allocation",
               "tune": "EXPERIMENTO 2 — tuned static controls: TUNE (seeds de validação)",
               "evaluate": "EXPERIMENTO 2 — tuned static controls: EVALUATE (config congelada, seeds 42-71)"}


def _fmt_dur(sec: float) -> str:
    sec = int(max(sec, 0))
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}min" if h else f"{m}min{sec % 60:02d}s"


def _progress_log(event: dict):
    p = REPO / "results" / "execution_logs" / "tuned_controls_and_step_permutation_progress.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **event}) + "\n")


def ensure_checkpoints(args, seeds) -> None:
    """Create the missing initial checkpoints (one subprocess with the venv python,
    so the runner itself never needs torch)."""
    missing = sorted({s for s in seeds if checkpoint_for(s, args.checkpoint_dir)[0] is None})
    if not missing:
        return
    print(f">>> Criando checkpoints iniciais comuns para {len(missing)} seed(s): {missing}", flush=True)
    seeds_s = ",".join(map(str, missing))
    cmd = [python_executable(args.python), str(Path(__file__).resolve()), "--make-checkpoints",
           "--seeds", seeds_s, "--val-seeds", seeds_s,
           "--checkpoint-dir", str(args.checkpoint_dir)]
    env = os.environ.copy(); env["CONTROLS_NO_REEXEC"] = "1"
    if subprocess.run(cmd, cwd=str(REPO), env=env).returncode != 0:
        raise SystemExit("ERRO: falha ao criar os checkpoints iniciais")
    still = sorted({s for s in seeds if checkpoint_for(s, args.checkpoint_dir)[0] is None})
    if still:
        raise SystemExit(f"ERRO: checkpoints ainda ausentes para {still}")


class Runner:
    """Sequential execution with [i/N] progress, ETA and a final summary."""

    def __init__(self, args, camp, git):
        self.args, self.camp, self.git = args, camp, git
        self.ok, self.failed, self.durations = [], [], []
        self.consecutive_failures = 0
        self.t_start = time.time()

    def eta(self, remaining: int) -> str:
        per = (sum(self.durations) / len(self.durations)) if self.durations else C.EST_SECONDS_PER_RUN
        return _fmt_dur(per * remaining) + ("" if self.durations else " (estimativa)")

    def pending(self, cells):
        todo, done = [], 0
        for c in cells:
            st, _ = status_of(c, self.camp, self.args.checkpoint_dir)
            if st == STATUS_COMPLETED and not self.args.force:
                done += 1; continue
            if self.args.rerun_failed and st != STATUS_FAILED:
                continue
            todo.append(c)
        return todo, done

    def run_stage(self, stage: str, cells, later_pending: int) -> bool:
        """Returns False if the pipeline must stop (max-runs or too many failures)."""
        args = self.args
        todo, done = self.pending(cells)
        print("\n" + "#" * 96)
        print(f"# {STAGE_TITLE[stage]}")
        print(f"# total {len(cells)} | já concluídas {done} | a executar {len(todo)} "
              f"| ETA desta etapa ~{self.eta(len(todo))}")
        print("#" * 96, flush=True)
        if not todo:
            write_manifest(cells, self.camp, args.checkpoint_dir)
            return True
        ensure_checkpoints(args, [c.seed for c in todo])
        for i, c in enumerate(todo, 1):
            if args.max_runs is not None and len(self.ok) + len(self.failed) >= args.max_runs:
                print(f"--max-runs {args.max_runs} atingido; parando.")
                return False
            head = (f"[{i}/{len(todo)}] {stage} | {c.arm if c.arm != 'static' else f'{c.method} lr={c.lr:g} E={c.epochs}'}"
                    f" | alpha={c.alpha:g} | seed={c.seed}")
            print(f"\n>>> INICIANDO {head} ({datetime.now():%Y-%m-%d %H:%M:%S})", flush=True)
            _progress_log({"event": "start", "stage": stage, "cell": c.cell_id, "out_root": _rel(c.out_root)})
            ok, err = execute(c, args, self.camp, args.checkpoint_dir, self.git)
            dur = (read_json(c.state_path) or {}).get("duration_s", 0.0)
            (self.ok if ok else self.failed).append((stage, c.cell_id, err))
            if ok:
                self.durations.append(float(dur)); self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            _progress_log({"event": "ok" if ok else "failed", "stage": stage, "cell": c.cell_id, "out_root": _rel(c.out_root),
                           "duration_s": dur, "error": err})
            remaining = len(todo) - i
            print(f"<<< {'OK' if ok else 'FALHOU'} {head} em {float(dur):.1f}s"
                  + ("" if ok else f" — {err} (log: {_rel(c.log_path)})"))
            print(f"    etapa: {done + i - sum(1 for f in self.failed if f[0] == stage)}/{len(cells)} concluídas"
                  f" | faltam {remaining} (~{self.eta(remaining)}) | falhas nesta sessão: {len(self.failed)}"
                  f" | ETA total conhecido ~{self.eta(remaining + later_pending)}", flush=True)
            write_manifest(cells, self.camp, args.checkpoint_dir)
            if self.consecutive_failures >= MAX_FALHAS_SEGUIDAS:
                print(f"\n!!! {MAX_FALHAS_SEGUIDAS} falhas seguidas — abortando para não queimar a fila. "
                      f"Veja o log acima; corrija e rode o mesmo comando para retomar.")
                return False
        return True

    def summary(self, note: str = ""):
        print("\n" + "=" * 96)
        print("RESUMO DAS EXECUÇÕES (esta sessão)")
        print("=" * 96)
        print(f"  concluídas agora: {len(self.ok)} | falhas: {len(self.failed)} | tempo: {_fmt_dur(time.time() - self.t_start)}")
        for st, cid, err in self.failed:
            print(f"  FALHOU  {st:16s} {cid}: {err}")
        if self.failed:
            print("  -> rode o mesmo comando de novo para refazer as que falharam/foram interrompidas.")
        if note:
            print(note)
        print("=" * 96, flush=True)


def run_pipeline(args) -> int:
    camp = campaign_fingerprint()
    git = git_info()
    runner = Runner(args, camp, git)
    known = {st: stage_cells(args, st) for st in args.stages}
    print("=" * 96)
    print(f"CONTROLS — etapas: {', '.join(args.stages)} | alphas {args.alphas} | métodos estáticos {args.methods}"
          f" | GPU {args.gpu if args.gpu is not None else '(padrão)'}")
    for st in args.stages:
        todo, done = runner.pending(known[st])
        extra = "" if known[st] or st != "evaluate" else " (aguarda frozen_config.json)"
        print(f"  {st:18s} total {len(known[st]):4d} | concluídas {done:4d} | pendentes {len(todo):4d}{extra}")
    print(f"  logs por execução: results/step_preserving_permutation/logs/ e results/tuned_controls_fedprox_validation/*/logs/ | progresso: results/execution_logs/tuned_controls_and_step_permutation_progress.jsonl")
    print("=" * 96, flush=True)
    note, interrupted = "", False
    try:
        for k, st in enumerate(args.stages):
            if st == "evaluate":
                if not args.frozen_config.exists():
                    if CONGELAR_AUTOMATICAMENTE and tune_complete(args, camp):
                        summarize_tune(args)
                        freeze(args, "budget_matched")
                    else:
                        note = pause_message(args, camp)
                        break
                known[st] = stage_cells(args, st)
            later = sum(len(runner.pending(known[s])[0]) for s in args.stages[k + 1:])
            n_before = len(runner.ok)
            if not runner.run_stage(st, known[st], later):
                break
            if st == "tune":
                if tune_complete(args, camp):
                    proposal = args.out_tuned / "tune" / "selection_proposal.json"
                    # regenerate only when there is something new, and never after freezing
                    # (frozen_config.json records the proposal's SHA-256)
                    if not args.frozen_config.exists() and (not proposal.exists() or len(runner.ok) > n_before):
                        print("\n>>> Tune completo — gerando resumo da validação (sem usar o conjunto de teste)")
                        summarize_tune(args)
                else:
                    note = "  Tune incompleto (há falhas); rode o mesmo comando para completar antes do evaluate."
                    break
    except KeyboardInterrupt:
        interrupted = True
        note = "  Interrompido (Ctrl+C). Rode o mesmo comando para continuar de onde parou."
    runner.summary(note)
    return 130 if interrupted else (0 if not runner.failed else 1)


def tune_complete(args, camp) -> bool:
    return all(status_of(c, camp, args.checkpoint_dir)[0] == STATUS_COMPLETED for c in plan_tune(args))


def pause_message(args, camp) -> str:
    if not tune_complete(args, camp):
        return ("  Evaluate não iniciado: o tune ainda não está completo.")
    return ("\n  PAUSA DE PROTOCOLO: o tune terminou, mas não há configuração congelada.\n"
            f"  Inspecione {_rel((args.out_tuned / 'tune' / 'validation_summary.csv'))} e\n"
            f"  {_rel((args.out_tuned / 'tune' / 'selection_proposal.json'))}, depois congele:\n"
            f"      python runners/{Path(__file__).name} --freeze budget_matched\n"
            "  (ou escreva frozen_config.json à mão). Em seguida rode o mesmo comando de sempre:\n"
            f"      python runners/{Path(__file__).name}\n"
            "  que continua direto no evaluate. Para pular esta pausa: CONGELAR_AUTOMATICAMENTE = True.")


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(REPO / "sanity_tests"), pattern="test_tuned_controls_policy.py")
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.print_counts:
        print(expected_counts_text(args)); return 0
    if args.make_checkpoints:
        seeds = sorted(set(args.seeds) | set(args.val_seeds))
        for r in make_checkpoints(seeds, args.checkpoint_dir, force=args.force):
            print(f"seed {r['seed']:3d}  {r['sha256']}  matches_component_analysis_init={r['matches_component_analysis_init']}")
        print(f"manifest: {ckpt_manifest_path(args.checkpoint_dir)}")
        return 0
    if args.summarize_tune:
        summarize_tune(args); return 0
    if args.freeze:
        freeze(args, args.freeze); return 0

    if args.dry_run or args.status:
        camp = campaign_fingerprint()
        cells = dedup([c for st in args.stages for c in stage_cells(args, st)])
        print_plan(cells, camp, args.checkpoint_dir, show_all=args.dry_run)
        if "evaluate" in args.stages and not args.frozen_config.exists():
            print(f"(evaluate não listado: {_rel(args.frozen_config) if args.frozen_config.is_relative_to(REPO) else args.frozen_config} "
                  "ainda não existe; é criado após o tune)")
        if args.status and cells:
            write_manifest(cells, camp, args.checkpoint_dir)
        return 0
    return run_pipeline(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:          # e.g. the leakage guard: report cleanly
        sys.exit(f"ERROR: {exc}")
    except KeyboardInterrupt:
        sys.exit("\nInterrompido. Rode o mesmo comando para continuar de onde parou.")
    except BrokenPipeError:            # output piped into `head`
        sys.exit(0)
