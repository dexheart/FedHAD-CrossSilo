# -*- coding: utf-8 -*-
"""
Diagnóstico do colapso do SCAFFOLD sob heterogeneidade extrema (Seção 6.9 do artigo).

Roda o SCAFFOLD_Final.py numa partição em que ele colapsa (MNIST, alpha = 0,01, semente 61)
com uma instrumentação somente de leitura que imprime, por cliente e por rodada: nº de amostras,
nº de passos locais, normas de c, c_i e da correção c - c_i, deslocamento local e c_i novo.
Também roda a regra original do artigo SEM momento (SGD puro), para mostrar que o colapso não
vem da adaptação ao momento. Nada é gravado em results/: tudo vai para uma pasta temporária.

Rodar:  python sanity_tests/scaffold_collapse_diagnostic.py [--rounds 3] [--seed 61] [--dataset MNIST]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "algoritmos" / "SCAFFOLD_Final.py"
PY = ROOT / "env_flwr_pt" / "bin" / "python"

PROBE = '''        _n = lambda v: float(torch.sqrt(sum((t ** 2).sum() for t in v)))
        print(f"PERCLI rnd={config.get('server_round','?')} cid={self.cid} n={len(self.trainloader.dataset)} "
              f"tau={tau_i} |c|={_n(c_global):.3g} |c_i|={_n(c_local):.3g} "
              f"|c-c_i|={_n([a - b for a, b in zip(c_global, c_local)]):.3g} "
              f"|x-y|={_n([a - b for a, b in zip(x_before, y_after)]):.3g} |c_i_novo|={_n(c_local_new):.3g}", flush=True)
'''


def instrumented_copy(dst: Path, sgd_without_momentum: bool) -> Path:
    s = SRC.read_text(encoding="utf-8")
    anchor = "        torch.save(c_local_new, state_path)\n"
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor + PROBE)
    s = s.replace('            cfg["scaffold_state_dir"] = str(self.state_dir)',
                  '            cfg["scaffold_state_dir"] = str(self.state_dir)\n            cfg["server_round"] = int(server_round)')
    if sgd_without_momentum:
        assert s.count("LOCAL_MOMENTUM = 0.9") == 1
        s = s.replace("LOCAL_MOMENTUM = 0.9", "LOCAL_MOMENTUM = 0.0")
    out = dst / ("scaffold_sgd.py" if sgd_without_momentum else "scaffold_momentum.py")
    out.write_text(s, encoding="utf-8")
    shutil.copy(ROOT / "algoritmos" / "drift_telemetry.py", dst / "drift_telemetry.py")
    return out


def run(script: Path, args, tmp: Path):
    env = dict(os.environ, FL_DATASET=args.dataset, FL_ALPHA=str(args.alpha), FL_SEED=str(args.seed),
               FL_NUM_ROUNDS=str(args.rounds), FL_CLIENT_SETUP="5", FL_USE_ENERGY="0", FL_RUN_TAG="diag",
               FL_RESULTS_DIR=str(tmp / "out"), PYTHONPATH=str(tmp), RAY_DEDUP_LOGS="0")
    r = subprocess.run([str(PY), "-u", str(script)], cwd=str(ROOT), env=env, capture_output=True, text=True)
    rows = sorted(set(re.findall(r"PERCLI (rnd=\d+ cid=\d+ .*?)(?:\x1b|$)", r.stdout + r.stderr, re.M)),
                  key=lambda x: (int(re.search(r"rnd=(\d+)", x).group(1)), int(re.search(r"cid=(\d+)", x).group(1))))
    accs = re.findall(r"Acurácia \(Servidor\): ([0-9.]+)", r.stdout)
    return rows, accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MNIST"); ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=61); ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="scaffold_diag_"))
    try:
        for label, sgd in (("SCAFFOLD como implementado (momento 0,9, passos efetivos a_i)", False),
                           ("regra original do artigo, SGD sem momento", True)):
            rows, accs = run(instrumented_copy(tmp, sgd), args, tmp)
            print(f"\n=== {label} | {args.dataset}, alpha={args.alpha}, semente {args.seed}")
            for r in rows:
                print("  " + r)
            print(f"  acurácia centralizada por rodada (rodada 0 = inicial): {accs}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
