# -*- coding: utf-8 -*-
"""
Testes de sanidade do SCAFFOLD (algoritmos/SCAFFOLD_Final.py).

Verificam que a implementação é de fato o SCAFFOLD de Karimireddy et al. (2020):
  1. a_i: o SGD do PyTorch com momento rho desloca os parâmetros em lr * a_i * g para
     um gradiente constante g (justifica dividir por a_i, e não por K, na opção II);
  2. com rho = 0 a atualização de c_i é exatamente a opção II do artigo;
  3. com gradiente constante, c_i+ recupera o gradiente local exato, com ou sem momento;
  4. invariante do servidor: com participação total, c = média dos c_i em toda rodada;
  5. com c = c_i = 0 o treino local é bit a bit o do FedAvg (train() do FedAVG_Final.py);
  6. correção do drift: num problema federado quadrático com curvaturas heterogêneas e
     muitos passos locais, o FedAvg converge para um ponto enviesado e o SCAFFOLD para o
     ótimo global; a versão com K*lr no lugar de a_i*lr não converge com momento 0,9;
  7. (com --flower) uma execução federada real de 2 rodadas no CIFAR-10.

Importar SCAFFOLD_Final.py executa o código de módulo (carrega o CIFAR-10 e mede FLOPs),
como no teste do FedNova; nenhuma simulação roda sem --flower.
Rodar:  python sanity_tests/test_scaffold_sanity.py [--flower]
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("FL_DATASET", "CIFAR10")
os.environ.setdefault("FL_USE_ENERGY", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algoritmos"))

import torch  # noqa: E402
import SCAFFOLD_Final as S  # noqa: E402
import FedNova_Final as N  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'OK  ' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# 1. a_i ----------------------------------------------------------------------
def test_effective_steps():
    for rho in (0.0, 0.5, 0.9):
        for tau in (1, 7, 140):
            w = torch.nn.Parameter(torch.zeros(3))
            opt = torch.optim.SGD([w], lr=0.01, momentum=rho)
            g = torch.tensor([1.0, -2.0, 0.5])
            for _ in range(tau):
                opt.zero_grad(); w.grad = g.clone(); opt.step()
            expected = -0.01 * S.effective_local_steps(tau, rho) * g
            if not torch.allclose(w.detach(), expected, rtol=1e-5, atol=1e-7):
                check(f"a_i reproduz o deslocamento do SGD (rho={rho}, tau={tau})", False); return
    check("a_i reproduz exatamente o deslocamento do SGD do PyTorch (rho = 0; 0,5; 0,9)", True)
    same = all(abs(S.effective_local_steps(t, r) - N.effective_local_steps(t, r)) < 1e-12
               for t in (1, 5, 100) for r in (0.0, 0.9))
    check("a_i idêntico ao do FedNova_Final.py", same)


# 2-3. atualização de c_i ------------------------------------------------------
def test_control_update():
    rng = np.random.default_rng(0)
    ci, c, x, y = (rng.normal(size=4) for _ in range(4))
    K, lr = 13, 0.05
    got = S.scaffold_new_control([ci], [c], [x], [y], S.effective_local_steps(K, 0.0), lr)[0]
    check("rho = 0: c_i+ = c_i - c + (x - y)/(K lr) (opção II do artigo)",
          np.allclose(got, ci - c + (x - y) / (K * lr)))
    for rho in (0.0, 0.9):
        g = torch.tensor([0.3, -1.2, 2.0])           # gradiente local constante
        c_i, c_g = torch.tensor([0.1, 0.0, -0.4]), torch.tensor([-0.2, 0.5, 0.1])
        w = torch.nn.Parameter(torch.zeros(3)); x0 = w.detach().clone()
        opt = torch.optim.SGD([w], lr=0.01, momentum=rho)
        K = 50
        for _ in range(K):
            opt.zero_grad(); w.grad = g + (c_g - c_i); opt.step()
        new = S.scaffold_new_control([c_i], [c_g], [x0], [w.detach()], S.effective_local_steps(K, rho), 0.01)[0]
        check(f"gradiente constante: c_i+ = gradiente local exato (rho = {rho})",
              torch.allclose(new, g, atol=1e-4), f"c_i+ = {new.numpy().round(4)}, g = {g.numpy()}")


# 4. invariante do servidor ------------------------------------------------------
def test_server_invariant():
    rng = np.random.default_rng(1)
    n = 5
    c_i = [[np.zeros(3), np.zeros((2, 2))] for _ in range(n)]
    c = [np.zeros(3), np.zeros((2, 2))]
    ok = True
    for _ in range(20):
        deltas = []
        for i in range(n):
            new = [v + rng.normal(size=v.shape) for v in c_i[i]]
            deltas.append([a - b for a, b in zip(new, c_i[i])]); c_i[i] = new
        c = S.scaffold_server_control(c, deltas, n)
        mean = [sum(ci[j] for ci in c_i) / n for j in range(2)]
        ok &= all(np.allclose(a, b) for a, b in zip(c, mean))
    check("participação total: c = média dos c_i em todas as 20 rodadas", ok)


# 5. equivalência com o FedAvg quando c = c_i = 0 --------------------------------
def test_reduces_to_fedavg():
    torch.manual_seed(0)
    x = torch.randn(64, 3, 32, 32); y = torch.randint(0, 10, (64,))
    loader = [(x[i:i + 16], y[i:i + 16]) for i in range(0, 64, 16)]
    net_a, net_b = S.get_net(), S.get_net()
    net_b.load_state_dict(net_a.state_dict())
    for net in (net_a, net_b):
        net.eval()                                   # dropout determinístico na comparação
        net.train = lambda mode=True, _n=net: torch.nn.Module.train(_n, False)
    zeros = [torch.zeros_like(p) for p in net_a.parameters()]
    S.train(net_a, loader, epochs=2)
    steps = S.train_scaffold(net_b, loader, 2, zeros, zeros)
    same = all(torch.equal(a, b) for a, b in zip(net_a.state_dict().values(), net_b.state_dict().values()))
    check("c = c_i = 0: treino local idêntico (bit a bit) ao train() do FedAvg", same and steps == 8)


# 6. correção do drift num problema quadrático federado ---------------------------
def federated_quadratic(method, rho, rounds=300, K=20, lr=0.02, seed=0, denominator="a_i"):
    rng = np.random.default_rng(seed)
    d, n = 4, 5
    A = [np.diag(rng.uniform(0.2, 3.0, d)) for _ in range(n)]     # curvaturas heterogêneas
    b = [rng.normal(scale=3.0, size=d) for _ in range(n)]
    w_star = np.linalg.solve(sum(A), sum(Ai @ bi for Ai, bi in zip(A, b)))
    x = np.zeros(d); c = [np.zeros(d)]; c_i = [[np.zeros(d)] for _ in range(n)]
    for _ in range(rounds):
        ys, deltas = [], []
        for i in range(n):
            w = torch.nn.Parameter(torch.tensor(x)); opt = torch.optim.SGD([w], lr=lr, momentum=rho)
            corr = torch.tensor(c[0] - c_i[i][0]) if method == "scaffold" else torch.zeros(d, dtype=torch.float64)
            for _ in range(K):
                opt.zero_grad()
                w.grad = torch.tensor(A[i] @ (w.detach().numpy() - b[i])) + corr
                opt.step()
            yv = w.detach().numpy()
            if method == "scaffold":
                a_i = S.effective_local_steps(K, rho) if denominator == "a_i" else float(K)
                new = S.scaffold_new_control(c_i[i], c, [x], [yv], a_i, lr)
                deltas.append([new[0] - c_i[i][0]]); c_i[i] = new
            ys.append(yv)
        x = sum(ys) / n                                              # pesos iguais
        if method == "scaffold":
            c = S.scaffold_server_control(c, deltas, n)
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e6:
            return float("inf")
    return float(np.linalg.norm(x - w_star) / np.linalg.norm(w_star))


def test_drift_correction():
    for rho in (0.0, 0.9):
        fa = federated_quadratic("fedavg", rho)
        sc = federated_quadratic("scaffold", rho)
        check(f"drift (rho = {rho}): FedAvg enviesado, SCAFFOLD no ótimo global",
              fa > 1e-2 and sc < 1e-6, f"erro relativo FedAvg = {fa:.2e}, SCAFFOLD = {sc:.2e}")
    naive = federated_quadratic("scaffold", 0.9, denominator="K")
    check("rho = 0,9 com K*lr (opção II literal) não converge; a_i*lr é necessário",
          not naive < 1e-6, f"erro relativo com K*lr = {naive:.2e}")


# 7. execução federada real (opcional) -------------------------------------------
def test_flower_run():
    out = ROOT / "sanity_tests" / "flower_smoke"
    env = dict(os.environ, FL_DATASET="CIFAR10", FL_NUM_ROUNDS="2", FL_ALPHA="0.1", FL_SEED="42",
               FL_USE_ENERGY="0", FL_RESULTS_DIR=str(out), PYTHONPATH=str(ROOT / "algoritmos"))
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", str(ROOT / "algoritmos" / "SCAFFOLD_Final.py")],
                       cwd=str(ROOT), env=env, capture_output=True, text=True)
    log = r.stdout + r.stderr
    reports = list(out.rglob("SCAFFOLD__*.txt"))
    check("execução federada real (CIFAR-10, 5 clientes, 2 rodadas) termina e gera relatório",
          r.returncode == 0 and len(reports) == 1, f"{time.time() - t0:.0f} s")
    line = [l for l in log.splitlines() if l.startswith("SCAFFOLD: norma de c por rodada")]
    check("variável de controle global c deixa de ser zero após a 1ª rodada",
          bool(line) and all(float(v) > 0 for v in line[0].split("[")[1].rstrip("]").split(",")),
          line[0] if line else "linha de telemetria ausente")
    accs = [float(l.split("Acurácia (Servidor):")[1]) for l in log.splitlines() if "Acurácia (Servidor):" in l]
    check("acurácia sobe acima do acaso (10%) em 2 rodadas", bool(accs) and accs[-1] > 0.2,
          f"acurácias centralizadas = {accs}")
    if r.returncode != 0:
        print(log[-3000:])


if __name__ == "__main__":
    test_effective_steps()
    test_control_update()
    test_server_invariant()
    test_reduces_to_fedavg()
    test_drift_correction()
    if "--flower" in sys.argv:
        test_flower_run()
    print("\n" + ("TODOS OS TESTES PASSARAM" if not FAILURES else f"FALHAS: {FAILURES}"))
    sys.exit(1 if FAILURES else 0)
