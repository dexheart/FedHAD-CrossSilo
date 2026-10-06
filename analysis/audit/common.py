"""Read-only helpers shared by the audit scripts.

Nothing here writes outside audit/out/. The report parser is written from scratch
(it does not reuse analysis/main_campaign/analisar_resultados.py) so the recomputation is
independent of the pipeline being audited.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
REV = ROOT / "results"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

CAMPAIGN_05 = RES / "main_campaign_default_alpha_0_5"
CAMPAIGN_001 = RES / "main_campaign_default_alpha_0_01"

_HDR = re.compile(r"^([A-Z_]+):\s*(.*)$")


def _literal_after(text: str, header: str):
    """Parse the Python literal printed after a '--- History (...) ---' header."""
    i = text.find(header)
    if i < 0:
        return None
    start = text.index("\n", i) + 1
    end = text.find("\n---", start)
    chunk = text[start:end if end >= 0 else None].strip()
    try:
        return ast.literal_eval(chunk)
    except Exception:
        try:  # reports of diverged runs contain bare nan/inf
            return eval(chunk, {"__builtins__": {}}, {"nan": float("nan"), "inf": float("inf")})
        except Exception:
            return None


def parse_report(path: Path) -> dict:
    t = path.read_text(encoding="utf-8", errors="replace")
    d: dict = {"path": str(path.relative_to(ROOT))}
    for line in t.splitlines()[:60]:
        m = _HDR.match(line.strip())
        if m and m.group(1) not in d:
            d[m.group(1)] = m.group(2).strip()
    m = re.search(r"^Teste:\s*(\S+)", t, re.M)
    d["method"] = m.group(1) if m else None
    m = re.search(r"^Dataset:\s*(\S+)", t, re.M)
    d["dataset"] = m.group(1) if m else None
    cen = _literal_after(t, "--- History (metrics, centralized) ---") or {}
    d["acc_hist"] = cen.get("accuracy", [])
    d["loss_hist"] = cen.get("loss", [])
    fit = _literal_after(t, "--- History (metrics, distributed, fit) ---") or {}
    d["flops_round"] = fit.get("total_flops_round", [])
    d["acc_final"] = d["acc_hist"][-1][1] if d["acc_hist"] else None
    d["total_flops"] = sum(v for _, v in d["flops_round"]) if d["flops_round"] else None
    m = re.search(r"FLOPS por amostra \(forward apenas\):\s*([0-9.e+]+)", t)
    d["flops_fwd"] = float(m.group(1)) if m else None
    m = re.search(r"FLOPS por amostra \(forward \+ backward, fator 3x\):\s*([0-9.e+]+)", t)
    d["flops_full"] = float(m.group(1)) if m else None
    # per-client lines (FedHAD) : Cliente k: amostras=N, avg_epochs=E, avg_lr=L
    cl = re.findall(r"Cliente (\d+): amostras=(\d+), avg_epochs=([0-9.]+), avg_lr=([0-9.]+)", t)
    # per-client FLOP lines (all methods): (N amostras × ~E épocas × R rodadas)
    cf = re.findall(r"Cliente (\d+): [0-9.]+ [GTMP]?FLOPs \((\d+) amostras × ~([0-9.]+) épocas × (\d+) rodadas\)", t)
    clients = {}
    for cid, n, e, r in cf:
        clients[int(cid)] = {"n": int(n), "epochs_avg": float(e), "rounds_participated": int(r)}
    for cid, n, e, lr in cl:
        c = clients.setdefault(int(cid), {"n": int(n)})
        c.update({"epochs_avg": float(e), "lr_avg": float(lr)})
    d["clients"] = clients
    m = re.search(r"^GPU:\s*(.*)$", t, re.M)
    d["gpu"] = m.group(1).strip() if m else None
    m = re.search(r"^CPU:\s*(.*)$", t, re.M)
    d["cpu"] = m.group(1).strip() if m else None
    m = re.search(r"^Data/Hora:\s*(.*)$", t, re.M)
    d["datetime"] = m.group(1).strip() if m else None
    return d


def method_dir(campaign: Path, method: str, tag: str, dataset: str) -> Path:
    ds = f"{dataset}-Standard" if method == "FedHAD" else dataset
    return campaign / f"{method}-results" / tag / ds


def seed_of(path: Path) -> int:
    return int(re.search(r"Seed-(\d+)", path.name).group(1))


def alpha_of(path: Path):
    m = re.search(r"Alpha-(\d+(?:_\d+)?)", path.name)
    return float(m.group(1).replace("_", ".")) if m else None
