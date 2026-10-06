# -*- coding: utf-8 -*-
"""A13 - Audit of the wall-clock / communication-delay figure (Figure 11, test3_comunicacao).

Recomputes, from the raw .txt reports, the mean elapsed time per (folder, method, dataset,
delay) and compares it with
  (i)  results/main_campaign_summaries/analise_resumo_test3_comunicacao.csv (the table the figure is drawn from,
       analysis/main_campaign/figuras_artigo.py::gerar_test3), and
  (ii) the numbers quoted in the manuscript text.
Also reports the hardware recorded in each report and byte-identical duplicates (the
nested FedAvgM-results/FedAvgM-results folder). Read-only: writes only analysis/audit/out/.
"""
import csv
import hashlib
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDERS = {"alpha-0_5 folder": ROOT / "results/main_campaign_default_alpha_0_5",
           "alpha-0_01 folder": ROOT / "results/main_campaign_default_alpha_0_01"}
RESUMO = ROOT / "results/main_campaign_summaries/analise_resumo_test3_comunicacao.csv"
OUT = Path(__file__).resolve().parent / "out"
METHOD_NAME = {"FedAVG": "FedAvg", "FedAvgM": "FedAvgM", "FedProx": "FedProx", "FedHAD": "FedHAD"}

rx = {k: re.compile(v) for k, v in {
    "method": r"^Teste: (\S+)", "dataset": r"^Dataset: (\S+)", "seed": r"^SEED: (\d+)",
    "delay": r"^COMMUNICATION_DELAY: ([\d.]+)", "alpha": r"^ALPHA: ([\d.]+)",
    "rounds": r"^NUM_ROUNDS: (\d+)", "clients": r"^NUM_CLIENTS: (\d+)",
    "time": r"^Tempo total \(s\): ([\d.]+)", "gpu": r"^GPU: (\d+ x .+)$", "cpu": r"^CPU: (.+)$",
    "tag": r"^EXPERIMENT_TAG: (\S+)"}.items()}


def parse(p: Path) -> dict:
    d = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        for k, r in rx.items():
            if k not in d:
                m = r.match(line)
                if m:
                    d[k] = m.group(1).strip()
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, dup = [], defaultdict(list)
    for label, folder in FOLDERS.items():
        seen = {}
        for p in sorted(folder.rglob("*.txt")):
            if "test3_comunicacao" not in p.parts:
                continue
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h in seen:
                dup[label].append((str(p.relative_to(ROOT)), seen[h]))
                continue
            seen[h] = str(p.relative_to(ROOT))
            d = parse(p)
            d.update(folder=label, file=str(p.relative_to(ROOT)))
            rows.append(d)

    # duplicates by (method, dataset, delay, seed) that are NOT byte-identical
    key_count = defaultdict(list)
    for r in rows:
        key_count[(r["folder"], r["method"], r["dataset"], r["delay"], r["seed"])].append(r["file"])
    conflicting = {k: v for k, v in key_count.items() if len(v) > 1}

    lines = ["# A13 - wall-clock / communication-delay audit (test3_comunicacao)", ""]
    for label in FOLDERS:
        sub = [r for r in rows if r["folder"] == label]
        hw = defaultdict(int)
        for r in sub:
            hw[(r.get("gpu", "?"), r.get("cpu", "?"))] += 1
        lines.append(f"## {label}: {len(sub)} unique reports "
                     f"({len(dup[label])} byte-identical duplicates skipped)")
        lines.append("hardware recorded: " + "; ".join(f"{g} / {c}: {n}" for (g, c), n in hw.items()))
        lines.append("alpha: " + ", ".join(sorted({r['alpha'] for r in sub})) +
                     " | rounds: " + ", ".join(sorted({r['rounds'] for r in sub})) +
                     " | clients: " + ", ".join(sorted({r['clients'] for r in sub})))
        lines.append("")
        lines.append("| method | dataset | delay | n | mean time (s) | sd |")
        lines.append("|---|---|---|---|---|---|")
        groups = defaultdict(list)
        for r in sub:
            groups[(r["method"], r["dataset"], r["delay"])].append(float(r["time"]))
        for (m, ds, dl), v in sorted(groups.items()):
            lines.append(f"| {m} | {ds} | {dl} | {len(v)} | {st.mean(v):.2f} | {st.stdev(v):.2f} |")
        lines.append("")
    lines.append(f"conflicting (non-identical) reports for the same cell: {len(conflicting)}")
    for k, v in list(conflicting.items())[:10]:
        lines.append(f"  {k}: {v}")
    lines.append("")

    # comparison with the table the figure is drawn from
    ref = {}
    for r in csv.DictReader(RESUMO.open(encoding="utf-8")):
        ref[(r["metodo"], r["dataset"], str(float(r["comm_delay"])))] = (float(r["tempo_s_media"]), int(float(r["n_seeds"])))
    lines.append("## analise_resumo_test3_comunicacao.csv vs raw reports (max |diff| of the means, s)")
    for label in FOLDERS:
        diffs, n_mismatch = [], 0
        for (m, ds, dl), (t, n) in ref.items():
            v = [float(r["time"]) for r in rows if r["folder"] == label and r["method"] == m
                 and r["dataset"] == ds and float(r["delay"]) == float(dl)]
            if not v:
                continue
            diffs.append(abs(st.mean(v) - t))
            n_mismatch += len(v) != n
        lines.append(f"  {label}: {len(diffs)} cells compared, max |diff| = "
                     f"{max(diffs) if diffs else float('nan'):.3f} s, n mismatches = {n_mismatch}")
    lines.append("")

    # numbers quoted in the manuscript (Section 5.7 and the CC additions)
    quoted = {("FedHAD", "CIFAR10", 0.05): 458.1, ("FedProx", "CIFAR10", 0.05): 563.7, ("FedAVG", "CIFAR10", 0.05): 387.6}
    lines.append("## numbers quoted in the text (alpha-0_5 folder)")
    for (m, ds, dl), q in quoted.items():
        v = [float(r["time"]) for r in rows if r["folder"] == "alpha-0_5 folder" and r["method"] == m
             and r["dataset"] == ds and float(r["delay"]) == dl]
        lines.append(f"  {m} {ds} {dl}s: text {q} | raw mean {st.mean(v):.2f} (n={len(v)}) | "
                     f"{'OK' if abs(round(st.mean(v), 1) - q) < 0.051 else 'MISMATCH'}")
    incs = []
    for m in ["FedAVG", "FedAvgM", "FedProx", "FedHAD"]:
        for ds in ["MNIST", "FashionMNIST", "CIFAR10"]:
            t = {}
            for dl in (0.05, 0.5):
                t[dl] = st.mean(float(r["time"]) for r in rows if r["folder"] == "alpha-0_5 folder"
                                and r["method"] == m and r["dataset"] == ds and float(r["delay"]) == dl)
            incs.append((m, ds, t[0.5] - t[0.05]))
    lines.append("  increment 0.05 -> 0.5 s (text: 'between 9 and 19 s'): " +
                 ", ".join(f"{m}/{ds} {d:+.1f}" for m, ds, d in incs) +
                 f" | min {min(d for *_, d in incs):.1f}, max {max(d for *_, d in incs):.1f}")
    # ordering check: FedHAD faster than FedProx, slower than FedAvg/FedAvgM at every delay
    order_ok = True
    for ds in ["MNIST", "FashionMNIST", "CIFAR10"]:
        for dl in (0.05, 0.1, 0.5):
            mean = {m: st.mean(float(r["time"]) for r in rows if r["folder"] == "alpha-0_5 folder"
                               and r["method"] == m and r["dataset"] == ds and float(r["delay"]) == dl)
                    for m in ["FedAVG", "FedAvgM", "FedProx", "FedHAD"]}
            order_ok &= mean["FedProx"] > mean["FedHAD"] > max(mean["FedAVG"], mean["FedAvgM"])
    lines.append(f"  ordering FedProx > FedHAD > FedAvg, FedAvgM at every dataset and delay: {order_ok}")

    (OUT / "a13_wallclock_delay.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUT / "a13_wallclock_delay_raw.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["folder", "method", "dataset", "delay", "alpha", "seed", "rounds",
                                           "clients", "time", "gpu", "cpu", "tag", "file"])
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
