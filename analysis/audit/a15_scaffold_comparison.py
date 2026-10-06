"""A15 - SCAFFOLD against FedHAD and the other baselines (drift-correction baseline, Section 6.9 of the manuscript).

SCAFFOLD was run on the same battery as FedNova (base point, heterogeneity family, natural
federation; 30 seeds, 10 rounds) and stored in the main campaign with default alpha = 0.5.
Pairs are aligned by seed (same data partition), as in the rest of the main campaign.
Wilcoxon two-sided; 10,000-resample percentile bootstrap of the paired mean; Holm within the
families of the paper: F1 = CIFAR-10 at alpha in {1.0, 0.5, 0.1, 0.01} (4 tests), F2 = MNIST,
FashionMNIST (4 alphas each) and FEMNIST (9 tests). Writes analysis/audit/out/a15_*.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "main_campaign"))
from analisar_resultados import parse_file  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
CAMP = ROOT / "results" / "main_campaign_default_alpha_0_5"
METHODS = ["FedAVG", "FedAvgM", "FedProx", "FedNova", "SCAFFOLD", "FedHAD"]
CELLS = ([("CIFAR10", a) for a in (1.0, 0.5, 0.1, 0.01)] + [("MNIST", a) for a in (1.0, 0.5, 0.1, 0.01)]
         + [("FashionMNIST", a) for a in (1.0, 0.5, 0.1, 0.01)] + [("FEMNIST", None)])
RNG = np.random.default_rng(20261004)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def load(method, ds, alpha):
    tag = "test8_femnist" if ds == "FEMNIST" else ("test1_convergencia" if alpha == 0.5 else "test2_robustez_alpha")
    root = CAMP / f"{method}-results" / tag
    out = {}
    for p in root.rglob("*.txt"):
        if f"__{ds}__" not in p.name or "Mode-ABLATION" in p.name:
            continue
        if alpha is not None and f"Alpha-{str(alpha).replace('.', '_')}__" not in p.name:
            continue
        r = parse_file(p)
        acc, diverged = r["acuracia_final"], False
        if acc is None:
            # parse_file cannot literal_eval a history that contains nan losses (a diverged run);
            # read the centralized accuracy list directly so the run is not silently dropped.
            text = p.read_text(errors="replace")
            block = re.search(r"--- History \(metrics, centralized\) ---\s*(\{.*?\})", text, re.S).group(1)
            accs = re.search(r"'accuracy':\s*\[(.*?)\]", block, re.S).group(1)
            acc = float(re.findall(r"\(\s*\d+,\s*([0-9.eE+-]+|nan)\)", accs)[-1])
            diverged = True
        out[int(re.search(r"Seed-(\d+)", p.name).group(1))] = {
            "acc": 100 * acc, "tflops": float(r["total_tflops"]), "time": float(r["tempo_s"]), "diverged": diverged}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, contr = [], []
    for ds, a in CELLS:
        data = {m: load(m, ds, a) for m in METHODS}
        for m in METHODS:
            v = data[m]
            rows.append({"dataset": ds, "alpha": a, "method": m, "n": len(v),
                         "diverged": sum(x["diverged"] for x in v.values()),
                         "acc_mean": np.mean([x["acc"] for x in v.values()]),
                         "acc_sd": np.std([x["acc"] for x in v.values()], ddof=1),
                         "tflops_mean": np.mean([x["tflops"] for x in v.values()]),
                         "time_mean_s": np.mean([x["time"] for x in v.values()])})
        for (other, subset) in (("FedHAD", "all"), ("FedNova", "all"), ("FedHAD", "non-diverged")):
            seeds = sorted(set(data["SCAFFOLD"]) & set(data[other]))
            if subset == "non-diverged":
                seeds = [s for s in seeds if not data["SCAFFOLD"][s]["diverged"]]
            d = np.array([data["SCAFFOLD"][s]["acc"] - data[other][s]["acc"] for s in seeds])
            boot = RNG.choice(d, size=(10000, len(d)), replace=True).mean(1)
            contr.append({"dataset": ds, "alpha": a, "contrast": f"SCAFFOLD - {other} ({subset} seeds)", "n": len(d),
                          "mean_pp": d.mean(), "ci_lo": np.quantile(boot, .025), "ci_hi": np.quantile(boot, .975),
                          "dz": d.mean() / d.std(ddof=1), "wins_scaffold": int((d > 0).sum()),
                          "wins_other": int((d < 0).sum()), "p": wilcoxon(d).pvalue,
                          "family": "F1" if ds == "CIFAR10" else "F2"})
    c = pd.DataFrame(contr)
    c["p_holm"] = np.nan
    for (fam, con), idx in c.groupby(["family", "contrast"]).groups.items():
        c.loc[idx, "p_holm"] = holm(c.loc[idx, "p"])
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "a15_methods_by_cell.csv", index=False)
    c.to_csv(OUT / "a15_scaffold_contrasts.csv", index=False)
    rank = (t.assign(rank=t.groupby(["dataset", "alpha"], dropna=False).acc_mean.rank(ascending=False))
              .pivot_table(index=["dataset", "alpha"], columns="method", values="rank", dropna=False))
    with (OUT / "a15_summary.txt").open("w") as o:
        o.write("Mean final accuracy (%) per cell\n")
        o.write(t.pivot_table(index=["dataset", "alpha"], columns="method", values="acc_mean", dropna=False)
                 [METHODS].round(2).to_string() + "\n\nRank of each method per cell (1 = most accurate)\n")
        o.write(rank[METHODS].to_string() + "\n\nPaired contrasts\n")
        o.write(c.round(4).to_string(index=False) + "\n")
    print((OUT / "a15_summary.txt").read_text())


if __name__ == "__main__":
    main()


def latex_table13(tex_path, out_path):
    """Table 13 extended with SCAFFOLD. FedNova columns are copied verbatim from the manuscript;
    SCAFFOLD columns come from this analysis (FedHAD - SCAFFOLD, Holm within this family)."""
    t = pd.read_csv(OUT / "a15_methods_by_cell.csv"); c = pd.read_csv(OUT / "a15_scaffold_contrasts.csv")
    tex = Path(tex_path).read_text(encoding="utf-8")
    body = tex[tex.index("\\label{tab:fednova_comparison}"):]
    body = body[body.index("\\midrule") + 8:body.index("\\bottomrule")]
    key = {"MNIST": "MNIST", "FashionMNIST": "FashionMNIST", "CIFAR-10": "CIFAR10", "FEMNIST": "FEMNIST"}
    fmt = lambda x: f"{x:+.2f}"
    pv = lambda x: "$<0.001$" if x < 0.001 else f"{x:.3f}"
    rows = []
    for line in [l.strip() for l in body.strip().splitlines() if l.strip()]:
        cells = [x.strip() for x in line.rstrip("\\").split("&")]
        ds = key[cells[0]]
        m = re.search(r"([0-9.]+)\$", cells[1])
        a = float(m.group(1)) if m else None
        sel = (t.dataset == ds) & ((t.alpha == a) if a is not None else t.alpha.isna())
        s = t[sel & (t.method == "SCAFFOLD")].iloc[0]
        cs = c[(c.dataset == ds) & ((c.alpha == a) if a is not None else c.alpha.isna())
               & (c.contrast == "SCAFFOLD - FedHAD (all seeds)")].iloc[0]
        acc_s = f"{s.acc_mean:.2f} $\\pm$ {s.acc_sd:.2f}"
        rows.append((cells, acc_s, f"{fmt(-cs.mean_pp)} [{fmt(-cs.ci_hi)}, {fmt(-cs.ci_lo)}]",
                     f"{pv(cs.p)} / {pv(cs.p_holm)}", ds, a))
    lines = []
    for cells, acc_s, d_s, p_s, ds, a in rows:
        v = load("SCAFFOLD", ds, a)
        n_coll = sum(1 for x in v.values() if x["acc"] < 20)
        acc_s = acc_s + (f"$^{{\\dagger{n_coll}}}$" if n_coll else "")
        # cells: dataset, setting, HAD acc, Nova acc, dAcc, pW, pH, TFLOPs, dFLOPs
        lines.append(" & ".join([cells[0], cells[1], cells[2], cells[3], acc_s, cells[4],
                                 f"{cells[5]} / {cells[6]}", d_s, p_s, cells[7], cells[8]]) + " \\\\")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines
