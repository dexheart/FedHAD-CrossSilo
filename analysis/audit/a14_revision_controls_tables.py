"""A14 - LaTeX tables of the control batteries (Tables 16 to 20).

Reads only the outputs of the three analysis scripts, never the raw runs:
  results/tuned_controls_fedprox_validation/tune/validation_summary.csv     (validation grid)
  results/step_preserving_permutation/analysis/{per_arm,contrasts,displacement}.csv
  results/lr_uniform_control/analysis/{contrasts,runs_tidy}.csv
  results/tuned_controls_fedprox_validation/evaluate_analysis/results.json
and writes analysis/audit/out/a14_*.tex, which are pasted verbatim into the manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
TUNE = ROOT / "results" / "tuned_controls_fedprox_validation" / "tune" / "validation_summary.csv"
STEP = ROOT / "results" / "step_preserving_permutation" / "analysis"
LRU = ROOT / "results" / "lr_uniform_control" / "analysis"
EVAL = ROOT / "results" / "tuned_controls_fedprox_validation" / "evaluate_analysis" / "results.json"
GG = ROOT / "results" / "global_gradient_diagnostic" / "analysis" / "tests.csv"

CAT = {1: "A superior", 2: "equivalent", 3: "B superior", 4: "inconclusive"}


def p(x):
    return "$<0.001$" if x < 0.001 else f"{x:.3f}"


def s(x, n=2):
    return f"{x:+.{n}f}"


def ci(lo, hi):
    return f"[{s(lo)}, {s(hi)}]"


def n_(x):
    return f"{x:,.0f}".replace(",", "{,}")


def validation_table():
    v = pd.read_csv(TUNE)
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"$\alpha$ & $\eta$ & $E = 2$ & $E = 3$ & $E = 4$ & $E = 5$ \\", r"\midrule"]
    for a in (0.01, 0.1):
        g = v[v["alpha"] == a]
        ref = g[g["arm"] == "full_fedhad"].iloc[0]
        st = g[g["arm"] == "static"]
        ratio = {e: st[st["epochs"] == e]["steps_mean"].iloc[0] / ref["steps_mean"] for e in (2, 3, 4, 5)}
        L.append(f"{a:g} & \\textit{{updates / FedHAD}} & " + " & ".join(f"\\textit{{{ratio[e]:.2f}}}" for e in (2, 3, 4, 5)) + r" \\")
        fz = json.loads((ROOT / "results" / "tuned_controls_fedprox_validation" / "frozen_config.json").read_text())
        sel = next(c for c in fz["configs"] if abs(float(c["alpha"]) - a) < 1e-12)
        for lr in (0.01, 0.0075, 0.005, 0.004):
            cells = []
            for e in (2, 3, 4, 5):
                r = st[(st["lr"] == lr) & (st["epochs"] == e)].iloc[0]
                txt = f"{100 * r['val_acc_mean']:.2f}"
                if abs(lr - sel["lr"]) < 1e-12 and e == sel["epochs"]:
                    txt = f"\\textbf{{{txt}}}"
                cells.append(txt)
            lr_txt = {0.01: r"$10^{-2}$", 0.0075: r"$7.5 \times 10^{-3}$", 0.005: r"$5 \times 10^{-3}$",
                      0.004: r"$4 \times 10^{-3}$"}[lr]
            L.append(f" & {lr_txt} & " + " & ".join(cells) + r" \\")
        L.append(f" & \\textit{{FedHAD}} & \\multicolumn{{4}}{{c}}{{{100 * ref['val_acc_mean']:.2f}}} \\\\")
        if a == 0.01:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def step_table():
    pa = pd.read_csv(STEP / "per_arm.csv")
    c = pd.read_csv(STEP / "contrasts.csv").set_index("alpha")
    d = pd.read_csv(STEP / "displacement.csv")
    L = [r"\begin{tabular}{llrrrrlrlrr}", r"\toprule",
         r"$\alpha$ & Arm & Accuracy (\%) & Updates & Examples & TFLOPs exec. & $\sum_k p_k \tau_k$ rel. & "
         r"$\Delta$ (pp) & 95\% CI & $d_z$ & $p_H$ \\", r"\midrule"]
    for a in (0.01, 0.1):
        g, k = d[d["alpha"] == a], c.loc[a]
        f = pa[(pa["alpha"] == a) & (pa["arm"] == "full_fedhad")].iloc[0]
        q = pa[(pa["alpha"] == a) & (pa["arm"] == "step_permuted_lrclient")].iloc[0]
        L.append(f"{a:g} & Full FedHAD & {f.acc_mean_pp:.2f} $\\pm$ {f.acc_sd_pp:.2f} & {n_(f.steps_mean)} & "
                 f"{n_(f.examples_mean)} & {f.tflops_executed_mean:.2f} & 1 & \\multicolumn{{4}}{{c}}{{reference}} \\\\")
        L.append(f" & Step-permuted & {q.acc_mean_pp:.2f} $\\pm$ {q.acc_sd_pp:.2f} & {n_(q.steps_mean)} & "
                 f"{n_(q.examples_mean)} & {q.tflops_executed_mean:.2f} & "
                 f"{g.ratio_sum_p_tau.median():.2f} [{g.ratio_sum_p_tau.min():.2f}, {g.ratio_sum_p_tau.max():.2f}] & "
                 f"{s(k.mean_diff_pp)} & {ci(k.ci95_boot_lo, k.ci95_boot_hi)} & {k.dz:.2f} & {p(k.p_wilcoxon_holm)} \\\\")
        if a == 0.01:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def lru_table():
    c = pd.read_csv(LRU / "contrasts.csv")
    c = c[c["analysis"] == "primary"]            # primary analysis = all 30 seeds
    t = pd.read_csv(LRU / "runs_tidy.csv")
    acc = t.groupby(["alpha", "arm"])["final_acc_pp"].mean()
    lab = {"full_fedhad": "Full", "u2": "U2", "lr_only_matched": "LR-only", "u1": "U1",
           "u2_arith": "U2-arith", "inv": "INV"}
    L = [r"\begin{tabular}{llrrrrrrrrl}", r"\toprule",
         r"Contrast (A $-$ B) & $\alpha$ & Acc. A & Acc. B & $\Delta$ (pp) & 95\% CI & 90\% CI & $\delta$ & "
         r"$p$ & $p_H$ & Category \\", r"\midrule"]
    order = [("full_fedhad", "u2"), ("lr_only_matched", "u1"), ("full_fedhad", "u2_arith"), ("full_fedhad", "inv")]
    for i, (A, B) in enumerate(order):
        if i:
            L.append(r"\addlinespace")
        for a in (0.01, 0.1):
            r = c[(c["A"] == A) & (c["B"] == B) & (c["alpha"] == a)].iloc[0]
            c95, c90 = json.loads(r.ci95), json.loads(r.ci90)
            L.append(f"{lab[A]} $-$ {lab[B]} & {a:g} & {acc[(a, A)]:.2f} & {acc[(a, B)]:.2f} & {s(r.mean_diff_pp)} & "
                     f"{ci(*c95)} & {ci(*c90)} & {r.delta:g} & {p(r.p_wilcoxon)} & "
                     f"{p(r.p_holm)} & {int(r.category)}: {CAT[int(r.category)]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def tuned_table():
    R = json.loads(EVAL.read_text())
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"$\alpha$ & Arm & $n$ & Accuracy (\%) & Updates & Updates rel. & TFLOPs exec. \\", r"\midrule"]
    for a in (0.01, 0.1):
        for i, r in enumerate([x for x in R["costs"] if x["alpha"] == a]):
            arm = {"FedHAD": "FedHAD", "FedProx tuned (budget_matched)": "FedProx, tuned",
                   "FedProx default (0.01, E=5)": "FedProx, default"}[r["arm"]]
            L.append(f"{(f'{a:g}' if i == 0 else '')} & {arm} & {r['n']} & {r['acc_mean_pp']:.2f} $\\pm$ {r['acc_sd_pp']:.2f} & "
                     f"{n_(r['steps_mean'])} & {r['steps_vs_fedhad']:.2f} & {r['tflops_exec_mean']:.1f} \\\\")
        if a == 0.01:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{6pt}", "",
          r"\begin{tabular}{llrrrrrrrl}", r"\toprule",
          r"Contrast (A $-$ B) & $\alpha$ & $n$ & $\Delta$ (pp) & 95\% CI & 90\% CI & $\delta$ & $p$ & $p_H$ & Category \\",
          r"\midrule"]
    short = {"FedHAD": "FedHAD", "FedProx tuned (budget_matched)": "FedProx, tuned",
             "FedProx default (0.01, E=5)": "FedProx, default"}
    rows = [("Primary", R["primary"]), ("Secondary", R["secondary"]), ("Sensitivity", R["sensitivity"])]
    for title, rs in rows:
        L.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for r in sorted(rs, key=lambda x: (x["contrast"], x["alpha"])):
            A, B = [short[x.strip()] for x in r["contrast"].split(" - ")]
            L.append(f"{A} $-$ {B} & {r['alpha']:g} & {r['n']} & {s(r['mean_diff_pp'])} & {ci(*r['ci95'])} & "
                     f"{ci(*r['ci90'])} & {r['delta']:g} & {p(r['p_wilcoxon'])} & {p(r['p_holm'])} & "
                     f"{r['category']}: {CAT[r['category']]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def gg_table():
    t = pd.read_csv(GG)
    def row(h, a, metric=None):
        q = t[(t["hypothesis"] == h) & (t["alpha"] == a)]
        if metric is not None:
            q = q[q["metric"] == metric]
        r = q.iloc[0]
        lo, hi = json.loads(r.ci95)
        mean = r["mean"] if pd.notna(r["mean"]) else r["mean_diff_pp"]
        return r, mean, lo, hi
    spec = [
        (r"\textit{Premise, equal client sizes (level 2)}", None),
        ("H1_primary", r"Gradient alignment $\rho(H_k, \cos(g_k, g_{-k}))$", (0.1, 0.2), None),
        ("H1_update", r"Update alignment $\rho(H_k, \cos(u_k, -g_{-k}))$", (0.1, 0.2), None),
        (r"\textit{Premise, partitions of the main campaign (level 1)}", None),
        ("H1_paper_partitions (raw)", r"Gradient alignment $\rho(H_k, \cos(g_k, g_{-k}))$", (0.1, 0.01), None),
        ("H1_paper_partitions (size-adjusted)", r"Coefficient of $H_k$ given $\log n_k$ and round", (0.1, 0.01), None),
        (r"\textit{Removed epochs, FedProx at five epochs (level 3)}", None),
        ("H2_removed_epochs", r"Epoch alignment, removed $-$ kept", (0.1, 0.01), None),
        ("H2_decline_by_skew", r"$\rho(H_k$, slope of epoch alignment$)$", (0.1, 0.01), None),
        (r"\textit{Allocation, equal client sizes (level 2), accuracy in pp}", None),
        ("H3_allocation (all seeds)", r"Full $-$ inverse epochs, all seeds", (0.1, 0.2), None),
        ("H3_allocation (seeds where allocations differ)", r"Full $-$ inverse epochs, differing allocations", (0.1, 0.2), None),
        (r"\textit{Class-sensitive metrics (levels 1 and 3), pp}", None),
        ("H4_classes (paper partitions)", r"Worst class, FedHAD $-$ FedProx default", (0.1, 0.01), ("worst_class_recall", "default")),
        ("H4_classes (paper partitions)", r"Worst class, FedHAD $-$ FedProx tuned", (0.1, 0.01), ("worst_class_recall", "tuned")),
        ("H4_classes (paper partitions)", r"Macro-F1, FedHAD $-$ FedProx default", (0.1, 0.01), ("macro_f1", "default")),
        ("H4_classes (paper partitions)", r"Macro-F1, FedHAD $-$ FedProx tuned", (0.1, 0.01), ("macro_f1", "tuned")),
    ]
    L = [r"\begin{tabular}{llrrrrl}", r"\toprule",
         r"Quantity & $\alpha$ & $n$ & Mean & 95\% CI & $p_H$ & Verdict \\", r"\midrule"]
    for item in spec:
        if item[1] is None:
            L.append(f"\\multicolumn{{7}}{{l}}{{{item[0]}}} \\\\")
            continue
        h, label, alphas, metric = item
        for a in alphas:
            q = t[(t["hypothesis"] == h) & (t["alpha"] == a)]
            if metric is not None:
                q = q[(q["metric"] == metric[0]) & q["statistic"].str.contains(f"fedprox_{metric[1]}")]
            r = q.iloc[0]
            lo, hi = json.loads(r.ci95)
            mean = r["mean"] if pd.notna(r["mean"]) else r["mean_diff_pp"]
            nd = 3 if abs(mean) < 0.1 else 2
            verdict = r.verdict
            if h.startswith("H3"):
                verdict = {"A superior": "FedHAD superior", "B superior": "inverse superior"}.get(verdict, verdict)
            if h.startswith("H4"):
                verdict = {"A superior": "FedHAD superior", "B superior": "FedProx superior"}.get(verdict, verdict)
            L.append(f"{label} & {a:g} & {int(r.n)} & {s(mean, nd)} & "
                     f"[{s(lo, nd)}, {s(hi, nd)}] & {p(r.p_holm)} & {verdict} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("validation", validation_table), ("step_permutation", step_table),
                     ("lr_uniform", lru_table), ("tuned", tuned_table), ("global_gradient", gg_table)):
        text = fn()
        (OUT / f"a14_{name}.tex").write_text(text + "\n", encoding="utf-8")
        print(f"%% ===== a14_{name}.tex\n{text}\n")


if __name__ == "__main__":
    main()
