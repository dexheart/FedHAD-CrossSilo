#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of the global-gradient diagnostic battery. Applies the pre-registered tests
mechanically and writes, to results/global_gradient_diagnostic/analysis/:

  seed_level.csv     one row per (hypothesis, alpha, seed): the seed-level statistic
  tests.csv          every pre-registered test (statistic, CI, p, Holm p, verdict)
  accuracy.csv       final accuracy, worst-class recall and macro-F1 per arm
  report.md          the numbers above as text

Refuses to run if the pre-registration does not match its hash, if a run was produced
by other code, or if a pre-registered cell is missing (no partial analysis).
Usage:  python analysis/global_gradient_diagnostic/analyze_global_gradient.py [--out DIR] [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[2] / "algoritmos" / "global_gradient_diagnostic"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lr_uniform_control"))
import config_gg as C      # noqa: E402
import gg_prereg as PR     # noqa: E402
import lru_stats as S      # noqa: E402  (same statistics and decision rule as the other revision batteries)

VERDICT = {1: "A superior", 2: "equivalent", 3: "B superior", 4: "inconclusive"}


def read_csv(p):
    with Path(p).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(out_root: Path, smoke: bool):
    p, h = out_root / "preregistration.json", out_root / "preregistration.sha256"
    doc = None
    if not smoke:
        if not p.exists() or not h.exists():
            sys.exit("ERROR: no pre-registration; the battery has not started.")
        text = p.read_text(encoding="utf-8")
        if PR.sha256_text(text) != h.read_text().strip():
            sys.exit("ERROR: preregistration.json does not match its hash. Refusing to analyse.")
        doc = json.loads(text)
    runs, missing = {}, []
    seeds = C.SMOKE_SEEDS if smoke else C.SEEDS
    for lvl, L in C.LEVELS.items():
        for a in L["alphas"]:
            for s in seeds:
                for arm in L["arms"]:
                    cid = f"{L['partition']}__{arm}__alpha-{a:g}__seed-{s}"
                    d = out_root / "runs" / cid
                    try:
                        res = json.loads((d / "result.json").read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        missing.append(cid); continue
                    if doc and res.get("code_fingerprint") != doc["code_fingerprint"]:
                        sys.exit(f"ERROR: {cid} was produced by code other than the pre-registered one.")
                    raw = d / "attempt"
                    res["_server"] = read_csv(raw / "gg_server.csv")
                    res["_epochs"] = read_csv(raw / "gg_epochs.csv") if (raw / "gg_epochs.csv").exists() else []
                    runs[(L["partition"], arm, float(a), int(s))] = res
    if missing:
        sys.exit(f"ERROR: {len(missing)} pre-registered runs missing or incomplete (e.g. {missing[:3]}); "
                 "no partial analysis.")
    return doc, runs, seeds


# --------------------------------------------------------------------------- #
# Seed-level statistics (pure)
# --------------------------------------------------------------------------- #
def rho_h(rows, metric):
    x = [(f(r["H_k"]), f(r[metric])) for r in rows if f(r["H_k"]) is not None and f(r.get(metric)) is not None]
    if len(x) < 3 or len({a for a, _ in x}) < 2:
        return None
    return float(spearmanr([a for a, _ in x], [b for _, b in x])[0])


def pooled_size_adjusted(run_rows_by_seed, metric, resamples=2000, seed=20261002):
    """Coefficient of H_k in OLS metric ~ H_k + log n_k + round FE over all client-rounds of
    all seeds; 95% cluster-bootstrap CI (resampling seeds) and two-sided bootstrap p."""
    data = []
    for s, rows in run_rows_by_seed.items():
        for r in rows:
            h, m, n = f(r["H_k"]), f(r.get(metric)), f(r["n_k"])
            if h is not None and m is not None and n:
                data.append((s, h, m, np.log(n), int(r["round"])))
    if len(data) < 10:
        return {"n": 0, "mean": None, "ci95": (None, None), "p": None}
    rounds = sorted({d[4] for d in data})

    def coef(sub):
        X = np.array([[d[1], d[3]] + [1.0 if d[4] == t else 0.0 for t in rounds] for d in sub])
        y = np.array([d[2] for d in sub])
        return float(np.linalg.lstsq(X, y, rcond=None)[0][0])
    by = {}
    for d in data:
        by.setdefault(d[0], []).append(d)
    seeds = sorted(by)
    if len(seeds) < 5:
        return {"n": len(seeds), "mean": coef(data), "ci95": (None, None), "p": None}
    est = coef(data)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(resamples):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        boot.append(coef([d for s in pick for d in by[s]]))
    boot = np.array(boot)
    p = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
    return {"n": len(seeds), "mean": est, "ci95": (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))),
            "p": p, "n_neg": None, "n_pos": None}


def removed_minus_kept(epoch_rows):
    """Seed mean over client-rounds with E_full < 5 of mean(marg_cos_loo, removed epochs)
    - mean(marg_cos_loo, kept epochs); plus the share of removed epochs that increase the
    rest of the federation's loss to first order."""
    by = {}
    for r in epoch_rows:
        by.setdefault((int(r["round"]), int(r["client_id"])), []).append(r)
    diffs, pos, tot = [], 0, 0
    for rows in by.values():
        ef = int(float(rows[0]["E_full"]))
        if ef >= 5:
            continue
        kept = [f(r["marg_cos_loo"]) for r in rows if int(r["epoch"]) <= ef and f(r.get("marg_cos_loo")) is not None]
        rem = [r for r in rows if int(r["epoch"]) > ef and f(r.get("marg_cos_loo")) is not None]
        if kept and rem:
            diffs.append(np.mean([f(r["marg_cos_loo"]) for r in rem]) - np.mean(kept))
            pos += sum(1 for r in rem if (f(r.get("marg_lin_loo")) or 0.0) > 0)
            tot += len(rem)
    return (float(np.mean(diffs)) if diffs else None), (pos / tot if tot else None)


def slope_rho(epoch_rows):
    by = {}
    for r in epoch_rows:
        by.setdefault((int(r["round"]), int(r["client_id"])), []).append(r)
    hs, sl = [], []
    for rows in by.values():
        pts = sorted((int(r["epoch"]), f(r["marg_cos_loo"])) for r in rows if f(r.get("marg_cos_loo")) is not None)
        if len(pts) >= 3:
            hs.append(f(rows[0]["H_k"]))
            sl.append(float(np.polyfit([e for e, _ in pts], [v for _, v in pts], 1)[0]))
    if len(hs) < 3 or len(set(hs)) < 2:
        return None
    return float(spearmanr(hs, sl)[0])


def one_sample(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) < 5:
        return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None, "ci95": (None, None),
                "p": None, "n_neg": int((v < 0).sum()), "n_pos": int((v > 0).sum())}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "ci95": S.bootstrap_ci(v, 0.95), "p": S.wilcoxon_two_sided(v),
            "n_neg": int((v < 0).sum()), "n_pos": int((v > 0).sum())}


def holm_into(tests):
    ps = [t["p"] if t["p"] is not None else 1.0 for t in tests]
    for t, ph in zip(tests, S.holm(ps)):
        t["p_holm"] = ph
    return tests


def directional_verdict(t, expected_sign=-1):
    if t["p"] is None:
        return "insufficient data"
    lo, hi = t["ci95"]
    if t["p_holm"] < 0.05 and ((expected_sign < 0 and hi < 0) or (expected_sign > 0 and lo > 0)):
        return "supports the premise"
    if t["p_holm"] < 0.05 and ((expected_sign < 0 and lo > 0) or (expected_sign > 0 and hi < 0)):
        return "contradicts the premise"
    return "not detected"


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyse(runs, seeds, margins):
    tests, seed_rows, acc_rows = [], [], []
    L1, L2, L3 = C.LEVELS[1], C.LEVELS[2], C.LEVELS[3]

    def add_seed_rows(name, a, vals):
        for s, v in zip(seeds, vals):
            seed_rows.append({"hypothesis": name, "alpha": a, "seed": s, "value": v})

    # H1 primary / update (equal partition)
    for metric, name in (("grad_cos_loo", "H1_primary"), ("upd_cos_loo", "H1_update")):
        fam = []
        for a in L2["alphas"]:
            vals = [rho_h(runs[("equal", "full_fedhad", a, s)]["_server"], metric) for s in seeds]
            add_seed_rows(name, a, vals)
            fam.append({"hypothesis": name, "alpha": a, "level": 2, "statistic": f"Spearman(H_k, {metric}) per seed",
                        **one_sample(vals)})
        for t in holm_into(fam):
            t["verdict"] = directional_verdict(t, -1)
        tests += fam

    # H1 on the paper's partitions (raw and size-adjusted), Holm over four
    fam = []
    for a in L1["alphas"]:
        vals = [rho_h(runs[("dirichlet", "full_fedhad", a, s)]["_server"], "grad_cos_loo") for s in seeds]
        add_seed_rows("H1_paper_raw", a, vals)
        fam.append({"hypothesis": "H1_paper_partitions (raw)", "alpha": a, "level": 1,
                    "statistic": "Spearman(H_k, grad_cos_loo) per seed", **one_sample(vals)})
        fam.append({"hypothesis": "H1_paper_partitions (size-adjusted)", "alpha": a, "level": 1,
                    "statistic": "pooled OLS coefficient of H_k given log n_k and round (seed bootstrap)",
                    **pooled_size_adjusted({s: runs[("dirichlet", "full_fedhad", a, s)]["_server"] for s in seeds},
                                           "grad_cos_loo")})
    for t in holm_into(fam):
        t["verdict"] = directional_verdict(t, -1)
    tests += fam

    # H2 removed epochs and decline by skew (fedprox_default, every client 5 epochs), Holm over four
    fam, shares = [], {}
    for a in L3["alphas"]:
        d, sh = zip(*[removed_minus_kept(runs[("dirichlet", "fedprox_default", a, s)]["_epochs"]) for s in seeds])
        add_seed_rows("H2_removed_epochs", a, list(d))
        shares[a] = [x for x in sh if x is not None]
        fam.append({"hypothesis": "H2_removed_epochs", "alpha": a, "level": 3,
                    "statistic": "mean marg_cos_loo removed - kept epochs, per seed", **one_sample(d)})
        r = [slope_rho(runs[("dirichlet", "fedprox_default", a, s)]["_epochs"]) for s in seeds]
        add_seed_rows("H2_decline_by_skew", a, r)
        fam.append({"hypothesis": "H2_decline_by_skew", "alpha": a, "level": 3,
                    "statistic": "Spearman(H_k, slope of marg_cos_loo over epochs) per seed", **one_sample(r)})
    for t in holm_into(fam):
        t["verdict"] = directional_verdict(t, -1)
        if t["hypothesis"] == "H2_removed_epochs" and shares.get(t["alpha"]):
            t["share_removed_epochs_increasing_others_loss"] = float(np.mean(shares[t["alpha"]]))
    tests += fam

    # H3 allocation (equal partition): accuracy, decision rule with margins
    def acc(part, arm, a, s):
        return 100 * float(runs[(part, arm, a, s)]["final_accuracy"])

    for subset in ("all seeds", "seeds where allocations differ"):
        fam = []
        for a in L2["alphas"]:
            ss = [s for s in seeds if subset == "all seeds" or
                  runs[("equal", "full_fedhad", a, s)]["clients"]["E_arm"] != runs[("equal", "inv_epochs", a, s)]["clients"]["E_arm"]]
            if len(ss) < 5:
                continue
            d = np.array([acc("equal", "full_fedhad", a, s) - acc("equal", "inv_epochs", a, s) for s in ss])
            fam.append({"hypothesis": f"H3_allocation ({subset})", "alpha": a, "level": 2,
                        "statistic": "final accuracy Full - inverse epochs (pp)", **S.contrast_summary(d, delta=margins[a])})
        for t, ph in zip(fam, S.holm([t["p_wilcoxon"] for t in fam])):
            t["p_holm"] = ph
            t["category"] = S.classify(p_holm=ph, ci95=t["ci95"], ci90=t["ci90"], delta=t["delta"])
            t["verdict"] = VERDICT[t["category"]]
            t["p"], t["mean"] = t["p_wilcoxon"], t["mean_diff_pp"]
        tests += fam

    # H4 class-sensitive metrics
    def cls(part, arm, a, s, k):
        return 100 * float(runs[(part, arm, a, s)]["final_class"][k])

    for block, part, pairs, alphas in (
            ("paper partitions", "dirichlet", [("full_fedhad", "fedprox_default"), ("full_fedhad", "fedprox_tuned")], L1["alphas"]),
            ("equal partition", "equal", [("full_fedhad", "inv_epochs")], L2["alphas"])):
        fam = []
        for k in ("worst_class_recall", "macro_f1"):
            for A, B in pairs:
                for a in alphas:
                    if len(seeds) < 5:
                        continue
                    d = np.array([cls(part, A, a, s, k) - cls(part, B, a, s, k) for s in seeds])
                    fam.append({"hypothesis": f"H4_classes ({block})", "alpha": a, "metric": k,
                                "statistic": f"{k} {A} - {B} (pp)", **S.contrast_summary(d, delta=1.0)})
        for t, ph in zip(fam, S.holm([t["p_wilcoxon"] for t in fam])):
            t["p_holm"] = ph
            lo, hi = t["ci95"]
            t["verdict"] = ("A superior" if ph < 0.05 and lo > 0 else
                            "B superior" if ph < 0.05 and hi < 0 else "not detected")
            t["p"], t["mean"] = t["p_wilcoxon"], t["mean_diff_pp"]
            t.pop("category", None)
        tests += fam

    # descriptive accuracy table
    for (part, arm, a, s), r in runs.items():
        acc_rows.append({"partition": part, "arm": arm, "alpha": a, "seed": s,
                         "accuracy": 100 * float(r["final_accuracy"]),
                         "worst_class_recall": 100 * float(r["final_class"]["worst_class_recall"]),
                         "macro_f1": 100 * float(r["final_class"]["macro_f1"]),
                         "steps": r["totals"]["optimizer_steps"]})
    return tests, seed_rows, acc_rows


def fmt(x, n=3):
    return "—" if x is None else (f"{x:+.{n}f}" if isinstance(x, float) else str(x))


def report(tests, acc_rows):
    L = ["# Global-gradient diagnostic battery: result of the pre-registered tests", "",
         "Seed-level statistics (one value per seed); Wilcoxon two-sided on the 30 values; Holm within "
         "the pre-registered family. 'Supports the premise' requires the Holm-adjusted test and the "
         "whole 95% CI on the predicted side.", "",
         "| Hypothesis | alpha | statistic | n | mean | 95% CI | p | p_H | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for t in tests:
        ci = t.get("ci95") or (None, None)
        L.append(f"| {t['hypothesis']}{' / ' + t['metric'] if 'metric' in t else ''} | {t['alpha']:g} | "
                 f"{t['statistic']} | {t.get('n')} | {fmt(t.get('mean'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
                 f"{fmt(t.get('p'), 4)} | {fmt(t.get('p_holm'), 4)} | {t['verdict']} |")
        if "share_removed_epochs_increasing_others_loss" in t:
            L.append(f"| ↳ share of removed epochs that increase the others' loss (first order) | {t['alpha']:g} | "
                     f"| | {t['share_removed_epochs_increasing_others_loss']:.3f} | | | | descriptive |")
    L += ["", "## Accuracy, worst-class recall and macro-F1 (mean over seeds, %)", "",
          "| partition | arm | alpha | accuracy | worst class | macro-F1 | updates |", "|---|---|---|---|---|---|---|"]
    groups = {}
    for r in acc_rows:
        groups.setdefault((r["partition"], r["arm"], r["alpha"]), []).append(r)
    for (part, arm, a), rs in sorted(groups.items()):
        m = lambda k: np.mean([r[k] for r in rs])
        L.append(f"| {part} | {arm} | {a:g} | {m('accuracy'):.2f} | {m('worst_class_recall'):.2f} | "
                 f"{m('macro_f1'):.2f} | {m('steps'):.0f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="analyse the smoke runs (pipeline check only)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = C.OUT_SMOKE if a.smoke else C.OUT
    doc, runs, seeds = load(root, a.smoke)
    margins = ({float(k): float(v) for k, v in doc["equivalence_margins_pp"].items()} if doc else C.load_margins())
    tests, seed_rows, acc_rows = analyse(runs, seeds, margins)
    out = a.out or (root / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "tests.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for t in tests for k in t})
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        w.writerows([{k: (json.dumps(v) if isinstance(v, (list, tuple)) else v) for k, v in t.items()} for t in tests])
    for name, rows in (("seed_level.csv", seed_rows), ("accuracy.csv", acc_rows)):
        with (out / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    text = report(tests, acc_rows)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
