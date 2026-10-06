#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of the LR-uniform control battery. Reads results/lr_uniform_control, applies
the pre-registered rule mechanically and writes, to results/lr_uniform_control/analysis/:

  runs_tidy.csv            one row per run
  contrasts.csv            every contrast, primary analysis and sensitivity analysis
  table_contrasts.tex/.md  table in the format of Table 15 of the paper, extended
  matching.csv             sum_k p_k eta_k tau_k matching between paired arms
  report.md                neutral report: category per alpha and the numbers behind it

It refuses to run if the pre-registration file does not match its recorded hash.
Usage:  python analysis/lr_uniform_control/analyze_lr_uniform_control.py [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2] / "algoritmos" / "lr_uniform_control"
sys.path.insert(0, str(HERE))
import config_lru as C      # noqa: E402
import lru_checks as K      # noqa: E402
import lru_prereg as PR     # noqa: E402
import lru_stats as S       # noqa: E402

REPO = C.REPO
ABL_RAW = REPO / "results" / "component_analysis_ablation" / "raw" / "official"
LABEL = {"full_fedhad": "Full", "u2": "U2", "lr_only_matched": "LR-only", "u1": "U1",
         "u2_arith": "U2-arith", "inv": "INV"}


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_prereg(out: Path):
    p, h = out / "preregistration.json", out / "preregistration.sha256"
    if not p.exists() or not h.exists():
        raise SystemExit(f"ERROR: no pre-registration in {out}; the battery has not started.")
    text = p.read_text(encoding="utf-8")
    if PR.sha256_text(text) != h.read_text().strip():
        raise SystemExit("ERROR: preregistration.json does not match its recorded hash. Refusing to analyse.")
    doc = json.loads(text)
    margins = {float(a): float(d) for a, d in doc["equivalence_margins_pp"].items()}
    return doc, margins


def expected_sha(seed):
    man = read_json(C.CKPT_DIR / "checkpoints_manifest.json") or {}
    return (man.get(str(seed)) or {}).get("sha256")


def load_runs(out: Path, code_fp: str):
    runs, invalid = {}, []
    for rp in sorted((out / "runs").glob("*/result.json")):
        res = read_json(rp)
        if not res:
            invalid.append((rp.parent.name, "unreadable")); continue
        ok, why = K.validate_result(res, arm=res.get("arm"), alpha=res.get("alpha"), seed=res.get("seed"),
                                    num_rounds=C.NUM_ROUNDS, expected_init_sha=expected_sha(res.get("seed")),
                                    code_fingerprint=code_fp)
        if not ok:
            invalid.append((rp.parent.name, why)); continue
        res["source"] = "this_battery"
        runs[(res["arm"], float(res["alpha"]), int(res["seed"]))] = res
    return runs, invalid


def load_section_610_reference(arm, alpha, seed):
    """Result-like record for full_fedhad / lr_only_matched from the Section 6.10 raw
    artifacts (used only when the pre-registration says reference_source=section_6_10_results).
    Those runs used unseeded worker trajectories; the pairing is on partition and
    initialization only."""
    tag = f"{arm}__CIFAR10__alpha-{alpha:g}__seed-{seed}"
    fp = read_json(ABL_RAW / f"fingerprint__{tag}.json")
    tel = ABL_RAW / f"telemetry__{tag}.csv"
    if fp is None or not tel.exists():
        return None
    rows = list(csv.DictReader(tel.open(encoding="utf-8")))
    n = fp["n_samples"]; tot = float(sum(n))
    last = max(int(r["round"]) for r in rows)
    acc = next(float(r["acc_centralized_round"]) for r in rows if int(r["round"]) == last)
    per = {}
    for r in rows:
        k = int(r["client_id"]); rnd = int(r["round"])
        per[rnd] = per.get(rnd, 0.0) + n[k] / tot * float(r["learning_rate"]) * float(r["minibatch_updates"])
    return {"arm": arm, "alpha": alpha, "seed": seed, "final_accuracy": acc, "source": "section_6_10",
            "sum_p_eta_tau_executed": float(np.mean(list(per.values()))),
            "clients": {"has_empty_client": any(int(x) == 0 for x in n), "n_samples": n},
            "init_sha256": fp.get("initial_weights_sha256"), "totals": {}, "eta_bar": {},
            "platform": {"device": fp.get("device")}, "elapsed_seconds": fp.get("elapsed_seconds"),
            "level": 1}


def tidy_rows(runs):
    rows = []
    for (arm, a, s), r in sorted(runs.items()):
        eb, tot, pl = r.get("eta_bar") or {}, r.get("totals") or {}, r.get("platform") or {}
        rows.append({
            "arm": arm, "alpha": a, "seed": s, "level": r.get("level"), "source": r.get("source"),
            "final_acc_pp": 100.0 * float(r["final_accuracy"]),
            "has_empty_client": r["clients"].get("has_empty_client"),
            "eta_bar_primary_p_tau": eb.get("eta_bar_primary_p_tau"), "eta_bar_tau_only": eb.get("eta_bar_tau_only"),
            "eta_bar_arithmetic": eb.get("eta_bar_arithmetic"),
            "sum_p_eta_tau_planned": r.get("sum_p_eta_tau_planned"),
            "sum_p_eta_tau_executed": r.get("sum_p_eta_tau_executed"),
            "optimizer_steps": tot.get("optimizer_steps"), "examples_processed": tot.get("examples_processed"),
            "flops_nominal": tot.get("flops_nominal"), "flops_executed": tot.get("flops_executed"),
            "elapsed_s": r.get("elapsed_seconds"), "init_sha256": r.get("init_sha256"),
            "gpu": pl.get("gpu"), "cpu": pl.get("cpu"),
        })
    return rows


def paired(runs, A, B, alpha, exclude_empty=False):
    seeds, d = [], []
    for s in C.SEEDS:
        ra, rb = runs.get((A, alpha, s)), runs.get((B, alpha, s))
        if not ra or not rb:
            continue
        if exclude_empty and (ra["clients"].get("has_empty_client") or rb["clients"].get("has_empty_client")):
            continue
        seeds.append(s); d.append(100.0 * (float(ra["final_accuracy"]) - float(rb["final_accuracy"])))
    return seeds, np.array(d)


def analyse(runs, margins, exclude_empty):
    out = []
    for a in C.ALPHAS:
        delta = margins[a]
        fams = {"primary": [C.PRIMARY], "secondary": C.SECONDARY}
        for fam, contrasts in fams.items():
            rows = []
            for A, B in contrasts:
                seeds, d = paired(runs, A, B, a, exclude_empty)
                row = {"analysis": "sensitivity_no_empty" if exclude_empty else "primary",
                       "family": fam, "alpha": a, "A": A, "B": B,
                       "contrast": f"{LABEL[A]} - {LABEL[B]}", "seeds": seeds}
                if len(d) >= 6:
                    row.update(S.contrast_summary(d, delta=delta))
                else:
                    row.update({"n": int(len(d)), "note": "fewer than 6 paired seeds; not computed"})
                rows.append(row)
            ps = [r["p_wilcoxon"] for r in rows if "p_wilcoxon" in r]
            adj = S.holm(ps) if ps else []
            it = iter(adj)
            for r in rows:
                if "p_wilcoxon" in r:
                    r["p_holm"] = next(it)
                    r["category"] = S.classify(p_holm=r["p_holm"], ci95=r["ci95"], ci90=r["ci90"], delta=r["delta"])
                    r["category_label"] = S.CATEGORY[r["category"]]
            out.extend(rows)
    return out


def matching(runs):
    rows = []
    for A, B in (("u2", "full_fedhad"), ("u1", "lr_only_matched")):
        for a in C.ALPHAS:
            rel = []
            for s in C.SEEDS:
                ra, rb = runs.get((A, a, s)), runs.get((B, a, s))
                if ra and rb and rb.get("sum_p_eta_tau_executed"):
                    rel.append((ra["sum_p_eta_tau_executed"] - rb["sum_p_eta_tau_executed"]) / rb["sum_p_eta_tau_executed"])
            if rel:
                rel = np.array(rel)
                rows.append({"arm": A, "reference": B, "alpha": a, "n": len(rel),
                             "mean_rel_diff": float(rel.mean()), "max_abs_rel_diff": float(np.abs(rel).max())})
    return rows


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_ci(ci):
    return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"


def write_tables(rows, adir: Path):
    hdr = ["Contrast", "alpha", "n", "Dacc (pp)", "95% CI", "90% CI", "d_z", "p", "p_H",
           "p TOST(-d)", "p TOST(+d)", "A/B/ties", "Category"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    tex = [r"\begin{tabular}{lrrrllrrrrrll}", r"\toprule",
           r"Contrast & $\alpha$ & $n$ & $\Delta$acc (pp) & 95\% CI & 90\% CI & $d_z$ & $p$ & $p_H$ & "
           r"$p_{-\delta}$ & $p_{+\delta}$ & A/B/ties & Category \\", r"\midrule"]
    for r in rows:
        if "p_wilcoxon" not in r:
            continue
        cells = [r["contrast"], f"{r['alpha']:g}", str(r["n"]), f"{r['mean_diff_pp']:+.2f}", fmt_ci(r["ci95"]),
                 fmt_ci(r["ci90"]), f"{r['d_z']:+.2f}", fmt_p(r["p_wilcoxon"]), fmt_p(r["p_holm"]),
                 fmt_p(r["p_tost_lower"]), fmt_p(r["p_tost_upper"]),
                 f"{r['n_favour_A']}/{r['n_favour_B']}/{r['n_ties']}", str(r["category"])]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(c.replace("<", "$<$") for c in cells).replace(" - ", r" $-$ ") + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(md), "\n".join(tex)


def report(prim, sens, match, margins, invalid, runs, doc):
    L = ["# LR-uniform control battery: result of the pre-registered rule", "",
         f"Pre-registration sha256: see preregistration.sha256. Reference source: {doc['reference_source']}.",
         f"Valid runs: {len(runs)}. Invalid or incomplete runs ignored: {len(invalid)}.", ""]
    for a in C.ALPHAS:
        L.append(f"## alpha = {a:g} (delta = {margins[a]:g} pp)")
        for r in prim:
            if r["alpha"] != a:
                continue
            tag = "Primary" if r["family"] == "primary" else "Secondary (supporting evidence)"
            if "p_wilcoxon" not in r:
                L.append(f"- {tag}, {r['contrast']}: n = {r['n']}; {r.get('note', '')}."); continue
            L.append(f"- {tag}, {r['contrast']}: n = {r['n']}, mean paired difference {r['mean_diff_pp']:+.2f} pp, "
                     f"95% CI {fmt_ci(r['ci95'])}, 90% CI {fmt_ci(r['ci90'])}, d_z {r['d_z']:+.2f}, "
                     f"Wilcoxon p {fmt_p(r['p_wilcoxon'])}, Holm p {fmt_p(r['p_holm'])}, "
                     f"one-sided p against -delta {fmt_p(r['p_tost_lower'])} and +delta {fmt_p(r['p_tost_upper'])}, "
                     f"seeds A/B/ties {r['n_favour_A']}/{r['n_favour_B']}/{r['n_ties']}. "
                     f"Category assigned: {r['category_label']}.")
            if r["category"] == 4:
                need = r["seeds_needed_for_equivalence"]
                L.append(f"  Inconclusive. 90% CI half-width {r['ci90_half_width']:.2f} pp; seeds for the 90% "
                         f"interval to fit within +/- delta by extrapolation from the SD of the differences: "
                         f"{need if need is not None else 'not attainable at the observed mean'}.")
        L.append("")
    L.append("## Sensitivity analysis (seeds with an empty client excluded)")
    for r in sens:
        if "p_wilcoxon" in r:
            L.append(f"- alpha {r['alpha']:g}, {r['contrast']}: n = {r['n']}, {r['mean_diff_pp']:+.2f} pp, "
                     f"95% CI {fmt_ci(r['ci95'])}, 90% CI {fmt_ci(r['ci90'])}, Holm p {fmt_p(r['p_holm'])}, "
                     f"category {r['category']}.")
        else:
            L.append(f"- alpha {r['alpha']:g}, {r['contrast']}: n = {r['n']}; {r.get('note', '')}.")
    L += ["", "## Matching of sum_k p_k eta_k tau_k (executed), relative to the reference arm"]
    for m in match:
        L.append(f"- {LABEL[m['arm']]} vs {LABEL[m['reference']]}, alpha {m['alpha']:g}: n = {m['n']}, "
                 f"mean relative difference {m['mean_rel_diff']:+.2e}, max |relative difference| {m['max_abs_rel_diff']:.2e}.")
    if invalid:
        L += ["", "## Runs ignored"] + [f"- {c}: {w}" for c, w in invalid]
    L += ["", "Levels of alpha are reported separately and not combined. Secondary contrasts are "
          "supporting evidence and do not replace the primary contrast."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=C.OUT)
    args = ap.parse_args(argv)
    doc, margins = load_prereg(args.out)
    runs, invalid = load_runs(args.out, doc["code_fingerprint"])
    if doc["reference_source"] == "section_6_10_results":
        for arm in C.REFERENCE_ARMS:
            for a in C.ALPHAS:
                for s in C.SEEDS:
                    if (arm, a, s) not in runs:
                        r = load_section_610_reference(arm, a, s)
                        if r:
                            runs[(arm, a, s)] = r
    adir = args.out / "analysis"; adir.mkdir(parents=True, exist_ok=True)
    tidy = tidy_rows(runs)
    if tidy:
        with (adir / "runs_tidy.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(tidy[0].keys())); w.writeheader(); w.writerows(tidy)
    prim = analyse(runs, margins, exclude_empty=False)
    sens = analyse(runs, margins, exclude_empty=True)
    allrows = prim + sens
    keys = sorted({k for r in allrows for k in r})
    with (adir / "contrasts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in allrows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, tuple)) else v) for k, v in r.items()})
    match = matching(runs)
    if match:
        with (adir / "matching.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(match[0].keys())); w.writeheader(); w.writerows(match)
    md, tex = write_tables(prim, adir)
    (adir / "table_contrasts.md").write_text(md + "\n", encoding="utf-8")
    (adir / "table_contrasts.tex").write_text(tex + "\n", encoding="utf-8")
    rep = report(prim, sens, match, margins, invalid, runs, doc)
    (adir / "report.md").write_text(rep, encoding="utf-8")
    print(rep); print(md)
    print(f"\nwritten to {adir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
