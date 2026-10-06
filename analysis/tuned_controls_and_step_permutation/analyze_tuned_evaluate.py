#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-registration and analysis of the DEFINITIVE evaluation of the tuned controls:
FedHAD vs a FedProx whose (lr, epochs) were tuned on separate validation seeds, and
vs the untuned FedProx default (lr=0.01, E=5).

    python analysis/tuned_controls_and_step_permutation/analyze_tuned_evaluate.py --preregister
        BEFORE any evaluate run: writes evaluate_preregistration.json + .sha256 with the
        frozen configuration, seed sets, margins, contrasts, decision rule and the
        interpretation of every outcome. Refuses to overwrite.

    python analysis/tuned_controls_and_step_permutation/analyze_tuned_evaluate.py
        AFTER the runs: applies the pre-registered rule mechanically. Refuses to run if
        the pre-registration, the frozen configuration or this file changed, or if any
        pre-registered cell is missing / failed. Writes to
        results/tuned_controls_fedprox_validation/evaluate_analysis/.

The statistics and the 4-category rule are the ones of the LR-uniform battery
(algoritmos/lr_uniform_control/lru_stats.py), so both batteries are read the same way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "algoritmos" / "lr_uniform_control"))
import lru_stats as S  # noqa: E402

STEP = REPO / "results" / "step_preserving_permutation"
TUNED = REPO / "results" / "tuned_controls_fedprox_validation"
FROZEN = TUNED / "frozen_config.json"
PREREG = TUNED / "evaluate_preregistration.json"
PREREG_SHA = TUNED / "evaluate_preregistration.sha256"
OUT = TUNED / "evaluate_analysis"
CKPT_MANIFEST = REPO / "results" / "initial_checkpoints_common" / "checkpoints_manifest.json"

# ---- pre-registered design (frozen into the pre-registration) --------------- #
SEEDS = {
    # alpha 0.1: paired SD of FedHAD - tuned FedProx on the 5 validation seeds ~1.0 pp
    #   -> 90% CI half-width ~0.30 pp with 30 seeds, well inside delta.
    0.1: list(range(42, 72)),
    # alpha 0.01: paired SD ~3.9 pp -> with delta = 1.0 a simulation of this rule gives
    #   60-74% inconclusive outcomes at 60 seeds and 22-40% at 120 (true difference
    #   0 to -1 pp). 120 seeds: 42-71 plus 77-166, the blocks after the validation seeds
    #   72-76, never used by any campaign.
    0.01: list(range(42, 72)) + list(range(77, 167)),
}
SENSITIVITY_SEEDS = {0.01: list(range(42, 72))}     # the paper's 30 seeds only
MARGINS_PP = {0.1: 0.75, 0.01: 1.0}                  # same deltas as the LR-uniform battery
ENDPOINT = "acc_centralized"                          # final-round centralized test accuracy
DEFAULT = {"lr": 0.01, "epochs": 5}

ARMS = {"fedhad": "FedHAD", "tuned": "FedProx tuned (budget_matched)", "default": "FedProx default (0.01, E=5)"}
PRIMARY = [("fedhad", "tuned")]
SECONDARY = [("fedhad", "default"), ("tuned", "default")]

INTERPRETATION = {
    "primary_1": "FedHAD more accurate than the tuned FedProx: the heterogeneity-driven policy "
                 "beats generic less-aggressive optimization; the accuracy claim survives tuning.",
    "primary_2": "Equivalent within +/- delta: FedHAD's accuracy is matched by tuning; the "
                 "contribution must be framed as automatic, tuning-free budget selection "
                 "(it reaches the tuned operating point without the validation grid). Compare "
                 "costs: if the tuned FedProx also uses fewer steps, FedHAD does not dominate it.",
    "primary_3": "Tuned FedProx more accurate: the reviewer's concern is confirmed; the accuracy "
                 "benefit is attributable to less aggressive optimization, not to heterogeneity "
                 "information; only the tuning-free argument (and its cost) remains.",
    "primary_4": "Inconclusive: report the CI and non-inferiority result; no accuracy claim "
                 "against tuned baselines may be made.",
    "non_inferiority": "Pre-specified secondary reading of the primary contrast: FedHAD is "
                       "non-inferior to the tuned FedProx at margin delta iff the lower end of the "
                       "90% CI (one-sided 95%) of FedHAD - tuned is > -delta.",
}


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canonical(doc) -> str:
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def frozen_by_alpha():
    fz = read_json(FROZEN)
    if not fz:
        sys.exit(f"ERROR: {FROZEN} missing; run run_tuned_controls_and_step_permutation.py --freeze budget_matched")
    out = {}
    for c in fz["configs"]:
        if c["method"] == "FedProx":
            out[float(c["alpha"])] = {"lr": float(c["lr"]), "epochs": int(c["epochs"])}
    if set(out) != set(SEEDS):
        sys.exit(f"ERROR: frozen config covers alphas {sorted(out)}, expected {sorted(SEEDS)}")
    return fz, out


def cell_dirs(alpha, seed, tuned):
    """Where the runner writes each arm (see run_tuned_controls_and_step_permutation.Cell)."""
    def static(lr, e):
        return TUNED / "evaluate" / "raw" / f"static_FedProx_lr{lr:g}_E{e}__CIFAR10__alpha-{alpha:g}__seed-{seed}"
    return {"fedhad": STEP / "raw" / f"full_fedhad__CIFAR10__alpha-{alpha:g}__seed-{seed}",
            "tuned": static(tuned["lr"], tuned["epochs"]),
            "default": static(DEFAULT["lr"], DEFAULT["epochs"])}


# --------------------------------------------------------------------------- #
# Pre-registration
# --------------------------------------------------------------------------- #
def build_prereg(fz, tuned):
    tune_dir = TUNED / "tune"
    return {
        # Title recorded verbatim in evaluate_preregistration.json; it must not change.
        "title": "R2 experiment 2 — definitive evaluation: FedHAD vs tuned FedProx",
        "question": ("Is FedHAD's final accuracy explained by heterogeneity-driven allocation, or is it "
                     "matched by a uniformly less aggressive FedProx (lower lr / fewer epochs) selected "
                     "on separate validation seeds?"),
        "substrate": "CIFAR-10, 5 clients, 10 rounds, FedProx mu=0.01, batch 32, common initial checkpoint per seed",
        "frozen_config": {str(a): v for a, v in tuned.items()},
        "frozen_config_sha256": sha_file(FROZEN),
        "selection": {"rule": fz.get("selection_rule"), "metric": fz.get("selection_metric"),
                      "validation_seeds": fz.get("validation_seeds"),
                      "validation_summary_sha256": sha_file(tune_dir / "validation_summary.csv"),
                      "selection_proposal_sha256": sha_file(tune_dir / "selection_proposal.json")},
        "default_baseline": DEFAULT,
        "evaluation_seeds": {str(a): s for a, s in SEEDS.items()},
        "sensitivity_seeds": {str(a): s for a, s in SENSITIVITY_SEEDS.items()},
        "endpoint": f"{ENDPOINT} at the last round, in percentage points; paired per seed (A - B)",
        "equivalence_margins_pp": {str(a): d for a, d in MARGINS_PP.items()},
        "contrasts": {"primary": [f"{a} - {b}" for a, b in PRIMARY],
                      "secondary": [f"{a} - {b}" for a, b in SECONDARY]},
        "multiplicity": ("Holm within each family: primary = the primary contrast at both alphas (2 tests); "
                         "secondary = the secondary contrasts at both alphas (4 tests)."),
        "tests": ("two-sided Wilcoxon signed-rank (zero_method=wilcox); percentile bootstrap CIs of the mean "
                  f"({S.BOOT_RESAMPLES} resamples, seed {S.BOOT_SEED}); TOST Wilcoxon against +/- delta reported"),
        "decision_rule": {
            "1": "p_holm < 0.05 and the whole 95% CI > 0  -> A superior",
            "3": "p_holm < 0.05 and the whole 95% CI < 0  -> B superior",
            "2": "otherwise, the whole 90% CI inside (-delta, +delta) -> practically equivalent",
            "4": "otherwise -> inconclusive",
        },
        "cost_report": ("executed optimizer steps, processed examples, executed and nominal TFLOPs per arm "
                        "(deterministic given the seed; reported, not tested), and the tuning cost: number of "
                        "validation runs and their executed steps"),
        "interpretation": INTERPRETATION,
        "completeness": "every pre-registered cell must be completed with the expected checkpoint SHA-256; "
                        "otherwise the analysis refuses to run (no partial analysis).",
        "analysis_code_sha256": sha_file(Path(__file__)),
        "stats_code_sha256": sha_file(REPO / "algoritmos" / "lr_uniform_control" / "lru_stats.py"),
    }


def preregister(late=False):
    fz, tuned = frozen_by_alpha()
    if PREREG.exists():
        sys.exit(f"ERROR: {PREREG} already exists; refusing to overwrite a pre-registration.")
    existing = sorted(d.name for a, seeds in SEEDS.items() for s in seeds
                      for k, d in cell_dirs(a, s, tuned[a]).items() if k != "fedhad" and d.exists())
    if existing and not late:
        sys.exit(f"ERROR: {len(existing)} evaluate runs already exist (e.g. {existing[0]}); "
                 "a pre-registration written after seeing results is not one. If they exist but were "
                 "not inspected, use --late to register them explicitly.")
    doc = build_prereg(fz, tuned)
    doc["registered_at"] = datetime.now().isoformat(timespec="seconds")
    # disclosed, never hidden: runs (finished or in progress) that existed when registering;
    # their results were not read by this function or by anyone before registration
    doc["evaluate_runs_existing_at_registration"] = existing
    text = canonical(doc)
    PREREG.write_text(text, encoding="utf-8")
    PREREG_SHA.write_text(sha_text(text) + "\n", encoding="utf-8")
    print(f"pre-registration written: {PREREG}\n  sha256 {sha_text(text)}")


def check_prereg(fz, tuned):
    if not PREREG.exists() or not PREREG_SHA.exists():
        sys.exit("ERROR: no pre-registration; run with --preregister BEFORE the evaluate runs.")
    text = PREREG.read_text(encoding="utf-8")
    if sha_text(text) != PREREG_SHA.read_text().strip():
        sys.exit("ERROR: evaluate_preregistration.json does not match its hash. Refusing to analyse.")
    doc = json.loads(text)
    now = build_prereg(fz, tuned)
    # This file was edited after registration ONLY to relocate paths (two repository
    # reorganisations). That edit is accepted solely when it is declared in
    # code_relocation.json with the registered and the current hash of this file and the
    # exact line diffs, which can be checked against the registered version in git.
    if doc.get("analysis_code_sha256") != now["analysis_code_sha256"]:
        rel = read_json(TUNED / "code_relocation.json") or {}
        if (rel.get("registered_sha256") != doc.get("analysis_code_sha256")
                or rel.get("current_sha256") != now["analysis_code_sha256"]):
            sys.exit("ERROR: 'analysis_code_sha256' differs from the pre-registration and the change is "
                     "not a declared path relocation. Refusing to analyse.")
        print(f"note: analysis code relocated after registration (paths only; see {TUNED / 'code_relocation.json'})")
        now["analysis_code_sha256"] = doc["analysis_code_sha256"]
    for k, v in now.items():
        if doc.get(k) != v:
            sys.exit(f"ERROR: '{k}' differs from the pre-registration (code, frozen config, seeds, margins "
                     "or selection changed after registering). Refusing to analyse.")
    return doc


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def load_cells(tuned):
    ck = read_json(CKPT_MANIFEST) or {}
    data, problems = {}, []
    for a, seeds in SEEDS.items():
        for s in seeds:
            expected = (ck.get(str(s)) or {}).get("sha256")
            for arm, d in cell_dirs(a, s, tuned[a]).items():
                fp = read_json(d / "fingerprint.json")
                if not fp or not fp.get("completed"):
                    problems.append(f"{d.name}: missing or incomplete"); continue
                if expected is None or fp["initial_checkpoint"]["sha256"] != expected:
                    problems.append(f"{d.name}: initial checkpoint SHA-256 does not match the manifest"); continue
                data[(a, s, arm)] = fp
            parts = {data[(a, s, arm)]["partition_hash_all"] for arm in ARMS if (a, s, arm) in data}
            if len(parts) > 1:
                problems.append(f"alpha={a:g} seed={s}: arms trained on different partitions")
    return data, problems


def diffs(data, a, seeds, A, B):
    return np.array([100 * (data[(a, s, A)]["final"][ENDPOINT] - data[(a, s, B)]["final"][ENDPOINT]) for s in seeds])


def contrast_rows(data, seed_sets, pairs, family):
    rows = []
    for a, seeds in seed_sets.items():
        for A, B in pairs:
            r = S.contrast_summary(diffs(data, a, seeds, A, B), delta=MARGINS_PP[a])
            r.update({"family": family, "alpha": a, "contrast": f"{ARMS[A]} - {ARMS[B]}"})
            r["non_inferior_A"] = bool(r["ci90"][0] > -MARGINS_PP[a])
            rows.append(r)
    for r, p in zip(rows, S.holm([r["p_wilcoxon"] for r in rows])):
        r["p_holm"] = p
        r["category"] = S.classify(p_holm=p, ci95=r["ci95"], ci90=r["ci90"], delta=r["delta"])
    return rows


def cost_rows(data):
    rows = []
    for a, seeds in SEEDS.items():
        for arm in ARMS:
            t = [data[(a, s, arm)]["totals"] for s in seeds]
            acc = [100 * data[(a, s, arm)]["final"][ENDPOINT] for s in seeds]
            rows.append({"alpha": a, "arm": ARMS[arm], "n": len(seeds),
                         "acc_mean_pp": float(np.mean(acc)), "acc_sd_pp": float(np.std(acc, ddof=1)),
                         "steps_mean": float(np.mean([x["optimizer_steps"] for x in t])),
                         "examples_mean": float(np.mean([x["examples_processed"] for x in t])),
                         "tflops_exec_mean": float(np.mean([x["flops_executed"] for x in t])) / 1e12,
                         "tflops_nominal_mean": float(np.mean([x["flops_nominal"] for x in t])) / 1e12})
        ref = next(r for r in rows if r["alpha"] == a and r["arm"] == ARMS["fedhad"])
        for r in rows:
            if r["alpha"] == a:
                r["steps_vs_fedhad"] = r["steps_mean"] / ref["steps_mean"]
                r["tflops_exec_vs_fedhad"] = r["tflops_exec_mean"] / ref["tflops_exec_mean"]
    return rows


def tuning_cost():
    import csv
    with (TUNED / "tune" / "manifest.csv").open() as fh:
        m = [r for r in csv.DictReader(fh) if r["status"] == "completed"]
    out = {}
    for a in SEEDS:
        grid = [r for r in m if float(r["alpha"]) == a and r["arm"] == "static"]
        fh_ = [r for r in m if float(r["alpha"]) == a and r["arm"] == "full_fedhad"]
        out[a] = {"grid_runs": len(grid), "grid_steps_total": sum(float(r["total_steps"]) for r in grid),
                  "fedhad_steps_per_run": float(np.mean([float(r["total_steps"]) for r in fh_]))}
    return out


def fmt(x):
    return f"{x:+.2f}"


def write_report(prim, sec, sens, costs, tcost, doc):
    cat = {1: "A superior", 2: "equivalente (±δ)", 3: "B superior", 4: "inconclusivo"}
    L = ["# Experimento 2 — avaliação definitiva (pré-registrada)", "",
         f"Pré-registro: `{os.path.relpath(PREREG, REPO)}` (registrado em {doc['registered_at']}). "
         "Regra aplicada mecanicamente; nada abaixo foi escolhido depois de ver os dados.", "",
         "## Acurácia e custo por braço", "",
         "| α | braço | n | acurácia (%) | DP | passos | passos/FedHAD | TFLOPs exec. | TFLOPs/FedHAD |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in costs:
        L.append(f"| {r['alpha']:g} | {r['arm']} | {r['n']} | {r['acc_mean_pp']:.2f} | {r['acc_sd_pp']:.2f} | "
                 f"{r['steps_mean']:.0f} | {r['steps_vs_fedhad']:.2f} | {r['tflops_exec_mean']:.1f} | {r['tflops_exec_vs_fedhad']:.2f} |")

    def table(rows, title):
        L.extend(["", f"## {title}", "",
                  "| α | contraste (A − B) | n | média (pp) | IC95% | IC90% | δ | A/B vitórias | p | p Holm | não-inferior A | categoria |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"])
        for r in rows:
            L.append(f"| {r['alpha']:g} | {r['contrast']} | {r['n']} | {fmt(r['mean_diff_pp'])} | "
                     f"[{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}] | [{fmt(r['ci90'][0])}, {fmt(r['ci90'][1])}] | "
                     f"{r['delta']:g} | {r['n_favour_A']}/{r['n_favour_B']} | {r['p_wilcoxon']:.4f} | {r['p_holm']:.4f} | "
                     f"{'sim' if r['non_inferior_A'] else 'não'} | **{r['category']}: {cat[r['category']]}** |")
    table(prim, "Contraste primário (Holm sobre os 2 alphas)")
    table(sec, "Contrastes secundários (Holm sobre os 4)")
    table(sens, "Sensibilidade: α=0,01 só com as 30 sementes do artigo (42–71)")
    L += ["", "## Custo do ajuste (fase de validação)", ""]
    for a, t in tcost.items():
        L.append(f"- α={a:g}: {t['grid_runs']} execuções de grade, {t['grid_steps_total']:.0f} passos no total "
                 f"= {t['grid_steps_total'] / t['fedhad_steps_per_run']:.0f}× uma execução do FedHAD.")
    L += ["", "## Leitura pré-registrada", ""]
    for r in prim:
        L.append(f"- α={r['alpha']:g}: categoria {r['category']} → {INTERPRETATION['primary_' + str(r['category'])]}")
        L.append(f"  Não-inferioridade (margem {r['delta']:g} pp): {'satisfeita' if r['non_inferior_A'] else 'não satisfeita'}.")
    return "\n".join(L) + "\n"


def analyse():
    fz, tuned = frozen_by_alpha()
    doc = check_prereg(fz, tuned)
    data, problems = load_cells(tuned)
    if problems:
        print(f"ERROR: {len(problems)} pre-registered cells are not usable; no partial analysis:")
        for p in problems[:20]:
            print("   ", p)
        sys.exit(1)
    prim = contrast_rows(data, SEEDS, PRIMARY, "primary")
    sec = contrast_rows(data, SEEDS, SECONDARY, "secondary")
    sens = contrast_rows(data, SENSITIVITY_SEEDS, PRIMARY, "sensitivity")
    costs, tcost = cost_rows(data), tuning_cost()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {"primary": prim, "secondary": sec, "sensitivity": sens, "costs": costs,
         "tuning_cost": {str(k): v for k, v in tcost.items()}}, indent=2, default=float), encoding="utf-8")
    per_seed = [{"alpha": a, "seed": s, **{f"acc_{arm}": 100 * data[(a, s, arm)]["final"][ENDPOINT] for arm in ARMS},
                 **{f"steps_{arm}": data[(a, s, arm)]["totals"]["optimizer_steps"] for arm in ARMS}}
                for a, seeds in SEEDS.items() for s in seeds]
    import csv
    with (OUT / "per_seed.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed[0])); w.writeheader(); w.writerows(per_seed)
    text = write_report(prim, sec, sens, costs, tcost, doc)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preregister", action="store_true", help="write the pre-registration (before the runs)")
    ap.add_argument("--late", action="store_true", help="with --preregister: allow and disclose runs that already exist")
    a = ap.parse_args()
    if a.preregister:
        preregister(late=a.late)
    else:
        analyse()


if __name__ == "__main__":
    main()
