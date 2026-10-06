# -*- coding: utf-8 -*-
"""Unit tests of the pure functions used by the step-permutation and tuned-control experiments.
No dataset is loaded and no model is trained."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "algoritmos" / "tuned_controls_and_step_permutation"
sys.path.insert(0, str(CODE))
import r2_policy as P  # noqa: E402

REPO = HERE.parent
ABL = REPO / "results" / "component_analysis_ablation" / "raw" / "official"


class TestStepPermutation(unittest.TestCase):
    def test_sum_and_multiset_preserved_synthetic(self):
        rng = np.random.default_rng(0)
        for trial in range(2000):
            k = int(rng.integers(2, 9))
            n = [int(x) for x in rng.integers(0, 20000, k)]
            n[int(rng.integers(0, k))] = 0 if trial % 3 == 0 else n[0]   # some empty clients
            b = [x // 32 for x in n]
            e = [int(x) for x in rng.integers(2, 6, k)]
            tau = P.step_budgets(e, b)
            lr = [float(x) for x in rng.uniform(0.004, 0.01, k)]
            dom = P.permutation_domain(n, b)
            if len(dom) < 2:
                continue
            out = P.step_permuted_allocation(tau, n, b, lr, perm_seed=trial)
            self.assertEqual(sum(out["tau_received"]), sum(tau))
            self.assertEqual(sorted(out["tau_received"][i] for i in dom), sorted(tau[i] for i in dom))
            for i in range(k):
                if i not in dom:           # empty / zero-batch clients: never receive
                    self.assertEqual(out["tau_received"][i], 0)
                    self.assertNotIn(str(i), out["mapping_receiver_to_donor"])
            donors = set(out["mapping_receiver_to_donor"].values())
            self.assertTrue(donors <= set(dom))  # never transfer from outside the domain
            for r, d in out["mapping_receiver_to_donor"].items():
                self.assertNotEqual(int(r), d)   # derangement: no client keeps its own budget slot
            self.assertEqual(out["lr_received"], lr)  # lrclient: LR stays with the client

    def test_lr_follows_variant(self):
        n, b = [3200, 6400, 0, 9600], [100, 200, 0, 300]
        tau = P.step_budgets([3, 4, 5, 2], b)
        lr = [0.005, 0.006, 0.01, 0.004]
        out = P.step_permuted_allocation(tau, n, b, lr, perm_seed=7, lr_follows=True)
        for r, d in out["mapping_receiver_to_donor"].items():
            self.assertEqual(out["lr_received"][int(r)], lr[d])
        self.assertEqual(out["tau_received"][2], 0)
        self.assertEqual(out["lr_received"][2], lr[2])

    def test_empty_client_with_phantom_epochs_never_transfers(self):
        # The historical failure: an empty client scored H=0 receives E=5 ("phantom epochs").
        n, b = [11304, 0, 8059, 2109, 8903], [353, 0, 251, 65, 278]
        e_full = [3, 5, 3, 2, 3]
        tau = P.step_budgets(e_full, b)       # empty client -> tau = 5 * 0 = 0
        self.assertEqual(tau[1], 0)
        for seed in range(200):
            out = P.step_permuted_allocation(tau, n, b, [0.005] * 5, perm_seed=seed)
            self.assertEqual(out["tau_received"][1], 0)
            self.assertEqual(sum(out["tau_received"]), sum(tau))

    def test_reproducible_by_seed(self):
        n, b = [100 * 32, 50 * 32, 70 * 32], [100, 50, 70]
        tau = P.step_budgets([3, 4, 2], b)
        a = P.step_permuted_allocation(tau, n, b, [0.01] * 3, perm_seed=P.permutation_seed("CIFAR10", 0.01, 42))
        c = P.step_permuted_allocation(tau, n, b, [0.01] * 3, perm_seed=P.permutation_seed("CIFAR10", 0.01, 42))
        self.assertEqual(a, c)

    def test_outside_domain_with_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            P.step_permuted_allocation([10, 5], [0, 32], [0, 1], [0.01, 0.01], perm_seed=1)


class TestSeeds(unittest.TestCase):
    def test_client_round_seed_is_arm_independent_and_stable(self):
        s1 = P.client_round_seed(42, "3", 7)
        s2 = P.client_round_seed(42, 3, 7)
        self.assertEqual(s1, s2)
        self.assertNotEqual(P.client_round_seed(42, 3, 7), P.client_round_seed(42, 3, 8))
        self.assertNotEqual(P.client_round_seed(42, 3, 7), P.client_round_seed(43, 3, 7))
        self.assertTrue(0 <= s1 < 2 ** 31)

    def test_parse_seed_list(self):
        self.assertEqual(P.parse_seed_list("42-45,50"), [42, 43, 44, 45, 50])
        self.assertEqual(len(P.parse_seed_list("42-71")), 30)

    def test_leakage_guard(self):
        self.assertTrue(P.check_seed_separation([72, 73, 74, 75, 76], range(42, 72)))
        with self.assertRaises(ValueError):
            P.check_seed_separation([70, 72], range(42, 72))


class TestChecksum(unittest.TestCase):
    def test_state_dict_sha256_matches_ablation_convention(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        sd = {"b": torch.ones(2), "a": torch.zeros(3)}
        h1 = P.state_dict_sha256(sd)
        h2 = P.state_dict_sha256({"a": torch.zeros(3), "b": torch.ones(2)})
        self.assertEqual(h1, h2)  # key-order independent

    def test_checkpoint_generator_reproduces_component_analysis_init(self):
        """Builds (in memory, no training) the seed-42 CIFAR-10 initial model the way
        the runner's --make-checkpoints does and compares its SHA-256 with the
        initial_weights_sha256 recorded by the historical component analysis."""
        fp = ABL / "fingerprint__full_fedhad__CIFAR10__alpha-0.01__seed-42.json"
        if not fp.exists():
            self.skipTest("historical fingerprint not available")
        try:
            import random
            import torch
        except ImportError:
            self.skipTest("torch not available")
        sys.path.insert(0, str(REPO / "runners"))
        import run_tuned_controls_and_step_permutation as R
        net_cls = R._net_class_from_source("Net_CIFAR10")
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        sha = P.state_dict_sha256(net_cls().state_dict())
        self.assertEqual(sha, json.loads(fp.read_text())["initial_weights_sha256"])


if __name__ == "__main__":
    unittest.main()


class TestBuildPlanSynthetic(unittest.TestCase):
    """build_plan on synthetic DataLoaders (no dataset download, no training)."""

    def _loaders(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset, Subset
        base = TensorDataset(torch.zeros(2000, 1), torch.zeros(2000, dtype=torch.long))
        sizes = [640, 0, 320, 96, 20]            # includes an empty and a zero-batch client
        out, start = [], 0
        for n in sizes:
            sub = Subset(base, list(range(start, start + n))); start += n
            out.append(DataLoader(sub, batch_size=32, shuffle=n > 0, drop_last=n > 1) if n else DataLoader([], batch_size=32))
        return out

    def _plan(self, arm, **env):
        import os
        old = dict(os.environ)
        os.environ["FL_R2_ARM"] = arm
        os.environ.update({k: str(v) for k, v in env.items()})
        try:
            H = [0.3, 0.0, 0.6, 0.9, 1.0]
            ef = lambda h, b, m: 5 if h == 0 else max(m, int(round(b - 3.0 * h)))
            lf = lambda h, b, m: b if h == 0 else max(b / (1 + 1.5 * h), m)
            return P.build_plan(trainloaders=self._loaders(), h_values=H, seed=42, dataset="CIFAR10", alpha=0.01,
                                epochs_fn=ef, lr_fn=lf, base_epochs=5, min_epochs=2, base_lr=0.01, min_lr=0.001,
                                flops_per_sample_full=1.0)
        finally:
            os.environ.clear(); os.environ.update(old)

    def test_arms(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available")
        full = self._plan("full_fedhad")
        self.assertEqual([r["n_batches"] for r in full], [20, 0, 10, 3, 0])
        self.assertEqual([r["tau"] for r in full], [r["E_full"] * r["n_batches"] for r in full])
        perm = self._plan("step_permuted_lrclient")
        self.assertEqual(sum(r["tau"] for r in perm), sum(r["tau"] for r in full))
        self.assertEqual(perm[1]["tau"], 0); self.assertEqual(perm[4]["tau"], 0)
        self.assertEqual([r["lr"] for r in perm], [r["lr"] for r in full])
        st = self._plan("static", FL_R2_STATIC_LR=0.005, FL_R2_STATIC_EPOCHS=3)
        self.assertEqual([r["tau"] for r in st], [3 * b for b in [20, 0, 10, 3, 0]])
        self.assertTrue(all(r["lr"] == 0.005 for r in st))


class TestResumeLogic(unittest.TestCase):
    """Fabricated artifacts in a temporary directory; no training."""

    def test_status_transitions(self):
        import tempfile, csv as _csv
        sys.path.insert(0, str(REPO / "runners"))
        import run_tuned_controls_and_step_permutation as R
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ck = tmp / "ckpt"; (ck / "CIFAR10").mkdir(parents=True)
            (ck / "CIFAR10" / "seed-42.pt").write_bytes(b"x")
            (ck / "checkpoints_manifest.json").write_text(json.dumps({"42": {"sha256": "abc"}}))
            cell = R.Cell("step_permutation", "full_fedhad", "FedHAD", 0.01, 42, out_root=tmp / "out")
            camp = {"fingerprint": "F1"}
            self.assertEqual(R.status_of(cell, camp, ck)[0], R.STATUS_PENDING)
            R.write_json_atomic(cell.state_path, {"status": R.STATUS_RUNNING, "campaign_fingerprint": "F1"})
            self.assertEqual(R.status_of(cell, camp, ck), (R.STATUS_PENDING, "interrupted while running"))
            cell.run_dir.mkdir(parents=True)
            (cell.run_dir / "fingerprint.json").write_text(json.dumps(
                {"completed": True, "arm": "full_fedhad", "seed": 42, "alpha": 0.01,
                 "initial_checkpoint": {"sha256": "abc"}}))
            with (cell.run_dir / "telemetry.csv").open("w", newline="") as fh:
                w = _csv.writer(fh); w.writerow(["x"])
                for _ in range(R.C.NUM_ROUNDS * R.C.CLIENT_SETUP):
                    w.writerow([1])
            self.assertEqual(R.status_of(cell, camp, ck)[0], R.STATUS_COMPLETED)
            self.assertEqual(R.status_of(cell, {"fingerprint": "F2"}, ck)[0], R.STATUS_PENDING)  # code changed
            (ck / "checkpoints_manifest.json").write_text(json.dumps({"42": {"sha256": "zzz"}}))
            self.assertEqual(R.status_of(cell, camp, ck)[0], R.STATUS_PENDING)                  # ckpt changed


class TestRaySerializable(unittest.TestCase):
    """Functions shipped to the Ray workers must not name torch.backends.cudnn:
    CudnnModule cannot be pickled by Ray's cloudpickle (every client fit failed
    with 'cannot pickle CudnnModule object' in the first real launch)."""

    def test_worker_functions_do_not_reference_cudnn_attribute(self):
        import ast
        src = (CODE / "fedhad_r2.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        allowed = {"set_global_seed"}          # runs only in the driver process
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in allowed:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and sub.attr == "cudnn":
                        offenders.append(node.name)
        self.assertEqual(offenders, [])
