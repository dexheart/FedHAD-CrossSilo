# -*- coding: utf-8 -*-
"""Unit tests of the pure parts of the global-gradient battery (no training).
Run: env_flwr_pt/bin/python -m pytest sanity_tests/test_global_gradient_diagnostic.py"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1] / "algoritmos" / "global_gradient_diagnostic"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lr_uniform_control"))
sys.path.insert(0, str(HERE.parents[1] / "analysis" / "global_gradient_diagnostic"))
import gg_checks as K      # noqa: E402
import gg_diag as D        # noqa: E402
import gg_policy as P      # noqa: E402
import analyze_global_gradient as A  # noqa: E402


class EqualPartition(unittest.TestCase):
    def setUp(self):
        self.labels = np.repeat(np.arange(10), 500)          # 10 classes x 500

    def test_exact_sizes_disjoint_and_deterministic(self):
        for alpha in (0.05, 0.2, 1.0):
            parts = P.equal_size_label_skew(self.labels, 5, alpha, 800, seed=7)
            self.assertEqual([len(p) for p in parts], [800] * 5)
            flat = [i for p in parts for i in p]
            self.assertEqual(len(flat), len(set(flat)))
            self.assertEqual(parts, P.equal_size_label_skew(self.labels, 5, alpha, 800, seed=7))

    def test_exhausted_classes_are_renormalised(self):
        parts = P.equal_size_label_skew(self.labels, 10, 0.01, 500, seed=3)   # uses every sample
        self.assertEqual(sorted(i for p in parts for i in p), list(range(5000)))

    def test_too_many_samples_rejected(self):
        with self.assertRaises(ValueError):
            P.equal_size_label_skew(self.labels, 5, 0.5, 2000, seed=1)


class Policy(unittest.TestCase):
    def test_inverse_epochs(self):
        e = P.inverse_epochs([4, 3, 2, 4, 3], [0.1, 0.5, 0.9, 0.2, 0.6], [10] * 5)
        self.assertEqual(e, [2, 3, 4, 3, 4])                  # largest H -> largest E
        self.assertEqual(sorted(e), sorted([4, 3, 2, 4, 3]))

    def test_inverse_keeps_empty_client(self):
        self.assertEqual(P.inverse_epochs([5, 2, 4], [0.0, 1.0, 0.3], [0, 10, 10]), [5, 4, 2])

    def test_equal_sizes_preserve_weighted_steps(self):
        e_full, h, n, b = [4, 3, 3, 4, 2], [0.2, 0.5, 0.6, 0.3, 0.9], [4500] * 5, [140] * 5
        inv = P.inverse_epochs(e_full, h, n)
        self.assertAlmostEqual(P.weighted_steps(P.step_budgets(e_full, b), n),
                               P.weighted_steps(P.step_budgets(inv, b), n))

    def test_static_arms(self):
        e, lr = P.arm_policy("fedprox_default", e_full=[3, 4], lr_full=[0.005, 0.006],
                             h_values=[0.5, 0.3], n_samples=[1, 1], alpha=0.1)
        self.assertEqual((e, lr), ([5, 5], [0.01, 0.01]))
        e, lr = P.arm_policy("fedprox_tuned", e_full=[3, 4], lr_full=[0.005, 0.006],
                             h_values=[0.5, 0.3], n_samples=[1, 1], alpha=0.01)
        self.assertEqual((e, lr), ([2, 2], [0.005, 0.005]))


class Diagnostics(unittest.TestCase):
    def test_loo_identity(self):
        g = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0]), None]
        n = [10, 30, 0]
        gall, loo = D.loo_gradients(g, n)
        self.assertTrue(torch.allclose(gall, torch.tensor([0.25, 1.5])))
        self.assertTrue(torch.allclose(loo[0], g[1]) and torch.allclose(loo[1], g[0]))
        self.assertIsNone(loo[2])

    def test_full_gradient_matches_autograd(self):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        x, y = torch.randn(10, 3), torch.randint(0, 2, (10,))
        loader = [(x[:6], y[:6]), (x[6:], y[6:])]
        g, n = D.full_gradient(net, loader, "cpu")
        net.zero_grad(); torch.nn.functional.cross_entropy(net(x), y).backward()
        self.assertEqual(n, 10)
        self.assertTrue(torch.allclose(g, D.flat_grad(net), atol=1e-6))

    def test_class_metrics(self):
        m = D.class_metrics([[8, 2], [5, 5]])
        self.assertAlmostEqual(m["worst_class_recall"], 0.5)
        self.assertAlmostEqual(m["accuracy"], 13 / 20)

    def test_epoch_tracker_records_each_epoch(self):
        net = torch.nn.Linear(2, 1)
        tr = D.EpochTracker(net, torch.ones(3), torch.ones(3))
        with torch.no_grad():
            for p in net.parameters():
                p -= 0.1
        tr(1)
        m = tr.as_metrics()
        self.assertAlmostEqual(m["gg_marg_cos_loo_e1"], 1.0, places=5)   # moved along -g
        self.assertLess(m["gg_marg_lin_loo_e1"], 0.0)


class Analysis(unittest.TestCase):
    def test_removed_minus_kept(self):
        rows = [{"round": 1, "client_id": 0, "E_full": 3, "epoch": e, "marg_cos_loo": c, "marg_lin_loo": l, "H_k": 0.5}
                for e, c, l in ((1, 0.4, -1), (2, 0.4, -1), (3, 0.4, -1), (4, 0.0, 1), (5, -0.2, 1))]
        d, share = A.removed_minus_kept(rows)
        self.assertAlmostEqual(d, -0.5)
        self.assertAlmostEqual(share, 1.0)

    def test_pooled_size_adjusted_recovers_h_effect(self):
        rng = np.random.default_rng(0); by = {}
        for s in range(12):
            n = rng.integers(500, 20000, 5); h = rng.uniform(0.2, 1.0, 5)
            by[s] = [{"H_k": h[k], "n_k": n[k], "round": t, "grad_cos_loo": -0.5 * h[k] + 0.02 * np.log(n[k]) + 0.01 * t}
                     for k in range(5) for t in (1, 2)]
        r = A.pooled_size_adjusted(by, "grad_cos_loo", resamples=200)
        self.assertAlmostEqual(r["mean"], -0.5, places=6)
        self.assertLess(r["ci95"][1], 0)

    def test_rho_h(self):
        rows = [{"H_k": h, "grad_cos_loo": -h, "n_k": 10} for h in (0.1, 0.4, 0.7, 0.9)]
        self.assertAlmostEqual(A.rho_h(rows, "grad_cos_loo"), -1.0)


class Checks(unittest.TestCase):
    def test_missing_epoch_rows_fail(self):
        tel = [{"round": 1, "client_id": 0, "n_k": 5, "steps_executed": 10}]
        srv = [{"round": 1, "client_id": 0}]
        cls = [{"round": 0}, {"round": 1}]
        ok, _ = K.check_diagnostics(n_samples=[5], num_rounds=1, telemetry_rows=tel, server_rows=srv,
                                    epoch_rows=[], class_rows=cls)
        self.assertFalse(ok)
        ok, _ = K.check_diagnostics(n_samples=[5], num_rounds=1, telemetry_rows=tel, server_rows=srv,
                                    epoch_rows=[{"round": 1, "client_id": 0}], class_rows=cls)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
