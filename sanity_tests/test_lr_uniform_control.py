# -*- coding: utf-8 -*-
"""Unit tests of the pure functions of the LR-uniform control battery.
No dataset is loaded and no model is trained (synthetic tensors only)."""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "algoritmos" / "lr_uniform_control"))
sys.path.insert(0, str(REPO / "algoritmos" / "tuned_controls_and_step_permutation"))
sys.path.insert(0, str(REPO / "algoritmos" / "component_analysis_ablation"))
import lru_checks as K   # noqa: E402
import lru_policy as P   # noqa: E402
import lru_stats as S    # noqa: E402

ef = lambda h, b, m: b if h == 0 else max(m, int(round(b - 3.0 * h)))        # FedHAD epochs
lf = lambda h, b, m: b if h == 0 else max(b / (1 + 1.5 * h), m)             # FedHAD LR


class TestEtaBar(unittest.TestCase):
    lr = [0.008, 0.005, 0.01, 0.004]
    tau = [300, 100, 0, 50]            # client 2 is empty
    n = [9600, 3200, 0, 1600]

    def test_primary_matches_displacement(self):
        eb = P.eta_bar_weighted(self.lr, self.tau, self.n)
        p = [x / sum(self.n) for x in self.n]
        num = sum(p[k] * self.lr[k] * self.tau[k] for k in (0, 1, 3))
        den = sum(p[k] * self.tau[k] for k in (0, 1, 3))
        self.assertAlmostEqual(eb, num / den, places=15)
        # a uniform rate eta_bar reproduces the aggregate first-order displacement exactly
        self.assertAlmostEqual(P.aggregate_displacement([eb] * 4, self.tau, self.n),
                               P.aggregate_displacement(self.lr, self.tau, self.n), places=12)

    def test_tau_only_and_arithmetic(self):
        self.assertAlmostEqual(P.eta_bar_tau(self.lr, self.tau, self.n),
                               (0.008 * 300 + 0.005 * 100 + 0.004 * 50) / 450, places=15)
        # empty client (lr 0.01) excluded from the arithmetic mean
        self.assertAlmostEqual(P.eta_bar_arith(self.lr, self.n), (0.008 + 0.005 + 0.004) / 3, places=15)

    def test_all_empty_raises(self):
        with self.assertRaises(ValueError):
            P.eta_bar_weighted([0.01], [0], [0])


class TestInverted(unittest.TestCase):
    def test_inverse_order_and_empty_client(self):
        H = [0.2, 0.8, 0.0, 0.5]          # client 2 empty (H=0, phantom LR 0.01)
        n = [100, 50, 0, 70]
        lr = [lf(h, 0.01, 0.001) for h in H]
        inv = P.inverted_lr(lr, H, n)
        self.assertEqual(sorted(inv[k] for k in (0, 1, 3)), sorted(lr[k] for k in (0, 1, 3)))  # same multiset
        self.assertEqual(inv[1], max(lr[k] for k in (0, 1, 3)))    # largest H -> largest LR
        self.assertEqual(inv[0], min(lr[k] for k in (0, 1, 3)))    # smallest H -> smallest LR
        self.assertEqual(inv[2], lr[2])                              # empty client keeps its own rate

    def test_ties_broken_by_index(self):
        H = [0.5, 0.5, 0.9]
        lr = [0.006, 0.004, 0.003]
        inv = P.inverted_lr(lr, H, [10, 10, 10])
        # ascending (H, index): client0, client1, client2 get ascending rates 0.003, 0.004, 0.006
        self.assertEqual(inv, [0.003, 0.004, 0.006])


def _loaders(sizes):
    import torch
    from torch.utils.data import DataLoader, Subset, TensorDataset
    base = TensorDataset(torch.zeros(sum(sizes) + 10, 1), torch.zeros(sum(sizes) + 10, dtype=torch.long))
    out, start = [], 0
    for n in sizes:
        if n == 0:
            out.append(DataLoader([], batch_size=32)); continue
        out.append(DataLoader(Subset(base, list(range(start, start + n))), batch_size=32,
                              shuffle=True, drop_last=n > 1))
        start += n
    return out


class TestFixedAllocationReuse(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available")
        self.sizes = [3200, 0, 1600, 640, 960]
        self.H = [0.2, 0.0, 0.5, 0.9, 0.7]
        self.kw = dict(seed=42, base_epochs=5, min_epochs=2, base_lr=0.01, epochs_fn=ef,
                       lr_fn=lf, min_lr=0.001, flops_per_sample_full=1.0)

    def test_same_as_section_610(self):
        import ablation_policy
        ref = ablation_policy.build_allocation(trainloaders=_loaders(self.sizes), h_values=self.H, **self.kw)
        ours = P.fixed_allocation(trainloaders=_loaders(self.sizes), h_values=self.H, **self.kw)
        self.assertEqual(ours["E_fixed"], ref["E_fixed"])
        self.assertEqual(ours["E_full"], ref["E_full"])
        self.assertEqual(ours["LR_full"], ref["LR_full"])
        nb = [len(dl) for dl in _loaders(self.sizes)]
        e, lr = P.arm_policy("u1", e_full=ours["E_full"], e_fixed=ours["E_fixed"], lr_full=ours["LR_full"],
                             h_values=self.H, n_samples=self.sizes, n_batches=nb)
        self.assertEqual(e, ref["E_fixed"])
        self.assertEqual(len(set(lr)), 1)
        e2, lr2 = P.arm_policy("lr_only_matched", e_full=ours["E_full"], e_fixed=ours["E_fixed"],
                               lr_full=ours["LR_full"], h_values=self.H, n_samples=self.sizes, n_batches=nb)
        self.assertEqual((e2, lr2), (ref["E_fixed"], ref["LR_full"]))

    def test_build_plan_u2_and_u1_match_their_references(self):
        import r2_policy
        disp = {}
        for arm in ("full_fedhad", "u2", "lr_only_matched", "u1"):
            table = P.build_plan(arm=arm, trainloaders=_loaders(self.sizes), h_values=self.H,
                                 dataset="CIFAR10", alpha=0.01, **self.kw)
            disp[arm] = r2_policy._STATE["planned_sum_p_eta_tau"]
            self.assertEqual(r2_policy._STATE["arm"], arm)
            self.assertEqual([row["tau"] for row in table], r2_policy._STATE["tau_received"])
            self.assertEqual(table[1]["tau"], 0)                              # empty client never trains
        self.assertAlmostEqual(disp["u2"], disp["full_fedhad"], places=12)
        self.assertAlmostEqual(disp["u1"], disp["lr_only_matched"], places=12)


def _synthetic_result(n_samples=(100, 0, 50), rounds=3, drop=None, sha="abc"):
    fp = {"n_samples": list(n_samples), "n_batches": [3, 0, 1], "H": [0.1, 0.0, 0.6],
          "E_arm": [5, 5, 3], "LR_arm": [0.007] * 3, "tau_received": [15, 0, 3],
          "initial_checkpoint": {"sha256": sha}, "final": {"acc_centralized": 0.5, "loss_centralized": 1.2},
          "totals": {"optimizer_steps": 54}, "planned_sum_p_eta_tau": 0.1, "hardware": {"gpu": "x"},
          "elapsed_seconds": 10.0, "eta_bar_primary_p_tau": 0.007}
    rows = []
    for r in range(1, rounds + 1):
        for k, n in enumerate(n_samples):
            if drop == (r, k):
                continue
            rows.append({"round": r, "client_id": k, "n_k": n, "learning_rate": 0.007,
                         "steps_executed": [15, 0, 3][k], "acc_centralized_round": 0.4, "loss_centralized_round": 1.3})
    cell = {"arm": "u2", "alpha": 0.01, "seed": 42, "num_rounds": rounds, "level": 1}
    return K.assemble_result(fingerprint=fp, telemetry_rows=rows, cell=cell, code_fingerprint="CF",
                             platform_info={"cpu": "c"}, started_at="t0", finished_at="t1")


class TestCompleteness(unittest.TestCase):
    kw = dict(arm="u2", alpha=0.01, seed=42, num_rounds=3, expected_init_sha="abc", code_fingerprint="CF")

    def test_complete(self):
        self.assertEqual(K.validate_result(_synthetic_result(), **self.kw), (True, ""))

    def test_empty_client_absent_is_fine(self):
        res = _synthetic_result(drop=(2, 1))                 # client 1 has no data
        self.assertTrue(K.validate_result(res, **self.kw)[0])

    def test_client_with_data_missing_in_a_round(self):
        ok, why = K.validate_result(_synthetic_result(drop=(2, 2)), **self.kw)
        self.assertFalse(ok); self.assertIn("clients with data missing", why)

    def test_missing_round(self):
        ok, why = K.validate_result(_synthetic_result(rounds=2), **self.kw)
        self.assertFalse(ok)

    def test_checksum_and_code_and_fields(self):
        self.assertFalse(K.validate_result(_synthetic_result(sha="zzz"), **self.kw)[0])
        self.assertFalse(K.validate_result(_synthetic_result(), **{**self.kw, "code_fingerprint": "OTHER"})[0])
        res = _synthetic_result(); res.pop("platform")
        self.assertFalse(K.validate_result(res, **self.kw)[0])
        self.assertFalse(K.validate_result(None, **self.kw)[0])

    def test_executed_displacement(self):
        res = _synthetic_result()
        self.assertAlmostEqual(res["sum_p_eta_tau_executed"], (100 / 150) * 0.007 * 15 + (50 / 150) * 0.007 * 3)


class TestClassification(unittest.TestCase):
    rng = np.random.default_rng(0)

    def _cat(self, d, delta=1.0):
        s = S.contrast_summary(np.asarray(d), delta=delta)
        p_h = S.holm([s["p_wilcoxon"]])[0]
        return S.classify(p_holm=p_h, ci95=s["ci95"], ci90=s["ci90"], delta=delta), s

    def test_category_1_superior(self):
        self.assertEqual(self._cat(2.0 + self.rng.normal(0, 0.5, 30))[0], 1)

    def test_category_3_uniform_superior(self):
        self.assertEqual(self._cat(-2.0 + self.rng.normal(0, 0.5, 30))[0], 3)

    def test_category_2_equivalent(self):
        cat, s = self._cat(self.rng.normal(0, 0.3, 30))
        self.assertEqual(cat, 2)
        self.assertLess(s["p_tost_lower"], 0.05); self.assertLess(s["p_tost_upper"], 0.05)

    def test_category_4_inconclusive(self):
        cat, s = self._cat(self.rng.normal(0.3, 3.0, 30))
        self.assertEqual(cat, 4)
        self.assertGreater(s["ci90_half_width"], 0)
        self.assertTrue(s["seeds_needed_for_equivalence"] is None or s["seeds_needed_for_equivalence"] > 30)

    def test_rule_is_mechanical(self):
        self.assertEqual(S.classify(p_holm=0.01, ci95=(0.1, 1.0), ci90=(0.2, 0.9), delta=1.0), 1)
        self.assertEqual(S.classify(p_holm=0.01, ci95=(-1.0, -0.1), ci90=(-0.9, -0.2), delta=1.0), 3)
        # significant but the 95% CI crosses zero -> not 1/3; inside the margin -> 2
        self.assertEqual(S.classify(p_holm=0.04, ci95=(-0.05, 0.6), ci90=(0.0, 0.5), delta=1.0), 2)
        self.assertEqual(S.classify(p_holm=0.5, ci95=(-2, 2), ci90=(-1.5, 1.5), delta=1.0), 4)

    def test_holm_and_seeds_needed(self):
        self.assertEqual(S.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertIsNone(S.seeds_needed(np.array([1.5, 1.6, 1.4, 1.5]), 1.0))   # |mean| >= delta

    def test_bootstrap_is_reproducible(self):
        d = self.rng.normal(0, 1, 30)
        self.assertEqual(S.bootstrap_ci(d, 0.95), S.bootstrap_ci(d, 0.95))


if __name__ == "__main__":
    unittest.main()
