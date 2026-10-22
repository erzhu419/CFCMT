"""Checks for paired accounting and the frozen effect/stability decision."""

import tempfile
from pathlib import Path
import unittest

from rule_common import HERE, P, case_spec, module, write
from collect_results import compare, load_reference, summarize, validate_day
from validate_runtime import compare_runtime


def day(case, cost=100., completion=.95, *, scene=None, seed=None):
    row = dict(case_spec(case), horizon_seconds=24000, sumo_version=[22, "SUMO 1.25.0"],
        deployment_hold_scale=1., split="test", generalized_seconds_per_departed_passenger=cost,
        passengers=dict(departed_passengers=100, completion_rate=completion, active_state_count_balance_error=0))
    if scene is not None:
        row.update(arm="fixed0250", scenario=scene, training_seed=seed, case=case+10, execution_actions={})
    return row


def paired_fixture():
    rules = {case: day(case) for case in range(10)}
    policies = {(scene, seed, case): day(case, 90., .96, scene=scene, seed=seed)
        for scene in P["scenarios"] for seed in P["training_seeds"] for case in range(10)}
    return policies, rules


class ConfirmationTests(unittest.TestCase):
    def test_case_mapping_and_existing_fixed_rule(self):
        self.assertEqual((case_spec(0)["demand_seed"], case_spec(9)["sumo_seed"], case_spec(9)["demand_scale"]), (68901, 69905, 1.25))
        rules = module("test_frozen_forward_rule", HERE / "fixed_rules.py")
        self.assertEqual(rules.hold_seconds("forward_proportional", 100, 400, 150, 150, True, True), 30.)
        self.assertEqual(rules.hold_seconds("forward_proportional", 0, 400, 150, 150, False, True), 0.)
        self.assertEqual(rules.hold_seconds("forward_proportional", 0, 400, 150, 150, True, True), 60.)

    def test_gate_keeps_all_seeds_and_discloses_a_regression(self):
        policies, rules = paired_fixture()
        for scene in P["scenarios"]:
            for case in range(10):
                policies[scene, P["training_seeds"][0], case]["generalized_seconds_per_departed_passenger"] = 101.
        result = compare(policies, rules)
        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(result["better_seeds"], 19)
        self.assertEqual(len(result["seed_deltas"]), 20)
        self.assertLess(result["macro_cost_delta_seconds"]["ci95"][1], 0)

    def test_completion_and_cell_regressions_stop_confirmation(self):
        policies, rules = paired_fixture()
        for seed in P["training_seeds"]:
            for case in range(5):
                policies["calibrated", seed, case]["passengers"]["completion_rate"] = .948
        result = compare(policies, rules)
        self.assertEqual(result["decision"], "STOP")
        self.assertFalse(result["conditions"]["cell_completion_preserved"])
        for seed in P["training_seeds"]:
            for case in range(5):
                policies["calibrated", seed, case]["generalized_seconds_per_departed_passenger"] = 101.
        self.assertFalse(compare(policies, rules)["conditions"]["no_cell_mean_cost_regression"])

    def test_uncertainty_cannot_be_hidden_by_a_good_mean(self):
        policies, rules = paired_fixture()
        for (scene, seed, case), row in policies.items():
            row["generalized_seconds_per_departed_passenger"] = 300. if seed == 63001 else 80.
        result = compare(policies, rules)
        self.assertTrue(result["conditions"]["mean_cost_improvement"])
        self.assertFalse(result["conditions"]["paired_ci_upper_below_zero"])
        self.assertEqual(result["decision"], "STOP")

    def test_wrong_pair_or_partial_horizon_is_rejected(self):
        row = day(5, scene="calibrated", seed=63001)
        validate_day(row, 5, scene="calibrated", seed=63001)
        row["sumo_seed"] = 69902
        with self.assertRaises(AssertionError):
            validate_day(row, 5, scene="calibrated", seed=63001)
        row = day(5, scene="calibrated", seed=63001)
        row["horizon_seconds"] = 1200
        with self.assertRaises(AssertionError):
            validate_day(row, 5, scene="calibrated", seed=63001)
        policies, rules = paired_fixture()
        policies["calibrated", 63001, 0]["passengers"]["departed_passengers"] = 99
        with self.assertRaises(AssertionError):
            compare(policies, rules)

    def test_missing_outputs_remain_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            result = summarize(root=Path(directory))
        self.assertFalse(result["complete"])
        self.assertEqual(result["decision"], "PENDING")
        self.assertEqual(result["missing_count"], 11)

    def test_runtime_change_stops_pairing_without_a_scientific_verdict(self):
        control = dict(day(0), method="daganzo_duet")
        row = dict(control, generalized_seconds_per_departed_passenger=control["generalized_seconds_per_departed_passenger"] + 1.)
        validation = compare_runtime(row, control)
        self.assertEqual(validation["decision"], "INVALID_RUNTIME")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "runtime_validation/results/validation.json", validation)
            result = summarize(root=root)
        self.assertEqual(result["decision"], "INVALID_RUNTIME")
        self.assertTrue(result["complete"])
        self.assertNotIn("primary", result)
        self.assertTrue(compare_runtime(control, control)["valid"])

    def test_reference_inventory_reads_only_its_own_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / P["reference_outputs"]
            write(reference / "completed_analysis.json", dict(complete=True, initialization_count=20,
                training_count=240, evaluation_count=4800, comparisons=[],
                protocol=dict(source_event_budget=25000, offline_updates=50000, continuation_updates=20000)))
            for scene in P["scenarios"]:
                for seed in P["training_seeds"]:
                    for case in range(10):
                        write(reference / "runs" / scene / f"fixed0250_seed{seed}" / "results" / f"eval_{case+10:02d}.json",
                            day(case, scene=scene, seed=seed))
            rows, _, missing = load_reference(root)
            self.assertEqual((len(rows), len(missing)), (600, 0))
            reference.rename(root / "unrelated_results")
            rows, _, missing = load_reference(root)
            self.assertEqual((len(rows), len(missing)), (0, 601))


if __name__ == "__main__":
    unittest.main()
