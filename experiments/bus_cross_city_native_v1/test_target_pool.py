"""Verify the fixed target union and preserve each sampled native transition."""

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from target_pool import ROWS_PER_DAY, assemble_pool, selected_indices


class TargetPoolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        cases = [dict(case=i, trajectory_id=12201+i, target_rows=2048,
            split="fit" if i < 6 else "stop" if i == 6 else "calibrate",
            method="mixture", episode_seconds=24000, demand_seed=80100+i,
            sumo_seed=81100+i, action_seed=82100+i) for i in range(8)]
        self.protocol = dict(collection_cases=cases, planned_training=dict(
            target_unique_transition_budget=16384,
            target_splits=dict(fit=12288, stop=2048, calibrate=2048),
            world_model_fit_rows_within_fit=8192))

    def tearDown(self):
        self.temporary.cleanup()

    def write_day(self, case, count, horizon_seconds=None):
        out = self.root / "runs" / f"case{case['case']:02d}"
        out.mkdir(parents=True, exist_ok=True)
        raw = np.zeros((count, 245), np.float32)
        raw[:, 0] = np.arange(count) + 10000 * case["case"]
        following = raw.copy()
        following[:, 0] += 1
        terminal = np.zeros(count, np.int8)
        terminal[-1] = 1
        metadata = dict(case, config={}, station_max_by_line=[35], line_ids=["100X"],
            sumo_version="1.25.0", asset_manifest=str(out / "assets.json"), scheduled_service_date="2026-05-19")
        with h5py.File(out / "transitions.h5", "w") as data:
            arrays = dict(observations=raw, next_observations=following,
                actions=np.zeros((count, 1), np.float32), rewards=-np.arange(count, dtype=np.float64),
                discounts=np.where(terminal, 0, 1), terminals=terminal,
                delta_t_seconds=(np.arange(count) % 3 != 0).astype(np.float64),
                trajectory_local_id=np.zeros(count, np.int64),
                event_bus_id=np.full(count, b"100X_000001", dtype="S12"))
            for name, values in arrays.items():
                data.create_dataset(name, data=values, compression="gzip")
            for name, value in metadata.items():
                data.attrs[name] = json.dumps(value)
        (out / "summary.json").write_text(json.dumps(dict(complete=True, diagnostic=False,
            horizon_seconds=case["episode_seconds"] if horizon_seconds is None else horizon_seconds)))

    def assemble(self, out="pool"):
        return assemble_pool(self.protocol, self.root / "runs", self.root / out,
            lambda raw, metadata: raw[:, :229], "transit-smdp-v7")

    def test_fixed_union_disjoint_days_original_pairs_and_terminal_tail(self):
        for case in self.protocol["collection_cases"]:
            self.write_day(case, 2201)
        summary = self.assemble()
        self.assertEqual(summary["total_unique_rows"], 16384)
        # The fixture's zero-dt rows leave 8160 eligible fit labels: no extra days fill the cap.
        self.assertEqual(summary["world_model_fit_row_cap"], 8160)
        self.assertFalse(summary["final_test_included"])
        with h5py.File(self.root / "pool/target_pool.h5") as data:
            case = data["collection_case"][:]
            source = data["source_row_index"][:]
            self.assertEqual(len(set(zip(case, source))), 16384)
            for index, role in enumerate(("fit", "stop", "calibrate")):
                mask = data["split_id"][:] == index
                self.assertEqual(int(np.sum(mask)), self.protocol["planned_training"]["target_splits"][role])
                self.assertEqual(set(data["target_role"][:][mask]), {role.encode()})
            days = data["global_trajectory_id"][:]
            for day in np.unique(days):
                mask = days == day
                self.assertEqual(len(set(data["split_id"][:][mask])), 1)
                self.assertEqual(np.sum(data["terminals"][:][mask]), 1)
                self.assertEqual(source[mask][-1], 2200)
                self.assertEqual(int(np.sum(mask)), ROWS_PER_DAY)
            self.assertEqual(data["observations"].shape, (16384, 229))
            self.assertEqual(data["raw_observations"].shape, (16384, 245))
            expected = source + 10000 * case
            np.testing.assert_array_equal(data["raw_observations"][:, 0], expected)
            np.testing.assert_array_equal(data["raw_next_observations"][:, 0], expected + 1)
            np.testing.assert_array_equal(data["rewards"][:], -source)
            eligible = data["world_model_fit_eligible"][:]
            self.assertFalse(np.any(eligible & (data["delta_t_seconds"][:] == 0)))
            self.assertFalse(np.any(eligible & (data["terminals"][:] != 0)))
            self.assertFalse(np.any(eligible & (data["split_id"][:] != 0)))
            np.testing.assert_array_equal(data["policy_seed"][:], 82100+case)
            np.testing.assert_array_equal(data["demand_realization_seed"][:], 80100+case)
            self.assertEqual(data.attrs["schema_version"], "transit-smdp-v7")

    def test_insufficient_day_does_not_expand_cases_or_write_a_pool(self):
        for case in self.protocol["collection_cases"]:
            self.write_day(case, 2047 if case["case"] == 7 else 2048)
        summary = self.assemble()
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["decision"], "INSUFFICIENT")
        self.assertEqual(summary["actual_cases"], 8)
        self.assertFalse((self.root / "pool/target_pool.h5").exists())

    def test_selection_uses_positions_and_keeps_final_row(self):
        indices = selected_indices(3000)
        self.assertEqual(len(np.unique(indices)), 2048)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 2999)
        self.assertIsNone(selected_indices(2047))

    def test_actual_early_termination_is_rejected_despite_full_horizon_metadata(self):
        for case in self.protocol["collection_cases"]:
            self.write_day(case, 2201, horizon_seconds=17900 if case["case"] == 0 else 24000)
        with self.assertRaisesRegex(ValueError, "case00.*17900.*24000"):
            self.assemble()
        self.assertFalse((self.root / "pool/target_pool.h5").exists())


if __name__ == "__main__":
    unittest.main()
