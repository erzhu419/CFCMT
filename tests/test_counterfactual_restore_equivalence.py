from __future__ import annotations

import numpy as np

from cf_h2o.eval.traffic_signal_counterfactual_restore_equivalence import (
    compare_counterfactual_implementations,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def _dataset(*, optimized: bool, target_offset: float = 0.0) -> MechanismDataset:
    features = np.asarray([[1.0], [2.0]], dtype=float)
    snapshot = {
        "reload_mode": "full_network_load",
        "branch_evaluation": "snapshot_full_reload_second_pass",
        **(
            {"halted_cost_readout": "libsumo_lane_subscription_v1"}
            if optimized
            else {}
        ),
    }
    return MechanismDataset(
        feature_names=("queue",),
        features=features,
        context_names=("lanes",),
        context=np.ones((2, 1), dtype=float),
        priors={"interval_cost": np.asarray([3.0, 4.0])},
        targets={"interval_cost": np.asarray([5.0, 6.0 + target_offset])},
        domains=np.asarray(["grid", "grid"]),
        metadata={
            "scenario": "grid",
            "seed": 2027,
            "duration_sec": 70.0,
            "control_interval_sec": 10,
            "warmup_sec": 60.0,
            "max_focal_tls": 1,
            "counterfactual_horizon_intervals": 45,
            "counterfactual_horizon_sec": 450,
            "action_group_ids": ["g0", "g0"],
            "row_tls": ["tls0", "tls0"],
            "row_times": [60.0, 60.0],
            "candidate_states": ["Gr", "rG"],
            "controllable_tls_ids": ["tls0"],
            "controllable_tls_count": 1,
            "covered_tls_ids": ["tls0"],
            "covered_tls_count": 1,
            "tls_coverage_fraction": 1.0,
            "counterfactual_branches": 2,
            "state_restores": 3,
            "state_snapshot_protocol": snapshot,
            "halted_cost_reader": (
                {
                    "protocol": "libsumo-lane-subscription-halted-cost-v1",
                    "aggregate_reads": 900,
                }
                if optimized
                else None
            ),
            "behavior_trace_sha256": "trace",
            "behavior_interval_count": 7,
            "captured_snapshot_count": 1,
            "protocol": "cache-v1",
            "strict_safety_monitoring": {"teleport": "fail"},
            "counterfactual_estimand": "pure-halted",
            "counterfactual_cost_mode": "halted_queue",
            "counterfactual_cost_scope": "network",
            "counterfactual_cost_population": "controlled_lanes",
            "counterfactual_cost_normalization": "mean",
            "mechanism_output_names": ["interval_cost"],
            "local_mechanism_horizon_sec": 10,
            "rollout_value_horizon_sec": 450,
            "rollout_prefix_horizons_sec": [450],
            "mechanism_estimands": {"interval_cost": "pure-halted"},
            "behavior_policy": "phase_pressure",
            "sumo_execution_protocol": {"api": "libsumo"},
            "phase_execution_audit": {"passed": True},
            "behavior_safety_audit": {"no_teleport_passed": True},
            "counterfactual_safety_audit": {
                "no_teleport_passed": True,
                "retained_branches": 2,
            },
            "counterfactual_replay_audit": {
                "checks": 1,
                "max_abs_difference": 0.0,
                "passed": True,
            },
        },
    )


def test_equivalence_authorizes_faster_semantically_identical_path() -> None:
    result = compare_counterfactual_implementations(
        _dataset(optimized=False),
        _dataset(optimized=True),
        legacy_elapsed_seconds=20.0,
        optimized_elapsed_seconds=5.0,
    )
    assert result["status"] == "PASS"
    assert result["gates"]["semantic_metadata"] is True
    assert result["runtime"]["speedup"] == 4.0


def test_equivalence_rejects_changed_counterfactual_label() -> None:
    result = compare_counterfactual_implementations(
        _dataset(optimized=False),
        _dataset(optimized=True, target_offset=0.1),
        legacy_elapsed_seconds=20.0,
        optimized_elapsed_seconds=5.0,
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["all_model_arrays"] is False
    assert result["array_comparisons"]["target__interval_cost"]["passed"] is False
