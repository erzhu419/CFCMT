import json
from pathlib import Path

from scripts.cluster.authorize_tsc_external_long_horizon_target_veto_cache import (
    AUDIT_DECISION,
    authorize_cache_collection,
)
from scripts.cluster.audit_tsc_external_hierarchical_closed_loop_development import (
    AUDIT_PROTOCOL as V64_AUDIT_PROTOCOL,
    REJECTION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_protocol import (
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import _sha256


def test_v65_authorization_rejects_existing_future_output(tmp_path: Path) -> None:
    v64 = tmp_path / "v64.json"
    v64.write_text(
        json.dumps(
            {
                "protocol": V64_AUDIT_PROTOCOL,
                "status": "PASS",
                "decision": REJECTION_DECISION,
                "integrity_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "v64_falsification_audit": {"sha256": _sha256(v64)},
                "long_horizon_collection": {
                    "duration_sec": 3600,
                    "counterfactual_horizon_intervals": 30,
                    "rollout_value_horizon_sec": 300,
                    "full_network_and_full_benchmark_horizon": True,
                    "seeds": [1, 2, 3, 4, 5, 6],
                    "scenario_count": 4,
                    "expected_cache_file_count": 384,
                },
                "target_veto_training": {
                    "classification": "target_simulator_labeled_few_shot_adaptation",
                    "zero_shot_claim_permitted": False,
                },
                "proposal": {"target_labels_used_by_proposal": False},
                "validation": {
                    "must_not_run_until_development_selector_and_analysis_hash_are_frozen": True
                },
                "prospective_confirmation": {
                    "must_not_run_until_validation_passes_and_prospective_analysis_is_frozen": True
                },
            }
        ),
        encoding="utf-8",
    )
    empty_cache = tmp_path / "cache"
    empty_training = tmp_path / "training"
    occupied = tmp_path / "closed_loop"
    occupied.mkdir()
    (occupied / "outcome.json").write_text("{}", encoding="utf-8")

    rejected = authorize_cache_collection(
        protocol_path=protocol,
        v64_audit_path=v64,
        future_cache_root=empty_cache,
        future_training_root=empty_training,
        future_closed_loop_root=occupied,
    )
    assert rejected["status"] == "FAIL"
    assert rejected["gate"]["future_outputs_absent"] is False

    (occupied / "outcome.json").unlink()
    accepted = authorize_cache_collection(
        protocol_path=protocol,
        v64_audit_path=v64,
        future_cache_root=empty_cache,
        future_training_root=empty_training,
        future_closed_loop_root=occupied,
    )
    assert accepted["status"] == "PASS"
    assert accepted["decision"] == AUDIT_DECISION
