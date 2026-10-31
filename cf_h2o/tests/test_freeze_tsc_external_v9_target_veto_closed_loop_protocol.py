import json
from pathlib import Path

from scripts.cluster.freeze_tsc_external_v9_target_veto_closed_loop_protocol import (
    _phase_reference_manifest,
)


def test_v67_reuses_exactly_32_v64_phase_references() -> None:
    parent = json.loads(
        Path(
            "cf_h2o/config/traffic_signal_tsc_v47_external_v9_long_horizon_target_veto.json"
        ).read_text(encoding="utf-8")
    )
    audit = json.loads(
        Path(
            "cf_h2o/results/cluster/tsc_v64r60_external_v9_hierarchical_closed_loop_development_20260810/development_selection_audit_v1.json"
        ).read_text(encoding="utf-8")
    )
    rows = _phase_reference_manifest(
        parent=parent,
        v64_audit=audit,
        v64_results_root=Path(
            "cf_h2o/results/cluster/tsc_v64r60_external_v9_hierarchical_closed_loop_development_20260810/formal_v1"
        ),
    )

    assert len(rows) == 32
    assert len({(row["city"], row["scenario"], row["seed"]) for row in rows}) == 32
    assert all(len(row["result"]["sha256"]) == 64 for row in rows)
    assert all(len(row["tripinfo"]["sha256"]) == 64 for row in rows)
