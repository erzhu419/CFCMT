from pathlib import Path

from scripts.cluster.run_tsc_external_proposal_conditional_veto_successor import (
    validate_executable_freeze,
)


def test_v49_executable_freeze_matches_current_sources() -> None:
    result = validate_executable_freeze(
        Path(
            "cf_h2o/config/traffic_signal_tsc_v49_external_v9_proposal_conditional_veto_successor.json"
        )
    )

    assert result["gate"]["passed"] is True
    assert all(result["source_gate"].values())
