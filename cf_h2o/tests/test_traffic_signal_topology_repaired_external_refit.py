from __future__ import annotations

import pytest

from cf_h2o.eval.traffic_signal_topology_context_repair import REPAIR_PROTOCOL
from cf_h2o.eval.traffic_signal_topology_repaired_external_refit import (
    validate_repaired_cache_contract,
)


def _repair_audit(scenarios: tuple[str, ...], files_per_scenario: int):
    return {
        "protocol": REPAIR_PROTOCOL,
        "file_count": len(scenarios) * files_per_scenario,
        "scenarios": {
            scenario: {"file_count": files_per_scenario}
            for scenario in scenarios
        },
    }


def test_repaired_cache_contract_accepts_exact_bank() -> None:
    scenarios = ("one", "two")
    repair = _repair_audit(scenarios, files_per_scenario=6)

    validate_repaired_cache_contract(
        label="source",
        repair_audit=repair,
        manifest_scenarios=scenarios,
        seed_count=2,
        collection_shards=3,
        cache_audit={
            "file_count": 12,
            "used_file_count": 12,
            "scenario_count": 2,
            "unique_identities": 12,
        },
    )


def test_repaired_cache_contract_rejects_manifest_that_silently_drops_cache() -> None:
    repair = _repair_audit(("one", "two"), files_per_scenario=6)

    with pytest.raises(ValueError, match="repair/manifest scenario mismatch"):
        validate_repaired_cache_contract(
            label="source",
            repair_audit=repair,
            manifest_scenarios=("one",),
            seed_count=2,
            collection_shards=3,
        )


def test_repaired_cache_contract_rejects_unused_files() -> None:
    repair = _repair_audit(("one", "two"), files_per_scenario=6)

    with pytest.raises(ValueError, match="loaded-cache contract mismatch"):
        validate_repaired_cache_contract(
            label="source",
            repair_audit=repair,
            manifest_scenarios=("one", "two"),
            seed_count=2,
            collection_shards=3,
            cache_audit={
                "file_count": 12,
                "used_file_count": 6,
                "scenario_count": 1,
                "unique_identities": 6,
            },
        )
