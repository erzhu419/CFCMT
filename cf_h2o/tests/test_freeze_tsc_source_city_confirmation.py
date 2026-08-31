from __future__ import annotations

from scripts.cluster.audit_tsc_source_city_component_screen import AUDIT_PROTOCOL
from scripts.cluster.audit_tsc_source_city_guard_screen import (
    AUDIT_PROTOCOL as GUARD_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_source_city_confirmation import (
    build_confirmation_freeze,
)


def test_confirmation_freeze_keeps_guards_matched_across_candidates() -> None:
    seeds = tuple(range(100, 156))
    base = {
        "protocol": AUDIT_PROTOCOL,
        "scientific_status": "development-only-not-confirmation",
        "_sha256": "parent",
    }
    guard = {
        "protocol": GUARD_AUDIT_PROTOCOL,
        "scientific_status": "development-only-not-confirmation",
        "_sha256": "guard",
        "selected_guard_by_city": {
            "los_angeles": {"guard_config": None},
            "jinan": {
                "guard_config": {
                    "risk_multiplier": 0.5,
                    "min_context_trust": 0.0,
                    "margin": 0.0,
                    "max_relative_rule_gap": 1.0,
                }
            },
        },
        "guard_selection_by_city": {
            "los_angeles": {"seeds": list(range(8))},
            "jinan": {"seeds": list(range(8))},
        },
    }
    payload = build_confirmation_freeze(
        identity_audit={
            **base,
            "selected_weights_by_city": {
                "los_angeles": {},
                "jinan": {"atlanta": 0.75},
            },
        },
        weight_audit={
            **base,
            "selected_weights_by_city": {
                "los_angeles": {},
                "jinan": {"atlanta": 1.0},
            },
        },
        guard_audit=guard,
        identity_candidates=(
            {
                "weights_by_city": {
                    "los_angeles": {"atlanta": 0.75},
                    "jinan": {"atlanta": 0.75},
                }
            },
            {
                "weights_by_city": {
                    "los_angeles": {"cologne": 0.75},
                    "jinan": {"cologne": 0.75},
                }
            },
        ),
        confirmation_seeds=seeds,
    )

    guards = [candidate["guard_by_city"] for candidate in payload["candidates"]]
    assert guards[0] == guards[1] == guards[2]
    assert payload["candidates"][-1]["weights_by_city"]["jinan"] == {
        "atlanta": 1.0
    }
    assert sum(
        payload["candidates"][1]["weights_by_city"]["jinan"].values()
    ) == 1.0
