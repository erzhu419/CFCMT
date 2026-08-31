from __future__ import annotations

from scripts.cluster.audit_tsc_source_city_component_screen import AUDIT_PROTOCOL
from scripts.cluster.freeze_tsc_source_city_guard_screen import build_guard_screen


def test_guard_screen_freezes_weight_curve_selection() -> None:
    payload = build_guard_screen(
        {
            "protocol": AUDIT_PROTOCOL,
            "scientific_status": "development-only-not-confirmation",
            "_sha256": "abc",
            "selected_weights_by_city": {
                "los_angeles": {},
                "jinan": {"atlanta": 1.0},
            },
        }
    )

    assert [row["key"] for row in payload["candidates"]] == [
        "target_only",
        "selected_source",
    ]
    assert payload["candidates"][0]["weights_by_city"] == {
        "los_angeles": {},
        "jinan": {},
    }
    assert payload["candidates"][1]["weights_by_city"] == {
        "los_angeles": {},
        "jinan": {"atlanta": 1.0},
    }
