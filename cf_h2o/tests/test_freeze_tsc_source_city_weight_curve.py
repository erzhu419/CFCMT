from __future__ import annotations

from scripts.cluster.audit_tsc_source_city_component_screen import AUDIT_PROTOCOL
from scripts.cluster.freeze_tsc_source_city_weight_curve import build_weight_curve


def test_weight_curve_preserves_selected_source_and_target_fallback() -> None:
    payload = build_weight_curve(
        {
            "protocol": AUDIT_PROTOCOL,
            "scientific_status": "development-only-not-confirmation",
            "_sha256": "abc",
            "selected_weights_by_city": {
                "los_angeles": {},
                "jinan": {"hangzhou": 0.75},
            },
        }
    )

    assert payload["selected_source_by_city"] == {
        "los_angeles": None,
        "jinan": "hangzhou",
    }
    assert payload["candidates"][0]["key"] == "target_only"
    assert payload["candidates"][0]["weights_by_city"]["jinan"] == {}
    assert payload["candidates"][-1]["weights_by_city"]["jinan"] == {
        "hangzhou": 1.0
    }
    assert payload["candidates"][-1]["weights_by_city"]["los_angeles"] == {}
