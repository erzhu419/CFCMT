from __future__ import annotations

from scripts.cluster.audit_tsc_source_city_guard_screen import (
    select_guard_profile_from_closed_loop,
)


def _row(profile: str, seed: int, waiting: float, collisions: int) -> dict:
    return {
        "city": "jinan",
        "scenario": "jinan_3x4_real",
        "seed": seed,
        "candidate_key": "selected_source",
        "profile": profile,
        "metrics": {
            "mean_tripinfo_waiting_time": waiting,
            "collision_incidents": collisions,
            "starting_teleports": 0,
            "ending_teleports": 0,
        },
    }


def test_guard_selector_accepts_safety_gain_with_bounded_waiting_cost() -> None:
    rows = {
        "raw": [_row("raw", 1, 100.0, 1), _row("raw", 2, 100.0, 1)],
        "guard": [
            _row("guard", 1, 100.5, 0),
            _row("guard", 2, 100.5, 0),
        ],
    }

    result = select_guard_profile_from_closed_loop(
        rows,
        city="jinan",
        scenarios=("jinan_3x4_real",),
        profile_keys=("raw", "guard"),
    )

    assert result["selected_profile_key"] == "guard"
    assert result["selection_reason"] == "safety_improving_guard"


def test_guard_selector_keeps_raw_without_material_gain() -> None:
    rows = {
        "raw": [_row("raw", 1, 100.0, 0), _row("raw", 2, 100.0, 0)],
        "guard": [
            _row("guard", 1, 99.8, 0),
            _row("guard", 2, 99.8, 0),
        ],
    }

    result = select_guard_profile_from_closed_loop(
        rows,
        city="jinan",
        scenarios=("jinan_3x4_real",),
        profile_keys=("raw", "guard"),
    )

    assert result["selected_profile_key"] == "raw"
    assert result["selection_reason"] == "raw_guard_fallback"
