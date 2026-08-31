from __future__ import annotations

from datetime import date

from scripts.data.audit_toronto_ptc_unseen_inputs import (
    DRIVABLE_CLASS_COUNTS,
    EXPECTED_DRIVABLE_FEATURE_COUNT,
    EXPECTED_MISSING_DATES,
    EXPECTED_PTC_DATES,
    _parse_ptc_hour,
    normalize_council,
    normalize_ward,
)


def test_geography_normalization_matches_published_labels() -> None:
    assert normalize_ward("10 - Spadina-Fort York") == "spadina-fort york"
    assert normalize_ward("Spadina-Fort York") == "spadina-fort york"
    assert normalize_council("Toronto and East York") == "toronto and east york"
    assert normalize_council("Toronto and East York Community Council") == (
        "toronto and east york"
    )


def test_frozen_ptc_date_inventory_excludes_only_documented_dates() -> None:
    assert len(EXPECTED_PTC_DATES) == 362
    assert EXPECTED_MISSING_DATES == {
        date(2025, 5, 26),
        date(2025, 5, 27),
        date(2025, 5, 28),
    }
    assert EXPECTED_PTC_DATES.isdisjoint(EXPECTED_MISSING_DATES)


def test_ptc_hour_parser_accepts_toronto_standard_and_daylight_offsets() -> None:
    assert _parse_ptc_hour("2025-01-01 00:00:00-05") == (
        date(2025, 1, 1),
        0,
        "-05",
    )
    assert _parse_ptc_hour("2025-07-01 23:00:00-04") == (
        date(2025, 7, 1),
        23,
        "-04",
    )


def test_amended_drivable_inventory_is_exact() -> None:
    assert len(DRIVABLE_CLASS_COUNTS) == 14
    assert sum(count for _, count in DRIVABLE_CLASS_COUNTS.values()) == (
        EXPECTED_DRIVABLE_FEATURE_COUNT
    )
    assert DRIVABLE_CLASS_COUNTS[201101][0] == "Expressway Ramp"
    assert 201800 not in DRIVABLE_CLASS_COUNTS
