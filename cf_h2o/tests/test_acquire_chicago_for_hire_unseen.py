from __future__ import annotations

from datetime import date
from urllib.parse import parse_qs, urlparse

import pytest

from scripts.data.acquire_chicago_for_hire_unseen import (
    BOUNDARY_SOURCE,
    DEFAULT_WORKERS,
    DATES,
    EXPECTED_SPEED_ROWS,
    EXPECTED_TRIP_ROWS,
    OSM_MD5,
    OSM_SIZE_BYTES,
    PAGE_SIZE,
    SPEED_COLUMNS,
    SPEED_SOURCE,
    SOURCES,
    TRIP_COLUMNS,
    TRIP_GROUP_FIELDS,
    TRIP_SOURCES,
    _day_where,
    speed_page_url,
    trip_page_url,
)
from scripts.data.acquire_chicago_for_hire_unseen_remote import (
    TRANSFER_MODE,
    remote_receive_command,
)


def test_frozen_week_and_source_inventory_are_complete() -> None:
    assert DATES == tuple(date(2023, 8, 21 + index) for index in range(7))
    assert len(SOURCES) == 4
    assert len({source.dataset_id for source in SOURCES}) == len(SOURCES)
    assert BOUNDARY_SOURCE.dataset_id == "igwz-8jzy"
    assert SPEED_SOURCE.dataset_id == "sxs8-h27x"
    assert DEFAULT_WORKERS == 8
    assert OSM_SIZE_BYTES == 231_147_462
    assert OSM_MD5 == "1ae6e180e1c996c5def4cbfb03684808"
    assert sum(EXPECTED_TRIP_ROWS["tnp"].values()) == 1_599_557
    assert sum(EXPECTED_TRIP_ROWS["taxi"].values()) == 118_756
    assert sum(row[0] for row in EXPECTED_SPEED_ROWS.values()) == 1_004_073
    assert sum(row[1] for row in EXPECTED_SPEED_ROWS.values()) == 565_514


def test_trip_query_groups_every_released_location_field() -> None:
    url = trip_page_url(TRIP_SOURCES[0], DATES[0], PAGE_SIZE)
    query = parse_qs(urlparse(url).query)

    assert query["$where"] == [_day_where("trip_start_timestamp", DATES[0])]
    assert query["$group"] == [",".join(TRIP_GROUP_FIELDS)]
    assert query["$order"] == [",".join(TRIP_GROUP_FIELDS)]
    assert query["$offset"] == [str(PAGE_SIZE)]
    assert "count(*) as trip_count" in query["$select"][0]
    assert tuple(item.split(" as ")[-1] for item in query["$select"][0].split(",")[-5:]) == TRIP_COLUMNS[-5:]


def test_speed_query_has_stable_record_identity_and_full_columns() -> None:
    url = speed_page_url(DATES[-1], 0)
    query = parse_qs(urlparse(url).query)

    assert query["$select"] == [",".join(SPEED_COLUMNS)]
    assert query["$order"] == ["time,segment_id,record_id"]
    assert query["$limit"] == [str(PAGE_SIZE)]
    assert query["$offset"] == ["0"]


def test_frozen_metadata_identities_are_nonempty_and_unique() -> None:
    assert all(source.name and source.rows_updated_at > 0 for source in SOURCES)
    assert len({source.key for source in SOURCES}) == len(SOURCES)
    with pytest.raises(KeyError):
        _ = EXPECTED_TRIP_ROWS["private_cars"]


def test_remote_receiver_stages_before_atomic_validation() -> None:
    from pathlib import PurePosixPath

    command = remote_receive_command(
        PurePosixPath("/remote/acquisition/.speed.csv.part")
    )

    assert "cat > /remote/acquisition/.speed.csv.part" in command
    assert "sha256sum /remote/acquisition/.speed.csv.part" in command
    assert "mv " not in command
    assert TRANSFER_MODE.endswith("no_local_raw_file")
