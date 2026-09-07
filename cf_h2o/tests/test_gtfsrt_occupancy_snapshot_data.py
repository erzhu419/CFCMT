import pandas as pd
import pytest
from google.transit import gtfs_realtime_pb2

from cf_h2o.eval.gtfsrt_occupancy_snapshot_data import FEEDS, _row_group_indices, _status_name, _vehicle_rows_from_feed


def test_status_name_maps_gtfsrt_occupancy_enum():
    assert _status_name(1) == "MANY_SEATS_AVAILABLE"
    assert _status_name(5) == "FULL"
    assert _status_name(None) is None


def test_row_group_indices_are_evenly_spaced():
    assert _row_group_indices(0, 4) == []
    assert _row_group_indices(3, 4) == [0, 1, 2]
    assert _row_group_indices(10, 4) == [0, 3, 6, 9]


def test_vehicle_rows_from_feed_extracts_position_and_occupancy():
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = 1_779_100_000
    entity = feed.entity.add()
    entity.id = "veh-1"
    vehicle = entity.vehicle
    vehicle.trip.trip_id = "trip-1"
    vehicle.trip.route_id = "route-1"
    vehicle.vehicle.id = "bus-1"
    vehicle.position.latitude = 42.35
    vehicle.position.longitude = -71.06
    vehicle.timestamp = 1_779_100_001
    vehicle.occupancy_status = gtfs_realtime_pb2.VehiclePosition.FEW_SEATS_AVAILABLE
    vehicle.occupancy_percentage = 42

    rows = _vehicle_rows_from_feed(feed.SerializeToString(), FEEDS["mbta_live"], pd.Timestamp("2026-05-19T00:00:00Z"))

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["route_id"] == "route-1"
    assert row["vehicle_id"] == "bus-1"
    assert row["occupancy_status"] == "FEW_SEATS_AVAILABLE"
    assert row["occupancy_percentage"] == 42
    assert row["latitude"] == pytest.approx(42.35)
