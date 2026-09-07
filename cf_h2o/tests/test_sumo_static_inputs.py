import gzip
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_resco_cfcmt_benchmark import (
    _route_departure_stats,
)
from cf_h2o.traffic_signal.demand_schedule_context import (
    build_demand_schedule_profile,
)
from cf_h2o.traffic_signal.network_context import _route_descriptors
from cf_h2o.traffic_signal.sumo_static_inputs import (
    demand_route_mixture,
    iter_static_demand_records,
    load_static_route_catalog,
    parse_sumo_time_seconds,
    static_demand_input_paths,
)


def _write_split_demand_case(tmp_path: Path) -> Path:
    (tmp_path / "net.xml").write_text(
        '<net><tlLogic id="t0" programID="0">'
        '<phase duration="30" state="G"/></tlLogic></net>',
        encoding="utf-8",
    )
    (tmp_path / "tls.add.xml").write_text("<additional/>", encoding="utf-8")
    (tmp_path / "routes.rou.xml").write_text(
        """<routes>
  <route id="r_short" edges="a b"/>
  <route id="r_long" edges="c d e f"/>
</routes>""",
        encoding="utf-8",
    )
    (tmp_path / "emitters.xml").write_text(
        """<additional>
  <routeDistribution id="mix">
    <route refId="r_short" probability="1"/>
    <route refId="r_long" probability="3"/>
  </routeDistribution>
  <vehicle id="v0" route="mix" depart="0:1:40"/>
  <vehicle id="v1" route="r_short" depart="0:11:40"/>
  <flow id="f0" route="r_long" begin="0:0" end="10:0" period="5:0"/>
</additional>""",
        encoding="utf-8",
    )
    config = tmp_path / "case.sumocfg"
    config.write_text(
        """<configuration><input>
  <net-file value="net.xml"/>
  <additional-files value="tls.add.xml,routes.rou.xml,emitters.xml"/>
</input><time><begin value="0"/><end value="1200"/></time></configuration>""",
        encoding="utf-8",
    )
    return config


def test_split_additional_demand_is_streamed_with_global_route_catalog(
    tmp_path: Path,
) -> None:
    config = _write_split_demand_case(tmp_path)
    paths = static_demand_input_paths(config)
    catalog = load_static_route_catalog(paths)
    records = list(iter_static_demand_records(paths))

    assert [record.kind for record in records] == ["vehicle", "vehicle", "flow"]
    assert demand_route_mixture(records[0], catalog) == (
        (("a", "b"), 0.25),
        (("c", "d", "e", "f"), 0.75),
    )


def test_parse_sumo_numeric_and_human_readable_time() -> None:
    assert parse_sumo_time_seconds("120") == 120.0
    assert parse_sumo_time_seconds("2:00") == 120.0
    assert parse_sumo_time_seconds("1:02:03") == 3723.0
    assert parse_sumo_time_seconds("1:19:46:50") == 157_610.0
    with pytest.raises(ValueError, match="invalid SUMO time"):
        parse_sumo_time_seconds("1:2:3:4:5")


def test_all_static_context_readers_include_additional_file_demand(
    tmp_path: Path,
) -> None:
    config = _write_split_demand_case(tmp_path)

    profile = build_demand_schedule_profile(config, horizon_sec=1200)
    lengths, weights = _route_descriptors(config)
    departure_stats = _route_departure_stats(config)

    assert profile.scheduled_vehicle_count == pytest.approx(4.0)
    assert profile.features[-1] == pytest.approx(3.375 / 20.0)
    assert sorted(zip(lengths.tolist(), weights.tolist())) == [
        (2.0, 1.25),
        (4.0, 2.75),
    ]
    assert departure_stats == {"trip_count": 4.0, "horizon_sec": 1200.0}


def test_static_context_readers_support_gzip_network_and_demand(
    tmp_path: Path,
) -> None:
    config = _write_split_demand_case(tmp_path)
    for filename in ("net.xml", "routes.rou.xml", "emitters.xml"):
        source = tmp_path / filename
        compressed = source.with_name(f"{source.name}.gz")
        with source.open("rb") as input_handle, gzip.open(
            compressed, "wb"
        ) as output_handle:
            output_handle.write(input_handle.read())
        source.unlink()
    config.write_text(
        """<configuration><input>
  <net-file value="net.xml.gz"/>
  <additional-files value="tls.add.xml,routes.rou.xml.gz,emitters.xml.gz"/>
</input><time><begin value="0"/><end value="1200"/></time></configuration>""",
        encoding="utf-8",
    )

    paths = static_demand_input_paths(config)
    catalog = load_static_route_catalog(paths)
    records = list(iter_static_demand_records(paths))
    profile = build_demand_schedule_profile(config, horizon_sec=1200)
    lengths, weights = _route_descriptors(config)
    departure_stats = _route_departure_stats(config)

    assert [record.kind for record in records] == ["vehicle", "vehicle", "flow"]
    assert demand_route_mixture(records[0], catalog) == (
        (("a", "b"), 0.25),
        (("c", "d", "e", "f"), 0.75),
    )
    assert profile.scheduled_vehicle_count == pytest.approx(4.0)
    assert sorted(zip(lengths.tolist(), weights.tolist())) == [
        (2.0, 1.25),
        (4.0, 2.75),
    ]
    assert departure_stats == {"trip_count": 4.0, "horizon_sec": 1200.0}


def test_triggered_vehicle_uses_person_plan_departure_and_nested_flow_is_ignored(
    tmp_path: Path,
) -> None:
    (tmp_path / "net.xml").write_text(
        '<net><tlLogic id="t0" programID="0">'
        '<phase duration="30" state="G"/></tlLogic></net>',
        encoding="utf-8",
    )
    (tmp_path / "demand.rou.xml").write_text(
        """<routes>
  <vehicle id="car0" depart="triggered"><route edges="a b c"/></vehicle>
  <person id="p0" depart="0:2:0"><ride from="a" to="c" lines="car0"/></person>
</routes>""",
        encoding="utf-8",
    )
    (tmp_path / "calib.add.xml").write_text(
        """<additional><calibrator id="c0" edge="a" pos="0">
  <flow begin="0" end="1200" type="passenger"/>
</calibrator></additional>""",
        encoding="utf-8",
    )
    config = tmp_path / "case.sumocfg"
    config.write_text(
        """<configuration><input><net-file value="net.xml"/>
  <route-files value="demand.rou.xml"/>
  <additional-files value="calib.add.xml"/>
</input><time><begin value="0"/><end value="1200"/></time></configuration>""",
        encoding="utf-8",
    )

    records = list(iter_static_demand_records(static_demand_input_paths(config)))
    profile = build_demand_schedule_profile(config, horizon_sec=1200)
    departure_stats = _route_departure_stats(config)

    assert [record.kind for record in records] == ["vehicle"]
    assert profile.scheduled_vehicle_count == pytest.approx(1.0)
    assert departure_stats == {"trip_count": 1.0, "horizon_sec": 1200.0}
