from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_benchmark import (
    FEATURE_NAMES,
    GENERIC_SCENARIO_SUMMARY,
    PhaseLocalState,
    PhaseTransitionSet,
    _build_actuated_sumocfg,
    _candidate_features,
    _green_lane_ids,
    _parse_tripinfo_metrics,
    _sim_next_queue,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import PhaseCandidate, TlsPhaseInfo


def _fake_info():
    return TlsPhaseInfo(
        tls_id="tls0",
        controlled_links=[
            [("lane_a", "out_a", "")],
            [("lane_b", "out_b", "")],
            [("lane_a", "out_c", "")],
        ],
        candidates=(
            PhaseCandidate(phase_index=0, state="Grr", duration=30.0, green_count=1),
            PhaseCandidate(phase_index=1, state="rGG", duration=20.0, green_count=2),
        ),
        incoming_lanes=("lane_a", "lane_b"),
    )


def test_green_lane_ids_deduplicates_controlled_links():
    info = _fake_info()

    assert _green_lane_ids(info, info.candidates[1]) == ("lane_b", "lane_a")


def test_candidate_features_have_stable_shape_and_pressure_terms():
    info = _fake_info()
    state = PhaseLocalState(
        tls_id="tls0",
        info=info,
        time_norm=0.25,
        q_by_lane={"lane_a": 10.0, "lane_b": 4.0},
        veh_by_lane={"lane_a": 5.0, "lane_b": 2.0},
        speed_by_lane={"lane_a": 3.0, "lane_b": 7.0},
        occ_by_lane={"lane_a": 30.0, "lane_b": 20.0},
        down_occ_by_lane={"lane_a": 50.0, "lane_b": 10.0},
        summary=GENERIC_SCENARIO_SUMMARY,
    )

    row = _candidate_features(state, info.candidates[0], 0)

    assert row.shape == (len(FEATURE_NAMES),)
    assert row[FEATURE_NAMES.index("total_q")] == 14.0
    assert row[FEATURE_NAMES.index("green_q")] == 10.0
    assert row[FEATURE_NAMES.index("red_q")] == 4.0
    assert row[FEATURE_NAMES.index("service_pressure")] > 0.0


def test_sim_next_queue_is_vectorized_and_finite():
    x = np.ones((3, len(FEATURE_NAMES)), dtype=float)
    x[:, FEATURE_NAMES.index("total_q")] = [5.0, 10.0, 15.0]
    x[:, FEATURE_NAMES.index("green_q")] = [3.0, 5.0, 6.0]
    x[:, FEATURE_NAMES.index("green_down_occ")] = [0.0, 40.0, 90.0]
    x[:, FEATURE_NAMES.index("green_link_ratio")] = 0.4
    x[:, FEATURE_NAMES.index("green_lane_ratio")] = 0.5
    ts = PhaseTransitionSet(
        scenario="fake",
        x=x,
        summary=np.repeat(GENERIC_SCENARIO_SUMMARY[None, :], 3, axis=0),
        next_queue=np.zeros(3, dtype=float),
    )

    pred = _sim_next_queue(ts, control_interval_sec=10)

    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))


def test_tripinfo_metrics_separate_write_unfinished_records(tmp_path):
    tripinfo = tmp_path / "tripinfo.xml"
    tripinfo.write_text(
        """<tripinfos>
<tripinfo id="done" arrival="100" duration="80" waitingTime="10" timeLoss="20" departDelay="0"/>
<tripinfo id="open" arrival="-1" duration="50" waitingTime="30" timeLoss="40" departDelay="0" vaporized="end"/>
</tripinfos>
""",
        encoding="utf-8",
    )

    metrics = _parse_tripinfo_metrics(tripinfo)

    assert metrics["tripinfo_count"] == 2.0
    assert metrics["tripinfo_completed_count"] == 1.0
    assert metrics["tripinfo_unfinished_count"] == 1.0
    assert metrics["tripinfo_unfinished_fraction"] == 0.5
    assert metrics["mean_tripinfo_waiting_time"] == 20.0
    assert metrics["mean_completed_tripinfo_waiting_time"] == 10.0


def test_build_actuated_sumocfg_generates_absolute_inputs(tmp_path):
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    (scenario_dir / "fake.rou.xml").write_text("<routes />", encoding="utf-8")
    (scenario_dir / "fake.net.xml").write_text(
        """
<net>
  <tlLogic id="tls0" type="static" programID="0" offset="0">
    <phase duration="20" state="Gr"/>
    <phase duration="3" state="yr"/>
  </tlLogic>
</net>
""".strip(),
        encoding="utf-8",
    )
    sumocfg = scenario_dir / "fake.sumocfg"
    sumocfg.write_text(
        """
<configuration>
  <input>
    <net-file value="fake.net.xml"/>
    <route-files value="fake.rou.xml"/>
  </input>
</configuration>
""".strip(),
        encoding="utf-8",
    )

    actuated_cfg = _build_actuated_sumocfg(sumocfg, "fake", tmp_path / "generated")
    cfg_root = ET.parse(actuated_cfg).getroot()
    net_value = cfg_root.find(".//net-file").attrib["value"]
    route_value = cfg_root.find(".//route-files").attrib["value"]
    assert net_value.startswith("/")
    assert route_value.startswith("/")

    net_root = ET.parse(net_value).getroot()
    logic = net_root.find(".//tlLogic")
    phases = net_root.findall(".//phase")
    assert logic.attrib["type"] == "actuated"
    assert phases[0].attrib["minDur"] == "10"
    assert phases[0].attrib["maxDur"] == "50"
    assert phases[1].attrib["minDur"] == "3"
    assert phases[1].attrib["maxDur"] == "3"
