from pathlib import Path

import numpy as np
import pytest

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.movement_arrival_timeline import (
    MOVEMENT_ARRIVAL_FEATURE_NAMES,
    augment_dataset_with_movement_arrivals,
    build_movement_arrival_timeline_context,
)


def _write_fixture(root: Path) -> Path:
    (root / "tiny.net.xml").write_text(
        """<net>
  <edge id="in_a" from="a" to="tls0"><lane id="in_a_0" index="0" speed="10" length="100"/></edge>
  <edge id="in_b" from="b" to="tls0"><lane id="in_b_0" index="0" speed="10" length="100"/></edge>
  <edge id="out_a" from="tls0" to="c"><lane id="out_a_0" index="0" speed="10" length="100"/></edge>
  <edge id="out_b" from="tls0" to="d"><lane id="out_b_0" index="0" speed="10" length="100"/></edge>
  <tlLogic id="tls0" type="static" programID="0" offset="0">
    <phase duration="30" state="GGr"/><phase duration="30" state="rrG"/>
  </tlLogic>
  <connection from="in_a" to="out_a" fromLane="0" toLane="0" tl="tls0" linkIndex="0"/>
  <connection from="in_a" to="out_a" fromLane="0" toLane="0" tl="tls0" linkIndex="1"/>
  <connection from="in_b" to="out_b" fromLane="0" toLane="0" tl="tls0" linkIndex="2"/>
</net>""",
        encoding="utf-8",
    )
    (root / "tiny.rou.xml").write_text(
        """<routes>
  <vType id="car" vClass="passenger"/>
  <vehicle id="a0" type="car" depart="0"><route edges="in_a out_a"/></vehicle>
  <vehicle id="b0" type="car" depart="10"><route edges="in_b out_b"/></vehicle>
  <vehicle id="a1" type="car" depart="15"><route edges="in_a out_a"/></vehicle>
</routes>""",
        encoding="utf-8",
    )
    cfg = root / "tiny.sumocfg"
    cfg.write_text(
        """<configuration><input><net-file value="tiny.net.xml"/><route-files value="tiny.rou.xml"/></input><time><begin value="0"/><end value="120"/></time></configuration>""",
        encoding="utf-8",
    )
    return cfg


def test_movement_projection_deduplicates_parallel_connections(tmp_path: Path) -> None:
    context = build_movement_arrival_timeline_context(
        _write_fixture(tmp_path), horizon_sec=120, bin_sec=1
    )

    assert context.scheduled_elements == 3
    assert context.projected_elements == 3
    assert context.projected_arrival_mass == pytest.approx(3.0)
    assert sum(sum(values) for values in context.movement_arrivals["tls0"].values()) == pytest.approx(3.0)
    assert context.movement_link_indices["tls0"][("in_a", "out_a")] == (0, 1)


def test_candidate_features_distinguish_served_arrival_wave(tmp_path: Path) -> None:
    context = build_movement_arrival_timeline_context(
        _write_fixture(tmp_path), horizon_sec=120, bin_sec=1
    )
    serve_a = context.vector_at("tls0", 0.0, "GGr")
    serve_b = context.vector_at("tls0", 0.0, "rrG")
    index = {name: i for i, name in enumerate(MOVEMENT_ARRIVAL_FEATURE_NAMES)}

    assert serve_a[index["served_arrivals_next_30_per_movement"]] == pytest.approx(2.0)
    assert serve_b[index["served_arrivals_next_30_per_movement"]] == pytest.approx(1.0)
    assert context.arrival_pressure_score("tls0", 0.0, "GGr") == pytest.approx(1.0)
    assert context.arrival_pressure_score("tls0", 0.0, "rrG") == pytest.approx(-1.0)
    assert not np.array_equal(serve_a, serve_b)


def test_future_window_excludes_past_arrivals(tmp_path: Path) -> None:
    context = build_movement_arrival_timeline_context(
        _write_fixture(tmp_path), horizon_sec=120, bin_sec=1
    )
    served, unserved, _, _ = context.candidate_window_totals(
        "tls0", 21.0, "GGr", 30
    )

    assert served == pytest.approx(1.0)
    assert unserved == pytest.approx(0.0)


def test_dataset_augmentation_preserves_targets_and_adds_aligned_rows(
    tmp_path: Path,
) -> None:
    context = build_movement_arrival_timeline_context(
        _write_fixture(tmp_path), horizon_sec=120, bin_sec=1
    )
    dataset = MechanismDataset(
        feature_names=("queue",),
        features=np.asarray([[1.0], [2.0]]),
        context_names=("demand",),
        context=np.asarray([[0.5], [0.5]]),
        priors={"cost": np.asarray([1.0, 2.0])},
        targets={"cost": np.asarray([1.5, 2.5])},
        domains=np.asarray(["tiny", "tiny"]),
        metadata={
            "row_tls": ["tls0", "tls0"],
            "row_times": [0.0, 0.0],
            "candidate_states": ["GGr", "rrG"],
        },
    )

    augmented = augment_dataset_with_movement_arrivals(dataset, context)

    assert augmented.features.shape == (2, 1 + len(MOVEMENT_ARRIVAL_FEATURE_NAMES))
    assert augmented.feature_names[-len(MOVEMENT_ARRIVAL_FEATURE_NAMES):] == MOVEMENT_ARRIVAL_FEATURE_NAMES
    assert not np.array_equal(augmented.features[0, 1:], augmented.features[1, 1:])
    np.testing.assert_array_equal(augmented.targets["cost"], dataset.targets["cost"])
    assert augmented.metadata["movement_arrival_protocol"] == context.protocol


def test_short_candidate_state_is_rejected(tmp_path: Path) -> None:
    context = build_movement_arrival_timeline_context(
        _write_fixture(tmp_path), horizon_sec=120, bin_sec=1
    )
    with pytest.raises(ValueError, match="shorter than controlled links"):
        context.vector_at("tls0", 0.0, "G")
