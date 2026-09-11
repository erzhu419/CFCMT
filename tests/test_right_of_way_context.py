from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.dataset_cache import save_mechanism_dataset
from cf_h2o.eval.traffic_signal_right_of_way_signature import (
    build_right_of_way_signatures,
)
from cf_h2o.traffic_signal.right_of_way_context import (
    NEIGHBOR_EXECUTION_STATE_FEATURES,
    RIGHT_OF_WAY_ACTION_FEATURES,
    augment_dataset_with_right_of_way_context,
    candidate_right_of_way_features,
    net_file_from_sumocfg,
    read_network_right_of_way_context,
)


def _write_network(tmp_path: Path) -> tuple[Path, Path]:
    network = tmp_path / "network.net.xml"
    network.write_text(
        """<net>
  <connection from="in_a" to="out" fromLane="0" toLane="0"
              tl="A" linkIndex="0" state="O"/>
  <connection from="in_a" to="unique" fromLane="1" toLane="0"
              tl="A" linkIndex="1" state="O"/>
  <connection from="major" to="out" fromLane="0" toLane="0" state="M"/>
  <connection from="in_b" to="other" fromLane="0" toLane="0"
              tl="B" linkIndex="0" state="O"/>
</net>\n""",
        encoding="utf-8",
    )
    sumocfg = tmp_path / "scenario.sumocfg"
    sumocfg.write_text(
        '<configuration><input><net-file value="network.net.xml"/>'
        "</input></configuration>\n",
        encoding="utf-8",
    )
    return network, sumocfg


def _dataset() -> MechanismDataset:
    names = (
        "current_phase_overlap",
        "current_green_elapsed_norm",
        "switch_indicator",
        "clearance_fraction",
    )
    return MechanismDataset(
        feature_names=names,
        features=np.asarray(
            [
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 2.0, 1.0, 0.3],
                [1.0, 3.0, 0.0, 0.0],
                [0.0, 3.0, 1.0, 0.2],
            ]
        ),
        context_names=("context",),
        context=np.ones((4, 1)),
        priors={"cost": np.zeros(4)},
        targets={"cost": np.arange(4.0)},
        domains=np.asarray(["city"] * 4),
        metadata={
            "scenario": "s",
            "action_group_ids": [
                "s:seed1:10:A",
                "s:seed1:10:A",
                "s:seed1:10:B",
                "s:seed1:10:B",
            ],
            "row_tls": ["A", "A", "B", "B"],
            "candidate_states": ["Gr", "rg", "G", "r"],
            "signal_routing_graph": {"adjacency": {"A": ["B"], "B": []}},
        },
    )


def test_candidate_right_of_way_features_distinguish_protected_and_yielding(
    tmp_path: Path,
) -> None:
    network, sumocfg = _write_network(tmp_path)
    assert net_file_from_sumocfg(sumocfg) == network.resolve()
    context = read_network_right_of_way_context(network)

    values = candidate_right_of_way_features(context, "A", "Gg")
    actual = dict(zip(RIGHT_OF_WAY_ACTION_FEATURES, values, strict=True))

    assert actual["protected_green_ratio"] == pytest.approx(0.5)
    assert actual["permissive_green_ratio"] == pytest.approx(0.5)
    assert actual["green_shared_receiver_ratio"] == pytest.approx(0.5)
    assert actual["green_uncontrolled_major_merge_ratio"] == pytest.approx(0.5)
    assert actual["protected_uncontrolled_major_merge_ratio"] == pytest.approx(0.5)
    assert actual["permissive_uncontrolled_major_merge_ratio"] == 0.0
    assert actual["green_receiver_competition_mean"] == pytest.approx(0.125)


def test_dataset_augmentation_uses_only_matched_neighbor_snapshots(
    tmp_path: Path,
) -> None:
    network, _ = _write_network(tmp_path)
    context = read_network_right_of_way_context(network)

    augmented = augment_dataset_with_right_of_way_context(
        _dataset(), {"s": context}
    )
    index = {name: i for i, name in enumerate(augmented.feature_names)}

    assert augmented.size == 4
    assert augmented.features[0, index["protected_green_ratio"]] == 1.0
    assert augmented.features[1, index["permissive_green_ratio"]] == 1.0
    assert augmented.features[0, index["neighbor_phase_observation_ratio"]] == 1.0
    assert (
        augmented.features[0, index["neighbor_current_protected_ratio_mean"]]
        == 1.0
    )
    assert (
        augmented.features[0, index["neighbor_current_green_elapsed_norm_mean"]]
        == 3.0
    )
    assert augmented.features[0, index["neighbor_switching_fraction"]] == 0.0
    assert augmented.metadata["right_of_way_context_protocol"].endswith("v1")
    assert tuple(NEIGHBOR_EXECUTION_STATE_FEATURES) == tuple(
        augmented.feature_names[-len(NEIGHBOR_EXECUTION_STATE_FEATURES) :]
    )


def test_dataset_augmentation_reports_missing_neighbor_coverage(
    tmp_path: Path,
) -> None:
    network, _ = _write_network(tmp_path)
    context = read_network_right_of_way_context(network)
    dataset = _dataset().subset(np.asarray([True, True, False, False]))
    dataset = MechanismDataset(
        feature_names=dataset.feature_names,
        features=dataset.features,
        context_names=dataset.context_names,
        context=dataset.context,
        priors=dataset.priors,
        targets=dataset.targets,
        domains=dataset.domains,
        metadata={
            **dict(_dataset().metadata),
            "action_group_ids": ["s:seed1:10:A"] * 2,
            "row_tls": ["A"] * 2,
            "candidate_states": ["Gr", "rg"],
        },
    )

    augmented = augment_dataset_with_right_of_way_context(dataset, {"s": context})
    index = {name: i for i, name in enumerate(augmented.feature_names)}

    assert np.all(
        augmented.features[:, index["neighbor_phase_observation_ratio"]] == 0.0
    )
    for name in NEIGHBOR_EXECUTION_STATE_FEATURES[1:]:
        assert np.all(augmented.features[:, index[name]] == 0.0)


def test_signature_builder_counts_and_skips_empty_cache_shards(
    tmp_path: Path,
) -> None:
    _, sumocfg = _write_network(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"scenarios":[{"scenario":"s","city_group":"city",'
        '"sumocfg":"scenario.sumocfg"}]}\n',
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    populated = _dataset()
    save_mechanism_dataset(cache / "s__populated.npz", populated)
    empty = MechanismDataset(
        feature_names=populated.feature_names,
        features=np.zeros((0, len(populated.feature_names))),
        context_names=populated.context_names,
        context=np.zeros((0, len(populated.context_names))),
        priors={},
        targets={},
        domains=np.asarray([], dtype=str),
        metadata={
            "scenario": "s",
            "action_group_ids": [],
            "row_tls": [],
            "candidate_states": [],
            "signal_routing_graph": {"adjacency": {"A": ["B"], "B": []}},
        },
    )
    save_mechanism_dataset(cache / "s__empty.npz", empty)

    result = build_right_of_way_signatures(
        source_manifest=manifest, source_cache_root=cache
    )

    assert result["cache_file_count"] == 2
    assert result["empty_cache_file_count"] == 1
    assert result["declared_scenarios_without_admitted_cache"] == []
    assert result["scenario_signatures"]["s"]["cache_file_count"] == 2
    assert result["scenario_signatures"]["s"]["empty_cache_file_count"] == 1
    assert result["scenario_signatures"]["s"]["row_count"] == populated.size
