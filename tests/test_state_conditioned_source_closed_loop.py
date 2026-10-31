from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL

import cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop as closed_loop
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    CONTRAST_FEATURES,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    FEATURE_NAMES_V3,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    RUNTIME_MODEL_PROTOCOL,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.mechanism_parameter_prior import (
    mechanism_parameter_design,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.right_of_way_context import (
    augment_dataset_with_right_of_way_context,
    read_network_right_of_way_context,
)
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    FittedUtilityGate,
    UTILITY_FEATURE_NAMES,
    UtilityGateConfig,
)


def _write_network(tmp_path: Path) -> Path:
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
    return network


def _absolute_dataset() -> MechanismDataset:
    index = {name: position for position, name in enumerate(FEATURE_NAMES_V3)}
    features = np.zeros((4, len(FEATURE_NAMES_V3)), dtype=float)
    features[:, index["current_phase_overlap"]] = [1.0, 0.0, 1.0, 0.0]
    features[:, index["current_green_elapsed_norm"]] = [2.0, 2.0, 3.0, 3.0]
    features[:, index["switch_indicator"]] = [0.0, 1.0, 0.0, 1.0]
    features[:, index["clearance_fraction"]] = [0.0, 0.3, 0.0, 0.2]
    return MechanismDataset(
        feature_names=FEATURE_NAMES_V3,
        features=features,
        context_names=("context",),
        context=np.ones((4, 1), dtype=float),
        priors={"cost": np.zeros(4, dtype=float)},
        targets={"cost": np.zeros(4, dtype=float)},
        domains=np.asarray(["city"] * 4),
        metadata={
            "occupancy_equation_protocol": OCCUPANCY_EQUATION_PROTOCOL,
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


def _subset(dataset: MechanismDataset, rows: slice, *, augmented: bool) -> MechanismDataset:
    return MechanismDataset(
        feature_names=dataset.feature_names,
        features=dataset.features[rows].copy(),
        context_names=dataset.context_names,
        context=dataset.context[rows].copy(),
        priors={name: values[rows].copy() for name, values in dataset.priors.items()},
        targets={name: values[rows].copy() for name, values in dataset.targets.items()},
        domains=dataset.domains[rows].copy(),
        metadata={
            "occupancy_equation_protocol": OCCUPANCY_EQUATION_PROTOCOL,
            "scenario": "s",
            "action_group_ids": ["s:seed1:10:A"] * 2,
            "row_tls": ["A"] * 2,
            "candidate_states": ["Gr", "rg"],
            "signal_routing_graph": {"adjacency": {"A": ["B"], "B": []}},
            **(
                {
                    "right_of_way_context_protocol": dataset.metadata[
                        "right_of_way_context_protocol"
                    ],
                    "right_of_way_feature_names": dataset.metadata[
                        "right_of_way_feature_names"
                    ],
                }
                if augmented
                else {}
            ),
        },
    )


@dataclass(frozen=True)
class _Timing:
    min_green_sec: float = 10.0


class _Executor:
    def __init__(self, state: str, *, elapsed: float) -> None:
        self.current_green_state = state
        self.target_green_state = state
        self.green_elapsed_sec = elapsed
        self.remaining_sec = 0.0
        self.is_switching = False
        self.timings = {state: _Timing()}


class _Graph:
    adjacency = {"A": ("B",), "B": ()}


def _payload(feature_names: tuple[str, ...], *, admitted: bool) -> dict:
    width = len(feature_names)
    disabled_gate = FittedUtilityGate(
        model=None,
        feature_mean=np.zeros(len(UTILITY_FEATURE_NAMES), dtype=float),
        feature_scale=np.ones(len(UTILITY_FEATURE_NAMES), dtype=float),
        error_margin=0.0,
        config=UtilityGateConfig(),
        diagnostics={"enabled": False},
    )
    return {
        "protocol": RUNTIME_MODEL_PROTOCOL,
        "occupancy_equation_protocol": OCCUPANCY_EQUATION_PROTOCOL,
        "city": "city",
        "target_budget": 100,
        "admitted": admitted,
        "mechanism_feature_names": feature_names,
        "feature_scale": np.ones(width, dtype=float),
        "target_beta": np.zeros(width, dtype=float),
        "candidate_betas": {"source|queue_service": np.zeros(width)},
        "placebo_betas": {"source|queue_service": np.zeros(width)},
        "rigid_model": object(),
        "source_utility_gate": disabled_gate,
        "placebo_utility_gate": disabled_gate,
    }


def test_runtime_context_reconstructs_offline_augmented_contrast(
    tmp_path: Path,
) -> None:
    context = read_network_right_of_way_context(_write_network(tmp_path))
    absolute = _absolute_dataset()
    offline_augmented = augment_dataset_with_right_of_way_context(
        absolute, {"s": context}
    )
    expected = build_action_contrast_dataset(
        _subset(offline_augmented, slice(0, 2), augmented=True),
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES,
    )
    original = build_action_contrast_dataset(
        _subset(absolute, slice(0, 2), augmented=False),
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    design = mechanism_parameter_design(expected)
    originator = closed_loop.StateConditionedSourceUtilityOriginator(
        runtime_payload=_payload(design.feature_names, admitted=False),
        network_context=context,
        arm="rigid_target_only",
    )
    originator.prepare_interval(
        states={"A": object(), "B": object()},
        executors={
            "A": _Executor("Gr", elapsed=20.0),
            "B": _Executor("G", elapsed=30.0),
        },
        routing_graph=_Graph(),
        control_interval_sec=10,
    )

    observed = originator._augment_contrast(original, reference_index=0)

    assert observed.feature_names == expected.feature_names
    assert np.allclose(observed.features, expected.features, rtol=0.0, atol=1e-12)
    assert tuple(observed.metadata["contrast_features"]) == CONTRAST_FEATURES


def test_rejected_source_city_is_exact_rigid_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    context = read_network_right_of_way_context(_write_network(tmp_path))
    absolute = _absolute_dataset()
    augmented = augment_dataset_with_right_of_way_context(absolute, {"s": context})
    design = mechanism_parameter_design(
        build_action_contrast_dataset(
            _subset(augmented, slice(0, 2), augmented=True),
            reference_policy="phase_pressure",
            contrast_features=CONTRAST_FEATURES,
        )
    )
    originator = closed_loop.StateConditionedSourceUtilityOriginator(
        runtime_payload=_payload(design.feature_names, admitted=False),
        network_context=context,
        arm="selected_source_corrected_rigid",
    )
    originator.prepare_interval(
        states={"A": object(), "B": object()},
        executors={
            "A": _Executor("Gr", elapsed=20.0),
            "B": _Executor("G", elapsed=30.0),
        },
        routing_graph=_Graph(),
        control_interval_sec=10,
    )
    contrast = build_action_contrast_dataset(
        _subset(absolute, slice(0, 2), augmented=False),
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    monkeypatch.setattr(
        closed_loop,
        "_rigid_score",
        lambda model, dataset: np.asarray([0.0, -1.0]),
    )

    decision = originator.select(contrast, reference_index=0)

    assert decision.selected_index == 1
    assert decision.rigid_selected_index == 1
    assert decision.utility_gate_selected is False
    assert decision.source_changed_rigid_action is False
    assert decision.priority == 1.0


def test_zero_incident_diagnostic_preserves_collision_evidence() -> None:
    admission = closed_loop.assess_zero_incident_diagnostic(
        {
            "starting_teleports": 0,
            "ending_teleports": 0,
            "collision_events": 2,
            "collision_incidents": 1,
        }
    )

    assert admission == {
        "protocol": "zero-teleport-zero-collision-observed-rollout-v1",
        "passed": False,
        "starting_teleports": 0,
        "ending_teleports": 0,
        "collision_events": 2,
        "collision_incidents": 1,
        "handling": "record_result_then_apply_paired_aggregate_safety_gate",
    }
