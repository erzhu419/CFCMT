from copy import deepcopy
from dataclasses import dataclass
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.eval import traffic_signal_state_conditioned_source_closed_loop as original
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    FrozenAnchoredBlendModel, constant_candidate_key,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3, FEATURE_NAMES_V3
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.fraction_target_rigid import FractionTargetRigidOriginator, MODEL_PROTOCOL
from cf_h2o.traffic_signal.mechanism_parameter_prior import mechanism_parameter_design
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL
from cf_h2o.traffic_signal.right_of_way_context import read_network_right_of_way_context
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    FittedUtilityGate, UTILITY_FEATURE_NAMES, UtilityGateConfig,
)


@dataclass
class _Prediction:
    scores: tuple

    def predict(self, dataset):
        return {"control_cost": {"mean": np.asarray(self.scores),
            "uncertainty": np.zeros(dataset.size), "context_trust": np.ones(dataset.size)}}


def setup_case(tmp_path, scores=(0., -1., 2.)):
    net = tmp_path / "net.xml"
    net.write_text('<net>' + ''.join(
        f'<connection from="in" to="out{i}" fromLane="{i}" toLane="0" tl="A" linkIndex="{i}" state="M"/>'
        for i in range(3)) + '</net>')
    context = read_network_right_of_way_context(net)
    model = FrozenAnchoredBlendModel(anchor_model=_Prediction(scores), correction_model=_Prediction(scores),
        candidate=constant_candidate_key(0.0), anchor_objective_mode="control_only", correction_objective_mode="control_only")
    payload = {"protocol": MODEL_PROTOCOL, "occupancy_equation_protocol": OCCUPANCY_EQUATION_PROTOCOL,
               "city": "cologne", "target_budget": 100, "rigid_model": model}
    features = np.zeros((3, len(FEATURE_NAMES_V3)))
    features[0, FEATURE_NAMES_V3.index("current_phase_overlap")] = 1.
    absolute = MechanismDataset(
        feature_names=FEATURE_NAMES_V3, features=features, context_names=("context",), context=np.ones((3, 1)),
        priors={"cost": np.zeros(3)}, targets={"cost": np.zeros(3)}, domains=np.asarray(["cologne"] * 3),
        metadata={"occupancy_equation_protocol": OCCUPANCY_EQUATION_PROTOCOL,
                  "scenario": "s", "action_group_ids": ["s:seed1:10:A"] * 3,
                  "row_tls": ["A"] * 3, "candidate_states": ["Grr", "rrG", "rGr"]})
    contrast = build_action_contrast_dataset(absolute, reference_policy="phase_pressure",
                                             contrast_features=CONTRAST_FEATURES_V3)
    preparation = {"states": {"A": object()}, "executors": {"A": SimpleNamespace(
        current_green_state="Grr", target_green_state="Grr", green_elapsed_sec=20.,
        remaining_sec=0., is_switching=False, timings={"Grr": SimpleNamespace(min_green_sec=10.)})},
        "routing_graph": SimpleNamespace(adjacency={"A": ()}), "control_interval_sec": 10}
    return payload, context, contrast, preparation


def original_payload(model, feature_names):
    width = len(feature_names)
    gate = FittedUtilityGate(model=None, feature_mean=np.zeros(len(UTILITY_FEATURE_NAMES)),
        feature_scale=np.ones(len(UTILITY_FEATURE_NAMES)), error_margin=0.,
        config=UtilityGateConfig(), diagnostics={"enabled": False})
    return {"protocol": original.RUNTIME_MODEL_PROTOCOL, "occupancy_equation_protocol": OCCUPANCY_EQUATION_PROTOCOL,
        "city": "cologne", "target_budget": 100, "admitted": False, "rigid_model": deepcopy(model),
        "mechanism_feature_names": feature_names, "feature_scale": np.ones(width), "target_beta": np.zeros(width),
        "candidate_betas": {"source": np.zeros(width)}, "placebo_betas": {"source": np.zeros(width)},
        "source_utility_gate": gate, "placebo_utility_gate": gate}


@pytest.mark.parametrize("scores,selected", [
    ((0., -1., 2.), 1), ((0., 0., 2.), 0), ((0., -1., -1.), 2),
])
def test_actual_rigid_blend_matches_original_target_only_decision_and_tie_break(tmp_path, scores, selected):
    payload, context, contrast, preparation = setup_case(tmp_path, scores)
    fresh = FractionTargetRigidOriginator(runtime_payload=payload, network_context=context)
    fresh.prepare_interval(**preparation)
    augmented = fresh._augment_contrast(contrast, reference_index=0)
    feature_names = mechanism_parameter_design(augmented).feature_names
    baseline = original.StateConditionedSourceUtilityOriginator(
        runtime_payload=original_payload(payload["rigid_model"], feature_names),
        network_context=context, arm="rigid_target_only")
    baseline.prepare_interval(**preparation)
    new_decision = fresh.select(contrast, reference_index=0)
    assert new_decision.to_dict() == baseline.select(contrast, reference_index=0).to_dict()
    assert new_decision.selected_index == selected
    assert payload["rigid_model"].audit.prediction_calls == 1
    diagnostics = fresh.diagnostics()
    assert diagnostics["decision_count"] == 1
    assert diagnostics["interval_count"] == 1
    assert diagnostics["utility_gate_selected_count"] == 0
    assert diagnostics["phase_pressure_override_count"] == int(selected != 0)
    assert not ({"candidate_betas", "source_utility_gate", "target_beta"} & fresh.payload.keys())


@pytest.mark.parametrize("mismatch", ["payload_units", "contrast_units", "budget", "model"])
def test_target_only_rejects_wrong_contract_or_wrong_model(tmp_path, mismatch):
    payload, context, contrast, preparation = setup_case(tmp_path)
    if mismatch == "payload_units":
        payload["occupancy_equation_protocol"] = "old-equations"
    elif mismatch == "budget":
        payload["target_budget"] = 25
    elif mismatch == "model":
        payload["rigid_model"] = _Prediction((0., -1., 2.))
    if mismatch != "contrast_units":
        with pytest.raises(ValueError):
            FractionTargetRigidOriginator(runtime_payload=payload, network_context=context)
    else:
        fresh = FractionTargetRigidOriginator(runtime_payload=payload, network_context=context)
        fresh.prepare_interval(**preparation)
        contrast.metadata["occupancy_equation_protocol"] = "old-equations"
        with pytest.raises(ValueError, match="contrast schema"):
            fresh.select(contrast, reference_index=0)


def test_plain_pickle_load_preserves_target_model_and_checks_city(tmp_path):
    payload, context, contrast, preparation = setup_case(tmp_path)
    path = tmp_path / "target.pkl"
    path.write_bytes(pickle.dumps(payload))
    fresh = FractionTargetRigidOriginator.load(path, expected_city="cologne", network_context=context)
    fresh.prepare_interval(**preparation)
    assert fresh.select(contrast, reference_index=0).selected_index == 1
    with pytest.raises(ValueError, match="another city"):
        FractionTargetRigidOriginator.load(path, expected_city="atlanta", network_context=context)
