"""Protocol inference tests with synthetic, manually specified coefficients."""
from copy import deepcopy
import json
from math import exp, log
from pathlib import Path

import pytest

from experiments.tsc_target_packet_kernel_20261109.local_model import (
    COEFFICIENT_VECTORS, FEATURES, FEATURE_NAMES, FrozenLocalModel,
    masked_lane_probabilities)


def context():
    row = {name: 0. for name in FEATURE_NAMES}
    row.update(accel_mps2=2., decel_mps2=4.5, length_m=5., tau_s=1.,
               allowed_speed_mps=15., speed_factor=1., signal_G=1.)
    return row


def coefficients():
    return {name: [0.]*len(FEATURES) for name in COEFFICIENT_VECTORS}


def set_weight(weights, head, feature, value):
    weights[head][FEATURE_NAMES.index(feature)] = value


def test_feature_order_divisors_and190_parameters_match_the_frozen_protocol():
    path = Path(__file__).resolve().parents[1]/'experiments/tsc_target_packet_protocol_20261109/protocol.json'
    protocol = json.loads(path.read_text())['model']
    assert FEATURES == tuple((row['name'], row['fixed_divisor']) for row in protocol['features'])
    assert len(FEATURES) == protocol['feature_count'] == 38
    assert len(COEFFICIENT_VECTORS)*len(FEATURES) == protocol['coefficient_count'] == 190
    weights = coefficients()
    weights['start'].pop()
    with pytest.raises(ValueError, match='38 finite'):
        FrozenLocalModel(weights)
    incomplete = context()
    del incomplete['signal_uncontrolled']
    with pytest.raises(KeyError, match='signal_uncontrolled'):
        FrozenLocalModel().predict(incomplete)


def test_default_zeros_are_disclosed_untrained_and_have_reference_stay_logit():
    model = FrozenLocalModel()
    prediction = model.predict(context())
    assert model.parameter_origin == 'untrained_zero_placeholder'
    assert model.fit_performed is False
    assert prediction['motion_residual_mps'] == 0
    assert prediction['start_probability'] == prediction['yield_probability'] == .5
    assert prediction['lane_probabilities'] == dict(left=1/3, stay=1/3, right=1/3)


def test_physical_divisors_bias_and_independent_heads_give_known_predictions():
    row, weights = context(), coefficients()
    row.update(bias=999., speed_mps=15., net_leader_gap_m=50.)
    set_weight(weights, 'motion', 'bias', .1)
    set_weight(weights, 'motion', 'speed_mps', .5)
    set_weight(weights, 'motion', 'accel_mps2', -.3)
    set_weight(weights, 'start', 'bias', log(3))
    set_weight(weights, 'yield_entry', 'net_leader_gap_m', log(9))
    set_weight(weights, 'lane_left', 'bias', log(2))
    set_weight(weights, 'lane_right', 'bias', log(3))
    model = FrozenLocalModel(weights)
    output = model.predict(row)
    assert model.features(row)[0] == 1.
    assert output['motion_residual_mps'] == pytest.approx(.4)
    assert output['start_probability'] == pytest.approx(.75)
    assert output['yield_probability'] == pytest.approx(.9)
    assert output['lane_probabilities'] == pytest.approx(dict(left=2/6, stay=1/6, right=3/6))
    # Supplying a coefficient array does not perform a fit; the model copies it.
    weights['motion'][0] = 1000.
    assert model.predict(row) == output
    assert model.parameter_origin == 'supplied_coefficients' and not model.fit_performed


def test_signed_motion_bounds_stable_large_logits_and_G_g_remain_distinct():
    weights, row = coefficients(), context()
    set_weight(weights, 'motion', 'bias', 1000.)
    set_weight(weights, 'start', 'bias', 1000.)
    set_weight(weights, 'yield_entry', 'bias', -1000.)
    set_weight(weights, 'lane_left', 'bias', 1000.)
    set_weight(weights, 'lane_right', 'bias', -1000.)
    output = FrozenLocalModel(weights).predict(row)
    assert output['motion_residual_mps'] == 2.
    assert output['start_probability'] == 1. and output['yield_probability'] == 0.
    assert output['lane_probabilities'] == dict(left=1., stay=0., right=0.)
    set_weight(weights, 'motion', 'bias', -1000.)
    assert FrozenLocalModel(weights).predict(row)['motion_residual_mps'] == -4.5
    weights = coefficients()
    set_weight(weights, 'start', 'signal_G', 1.)
    set_weight(weights, 'start', 'signal_g', -1.)
    model = FrozenLocalModel(weights)
    assert model.predict(row)['start_probability'] == pytest.approx(1/(1+exp(-1)))
    row.update(signal_G=0., signal_g=1.)
    assert model.predict(row)['start_probability'] == pytest.approx(1/(1+exp(1)))


def test_extra_ids_and_future_fields_are_never_read_or_converted():
    class CurrentFieldsOnly(dict):
        def __getitem__(self, name):
            assert name in FEATURE_NAMES
            return super().__getitem__(name)
    row = CurrentFieldsOnly(context())
    row.update(vehicle_id=object(), actual_next_link=object(), future_PP=object(), cost450=object())
    saved = deepcopy({name: row[name] for name in FEATURE_NAMES})
    assert FrozenLocalModel().predict(row) == FrozenLocalModel().predict(context())
    assert {name: row[name] for name in FEATURE_NAMES} == saved


def test_mask_precedes_softmax_even_when_forbidden_logit_causes_underflow():
    weights = coefficients()
    set_weight(weights, 'lane_left', 'bias', 1000.)
    set_weight(weights, 'lane_right', 'bias', log(3))
    output = FrozenLocalModel(weights).predict(context())
    assert output['lane_logits'] == dict(left=1000., stay=0., right=log(3))
    assert output['lane_probabilities']['stay'] == output['lane_probabilities']['right'] == 0.
    masked = masked_lane_probabilities(output['lane_logits'], dict(left=False, stay=True, right=True))
    assert masked == pytest.approx(dict(left=0., stay=.25, right=.75))
    assert masked_lane_probabilities(output['lane_logits'], dict(left=False, stay=True, right=False)) == dict(left=0., stay=1., right=0.)
