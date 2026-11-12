"""Raw calibration boundaries and signed fixed-action attribution identities."""
import json

import numpy as np
import pytest

from experiments.tsc_corridor_probability_calibration_20261109.calibration_metrics import (
    DecompositionAccumulator, ProbabilityCalibration, decision_decomposition,
)


def test_fixed_decile_boundaries_and_unthresholded_endpoint():
    calibration = ProbabilityCalibration()
    for index in range(11):
        p = np.full((3, 14), .24)
        p[:, 0] = index/10
        truth = np.zeros(14); truth[-1] = 1
        calibration.update(p, truth)
    result = calibration.result()
    for arm in ('T', 'S', 'P'):
        service = result['arms'][arm]['service_1']
        assert service['n'] == 11
        assert [row['n'] for row in service['bins']] == [1]*9+[2]
        assert service['bins'][-1]['p_mean'] == pytest.approx(.95)
        endpoint = result['arms'][arm]['detector_10']
        assert endpoint['p_mean'] == pytest.approx(.24)
        assert endpoint['bias'] == pytest.approx(-.76)
        assert endpoint['brier'] == pytest.approx(.76**2)
    json.dumps(result, allow_nan=False)


def test_masks_and_merging_keep_query_denominators_and_raw_probability_sums():
    p, y = np.full((3, 14), .6), np.ones(14)
    mask = np.zeros(14, dtype=bool); mask[[0, 12]] = True
    first = ProbabilityCalibration().update(p, y, mask)
    second = ProbabilityCalibration().update(p*.5, y*0, mask)
    merged = ProbabilityCalibration().merge(first.result()['raw_sums']).merge(second)
    direct = ProbabilityCalibration().update(p, y, mask).update(p*.5, y*0, mask)
    assert merged.result()['raw_sums'] == direct.result()['raw_sums']
    result = merged.result()['arms']['S']
    assert result['service_1']['n'] == 2
    assert result['service_1']['p_mean'] == pytest.approx(.45)
    assert result['service_13']['n'] == 2
    assert result['detector_10']['n'] == 0  # Endpoint never joins a signed service-query mask.
    assert result['detector_10']['brier'] is None


def example():
    aligned = dict(base=[2, 1], protected={'a': [1, 2], 'b': [2, 1]})
    flights = [dict(edge='a'), dict(edge='a'), dict(edge='b')]
    specifications = [
        ([[.2, .8], [.6, .4], [.3, .7]], [1, 0], ('same', 1)),
        ([[.7, .3], [.8, .2], [.6, .4]], [0, 1], ('same', 1)),
        ([[.1, .9], [.2, .8], [.3, .7]], [0, 1], ('other', 2)),
    ]
    captures = []
    for values, truth, context in specifications:
        raw = np.zeros((5, 14)); raw[:3, :2] = values
        raw[:, -1] = .87  # The hard endpoint must not enter service-score attribution.
        observed = np.zeros(14); observed[:2] = truth
        captures.append(dict(raw=raw, truth=observed, context=context, eta=3,
                             n_target=20, n_source=40, source_weight=.5))
    actions = dict(S=0, T=1, P=1)
    # Compute factual nominal scores independently using each request's mask.
    labels = np.asarray(aligned['base'], dtype=float)
    for flight, capture in zip(flights, captures):
        for request, mask in enumerate(aligned['protected'][flight['edge']]):
            labels[request] += sum(capture['truth'][bit] for bit in range(13) if mask & (1 << bit))
    return aligned, flights, captures, actions, labels


def test_signed_masks_preserve_both_directions_and_match_fixed_request_scores():
    result = decision_decomposition(*example())
    assert result['signed_masks']['a'][:2] == [1, -1]
    assert result['signed_masks']['b'][:2] == [-1, 1]
    assert result['queue_base_margin'] == 1
    assert result['predicted_margins'] == pytest.approx(dict(T=1.6, S=2.4, P=1.2))
    assert result['label_margin'] == 2
    assert result['source_correction_sum'] == pytest.approx(.8)
    assert result['baseline_error_sum'] == pytest.approx(-.4)
    assert result['source_error_sum'] == pytest.approx(.4)
    assert result['positive_correction_sum'] == pytest.approx(1)
    assert result['negative_correction_sum'] == pytest.approx(-.2)
    assert result['generic_count_component'] == pytest.approx(-.4)
    assert result['conditional_source_component'] == pytest.approx(1.2)
    positive, negative = (result['directions'][key] for key in ('w_positive', 'w_negative'))
    assert positive['queries'] == negative['queries'] == 3
    assert positive['expected_credits']['S'] == pytest.approx(2.2)
    assert positive['observed_credits'] == 2
    assert positive['signed_errors']['S'] == pytest.approx(.2)
    assert negative['expected_credits']['S'] == pytest.approx(.8)
    assert negative['observed_credits'] == 1
    assert negative['signed_errors']['S'] == pytest.approx(.2)
    assert positive['source_correction_sum']+negative['source_correction_sum'] == pytest.approx(.8)
    json.dumps(result, allow_nan=False)


def test_full_context_aggregation_and_merge_retain_additive_loss_strata():
    inputs = example()
    result = decision_decomposition(*inputs)
    contexts = {tuple(row['context']): row for row in result['contexts']}
    same = contexts[('same', 1)]
    assert same['flights'] == 2
    assert same['service_queries'] == 26
    assert same['active_queries'] == 4
    assert same['source_weight_sum'] == 1
    assert same['n_target_sum'] == 40
    assert same['source_correction_sum'] == pytest.approx(1)
    assert same['generic_count_component'] == pytest.approx(0)
    assert same['conditional_source_component'] == pytest.approx(1)
    first = DecompositionAccumulator().update(result)
    reverse_actions = dict(S=1, T=0, P=0)
    reverse = decision_decomposition(*inputs[:3], reverse_actions, inputs[4])
    combined = DecompositionAccumulator().merge(first.result()).update(reverse).result()
    assert combined['decisions'] == 2
    assert combined['by_label_sign']['positive']['decisions'] == 1
    assert combined['by_label_sign']['negative']['decisions'] == 1
    assert next(row for row in combined['by_label_sign']['negative']['contexts']
                if row['context'] == ['same', 1])['source_correction_sum'] == pytest.approx(-1)
    assert next(row for row in combined['by_label_sign']['positive']['contexts']
                if row['context'] == ['same', 1])['source_correction_sum'] == pytest.approx(1)
    assert combined['totals']['source_correction_sum'] == pytest.approx(0)
    same = next(row for row in combined['contexts'] if row['context'] == ['same', 1])
    assert same['decisions'] == 2
    assert same['flights'] == 4
    assert same['source_weight_sum'] == 2


def test_placebo_contrast_and_inconsistent_label_identity():
    aligned, flights, captures, actions, labels = example()
    result = decision_decomposition(aligned, flights, captures, actions, labels, contrast=('S', 'P'))
    assert result['source_correction_sum'] == pytest.approx(1.2)
    assert result['baseline_error_sum'] == pytest.approx(-.8)
    assert result['source_error_sum'] == pytest.approx(.4)
    with pytest.raises(ValueError, match='does not reproduce'):
        decision_decomposition(aligned, flights, captures, actions, labels+[1, 0])
