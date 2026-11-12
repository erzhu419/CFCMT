"""Exact-key joins and actual-label support in saved-array comparisons."""
import numpy as np
import pytest

from experiments.tsc_packet_discrete_braking_20261110.compare import (
    _align, _arrays, _feature_changes, _head_comparison, _log_probability, _motion_vectors, _sum_counts,
)


def row(pid, y):
    return dict(head='motion', reference=dict(seed=76116901, time_sec=3., packet_id=pid, y=y),
                candidate_intents=['END'], prior=[1.])


def test_exact_key_alignment_reorders_rows_arrays_and_feature_change_columns():
    first, second = row('a', 2.), row('b', 4.)
    order = _align([first, second], [second, first])
    np.testing.assert_array_equal(order, [1, 0])
    with pytest.raises(ValueError, match='label changed'):
        _align([first, second], [row('b', 5.), first])
    archive = dict(motion_prior=np.ones((2, 1)), motion_risk=np.ones((2, 1), bool),
        motion_y=np.asarray([4., 2.]), motion_retained=np.ones(2, bool),
        motion_zero_log_probability=np.zeros(2), motion_mean0_mps=np.asarray([[4.], [2.]]),
        motion_physical_ceiling_mps=np.asarray([[6.], [5.]]))
    aligned = _arrays(archive, 'motion', order)
    np.testing.assert_array_equal(aligned['y'], [2., 4.])
    old_x = np.zeros((2, 1, 38)); new_x = old_x.copy()
    new_x[0, 0, 7] = 3.
    names = ['feature'+str(index) for index in range(38)]
    assert _feature_changes(old_x, new_x, order, np.ones((2, 1)), names) == [
        dict(name='feature7', changed_components=1, maximum_absolute_change=3.)]


def test_support_is_label_and_positive_prior_specific_not_retention_or_zero_mass_ceiling():
    old = dict(prior=np.asarray([[0., 1.], [0., 1.]]), risk=np.ones((2, 2), bool), y=np.asarray([0., 1.]),
        retained=np.ones(2, bool), mask=np.asarray([[[True, True, False], [False, True, True]],
                                                  [[True, True, False], [False, True, True]]]))
    new = {key: value.copy() for key, value in old.items()}
    new['mask'][0, 1] = [True, True, False]
    before, after = _log_probability('lane_change', old), _log_probability('lane_change', new)
    result = _head_comparison('lane_change', old, new, before, after)
    assert result['baseline']['retained'] == result['candidate']['retained'] == 2
    assert result['positive_labels']['gained'] == 1 and result['positive_labels']['lost'] == 0
    assert result['negative_labels']['supported_both'] == 1
    assert result['baseline']['full_mean_nll'] is None
    assert result['candidate']['full_mean_nll'] == pytest.approx(np.log(2))
    assert _sum_counts([dict(gained=0, lost=0), dict(gained=1, lost=0)]) == dict(gained=1, lost=0)
    motion = dict(prior=np.asarray([[0., 1.]]), y=np.asarray([5.]),
        mean0_mps=np.asarray([[100., 3.]]), physical_ceiling_mps=np.asarray([[100., 4.]]))
    vectors = _motion_vectors(motion)
    assert vectors['physical_deficit_mps'][0] == 1
    assert vectors['mean_deficit_mps'][0] == 2

    # y0 remains supported when risk changes, but its permission can be newly
    # granted; these two quantities must remain distinct in the comparison.
    old_binary = dict(prior=np.ones((2, 1)), risk=np.zeros((2, 1), bool),
                      y=np.asarray([0., 1.]), retained=np.zeros(2, bool))
    new_binary = {key: value.copy() for key, value in old_binary.items()}
    new_binary['risk'][:] = True; new_binary['retained'][:] = True
    compared = _head_comparison('yield_entry', old_binary, new_binary,
        _log_probability('yield_entry', old_binary), _log_probability('yield_entry', new_binary))
    assert compared['support_by_label']['0']['gained'] == 0
    assert compared['risk_permission_transition_by_label']['0']['gained'] == 1
    assert compared['support_by_label']['1']['gained'] == 1
