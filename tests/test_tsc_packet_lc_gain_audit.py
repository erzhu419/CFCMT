"""Pure arithmetic checks of the fixed-cohort matched-gain decomposition."""

from copy import deepcopy
from math import log

import pytest

from experiments.tsc_packet_lc_gain_audit_20261111.decompose import audit_edges


METRICS = ('direction_log_probability', 'conditional_log_probability')


def probe(probability, *, conditional=True):
    value = dict(state='finite', value=log(probability)) if probability else dict(state='negative_infinity', value=None)
    return dict(direction_log_probability=value,
        conditional_log_probability=dict(value) if conditional else dict(state='undefined', value=None))


def records(specifications, links):
    """Use stored scalar scores and metadata only, with no model construction."""
    old, new = [], []
    keys = {name:[76116901, 10+index, name] for index, (name, *_) in enumerate(specifications)}
    for index, (name, label, method, weight, before, after, conditional) in enumerate(specifications):
        metadata = dict(case_index=index, seed=76116901,
            reference=dict(zip(('seed', 'time_sec', 'packet_id'), keys[name])), y=label,
            raw_weight=weight, objective_weight=weight/30000., proposed_method=method,
            directions={direction:dict(matched_positive_keys=[keys[p] for p in links.get(name, ())]
                if direction == 'left' else []) for direction in ('left', 'right')})
        old.append(dict(deepcopy(metadata), points={'old':{'0':probe(before, conditional=conditional)}}))
        new.append(dict(deepcopy(metadata), points={'new':{'0':probe(after, conditional=conditional)}}))
    return old, new


def summaries(positives, covered, edges, before, after, *, conditional_before=None, conditional_after=None):
    result = []
    for marginal, conditional in ((before, conditional_before), (after, conditional_after)):
        conditional = marginal if conditional == 'same' else conditional
        left = dict(positive_count=positives, positive_with_selected_neighbors=covered,
            zero_neighbor_positive_count=positives-covered, registered_positive_stay_edges=edges,
            metrics={metric:dict(covered_positive_mean_of_weighted_neighbor_concordance=value,
                equal_positive_mean_of_weighted_neighbor_concordance=value if positives == covered else None)
                for metric, value in zip(METRICS, (marginal, conditional), strict=True)})
        right = dict(positive_count=0, positive_with_selected_neighbors=0,
            zero_neighbor_positive_count=0, registered_positive_stay_edges=0,
            metrics={metric:dict(covered_positive_mean_of_weighted_neighbor_concordance=None,
                equal_positive_mean_of_weighted_neighbor_concordance=None) for metric in METRICS})
        result.append(dict(left={'all':left}, right={'all':right}))
    return result


def mixed_fixture():
    # B has only mixed neighbours. S and T are each reused across positives.
    specifications = [
        ('A', 0, 'exact', 1., .2, .8, True),
        ('B', 0, 'exact', 1., .6, .6, True),
        ('C', 0, 'mixture256', 1., .4, .5, True),
        ('D', 0, 'exact', 1., .3, .3, True),
        ('S', 1, 'exact', 1., .3, .3, True),
        ('T', 1, 'mixture256', 3., .7, .6, True),
        ('U', 1, 'mixture256', 1., .4, .4, True),
    ]
    old, new = records(specifications, {'S':('A', 'C'), 'T':('A', 'B'), 'U':('C',)})
    before, after = summaries(4, 3, 5, 1/4, 5/6, conditional_before='same', conditional_after='same')
    return old, new, before, after


def test_fixed_covered_cohort_full_neighbor_denominators_and_all_four_method_groups():
    result = audit_edges(*mixed_fixture())
    left = result['directions']['left']
    assert left['positive_count'] == 4
    assert left['positive_with_selected_neighbors'] == 3
    assert left['zero_neighbor_positive_count'] == 1
    assert left['positives_with_EE_neighbors'] == 1
    assert left['positives_with_only_mixed_neighbors'] == 2
    for metric in METRICS:
        value = left['metrics'][metric]
        groups = value['groups']
        assert {name:groups[name]['edges'] for name in groups} == dict(EE=1, EM=2, ME=1, MM=1)
        assert {name:groups[name]['positive_count'] for name in groups} == dict(EE=1, EM=2, ME=1, MM=1)
        for name, mass in dict(EE=1/12, EM=7/12, ME=1/6, MM=1/6).items():
            assert groups[name]['coefficient_mass'] == pytest.approx(mass)
        assert groups['MM']['old_rank_counts']['ties'] == 1
        assert groups['EM']['new_rank_counts']['ties'] == 1
        assert groups['EE']['old_contribution'] == 0.
        assert groups['EE']['new_contribution'] == pytest.approx(1/12)
        assert groups['ME']['old_contribution'] == pytest.approx(1/6)
        assert groups['MM']['old_contribution'] == pytest.approx(1/12)
        assert value['old_score'] == pytest.approx(1/4)
        assert value['new_score'] == pytest.approx(5/6)
        assert value['gain'] == pytest.approx(7/12)
        assert value['exact_gain'] == pytest.approx(1/12)
        assert value['mixed_coefficient_mass'] == pytest.approx(11/12)
        assert value['old_score_interval'] == pytest.approx([0., 11/12])
        assert value['new_score_interval'] == pytest.approx([1/12, 1.])
        assert value['gain_interval'] == pytest.approx([-5/6, 1.])
        assert value['gain_lower_bound_positive'] is False


def test_gain_lower_bound_uses_delta_exact_plus_or_minus_mixed_mass_once():
    old, new = records([
        ('A', 0, 'exact', 1., .1, .8, True),
        ('E', 1, 'exact', 9., .5, .5, True),
        ('M', 1, 'mixture256', 1., .4, .4, True)], {'E':('A',), 'M':('A',)})
    before, after = summaries(1, 1, 2, 0., 1., conditional_before='same', conditional_after='same')
    result = audit_edges(old, new, before, after)['directions']['left']['metrics']['direction_log_probability']
    assert result['exact_gain'] == pytest.approx(.9)
    assert result['mixed_coefficient_mass'] == pytest.approx(.1)
    assert result['gain_interval'] == pytest.approx([.8, 1.])
    assert result['gain_lower_bound_positive'] is True


def test_undefined_conditional_edge_keeps_its_mass_and_is_neither_tie_nor_deleted():
    old, new = records([
        ('A', 0, 'exact', 1., .5, .6, True),
        ('M', 1, 'mixture256', 2., 0., 0., False)], {'M':('A',)})
    before, after = summaries(1, 1, 1, 1., 1.)
    result = audit_edges(old, new, before, after)['directions']['left']
    marginal = result['metrics']['direction_log_probability']
    assert marginal['old_score'] == marginal['new_score'] == 1.
    conditional = result['metrics']['conditional_log_probability']
    group = conditional['groups']['EM']
    assert group['coefficient_mass'] == 1.
    assert group['old_undefined_coefficient_mass'] == group['new_undefined_coefficient_mass'] == 1.
    assert group['old_rank_counts']['undefined'] == group['new_rank_counts']['undefined'] == 1
    assert group['old_rank_counts']['ties'] == group['new_rank_counts']['ties'] == 0
    assert group['old_contribution'] is group['new_contribution'] is None
    assert conditional['old_score'] is conditional['new_score'] is conditional['gain'] is None
    assert conditional['gain_interval'] is None


def test_saved_summary_reconstruction_mismatch_stops_the_arithmetic_audit():
    old, new, before, after = mixed_fixture()
    after['left']['all']['metrics']['direction_log_probability']['covered_positive_mean_of_weighted_neighbor_concordance'] = .5
    with pytest.raises((ValueError, ArithmeticError)):
        audit_edges(old, new, before, after)
