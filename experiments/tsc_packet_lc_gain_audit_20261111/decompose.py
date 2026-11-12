"""Arithmetic exact/mixed contribution audit of saved LC concordances."""

from collections import Counter, defaultdict
from math import fsum, isclose, isfinite


DIRECTIONS = {0: 'left', 2: 'right'}
METRICS = ('direction_log_probability', 'conditional_log_probability')
GROUPS = ('EE', 'EM', 'ME', 'MM')
TOLERANCE = 1e-10


def _identity(row):
    ref = row['reference']
    return int(ref['seed']), int(ref['time_sec']), ref['packet_id']


def _records(rows):
    result = {}
    for row in rows:
        key = _identity(row)
        if key in result:
            raise ValueError('The fixed readout records contain duplicate references')
        if row['proposed_method'] not in ('exact', 'mixture256'):
            raise ValueError('Only the registered exact and mixture256 methods are allowed')
        if not isfinite(row['objective_weight']) or row['objective_weight'] <= 0:
            raise ValueError('Every registered objective weight must remain finite and positive')
        result[key] = row
    return result


def _score(row, point, direction, metric):
    value = row['points'][point][str(direction)][metric]
    if value['state'] == 'finite':
        if not isfinite(value['value']):
            raise ValueError('A saved finite log score is nonfinite')
        return value['value']
    if value['state'] == 'negative_infinity':
        return -float('inf')
    if value['state'] == 'undefined':
        return None
    raise ValueError('An unexpected saved log-probability state was encountered')


def _rank(a, b):
    if a is None or b is None:
        return 'undefined'
    if a == b or abs(a-b) <= TOLERANCE:
        return 'ties'
    return 'wins' if a > b else 'losses'


def _concordance(outcome):
    return {'wins':1., 'ties':.5, 'losses':0., 'undefined':None}[outcome]


def _group_readout(edges):
    result = dict(edges=len(edges), positive_count=len({edge['positive'] for edge in edges}),
                  coefficient_mass=fsum(edge['coefficient'] for edge in edges))
    for point in ('old', 'new'):
        defined = [edge for edge in edges if edge[point] != 'undefined']
        unknown = [edge for edge in edges if edge[point] == 'undefined']
        contribution = fsum(edge['coefficient']*_concordance(edge[point]) for edge in defined)
        counts = Counter(edge[point] for edge in edges)
        result.update({f'{point}_contribution':contribution if not unknown else None,
            f'{point}_defined_contribution':contribution,
            f'{point}_undefined_edge_count':len(unknown),
            f'{point}_undefined_coefficient_mass':fsum(edge['coefficient'] for edge in unknown),
            f'{point}_rank_counts':{name:counts[name] for name in ('wins','ties','losses','undefined')}})
    jointly_defined = [edge for edge in edges if edge['old'] != 'undefined' and edge['new'] != 'undefined']
    unknown = [edge for edge in edges if edge['old'] == 'undefined' or edge['new'] == 'undefined']
    gain = fsum(edge['coefficient']*(_concordance(edge['new'])-_concordance(edge['old'])) for edge in jointly_defined)
    result.update(gain_contribution=gain if not unknown else None,
        jointly_defined_gain_contribution=gain, undefined_gain_edge_count=len(unknown),
        undefined_gain_coefficient_mass=fsum(edge['coefficient'] for edge in unknown))
    return result


def _same_number(actual, expected, message):
    if actual is None or expected is None:
        if actual is not expected:
            raise ArithmeticError(message)
        return 0.
    if not isclose(actual, expected, rel_tol=0., abs_tol=1e-12):
        raise ArithmeticError(message)
    return abs(actual-expected)


def _metric_readout(edges, old_expected, new_expected, positive_count, covered_count):
    groups = {name:_group_readout([edge for edge in edges if edge['group'] == name]) for name in GROUPS}
    all_edges = _group_readout(edges)
    old_score = all_edges['old_contribution'] if covered_count else None
    new_score = all_edges['new_contribution'] if covered_count else None
    gain = new_score-old_score if old_score is not None and new_score is not None else None
    exact_old, exact_new = groups['EE']['old_contribution'], groups['EE']['new_contribution']
    exact_gain = exact_new-exact_old if exact_old is not None and exact_new is not None else None
    mixed_mass = fsum(groups[name]['coefficient_mass'] for name in ('EM','ME','MM'))
    defined = covered_count > 0 and old_score is not None and new_score is not None
    errors = []
    for actual, expected in ((old_score, old_expected['covered_positive_mean_of_weighted_neighbor_concordance']),
                             (new_score, new_expected['covered_positive_mean_of_weighted_neighbor_concordance'])):
        errors.append(_same_number(actual, expected, 'Saved covered-positive concordance did not reconstruct'))
    for point, actual, expected in (('old', old_score, old_expected), ('new', new_score, new_expected)):
        full = actual if covered_count == positive_count else None
        errors.append(_same_number(full, expected['equal_positive_mean_of_weighted_neighbor_concordance'],
                                   f'Saved {point} full-positive concordance did not reconstruct'))
    if defined:
        errors.append(_same_number(fsum(groups[name]['old_contribution'] for name in GROUPS), old_score,
                                   'Old component contributions did not sum to the fixed score'))
        errors.append(_same_number(fsum(groups[name]['new_contribution'] for name in GROUPS), new_score,
                                   'New component contributions did not sum to the fixed score'))
        errors.append(_same_number(fsum(groups[name]['gain_contribution'] for name in GROUPS), gain,
                                   'Component gains did not sum to the fixed score change'))
    interval = [exact_gain-mixed_mass, exact_gain+mixed_mass] if defined else None
    return dict(groups=groups, old_score=old_score, new_score=new_score, gain=gain,
        exact_old_contribution=exact_old, exact_new_contribution=exact_new, exact_gain=exact_gain,
        mixed_coefficient_mass=mixed_mass,
        old_score_interval=[exact_old, exact_old+mixed_mass] if covered_count and old_score is not None else None,
        new_score_interval=[exact_new, exact_new+mixed_mass] if covered_count and new_score is not None else None,
        gain_interval=interval, gain_lower_bound_positive=interval[0] > 0 if interval is not None else None,
        old_undefined_edge_count=all_edges['old_undefined_edge_count'],
        new_undefined_edge_count=all_edges['new_undefined_edge_count'],
        old_undefined_coefficient_mass=all_edges['old_undefined_coefficient_mass'],
        new_undefined_coefficient_mass=all_edges['new_undefined_coefficient_mass'],
        bounds_state='DEFINED' if defined else 'UNDEFINED_SAVED_CONCORDANCE' if covered_count else 'NO_COVERED_POSITIVES',
        maximum_reconstruction_error=max(errors, default=0.))


def audit_edges(old_records, new_records, old_summary, new_summary):
    """Keep the original covered positives and full per-positive denominators.

    Inputs contain saved extended log scores at points ``old`` and ``new``.
    Summaries are the original matched-summary dictionaries keyed left/right.
    Undefined comparisons remain undefined; no completion or renormalization is
    used. Bounds concern mixed-edge numerical uncertainty in this fixed panel.
    """
    old, new = _records(old_records), _records(new_records)
    if old.keys() != new.keys():
        raise ValueError('Old and new readouts must cover identical registered references')
    for key, row in old.items():
        other = new[key]
        if any(row[name] != other[name] for name in
               ('case_index','seed','y','raw_weight','objective_weight','proposed_method')):
            raise ValueError('A registered identity, label, weight or method changed')
        if row['y'] == 1:
            for name in DIRECTIONS.values():
                if row['directions'][name]['matched_positive_keys'] != other['directions'][name]['matched_positive_keys']:
                    raise ValueError('The original registered neighbour edges changed')
    positives = {key:row for key,row in old.items() if row['y'] != 1}
    neighbors = defaultdict(list)
    for key, stay in old.items():
        if stay['y'] != 1:
            continue
        for direction, name in DIRECTIONS.items():
            matched = [tuple(value) for value in stay['directions'][name]['matched_positive_keys']]
            if len(matched) != len(set(matched)):
                raise ValueError('A registered positive-stay edge was duplicated')
            for positive in matched:
                if positive not in positives or positives[positive]['y'] != direction:
                    raise ValueError('An edge points outside the fixed direction-positive cohort')
                neighbors[positive].append(key)
    output = {}
    for direction, name in DIRECTIONS.items():
        selected = [key for key,row in positives.items() if row['y'] == direction]
        covered = [key for key in selected if neighbors[key]]
        count = len(covered)
        edges = []
        for positive in covered:
            denominator = fsum(old[stay]['objective_weight'] for stay in neighbors[positive])
            for stay in neighbors[positive]:
                group = ''.join('E' if old[key]['proposed_method'] == 'exact' else 'M' for key in (positive, stay))
                edges.append(dict(positive=positive, stay=stay, group=group,
                                  coefficient=old[stay]['objective_weight']/(count*denominator)))
        expected = dict(positive_count=len(selected), positive_with_selected_neighbors=count,
                        zero_neighbor_positive_count=len(selected)-count, registered_positive_stay_edges=len(edges))
        for summary in (old_summary[name]['all'], new_summary[name]['all']):
            if any(summary[key] != value for key,value in expected.items()):
                raise ValueError('The saved matched denominator or positive cohort changed')
        mass = fsum(edge['coefficient'] for edge in edges)
        _same_number(mass, 1. if count else 0., 'The fixed per-positive edge coefficients do not sum to one')
        metrics = {}
        for metric in METRICS:
            comparisons = [dict(edge, old=_rank(_score(old[edge['positive']], 'old', direction, metric),
                _score(old[edge['stay']], 'old', direction, metric)),
                new=_rank(_score(new[edge['positive']], 'new', direction, metric),
                _score(new[edge['stay']], 'new', direction, metric))) for edge in edges]
            metrics[metric] = _metric_readout(comparisons, old_summary[name]['all']['metrics'][metric],
                new_summary[name]['all']['metrics'][metric], len(selected), count)
        ee_positives = {edge['positive'] for edge in edges if edge['group'] == 'EE'}
        output[name] = dict(**expected, coefficient_mass=mass,
            positives_with_EE_neighbors=len(ee_positives), positives_with_only_mixed_neighbors=count-len(ee_positives),
            metrics=metrics)
    return dict(directions=output, saved_summary_reconstructed=True,
        coefficient_definition='a_pj = stay objective_weight / (covered positive count * full registered neighbor weight sum for positive p)',
        bounds_scope='Exact-exact contribution fixed; all mixed-affected concordances may vary in [0,1]. Same covered positives and all neighbor denominators at both points. No independence or control-benefit claim.',
        undefined_policy='Saved undefined comparisons retain their counts and coefficient mass; full scores, gains and bounds remain undefined rather than being completed or dropped.')
