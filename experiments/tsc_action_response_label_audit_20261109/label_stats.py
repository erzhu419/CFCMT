"""Exact within-state response classes and their observed stopping-cost ranges."""
from itertools import combinations
from math import isfinite
from statistics import mean


def _integer(value):
    if not isfinite(value) or value < 0 or value != int(value):
        raise ValueError('Measured counts and stopping seconds must be nonnegative integers')
    return int(value)


def _range(members, stopped_seconds, horizon, lanes):
    minimum, maximum = min(stopped_seconds[a] for a in members), max(stopped_seconds[a] for a in members)
    return dict(stopped_vehicle_seconds_min=minimum, stopped_vehicle_seconds_max=maximum,
        stopped_vehicle_seconds_span=maximum-minimum,
        mean_halted_per_lane_min=minimum/(horizon*lanes),
        mean_halted_per_lane_max=maximum/(horizon*lanes),
        mean_halted_per_lane_span=(maximum-minimum)/(horizon*lanes))


def audit_group(case_id, city, seed, Y, costs_450, reference_index, controlled_lane_count,
                stopped_seconds_10, stopped_seconds_450):
    """Inputs use the verified canonical candidate order 0..7, including PP.

    Integer stopping totals come from the runner's native-grid/series audit.
    Response comparison uses every ordered movement and both measured outputs.
    """
    if len(Y) != 8 or len(costs_450) != 8 or len(stopped_seconds_10) != 8 or len(stopped_seconds_450) != 8:
        raise ValueError('The fixed group requires eight canonical action rows')
    if reference_index not in range(8) or controlled_lane_count <= 0:
        raise ValueError('The PP reference and controlled lane count must match the native group')
    movement_count = len(Y[0])
    if not movement_count or any(len(action) != movement_count or any(len(row) != 2 for row in action) for action in Y):
        raise ValueError('Full responses must have a common ordered movement-by-two-output roster')
    stopped10 = [_integer(value) for value in stopped_seconds_10]
    stopped450 = [_integer(value) for value in stopped_seconds_450]
    classes = {}
    for action, response in enumerate(Y):
        vector = tuple(_integer(value) for movement in response for value in movement)
        classes.setdefault(vector, []).append(action)
    output, identical_pairs, different10, different450, equal10_different450 = [], 0, 0, 0, 0
    for members in classes.values():
        for a, b in combinations(members, 2):
            identical_pairs += 1
            different10 += stopped10[a] != stopped10[b]
            different450 += stopped450[a] != stopped450[b]
            equal10_different450 += stopped10[a] == stopped10[b] and stopped450[a] != stopped450[b]
        output.append(dict(class_index=len(output), members=members, action_count=len(members),
            contains_pp=reference_index in members,
            range_450=_range(members, stopped450, 450, controlled_lane_count),
            range_10=_range(members, stopped10, 10, controlled_lane_count)))
    pp_class = next(row for row in output if row['contains_pp'])
    deltas = [dict(candidate_index=action,
        stopped_vehicle_seconds_450=stopped450[action]-stopped450[reference_index],
        mean_halted_per_lane_450=(stopped450[action]-stopped450[reference_index])/(450*controlled_lane_count),
        stopped_vehicle_seconds_10=stopped10[action]-stopped10[reference_index],
        mean_halted_per_lane_10=(stopped10[action]-stopped10[reference_index])/(10*controlled_lane_count))
        for action in pp_class['members'] if action != reference_index]
    statistics = dict(response_classes=len(output), identical_Y_pairs=identical_pairs,
        sameY_pair_cost10_different=different10, sameY_pair_cost450_different=different450,
        sameY_pair_cost10_equal_but450_different=equal10_different450,
        has_identical_response_pair=int(identical_pairs > 0))
    for horizon in (450, 10):
        ranges = [row['range_'+str(horizon)] for row in output]
        physical = [row['stopped_vehicle_seconds_span'] for row in ranges]
        per_lane = [row['mean_halted_per_lane_span'] for row in ranges]
        pp = pp_class['range_'+str(horizon)]
        statistics.update({
            'alias_group_'+str(horizon): int(any(value > 0 for value in physical)),
            'pp_alias_group_'+str(horizon): int(pp['stopped_vehicle_seconds_span'] > 0),
            'mean_class_span_'+str(horizon)+'_vehicle_seconds': mean(physical),
            'max_class_span_'+str(horizon)+'_vehicle_seconds': max(physical),
            'pp_class_span_'+str(horizon)+'_vehicle_seconds': pp['stopped_vehicle_seconds_span'],
            'mean_class_span_'+str(horizon)+'_per_lane': mean(per_lane),
            'max_class_span_'+str(horizon)+'_per_lane': max(per_lane),
            'pp_class_span_'+str(horizon)+'_per_lane': pp['mean_halted_per_lane_span']})
    return dict(case_id=case_id, city=city, seed=seed, reference_index=reference_index,
        controlled_lane_count=controlled_lane_count, movement_count=movement_count,
        costs_450=[float(value) for value in costs_450],
        stopped_seconds_10=stopped10, stopped_seconds_450=stopped450,
        classes=output, pp_class_index=pp_class['class_index'], pp_same_response_deltas=deltas,
        statistics=statistics)


def summarize(groups):
    """Two groups within seed, then equal seeds; cities remain separate."""
    cities = {}
    for city in sorted({group['city'] for group in groups}):
        selected = [group for group in groups if group['city'] == city]
        per_seed = []
        for seed in sorted({group['seed'] for group in selected}):
            rows = [group for group in selected if group['seed'] == seed]
            if len(rows) != 2:
                raise ValueError('Every declared seed must retain both fixed native groups')
            per_seed.append(dict(seed=seed, group_ids=[row['case_id'] for row in rows],
                group_mean_statistics={key: mean(row['statistics'][key] for row in rows) for key in rows[0]['statistics']}))
        totals = {key: sum(group['statistics'][key] for group in selected)
                  for key in ('identical_Y_pairs', 'sameY_pair_cost10_different', 'sameY_pair_cost450_different',
                              'sameY_pair_cost10_equal_but450_different', 'has_identical_response_pair',
                              'alias_group_450', 'pp_alias_group_450', 'alias_group_10', 'pp_alias_group_10')}
        cities[city] = dict(seed_count=len(per_seed), group_count=len(selected), groups=selected, per_seed=per_seed,
            equal_seed_statistics={key: mean(row['group_mean_statistics'][key] for row in per_seed)
                                   for key in per_seed[0]['group_mean_statistics']},
            descriptive_totals=totals)
    return dict(group_count=len(groups), branch_count=8*len(groups),
        aggregation='All response classes including singletons and zero spans; equal groups within seed, then equal seeds within city.',
        pair_counts='Descriptive overlapping action pairs, not independent observations.', cities=cities)
