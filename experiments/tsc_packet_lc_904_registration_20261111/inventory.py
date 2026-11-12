"""Inventory paid 904 laws from scalar receipts and register symbolic costs."""

import json
from math import prod
from pathlib import Path

from experiments.tsc_packet_positive_covered_registration_20261111.inventory import (
    cache_inventory as old_inventory,
)


SEED = 76116904


def key(row):
    return int(row['seed']), int(row['time_sec']), row['packet_id']


def cache_inventory(base, wanted_keys):
    """Add existing paired exact laws without opening any numerical cache."""
    base, wanted = Path(base), set(wanted_keys)
    if any(identity[0] != SEED for identity in wanted):
        raise ValueError('Only the frozen 904 population may enter this inventory')
    available, previous = old_inventory(base, wanted)
    anchor = base/'target_packet_particle_first_fit/fit/coefficients_server_only.json'
    for pools in available.values():
        for receipt in pools['mixture']:
            if Path(receipt['anchor_file']) != anchor:
                raise ValueError('A mixed receipt changed the original frozen proposal anchor')
    root = base/'target_packet_lane_change_holdout/evaluation'
    result = json.loads((root/'result.json').read_text())
    tasks = json.loads((root/'registered_server_only.json').read_text())['cases']
    if (result['status'] != 'COMPLETE_904_EXACT_PAIRED_HOLDOUT'
            or result['defined_cases'] != 124 or result['failures'] or result['incomplete_started_cases']
            or len(tasks) != 124 or len({key(task['reference']) for task in tasks}) != 124
            or any(task['reference']['seed'] != SEED for task in tasks)):
        raise ValueError('The complete original 124-case 904 exact receipts are required')
    added = 0
    for task in tasks:
        identity = key(task['reference'])
        if identity not in wanted:
            continue
        summary = json.loads((root/task['stem']/'summary.json').read_text())
        if (summary['status'] != 'COMPLETE_EXACT_CASE' or summary['reference'] != task['reference']
                or any(summary[name] != task[name] for name in
                       ('case_index', 'pair_index', 'role', 'direction', 'time_sec'))):
            raise ValueError('A paid exact law changed its registered 904 identity')
        path = root/task['stem']/'cache_server_only.json.gz'
        if not path.is_file():
            raise FileNotFoundError(path)
        if any(receipt['particle_path'] == str(path) for receipt in available[identity]['exact']):
            continue
        available[identity]['exact'].append(dict(
            kind='exact', source='lane_change_holdout', particle_path=str(path), particle_count=None,
            source_case_index=task['case_index'], reference=task['reference'],
            stochastic_participants=task['required_participants'], frame_actor_count=task['frame_actor_count']))
        added += 1
    summary = dict(previous_inventory=previous, added_completed_904_exact_caches=added,
        references_with_exact=sum(bool(pools['exact']) for pools in available.values()),
        references_with_mixture=sum(bool(pools['mixture']) for pools in available.values()),
        references_with_q0=sum(bool(pools['q0']) for pools in available.values()),
        references_with_reusable_kind=sum(bool(pools['exact'] or pools['mixture']) for pools in available.values()),
        references_without_reusable_kind=sum(not (pools['exact'] or pools['mixture']) for pools in available.values()),
        numerical_cache_arrays_read=False, coefficient_arrays_read=False,
        new_full_forward_calls=0, new_canonical_feature_evaluations=0)
    return available, summary


def cost_plan(budget, participants, frame_actors, available):
    """Prefer compatible paid laws before deciding how to build a missing law.

    Mixture receipts originate from cache_inventory's original-anchor inventory;
    a q0 pool is never a substitute for the fixed balanced proposal.
    """
    if budget['status'] != 'REGISTERED_SYMBOLIC_UPPER_BOUND':
        return dict(proposed_method=None, proposed_reused_cache=None,
            current_goal_combinations=None, all_exact_full_forward_ceiling=None,
            all_exact_feature_ceiling=None, all_exact_leaf_ceiling=None,
            all_mixture_full_forward_ceiling=None, all_mixture_feature_ceiling=None,
            proposed_new_full_forward_ceiling=None, proposed_new_feature_ceiling=None,
            proposed_new_exact_leaf_ceiling=None)
    actors = len(participants)
    goals = prod(row['current_positive_goals'] for row in budget['per_actor'].values())

    def compatible(receipt):
        return (receipt['stochastic_participants'] == participants
                and receipt['frame_actor_count'] == frame_actors)

    exact = [receipt for receipt in available['exact']
             if compatible(receipt) and receipt['kind'] == 'exact' and receipt['particle_count'] is None]
    mixed = [receipt for receipt in available['mixture']
             if compatible(receipt) and receipt['kind'] == 'mixture' and receipt['particle_count'] == 256]
    if exact:
        method, reused = 'exact', exact[0]
    elif mixed:
        method, reused = 'mixture256', mixed[0]
    else:
        method = 'exact' if budget['physical_resolver_calls_upper_bound'] <= 256 else 'mixture256'
        reused = None
    return dict(proposed_method=method, proposed_reused_cache=reused,
        current_goal_combinations=goals,
        all_exact_full_forward_ceiling=budget['physical_resolver_calls_upper_bound'],
        all_exact_feature_ceiling=actors*goals,
        all_exact_leaf_ceiling=budget['lateral_leaf_branches_upper_bound'],
        all_mixture_full_forward_ceiling=256,
        all_mixture_feature_ceiling=actors*min(goals, 256),
        proposed_new_full_forward_ceiling=0 if reused else budget['physical_resolver_calls_upper_bound'] if method == 'exact' else 256,
        proposed_new_feature_ceiling=0 if reused else actors*(goals if method == 'exact' else min(goals, 256)),
        proposed_new_exact_leaf_ceiling=0 if reused or method != 'exact' else budget['lateral_leaf_branches_upper_bound'])
