"""Reuse completed numerical caches by reference and proposal identity."""

import json
from pathlib import Path

from experiments.tsc_packet_positive_covered_registration_20261111.inventory import cache_inventory as old_inventory


def key(row):
    return int(row['seed']), int(row['time_sec']), row['packet_id']


def cache_inventory(base, wanted):
    available, old_summary = old_inventory(base, wanted)
    root = base/'target_packet_positive_covered_cache/evaluation'
    result = json.loads((root/'result.json').read_text())
    if result['status'] != 'COMPLETE_POSITIVE_COVERED_TWO_POINT_CACHE' or not result['primary_two_point_support_passed']:
        raise ValueError('The completed424-case cache receipts are required')
    prepared = json.loads((root/'prepared_server_only.json').read_text())['cases']
    added = {'exact':0, 'mixture':0}
    for task in prepared:
        identity = key(task['reference'])
        if identity not in available:
            continue
        row = json.loads((root/task['stem']/'summary.json').read_text())
        if (row['status'] != 'COMPLETE_POSITIVE_COVERED_CACHE_CASE'
                or row['reference'] != task['reference'] or row['proposed_method'] != task['proposed_method']):
            raise ValueError('A completed cache receipt changed identity or method')
        kind = 'exact' if row['proposed_method']=='exact' else 'mixture'
        path = Path(row['cache_path'])
        if not path.is_file():
            raise FileNotFoundError(path)
        if any(old['particle_path']==str(path) for old in available[identity][kind]):
            continue
        receipt = dict(kind=kind, source='positive_covered_cache', particle_path=str(path),
            particle_count=None if kind=='exact' else 256, source_case_index=task['case_index'],
            reference=task['reference'], stochastic_participants=task['required_participants'],
            frame_actor_count=task['frame_actor_count'])
        if kind=='mixture':
            receipt.update(anchor_file=str(base/'target_packet_particle_first_fit/fit/coefficients_server_only.json'),
                denominator='0.5*q0 + 0.5*qf at the original saved frozen anchor',
                seed_zero=task['seed_zero'], seed_fitted=task['seed_fitted'])
        available[identity][kind].append(receipt)
        added[kind]+=1
    summary=dict(previous_inventory=old_summary,added_completed_caches=added,
        references_with_exact=sum(bool(v['exact']) for v in available.values()),
        references_with_mixture=sum(bool(v['mixture']) for v in available.values()),
        references_with_any_cache=sum(any(v.values()) for v in available.values()),
        references_without_cache=sum(not any(v.values()) for v in available.values()),
        numerical_cache_arrays_read=False)
    return available,summary


def cost_plan(budget, participants, frame_actors, available):
    """Choose the existing exact/mixture rule; never construct a numerical law."""
    if budget['status']!='REGISTERED_SYMBOLIC_UPPER_BOUND':
        return dict(proposed_method=None,proposed_reused_cache=None,
            proposed_new_full_forward_ceiling=None,proposed_new_feature_ceiling=None,
            proposed_new_exact_leaf_ceiling=None)
    from math import prod
    actors=len(participants)
    goals=prod(row['current_positive_goals'] for row in budget['per_actor'].values())
    method='exact' if budget['physical_resolver_calls_upper_bound']<=256 else 'mixture256'
    kind='exact' if method=='exact' else 'mixture'
    for caches in available.values():
        for row in caches:
            if row['stochastic_participants']!=participants or row['frame_actor_count']!=frame_actors:
                raise ValueError('An existing cache differs from its current geometric domain')
    compatible=[r for r in available[kind] if kind=='exact' or r['particle_count']==256]
    reused=compatible[0] if compatible else None
    return dict(proposed_method=method,proposed_reused_cache=reused,
        current_goal_combinations=goals,
        all_exact_full_forward_ceiling=budget['physical_resolver_calls_upper_bound'],
        all_exact_feature_ceiling=actors*goals,
        all_exact_leaf_ceiling=budget['lateral_leaf_branches_upper_bound'],
        all_mixture_full_forward_ceiling=256,
        all_mixture_feature_ceiling=actors*min(goals,256),
        proposed_new_full_forward_ceiling=0 if reused else budget['physical_resolver_calls_upper_bound'] if method=='exact' else 256,
        proposed_new_feature_ceiling=0 if reused else actors*(goals if method=='exact' else min(goals,256)),
        proposed_new_exact_leaf_ceiling=0 if reused or method!='exact' else budget['lateral_leaf_branches_upper_bound'])
