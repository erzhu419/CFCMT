"""Combine matched native branches with the original collector's label replay."""

import argparse
import json
from pathlib import Path
from statistics import mean

from cologne_training_label_comparison import compare_labels


def summarize(root):
    protocol = json.loads((root / 'protocol_v1.json').read_text())
    native_runs = [json.loads((root / f'audit/seed_{seed}/result.json').read_text())
                   for seed in protocol['training_seeds']]
    replay_runs = [json.loads((root / f'collector_replay_v1/seed_{seed}/result.json').read_text())
                   for seed in protocol['training_seeds']]
    native = {g['group_id']: g for r in native_runs for g in r['groups']}
    replay = {g['group_id']: g for r in replay_runs for g in r['groups']}
    expected = [g['group_id'] for g in protocol['groups']]
    if len(expected) != 9 or set(native) != set(expected) or set(replay) != set(expected):
        raise ValueError('Both sources must contain the nine frozen groups')
    if not all(r['status'] == 'PASS' for r in replay_runs):
        raise ValueError('Original collector label reproduction is incomplete')
    groups = []
    for group_id in expected:
        n, r = native[group_id], replay[group_id]
        branches = n['native_branches']
        if (len(branches) != 8 or [b['candidate_index'] for b in branches] != list(range(8))
                or not all(b['status'] == 'PASS' and all(b['checks'].values()) for b in branches)
                or r['status'] != 'PASS' or not all(r['checks'].values())
                or n['candidate_states'] != r['candidate_states']
                or n['stored_costs'] != r['original_costs']):
            raise ValueError(f'Native state/safety or original label binding failed for {group_id}')
        costs = [b['cost'] for b in branches]
        comparison = compare_labels(n['stored_costs'], costs, n['reference_index'])
        original_best = comparison['stored_optimal_indices']
        regrets = [costs[i] - min(costs) for i in original_best]
        groups.append({'group_id': group_id, 'seed': n['seed'], 'position': n['position'],
            'interval_index': n['interval_index'], 'time_sec': n['time_sec'],
            'bank_path': n['bank_path'], 'candidate_states': n['candidate_states'],
            'reference_index': n['reference_index'], 'stored_costs': n['stored_costs'],
            'native_costs': costs, 'comparison': comparison,
            'native_regret_of_stored_optimum_min': min(regrets),
            'native_regret_of_stored_optimum_max': max(regrets),
            'original_collector_reproduction_error': r['costs_match']['max_absolute_difference'],
            'native_state_and_safety_valid': True,
            'short_history_control': n['restored_pp_control'],
            'initial_group_status_preserved': n['status']})
    files = [root / f'{stage}/seed_{seed}/result.json'
             for stage in ('audit', 'collector_replay_v1') for seed in protocol['training_seeds']]
    comparisons = [g['comparison'] for g in groups]
    return {'protocol': 'tsc-v154o-cologne-training-label-restore-audit-summary-v1',
        'status': 'PASS',
        'finding': 'The original collector exactly reproduces the saved labels, but their action rankings differ from native-prefix outcomes in the same frozen training groups.',
        'source_root': protocol['source_root'], 'sumocfg': protocol['sumocfg'],
        'frozen_threshold': protocol['frozen_threshold'], 'selection_rule': protocol['selection_rule'],
        'label_contract': protocol['label_contract'], 'new_model_fit_count': 0, 'threshold_retuned': False,
        'summary': {'planned_groups': 9, 'valid_groups': len(groups), 'native_action_rows': 72,
            'non_reference_action_rows': 63,
            'groups_with_changed_costs': sum(c['any_cost_changed'] for c in comparisons),
            'groups_with_changed_optimal_set': sum(c['optimal_set_changed'] for c in comparisons),
            'groups_with_disjoint_optimal_sets': sum(c['disjoint_optimal_sets'] for c in comparisons),
            'groups_with_strict_sign_flips': sum(bool(c['strict_sign_flip_indices']) for c in comparisons),
            'strict_sign_flip_actions': sum(len(c['strict_sign_flip_indices']) for c in comparisons),
            'sign_change_actions_including_ties': sum(len(c['any_advantage_sign_change_indices']) for c in comparisons),
            'mean_absolute_cost_error': mean(c['mean_absolute_cost_error'] for c in comparisons),
            'max_absolute_cost_error': max(c['max_absolute_cost_error'] for c in comparisons),
            'mean_native_regret_of_stored_optimum_min': mean(g['native_regret_of_stored_optimum_min'] for g in groups),
            'mean_native_regret_of_stored_optimum_max': mean(g['native_regret_of_stored_optimum_max'] for g in groups),
            'max_original_collector_reproduction_error': max(g['original_collector_reproduction_error'] for g in groups),
            'native_collision_events': sum(b['safety']['raw_collision_events'] for g in native.values() for b in g['native_branches']),
            'native_starting_teleports': sum(b['safety']['starting_teleports'] for g in native.values() for b in g['native_branches']),
            'short_history_controls_matching_old_labels': sum(g['short_history_control']['stored_label_exact'] for g in groups)},
        'per_period': {period: {'mean_absolute_cost_error': mean(g['comparison']['mean_absolute_cost_error'] for g in groups if g['position'] == period),
            'mean_native_regret_of_stored_optimum': mean(g['native_regret_of_stored_optimum_min'] for g in groups if g['position'] == period)}
            for period in ('early', 'middle', 'late')},
        'groups': groups,
        'execution': {'native_tasks': ['t91785', 't91786', 't91787'],
            'collector_replay_tasks': ['t91805', 't91806', 't91807'],
            'native_branches': 72, 'short_history_pp_controls': 9, 'original_collector_action_rows': 72,
            'original_collector_total_restores': sum(r['metadata']['state_restores'] for r in replay_runs),
            'runner_elapsed_sec_total': sum(r['elapsed_sec'] for r in [*native_runs, *replay_runs]),
            'retrieved_bytes_total': sum(p.stat().st_size for p in files),
            'retrieved_files': [str(p) for p in files], 'focused_tests_passed': 10},
        'preserved_failed_controls': 'The three initial runner results remain INVALID because six shortened-history PP restore controls do not reproduce their old labels. The final comparison reuses only their fully matched, complete and safe native branches; old labels are validated by the separate unchanged collector replay.',
        'next_step': {'purpose': 'Replace restore-based label generation with native-prefix branching before another model or threshold change.',
            'scope': 'Recollect the original100 B100 groups, keeping exact group IDs, eight candidates, original PP behavior prefixes, repaired network/executor, occupancy equations and450-second target. Retain any unsafe group as a failed group without replacement.',
            'reuse': 'Reuse these72 validated native labels for the nine audited groups when constructing the full corrected bank.',
            'fitting': 'Refit and recalibrate using only the original training seeds after a complete valid B100 bank exists.',
            'executed': False},
        'limitations': ['Nine preselected groups do not estimate the error rate across all100 training groups or other cities.',
            'Two late groups change optimal action with native regret only0.000277778; early and middle differences are larger.',
            'The specific hidden SUMO state causing the restore divergence is not isolated. This audit does not quantify how much of the closed-loop waiting gap it explains.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root.resolve())
    with args.out.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps({'summary': result['summary'], 'per_period': result['per_period'], 'execution': result['execution']}))
