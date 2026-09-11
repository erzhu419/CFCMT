"""Natively label the fixed95 additions; collisions remain diagnostic evidence."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

if __package__:
    from .cologne_action_branch_checks import physical_snapshot, compare_physical
    from .cologne_native_bank import B195_PROTOCOL, build_native_group, assemble_native_bank
    from .run_cologne_training_label_restore_audit import native_prefix, feature_match, TLS
else:
    from cologne_action_branch_checks import physical_snapshot, compare_physical
    from cologne_native_bank import B195_PROTOCOL, build_native_group, assemble_native_bank
    from run_cologne_training_label_restore_audit import native_prefix, feature_match, TLS


PROTOCOL = 'tsc-v155k-cologne-expanded-data-oof-v1'


def shard_groups(protocol, seed, shard):
    additional, original = protocol['additional_group_ids'], protocol['original_group_ids']
    specs = protocol['groups']
    if (len(additional) != 95 or len(set(additional)) != 95
            or len(original) != 100 or set(additional) & set(original)
            or [row['group_id'] for row in specs] != additional or shard not in (0, 1)):
        raise ValueError('Use the fixed disjoint original100 and additional95 group rosters')
    for row in specs:
        if (row['seed'] != int(row['group_id'].split(':')[1][4:])
                or row['interval_index'] != int(row['group_id'].split(':')[2])
                or row['time_sec'] != 25200 + 10*row['interval_index']):
            raise ValueError('Group seed, interval and original capture time must agree')
    return [row for row in specs if row['seed'] == seed][shard::2]


def audit_native_outcome(api, v3, infos, context, executors, states, phase, lanes, traces, key):
    """The original45-interval outcome, with collision_mode=audit at each advance."""
    ledger, costs = v3._new_safety_ledger(), []
    candidate = v3._candidate_for_state(infos[TLS], phase)
    one_step = None
    try:
        for interval in range(45):
            rollout = states if interval == 0 else {
                tls: v3._read_local_state(api, tls, info, context) for tls, info in sorted(infos.items())}
            actions = {tls: v3._rule_candidate(rollout[tls], executors[tls], 'phase_pressure').state
                       for tls in sorted(infos)}
            if interval == 0:
                actions[TLS] = candidate.state
            costs.extend(float(value)/8. for value in v3._advance_with_actions(
                sumo_api=api, executors=executors, actions=actions, control_interval_sec=10,
                collision_mode='audit', safety_ledger=ledger, cost_lanes=lanes,
                system_cost_lane_count=None))
            if interval == 0:
                one_step = v3._candidate_aggregates(v3._read_local_state(api, TLS, infos[TLS], context), candidate)
        terminal = v3._candidate_aggregates(v3._read_local_state(api, TLS, infos[TLS], context), candidate)
        aggregates = ('total_q', 'green_q', 'red_q', 'green_down_occ', 'mean_speed')
        result = {'status': 'COMPLETE', 'cost': float(np.mean(costs)),
                  'one_step': {name: one_step[name] for name in aggregates},
                  'terminal': {name: terminal[name] for name in aggregates}}
    except Exception as error:
        result = {'status': 'INVALID', 'error': f'{type(error).__name__}: {error}'}
    traces[key] = costs
    result.update({'sample_count': len(costs), 'first_interval_costs': costs[:10],
                   'end_time_sec': float(api.simulation.getTime()),
                   'safety': v3._finalize_safety_ledger(ledger)})
    return result


def collect_group(spec, original, protocol, api, v3, traces):
    indices = np.flatnonzero(np.asarray(original.metadata['action_group_ids']) == spec['group_id'])
    phases = np.asarray(original.metadata['candidate_states'])[indices].tolist()
    if (len(indices) != 8 or len(set(phases)) != 8
            or tuple(original.feature_names) != tuple(v3.FEATURE_NAMES_V3)):
        raise ValueError('Keep the original eight candidate rows and frozen V3 features')
    result = {**spec, 'status': 'INVALID', 'candidate_states': phases, 'native_branches': []}
    first_start, first_pending = None, None
    for index, phase in enumerate(phases):
        started = time.monotonic()
        branch = {'candidate_index': index, 'phase': phase, 'status': 'INVALID',
                  'checks': {}, 'prefix_replay_sec': 0., 'window_simulation_sec': 0.}
        result['native_branches'].append(branch)
        try:
            infos, context, executors, states, prefix_safety = native_prefix(
                api, v3, Path(protocol['sumocfg']), spec['seed'], spec['interval_index'])
            branch.update({'prefix_safety': prefix_safety, 'prefix_wall_sec': time.monotonic()-started,
                           'prefix_replay_sec': float(api.simulation.getTime())-25200})
            lanes = tuple(sorted(v3._controlled_lane_set(infos)))
            physical = physical_snapshot(api, executors)
            pending = sorted(api.simulation.getPendingVehicles())
            if first_start is None:
                first_start, first_pending = physical, pending
                result['native_capture_server_only'] = {'physical': physical, 'pending_ids': pending}
            features = feature_match(np.vstack([v3.candidate_features_v3(states[TLS],
                v3._candidate_for_state(infos[TLS], candidate), executors[TLS], control_interval_sec=10)
                for candidate in phases]), original.features[indices], original.feature_names)
            contexts = feature_match(np.tile(context, (8, 1)), original.context[indices], original.context_names)
            physical_match = compare_physical(first_start, physical, tolerance=0)
            checks = {'time_exact': physical['time_sec'] == spec['time_sec'],
                'stored_times_exact': bool(np.all(np.asarray(original.metadata['row_times'])[indices] == spec['time_sec'])),
                'single_tls_eight_lanes': list(infos) == [TLS] and len(lanes) == 8,
                'candidate_phases_exact': list(executors[TLS].feasible_states_now()) == phases,
                'features_exact': features['passed'], 'contexts_exact': contexts['passed'],
                'native_start_exact': physical_match['passed'], 'pending_start_exact': first_pending == pending,
                'prefix_no_teleport': prefix_safety['no_teleport_passed']}
            branch.update({'checks': checks, 'feature_match': features, 'context_match': contexts,
                           'physical_match': physical_match})
            if not all(checks.values()):
                continue
            outcome_started = time.monotonic()
            branch.update(audit_native_outcome(api, v3, infos, context, executors, states, phase, lanes,
                                               traces, f'{spec["group_id"]}:native:{index}'))
            branch['window_wall_sec'] = time.monotonic()-outcome_started
            branch['window_simulation_sec'] = branch['end_time_sec']-spec['time_sec']
            checks['full_horizon'] = branch['sample_count'] == 450 and branch['end_time_sec'] == spec['time_sec']+450
            checks['window_no_teleport'] = branch['safety']['no_teleport_passed']
            if not all(checks.values()):
                branch['status'] = 'INVALID'
        except Exception as error:
            branch['error'] = f'{type(error).__name__}: {error}'
            if hasattr(error, 'safety_audit'):
                branch['prefix_safety'] = error.safety_audit
            try:
                branch['prefix_replay_sec'] = min(float(api.simulation.getTime()), spec['time_sec'])-25200
            except Exception:
                pass
        finally:
            branch['elapsed_sec'] = time.monotonic()-started
            try:
                api.close()
            except Exception as error:
                branch.update({'status': 'INVALID', 'close_error': f'{type(error).__name__}: {error}'})
    if all(branch['status'] == 'COMPLETE' and all(branch['checks'].values()) for branch in result['native_branches']):
        result['status'] = 'COMPLETE'
    return result


def compact_group(group):
    branches = group['native_branches']
    return {'group_id': group['group_id'], 'seed': group['seed'], 'time_sec': group['time_sec'],
        'status': group['status'], 'native_branches_attempted': len(branches),
        'completed_branches': sum(branch['status'] == 'COMPLETE' for branch in branches),
        'native_costs': [branch.get('cost') for branch in branches],
        'prefix_replay_sec': sum(branch.get('prefix_replay_sec', 0.) for branch in branches),
        'window_simulation_sec': sum(branch.get('window_simulation_sec', 0.) for branch in branches),
        'prefix_collision_events': sum(branch.get('prefix_safety', {}).get('raw_collision_events', 0) for branch in branches),
        'window_collision_events': sum(branch.get('safety', {}).get('raw_collision_events', 0) for branch in branches),
        'failures': [{key: branch.get(key) for key in ('candidate_index', 'phase', 'checks', 'error', 'close_error')}
                     for branch in branches if branch['status'] != 'COMPLETE'], 'error': group.get('error')}


def run(protocol, seed, shard, output):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.eval import traffic_signal_cologne_collision_replay as runtime
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset, save_mechanism_dataset
    started = time.monotonic()
    if protocol['protocol'] != PROTOCOL or os.environ.get('CFCMT_SOURCE_ROOT') != protocol['source_root']:
        raise ValueError('Use the frozen K protocol and unchanged J simulation source')
    specs = shard_groups(protocol, seed, shard)
    if output.exists():
        raise FileExistsError(output)
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(Path(protocol['sumocfg'])))
    if geometry != json.loads(Path(protocol['geometry_reference']).read_text())['bilateral_static_geometry']:
        raise ValueError('The fixed repaired network changed')
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = protocol['conversion_root']
    if '1.22.0' not in str(runtime.libsumo_version()):
        raise ValueError('Use SUMO1.22.0')
    api = runtime.load_libsumo()
    output.mkdir(parents=True)
    banks, parts, summaries, failed = {}, [], [], []
    for spec in specs:
        group_output = output / 'groups' / f'interval_{spec["interval_index"]}'
        group_output.mkdir(parents=True)
        traces = {}
        group = {**spec, 'status': 'INVALID', 'native_branches': []}
        try:
            if spec['bank_path'] not in banks:
                banks[spec['bank_path']] = load_mechanism_dataset(Path(spec['bank_path']))
            original = banks[spec['bank_path']]
            meta = original.metadata
            if (meta['sumocfg'] != protocol['sumocfg'] or meta['seed'] != seed
                    or meta['behavior_policy'] != 'phase_pressure' or meta['counterfactual_horizon_sec'] != 450
                    or meta['counterfactual_cost_mode'] != 'halted_queue'):
                raise ValueError('Original prefix policy, labels or network binding changed')
            group = collect_group(spec, original, protocol, api, v3, traces)
            group['source_result_path'] = str(group_output / 'native_outcomes_server_only.json')
            if group['status'] == 'COMPLETE':
                native = build_native_group(original, group, traces, native_protocol=B195_PROTOCOL)
                save_mechanism_dataset(group_output / 'bank.npz', native)
                parts.append(native)
        except Exception as error:
            group.update({'status': 'INVALID', 'error': f'{type(error).__name__}: {error}'})
        if group['status'] != 'COMPLETE':
            failed.append(spec['group_id'])
        summary = compact_group(group)
        summaries.append(summary)
        for name, content in (('native_outcomes_server_only.json', group), ('cost_traces_server_only.json', traces)):
            (group_output / name).write_text(json.dumps(content, separators=(',', ':'), allow_nan=False)+'\n')
        print(json.dumps(summary), flush=True)
    expected = [spec['group_id'] for spec in specs]
    bank_path = output / 'bank.npz'
    data = assemble_native_bank(parts, expected, failed_groups=failed, native_protocol=B195_PROTOCOL) if parts else None
    if data is not None:
        save_mechanism_dataset(bank_path, data)
    result = {'protocol': PROTOCOL, 'status': 'INVALID' if failed else 'COMPLETE',
        'source_root': protocol['source_root'], 'sumocfg': protocol['sumocfg'], 'seed': seed, 'shard': shard,
        'requested_groups': expected, 'retained_groups': [row['group_id'] for row in summaries if row['status'] == 'COMPLETE'],
        'invalid_groups': failed, 'rows': data.size if data is not None else 0,
        'bank': str(bank_path) if data is not None else None, 'bank_protocol': B195_PROTOCOL,
        'groups': summaries, 'state_restores': 0, 'collision_admission_gate': False,
        'new_branches_attempted': sum(row['native_branches_attempted'] for row in summaries),
        **{key: sum(row[key] for row in summaries) for key in ('completed_branches', 'prefix_replay_sec',
            'window_simulation_sec', 'prefix_collision_events', 'window_collision_events')},
        'elapsed_sec': time.monotonic()-started}
    (output / 'result.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'groups'}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(2027, 3037, 4047), required=True)
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if run(json.loads(args.protocol.read_text()), args.seed, args.shard, args.output)['status'] != 'COMPLETE':
        raise SystemExit(2)
