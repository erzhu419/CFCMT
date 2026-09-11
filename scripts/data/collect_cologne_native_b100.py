"""Complete the original B100 roster using fresh native prefixes for each action."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from cologne_action_branch_checks import physical_snapshot, compare_physical
from cologne_native_bank import build_native_group, assemble_native_bank
from run_cologne_training_label_restore_audit import native_prefix, outcome, feature_match, TLS


PROTOCOL = 'tsc-v154p-cologne-native-b100-v1'


def collect_group(spec, original, protocol, api, v3, traces):
    rows = np.flatnonzero(np.asarray(original.metadata['action_group_ids']) == spec['group_id'])
    phases = np.asarray(original.metadata['candidate_states'])[rows].tolist()
    if len(rows) != 8 or len(set(phases)) != 8:
        raise ValueError('Every original B100 group must have eight candidate rows')
    if tuple(original.feature_names) != tuple(v3.FEATURE_NAMES_V3):
        raise ValueError('Original feature columns differ from the frozen core')
    result = {**spec, 'status': 'INVALID', 'candidate_states': phases, 'native_branches': []}
    first_start = None
    for index, phase in enumerate(phases):
        try:
            infos, context, executors, states, prefix_safety = native_prefix(
                api, v3, Path(protocol['sumocfg']), spec['seed'], spec['interval_index'])
            lanes = tuple(sorted(v3._controlled_lane_set(infos)))
            physical = physical_snapshot(api, executors)
            if first_start is None:
                first_start = physical
            actual = np.vstack([v3.candidate_features_v3(states[TLS],
                v3._candidate_for_state(infos[TLS], p), executors[TLS], control_interval_sec=10)
                for p in phases])
            features = feature_match(actual, original.features[rows], original.feature_names)
            contexts = feature_match(np.tile(context, (8, 1)), original.context[rows], original.context_names)
            start = compare_physical(first_start, physical, tolerance=0)
            checks = {'time_exact': physical['time_sec'] == spec['time_sec'],
                'stored_times_exact': bool(np.all(np.asarray(original.metadata['row_times'])[rows] == spec['time_sec'])),
                'single_tls_eight_lanes': list(infos) == [TLS] and len(lanes) == 8,
                'candidate_phases_exact': list(executors[TLS].feasible_states_now()) == phases,
                'features_exact': features['passed'], 'contexts_exact': contexts['passed'],
                'native_start_exact': start['passed'],
                'pending_start_exact': first_start['pending_raw'] == physical['pending_raw'],
                'prefix_safe': prefix_safety['raw_collision_events'] == 0 and prefix_safety['no_teleport_passed']}
            branch = {'candidate_index': index, 'phase': phase, 'checks': checks,
                      'prefix_safety': prefix_safety, 'feature_match': features,
                      'context_match': contexts, 'physical_match': start}
            result['native_branches'].append(branch)
            if not all(checks.values()):
                branch['status'] = 'INVALID' if checks['prefix_safe'] else 'FAIL_UNSAFE'
                break
            branch.update(outcome(api, v3, infos, context, executors, states, phase, lanes,
                                  traces, f'{spec["group_id"]}:native:{index}'))
            checks['full_horizon'] = branch.get('sample_count') == 450 and branch['end_time_sec'] == spec['time_sec'] + 450
        finally:
            api.close()
    branches = result['native_branches']
    if any(b['status'] == 'FAIL_UNSAFE' for b in branches):
        result['status'] = 'FAIL_UNSAFE'
    elif len(branches) == 8 and all(b['status'] == 'PASS' and all(b['checks'].values()) for b in branches):
        result['status'] = 'PASS'
    return result


def compact_group(group, reused):
    branches = group['native_branches']
    failures = [{'candidate_index': b['candidate_index'], 'phase': b['phase'],
                 'status': b['status'], 'checks': b['checks'],
                 'safety': b.get('safety'), 'prefix_safety': b.get('prefix_safety'),
                 'error': b.get('error'), 'end_time_sec': b.get('end_time_sec'),
                 'feature_match': b.get('feature_match'), 'physical_match': b.get('physical_match')}
                for b in branches if b['status'] != 'PASS' or not all(b['checks'].values())]
    # Reused V154O groups can have an INVALID outer status from their discarded restore control.
    native_valid = len(branches) == 8 and not failures
    status = 'PASS' if native_valid else 'FAIL_UNSAFE' if any(b['status'] == 'FAIL_UNSAFE' for b in branches) else 'INVALID'
    return {'group_id': group['group_id'], 'seed': group['seed'], 'time_sec': group['time_sec'],
            'reused': reused, 'status': status,
            'native_branches_attempted': sum('end_time_sec' in b for b in branches),
            'safe_branches': sum(b['status'] == 'PASS' for b in branches), 'failures': failures,
            'native_costs': [b.get('cost') for b in branches]}


def run(protocol, seed, output):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.eval import traffic_signal_cologne_collision_replay as runtime
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset, save_mechanism_dataset
    started = time.monotonic()
    if protocol['protocol'] != PROTOCOL or os.environ['CFCMT_SOURCE_ROOT'] != protocol['source_root']:
        raise ValueError('Use the frozen V154P protocol and V154J simulation core')
    if output.exists():
        raise FileExistsError(output)
    if json.loads(Path(protocol['frozen_threshold_path']).read_text()) != protocol['frozen_threshold']:
        raise ValueError('The original model/threshold binding changed')
    fit = json.loads(Path(protocol['fit_result']).read_text())
    if fit['selected_groups'] != protocol['training_group_ids'] or len(fit['selected_groups']) != 100:
        raise ValueError('The original B100 roster changed')
    groups = [g for g in protocol['groups'] if g['seed'] == seed]
    expected = [g for g in fit['selected_groups'] if g.startswith(f'cologne1:seed{seed}:')]
    if [g['group_id'] for g in groups] != expected:
        raise ValueError('The seed collection roster changed')
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(Path(protocol['sumocfg'])))
    if geometry != json.loads(Path(protocol['geometry_reference']).read_text())['bilateral_static_geometry']:
        raise ValueError('The fixed repaired geometry changed')
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = protocol['conversion_root']
    if '1.22.0' not in str(runtime.libsumo_version()):
        raise ValueError('Use SUMO 1.22.0')
    reused_run_path = Path(protocol['reuse_root']) / f'audit/seed_{seed}/result.json'
    reused_run = json.loads(reused_run_path.read_text())
    reused = {g['group_id']: g for g in reused_run['groups']}
    old_replay = json.loads((Path(protocol['reuse_root']) / f'collector_replay_v1/seed_{seed}/result.json').read_text())
    if old_replay['status'] != 'PASS' or reused_run['seed'] != seed:
        raise ValueError('The previously validated native inputs are not available')
    reuse_traces = json.loads(Path(reused_run['cost_traces_server_only']).read_text())
    api = runtime.load_libsumo()
    banks, parts, records, summaries, traces, failed = {}, [], [], [], {}, []
    output.mkdir(parents=True)
    for spec in groups:
        path = spec['bank_path']
        if path not in banks:
            banks[path] = load_mechanism_dataset(Path(path))
        original = banks[path]
        if (original.metadata['sumocfg'] != protocol['sumocfg'] or original.metadata['seed'] != seed
                or original.metadata['behavior_policy'] != 'phase_pressure'):
            raise ValueError('The original collection source differs from the fixed protocol')
        if spec['reused']:
            group = {**reused[spec['group_id']], 'reused': True,
                     'source_result_path': str(reused_run_path)}
            if group['bank_path'] != path or group['time_sec'] != spec['time_sec']:
                raise ValueError('The reused group belongs to another original state')
            for i in range(8):
                key = f'{spec["group_id"]}:native:{i}'
                traces[key] = reuse_traces[key]
        else:
            group = collect_group(spec, original, protocol, api, v3, traces)
            group['source_result_path'] = str(output / 'native_outcomes_server_only.json')
        summary = compact_group(group, spec['reused'])
        if summary['status'] == 'PASS':
            parts.append(build_native_group(original, group, traces))
        else:
            failed.append(spec['group_id'])
        records.append(group)
        summaries.append(summary)
        print(json.dumps({'group': spec['group_id'], 'reused': spec['reused'], 'status': summary['status'],
                          'completed_groups': len(summaries), 'planned_groups': len(groups)}), flush=True)
    data = assemble_native_bank(parts, expected, failed_groups=failed)
    bank_path = output / 'bank.npz'
    save_mechanism_dataset(bank_path, data)
    (output / 'native_outcomes_server_only.json').write_text(json.dumps(records, separators=(',', ':'), allow_nan=False) + '\n')
    (output / 'cost_traces_server_only.json').write_text(json.dumps(traces, separators=(',', ':'), allow_nan=False) + '\n')
    unsafe = [g['group_id'] for g in summaries if g['status'] == 'FAIL_UNSAFE']
    invalid = [g['group_id'] for g in summaries if g['status'] == 'INVALID']
    result = {'protocol': PROTOCOL, 'status': 'FAIL_UNSAFE' if unsafe else 'INVALID' if invalid else 'PASS',
        'seed': seed, 'source_root': protocol['source_root'], 'sumocfg': protocol['sumocfg'],
        'bank': str(bank_path), 'requested_groups': expected,
        'retained_groups': [g['group_id'] for g in summaries if g['status'] == 'PASS'],
        'unsafe_groups': unsafe, 'invalid_groups': invalid, 'rows': data.size, 'groups': summaries,
        'reused_group_count': sum(g['reused'] for g in summaries),
        'new_group_count': sum(not g['reused'] for g in summaries),
        'new_branches_attempted': sum(g['native_branches_attempted'] for g in summaries if not g['reused']),
        'unsafe_native_branches': sum(b['status'] == 'FAIL_UNSAFE' for g in records for b in g['native_branches']),
        'bank_protocol': data.metadata['native_bank_protocol'], 'elapsed_sec': time.monotonic() - started}
    (output / 'result.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('protocol', 'status', 'seed', 'rows', 'new_branches_attempted', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(2027, 3037, 4047), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if run(json.loads(args.protocol.read_text()), args.seed, args.output)['status'] != 'PASS':
        raise SystemExit(2)
