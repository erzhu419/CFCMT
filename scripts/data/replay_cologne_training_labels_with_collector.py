"""Replay the nine frozen training groups through the original two-pass collector."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

PROTOCOL = 'tsc-v154o-cologne-training-label-restore-audit-v1'
TARGET = 'prefix_mean_cost_450s'


def difference(actual, expected, tolerance=0):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return {'passed': False, 'actual_shape': list(actual.shape), 'expected_shape': list(expected.shape)}
    delta = actual - expected
    return {'passed': bool(np.all(np.abs(delta) <= tolerance)),
        'max_absolute_difference': float(np.max(np.abs(delta))),
        'mismatch_count': int(np.sum(np.abs(delta) > tolerance))}


def compare_group(spec, actual, load_dataset):
    original = load_dataset(Path(spec['bank_path']))
    old_rows = np.flatnonzero(np.asarray(original.metadata['action_group_ids']) == spec['group_id'])
    rows = np.flatnonzero(np.asarray(actual.metadata['action_group_ids']) == spec['group_id'])
    old_phases = np.asarray(original.metadata['candidate_states'])[old_rows].tolist()
    phases = np.asarray(actual.metadata['candidate_states'])[rows].tolist()
    result = {**spec, 'status': 'INVALID', 'original_rows': len(old_rows), 'replayed_rows': len(rows),
        'candidate_states': phases, 'original_candidate_states': old_phases,
        'original_costs': original.targets[TARGET][old_rows].tolist(), 'checks': {}}
    if spec['group_id'] in actual.metadata['counterfactual_safety_audit']['censored_group_ids']:
        result['status'] = 'FAIL_UNSAFE'
        return result
    result['checks']['eight_candidate_phases_exact'] = len(rows) == len(old_rows) == 8 and phases == old_phases
    if not result['checks']['eight_candidate_phases_exact']:
        return result
    result['checks']['feature_names_exact'] = actual.feature_names == original.feature_names
    result['checks']['context_names_exact'] = actual.context_names == original.context_names
    for key, observed, expected, tolerance in (
            ('features', actual.features[rows], original.features[old_rows], 0),
            ('contexts', actual.context[rows], original.context[old_rows], 0),
            ('row_times', np.asarray(actual.metadata['row_times'])[rows],
             np.asarray(original.metadata['row_times'])[old_rows], 0),
            ('costs', actual.targets[TARGET][rows], original.targets[TARGET][old_rows], 1e-12),
            ('interval_target', actual.targets[TARGET][rows], actual.targets['interval_cost'][rows], 1e-12)):
        result[key + '_match'] = difference(observed, expected, tolerance)
        result['checks'][key + '_match'] = result[key + '_match']['passed']
    result['replayed_costs'] = actual.targets[TARGET][rows].tolist()
    result['cost_deltas'] = (actual.targets[TARGET][rows] - original.targets[TARGET][old_rows]).tolist()
    result['status'] = 'PASS' if all(result['checks'].values()) else 'INVALID'
    return result


def run(protocol_path, seed, output):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    started = time.monotonic()
    protocol = json.loads(protocol_path.read_text())
    if protocol['protocol'] != PROTOCOL or os.environ['CFCMT_SOURCE_ROOT'] != protocol['source_root']:
        raise ValueError('Use the original V154O protocol and frozen source')
    if output.exists():
        raise FileExistsError(output)
    groups = [g for g in protocol['groups'] if g['seed'] == seed]
    if len(groups) != 3:
        raise ValueError('Use the three frozen groups for this training seed')
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = protocol['conversion_root']
    api = load_libsumo()
    if '1.22.0' not in str(libsumo_version()):
        raise ValueError('Use SUMO 1.22.0')
    result = {'protocol': 'tsc-v154o-cologne-collector-replay-v1', 'status': 'INVALID',
        'seed': seed, 'source_protocol': str(protocol_path), 'source_root': protocol['source_root'],
        'sumocfg': protocol['sumocfg'], 'requested_groups': [g['group_id'] for g in groups],
        'groups': [], 'planned_groups': 3, 'planned_action_rows': 24}
    try:
        data = v3.collect_counterfactual_transitions_v3(sumo_api=api,
            sumocfg=Path(protocol['sumocfg']), scenario='cologne1', seed=seed,
            duration_sec=3600, control_interval_sec=10, warmup_sec=60, max_focal_tls=4,
            counterfactual_horizon_intervals=45, rollout_prefix_horizons_sec=[450],
            behavior_policy='phase_pressure', counterfactual_cost_mode='halted_queue',
            collection_shard_index=0, collection_shard_count=1,
            selected_action_group_ids=result['requested_groups'])
        result['groups'] = [compare_group(g, data, load_mechanism_dataset) for g in groups]
        result['metadata'] = data.metadata
        result['replayed_rows'] = data.size
        result['matched_groups'] = sum(g['status'] == 'PASS' for g in result['groups'])
        result['unsafe_groups'] = sum(g['status'] == 'FAIL_UNSAFE' for g in result['groups'])
        result['invalid_groups'] = sum(g['status'] == 'INVALID' for g in result['groups'])
        behavior = data.metadata['behavior_safety_audit']
        result['behavior_safe'] = behavior['raw_collision_events'] == 0 and behavior['no_teleport_passed']
        result['status'] = ('FAIL_UNSAFE' if result['unsafe_groups'] or not result['behavior_safe']
                            else 'PASS' if result['matched_groups'] == 3 else 'INVALID')
    except Exception as error:
        result.update(status='FAIL', error=f'{type(error).__name__}: {error}')
    result['elapsed_sec'] = time.monotonic() - started
    output.mkdir(parents=True)
    (output / 'result.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('protocol', 'status', 'seed', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(2027, 3037, 4047), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if run(args.protocol, args.seed, args.output)['status'] != 'PASS':
        raise SystemExit(2)
