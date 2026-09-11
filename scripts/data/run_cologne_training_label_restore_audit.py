"""Audit nine frozen B100 groups using native PP prefixes and saved labels."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from cologne_action_branch_checks import physical_snapshot, compare_physical
from cologne_training_label_comparison import compare_labels


PROTOCOL = 'tsc-v154o-cologne-training-label-restore-audit-v1'
TLS = 'cluster_357187_359543'
TARGET = 'prefix_mean_cost_450s'


def feature_match(actual, expected, names):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return {'passed': False, 'actual_shape': list(a.shape), 'expected_shape': list(b.shape)}
    changed = np.argwhere(a != b)
    return {'passed': bool(not len(changed)), 'mismatch_count': len(changed),
        'max_absolute_difference': float(np.max(np.abs(a - b))),
        'mismatches': [{'row': int(i), 'feature': names[j], 'actual': float(a[i, j]),
                        'expected': float(b[i, j])} for i, j in changed[:12]]}


def original_group(spec, v3):
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    data = load_mechanism_dataset(Path(spec['bank_path']))
    groups = np.asarray(data.metadata['action_group_ids'], dtype=str)
    rows = np.flatnonzero(groups == spec['group_id'])
    phases = np.asarray(data.metadata['candidate_states'], dtype=str)[rows].tolist()
    if len(rows) != 8 or len(set(phases)) != 8 or tuple(data.feature_names) != tuple(v3.FEATURE_NAMES_V3):
        raise ValueError('Expected eight original candidate rows with the frozen feature binding')
    if not np.all(np.asarray(data.metadata['row_times'])[rows] == spec['time_sec']):
        raise ValueError('Stored capture times differ from the frozen group')
    if not np.array_equal(data.targets[TARGET][rows], data.targets['interval_cost'][rows]):
        raise ValueError('The original fitted target is not the450-second label')
    return {'phases': phases, 'features': data.features[rows], 'context': data.context[rows],
            'feature_names': list(data.feature_names), 'context_names': list(data.context_names),
            'stored_costs': data.targets[TARGET][rows].tolist(), 'metadata': data.metadata}


def native_prefix(api, v3, sumocfg, seed, interval):
    """The unchanged collector's first pass, stopping before the target request."""
    v3._start_sumo(api, sumocfg, seed)
    infos = v3._tls_phase_infos(api)
    graph = v3.build_signal_routing_graph(api, infos)
    context = v3._scenario_context_v3(sumocfg=sumocfg, infos=infos, control_interval_sec=10)
    executors = v3.build_safe_phase_executors(api, infos)
    ledger = v3._new_safety_ledger()
    for _ in range(interval):
        states = v3._read_graph_states_v3(api, infos, context, graph)
        actions = {tls: v3._rule_candidate(states[tls], executors[tls], 'phase_pressure').state
                   for tls in sorted(infos)}
        v3._advance_with_actions(sumo_api=api, executors=executors, actions=actions,
                                control_interval_sec=10, collision_mode='audit', safety_ledger=ledger)
    states = v3._read_graph_states_v3(api, infos, context, graph)
    return infos, context, executors, states, v3._finalize_safety_ledger(ledger)


def outcome(api, v3, infos, context, executors, states, phase, lanes, traces, key):
    ledger = v3._new_safety_ledger()
    try:
        costs, one_step, terminal = v3._counterfactual_branch_outcome_v3(
            sumo_api=api, infos=infos, states=states, executors=executors, focal_id=TLS,
            candidate=v3._candidate_for_state(infos[TLS], phase), context=context,
            control_interval_sec=10, counterfactual_horizon_intervals=45,
            controlled_lane_count=8, controlled_lanes=lanes,
            counterfactual_cost_mode='halted_queue', safety_ledger=ledger)
    except v3.UnsafeRolloutError as error:
        return {'status': 'FAIL_UNSAFE', 'error': str(error), 'safety': error.safety_audit,
                'end_time_sec': float(api.simulation.getTime())}
    traces[key] = costs
    return {'status': 'PASS', 'cost': float(np.mean(costs)), 'sample_count': len(costs),
            'first_interval_costs': costs[:10], 'end_time_sec': float(api.simulation.getTime()),
            'one_step': {k: one_step[k] for k in ('total_q', 'green_q', 'red_q', 'green_down_occ', 'mean_speed')},
            'terminal': {k: terminal[k] for k in ('total_q', 'green_q', 'red_q', 'green_down_occ', 'mean_speed')},
            'safety': v3._finalize_safety_ledger(ledger)}


def audit_group(spec, bank, protocol, api, v3, scratch, traces):
    sumocfg, seed = Path(protocol['sumocfg']), spec['seed']
    phases = bank['phases']
    result = {**spec, 'status': 'INVALID', 'candidate_states': phases,
              'stored_costs': bank['stored_costs'], 'native_branches': [], 'checks': {}}
    starts = []
    for index, phase in enumerate(phases):
        try:
            infos, context, executors, states, prefix_safety = native_prefix(
                api, v3, sumocfg, seed, spec['interval_index'])
            lanes = tuple(sorted(v3._controlled_lane_set(infos)))
            physical = physical_snapshot(api, executors)
            actual_phases = list(executors[TLS].feasible_states_now())
            actual_features = np.vstack([v3.candidate_features_v3(states[TLS],
                v3._candidate_for_state(infos[TLS], p), executors[TLS], control_interval_sec=10)
                for p in phases])
            features = feature_match(actual_features, bank['features'], bank['feature_names'])
            contexts = feature_match(np.tile(context, (8, 1)), bank['context'], bank['context_names'])
            start_match = compare_physical(starts[0] if starts else physical, physical, tolerance=0)
            pending_match = not starts or starts[0]['pending_raw'] == physical['pending_raw']
            starts.append(physical)
            ref_phase = v3._rule_candidate(states[TLS], executors[TLS], 'phase_pressure').state
            checks = {'time_exact': physical['time_sec'] == spec['time_sec'],
                'single_tls_eight_lanes': list(infos) == [TLS] and len(lanes) == 8,
                'candidate_phases_exact': actual_phases == phases,
                'features_exact': features['passed'], 'contexts_exact': contexts['passed'],
                'native_start_exact': start_match['passed'], 'pending_start_exact': bool(pending_match),
                'prefix_safe': prefix_safety['raw_collision_events'] == 0 and prefix_safety['no_teleport_passed']}
            branch = {'candidate_index': index, 'phase': phase, 'checks': checks,
                      'feature_match': features, 'context_match': contexts, 'physical_match': start_match,
                      'prefix_safety': prefix_safety}
            result['native_branches'].append(branch)
            if not all(checks.values()):
                branch['status'] = 'INVALID'
                break
            reference_index = phases.index(ref_phase)
            result['reference_index'] = reference_index
            if index == reference_index:
                snapshots = {tls: ex.snapshot() for tls, ex in executors.items()}
                state_path = scratch / f'seed{seed}_interval{spec["interval_index"]}_pp.xml'
                api.simulation.saveState(str(state_path))
            branch.update(outcome(api, v3, infos, context, executors, states, phase, lanes,
                                  traces, f'{spec["group_id"]}:native:{index}'))
            branch['checks']['full_horizon'] = (branch.get('sample_count') == 450
                and branch['end_time_sec'] == spec['time_sec'] + 450)
            if index == reference_index:
                v3._reload_sumo_state(api, sumocfg, seed, state_path, begin_time=spec['time_sec'])
                v3._restore_executors(executors, snapshots)
                restored = physical_snapshot(api, executors)
                restore_match = compare_physical(physical, restored)
                control = {'phase': phase, 'restoration': restore_match,
                    'original_pending': physical['pending_raw'], 'restored_pending': restored['pending_raw']}
                control.update(outcome(api, v3, infos, context, executors, states, phase, lanes,
                                      traces, f'{spec["group_id"]}:restored_pp'))
                control['stored_label_exact'] = (control.get('status') == 'PASS'
                    and abs(control['cost'] - bank['stored_costs'][reference_index]) <= 1e-12)
                control['full_horizon'] = control.get('sample_count') == 450 and control['end_time_sec'] == spec['time_sec'] + 450
                if branch['status'] == control['status'] == 'PASS':
                    control['native_minus_restored_cost'] = branch['cost'] - control['cost']
                    control['first_interval_max_difference'] = float(np.max(np.abs(
                        np.asarray(branch['first_interval_costs']) - control['first_interval_costs'])))
                result['restored_pp_control'] = control
        finally:
            api.close()
    branches, control = result['native_branches'], result.get('restored_pp_control', {})
    result['checks'] = {'all_eight_native_branches': len(branches) == 8,
        'all_native_valid': all(all(row['checks'].values()) for row in branches),
        'all_native_safe': all(row.get('status') == 'PASS' for row in branches),
        'restored_pp_safe': control.get('status') == 'PASS',
        'restored_pp_start_matches': control.get('restoration', {}).get('passed') is True,
        'restored_pp_full_horizon': control.get('full_horizon') is True,
        'restored_pp_matches_stored_label': control.get('stored_label_exact') is True}
    if all(result['checks'].values()):
        result['status'] = 'PASS'
        result['native_costs'] = [row['cost'] for row in branches]
        result['comparison'] = compare_labels(bank['stored_costs'], result['native_costs'], result['reference_index'])
    elif any(row.get('status') == 'FAIL_UNSAFE' for row in [*branches, control]):
        result['status'] = 'FAIL_UNSAFE'
    print(json.dumps({'group': spec['group_id'], 'status': result['status'], 'checks': result['checks']}), flush=True)
    return result


def run(protocol, seed, output):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.eval import traffic_signal_cologne_collision_replay as runtime
    started = time.monotonic()
    if protocol['protocol'] != PROTOCOL or os.environ['CFCMT_SOURCE_ROOT'] != protocol['source_root']:
        raise ValueError('Use the frozen V154O protocol and V154J source')
    if output.exists():
        raise FileExistsError(output)
    frozen = json.loads(Path(protocol['frozen_threshold_path']).read_text())
    if frozen != protocol['frozen_threshold']:
        raise ValueError('The frozen threshold/model binding changed')
    fit = json.loads(Path(protocol['fit_result']).read_text())
    ordered = sorted([g for g in fit['selected_groups'] if g.startswith(f'cologne1:seed{seed}:')],
                     key=lambda g: int(g.split(':')[2]))
    groups = [g for g in protocol['groups'] if g['seed'] == seed]
    if [g['group_id'] for g in groups] != [ordered[i] for i in (0, (len(ordered) - 1) // 2, len(ordered) - 1)]:
        raise ValueError('The three original B100 positions changed')
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(Path(protocol['sumocfg'])))
    if geometry != json.loads(Path(protocol['geometry_reference']).read_text())['bilateral_static_geometry']:
        raise ValueError('The fixed repaired network changed')
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = protocol['conversion_root']
    api = runtime.load_libsumo()
    if '1.22.0' not in str(runtime.libsumo_version()):
        raise ValueError('Use SUMO1.22.0')
    scratch = Path(protocol['scratch_root']) / f'seed_{seed}'
    scratch.mkdir(parents=True, exist_ok=False)
    results, traces = [], {}
    for group in groups:
        bank = original_group(group, v3)
        meta = bank['metadata']
        if (meta['sumocfg'] != protocol['sumocfg'] or meta['seed'] != seed
                or meta['behavior_policy'] != 'phase_pressure' or meta['counterfactual_horizon_sec'] != 450
                or meta['counterfactual_cost_mode'] != 'halted_queue'):
            raise ValueError('The bank collection settings changed')
        results.append(audit_group(group, bank, protocol, api, v3, scratch, traces))
    output.mkdir(parents=True)
    trajectory_path = output / 'cost_traces_server_only.json'
    trajectory_path.write_text(json.dumps(traces, separators=(',', ':')) + '\n')
    status = 'PASS' if all(g['status'] == 'PASS' for g in results) else 'FAIL_UNSAFE' if any(g['status'] == 'FAIL_UNSAFE' for g in results) else 'INVALID'
    result = {'protocol': PROTOCOL, 'status': status, 'seed': seed, 'groups': results,
        'source_root': protocol['source_root'], 'sumocfg': protocol['sumocfg'], 'frozen_threshold': frozen,
        'planned_native_branches': 24, 'planned_restored_pp_controls': 3,
        'native_branches_attempted': sum('end_time_sec' in b for g in results for b in g['native_branches']),
        'restored_pp_controls_attempted': sum('restored_pp_control' in g for g in results),
        'cost_traces_server_only': str(trajectory_path), 'elapsed_sec': time.monotonic() - started}
    (output / 'result.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('protocol', 'status', 'seed', 'elapsed_sec')}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(2027, 3037, 4047), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if run(json.loads(args.protocol.read_text()), args.seed, args.output)['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
