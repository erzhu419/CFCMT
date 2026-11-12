#!/usr/bin/env python3
"""Two fixed native action groups to validate the joint response input interface."""
import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_tsc_physical_response_trial as native
from tsc_joint_response_interface import build_joint_response_interface
from tsc_joint_response_observer import JointResponseObserver

PROTOCOL = 'tsc-joint-response-measurement-v1'
write_json = native.write_json


def validate_spec(spec):
    if (spec['protocol'] != PROTOCOL or spec['max_new_branches'] != 16
            or spec['new_fits'] != 0 or spec['response_horizon_sec'] != 10
            or spec['cost_horizon_sec'] != 450 or spec['checkpoint_search_sec'] != 60
            or len(spec['cases']) != 2
            or [c['city'] for c in spec['cases']] != ['hangzhou', 'jinan']
            or any(c['checkpoint_sec'] != 600 for c in spec['cases'])):
        raise ValueError('Use the fixed Hangzhou/Jinan sixteen-branch measurement pilot')


def audit_branch(branch, observation, static, tls, first_frame):
    frame = observation['frames'][0]
    movements = [m for m in observation['catalog'].values() if m['tls_id'] == tls]
    lanes = {lane for m in movements for role in ['incoming_lanes', 'outgoing_lanes'] for lane in m[role]}
    direct = frame.get('focal_lane_measurements', {})
    fields = ['vehicles', 'halted', 'occupancy_raw', 'mean_speed_mps']
    complete = set(direct) == lanes and all(
        all(k in r and isinstance(r[k], (int, float)) and math.isfinite(r[k]) for k in fields)
        for r in direct.values())
    count_matches = complete and all(direct[lane]['vehicles'] == sum(
        v['lane_id'] == lane for v in frame['vehicles'].values()) for lane in lanes)
    nominal = branch['preaction']['nominal_states'][tls]
    actual = [r['applied_states'][tls] for r in observation['frames'][1:]]
    mismatches = [i for i, (a, b) in enumerate(zip(nominal, actual)) if a != b]
    conflicts = static['conflicts_by_tls'].get(tls, {})
    checks = dict(direct_t0_complete=complete, direct_counts_match_positions=count_matches,
        t0_time_exact=frame['offset_sec'] == 0 and frame['time_sec'] == branch['resolved_checkpoint_sec'],
        same_t0_observations=frame == first_frame,
        full_conflict_mapping=conflicts.get('full_context_complete', False),
        exact_static_geometry=all(observation['lane_geometry'][lane]['length_m'] == static['lane_lengths'][lane]
            for m in movements for role in ['incoming_lanes', 'outgoing_lanes', 'internal_lanes'] for lane in m[role]),
        nominal_actual_focal_signals_equal=len(actual) == len(nominal) == 10 and not mismatches,
        prefix_no_collision=branch['prefix_safety']['raw_collision_events'] == 0,
        prefix_no_teleport=branch['prefix_safety']['no_teleport_passed'],
        branch_no_collision=branch['safety']['raw_collision_events'] == 0,
        branch_no_teleport=branch['safety']['no_teleport_passed'])
    return checks, dict(direct_lane_count=len(direct), nominal_actual_mismatch_step_indices=mismatches,
                        nominal_states=nominal, actual_states=actual)


def collect_case(api, v3, case, output, static, progress):
    from cologne_action_branch_checks import compare_physical
    from tsc_physical_response_data import build_preaction_rows, extract_response_labels
    output.mkdir(parents=True, exist_ok=False)
    indices, branches, summaries = [None], [], []
    resolved, first, first_observation = dict(case), None, None
    observer = partial(JointResponseObserver, focal_tls=case['tls_id'])
    for execution_index in range(8):
        index = indices[execution_index]
        directory = output / ('first_pp' if index is None else f'action_{index}')
        directory.mkdir()
        summary = dict(candidate_index=index, status='INVALID')
        try:
            branch = native.collect_branch(api, v3, resolved, index, directory, observer,
                                           build_preaction_rows, extract_response_labels)
            obs = json.loads((directory / 'observations_server_only.json').read_text())
            index = branch['candidate_index']; summary['candidate_index'] = index
            if first is None:
                first, first_observation = branch, obs
                resolved['resolved_checkpoint_sec'] = branch['resolved_checkpoint_sec']
                indices.extend(i for i in range(8) if i != branch['reference_index'])
                write_json(output / 'physical_start_server_only.json', first['start'])
            extra, audit = audit_branch(branch, obs, static, case['tls_id'], first_observation['frames'][0])
            checks = {**branch['checks'], **extra,
                'same_native_state': compare_physical(first['start'], branch['start'], tolerance=0)['passed'],
                'same_pending_population': first['start']['pending_ids'] == branch['start']['pending_ids'],
                'same_candidate_roster': first['candidate_states'] == branch['candidate_states'],
                'same_pp_reference': first['reference_index'] == branch['reference_index'],
                'same_movement_roster': first['preaction']['movement_ids'] == branch['preaction']['movement_ids'],
                'same_network_preaction': first['network_preaction'] == branch['network_preaction']}
            status = 'COMPLETE' if all(checks.values()) else 'INVALID'
            if not checks['nominal_actual_focal_signals_equal']:
                status = 'HOLD_EXECUTION'
            summary.update(status=status, checks=checks, audit=audit,
                measured_stopline_entries=sum(r[0] for r in branch['labels']),
                measured_receiving_entries=sum(r[1] for r in branch['labels']),
                unresolved_events=branch['measurement']['unresolved_count'],
                unobserved_first_receiving_count=branch['unobserved_first_receiving_count'],
                prefix_safety=branch['prefix_safety'], safety=branch['safety'])
            branches.append(branch)
        except native.IneligibleCase as error:
            summary.update(status='INELIGIBLE', error=f'{type(error).__name__}: {error}')
        except Exception as error:
            summary['error'] = f'{type(error).__name__}: {error}'
        finally:
            try:
                api.close()
            except Exception:
                pass
        summaries.append(summary)
        write_json(directory / 'result.json', summary)
        progress(case, summary)
        if summary['status'] != 'COMPLETE':
            break
    complete = len(summaries) == 8 and all(r['status'] == 'COMPLETE' for r in summaries)
    result = dict(case_id=case['case_id'], city=case['city'], seed=case['seed'],
        status='COMPLETE' if complete else summaries[-1]['status'],
        resolved_checkpoint_sec=resolved.get('resolved_checkpoint_sec'),
        new_branches_attempted=len(summaries), branches=summaries)
    if branches:
        branches.sort(key=lambda b: b['candidate_index'])
        group = dict(case=case, case_id=case['case_id'], city=case['city'], seed=case['seed'],
            complete=complete, reference_index=first['reference_index'], candidate_states=first['candidate_states'],
            feature_names=first['preaction']['feature_names'], movement_ids=first['preaction']['movement_ids'],
            X=[b['preaction']['features'] for b in branches], Y=[b['labels'] for b in branches],
            branches=[{k: b[k] for k in ['candidate_index', 'preaction', 'labels', 'cost', 'costs']} for b in branches])
        write_json(output / 'group_data_server_only.json', group)
    if complete:
        clean = {k: group[k] for k in ['case_id', 'reference_index', 'feature_names', 'movement_ids', 'X']}
        clean['branches'] = [{'candidate_index': b['candidate_index'], 'preaction': {
            k: b['preaction'][k] for k in ['nominal_states', 'nominal_request_accepted']}} for b in branches]
        graph = build_joint_response_interface(clean, first_observation['frames'][0], first_observation['catalog'],
            first_observation['lane_geometry'], static['lane_speed_limits'],
            conflict_relations=static['conflicts_by_tls'][case['tls_id']])
        write_json(output / 'joint_interface_server_only.json', graph)
        incoming = [n for n in graph['lane_nodes'] if 'incoming' in n['roles']]
        result['input_summary'] = dict(interface_protocol=graph['protocol'], candidate_views=len(graph['candidate_views']),
            incoming_lane_count=len(incoming), incoming_vehicle_count=sum(n['vehicle_count'] for n in incoming),
            near_stopline_proxy_vehicle_count=sum(n['near_stopline_vehicle_count'] for n in incoming),
            direct_incoming_lane_count=sum(n['lane_measurements_source'] == 'direct_t0_focal_lane' for n in incoming),
            incoming_halted_count=sum(n['lane_measurements']['halted'] for n in incoming))
    write_json(output / 'result.json', result)
    return result


def run(spec, output):
    validate_spec(spec)
    output.mkdir(parents=True, exist_ok=False)
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = spec['conversion_root']
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
    from static_context import read_static_context
    if '1.22.0' not in str(libsumo_version()):
        raise ValueError('Use the frozen SUMO 1.22.0 runtime')
    started, attempted, groups = time.monotonic(), 0, []

    def progress(case=None, branch=None):
        nonlocal attempted
        attempted += int(branch is not None)
        print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=attempted, total=16, unit='branch',
            elapsed_seconds=time.monotonic() - started, case_id=case['case_id'] if case else None,
            status=branch['status'] if branch else 'START')), flush=True)

    progress()
    api = load_libsumo()
    for original in spec['cases']:
        case = {**original, 'checkpoint_search_sec': spec['checkpoint_search_sec']}
        static = read_static_context(Path(case['sumocfg']))
        result = collect_case(api, v3, case, output / case['case_id'], static, progress)
        groups.append(result)
        if result['status'] != 'COMPLETE':
            break
    result = dict(protocol=PROTOCOL, status='PASS_MEASUREMENT' if len(groups) == 2
        and all(g['status'] == 'COMPLETE' for g in groups) else 'HOLD_MEASUREMENT',
        groups=groups, new_fits=0, new_branches_attempted=attempted,
        elapsed_sec=time.monotonic() - started, old_response_gate='HOLD_RESPONSE')
    write_json(output / 'result.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'new_fits', 'new_branches_attempted']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.spec.read_text()), args.output)
