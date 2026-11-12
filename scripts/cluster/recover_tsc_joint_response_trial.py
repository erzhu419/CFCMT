#!/usr/bin/env python3
"""Reuse completed native branches and fill only missing v2 trial actions."""
import argparse
from copy import deepcopy
from functools import partial
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_tsc_joint_response_measurement as measured
from tsc_clearance_execution_audit import audit_execution, read_clearance_lanes

PROTOCOL = 'tsc-joint-response-development-v2'
write_json = measured.write_json


def read_json(path):
    return json.loads(path.read_text())


def revised_audit(branch, observation, static, tls, first_frame, clearance_lanes):
    checks, audit = measured.audit_branch(branch, observation, static, tls, first_frame)
    audit['nominal_actual_focal_signals_equal'] = checks.pop('nominal_actual_focal_signals_equal')
    audit['execution'] = audit_execution(observation, branch['preaction']['nominal_states'][tls],
                                        clearance_lanes, tls=tls)
    checks['execution_conforms_to_clearance_rule'] = audit['execution']['matches']
    return checks, audit


def save_group(case, output, first, first_observation, branches, summaries, static, old_attempts, new_attempts):
    complete = len(summaries) == 8 and all(row['status'] == 'COMPLETE' for row in summaries)
    result = dict(case_id=case['case_id'], city=case['city'], seed=case['seed'],
        status='COMPLETE' if complete else summaries[-1]['status'],
        resolved_checkpoint_sec=first['resolved_checkpoint_sec'] if first else None,
        new_branches_attempted=old_attempts + new_attempts,
        reused_old_attempts=old_attempts, recovery_new_branches_attempted=new_attempts,
        branches=summaries, recovery_protocol=PROTOCOL)
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
        graph = measured.build_joint_response_interface(clean, first_observation['frames'][0],
            first_observation['catalog'], first_observation['lane_geometry'], static['lane_speed_limits'],
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


def recover_case(api, v3, case, output, static, clearance_lanes, progress, remaining_budget):
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / 'result.json'
    receipt = read_json(receipt_path) if receipt_path.exists() else None
    if receipt and receipt['status'] == 'COMPLETE':
        return receipt, 0
    from cologne_action_branch_checks import compare_physical
    from tsc_physical_response_data import build_preaction_rows, extract_response_labels
    summaries, branches, first, first_observation = [], [], None, None
    old_attempts = receipt['new_branches_attempted'] if receipt else 0
    resolved, new_attempts, network_baseline = dict(case), 0, None
    if receipt:
        data = read_json(output / 'group_data_server_only.json')
        branches = data['branches']
        first_row = next(b for b in branches if b['candidate_index'] == data['reference_index'])
        first_observation = read_json(output / 'first_pp/observations_server_only.json')
        first = dict(start=read_json(output / 'physical_start_server_only.json'),
            reference_index=data['reference_index'], candidate_states=data['candidate_states'],
            preaction=first_row['preaction'], resolved_checkpoint_sec=receipt['resolved_checkpoint_sec'])
        resolved['resolved_checkpoint_sec'] = first['resolved_checkpoint_sec']
        by_index = {b['candidate_index']: b for b in branches}
        for original in receipt['branches']:
            summary = deepcopy(original)
            index = summary['candidate_index']
            directory = output / ('first_pp' if index == first['reference_index'] else f'action_{index}')
            if summary['status'] != 'COMPLETE':
                saved = by_index[index]
                branch = {**saved, 'resolved_checkpoint_sec': first['resolved_checkpoint_sec'],
                          'prefix_safety': summary['prefix_safety'], 'safety': summary['safety']}
                obs = read_json(directory / 'observations_server_only.json')
                extra, audit = revised_audit(branch, obs, static, case['tls_id'],
                                            first_observation['frames'][0], clearance_lanes)
                checks = dict(summary['checks'])
                checks.pop('nominal_actual_focal_signals_equal', None)
                checks.update(extra)
                summary.update(status='COMPLETE' if all(checks.values()) else 'INVALID',
                    checks=checks, audit=audit, reused_previous_status=original['status'])
                write_json(directory / 'result.json', summary)
            summaries.append(summary)
            if summary['status'] != 'COMPLETE':
                return save_group(case, output, first, first_observation, branches, summaries,
                                  static, old_attempts, 0), 0
        missing = [i for i in range(8) if i not in by_index]
    else:
        missing = [None]
    observer = partial(measured.JointResponseObserver, focal_tls=case['tls_id'])
    while missing:
        if new_attempts >= remaining_budget:
            raise ValueError('Recovery cannot exceed the original 224-branch budget')
        index = missing.pop(0)
        directory = output / ('first_pp' if index is None else f'action_{index}')
        directory.mkdir(exist_ok=False)
        summary = dict(candidate_index=index, status='INVALID')
        new_attempts += 1
        try:
            branch = measured.native.collect_branch(api, v3, resolved, index, directory, observer,
                                                     build_preaction_rows, extract_response_labels)
            obs = read_json(directory / 'observations_server_only.json')
            index = branch['candidate_index']; summary['candidate_index'] = index
            if first is None:
                first, first_observation = branch, obs
                resolved['resolved_checkpoint_sec'] = branch['resolved_checkpoint_sec']
                missing.extend(i for i in range(8) if i != branch['reference_index'])
                write_json(output / 'physical_start_server_only.json', first['start'])
            if network_baseline is None:
                network_baseline = branch['network_preaction']
            extra, audit = revised_audit(branch, obs, static, case['tls_id'],
                                        first_observation['frames'][0], clearance_lanes)
            checks = {**branch['checks'], **extra,
                'same_native_state': compare_physical(first['start'], branch['start'], tolerance=0)['passed'],
                'same_pending_population': first['start']['pending_ids'] == branch['start']['pending_ids'],
                'same_candidate_roster': first['candidate_states'] == branch['candidate_states'],
                'same_pp_reference': first['reference_index'] == branch['reference_index'],
                'same_movement_roster': first['preaction']['movement_ids'] == branch['preaction']['movement_ids'],
                'same_recovery_network_preaction': network_baseline == branch['network_preaction']}
            checks.pop('nominal_actual_focal_signals_equal', None)
            summary.update(status='COMPLETE' if all(checks.values()) else 'INVALID', checks=checks, audit=audit,
                measured_stopline_entries=sum(r[0] for r in branch['labels']),
                measured_receiving_entries=sum(r[1] for r in branch['labels']),
                unresolved_events=branch['measurement']['unresolved_count'],
                unobserved_first_receiving_count=branch['unobserved_first_receiving_count'],
                prefix_safety=branch['prefix_safety'], safety=branch['safety'])
            branches.append(branch)
        except measured.native.IneligibleCase as error:
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
    return save_group(case, output, first, first_observation, branches, summaries,
                      static, old_attempts, new_attempts), new_attempts


def run(spec, old_root, output):
    if spec['protocol'] != PROTOCOL or spec['max_new_branches'] != 224:
        raise ValueError('Recovery requires the v2 measurement amendment and original 224 budget')
    seeds = spec['source_seeds'] + spec['target_seeds']
    previous = {seed: read_json(old_root / f'seed_{seed}/result.json') for seed in seeds}
    old_attempts = sum(r['new_branches_attempted'] for r in previous.values())
    if old_attempts > 224:
        raise ValueError('Original collection has already exceeded its budget')
    shutil.copytree(old_root, output)
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = spec['conversion_root']
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
    from static_context import read_static_context
    if '1.22.0' not in str(libsumo_version()):
        raise ValueError('Use the frozen SUMO 1.22.0 runtime')
    api, started, new_attempts = load_libsumo(), time.monotonic(), 0
    contexts, clearances, recovered = {}, {}, []

    def progress(case, branch):
        nonlocal new_attempts
        new_attempts += 1
        print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=new_attempts, total=224 - old_attempts,
            unit='new_branch', case_id=case['case_id'], status=branch['status'],
            elapsed_seconds=time.monotonic() - started)), flush=True)

    for original in spec['cases']:
        directory = output / f"seed_{original['seed']}" / original['case_id']
        existing = read_json(directory / 'result.json') if (directory / 'result.json').exists() else None
        if existing and existing['status'] == 'COMPLETE':
            continue
        case = {**original, 'checkpoint_search_sec': spec['checkpoint_search_sec']}
        config = case['sumocfg']; key = (config, case['tls_id'])
        if config not in contexts:
            contexts[config] = read_static_context(Path(config))
        if key not in clearances:
            clearances[key] = read_clearance_lanes(Path(config), case['tls_id'])
        result, _ = recover_case(api, v3, case, directory, contexts[config], clearances[key],
                                 progress, 224 - old_attempts - new_attempts)
        recovered.append(result)
        if result['status'] != 'COMPLETE':
            break
    for seed in {row['seed'] for row in recovered}:
        groups = [read_json(output / f'seed_{seed}' / case['case_id'] / 'result.json')
                  for case in spec['cases'] if case['seed'] == seed
                  and (output / f'seed_{seed}' / case['case_id'] / 'result.json').exists()]
        write_json(output / f'seed_{seed}/result.json', dict(protocol=PROTOCOL, seed=seed,
            status='COMPLETE' if len(groups) == 2 and all(r['status'] == 'COMPLETE' for r in groups) else 'HOLD_COLLECTION',
            groups=groups, new_fits=0, new_branches_attempted=sum(g['new_branches_attempted'] for g in groups),
            recovery_new_branches_attempted=sum(g.get('recovery_new_branches_attempted', 0) for g in groups),
            parent_collection=str(old_root)))
    complete = all((output / f"seed_{c['seed']}" / c['case_id'] / 'result.json').exists()
                   and read_json(output / f"seed_{c['seed']}" / c['case_id'] / 'result.json')['status'] == 'COMPLETE'
                   for c in spec['cases'])
    result = dict(protocol=PROTOCOL, status='COMPLETE' if complete else 'HOLD_COLLECTION',
        old_attempts=old_attempts, new_branches_attempted=new_attempts,
        total_protocol_attempts=old_attempts + new_attempts, new_fits=0,
        recovered_groups=recovered, parent_collection=str(old_root), elapsed_sec=time.monotonic() - started)
    write_json(output / 'recovery_result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'old-collection-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(read_json(args.spec), args.old_collection_root, args.output)
    print(json.dumps({k: result[k] for k in ['status', 'old_attempts', 'new_branches_attempted', 'total_protocol_attempts']}), flush=True)
