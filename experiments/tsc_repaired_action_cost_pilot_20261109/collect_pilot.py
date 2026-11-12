"""Two repaired-network, native-prefix action-cost measurement groups; no fits."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from advance_measurement import advance_horizon


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n')


def collect_branch(api, v3, case, candidate_index, output):
    import collect_tsc_physical_response_trial as native
    from cologne_action_branch_checks import physical_snapshot
    from tsc_joint_response_observer import JointResponseObserver
    from tsc_physical_response_data import build_preaction_rows, extract_response_labels

    infos, graph, context, executors, states, prefix = native.native_prefix(api, v3, case)
    candidates, reference = native.candidate_roster(v3, states, executors, case)
    index = reference if candidate_index is None else candidate_index
    t0 = int(api.simulation.getTime())
    start = physical_snapshot(api, executors)
    start['pending_ids'] = sorted(api.simulation.getPendingVehicles())
    lanes = tuple(sorted(v3._controlled_lane_set(infos)))
    actions = {tls: v3._rule_candidate(states[tls], executors[tls], 'phase_pressure').state
               for tls in sorted(infos)}
    actions[case['tls_id']] = candidates[index]
    observer = JointResponseObserver(api, infos, t0, windows=(10,), focal_tls=case['tls_id'])
    preaction = build_preaction_rows(api, infos, executors, actions, observer.movements,
                                    focal_tls=case['tls_id'], horizon_sec=10)
    network = native.network_preaction(v3, states, executors, graph)
    observer.sample(t0)
    ledger = v3._new_safety_ledger()
    future = advance_horizon(api, v3, infos, graph, context, executors, actions, lanes, ledger, observer)
    observation = dict(preaction=preaction, events=observer.events, frames=observer.frames,
                       catalog=observer.movements, lane_geometry=observer.lane_geometry,
                       costs=future['costs450'])
    write_json(output/'observations_server_only.json', observation)
    write_json(output/'future_execution_server_only.json', future)
    safety = v3._finalize_safety_ledger(ledger)
    checks = dict(future['checks'])
    checks.update(native_checkpoint_exact=t0 == case.get('resolved_checkpoint_sec', t0),
                  inside_frozen_window=case['checkpoint_sec'] <= t0 <= case['checkpoint_sec']+60,
                  prefix_no_teleport=prefix['no_teleport_passed'])
    labels, measurement, first_receiving = None, None, None
    if all(checks.values()):
        measurement = observer.summary()
        first_receiving = native.unobserved_first_receiving_count(observer.frames, observer.movements)
        labels = extract_response_labels(measurement, preaction['movement_ids'], horizon_sec=10)
        checks.update(response_samples_complete=measurement['sample_count'] == 10,
                      movement_accounting_complete=measurement['complete'],
                      first_receiving_entries_observed=first_receiving == 0)
    return dict(start=start, candidate_states=candidates, reference_index=reference, candidate_index=index,
                resolved_checkpoint_sec=t0, preaction=preaction, network_preaction=network,
                labels=labels, measurement=measurement, checks=checks, observation=observation,
                cost=future['cost'], integrals=future['integrals'], segments=future['segments'],
                controlled_lane_count=len(lanes), future=future, prefix_safety=prefix, safety=safety)


def collect_case(api, v3, case, output, static, clearance_lanes):
    from cologne_action_branch_checks import compare_physical
    from recover_tsc_joint_response_trial import revised_audit

    output.mkdir(parents=True, exist_ok=False)
    resolved, indices, first = dict(case), [None], None
    summaries, complete_rows = [], []
    for position in range(8):
        requested = indices[position]
        directory = output/('first_pp' if requested is None else f'action_{requested}')
        directory.mkdir()
        summary = dict(candidate_index=requested, status='INVALID')
        try:
            branch = collect_branch(api, v3, resolved, requested, directory)
            index = branch['candidate_index']
            summary['candidate_index'] = index
            if first is None:
                first = branch
                resolved['resolved_checkpoint_sec'] = branch['resolved_checkpoint_sec']
                indices.extend(i for i in range(8) if i != branch['reference_index'])
                write_json(output/'physical_start_server_only.json', branch['start'])
            checks = dict(branch['checks'])
            if all(checks.values()):
                extra, audit = revised_audit(branch, branch['observation'], static, case['tls_id'],
                                             first['observation']['frames'][0], clearance_lanes)
                # Counts remain in safety metadata; admission concerns executable, accounted trajectories.
                extra.pop('prefix_no_collision')
                extra.pop('branch_no_collision')
                checks.update(extra)
                summary['execution_audit'] = audit
                checks.update(
                    same_native_state=compare_physical(first['start'], branch['start'], tolerance=0)['passed'],
                    same_pending_population=first['start']['pending_ids'] == branch['start']['pending_ids'],
                    same_candidate_roster=first['candidate_states'] == branch['candidate_states'],
                    same_pp_reference=first['reference_index'] == branch['reference_index'],
                    same_movement_roster=first['preaction']['movement_ids'] == branch['preaction']['movement_ids'],
                    same_network_preaction=first['network_preaction'] == branch['network_preaction'])
            status = 'COMPLETE' if all(checks.values()) else 'INVALID'
            summary.update(status=status, checks=checks, cost=branch['cost'], integrals=branch['integrals'],
                segments=branch['segments'], labels_10s=branch['labels'],
                prefix_safety=branch['prefix_safety'], safety=branch['safety'],
                elapsed_sec=api.simulation.getTime()-branch['resolved_checkpoint_sec'],
                advance_error=branch['future'].get('error'))
            if status == 'COMPLETE':
                complete_rows.append(dict(candidate_index=index, candidate_state=branch['candidate_states'][index],
                    labels=branch['labels'], cost=branch['cost'], integrals=branch['integrals'],
                    segments=branch['segments'], preaction=branch['preaction'],
                    raw_directory=directory.name))
        except Exception as error:
            summary['error'] = f'{type(error).__name__}: {error}'
        finally:
            try:
                api.close()
            except Exception:
                pass
        summaries.append(summary)
        write_json(directory/'result.json', summary)
        print(json.dumps(dict(case_id=case['case_id'], attempted=len(summaries), total=8,
                              candidate_index=summary['candidate_index'], status=summary['status'])), flush=True)
        if summary['status'] != 'COMPLETE':
            break
    complete = len(summaries) == 8 and all(row['status'] == 'COMPLETE' for row in summaries)
    result = dict(case=case, status='COMPLETE' if complete else 'HOLD_INCOMPLETE_MEASUREMENT',
                  resolved_checkpoint_sec=resolved.get('resolved_checkpoint_sec'),
                  new_branches_attempted=len(summaries), branches=summaries)
    if first is not None:
        result.update(reference_index=first['reference_index'], candidate_states=first['candidate_states'],
                      controlled_lane_count=first['controlled_lane_count'])
        rows = sorted(complete_rows, key=lambda row: row['candidate_index'])
        write_json(output/'group_data_server_only.json', dict(case=case, complete=complete,
            reference_index=first['reference_index'], controlled_lane_count=first['controlled_lane_count'],
            candidate_states=first['candidate_states'], movement_ids=first['preaction']['movement_ids'], branches=rows))
        if complete:
            pp = next(row for row in rows if row['candidate_index'] == first['reference_index'])
            result['action_table'] = [dict(candidate_index=row['candidate_index'], cost=row['cost'],
                cost_minus_pp=row['cost']-pp['cost'], integrals=row['integrals'],
                labels_10s=row['labels']) for row in rows]
    write_json(output/'result.json', result)
    return result


def run(base):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
    from static_context import read_static_context
    from tsc_clearance_execution_audit import read_clearance_lanes

    protocol = json.loads((HERE/'protocol.json').read_text())
    api = load_libsumo()
    if libsumo_version(api) != '1.22.0':
        raise ValueError('Use the fixed SUMO1.22 runtime')
    results = []
    for spec in protocol['cases']:
        filename = 'hangzhou_aligned.sumocfg' if spec['city'] == 'hangzhou' else 'jinan_3x4_real.sumocfg'
        cfg = base/'inputs'/spec['city']/filename
        case = dict(spec, sumocfg=str(cfg))
        results.append(collect_case(api, v3, case, base/'measurement'/case['case_id'],
                                   read_static_context(cfg), read_clearance_lanes(cfg, case['tls_id'])))
        if results[-1]['status'] != 'COMPLETE':
            break
    complete = len(results) == 2 and all(row['status'] == 'COMPLETE' for row in results)
    result = dict(status='PASS_MEASUREMENT' if complete else 'HOLD_INCOMPLETE_MEASUREMENT',
                  protocol=protocol, groups=results, scheduled_native_branches=16,
                  new_branches_attempted=sum(row['new_branches_attempted'] for row in results), new_fits=0)
    write_json(base/'result.json', result)
    print(result['status'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base)
    raise SystemExit(0 if result['status'] == 'PASS_MEASUREMENT' else 2)
