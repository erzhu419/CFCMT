"""Eight native Jinan action branches for dense packet/executor measurement."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from packet_observer import PacketObserver, read_static
from measurement_audit import MeasurementAuditor


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n')


def dump_line(stream, value):
    stream.write(json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n')


def executor_rows(executors):
    return {tls: dict(snapshot=ex.snapshot(), feasible_states=list(ex.feasible_states_now()),
                     minimum_green_sec=float(ex.timings[ex.current_green_state].min_green_sec))
            for tls, ex in sorted(executors.items())}


def executor_seed(executors):
    return {tls: dict(snapshot=ex.snapshot(), timings=[asdict(timing) for timing in ex.timings.values()],
                     clearance_lane_ids=list(ex.clearance_lane_ids), clearance_poll_sec=ex.clearance_poll_sec)
            for tls, ex in sorted(executors.items())}


def due_schedule(cfg):
    """Sealed supervision only; this mapping never enters the observer or PP."""
    cfg = Path(cfg)
    config = ET.parse(cfg)
    path = (cfg.parent/config.find('input/route-files').attrib['value']).resolve()
    root = ET.parse(path).getroot()
    if root.findall('flow') or root.findall('trip'):
        raise ValueError('The frozen Jinan route uses explicit integer-departure vehicles')
    by_time, records = {}, []
    for node in root.findall('vehicle'):
        depart = float(node.attrib['depart'])
        route = node.find('route')
        if not depart.is_integer() or route is None:
            raise ValueError('Use the fixed explicit integer-departure Jinan input')
        row = dict(vehicle_id=node.attrib['id'], due_time_s=int(depart),
                   entry_edge=route.attrib['edges'].split()[0], role='sealed_demand_supervision')
        by_time.setdefault(int(depart), []).append(row['vehicle_id'])
        records.append(row)
    return by_time, records


def native_prefix(api, v3, case, observer):
    """Same requests/steps/advance order as the original prefix; passive samples."""
    v3._start_sumo(api, Path(case['sumocfg']), int(case['seed']))
    infos = v3._tls_phase_infos(api)
    graph = v3.build_signal_routing_graph(api, infos)
    context = v3._scenario_context_v3(sumocfg=Path(case['sumocfg']), infos=infos, control_interval_sec=10)
    executors = v3.build_safe_phase_executors(api, infos)
    begin = float(api.simulation.getTime())
    requested = float(case.get('resolved_checkpoint_sec', case['checkpoint_sec']))
    if requested < begin or (requested-begin) % 10:
        raise ValueError('The frozen checkpoint must be a native ten-second epoch')
    ledger = v3._new_safety_ledger()
    last = observer.capture(api, begin, 'prefix_observation')
    samples, failed_prefix_samples = 1, int(not all(last['checks'].values()))

    def advance_ten():
        nonlocal last, samples, failed_prefix_samples
        states = v3._read_graph_states_v3(api, infos, context, graph)
        requests = {tls: v3._rule_candidate(states[tls], executors[tls], 'phase_pressure').state
                    for tls in sorted(infos)}
        for substep in range(10):
            v3._advance_with_actions(sumo_api=api, executors=executors,
                actions=requests if substep == 0 else {}, control_interval_sec=1,
                collision_mode='audit', safety_ledger=ledger)
            # _advance_with_actions has already advanced all live executors.
            last = observer.capture(api, float(api.simulation.getTime()), 'prefix_observation')
            samples += 1
            failed_prefix_samples += int(not all(last['checks'].values()))
            if samples % 120 == 1:
                print(json.dumps(dict(stage='prefix', time_s=last['time_s'],
                    packet_count=len(last.get('packets', {})), observer_checks=last['checks'])), flush=True)

    for _ in range(int((requested-begin)//10)):
        advance_ten()
    if case['tls_id'] not in executors:
        raise ValueError('Frozen focal TLS is absent')
    if 'resolved_checkpoint_sec' not in case:
        deadline = case['checkpoint_sec']+case['checkpoint_search_sec']
        while len(executors[case['tls_id']].feasible_states_now()) != 8:
            if float(api.simulation.getTime()) >= deadline:
                raise ValueError('No eight-action opportunity in the fixed anchor window')
            advance_ten()
    states = v3._read_graph_states_v3(api, infos, context, graph)
    prefix = dict(safety=v3._finalize_safety_ledger(ledger), samples=samples,
                  failed_prefix_samples=failed_prefix_samples, last_frame=last)
    return infos, graph, context, executors, states, prefix


def collect_branch(api, v3, case, candidate_index, output, static, sealed_due):
    import collect_tsc_physical_response_trial as native
    from cologne_action_branch_checks import physical_snapshot

    observer = PacketObserver(static)
    try:
        infos, graph, context, executors, states, prefix = native_prefix(api, v3, case, observer)
    except Exception:
        if observer.last_frame is not None:
            with gzip.open(output/'last_prefix_frame_server_only.json.gz', 'xt', compresslevel=1) as stream:
                dump_line(stream, observer.last_frame)
        raise
    candidates, reference = native.candidate_roster(v3, states, executors, case)
    index = reference if candidate_index is None else candidate_index
    t0 = float(api.simulation.getTime())
    snapshot = physical_snapshot(api, executors)  # Sealed pairing audit; never a packet feature source.
    snapshot['pending_ids'] = sorted(api.simulation.getPendingVehicles())
    cost_lanes = tuple(sorted(v3._controlled_lane_set(infos)))
    frame0 = dict(prefix.pop('last_frame'), role='preaction_observation', offset_sec=0,
                  executors=executor_rows(executors))
    write_json(output/'prefix_summary.json', prefix)
    write_json(output/'pairing_snapshot_server_only.json', snapshot)
    write_json(output/'executor_seed_server_only.json', executor_seed(executors))
    auditor = MeasurementAuditor(v3, infos, graph, context, executors, cost_lanes,
                                 focal_tls=case['tls_id'], horizon=450)
    error, steps, halted, last, inflight = None, 0, [], frame0, None
    checks = dict(prefix_no_teleport=prefix['safety']['no_teleport_passed'],
                  checkpoint_exact=t0 == case.get('resolved_checkpoint_sec', t0),
                  checkpoint_in_window=case['checkpoint_sec'] <= t0 <= case['checkpoint_sec']+60)
    ledger = v3._new_safety_ledger()
    with gzip.open(output/'frames_server_only.jsonl.gz', 'xt', compresslevel=1) as frames, \
            gzip.open(output/'execution_server_only.jsonl.gz', 'xt', compresslevel=1) as execution:
        dump_line(frames, frame0)
        try:
            if not auditor.observe(frame0, None) or not all(checks.values()):
                raise ValueError('Initial dense state or prefix admission failed')
            for step in range(1, 451):
                offset = step-1
                now = float(api.simulation.getTime())
                if now != t0+offset:
                    raise ValueError('Native clock skipped a dense measurement step')
                before = executor_rows(executors)
                requests, returns = {}, {}
                if offset % 10 == 0:
                    states = v3._read_graph_states_v3(api, infos, context, graph)
                    requests = {tls: v3._rule_candidate(states[tls], executors[tls], 'phase_pressure').state
                                for tls in sorted(executors)}
                    if offset == 0:
                        requests[case['tls_id']] = candidates[index]
                    for tls in sorted(executors):
                        returns[tls] = bool(executors[tls].request(requests[tls]))
                after = executor_rows(executors)
                applied = {tls: str(api.trafficlight.getRedYellowGreenState(tls)) for tls in sorted(executors)}
                inflight = dict(offset_sec=offset, time_sec=now, requests=requests, before=before,
                    after=after, request_returns=returns, applied_states=applied,
                    due_ids=sealed_due.get(int(now), []), role='started_step_context')
                raw = v3._advance_with_actions(sumo_api=api, executors=executors, actions={},
                    control_interval_sec=1, cost_lanes=cost_lanes, collision_mode='audit', safety_ledger=ledger)
                steps = step
                if len(raw) != 1 or raw[0] != round(raw[0]):
                    raise ValueError('Native measurement did not return an integer halting count')
                halted.append(int(raw[0]))
                last = observer.capture(api, float(api.simulation.getTime()), 'future_measurement_label')
                last.update(offset_sec=step, executors=executor_rows(executors))
                transition = dict(offset_sec=offset, time_sec=now, requests=requests, before=before,
                    after=after, request_returns=returns, applied_states=applied,
                    after_executors=last['executors'], halted_int=halted[-1],
                    due_ids=sealed_due.get(int(now), []))
                # Persist a failed frame as well as its transition before stopping.
                dump_line(frames, last)
                dump_line(execution, transition)
                if not auditor.observe(last, transition):
                    raise ValueError('Dense observation or retrospective execution audit failed')
                inflight = None
                if step % 60 == 0:
                    print(json.dumps(dict(stage='measurement', candidate_index=index,
                        completed_steps=step, total_steps=450)), flush=True)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            if inflight is not None:
                write_json(output/'failed_step_context_server_only.json', inflight)
    audit = auditor.summary(last)
    checks.update(audit.get('checks', {}))
    checks.update(no_execution_error=error is None, horizon_complete=steps == 450,
                  final_clock_exact=float(api.simulation.getTime()) == t0+450,
                  frame_observations_complete=all(last['checks'].values()))
    result = dict(status='COMPLETE' if all(checks.values()) else 'HOLD_DENSE_MEASUREMENT',
        candidate_index=index, reference_index=reference, candidate_states=candidates,
        resolved_checkpoint_sec=t0, completed_steps=steps, checks=checks, error=error,
        controlled_lane_count=len(cost_lanes), halted_integral=sum(halted) if steps == 450 else None,
        cost=sum(halted)/(450*len(cost_lanes)) if steps == 450 else None,
        prefix_safety=prefix['safety'], safety=v3._finalize_safety_ledger(ledger),
        audit=audit, snapshot=snapshot)
    return result


def collect_case(api, v3, case, output, static, sealed_due):
    from cologne_action_branch_checks import compare_physical

    output.mkdir(parents=True, exist_ok=False)
    resolved, indices, first, summaries = dict(case), [None], None, []
    for position in range(8):
        requested = indices[position]
        directory = output/('first_pp' if requested is None else f'action_{requested}')
        directory.mkdir()
        summary = dict(status='HOLD_DENSE_MEASUREMENT', candidate_index=requested)
        try:
            summary = collect_branch(api, v3, resolved, requested, directory, static, sealed_due)
            if first is None:
                first = summary
                resolved['resolved_checkpoint_sec'] = summary['resolved_checkpoint_sec']
                indices.extend(index for index in range(8) if index != first['reference_index'])
            summary['checks'].update(
                same_native_state=compare_physical(first['snapshot'], summary['snapshot'], tolerance=0)['passed'],
                same_pending=first['snapshot']['pending_ids'] == summary['snapshot']['pending_ids'],
                same_roster=first['candidate_states'] == summary['candidate_states'],
                same_pp=first['reference_index'] == summary['reference_index'])
            if not all(summary['checks'].values()):
                summary['status'] = 'HOLD_DENSE_MEASUREMENT'
        except Exception as exc:
            summary.update(status='HOLD_DENSE_MEASUREMENT', error=f'{type(exc).__name__}: {exc}')
        finally:
            try:
                api.close()
            except Exception:
                pass
        compact = {key: value for key, value in summary.items() if key != 'snapshot'}
        summaries.append(compact)
        write_json(directory/'result.json', compact)
        receipt = dict(status='PASS_DENSE_MEASUREMENT' if len(summaries) == 8 and
            all(row['status'] == 'COMPLETE' for row in summaries) else
            'COLLECTING' if compact['status'] == 'COMPLETE' else 'HOLD_DENSE_MEASUREMENT',
            case=case, resolved_checkpoint_sec=resolved.get('resolved_checkpoint_sec'),
            native_start_budget=8, new_native_starts=len(summaries), new_fits=0, branches=summaries)
        write_json(output/'result.json', receipt)
        print(json.dumps(dict(branches_completed=len(summaries), total=8,
            candidate_index=compact['candidate_index'], status=compact['status'],
            completed_steps=compact.get('completed_steps'), error=compact.get('error'))), flush=True)
        if compact['status'] != 'COMPLETE':
            break
    return receipt


def run(base, input_base):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
    from static_context import read_static_context

    if (base/'measurement').exists():
        raise ValueError('This measurement cohort has already started; never rerun or replace it')
    protocol = json.loads((HERE/'protocol.json').read_text())
    cfg = input_base/'inputs/jinan/jinan_3x4_real.sumocfg'
    case = dict(protocol['case'], sumocfg=str(cfg))
    static = read_static(cfg)
    static['conflicts_by_tls'] = read_static_context(cfg)['conflicts_by_tls']
    write_json(base/'static_graph_server_only.json', static)
    sealed_due, labels = due_schedule(cfg)
    # These are labels, separately stored from any observed packet state.
    write_json(base/'sealed_demand_labels_server_only.json', labels)
    api = load_libsumo()
    if libsumo_version(api) != '1.22.0':
        raise ValueError('Use the frozen SUMO1.22 runtime')
    result = collect_case(api, v3, case, base/'measurement', static, sealed_due)
    write_json(base/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--input-base', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base, args.input_base)
    raise SystemExit(0 if result['status'] == 'PASS_DENSE_MEASUREMENT' else 2)
