"""Frozen PP single trajectories; six-start maximum, stop on first failure."""
import argparse
import gzip
import json
from pathlib import Path

from experiments.tsc_network_dense_measurement_v2_20261109.collect_dense import (
    executor_rows, executor_seed, due_schedule, dump_line, write_json,
)
from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver, read_static
from experiments.tsc_network_dense_measurement_v2_20261109.measurement_audit import MeasurementAuditor


def collect_one(api, v3, case, output, static, due):
    """No discovery fork or checkpoint restore: collect every second from zero."""
    v3._start_sumo(api, Path(case['sumocfg']), case['seed'])
    infos = v3._tls_phase_infos(api)
    graph = v3.build_signal_routing_graph(api, infos)
    context = v3._scenario_context_v3(sumocfg=Path(case['sumocfg']), infos=infos, control_interval_sec=10)
    executors = v3.build_safe_phase_executors(api, infos)
    lanes = tuple(sorted(v3._controlled_lane_set(infos)))
    if len(lanes) != 144:
        raise ValueError('The frozen Jinan cost roster has144 controlled lanes')
    static['controlled_lane_ids'] = list(lanes)
    static['incoming_lanes_by_tls'] = {tls:list(info.incoming_lanes) for tls,info in infos.items()}
    static['pp_lane_by_link'] = {
        tls:{index:str(group[0][0]) for index,group in enumerate(info.controlled_links) if group and group[0]}
        for tls,info in infos.items()}
    write_json(output.parent.parent/'static_graph_server_only.json',static)
    write_json(output/'executor_seed_server_only.json',executor_seed(executors))
    observer = PacketObserver(static)
    start = float(api.simulation.getTime())
    if start != 0:
        raise ValueError('A training/calibration trajectory must start natively at zero')
    last = observer.capture(api,start,'observed_training_state' if case['role']=='training' else 'observed_calibration_state')
    last.update(offset_sec=0,executors=executor_rows(executors))
    auditor = MeasurementAuditor(v3,infos,graph,context,executors,lanes,
                                 focal_tls=sorted(infos)[0],horizon=case['horizon_sec'])
    ledger = v3._new_safety_ledger()
    steps,error,inflight = 0,None,None
    with gzip.open(output/'frames_server_only.jsonl.gz','xt',compresslevel=1) as frames, \
            gzip.open(output/'execution_server_only.jsonl.gz','xt',compresslevel=1) as execution:
        dump_line(frames,last)
        try:
            if not auditor.observe(last,None):
                raise ValueError('Initial observation admission failed')
            for step in range(1,case['horizon_sec']+1):
                now=float(api.simulation.getTime())
                if now!=step-1:
                    raise ValueError('The native one-second clock skipped a step')
                before=executor_rows(executors)
                requests={}
                if (step-1)%10==0:
                    states=v3._read_graph_states_v3(api,infos,context,graph)
                    requests={tls:v3._rule_candidate(states[tls],executors[tls],'phase_pressure').state
                              for tls in sorted(executors)}
                returns={tls:bool(executors[tls].request(value)) for tls,value in sorted(requests.items())}
                after=executor_rows(executors)
                applied={tls:str(api.trafficlight.getRedYellowGreenState(tls)) for tls in sorted(executors)}
                inflight=dict(offset_sec=step-1,time_sec=now,requests=requests,before=before,after=after,
                              request_returns=returns,applied_states=applied,due_ids=due.get(int(now),[]))
                raw=v3._advance_with_actions(sumo_api=api,executors=executors,actions={},control_interval_sec=1,
                                            cost_lanes=lanes,collision_mode='audit',safety_ledger=ledger)
                steps=step
                if len(raw)!=1 or raw[0]!=round(raw[0]):
                    raise ValueError('The native producer did not return one integer halted count')
                last=observer.capture(api,float(api.simulation.getTime()),'observed_training_state'
                    if case['role']=='training' else 'observed_calibration_state')
                last.update(offset_sec=step,executors=executor_rows(executors))
                transition=dict(inflight,after_executors=last['executors'],halted_int=int(raw[0]))
                dump_line(frames,last);dump_line(execution,transition)
                if not auditor.observe(last,transition):
                    raise ValueError('Dense trajectory measurement/executor admission failed')
                inflight=None
                if step%120==0:
                    frames.flush();execution.flush()
                    print(json.dumps(dict(stage='collect',role=case['role'],seed=case['seed'],step=step,
                                          horizon=case['horizon_sec'])),flush=True)
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            if inflight is not None:
                write_json(output/'failed_step_context_server_only.json',inflight)
    audit=auditor.summary(last)
    # Auditor is shared with the450s measurement pilot; do not mislabel1800s costs.
    audit['I_horizon']=audit.pop('I450');audit['C_horizon']=audit.pop('C450')
    audit['horizon_sec']=case['horizon_sec']
    passed=error is None and steps==case['horizon_sec'] and all(audit['checks'].values())
    return dict(status='COMPLETE' if passed else 'HOLD_COLLECTION',case=case,completed_steps=steps,
                error=error,audit=audit,safety=v3._finalize_safety_ledger(ledger),new_native_starts=1,
                new_fits=0,source_data_used=False)


def collect_group(base,input_base,role):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import load_libsumo,libsumo_version
    from static_context import read_static_context

    protocol=json.loads((base/'protocol.json').read_text())
    if role not in ('training','calibration'):
        raise ValueError('Only the two frozen trajectory roles exist')
    output=base/role
    if output.exists():
        raise ValueError('This cohort has already started; never rerun or replace it')
    if role=='calibration':
        training=json.loads((base/'training/result.json').read_text())
        fit=json.loads((base/'fit_result.json').read_text())
        if training['status']!='COMPLETE' or fit['status']!='PASS_FIT' or not (base/'model.json').exists():
            raise ValueError('Freeze a complete training-only model before calibration collection')
    cfg=input_base/'inputs/jinan/jinan_3x4_real.sumocfg'
    static=read_static(cfg)
    static['conflicts_by_tls']=read_static_context(cfg)['conflicts_by_tls']
    due,labels=due_schedule(cfg)
    # Sealed measurement supervision is never a learner or free-rollout input.
    if role=='training':
        write_json(base/'sealed_due_labels_server_only.json',labels)
    api=load_libsumo()
    if libsumo_version(api)!='1.22.0':
        raise ValueError('Use the frozen SUMO1.22 runtime')
    seeds=protocol['cohorts']['training_seeds' if role=='training' else 'calibration_seeds']
    horizon=protocol['proposed_budget']['seconds_per_single_trajectory']
    output.mkdir()
    receipt=dict(status='COLLECTING',role=role,new_native_starts=0,new_fits=0,seeds=[],native_start_budget=len(seeds))
    for seed in seeds:
        directory=output/f'seed{seed}'
        directory.mkdir()
        row=dict(status='STARTING',seed=seed,new_native_starts=1)
        receipt['seeds'].append(row);receipt['new_native_starts']+=1
        write_json(directory/'result.json',row);write_json(output/'result.json',receipt)
        case=dict(seed=seed,role=role,horizon_sec=horizon,sumocfg=str(cfg))
        try:
            row.update(collect_one(api,v3,case,directory,static,due))
        except Exception as exc:
            row.update(status='HOLD_COLLECTION',error=f'{type(exc).__name__}: {exc}')
        finally:
            try:
                api.close()
            except Exception:
                pass
        write_json(directory/'result.json',row)
        receipt['status']='COMPLETE' if len(receipt['seeds'])==len(seeds) and all(
            s['status']=='COMPLETE' for s in receipt['seeds']) else 'COLLECTING' if row['status']=='COMPLETE' else 'HOLD_COLLECTION'
        write_json(output/'result.json',receipt)
        print(json.dumps(dict(stage='seed_end',role=role,seed=seed,status=row['status'],steps=row.get('completed_steps'),
                              error=row.get('error'))),flush=True)
        if row['status']!='COMPLETE':
            break
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--input-base',type=Path,required=True)
    parser.add_argument('--role',choices=['training','calibration'],required=True)
    args=parser.parse_args()
    result=collect_group(args.base,args.input_base,args.role)
    raise SystemExit(0 if result['status']=='COMPLETE' else 2)
