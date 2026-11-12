"""Exercise real static geometry using entirely invented packets and priors.

Reads static graph and executor timings only. No native simulator or recorded
vehicle trajectory is loaded, and no fitting/calibration takes place.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

from .kernel import PacketKernel
from .synthetic_fixtures import packet, initial_observation, PARAMETERS


def run(static, configurations, *, count=700, steps=10):
    ordinary=[lane for lane,row in static['lanes'].items() if not row['internal']]
    packets=[]
    for index in range(count):
        lane=ordinary[index%len(ordinary)]
        position=static['lanes'][lane]['length_m']-8*(1+index//len(ordinary))
        if position<5:
            raise ValueError('The artificial packet inventory does not fit this graph')
        packets.append(packet(f'synthetic:{index}',lane,position,5.+index%3))
    configs={}
    for tls,seed in configurations.items():
        first=seed['timings'][0]
        configs[tls]=dict(timings=seed['timings'],clearance_lane_ids=seed['clearance_lane_ids'],
            clearance_poll_sec=seed['clearance_poll_sec'],snapshot=dict(tls_id=tls,mode='green',
            current_state=first['state'],current_green_state=first['state'],target_green_state=first['state'],
            green_elapsed_sec=first['min_green_sec'],remaining_sec=0.,pending_all_red_sec=0.,audit={}))
    choices={row['edge_id']:{'END'} for row in static['lanes'].values() if not row['internal']}
    for connection in static['connections']:
        if not static['lanes'][connection['from_lane']]['internal']:
            choices[static['lanes'][connection['from_lane']]['edge_id']].add(static['lanes'][connection['to_lane']]['edge_id'])
    target=dict(intent_probs={road:{choice:1/len(values) for choice in sorted(values)} for road,values in choices.items()},
                due_rate_per_sec=0.,entry_probs={},profiles=[dict(parameters=PARAMETERS,type_id='synthetic')])
    kernel=PacketKernel(static,initial_observation(static,packets,executors=configs),configs,target,random_seed=424242)
    times=[]
    rows=[]
    for offset in range(steps):
        begin=perf_counter()
        row=kernel.step()
        times.append(perf_counter()-begin)
        rows.append(row)
        print(json.dumps(dict(synthetic_step=offset+1,active=row['active_count'],wall_sec=times[-1])),flush=True)
    return dict(status='PASS_SYNTHETIC_STATIC_SCALE',initial_artificial_packets=count,completed_steps=steps,
                initial_active=count,final_active=len(kernel.state['packets']),
                exits=sum(len(row['exited_ids']) for row in rows),identity_all_passed=all(row['ledger_passed'] for row in rows),
                total_wall_sec=sum(times),seconds_per_model_step=sum(times)/len(times),
                lane_count=len(static['lanes']),native_starts=0,real_data_fits=0,
                parameter_origin='untrained_zero_placeholder',
                role='Only static-geometry/API/scale smoke with invented packets, uniform intent laws and static first-green executors. Not observed-state prediction or calibration.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--static-json',type=Path,required=True)
    parser.add_argument('--executor-seeds-json',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(json.loads(args.static_json.read_text()),json.loads(args.executor_seeds_json.read_text()))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
