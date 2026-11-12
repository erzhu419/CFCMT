"""Replay the preserved unsupported row under the clock correction; zero fit."""
import argparse
import json
from pathlib import Path

from .training_data import iter_transitions
from .conditional_features import prepare_feature_kernel,candidate_features


def run(base):
    failure=json.loads((base/'support_failure_summary.json').read_text())['earliest_by_time']
    seed,time_s,key=failure['seed'],failure['time_sec'],failure['packet_id']
    directory=base/'training'/f'seed{seed}'
    static=json.loads((base/'static_graph_server_only.json').read_text())
    seeds=json.loads((directory/'executor_seed_server_only.json').read_text())
    laws=json.loads((base/'target_laws_server_only.json').read_text())
    for before,actual,action in iter_transitions(directory):
        if before['time_s']==time_s:
            break
    else:
        raise ValueError('The preserved failing observation is absent')
    kernel=prepare_feature_kernel(static,before,seeds,laws,action['requests'],76116999+seed*100000+int(time_s))
    candidates=[]
    probability=0.
    for old in failure['candidates']:
        new=candidate_features(kernel,key,old['intent'])
        mask=[bool(new['lane_mask'][choice]) for choice in ('left','stay','right')]
        risk=mask[0] or mask[2]
        effective=mask if risk else [False,True,False]
        if effective[failure['label']]:
            probability+=old['prior']/sum(effective)  # All three logits are zero; no fit is performed.
        candidates.append(dict(intent=old['intent'],prior=old['prior'],old_mask=old['effective_mask'],
                               fixed_mask=effective,upper=new['upper']))
    old_supported=any(row['prior']>0 and row['effective_mask'][failure['label']] for row in failure['candidates'])
    result=dict(status='PASS_KNOWN_CLOCK_CASE' if not old_supported and probability>0 else 'HOLD_KNOWN_CLOCK_CASE',
                seed=seed,time_sec=time_s,packet_id=key,label=failure['label'],old_has_support=old_supported,
                fixed_has_support=probability>0,fixed_zero_coefficient_label_probability=probability,
                candidates=candidates,observed_front_path=[event['observed_lane_path'] for event in actual['events']
                    if event.get('vehicle_id')==key and event['kind']=='lane_transition'],
                new_native_starts=0,new_fits=0,original_failed_result_unchanged=True,
                future_observations_used_for_prediction=False,
                scope='Known failed training row only; all neighbours are predicted from the same current state. No optimizer run or full-array success claim.')
    (base/'clock_fix_replay_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.base)),flush=True)
