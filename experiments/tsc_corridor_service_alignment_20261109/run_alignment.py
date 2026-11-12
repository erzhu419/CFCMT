"""Evaluate one service-window alignment using the frozen count-fusion forecaster."""
import argparse
import gzip
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
for sibling in ('tsc_corridor_arrival_transfer_20261107','tsc_corridor_residual_transfer_20261109','tsc_corridor_count_fusion_20261109'):
    sys.path.insert(0,str(HERE.parent/sibling))
sys.path.insert(0,str(HERE))
from arrival_policy import ArrivalPolicy,fit_hours as old_fit
from model_gate import hour,load_training,verify_protocol
from residual_model import fit_hours,fit_source,PSEUDOCOUNT,PERMUTATION_SEED
from aligned_predictor import AlignedPredictor
from evaluate_alignment import evaluate_hour,summarize


def run(root,cohort_file,frozen_result,coverage_file,output):
    plan=json.loads((HERE/'protocol.json').read_text())
    cohort=json.loads(cohort_file.read_text())['cities']
    expected=json.loads(frozen_result.read_text())
    coverage=json.loads(coverage_file.read_text())
    verify_protocol(cohort)
    if coverage['status']!='PASS_EXACT_RECONSTRUCTION':
        raise ValueError('Original unlocked records are not fully reconstructible')
    for city in cohort:
        original_unlocked=sum(r['counts']['unlocked_decisions'] for r in expected['cities'][city]['by_seed'])
        if original_unlocked!=coverage['cities'][city]['counts']['unlocked_reconstructed']:
            raise ValueError('Coverage check changed the original unlocked denominator')
    if (PSEUDOCOUNT!=plan['physical_pseudocount'] or PERMUTATION_SEED!=plan['permutation_seed']
            or AlignedPredictor.service_seconds!=plan['service_score_horizon_sec']
            or AlignedPredictor.detector_seconds!=plan['detector_prediction_horizon_sec']):
        raise ValueError('Predictor constants differ from the fixed alignment protocol')
    sources,old_sources={},{}
    metadata=dict(source={},target={})
    for city,spec in cohort.items():
        sources[city]=fit_source(root,'donor',city,spec['donor'])
        labels=load_training(root,'donor',city,spec['donor'])
        old_sources[city]=old_fit(labels),old_fit(labels,permute=True)
        metadata['source'][city]=dict(real=sources[city][0].metadata,placebo=sources[city][1].metadata)
        if metadata['source'][city]!=expected['training']['source'][city]:
            raise ValueError('Frozen source training metadata changed')
        print(f'Restored unchanged source/{city}',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    summaries={}
    with gzip.open(output/'aligned_request_disagreements_server_only.jsonl.gz','wt') as stream:
        for city,donor in (('hangzhou','jinan'),('jinan','hangzhou')):
            previous={r['seed']:r for r in expected['cities'][city]['by_seed']}
            metadata['target'][city]=[];panel=[]
            for block,seeds in enumerate(cohort[city]['adapt']):
                target=fit_hours(root,'adapt',city,seeds)
                record=dict(block=block,**target.metadata)
                if record!=expected['training']['target'][city][block]:
                    raise ValueError('Frozen target adaptation metadata changed')
                metadata['target'][city].append(record)
                policy=ArrivalPolicy(old_fit(load_training(root,'adapt',city,seeds)),*old_sources[donor])
                for seed in cohort[city]['dev'][2*block:2*block+2]:
                    predictor=AlignedPredictor(target,*sources[donor])
                    result=evaluate_hour(city,seed,block,hour(root,'dev',city,seed),predictor,policy,previous[seed],stream)
                    result['prediction_support']=dict(predictor.counts);panel.append(result)
                    primary=result['scopes']['original_common_scored']
                    print(f'Aligned {city}/{seed}; old predictions/requests reproduced; ST={primary["action_differences"]["S_vs_T"]}, SP={primary["action_differences"]["S_vs_P"]}',flush=True)
            summaries[city]=summarize(panel)
    passed=all(c['scopes']['original_common_scored']['passed'] for c in summaries.values())
    result=dict(status='DEVELOPMENT_SIGNAL_ONLY' if passed else 'HOLD_SERVICE_ALIGNMENT_DEVELOPMENT',
        previous_status=expected['status'],model_changed=False,score_horizon_changed=True,
        new_simulations=0,newly_collected_labels=0,independent_confirmation=0,action_branches=0,
        reused_development_seeds=True,training=metadata,coverage=coverage,cities=summaries,
        previous_10s_summary={city:c['scopes'] for city,c in expected['cities'].items()},model_protocol=plan,
        label_score_interpretation='PP observed detector arrivals inserted into nominal13s service plans; not actual counterfactual cost or realized service.')
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','cohort-file','frozen-result','coverage-file','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.cohort_file,a.frozen_result,a.coverage_file,a.output)
