"""Register existing904 current states and cache costs, without model queries."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from math import fsum, isclose
from pathlib import Path
import time
import traceback

from experiments.tsc_packet_conditional_census_20261110.census import FrameSelector, current_eligibility, selected_current_records
from experiments.tsc_packet_conditional_domain_20261110.budget import symbolic_budget
from experiments.tsc_packet_conditional_domain_20261110.current import GeometryView
from experiments.tsc_packet_conditional_domain_20261110.register import RESERVOIR_SEED
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden

from .cohort import census_and_sample
from .inventory import cache_inventory, cost_plan


SEED=76116904
COSTS=('all_exact_full_forward_ceiling','all_exact_feature_ceiling','all_exact_leaf_ceiling',
       'all_mixture_full_forward_ceiling','all_mixture_feature_ceiling',
       'proposed_new_full_forward_ceiling','proposed_new_feature_ceiling','proposed_new_exact_leaf_ceiling')


def read(path):
    return json.loads(path.read_text())


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def key(row):
    return int(row['seed']),int(row['time_sec']),row['packet_id']


def freeze_models(base,output):
    models={}
    for name,folder,status in (
            ('old_lc','target_packet_lc_refit','COMPLETE_LC_ONLY_REFIT'),
            ('new_lc','target_packet_lc_stratified_refit','COMPLETE_STRATIFIED_LC_REFIT')):
        receipt=read(base/folder/'execution.json')
        path=base/folder/'fit/coefficients_server_only.json'
        if receipt['status']!=status or receipt['new_fits']!=1 or receipt['optimizer_invocations']!=1 or not path.is_file():
            raise ValueError('The completed frozen old/new single LC fits are required')
        models[name]=dict(coefficient_file=str(path),completed_at_utc=receipt['completed_at_utc'])
    original=base/'target_packet_particle_first_fit/fit/coefficients_server_only.json'
    if not original.is_file():
        raise FileNotFoundError(original)
    models['frequency_bias']=dict(non_lc_rows_source=str(original),training_class_counts=dict(left=165,right=166,stay=29669),
        recipe='LC nonbias coefficients0; left/right bias log(training direction count/29669). Do not replace these counts with904 counts.')
    save(output/'frozen_models_server_only.json',dict(models=models,coefficient_arrays_read=False,
        frozen_before_sampling_at_utc=datetime.now(timezone.utc).isoformat()))
    return models


def register_current_cases(base,output,refs,available):
    folder=output/f'seed{SEED}'
    folder.mkdir(exist_ok=False)
    static=read(base/'target_packet_training/static_graph_server_only.json')
    laws=read(base/'target_packet_training/target_laws_server_only.json')
    grouped=defaultdict(list)
    for row in refs:
        grouped[row['time_sec']].append(row)
    cases=[]
    frames=0
    for frame,action,_ in selected_current_records(base/'target_packet_training/training'/f'seed{SEED}',grouped):
        t=int(frame['time_s'])
        view=GeometryView(static,frame,laws,action['applied_states'],RESERVOIR_SEED+SEED*100000+t)
        selector=FrameSelector(view)
        filename=f't{t}_current_server_only.json.gz'
        with gzip.open(folder/filename,'wt',compresslevel=1) as stream:
            json.dump(dict(frame=frame,action=action,internal_bindings=view.internal_bindings),stream,allow_nan=False)
        frames+=1
        for ref in grouped[t]:
            selection=selector.select(ref['packet_id'],max_stochastic=10)
            eligible=current_eligibility(view,view.state['packets'][ref['packet_id']])
            if (selection['required_participants']!=ref['required_participants']
                    or len(selection['required_participants'])!=ref['required_actor_count']
                    or len(view.state['packets'])!=ref['frame_actor_count'] or eligible!=ref['eligibility']):
                raise ValueError(f'The archived current domain changed at {key(ref)}')
            budget=symbolic_budget(view,selection)
            status=budget['status'] if eligible['lane_change'] else 'OUTSIDE_LC_SELECTOR'
            if status!='REGISTERED_SYMBOLIC_UPPER_BOUND':
                budget=dict(budget,status=status)
            for field,oldfield in (('physical_resolver_calls_upper_bound','physical_calls_upper'),
                                   ('lateral_leaf_branches_upper_bound','leaf_visits_upper')):
                if ref.get(oldfield) is not None and ref[oldfield]!=budget[field]:
                    raise ValueError(f'A known symbolic cost changed at {key(ref)}')
            plan=cost_plan(budget,selection['required_participants'],ref['frame_actor_count'],available[key(ref)])
            cases.append(dict(ref,current_cache=f'seed{SEED}/{filename}',eligibility=eligible,
                symbolic_status=status,symbolic_budget=budget,available_caches=available[key(ref)],**plan))
        if frames%100==0:
            print(json.dumps(dict(stage='904_symbolic_progress',frames=frames,cases=len(cases))),flush=True)
    if len(cases)!=340 or {key(r) for r in cases}!={key(r) for r in refs}:
        raise ValueError('Every selected904 row must remain registered')
    cases.sort(key=key)
    save(folder/'registered_server_only.json',dict(seed=SEED,references=cases))
    ready=[r for r in cases if r['symbolic_status']=='REGISTERED_SYMBOLIC_UPPER_BOUND']
    totals={field:sum(row[field] for row in ready) for field in COSTS}
    methods={}
    for method in ('exact','mixture256'):
        rows=[r for r in ready if r['proposed_method']==method]
        methods[method]=dict(cases=len(rows),reused=sum(r['proposed_reused_cache'] is not None for r in rows),
            new=sum(r['proposed_reused_cache'] is None for r in rows),
            new_full_forward_ceiling=sum(r['proposed_new_full_forward_ceiling'] for r in rows),
            new_feature_ceiling=sum(r['proposed_new_feature_ceiling'] for r in rows),
            new_exact_leaf_ceiling=sum(r['proposed_new_exact_leaf_ceiling'] for r in rows))
    return dict(reference_count=len(cases),current_frames_decoded=frames,registered_cases=len(ready),
        unregistered_cases=len(cases)-len(ready),symbolic_status_counts=dict(Counter(r['symbolic_status'] for r in cases)),
        actor_count_histogram=dict(Counter(r['required_actor_count'] for r in cases)),
        methods=methods,proposed_reused_cases=sum(r['proposed_reused_cache'] is not None for r in ready),
        reused_positive_cases=sum(r['y']!=1 and r['proposed_reused_cache'] is not None for r in ready),
        reused_stay_cases=sum(r['y']==1 and r['proposed_reused_cache'] is not None for r in ready),
        partial_registered_costs=totals,complete_costs=totals if len(ready)==len(cases) else None)


def run(base,output):
    output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    result=dict(status='HOLD_904_NATURAL_REGISTRATION',new_full_forward_calls=0,new_canonical_feature_evaluations=0,
        new_native_starts=0,new_fits=0,optimizer_invocations=0,new_numerical_likelihood_readouts=0,
        new_nuisance_estimations=0,new_particle_pools=0,native_next_pairs_formed=0,
        coefficient_arrays_read=False,models_recalibrated=False,model_admitted=False,
        exposure='904 has prior and historical development exposure; internal regression only.',
        started_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'started.json',result)
    try:
        with physical_forbidden():
            result['frozen_models']=freeze_models(base,output)
            refs,result['cohort']=census_and_sample(base,output/'cohort')
            refs.sort(key=key)
            for index,row in enumerate(refs):
                row['case_index']=index
            if (len(refs)!=340 or len({key(r) for r in refs})!=340 or any(r['seed']!=SEED for r in refs)
                    or Counter(r['y'] for r in refs)!={0:56,1:223,2:61}
                    or not isclose(fsum(r['raw_weight'] for r in refs),10000.,abs_tol=1e-9)
                    or not isclose(fsum(r['objective_weight'] for r in refs),1.,abs_tol=1e-12)):
                raise ValueError('The frozen340-row904 natural-weight cohort changed')
            save(output/'cohort_server_only.json',dict(result['cohort'],references=refs))
            print(json.dumps(dict(stage='904_cohort_frozen',summary=result['cohort'])),flush=True)
            available,result['inventory']=cache_inventory(base,{key(r) for r in refs})
            result['registration']=register_current_cases(base,output,refs,available)
            result['status']=('COMPLETE_904_NATURAL_REGISTRATION'
                if result['registration']['unregistered_cases']==0 else 'HOLD_904_CURRENT_DOMAIN')
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    result.update(elapsed_sec=time.monotonic()-began,completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('status','error','elapsed_sec')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if run(args.base,args.output)['status']=='COMPLETE_904_NATURAL_REGISTRATION' else 2)
