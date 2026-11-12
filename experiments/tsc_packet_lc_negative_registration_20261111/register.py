"""Current-only stratified negative registration and symbolic cache costs."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from math import fsum
from pathlib import Path
import time
import traceback

from experiments.tsc_packet_conditional_census_20261110.census import (
    FrameSelector, current_eligibility, selected_current_records,
)
from experiments.tsc_packet_conditional_domain_20261110.budget import symbolic_budget
from experiments.tsc_packet_conditional_domain_20261110.current import GeometryView
from experiments.tsc_packet_conditional_domain_20261110.register import RESERVOIR_SEED
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden

from .census import census_population
from .inventory import cache_inventory, cost_plan, key
from .sample import SEEDS, select_cohort


COSTS=('all_exact_full_forward_ceiling','all_exact_feature_ceiling','all_exact_leaf_ceiling',
       'all_mixture_full_forward_ceiling','all_mixture_feature_ceiling',
       'proposed_new_full_forward_ceiling','proposed_new_feature_ceiling','proposed_new_exact_leaf_ceiling')


def read(path):
    return json.loads(path.read_text())


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def coverage_counts(population,selected,old_stay_keys):
    pools={'population':population,'sampled':selected,
           'old93':[r for r in population if r['y']==1 and key(r) in old_stay_keys]}
    counters={}
    for name,rows in pools.items():
        counts=Counter()
        for row in rows:
            if row['y']==1:
                for direction in ('left','right'):
                    for identity in row['directions'][direction]['matched_positive_keys']:
                        counts[tuple(identity)]+=1
        counters[name]=counts
    details=[]
    for row in population:
        if row['y']!=1:
            details.append(dict(seed=row['seed'],time_sec=row['time_sec'],packet_id=row['packet_id'],y=row['y'],
                **{name+'_matching_stays':counts[key(row)] for name,counts in counters.items()}))
    summary={}
    for y,direction in ((0,'left'),(2,'right')):
        rows=[r for r in details if r['y']==y]
        summary[direction]=dict(positive_count=len(rows),
            population_covered=sum(r['population_matching_stays']>0 for r in rows),
            sampled_covered=sum(r['sampled_matching_stays']>0 for r in rows),
            old93_covered=sum(r['old93_matching_stays']>0 for r in rows),
            per_seed={str(seed):dict(positive_count=sum(r['seed']==seed for r in rows),
                **{name+'_covered':sum(r['seed']==seed and r[name+'_matching_stays']>0 for r in rows)
                   for name in counters}) for seed in SEEDS})
    return summary,details


def seed_registration(seed,refs,base,output,static,laws,available):
    began=time.monotonic()
    folder=output/f'seed{seed}'
    folder.mkdir(exist_ok=False)
    grouped=defaultdict(list)
    for row in refs:
        grouped[row['time_sec']].append(row)
    cases=[]
    frames=0
    with physical_forbidden():
        for frame,action,_ in selected_current_records(base/'target_packet_training/training'/f'seed{seed}',grouped):
            t=int(frame['time_s'])
            view=GeometryView(static,frame,laws,action['applied_states'],RESERVOIR_SEED+seed*100000+t)
            selector=FrameSelector(view)
            name=f't{t}_current_server_only.json.gz'
            with gzip.open(folder/name,'wt',compresslevel=1) as stream:
                json.dump(dict(frame=frame,action=action,internal_bindings=view.internal_bindings),stream,allow_nan=False)
            frames+=1
            for ref in grouped[t]:
                focal=ref['packet_id']
                selection=selector.select(focal,max_stochastic=10)
                eligible=current_eligibility(view,view.state['packets'][focal])
                if (selection['required_participants']!=ref['required_participants']
                        or len(selection['required_participants'])!=ref['required_actor_count']
                        or len(view.state['packets'])!=ref['frame_actor_count']
                        or eligible!=ref['eligibility']):
                    raise ValueError(f'The archived current domain changed at {key(ref)}')
                budget=symbolic_budget(view,selection)
                status=budget['status'] if eligible['lane_change'] else 'OUTSIDE_LC_SELECTOR'
                if status!='REGISTERED_SYMBOLIC_UPPER_BOUND':
                    budget=dict(budget,status=status)
                for field,oldfield in (('physical_resolver_calls_upper_bound','physical_calls_upper'),
                                       ('lateral_leaf_branches_upper_bound','leaf_visits_upper')):
                    if ref.get(oldfield) is not None and ref[oldfield]!=budget[field]:
                        raise ValueError(f'A previously registered symbolic cost changed at {key(ref)}')
                plan=cost_plan(budget,selection['required_participants'],ref['frame_actor_count'],available[key(ref)])
                cases.append(dict(ref,current_cache=f'seed{seed}/{name}',eligibility=eligible,
                    symbolic_status=status,symbolic_budget=budget,available_caches=available[key(ref)],**plan))
            if frames%100==0:
                print(json.dumps(dict(stage='selected_symbolic_progress',seed=seed,frames=frames,cases=len(cases))),flush=True)
    if len(cases)!=len(refs) or {key(r) for r in cases}!={key(r) for r in refs}:
        raise ValueError('Every selected row must retain its current-domain registration')
    cases.sort(key=key)
    save(folder/'registered_server_only.json',dict(seed=seed,references=cases))
    ready=[r for r in cases if r['symbolic_status']=='REGISTERED_SYMBOLIC_UPPER_BOUND']
    partial={field:sum(row[field] for row in ready) for field in COSTS}
    summary=dict(seed=seed,reference_count=len(cases),current_frames_decoded=frames,
        symbolic_status_counts=dict(Counter(r['symbolic_status'] for r in cases)),
        registered_cases=len(ready),unregistered_cases=len(cases)-len(ready),
        actor_count_histogram=dict(Counter(r['required_actor_count'] for r in cases)),
        proposed_method_counts=dict(Counter(r['proposed_method'] for r in ready)),
        proposed_reused_cases=sum(r['proposed_reused_cache'] is not None for r in ready),
        reused_positive_cases=sum(r['y']!=1 and r['proposed_reused_cache'] is not None for r in ready),
        reused_stay_cases=sum(r['y']==1 and r['proposed_reused_cache'] is not None for r in ready),
        partial_registered_costs=partial,complete_costs=partial if len(ready)==len(cases) else None,
        elapsed_sec=time.monotonic()-began)
    save(folder/'summary.json',summary)
    print(json.dumps(dict(stage='selected_symbolic_complete',**summary)),flush=True)
    return summary


def run(base,output):
    output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    result=dict(status='HOLD_STRATIFIED_NEGATIVE_REGISTRATION',new_full_forward_calls=0,
        new_canonical_feature_evaluations=0,new_native_starts=0,new_fits=0,optimizer_invocations=0,
        new_nuisance_estimations=0,new_particle_pools=0,numerical_likelihood_readouts=0,
        native_next_pairs_formed=0,calibration_seeds_used=False,regression_seed_used_for_sampling=False,
        regression_current_frames_decoded=0,
        original_fit_hold_preserved=True,model_admitted=False,started_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'started.json',result)
    try:
        with physical_forbidden():
            population,result['census']=census_population(base,output/'census')
            cohort,result['cohort']=select_cohort(population)
            for index,row in enumerate(cohort):
                row['case_index']=index
            save(output/'cohort_server_only.json',dict(result['cohort'],references=cohort))
            old=read(base/'target_packet_positive_covered_registration/registration/cohort_server_only.json')['references']
            old_keys={key(r) for r in old if r['y']==1}
            if len(old_keys)!=93:
                raise ValueError('The previous93 stays must remain available for coverage comparison')
            result['coverage'],details=coverage_counts(population,cohort,old_keys)
            save(output/'positive_coverage_server_only.json',details)
            result['old93_selected']=sum(r['y']==1 and key(r) in old_keys for r in cohort)
            del population
            print(json.dumps(dict(stage='cohort_frozen',coverage=result['coverage'],cohort=result['cohort'])),flush=True)
            available,result['inventory']=cache_inventory(base,{key(r) for r in cohort})
            cache=base/'target_packet_training'
            static,laws=read(cache/'static_graph_server_only.json'),read(cache/'target_laws_server_only.json')
            with ProcessPoolExecutor(max_workers=3) as pool:
                futures=[pool.submit(seed_registration,seed,[r for r in cohort if r['seed']==seed],
                                     base,output,static,laws,available) for seed in SEEDS]
                result['seeds']=[future.result() for future in futures]
            summaries=result['seeds']
            if sum(s['reference_count'] for s in summaries)!=1000:
                raise ValueError('The full1000-row cohort was not registered')
            ready=all(s['unregistered_cases']==0 for s in summaries)
            partial={field:sum(s['partial_registered_costs'][field] for s in summaries) for field in COSTS}
            methods,actors=Counter(),Counter()
            for s in summaries:
                methods.update(s['proposed_method_counts'])
                actors.update(s['actor_count_histogram'])
            result.update(status='COMPLETE_STRATIFIED_NEGATIVE_REGISTRATION' if ready else 'HOLD_STRATIFIED_NEGATIVE_CURRENT_DOMAIN',
                total_selected_rows=1000,total_current_frames_decoded=sum(s['current_frames_decoded'] for s in summaries),
                current_actor_count_histogram=dict(actors),proposed_method_counts=dict(methods),
                proposed_reused_cases=sum(s['proposed_reused_cases'] for s in summaries),
                reused_positive_cases=sum(s['reused_positive_cases'] for s in summaries),
                reused_stay_cases=sum(s['reused_stay_cases'] for s in summaries),
                partial_registered_costs=partial,complete_costs=partial if ready else None)
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    result.update(elapsed_sec=time.monotonic()-began,completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('status','error','coverage','complete_costs','elapsed_sec')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if run(args.base,args.output)['status']=='COMPLETE_STRATIFIED_NEGATIVE_REGISTRATION' else 2)
