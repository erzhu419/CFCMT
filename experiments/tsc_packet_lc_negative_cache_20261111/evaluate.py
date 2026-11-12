"""Build only missing frozen training caches, then read four fixed points."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import gzip
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from experiments.tsc_packet_conditional_cache_20261110 import cache as exact
from experiments.tsc_packet_conditional_cache_20261110.cache import CachedLaw, _vectors
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_packet_lane_change_holdout_20261111.evaluate import exact_budget
from experiments.tsc_packet_lc_refit_20261111.fit import frequency_reference
from experiments.tsc_packet_mixture_confirmation_20261111.evaluate import frozen_candidate, read_json
from experiments.tsc_packet_mixture_confirmation_20261111.readout import MixtureParticleLaw
from experiments.tsc_packet_mixture_confirmation_20261111.sampler import MixtureSampler
from experiments.tsc_packet_real_particle_pilot_20261110.evaluate import collect_receipts, primary_response_budget
from experiments.tsc_packet_real_particle_pilot_20261110.register import save
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES

from .prepare import prepare
from .readout import aggregate_records, read_case


LIMITS=dict(physical_calls=32384,canonical_feature_evaluations=60888,leaf_visits=110478)
_kernels=None


def set_worker_kernels(kernels):
    global _kernels
    _kernels=kernels


def frozen_points(base):
    original=frozen_candidate(base)
    root=base/'target_packet_lc_refit/fit'
    fitted=read_json(root/'result.json')
    saved=read_json(root/'coefficients_server_only.json')
    returned=_vectors(saved['coefficients'])
    if (fitted['status']!='COMPLETE_LC_ONLY_REFIT' or not fitted['optimizer']['success']
            or saved['parameter_names']!=list(PARAMETERS) or saved['feature_names']!=list(FEATURE_NAMES)
            or not np.array_equal(returned[:3],original[:3])):
        raise ValueError('The single frozen LC-refitted point and its unchanged non-LC rows are required')
    return dict(zero=np.zeros_like(original),original=original,
                frequency_bias=frequency_reference(original),refitted=returned)


def load_reused_law(task,anchor):
    receipt=task['proposed_reused_cache']
    with gzip.open(receipt['particle_path'],'rt') as stream:
        data=json.load(stream)
    if (data['focal_id']!=task['reference']['packet_id']
            or data['stochastic_participants']!=task['required_participants']
            or data['frame_actor_count']!=task['frame_actor_count'] or data['eligibility']!=task['eligibility']
            or data['parameter_names']!=list(PARAMETERS) or data['feature_names']!=list(FEATURE_NAMES)):
        raise ValueError('A reused cache changed its reference, layout or current domain')
    if task['proposed_method']=='exact':
        if receipt['kind']!='exact':
            raise ValueError('An exact reference requires an exact cache')
        return CachedLaw(data)
    if (receipt['kind']!='mixture' or receipt['particle_count']!=256 or len(data['particles'])!=256
            or data['proposal']['design_weights']!={'0':.5,'1':.5}
            or data['proposal']['seeds']!={'0':receipt['seed_zero'],'1':receipt['seed_fitted']}):
        raise ValueError('A reused mixture changed its fixed equal proposal or recorded seeds')
    return MixtureParticleLaw(data,anchor)


def inspect_case(task,points,output):
    directory=output/task['stem']
    directory.mkdir(exist_ok=False)
    save(directory/'started.json',task)
    began=time.monotonic()
    counts=dict(physical_calls=0,canonical_feature_evaluations=0,leaf_visits=0)
    sampler=None
    row={k:task[k] for k in ('case_index','reference','seed','y','raw_weight','objective_weight','proposed_method')}
    row['reused_cache']=task['proposed_reused_cache'] is not None
    try:
        if row['reused_cache']:
            with physical_forbidden():
                law=load_reused_law(task,points['original'])
            row['cache_path']=task['proposed_reused_cache']['particle_path']
        else:
            kernel=_kernels[(task['seed'],task['reference']['time_sec'])]
            if task['proposed_method']=='exact':
                with exact_budget(task,counts,directory/'progress.json'):
                    law=exact.build_cache(kernel,task['reference']['packet_id'],max_stochastic=10)
                if (counts['physical_calls']!=law.stats['physical_calls']
                        or counts['leaf_visits']!=law.stats['leaf_visits']
                        or counts['canonical_feature_evaluations']!=len(law.data['goal_configurations'])*task['required_actor_count']):
                    raise ArithmeticError('Exact construction disagrees with its consumed counters')
                filename='exact_cache_server_only.json.gz'
            else:
                sampler=MixtureSampler(kernel,task['reference']['packet_id'],points['original'],max_stochastic=10,
                    physical_limit=task['physical_limit'],feature_limit=task['feature_limit'],
                    progress=lambda stats:save(directory/'progress.json',stats))
                with primary_response_budget(sampler,256) as accounting:
                    pool=sampler.draw_mixture(256,task['seed_zero'],task['seed_fitted'])
                counts.update(physical_calls=sampler.stats['physical_calls'],
                              canonical_feature_evaluations=sampler.stats['canonical_feature_evaluations'])
                if accounting['resolver_constructions']!=counts['physical_calls']:
                    raise ArithmeticError('Mixed-pool and resolver counters disagree')
                law=MixtureParticleLaw(pool.data,points['original'])
                filename='mixture_cache_server_only.json.gz'
            if law.data['stochastic_participants']!=task['required_participants']:
                raise ValueError('A new cache changed its complete registered actor set')
            row['cache_path']=str(directory/filename)
            with gzip.open(directory/filename,'wt',compresslevel=1) as stream:
                json.dump(law.data,stream,allow_nan=False)
        with physical_forbidden():
            scalar,gradients=read_case(law,points,task)
        save(directory/'readouts_server_only.json',scalar)
        save(directory/'gradients_server_only.json',gradients)
        row['native_support_passed']=scalar['native_support_passed']
        row['status']='COMPLETE_STRATIFIED_CACHE_CASE' if row['native_support_passed'] else 'HOLD_STRATIFIED_NATIVE_SUPPORT'
    except Exception as error:
        row.update(status='HOLD_STRATIFIED_CACHE_CASE',error=f'{type(error).__name__}: {error}')
        (directory/'error_server_only.txt').write_text(traceback.format_exc())
    if sampler is not None:
        counts.update(physical_calls=sampler.stats['physical_calls'],
                      canonical_feature_evaluations=sampler.stats['canonical_feature_evaluations'])
    row.update(**counts,elapsed_sec=time.monotonic()-began)
    save(directory/'summary.json',row)
    print(json.dumps({k:row[k] for k in ('status','case_index','proposed_method','reused_cache',
        'physical_calls','canonical_feature_evaluations','leaf_visits','elapsed_sec')}),flush=True)
    return row


def run(base,output,workers):
    output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    completed,tasks,registration=[],[],{}
    points={}
    error,stopped=None,False
    save(output/'started.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        limits=LIMITS,new_native_starts=0,new_fits=0,optimizer_invocations=0))
    try:
        proof=read_json(base/'target_packet_mixture_confirmation/synthetic_result.json')
        if proof['status']!='PASS_SYNTHETIC_MIXTURE_GATE' or proof['full_forward_calls']!=854:
            raise ValueError('The existing paid exact/mixture equivalence gate is required')
        points=frozen_points(base)
        save(output/'points_server_only.json',{name:theta.tolist() for name,theta in points.items()})
        tasks,kernels,registration=prepare(base,output)
        print(json.dumps(dict(stage='stratified_cache_prepared',**registration)),flush=True)
        # Existing caches are read first. Each case is submitted once; after a
        # failure, already running cases finish and retain their consumed costs.
        ordered=iter(sorted(tasks,key=lambda task:(not bool(task['proposed_reused_cache']),
            -task['physical_limit']*task['frame_actor_count'],task['case_index'])))
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'),
                                 initializer=set_worker_kernels,initargs=(kernels,)) as pool:
            pending={pool.submit(inspect_case,next(ordered),points,output) for _ in range(workers)}
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    row=future.result()
                    completed.append(row)
                    stopped |= row['status']!='COMPLETE_STRATIFIED_CACHE_CASE'
                if not stopped:
                    for _ in range(workers-len(pending)):
                        task=next(ordered,None)
                        if task is not None:
                            pending.add(pool.submit(inspect_case,task,points,output))
    except Exception as failure:
        error=f'{type(failure).__name__}: {failure}'
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    completed,unknown=collect_receipts(output,completed)
    physical=sum(r['physical_calls'] for r in completed)+sum(r['physical_calls_lower_bound'] for r in unknown)
    features=sum(r['canonical_feature_evaluations'] for r in completed)+sum(r['canonical_feature_evaluations_lower_bound'] for r in unknown)
    leaves=sum(r['leaf_visits'] for r in completed)
    if physical>LIMITS['physical_calls'] or features>LIMITS['canonical_feature_evaluations'] or leaves>LIMITS['leaf_visits']:
        error='Frozen cache construction budget exceeded'
    complete=(error is None and not unknown and len(completed)==1000
              and all(r['status']=='COMPLETE_STRATIFIED_CACHE_CASE' for r in completed))
    aggregate=None
    if complete:
        try:
            records,gradients=[],{}
            with physical_forbidden():
                for task in tasks:
                    records.append(read_json(output/task['stem']/'readouts_server_only.json'))
                    gradients[task['case_index']]=read_json(output/task['stem']/'gradients_server_only.json')
                aggregate=aggregate_records(records,gradients,points,expected_references=1000)
                if not aggregate['native_support_passed'] or not all(
                    value[view]['all_references_supported']
                    and value[view]['weighted_nll_contribution']['state']=='finite'
                    and value[view]['weighted_gradient']['defined']
                    for value in aggregate['native']['points'].values() for view in ('primary','128')):
                    raise ArithmeticError('The complete four-point native support gate failed')
        except Exception as failure:
            error=f'{type(failure).__name__}: {failure}'
            (output/'error_server_only.txt').write_text(traceback.format_exc())
            complete=False
    result=dict(status='COMPLETE_STRATIFIED_FOUR_POINT_CACHE' if complete else 'HOLD_STRATIFIED_CACHE',
        error=error,registration=registration,aggregate=aggregate,completed_case_attempts=len(completed),
        all_four_points_supported=bool(complete),failed_cases=[r for r in completed if r['status']!='COMPLETE_STRATIFIED_CACHE_CASE'],
        incomplete_started_cases=unknown,stopped_after_failure=stopped or error is not None,
        undispatched_case_indices=[t['case_index'] for t in tasks if t['case_index'] not in {r['case_index'] for r in completed+unknown}],
        consumed_counters_exact=not unknown,full_forward_calls_lower_bound=physical,
        canonical_feature_evaluations_lower_bound=features,exact_leaf_visits_lower_bound=leaves,
        new_full_forward_calls=physical if not unknown else None,
        new_canonical_feature_evaluations=features if not unknown else None,exact_leaf_visits=leaves if not unknown else None,
        reused_case_count=sum(r['reused_cache'] for r in completed),new_native_starts=0,new_nuisance_estimations=0,
        new_fits=0,optimizer_invocations=0,new_synthetic_full_forward_calls=0,calibration_seeds_used=False,
        original_fit_hold_preserved=True,model_admitted=False,
        elapsed_sec=time.monotonic()-began,completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'result.json',result)
    execution_file=output.parent/'execution.json'
    if execution_file.is_file():
        execution=read_json(execution_file)
        execution.update({name:result[name] for name in ('status','completed_at_utc','elapsed_sec',
            'completed_case_attempts','new_full_forward_calls','new_canonical_feature_evaluations',
            'exact_leaf_visits','full_forward_calls_lower_bound','canonical_feature_evaluations_lower_bound',
            'exact_leaf_visits_lower_bound','reused_case_count','all_four_points_supported',
            'new_native_starts','new_fits','optimizer_invocations')})
        execution['counters_finalized']=result['consumed_counters_exact']
        save(execution_file,execution)
    print(json.dumps({k:result[k] for k in ('status','error','completed_case_attempts','new_full_forward_calls',
        'new_canonical_feature_evaluations','exact_leaf_visits','elapsed_sec')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3,choices=(1,2,3))
    args=parser.parse_args()
    raise SystemExit(0 if run(args.base,args.output,args.workers)['status']=='COMPLETE_STRATIFIED_FOUR_POINT_CACHE' else 2)
