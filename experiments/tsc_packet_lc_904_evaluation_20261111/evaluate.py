"""Bounded904 cache construction and three frozen-model readouts."""

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
from experiments.tsc_packet_conditional_cache_20261110.cache import _vectors
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_packet_lane_change_holdout_20261111.evaluate import exact_budget
from experiments.tsc_packet_lc_negative_cache_20261111.evaluate import load_reused_law
from experiments.tsc_packet_lc_refit_20261111.fit import frequency_reference
from experiments.tsc_packet_mixture_confirmation_20261111.evaluate import frozen_candidate, read_json
from experiments.tsc_packet_mixture_confirmation_20261111.readout import MixtureParticleLaw
from experiments.tsc_packet_mixture_confirmation_20261111.sampler import MixtureSampler
from experiments.tsc_packet_real_particle_pilot_20261110.evaluate import collect_receipts, primary_response_budget
from experiments.tsc_packet_real_particle_pilot_20261110.register import save
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES

from .prepare import prepare
from .readout import aggregate_records, read_case


LIMITS=dict(physical_calls=11820,canonical_feature_evaluations=24206,leaf_visits=35280)
_kernels=None


def set_worker_kernels(kernels):
    global _kernels
    _kernels=kernels


def frozen_points(base):
    manifest=read_json(base/'target_packet_lc_904_registration/registration/frozen_models_server_only.json')['models']
    anchor=frozen_candidate(base)
    points={}
    for name,folder in (('old_lc','target_packet_lc_refit'),('new_lc','target_packet_lc_stratified_refit')):
        path=base/folder/'fit/coefficients_server_only.json'
        if Path(manifest[name]['coefficient_file'])!=path:
            raise ValueError('A registered frozen coefficient source changed')
        saved=read_json(path)
        theta=_vectors(saved['coefficients'])
        if (saved['parameter_names']!=list(PARAMETERS) or saved['feature_names']!=list(FEATURE_NAMES)
                or not np.array_equal(theta[:3],anchor[:3])):
            raise ValueError('A fixed model changed its layout or non-LC coefficient rows')
        points[name]=theta
    reference=manifest['frequency_bias']
    if (reference['training_class_counts']!={'left':165,'right':166,'stay':29669}
            or Path(reference['non_lc_rows_source'])!=base/'target_packet_particle_first_fit/fit/coefficients_server_only.json'):
        raise ValueError('The training-frequency reference changed using904 information')
    points['frequency_bias']=frequency_reference(anchor)
    return anchor,points


def inspect_case(task,points,anchor,output):
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
                law=load_reused_law(task,anchor)
            row['cache_path']=task['proposed_reused_cache']['particle_path']
        else:
            kernel=_kernels[(task['seed'],task['reference']['time_sec'])]
            if task['proposed_method']=='exact':
                with exact_budget(task,counts,directory/'progress.json'):
                    law=exact.build_cache(kernel,task['reference']['packet_id'],max_stochastic=10)
                if (counts['physical_calls']!=law.stats['physical_calls']
                        or counts['leaf_visits']!=law.stats['leaf_visits']
                        or counts['canonical_feature_evaluations']!=len(law.data['goal_configurations'])*task['required_actor_count']):
                    raise ArithmeticError('Exact construction and its charged counters disagree')
                filename='exact_cache_server_only.json.gz'
            else:
                sampler=MixtureSampler(kernel,task['reference']['packet_id'],anchor,max_stochastic=10,
                    physical_limit=task['physical_limit'],feature_limit=task['feature_limit'],
                    progress=lambda stats:save(directory/'progress.json',stats))
                with primary_response_budget(sampler,256) as accounting:
                    pool=sampler.draw_mixture(256,task['seed_zero'],task['seed_fitted'])
                counts.update(physical_calls=sampler.stats['physical_calls'],
                              canonical_feature_evaluations=sampler.stats['canonical_feature_evaluations'])
                if accounting['resolver_constructions']!=counts['physical_calls']:
                    raise ArithmeticError('Mixed-pool and resolver counters disagree')
                law=MixtureParticleLaw(pool.data,anchor)
                filename='mixture_cache_server_only.json.gz'
            if law.data['stochastic_participants']!=task['required_participants']:
                raise ValueError('The new cache changed its full registered actor set')
            row['cache_path']=str(directory/filename)
            with gzip.open(directory/filename,'wt',compresslevel=1) as stream:
                json.dump(law.data,stream,allow_nan=False)
        with physical_forbidden():
            scalar,gradients=read_case(law,points,task)
        save(directory/'readouts_server_only.json',scalar)
        save(directory/'gradients_server_only.json',gradients)
        row.update(native_support_passed=scalar['native_support_passed'],
                   probability_normalization_passed=scalar['probability_normalization_passed'])
        row['status']=('COMPLETE_904_CACHE_CASE' if row['native_support_passed']
                       and row['probability_normalization_passed'] else 'HOLD_904_NATIVE_OR_PROBABILITY')
    except Exception as error:
        row.update(status='HOLD_904_CACHE_CASE',error=f'{type(error).__name__}: {error}')
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
    tasks,completed,registration=[],[],{}
    points={}
    error,stopped=None,False
    save(output/'started.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),limits=LIMITS,
        new_native_starts=0,new_fits=0,optimizer_invocations=0))
    try:
        proof=read_json(base/'target_packet_mixture_confirmation/synthetic_result.json')
        if proof['status']!='PASS_SYNTHETIC_MIXTURE_GATE' or proof['full_forward_calls']!=854:
            raise ValueError('The existing exact/mixture equivalence gate is required')
        anchor,points=frozen_points(base)
        save(output/'points_server_only.json',{name:theta.tolist() for name,theta in points.items()})
        tasks,kernels,registration=prepare(base,output)
        print(json.dumps(dict(stage='904_evaluation_prepared',**registration)),flush=True)
        ordered=iter(sorted(tasks,key=lambda task:(not bool(task['proposed_reused_cache']),
            -task['physical_limit']*task['frame_actor_count'],task['case_index'])))
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'),
                                 initializer=set_worker_kernels,initargs=(kernels,)) as pool:
            pending={pool.submit(inspect_case,next(ordered),points,anchor,output) for _ in range(workers)}
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    row=future.result()
                    completed.append(row)
                    stopped |= row['status']!='COMPLETE_904_CACHE_CASE'
                if not stopped:
                    for _ in range(workers-len(pending)):
                        task=next(ordered,None)
                        if task is not None:
                            pending.add(pool.submit(inspect_case,task,points,anchor,output))
    except Exception as failure:
        error=f'{type(failure).__name__}: {failure}'
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    completed,unknown=collect_receipts(output,completed)
    physical=sum(r['physical_calls'] for r in completed)+sum(r['physical_calls_lower_bound'] for r in unknown)
    features=sum(r['canonical_feature_evaluations'] for r in completed)+sum(r['canonical_feature_evaluations_lower_bound'] for r in unknown)
    leaves=sum(r['leaf_visits'] for r in completed)
    if physical>LIMITS['physical_calls'] or features>LIMITS['canonical_feature_evaluations'] or leaves>LIMITS['leaf_visits']:
        error='Frozen904 cache budget exceeded'
    complete=(error is None and not unknown and len(completed)==340
              and all(r['status']=='COMPLETE_904_CACHE_CASE' for r in completed))
    aggregate=None
    if complete:
        try:
            records,gradients=[],{}
            with physical_forbidden():
                for task in tasks:
                    records.append(read_json(output/task['stem']/'readouts_server_only.json'))
                    gradients[task['case_index']]=read_json(output/task['stem']/'gradients_server_only.json')
                aggregate=aggregate_records(records,gradients,points,expected_references=340)
                if (not aggregate['all_native_support_passed'] or not aggregate['all_probability_normalization_passed']
                        or not all(v[view]['all_references_supported'] and v[view]['weighted_gradient']['defined']
                            and v[view]['weighted_nll_contribution']['state']=='finite'
                            for v in aggregate['native']['points'].values() for view in ('primary','128'))):
                    raise ArithmeticError('The complete904 native/probability gate failed')
        except Exception as failure:
            error=f'{type(failure).__name__}: {failure}'
            (output/'error_server_only.txt').write_text(traceback.format_exc())
            complete=False
    result=dict(status='COMPLETE_904_NATURAL_THREE_POINT_EVALUATION' if complete else 'HOLD_904_NATURAL_EVALUATION',
        error=error,registration=registration,aggregate=aggregate,completed_case_attempts=len(completed),
        all_three_points_supported=bool(complete),failed_cases=[r for r in completed if r['status']!='COMPLETE_904_CACHE_CASE'],
        incomplete_started_cases=unknown,stopped_after_failure=stopped or error is not None,
        undispatched_case_indices=[t['case_index'] for t in tasks if t['case_index'] not in {r['case_index'] for r in completed+unknown}],
        consumed_counters_exact=not unknown,full_forward_calls_lower_bound=physical,
        canonical_feature_evaluations_lower_bound=features,exact_leaf_visits_lower_bound=leaves,
        new_full_forward_calls=physical if not unknown else None,new_canonical_feature_evaluations=features if not unknown else None,
        exact_leaf_visits=leaves if not unknown else None,reused_case_count=sum(r['reused_cache'] for r in completed),
        new_native_starts=0,new_fits=0,optimizer_invocations=0,new_nuisance_estimations=0,new_synthetic_full_forward_calls=0,
        exposure='Previously exposed904; internal natural-frequency regression, not independent confirmation.',
        old62_pair_regression_recomputed=False,original_fit_hold_preserved=True,model_admitted=False,
        elapsed_sec=time.monotonic()-began,completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'result.json',result)
    execution_file=output.parent/'execution.json'
    if execution_file.exists():
        e=read_json(execution_file)
        e.update({k:result[k] for k in ('status','completed_at_utc','elapsed_sec','completed_case_attempts',
            'new_full_forward_calls','new_canonical_feature_evaluations','exact_leaf_visits','full_forward_calls_lower_bound',
            'canonical_feature_evaluations_lower_bound','reused_case_count','all_three_points_supported')})
        e['counters_finalized']=result['consumed_counters_exact']
        save(execution_file,e)
    print(json.dumps({k:result[k] for k in ('status','error','completed_case_attempts','new_full_forward_calls',
        'new_canonical_feature_evaluations','exact_leaf_visits','elapsed_sec')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    args=parser.parse_args()
    raise SystemExit(0 if run(args.base,args.output,args.workers)['status']=='COMPLETE_904_NATURAL_THREE_POINT_EVALUATION' else 2)
