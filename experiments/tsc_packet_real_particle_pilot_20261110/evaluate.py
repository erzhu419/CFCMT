"""One frozen fifteen-state real particle pilot, with no fitting or rollout."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import json
from math import isfinite, log
import multiprocessing
from pathlib import Path
import resource
import time
import traceback
from unittest.mock import patch

import numpy as np

from experiments.tsc_packet_conditional_domain_20261110.register import RESERVOIR_SEED
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_conditional_panel_20261110.initialize import prepare_zero_kernel
from experiments.tsc_packet_conditional_particles_20261110.verify import (
    all_queries, consistency, physical_forbidden, read,
)
from experiments.tsc_packet_conditional_cache_20261110.verify import parameter_probes
from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from .register import reference_key, save
from .sampler import make_sampler


PARTICLE_SIZES = (256, 1024)


def case_seed(index):
    return 20261111+100000*index


def counters_from_sampler(sampler):
    stats = {} if sampler is None else dict(sampler.stats)
    return dict(physical_calls=int(stats.get('physical_calls', 0)),
        canonical_feature_evaluations=int(stats.get('canonical_feature_evaluations', 0)),
        sampler_stats=stats)


def load_labels(manifest_path, cases, exact_file):
    """Called only after the labels-free cohort and synthetic gate passed."""
    wanted = {reference_key(row['reference']) for row in cases}
    result = {key: dict(observations=[]) for key in wanted}
    reservoirs = json.loads(Path(manifest_path).read_text())
    for seed, heads in reservoirs.items():
        for head, refs in heads.items():
            for ordinal, row in enumerate(refs):
                key = reference_key(row)
                if key in wanted:
                    y = row['y']
                    if not isinstance(y, (int, float)) or not isfinite(y):
                        raise ValueError('Available native head labels must be finite numbers')
                    result[key]['observations'].append(dict(head=head, y=y, manifest_ordinal=ordinal))
    exact = {reference_key(row['reference']): row for row in json.loads(Path(exact_file).read_text())['cases']}
    for case in cases:
        key = reference_key(case['reference'])
        if not result[key]['observations']:
            raise ValueError('Every frozen current case must retain its original native head reference')
        if case['case_index'] < 9:
            if key not in exact:
                raise ValueError('Each original state needs its saved exact probability readout')
            old = exact[key]
            result[key]['exact'] = old
            result[key]['native_speed_mps'] = old['native_speed_mps']
            result[key]['native_lc_label'] = old['observed_lc_label']
            if not isfinite(old['native_speed_mps']):
                raise ValueError('The original speed label must remain finite')
            if any(int(obs['y']) != old['observed_lc_label'] for obs in result[key]['observations']
                   if obs['head'] == 'lane_change'):
                raise ValueError('Manifest and saved exact native LC labels differ')
    return result


@contextmanager
def primary_response_budget(sampler, limit):
    """A constructor is allowed only after the inherited primary counter rises.

    Feature/context work precedes that rise, so an unexpected feature resolver
    is rejected before construction, rather than silently charging its cost0.
    """
    counts = dict(resolver_constructions=0, forbidden_feature_or_extra_resolver_attempts=0)
    original = MotionResolver.__init__
    def initialize(resolver, *args, **kwargs):
        if (int(sampler.stats['physical_calls']) != counts['resolver_constructions']+1
                or counts['resolver_constructions'] >= limit):
            counts['forbidden_feature_or_extra_resolver_attempts'] += 1
            raise RuntimeError('Feature work or an extra path attempted an unregistered full-forward resolver')
        counts['resolver_constructions'] += 1
        return original(resolver, *args, **kwargs)
    with patch.object(MotionResolver, '__init__', initialize):
        yield counts


def compact(row):
    gradient = row.get('gradient')
    return {**{key: row.get(key) for key in ('defined', 'supported', 'likelihood',
        'survivor_probability', 'global_ess', 'survivor_ess', 'posterior_ess', 'max_normalized_weight')},
        'log_likelihood': row['log_likelihood'] if row.get('log_likelihood') is not None and isfinite(row['log_likelihood']) else None,
        'nll': row['nll'] if row.get('nll') is not None and isfinite(row['nll']) else None,
        'nll_infinite': bool(row.get('nll') is not None and not isfinite(row['nll'])),
        'likelihood_state': ('NO_SURVIVOR_SAMPLES' if not row['defined'] else
                            'POSITIVE' if row.get('supported', False) else 'ZERO_NUMERATOR'),
        'gradient_l2': float(np.linalg.norm(gradient)) if gradient is not None else None,
        'gradient_finite': bool(np.isfinite(gradient).all()) if gradient is not None else None}


def probability_error(approximate, exact):
    if approximate is None:
        return dict(exact=exact, approximate=None, absolute_error=None, missed_known_positive=exact > 0.,
                    structural_zero_false_positive=False, nll_error=None, nll_error_infinite=exact > 0.)
    positive = exact > 0.
    return dict(exact=exact, approximate=approximate, absolute_error=abs(approximate-exact),
        missed_known_positive=positive and approximate <= 0.,
        structural_zero_false_positive=exact == 0. and approximate > 0.,
        nll_error=-log(approximate)+log(exact) if positive and approximate > 0. else None,
        nll_error_infinite=positive and approximate <= 0.)


def original_comparison(pool, theta, n, label):
    old, speed = label['exact'], label['native_speed_mps']
    lc = [pool.marginal(theta, 'lane_change', y, n=n)['likelihood'] for y in range(3)]
    motion = pool.marginal(theta, 'motion', speed, n=n)['likelihood']
    joint = [pool.class_speed(theta, y, speed, n=n)['likelihood'] for y in range(3)]
    return dict(lane_change=[probability_error(a,b) for a,b in zip(lc,old['conditional_lc_probabilities'])],
        observed_speed=probability_error(motion,old['observed_speed_likelihood']['likelihood']),
        class_speed=[probability_error(a,b) for a,b in zip(joint,old['conditional_class_speed_likelihoods'])])


def inspect_case(task):
    cache, preflight, width_file, registered, label, output = task
    ref, index = registered['reference'], registered['case_index']
    seed, t, focal = reference_key(ref)
    directory = output/f'case{index:02d}_seed{seed}_t{t}'
    directory.mkdir(exist_ok=False)
    began = time.monotonic()
    sampler, accounting = None, dict(resolver_constructions=0, forbidden_feature_or_extra_resolver_attempts=0)
    readouts, comparisons = {}, {}
    save(directory/'started.json', dict(reference=ref, case_index=index, particle_seed=case_seed(index),
        started_at_utc=datetime.now(timezone.utc).isoformat(), physical_limit=1024,
        canonical_feature_limit=registered['canonical_feature_ceiling']))
    try:
        current = json.loads((preflight/registered['current_cache']).read_text())
        static, unused = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        executors = json.loads((cache/'training'/f'seed{seed}'/'executor_seed_server_only.json').read_text())
        width, unused = verify_entity_preflight(width_file)
        geometry = build_entity_geometry(RoadTopology(static), width)
        kernel = prepare_zero_kernel(static,current['frame'],executors,laws,current['action'],
                                    RESERVOIR_SEED+seed*100000+t,geometry)
        selected = select_stochastic(kernel,focal,max_stochastic=10)
        saved = json.loads((preflight/registered['selection_cache']).read_text())
        if json.loads(json.dumps(selected)) != saved['selection']:
            raise ValueError('Response initializer changed the registered complete geometry')
        if any(kernel.state['packets'][key]['_movement_key'] != item['movement_key']
               for key,item in saved['internal_bindings'].items()):
            raise ValueError('Response initializer changed a current committed movement')
        def progress(counts):
            save(directory/'progress.json',counts)
            print(json.dumps(dict(stage='real_particle_progress',case_index=index,reference=ref,**counts)),flush=True)
        sampler = make_sampler(kernel,focal,max_stochastic=10,progress=progress,physical_limit=1024,
                               feature_limit=registered['canonical_feature_ceiling'])
        for obs in label['observations']:
            if not pool_eligibility(kernel,focal)[obs['head']]:
                raise ValueError('An original head reference is outside its registered current selector')
        with primary_response_budget(sampler,1024) as accounting:
            pool = sampler.draw(1024,case_seed(index))
        with gzip.open(directory/'particles_server_only.json.gz','wt',compresslevel=1) as stream:
            json.dump(pool.to_dict(),stream,allow_nan=False)
        max_error, all_supported = 0., True
        eligibility = pool.data['eligibility']
        with physical_forbidden():
            for n in PARTICLE_SIZES:
                point_rows = {}
                for point,theta in enumerate(parameter_probes()):
                    error, summary = consistency(pool,theta,eligibility,n)
                    max_error = max(max_error,error)
                    observed = []
                    for obs in label['observations']:
                        value = pool.marginal(theta,obs['head'],obs['y'],n=n)
                        observed.append(dict(obs,**compact(value)))
                        all_supported &= bool(value['defined'] and value.get('supported',False))
                    diagnostics = {':'.join(map(str,query)):compact(read(pool,theta,query,n))
                                   for query in all_queries(eligibility)}
                    point_rows[str(point)] = dict(summary={k:v for k,v in summary.items() if k!='weighted_score'},
                        observed_heads=observed,diagnostics=diagnostics)
                    if index < 9:
                        speed,lc_y = label['native_speed_mps'],label['native_lc_label']
                        point_rows[str(point)]['saved_native_speed'] = compact(pool.marginal(theta,'motion',speed,n=n))
                        point_rows[str(point)]['saved_native_class_speed'] = compact(pool.class_speed(theta,lc_y,speed,n=n))
                        if point == 0:
                            comparisons[str(n)] = original_comparison(pool,theta,n,label)
                readouts[str(n)] = point_rows
        if accounting['resolver_constructions'] != sampler.stats['physical_calls']:
            raise ArithmeticError('Primary resolver and registered particle counts differ')
        missed = any(item['missed_known_positive'] for comp in comparisons.values()
                     for item in comp['lane_change']+[comp['observed_speed']]+comp['class_speed'])
        false_positive = any(item['structural_zero_false_positive'] for comp in comparisons.values()
                             for item in comp['lane_change']+[comp['observed_speed']]+comp['class_speed'])
        if false_positive:
            raise ArithmeticError('The approximation assigned mass outside a saved exact structural support')
        row = dict(status='PASS_REAL_PARTICLE_CASE' if all_supported and not missed else 'DEFINED_PARTICLE_SUPPORT_FAILURE',
            reference=ref,case_index=index,origin=registered['origin'],particle_seed=case_seed(index),
            stochastic_participants=list(sampler.participants),stochastic_actor_count=len(sampler.participants),
            frame_actor_count=len(kernel.state['packets']),**counters_from_sampler(sampler),
            resolver_constructions=accounting['resolver_constructions'],feature_stage_resolver_calls=0,
            forbidden_feature_or_extra_resolver_attempts=accounting['forbidden_feature_or_extra_resolver_attempts'],
            maximum_consistency_error=max_error,normalized=True,all_observed_heads_supported=all_supported,
            observed_head_support_by_particle_size={n:all(item['defined'] and item['supported']
                for point in point_rows.values() for item in point['observed_heads'])
                for n,point_rows in readouts.items()},
            missed_saved_exact_positive_by_particle_size={n:any(item['missed_known_positive']
                for item in comp['lane_change']+[comp['observed_speed']]+comp['class_speed'])
                for n,comp in comparisons.items()},
            missed_saved_exact_positive=missed,readouts=readouts,saved_exact_comparisons=comparisons,
            reuse_physical_calls=0,stop_new_cases=False,elapsed_sec=time.monotonic()-began)
    except Exception as error:
        (directory/'error_server_only.txt').write_text(traceback.format_exc())
        row = dict(status='HOLD_REAL_PARTICLE_CASE_UNDEFINED',reference=ref,case_index=index,
            origin=registered['origin'],particle_seed=case_seed(index),error=f'{type(error).__name__}: {error}',
            **counters_from_sampler(sampler),resolver_constructions=accounting['resolver_constructions'],
            feature_stage_resolver_calls=0,
            forbidden_feature_or_extra_resolver_attempts=accounting['forbidden_feature_or_extra_resolver_attempts'],
            current_goal_tuple=getattr(error,'current_goal_tuple',None),normalized=False,
            readouts=readouts,saved_exact_comparisons=comparisons,
            all_observed_heads_supported=False,stop_new_cases=True,elapsed_sec=time.monotonic()-began)
    row['worker_process_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    save(directory/'summary.json',row)
    print(json.dumps(dict(stage='real_particle_case_complete',case_index=index,status=row['status'],
        physical_calls=row['physical_calls'],canonical_feature_evaluations=row['canonical_feature_evaluations'],
        elapsed_sec=row['elapsed_sec'])),flush=True)
    return row


def pool_eligibility(kernel,focal):
    from experiments.tsc_packet_joint_law_definition_20261110.law import _eligibility
    return _eligibility(kernel,kernel.state['packets'][focal])


def collect_receipts(output, completed):
    """Account for a worker that wrote a receipt but could not return it."""
    rows = {row['case_index']: row for row in completed}
    unknown = []
    for directory in sorted(output.glob('case*_seed*_t*')):
        summary_file = directory/'summary.json'
        if summary_file.exists():
            try:
                row = json.loads(summary_file.read_text())
                rows[row['case_index']] = row
                continue
            except (ValueError, KeyError):
                pass
        started_file = directory/'started.json'
        if not started_file.exists():
            continue
        try:
            started = json.loads(started_file.read_text())
        except ValueError:
            # started.json is completed before kernel construction or responses.
            continue
        if started['case_index'] in rows:
            continue
        progress = {}
        if (directory/'progress.json').exists():
            try:
                progress = json.loads((directory/'progress.json').read_text())
            except ValueError:
                pass
        unknown.append(dict(reference=started['reference'],case_index=started['case_index'],
            physical_calls_lower_bound=progress.get('physical_calls',0),
            canonical_feature_evaluations_lower_bound=progress.get('canonical_feature_evaluations',0),
            consumed_counters_exact=False,reason='Started worker lacks a complete final receipt'))
    return sorted(rows.values(),key=lambda row:row['case_index']),unknown


def run(cache,preflight,synthetic_result,exact_file,width_file,output,workers=4):
    output.mkdir(parents=True,exist_ok=False)
    began,completed = time.monotonic(),[]
    registered,stopped = [],False
    try:
        proof = json.loads(synthetic_result.read_text())
        frozen = json.loads((preflight/'summary.json').read_text())
        if proof['status'] != 'PASS_LIGHT_SAMPLER_EQUIVALENCE':
            raise ValueError('Numeric-only preparation equivalence must pass before real responses')
        if (frozen['status'] != 'PASS_FROZEN_REAL_PARTICLE_COHORT' or len(frozen['cases'])!=15
                or frozen['particle_full_forward_ceiling']!=15360 or frozen['canonical_feature_ceiling']!=68608):
            raise ValueError('The complete frozen15-case budget must pass before responses')
        registered = frozen['cases']
        labels = load_labels(cache/'training_reservoir_server_only.json',registered,exact_file)
        save(output/'started.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
            requested_states=15,particle_full_forward_ceiling=15360,canonical_feature_ceiling=68608,
            dispatch_order='Descending required actor count; fixed original case indices and seeds unchanged.'))
        ordered = sorted(registered,key=lambda row:(-row['required_actor_count'],row['case_index']))
        tasks = iter((cache,preflight,width_file,row,labels[reference_key(row['reference'])],output) for row in ordered)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {}
            for unused in range(workers):
                task = next(tasks,None)
                if task is not None:
                    pending[pool.submit(inspect_case,task)] = task[3]['case_index']
            while pending:
                done,unused = wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    completed.append(row)
                    del pending[future]
                    stopped |= row['stop_new_cases']
                if not stopped:
                    for unused in range(workers-len(pending)):
                        task = next(tasks,None)
                        if task is None:
                            break
                        pending[pool.submit(inspect_case,task)] = task[3]['case_index']
        physical = sum(row['physical_calls'] for row in completed)
        features = sum(row['canonical_feature_evaluations'] for row in completed)
        if physical>15360 or features>68608:
            raise ArithmeticError('The pilot exceeded its frozen computation budget')
        passed = len(completed)==15 and all(row['status']=='PASS_REAL_PARTICLE_CASE' for row in completed)
        result = dict(status='PASS_BOUNDED_REAL_PARTICLE_PILOT' if passed else 'HOLD_BOUNDED_REAL_PARTICLE_PILOT',
            cases=sorted(completed,key=lambda row:row['case_index']),requested_states=15,
            completed_case_attempts=len(completed),
            not_completed_references=[row['reference'] for row in registered if row['case_index'] not in {x['case_index'] for x in completed}],
            actual_full_forward_calls=physical,actual_canonical_feature_evaluations=features,
            feature_stage_resolver_calls=0,reuse_physical_calls=0,stopped_after_failure=stopped,
            consumed_counters_exact=True,real_additional_exact_reference_full_forward_calls=0,
            model_admitted=False,scope='Selected15 training states: runtime/support and saved-exact theta0 approximation only; no full-target, source-transfer, prediction or control admission.')
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        completed,unknown = collect_receipts(output,completed)
        physical_lower = sum(row['physical_calls'] for row in completed)+sum(row['physical_calls_lower_bound'] for row in unknown)
        feature_lower = sum(row['canonical_feature_evaluations'] for row in completed)+sum(row['canonical_feature_evaluations_lower_bound'] for row in unknown)
        result = dict(status='HOLD_REAL_PARTICLE_PILOT_TECHNICAL',error=f'{type(error).__name__}: {error}',
            cases=completed,incomplete_started_cases=unknown,consumed_counters_exact=not unknown,
            actual_full_forward_calls=physical_lower if not unknown else None,
            actual_canonical_feature_evaluations=feature_lower if not unknown else None,
            full_forward_calls_lower_bound=physical_lower,
            canonical_feature_evaluations_lower_bound=feature_lower,model_admitted=False)
    result.update(new_native_starts=0,new_fits=0,new_nuisance_estimations=0,optimizer_calls=0,
        free_rollouts=0,calibration_data_used=False,elapsed_sec=time.monotonic()-began,
        memory_metric='Linux worker-process ru_maxrss; includes any earlier case in the same worker.')
    save(output/'summary.json',result)
    print(json.dumps({key:result[key] for key in ('status','elapsed_sec')}),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ('cache','preflight','synthetic-result','exact-file','width-file','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2,3,4),default=4)
    args=parser.parse_args()
    result=run(args.cache,args.preflight,args.synthetic_result,args.exact_file,args.width_file,args.output,args.workers)
    raise SystemExit(0 if result['status']=='PASS_BOUNDED_REAL_PARTICLE_PILOT' else 2)
