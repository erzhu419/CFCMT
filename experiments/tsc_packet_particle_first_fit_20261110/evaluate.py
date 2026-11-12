"""Build fixed256 pools once, then stop unless every selected label is supported."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import gzip
import json
from math import isfinite
import multiprocessing
from pathlib import Path
import resource
import time
import traceback

import numpy as np

from experiments.tsc_packet_conditional_domain_20261110.register import RESERVOIR_SEED
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_conditional_panel_20261110.initialize import prepare_zero_kernel
from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from experiments.tsc_packet_real_particle_pilot_20261110.evaluate import (
    collect_receipts, compact, counters_from_sampler, pool_eligibility, primary_response_budget,
)
from experiments.tsc_packet_real_particle_pilot_20261110.sampler import make_sampler
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from .register import PARTICLES, reference_key, save


def load_labels(manifest_path, cases):
    """Recover only the preselected head ordinals after complete current freeze."""
    manifest = json.loads(Path(manifest_path).read_text())
    result = {}
    for case in cases:
        observations = []
        for selected in case['references']:
            old = manifest[str(selected['seed'])][selected['head']][selected['manifest_ordinal']]
            if reference_key(old) != reference_key(selected):
                raise ValueError('A selected manifest ordinal changed its current reference')
            y = old['y']
            if not isinstance(y, (int,float)) or not isfinite(y):
                raise ValueError('Every selected native label must be finite')
            if selected['head'] == 'motion' and y < 0:
                raise ValueError('A selected native speed must be nonnegative')
            if selected['head'] == 'lane_change' and y not in (0,1,2):
                raise ValueError('A selected LC label must be left0/stay1/right2')
            if selected['head'] in ('start','yield_entry') and y not in (0,1):
                raise ValueError('A selected event label must be Bernoulli')
            observations.append(dict(selected,y=y))
        result[reference_key(case['reference'])] = observations
    if sum(map(len,result.values())) != 512:
        raise ValueError('Exactly512 selected labels, with their original multiplicity, are required')
    return result


def observations_supported(rows):
    return bool(rows) and all(row.get('defined') and row.get('supported')
        and row.get('gradient_finite') and row.get('nll') is not None and isfinite(row['nll']) for row in rows)


def inspect_case(task):
    cache, preflight, width_file, registered, observations, output = task
    ref, index = registered['reference'], registered['case_index']
    seed,t,focal = reference_key(ref)
    directory = output/f'case{index:03d}_seed{seed}_t{t}'
    directory.mkdir(exist_ok=False)
    began, sampler = time.monotonic(), None
    accounting = dict(resolver_constructions=0, forbidden_feature_or_extra_resolver_attempts=0)
    observed, particle_path = [], None
    reused = registered['reuse_pilot_pool']
    save(directory/'started.json',dict(reference=ref,case_index=index,particle_seed=registered['particle_seed'],
        reused_pilot_pool=reused,started_at_utc=datetime.now(timezone.utc).isoformat(),
        physical_limit=registered['particle_full_forward_ceiling'],
        canonical_feature_limit=registered['canonical_feature_ceiling']))
    try:
        current = json.loads((preflight/registered['current_cache']).read_text())
        static,unused = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        executors = json.loads((cache/'training'/f'seed{seed}'/'executor_seed_server_only.json').read_text())
        width,unused = verify_entity_preflight(width_file)
        geometry = build_entity_geometry(RoadTopology(static),width)
        kernel = prepare_zero_kernel(static,current['frame'],executors,laws,current['action'],
                                    RESERVOIR_SEED+seed*100000+t,geometry)
        selected = select_stochastic(kernel,focal,max_stochastic=10)
        saved = json.loads((preflight/registered['selection_cache']).read_text())
        if json.loads(json.dumps(selected)) != saved['selection']:
            raise ValueError('The response initializer changed the complete registered selection')
        if any(kernel.state['packets'][key]['_movement_key'] != item['movement_key']
               for key,item in saved['internal_bindings'].items()):
            raise ValueError('The response initializer changed an internal committed movement')
        eligible = pool_eligibility(kernel,focal)
        if eligible != registered['eligibility'] or any(not eligible[obs['head']] for obs in observations):
            raise ValueError('An original selected head lost its current selector eligibility')
        if reused:
            with physical_forbidden():
                with gzip.open(registered['reuse_particle_path'],'rt') as stream:
                    payload = json.load(stream)
                if (len(payload['particles']) != 1024 or payload['focal_id'] != focal
                        or payload['stochastic_participants'] != selected['participants']
                        or payload['frame_actor_count'] != registered['frame_actor_count']
                        or payload['eligibility'] != eligible):
                    raise ValueError('The reusable pilot pool does not match the frozen current law')
                payload = dict(payload,particles=payload['particles'][:PARTICLES],
                    stats=dict(payload['stats'],source_particle_count=1024,
                        source_physical_calls=payload['stats']['physical_calls'],
                        physical_calls=0,particle_count=PARTICLES,reused_pilot_pool=True))
                pool = ParticleLaw(payload)
        else:
            def progress(counts):
                save(directory/'progress.json',counts)
                print(json.dumps(dict(stage='first_fit_particle_progress',case_index=index,**counts)),flush=True)
            sampler = make_sampler(kernel,focal,max_stochastic=10,progress=progress,
                physical_limit=PARTICLES,feature_limit=registered['canonical_feature_ceiling'])
            with primary_response_budget(sampler,PARTICLES) as accounting:
                pool = sampler.draw(PARTICLES,registered['particle_seed'])
            if accounting['resolver_constructions'] != sampler.stats['physical_calls']:
                raise ArithmeticError('The primary particle and resolver counters differ')
        particle_path = str(Path(directory.name)/'particles_server_only.json.gz')
        with gzip.open(output/particle_path,'wt',compresslevel=1) as stream:
            json.dump(pool.to_dict(),stream,allow_nan=False)
        with physical_forbidden():
            theta = np.zeros((5,len(FEATURE_NAMES)))
            for obs in observations:
                observed.append(dict(obs,**compact(pool.marginal(theta,obs['head'],obs['y']))))
        supported = observations_supported(observed)
        row = dict(status='PASS_FIRST_FIT_PARTICLE_CASE' if supported else 'HOLD_FIRST_FIT_LABEL_SUPPORT',
            observations=observed,all_observed_heads_supported=supported,stop_new_cases=not supported,
            stochastic_participants=selected['participants'],required_actor_count=len(selected['participants']),
            frame_actor_count=len(kernel.state['packets']))
    except Exception as error:
        (directory/'error_server_only.txt').write_text(traceback.format_exc())
        row = dict(status='HOLD_FIRST_FIT_PARTICLE_CASE',error=f'{type(error).__name__}: {error}',
            observations=observed,selected_observations=observations,
            all_observed_heads_supported=False,stop_new_cases=True,
            current_goal_tuple=getattr(error,'current_goal_tuple',None))
    row.update(reference=ref,case_index=index,particle_seed=registered['particle_seed'],
        particle_path=particle_path,reused_pilot_pool=reused,
        reused_pilot_case_index=registered.get('reused_pilot_case_index'),
        particle_count=PARTICLES if particle_path else None,
        **counters_from_sampler(sampler),resolver_constructions=accounting['resolver_constructions'],
        feature_stage_resolver_calls=0,
        forbidden_feature_or_extra_resolver_attempts=accounting['forbidden_feature_or_extra_resolver_attempts'],
        elapsed_sec=time.monotonic()-began,
        worker_process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
    save(directory/'summary.json',row)
    print(json.dumps(dict(stage='first_fit_particle_case_complete',case_index=index,status=row['status'],
        physical_calls=row['physical_calls'],canonical_feature_evaluations=row['canonical_feature_evaluations'],
        elapsed_sec=row['elapsed_sec'])),flush=True)
    return row


def reference_dispositions(registered,completed,unknown=()):
    receipts = {row['case_index']:row for row in completed}
    unknown_indices = {row['case_index'] for row in unknown}
    rows = []
    for case in registered:
        receipt = receipts.get(case['case_index'])
        by_label = {(obs['head'],obs['manifest_ordinal']):obs for obs in receipt.get('observations',[])} if receipt else {}
        for ref in case['references']:
            obs = by_label.get((ref['head'],ref['manifest_ordinal']))
            status = ('SUPPORTED' if obs and observations_supported([obs]) else
                'UNSUPPORTED_OR_NONFINITE' if obs else receipt['status'] if receipt else
                'STARTED_COUNTERS_UNKNOWN' if case['case_index'] in unknown_indices else 'NOT_STARTED')
            rows.append(dict(ref,case_index=case['case_index'],disposition=status))
    return rows


def run(cache,preflight,synthetic_result,width_file,output,workers=8):
    output.mkdir(parents=True,exist_ok=False)
    began,completed,registered = time.monotonic(),[],[]
    unknown,stopped = [],False
    try:
        proof = json.loads(synthetic_result.read_text())
        frozen = json.loads((preflight/'summary.json').read_text())
        if (proof['status'] != 'PASS_FIRST_FIT_OBJECTIVE_GRADIENT'
                or proof['full_forward_calls'] > 1024
                or any(proof.get(key,0) != 0 for key in ('new_native_starts','new_fits','optimizer_calls'))):
            raise ValueError('The bounded synthetic objective-gradient gate must pass before real responses')
        if (frozen['status'] != 'PASS_FROZEN_FIRST_FIT_COHORT' or frozen['selected_references'] != 512
                or sum(len(case['references']) for case in frozen['cases']) != 512
                or frozen['particle_full_forward_ceiling'] > 131072
                or frozen['canonical_feature_ceiling'] > 1310720 or not 1 <= workers <= 8):
            raise ValueError('The complete frozen512-reference domain and budget must pass before labels')
        registered = frozen['cases']
        labels = load_labels(cache/'training_reservoir_server_only.json',registered)
        save(output/'started.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
            requested_references=512,requested_states=len(registered),workers=workers,
            particle_full_forward_ceiling=frozen['particle_full_forward_ceiling'],
            canonical_feature_ceiling=frozen['canonical_feature_ceiling'],
            dispatch_order='Descending J, fixed lexicographic case indices and seeds unchanged.'))
        ordered = sorted(registered,key=lambda row:(-row['required_actor_count'],row['case_index']))
        tasks = iter((cache,preflight,width_file,row,labels[reference_key(row['reference'])],output) for row in ordered)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as executor:
            pending = {}
            for unused in range(workers):
                task = next(tasks,None)
                if task is not None:
                    pending[executor.submit(inspect_case,task)] = task[3]['case_index']
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
                        pending[executor.submit(inspect_case,task)] = task[3]['case_index']
        physical = sum(row['physical_calls'] for row in completed)
        features = sum(row['canonical_feature_evaluations'] for row in completed)
        if physical > frozen['particle_full_forward_ceiling'] or features > frozen['canonical_feature_ceiling']:
            raise ArithmeticError('The frozen first-fit response budget was exceeded')
        passed = len(completed)==len(registered) and all(row['status']=='PASS_FIRST_FIT_PARTICLE_CASE' for row in completed)
        result = dict(status='PASS_FROZEN_PARTICLE_TRAINING_POOLS' if passed else 'HOLD_FIRST_FIT_PARTICLE_SUPPORT',
            cases=sorted(completed,key=lambda row:row['case_index']),consumed_counters_exact=True,
            actual_full_forward_calls=physical,actual_canonical_feature_evaluations=features,
            stopped_after_failure=stopped)
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        completed,unknown = collect_receipts(output,completed)
        physical = sum(row['physical_calls'] for row in completed)+sum(row['physical_calls_lower_bound'] for row in unknown)
        features = sum(row['canonical_feature_evaluations'] for row in completed)+sum(row['canonical_feature_evaluations_lower_bound'] for row in unknown)
        result = dict(status='HOLD_FIRST_FIT_PARTICLE_TECHNICAL',error=f'{type(error).__name__}: {error}',
            cases=completed,incomplete_started_cases=unknown,consumed_counters_exact=not unknown,
            actual_full_forward_calls=None if unknown else physical,
            actual_canonical_feature_evaluations=None if unknown else features,
            full_forward_calls_lower_bound=physical,canonical_feature_evaluations_lower_bound=features)
    result.update(requested_references=512,requested_states=len(registered),
        completed_case_attempts=len(completed),reference_dispositions=reference_dispositions(registered,completed,unknown),
        feature_stage_resolver_calls=0,parameter_reuse_full_forward_calls=0,
        new_exact_reference_full_forward_calls=0,new_native_starts=0,new_fits=0,
        new_nuisance_estimations=0,optimizer_calls=0,free_rollouts=0,calibration_data_used=False,
        history_exposure='Frozen laws used all four native training trajectories,7200 seconds;904 is internal coefficient development, not independent validation.',
        elapsed_sec=time.monotonic()-began,
        memory_metric='Linux worker-process ru_maxrss, including any earlier case in that worker.')
    save(output/'summary.json',result)
    print(json.dumps({key:result[key] for key in ('status','elapsed_sec','completed_case_attempts','actual_full_forward_calls')}),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache','preflight','synthetic-result','width-file','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=range(1,9),default=8)
    args = parser.parse_args()
    result = run(args.cache,args.preflight,args.synthetic_result,args.width_file,args.output,args.workers)
    raise SystemExit(0 if result['status']=='PASS_FROZEN_PARTICLE_TRAINING_POOLS' else 2)
