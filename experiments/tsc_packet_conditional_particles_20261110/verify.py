"""Fixed synthetic SNIS comparison, retaining finite-budget support failures."""
import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
from math import fsum
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np

from experiments.tsc_packet_conditional_cache_20261110.cache import build_cache
from experiments.tsc_packet_conditional_cache_20261110 import cache as exact_implementation
from experiments.tsc_packet_conditional_cache_20261110.verify import (
    ABS_TOL, MASS_TOL, REL_TOL, STEP, _fold, _queries, _rng_state,
    count_physical_preparation, gradient_directions, parameter_probes,
    physical_evaluation_forbidden as old_physical_forbidden,
)
from experiments.tsc_packet_conditional_law_20261110.law import enumerate_conditional
from experiments.tsc_packet_conditional_law_20261110 import law as old_implementation
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS, _eligibility
from .fixtures import CASE_NAMES, MotionOracle, cases
from . import fixtures as fixture_implementation
from . import particles


REPLICATES = 32
PARTICLE_SIZES = (256, 1024)
MAX_PARTICLE_CALLS = 491520
MAX_REFERENCE_CALLS = 3000
_WORKER_CASES = None


def initialize_worker():
    """Construct kernels locally: signalled fixtures contain nonpickleable callbacks."""
    global _WORKER_CASES
    _WORKER_CASES = cases()


def worker_kernel(name):
    if _WORKER_CASES is None:
        initialize_worker()
    return _WORKER_CASES[name]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def query_key(query):
    return ':'.join(map(str, query))


def all_queries(eligibility):
    result = [('motion', y) for y in (0., .05, .1, 1.)]
    for head, labels in (('start', (0, 1)), ('yield_entry', (0, 1)), ('lane_change', (0, 1, 2))):
        if eligibility[head]:
            result.extend((head, y) for y in labels)
    if eligibility['lane_change']:
        result.extend(('class_speed', label, y) for label in range(3) for y in (0., 1.))
    return result


def read(pool, theta, query, n=None):
    if query[0] == 'class_speed':
        return pool.class_speed(theta, query[1], query[2], n=n)
    return pool.marginal(theta, query[0], query[1], n=n)


@contextmanager
def physical_forbidden():
    with old_physical_forbidden(), ExitStack() as stack:
        def forbidden(*args, **kwargs):
            raise AssertionError('Particle reuse attempted physical resolution or feature preparation')
        for name in ('_priority_order', 'prepare_receiving_context', 'build_context', 'canonical_features'):
            if hasattr(particles, name):
                stack.enter_context(patch.object(particles, name, forbidden))
        yield


def exact_reference(kernel, focal, name):
    calls = 0
    def counted(original):
        def resolve(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)
        return resolve
    with ExitStack() as stack:
        for module in (exact_implementation, old_implementation, fixture_implementation):
            stack.enter_context(patch.object(module,'_priority_order',counted(module._priority_order)))
        try:
            truth,statistics = _exact_reference(kernel,focal,name)
            assert statistics['physical_calls']==calls
            return truth,statistics
        except Exception as error:
            error.reference_full_forward_calls = calls
            raise


def _exact_reference(kernel, focal, name):
    """Independent physical enumeration verifies the exact cached oracle."""
    if name == 'cap10_motion_oracle':
        with count_physical_preparation() as preparation:
            oracle = MotionOracle(kernel, focal)
        queries = [('motion', y) for y in (0., .05, .1, 1.)]
        truth = [{query: oracle.marginal(theta, *query) for query in queries}
                 for theta in parameter_probes()]
        return truth, dict(kind='CLOSED_FOCAL_MOTION_ONLY', **oracle.stats,
                           reference_resolver_constructions=preparation['resolver_constructions'])
    maximum = 5 if name == 'cap5_focal_last' else 4
    with count_physical_preparation() as preparation:
        oracle = build_cache(kernel, focal, max_stochastic=maximum)
        references = [enumerate_conditional(kernel, focal, dict(zip(PARAMETERS, theta)),
                      max_stochastic=maximum) for theta in parameter_probes()]
    truth, atom_error, likelihood_error = [], 0., 0.
    with physical_forbidden():
        for theta, reference in zip(parameter_probes(), references):
            expected, actual = _fold(reference, focal), _fold(oracle.law(theta), focal)
            assert expected.keys() == actual.keys()
            atom_error = max(atom_error, max(abs(actual[key]-expected[key]) for key in expected))
            assert atom_error <= MASS_TOL
            row = {}
            for query, likelihood in _queries(reference, focal):
                value = (oracle.class_speed(theta, query[1], query[2]) if query[0] == 'class_speed'
                         else oracle.marginal(theta, query[0], query[1]))
                likelihood_error = max(likelihood_error, abs(value['likelihood']-likelihood))
                assert likelihood_error <= MASS_TOL
                row[query] = value
            truth.append(row)
    calls = oracle.stats['physical_calls']+sum(x['diagnostics']['physical_branches'] for x in references)
    return truth, dict(kind='EXACT_CACHE_PLUS_INDEPENDENT_ENUMERATION', physical_calls=calls,
        cache_physical_calls=oracle.stats['physical_calls'],
        independent_physical_calls=calls-oracle.stats['physical_calls'],
        reference_resolver_constructions=preparation['resolver_constructions'],
        maximum_atom_error=atom_error, maximum_likelihood_error=likelihood_error)


def consistency(pool, theta, eligibility, n):
    summary = pool.summary(theta, n=n)
    errors = [abs(summary['total_probability']-1.),
              abs(summary['survivor_probability']+summary['end_probability']-1.)]
    if 'weighted_score' in summary:
        errors.append(float(np.max(np.abs(summary['weighted_score']))))
    if summary['survivor_probability'] > 0.:
        for head, labels in (('start', (0, 1)), ('yield_entry', (0, 1)), ('lane_change', (0, 1, 2))):
            if eligibility[head]:
                errors.append(abs(fsum(pool.marginal(theta, head, y, n=n)['likelihood'] for y in labels)-1.))
        if eligibility['lane_change']:
            for speed in (0., 1.):
                total = fsum(pool.class_speed(theta, label, speed, n=n)['likelihood'] for label in range(3))
                errors.append(abs(total-pool.marginal(theta, 'motion', speed, n=n)['likelihood']))
    if eligibility['yield_entry']:
        errors.append(abs(fsum(pool.marginal(theta, 'yield_entry', y, n=n)['likelihood'] for y in (0, 1))-1.))
    error = max(errors)
    assert error <= MASS_TOL, ('finite common-mixture consistency', error)
    return error, summary


def directional_check(pool, theta, query, n):
    row = read(pool, theta, query, n)
    if not row['defined'] or row['gradient'] is None:
        return 0, 0.
    gradient, largest = np.asarray(row['gradient']), 0.
    assert gradient.shape == theta.shape and np.isfinite(gradient).all()
    for direction in gradient_directions():
        upper = read(pool, theta+STEP*direction, query, n)['nll']
        lower = read(pool, theta-STEP*direction, query, n)['nll']
        difference = (upper-lower)/(2*STEP)
        analytic = float(np.sum(gradient*direction))
        error = abs(difference-analytic)
        assert error <= ABS_TOL+REL_TOL*max(abs(difference), abs(analytic)), (query, difference, analytic)
        largest = max(largest, error)
    return 6, largest


def error_summary(oracle, observations):
    """Failed replicates make overall errors infinite; finite-only is labelled."""
    positive = oracle['likelihood'] > 0.
    supported = [x for x in observations if x['supported']]
    failed = [x for x in observations if not x['supported']]
    result = dict(oracle_likelihood=float(oracle['likelihood']), oracle_positive=positive,
        replicates=len(observations), zero_numerator_replicates=sum(x['failure']=='ZERO_NUMERATOR' for x in failed),
        zero_survival_replicates=sum(x['failure']=='ZERO_SURVIVAL' for x in failed),
        failed_replicates=[{key:x[key] for key in ('replicate','seed','failure')} for x in failed],
        global_ess_min=min(x['global_ess'] for x in observations),
        survivor_ess_min=min(x['survivor_ess'] for x in observations),
        posterior_ess_min=min(x['posterior_ess'] for x in supported) if supported else 0.,
        maximum_normalized_weight=max(x['maximum_normalized_weight'] for x in observations))
    if not positive:
        result.update(structural_zero=True, false_positive_replicates=len(supported))
        assert not supported, 'A particle estimator assigned mass outside the exact oracle support'
        return result
    values = supported
    if values:
        nll = np.array([x['nll_error'] for x in values])
        gradient = np.array([x['gradient_error'] for x in values])
        norms = np.linalg.norm(gradient.reshape(len(gradient), -1), axis=1)
        scale = 1.+float(np.linalg.norm(oracle['gradient']))
        finite = dict(replicates=len(values), nll_bias=float(nll.mean()),
            nll_rmse=float(np.sqrt(np.mean(nll*nll))), nll_absolute_error_p95=float(np.quantile(np.abs(nll), .95)),
            gradient_bias_l2=float(np.linalg.norm(gradient.mean(axis=0))),
            gradient_rmse_l2=float(np.sqrt(np.mean(norms*norms))),
            gradient_error_l2_p95=float(np.quantile(norms,.95)),
            normalized_gradient_bias_l2=float(np.linalg.norm(gradient.mean(axis=0))/scale),
            normalized_gradient_error_p95=float(np.quantile(norms,.95)/scale))
    else:
        finite = None
    result.update(overall_errors_infinite=bool(failed),
        nll_bias=finite['nll_bias'] if not failed else None,
        nll_rmse=finite['nll_rmse'] if not failed else None,
        nll_absolute_error_p95=finite['nll_absolute_error_p95'] if not failed else None,
        gradient_bias_l2=finite['gradient_bias_l2'] if not failed else None,
        gradient_rmse_l2=finite['gradient_rmse_l2'] if not failed else None,
        gradient_error_l2_p95=finite['gradient_error_l2_p95'] if not failed else None,
        normalized_gradient_bias_l2=finite['normalized_gradient_bias_l2'] if not failed else None,
        normalized_gradient_error_p95=finite['normalized_gradient_error_p95'] if not failed else None,
        finite_subset_only=finite if failed else None)
    return result


def _ess(row, summary, key):
    value = row.get(key, summary.get(key, 0.))
    return float(value) if value is not None else 0.


def verify_case(task):
    name, index, output = task
    kernel, focal = worker_kernel(name)
    began = time.monotonic()
    stats = dict(particle_physical_calls=0, reference_physical_calls=0)
    sampler = None
    try:
        state, connections = deepcopy(kernel.state), deepcopy(kernel._connections)
        context, rng = deepcopy(kernel._mean_alignment_context), _rng_state(kernel)
        truth, reference = exact_reference(kernel, focal, name)
        stats['reference_physical_calls'] = reference['physical_calls']
        with count_physical_preparation() as preparation:
            sampler = particles.make_sampler(kernel, focal, max_stochastic=10)
            eligibility = _eligibility(kernel, kernel.state['packets'][focal])
            queries = all_queries(eligibility)
            observations = defaultdict(list)
            max_consistency = max_gradient_error = 0.
            gradient_checks = 0
            for replicate in range(REPLICATES):
                seed = 20261110+100000*index+replicate
                pool = sampler.draw(PARTICLE_SIZES[-1], seed)
                stats['particle_physical_calls'] += pool.stats['physical_calls']
                with physical_forbidden():
                    for n in PARTICLE_SIZES:
                        for theta_index, theta in enumerate(parameter_probes()):
                            error, summary = consistency(pool, theta, eligibility, n)
                            max_consistency = max(max_consistency,error)
                            for query in queries:
                                row = read(pool,theta,query,n)
                                if query in truth[theta_index]:
                                    oracle = truth[theta_index][query]
                                    valid = bool(row['defined'] and row.get('supported',row['likelihood']>0.))
                                    failure = (None if valid else 'ZERO_SURVIVAL' if not row['defined'] else 'ZERO_NUMERATOR')
                                    item = dict(replicate=replicate,seed=seed,supported=valid,failure=failure,
                                        global_ess=_ess(row,summary,'global_ess'),
                                        survivor_ess=_ess(row,summary,'survivor_ess'),
                                        posterior_ess=_ess(row,summary,'posterior_ess'),
                                        maximum_normalized_weight=_ess(row,summary,'max_normalized_weight'))
                                    if valid and oracle['likelihood']>0.:
                                        item.update(nll_error=float(row['nll']-oracle['nll']),
                                            gradient_error=np.asarray(row['gradient'])-np.asarray(oracle['gradient']))
                                    observations[(n,theta_index,query)].append(item)
                                if replicate == 0 and n == PARTICLE_SIZES[-1]:
                                    count,error = directional_check(pool,theta,query,n)
                                    gradient_checks += count
                                    max_gradient_error = max(max_gradient_error,error)
                if (replicate+1)%4 == 0:
                    progress = dict(case=name,replicates_completed=replicate+1,
                        particle_physical_calls=stats['particle_physical_calls'],
                        reference_physical_calls=stats['reference_physical_calls'],elapsed_sec=time.monotonic()-began)
                    save(output/(name+'_progress.json'),progress)
                    print(json.dumps(dict(stage='particle_case_progress',**progress)),flush=True)
                del pool
        assert kernel.state == state and kernel._connections == connections
        assert kernel._mean_alignment_context == context and _rng_state(kernel) == rng
        metrics = {str(n):{str(t):{query_key(query):error_summary(truth[t][query],observations[(n,t,query)])
                        for query in truth[t]} for t in range(2)} for n in PARTICLE_SIZES}
        failures = sum(row['zero_numerator_replicates']+row['zero_survival_replicates']
            for point in metrics[str(PARTICLE_SIZES[-1])].values() for row in point.values() if row['oracle_positive'])
        result = dict(status='HOLD_PARTICLE_SUPPORT' if failures else 'PASS_FIXED_PARTICLE_CASE',
            case=name,case_index=index,replicates=REPLICATES,particle_sizes=list(PARTICLE_SIZES),
            parameter_probes=2,stochastic_actor_count=len(sampler.participants),
            frame_actor_count=len(state['packets']),reference=reference,**stats,
            preparation_resolver_constructions=preparation['resolver_constructions'],
            nonparticle_resolver_constructions=preparation['resolver_constructions']-stats['particle_physical_calls'],
            maximum_consistency_error=max_consistency,gradient_direction_checks=gradient_checks,
            maximum_finite_estimator_gradient_error=max_gradient_error,reuse_physical_calls=0,
            oracle_positive_support_failures_at_1024=failures,metrics=metrics,
            input_state_context_and_rng_unchanged=True,elapsed_sec=time.monotonic()-began)
    except Exception as error:
        stats['reference_physical_calls'] = max(stats['reference_physical_calls'],
                                              getattr(error,'reference_full_forward_calls',0))
        if sampler is not None:
            stats['particle_physical_calls'] = max(stats['particle_physical_calls'],
                                                   sampler.stats['physical_calls'])
        save(output/(name+'_failure.json'),dict(case=name,error=f'{type(error).__name__}: {error}',
            traceback=traceback.format_exc(),**stats,
            sampler_stats=getattr(sampler,'stats',None)))
        raise
    save(output/(name+'_result.json'),result)
    print(json.dumps(dict(stage='particle_case_complete',case=name,status=result['status'],
                         elapsed_sec=result['elapsed_sec'])),flush=True)
    return result


def completed_receipts(source):
    """Reuse the completed cases from this panel's zero-work transport failure."""
    source = Path(source)
    summary = json.loads((source/'result.json').read_text())
    if (summary['requested_cases']!=15 or summary['replicates']!=REPLICATES or
            summary['particle_sizes']!=list(PARTICLE_SIZES)):
        raise ValueError('The prior panel receipt does not match the frozen comparison')
    if (summary['failed_case_particle_physical_calls']!=0 or
            summary['failed_case_reference_physical_calls']!=0):
        raise ValueError('A partially evaluated failed case cannot be restarted by this resume')
    rows = []
    seen = set()
    for entry in summary['cases']:
        name = entry['case']
        if name not in CASE_NAMES or name in seen:
            raise ValueError('Prior completed case identities must be unique fixed cohort names')
        row = json.loads((source/(name+'_result.json')).read_text())
        if (row['case']!=name or row['case_index']!=CASE_NAMES.index(name) or
                row['status'] not in ('PASS_FIXED_PARTICLE_CASE','HOLD_PARTICLE_SUPPORT') or
                row['replicates']!=REPLICATES or row['particle_sizes']!=list(PARTICLE_SIZES) or
                row['parameter_probes']!=2 or row['particle_physical_calls']!=REPLICATES*PARTICLE_SIZES[-1] or
                set(row['metrics'])!=set(map(str,PARTICLE_SIZES))):
            raise ValueError('A completed case receipt differs from the frozen case/protocol shape')
        for n in PARTICLE_SIZES:
            if set(row['metrics'][str(n)])!={'0','1'}:
                raise ValueError('Both fixed parameter probes must be present')
            for point in row['metrics'][str(n)].values():
                if not point or any(metric['replicates']!=REPLICATES for metric in point.values()):
                    raise ValueError('All completed query metrics must include every fixed replicate')
        seen.add(name)
        rows.append(dict(row,reused_from=str(source)))
    if (len(rows)!=summary['completed_cases'] or
            sum(row['particle_physical_calls'] for row in rows)!=summary['particle_physical_calls'] or
            sum(row['reference_physical_calls'] for row in rows)!=summary['reference_physical_calls']):
        raise ValueError('Completed receipts do not account for all prior consumed work')
    return rows


def run(output, workers=4, resume_from=None):
    output = Path(output)
    reused = [] if resume_from is None else completed_receipts(resume_from)
    output.mkdir(parents=True,exist_ok=False)
    began = time.monotonic()
    for row in reused:
        save(output/(row['case']+'_result.json'),row)
    reused_names = {row['case'] for row in reused}
    tasks = [(name,index,output) for index,name in enumerate(CASE_NAMES) if name not in reused_names]
    completed, failure = list(reused), None
    if workers == 1:
        for task in tasks:
            try:
                completed.append(verify_case(task))
            except Exception as error:
                failure = f'{task[0]}: {type(error).__name__}: {error}'
                break
    else:
        iterator = iter(tasks)
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as executor:
            pending = {executor.submit(verify_case,task):task[0] for task in [next(iterator) for _ in range(min(workers,len(tasks)))]}
            while pending:
                done,unused = wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    name = pending.pop(future)
                    try:
                        completed.append(future.result())
                    except Exception as error:
                        failure = f'{name}: {type(error).__name__}: {error}'
                if failure is None:
                    for unused_index in range(len(done)):
                        task = next(iterator,None)
                        if task is not None:
                            pending[executor.submit(verify_case,task)] = task[0]
    failed_receipts = [json.loads(path.read_text()) for path in sorted(output.glob('*_failure.json'))]
    completed_particle_calls = sum(row['particle_physical_calls'] for row in completed)
    completed_reference_calls = sum(row['reference_physical_calls'] for row in completed)
    partial_particle_calls = sum(row['particle_physical_calls'] for row in failed_receipts)
    partial_reference_calls = sum(row['reference_physical_calls'] for row in failed_receipts)
    particle_calls = completed_particle_calls+partial_particle_calls
    reference_calls = completed_reference_calls+partial_reference_calls
    positive_failures = sum(row['oracle_positive_support_failures_at_1024'] for row in completed)
    if failure is None:
        assert len(completed)==15 and particle_calls==MAX_PARTICLE_CALLS
        assert reference_calls<=MAX_REFERENCE_CALLS
    result = dict(status='HOLD_TECHNICAL_PARTICLE_VERIFICATION' if failure is not None else
        'HOLD_FIXED_PARTICLE_SUPPORT' if positive_failures else 'PASS_FIXED_SYNTHETIC_PARTICLE_COMPARISON',
        cases=sorted(completed,key=lambda row:row['case_index']),technical_failure=failure,
        completed_cases=len(completed),requested_cases=15,replicates=32,particle_sizes=list(PARTICLE_SIZES),
        particle_physical_calls=particle_calls,reference_physical_calls=reference_calls,
        completed_case_particle_physical_calls=completed_particle_calls,
        completed_case_reference_physical_calls=completed_reference_calls,
        failed_case_particle_physical_calls=partial_particle_calls,
        failed_case_reference_physical_calls=partial_reference_calls,
        failed_case_receipts=failed_receipts,
        resume_from=None if resume_from is None else str(resume_from),
        reused_completed_case_names=[row['case'] for row in reused],
        reused_particle_physical_calls=sum(row['particle_physical_calls'] for row in reused),
        reused_reference_physical_calls=sum(row['reference_physical_calls'] for row in reused),
        newly_evaluated_particle_physical_calls=particle_calls-sum(row['particle_physical_calls'] for row in reused),
        newly_evaluated_reference_physical_calls=reference_calls-sum(row['reference_physical_calls'] for row in reused),
        oracle_positive_support_failures_at_1024=positive_failures,
        maximum_consistency_error=max((row['maximum_consistency_error'] for row in completed),default=0.),
        maximum_finite_estimator_gradient_error=max((row['maximum_finite_estimator_gradient_error'] for row in completed),default=0.),
        gradient_direction_checks=sum(row['gradient_direction_checks'] for row in completed),
        reuse_physical_calls=0,real_record_response_queries=0,new_native_starts=0,new_fits=0,
        optimizer_calls=0,calibration_seeds_used=False,model_admitted=False,elapsed_sec=time.monotonic()-began,
        scope='Finite-budget synthetic approximation only; finite SNIS likelihood/gradient may be biased. Cap10 oracle covers motion only; real training/source/control benefits are not evaluated.')
    save(output/'result.json',result)
    print(json.dumps({key:result[key] for key in ('status','completed_cases','particle_physical_calls',
        'reference_physical_calls','oracle_positive_support_failures_at_1024','elapsed_sec')}),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume-from',type=Path)
    args=parser.parse_args()
    result=run(args.output,args.workers,args.resume_from)
    raise SystemExit(2 if result['technical_failure'] else 0)
