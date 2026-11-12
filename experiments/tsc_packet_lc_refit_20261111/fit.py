"""One cached 76-variable LC fit and the frozen training/904 comparisons."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from math import log
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy
from scipy.optimize import minimize

from experiments.tsc_packet_conditional_cache_20261110.cache import CachedLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_packet_mixture_confirmation_20261111.evaluate import frozen_candidate, read_json
from experiments.tsc_packet_mixture_confirmation_20261111.readout import MixtureParticleLaw
from experiments.tsc_packet_real_particle_pilot_20261110.register import reference_key, save
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES

from .objective import LCObjective
from .readout import regression_readouts, training_readouts


def load_training(base, anchor):
    root = base/'target_packet_positive_covered_cache/evaluation'
    receipt = read_json(root/'result.json')
    tasks = read_json(root/'prepared_server_only.json')['cases']
    if (receipt['status'] != 'COMPLETE_POSITIVE_COVERED_TWO_POINT_CACHE'
            or not receipt['primary_two_point_support_passed'] or len(tasks) != 424):
        raise ValueError('All424 cached cases must pass the complete two-point support gate')
    laws, references = {}, []
    for task in tasks:
        index, ref = task['case_index'], task['reference']
        row = read_json(root/task['stem']/'summary.json')
        if (row['status'] != 'COMPLETE_POSITIVE_COVERED_CACHE_CASE' or index in laws
                or row['reference'] != ref or ref['seed'] not in (76116901, 76116902, 76116903)
                or any(row[k] != task[k] for k in ('y', 'raw_weight', 'objective_weight', 'proposed_method'))):
            raise ValueError('A cached training reference changed its identity or weight')
        with gzip.open(row['cache_path'], 'rt') as stream:
            payload = json.load(stream)
        if (payload['focal_id'] != ref['packet_id']
                or payload['stochastic_participants'] != task['required_participants']
                or payload['frame_actor_count'] != task['frame_actor_count']
                or payload['parameter_names'] != list(PARAMETERS)
                or payload['feature_names'] != list(FEATURE_NAMES) or not payload['eligibility']['lane_change']):
            raise ValueError('A training cache changed its frozen model or current domain')
        if row['proposed_method'] == 'exact':
            laws[index] = CachedLaw(payload)
        else:
            if len(payload['particles']) != 256:
                raise ValueError('Every mixed training law must retain all256 particles')
            laws[index] = MixtureParticleLaw(payload, anchor)
        references.append(dict(task, cache_path=row['cache_path']))
    if (Counter(r['y'] for r in references) != {0:165, 1:93, 2:166}
            or Counter(r['proposed_method'] for r in references) != {'exact':387, 'mixture256':37}
            or len({reference_key(r['reference']) for r in references}) != 424):
        raise ValueError('The fixed positive-covered training target changed')
    return laws, references, receipt


def load_regression(base):
    root = base/'target_packet_lane_change_holdout/evaluation'
    receipt = read_json(root/'result.json')
    cases = read_json(root/'registered_server_only.json')['cases']
    if receipt['status'] != 'COMPLETE_904_EXACT_PAIRED_HOLDOUT' or len(cases) != 124:
        raise ValueError('The complete fixed90462-pair cache is required')
    laws = {}
    for case in cases:
        ref = case['reference']
        if ref['seed'] != 76116904 or case['case_index'] in laws:
            raise ValueError('Only the original904 regression cases may enter this comparison')
        with gzip.open(root/case['stem']/'cache_server_only.json.gz', 'rt') as stream:
            data = json.load(stream)
        if (data['focal_id'] != ref['packet_id']
                or data['stochastic_participants'] != case['required_participants']
                or data['parameter_names'] != list(PARAMETERS) or data['feature_names'] != list(FEATURE_NAMES)):
            raise ValueError('A fixed904 exact cache changed its identity or coefficient layout')
        laws[case['case_index']] = CachedLaw(data)
    if Counter(case['direction'] for case in cases if case['role'] == 'positive') != {0:32, 2:30}:
        raise ValueError('The fixed904 directional pair counts changed')
    return laws, cases


def frequency_reference(anchor):
    theta = anchor.copy()
    theta[3:] = 0.
    bias = list(FEATURE_NAMES).index('bias')
    theta[3, bias] = log(165./29669.)
    theta[4, bias] = log(166./29669.)
    return theta


def compact(value):
    return {key:item for key,item in value.items() if key != 'gradient'}


def run(base, output):
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    result = dict(status='HOLD_LC_REFIT_PRECHECK', new_fits=0, optimizer_invocations=0,
        new_full_forward_calls=0, new_canonical_feature_evaluations=0, new_native_starts=0,
        new_nuisance_estimations=0, source_fits=0, free_rollouts=0, calibration_seeds_used=False,
        parameter_count=76, versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    objective = None
    save(output/'started.json', dict(result, started_at_utc=datetime.now(timezone.utc).isoformat()))
    try:
        with physical_forbidden():
            anchor = frozen_candidate(base)
            laws, references, old_receipt = load_training(base, anchor)
            regression_laws, cases = load_regression(base)
            points = dict(zero=np.zeros_like(anchor), original=anchor, frequency_bias=frequency_reference(anchor))
            save(output/'frozen_reference_server_only.json', dict(
                class_counts=dict(left=165, stay=29669, right=166),
                frequency_bias_coefficients=points['frequency_bias'].tolist(),
                first_three_rows_fixed=True, initialization='original76',
                optimizer_options=dict(maxiter=200, gtol=1e-7), regularization=.001,
                frozen_at_utc=datetime.now(timezone.utc).isoformat()))

            def record(row):
                with (output/'objective_trace.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                print(json.dumps(dict(stage='lc_refit_objective', **row)), flush=True)

            objective = LCObjective(laws, references, anchor, record=record)
            initial = objective.evaluate(objective.initial_vector)
            expected = old_receipt['aggregate']['views']['primary']['points']['candidate']['weighted_nll_contribution']['value']
            if abs(initial['unpenalized']-expected) > 1e-10:
                raise ArithmeticError('The original candidate weighted loss did not reproduce the support audit')
            result['initial'] = compact(initial)
            print(json.dumps(dict(stage='lc_refit_precheck_passed', references=len(references),
                regression_cases=len(cases), unpenalized=initial['unpenalized'])), flush=True)
            # Baselines are fixed and stored before the one optimization attempt.
            result['training'] = training_readouts(laws, references, points)
            result['regression'] = regression_readouts(regression_laws, cases, points)
            save(output/'baseline_readouts.json', dict(training=result['training'], regression=result['regression']))
            save(output/'optimizer_attempt.json', dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                initialization='saved_candidate76', method='L-BFGS-B', jac=True,
                options=dict(maxiter=200, gtol=1e-7), optimized_rows=[3,4], new_fits=1, optimizer_invocations=1))
            result.update(new_fits=1, optimizer_invocations=1)
            optimized = minimize(objective.fun, objective.initial_vector, jac=True,
                                 method='L-BFGS-B', options=dict(maxiter=200, gtol=1e-7))
            result['optimizer'] = dict(success=bool(optimized.success), status=int(optimized.status),
                message=str(optimized.message), nit=int(optimized.nit), nfev=int(optimized.nfev), njev=int(optimized.njev),
                objective=float(optimized.fun) if np.isfinite(optimized.fun) else None,
                gradient_l2=float(np.linalg.norm(optimized.jac)) if np.isfinite(optimized.jac).all() else None)
            theta = objective.coefficients(optimized.x)
            if not np.array_equal(theta[:3], anchor[:3]):
                raise ArithmeticError('LC-only fitting changed a frozen longitudinal coefficient row')
            save(output/'coefficients_server_only.json', dict(parameter_names=list(PARAMETERS),
                feature_names=list(FEATURE_NAMES), coefficients=theta.tolist(), optimized_rows=[3,4],
                initialization='original_candidate76', training_scope='424 natural-weight LC references;904 fixed regression'))
            result['coefficient_file'] = 'coefficients_server_only.json'
            result['final'] = compact(objective.evaluate(optimized.x))
            returned_points = dict(points, refitted=theta)
            result['training'] = training_readouts(laws, references, returned_points)
            result['regression'] = regression_readouts(regression_laws, cases, returned_points)
            result['status'] = 'COMPLETE_LC_ONLY_REFIT' if optimized.success else 'HOLD_LC_REFIT_OPTIMIZER_TERMINATION'
    except Exception as failure:
        result.update(status='HOLD_LC_REFIT_ATTEMPT' if result['optimizer_invocations'] else 'HOLD_LC_REFIT_PRECHECK',
                      error=f'{type(failure).__name__}: {failure}')
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    result.update(objective_trace_file='objective_trace.jsonl', model_admitted=False,
        original_fit_hold_preserved=True, objective_function_attempts=0 if objective is None else len(objective.trace),
        elapsed_sec=time.monotonic()-began, completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'result.json', result)
    print(json.dumps({k:result[k] for k in ('status','new_fits','optimizer_invocations','elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.base, args.output)
    raise SystemExit(0 if result['status'] == 'COMPLETE_LC_ONLY_REFIT' else 2)
