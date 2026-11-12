"""A single target-only coefficient-fit invocation after complete pool support."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy
from scipy.optimize import minimize

from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES
from .model import CompositeObjective, DEVELOPMENT_SEED, HEADS, SHAPE, TRAIN_SEEDS


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def compact(value):
    return {key:item for key,item in value.items() if key != 'gradient'}


def load_pools(directory):
    directory = Path(directory)
    receipt = json.loads((directory/'summary.json').read_text())
    if receipt['status'] != 'PASS_FROZEN_PARTICLE_TRAINING_POOLS':
        raise ValueError('Fitting requires the complete frozen cohort support gate')
    pools, references = {}, []
    for case in receipt['cases']:
        index = int(case['case_index'])
        if index in pools:
            raise ValueError('Duplicate current-case receipt')
        path = Path(case['particle_path'])
        if not path.is_absolute():
            path = directory/path
        with gzip.open(path, 'rt') as stream:
            payload = json.load(stream)
        if len(payload['particles']) != 256:
            raise ValueError('Every frozen current case needs its original256-particle pool')
        pools[index] = ParticleLaw(payload)
        for obs in case['observations']:
            references.append(dict(obs, case_index=index,
                seed=int(case['reference']['seed']),
                time_sec=int(case['reference']['time_sec']),
                packet_id=case['reference']['packet_id']))
    strata = Counter((ref['seed'], ref['head']) for ref in references)
    expected = {(seed, head):32 for seed in (*TRAIN_SEEDS, DEVELOPMENT_SEED) for head in HEADS}
    ids = [(ref['seed'], ref['head'], ref['manifest_ordinal']) for ref in references]
    if dict(strata) != expected or len(ids) != 512 or len(set(ids)) != 512:
        raise ValueError('All512 frozen stratified references and multiplicities must remain')
    return receipt, pools, references


def run(particles, output):
    particles, output = Path(particles), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    result = dict(status='HOLD_FIRST_FIT_PRECHECK', new_fits=0, optimizer_invocations=0,
                  new_particle_full_forward_calls=0, new_exact_full_forward_calls=0,
                  new_nuisance_estimations=0, new_native_starts=0, free_rollouts=0,
                  calibration_seeds_used=False,
                  data_scope='Shared-history coefficient holdout only; all four training trajectories totaling7200 seconds entered frozen priors. Not independent validation.',
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    objective = None
    save(output/'started.json', dict(result, started_at_utc=datetime.now(timezone.utc).isoformat()))
    try:
        unused_receipt, pools, references = load_pools(particles)
        def record(row):
            with (output/'objective_trace.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps(dict(stage='first_fit_objective', **row)), flush=True)
        objective = CompositeObjective(pools, references, record=record)
        theta0 = np.zeros(SHAPE)
        with physical_forbidden():
            before = {split:compact(objective.evaluate(theta0, split, details=True))
                      for split in ('train', 'development')}
            result['zero'] = before
            result.update(selected_references=len(references), unique_current_cases=len(pools),
                          new_fits=1, optimizer_invocations=1)
            save(output/'optimizer_attempt.json', dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                initialization='zero190', method='L-BFGS-B', jac=True,
                options=dict(maxiter=200, gtol=1e-7), new_fits=1, optimizer_invocations=1))
            optimized = minimize(objective.fun, theta0.reshape(-1), jac=True,
                                 method='L-BFGS-B', options=dict(maxiter=200, gtol=1e-7))
            result['optimizer'] = dict(success=bool(optimized.success),
                status=int(optimized.status), message=str(optimized.message), nit=int(optimized.nit),
                nfev=int(optimized.nfev), njev=int(optimized.njev),
                objective=float(optimized.fun) if np.isfinite(optimized.fun) else None,
                gradient_l2=float(np.linalg.norm(optimized.jac)) if np.isfinite(optimized.jac).all() else None)
            theta = np.asarray(optimized.x, dtype=float).reshape(SHAPE)
            if not np.isfinite(theta).all():
                raise ValueError('Optimizer returned nonfinite coefficients')
            save(output/'coefficients_server_only.json', dict(parameter_names=list(PARAMETERS),
                feature_names=list(FEATURE_NAMES), coefficients=theta.tolist(),
                initialization='zero190', history_scope=result['data_scope']))
            result['coefficient_file'] = 'coefficients_server_only.json'
            final = {split:compact(objective.evaluate(theta, split, details=True))
                     for split in ('train', 'development')}
        result.update(final=final,
            status='COMPLETE_TARGET_ONLY_FIRST_FIT' if optimized.success else 'HOLD_FIRST_FIT_OPTIMIZER_TERMINATION',
            train_unpenalized_change=final['train']['unpenalized_sum']-before['train']['unpenalized_sum'],
            internal_development_unpenalized_change=final['development']['unpenalized_sum']-before['development']['unpenalized_sum'])
    except Exception as error:
        result.update(status='HOLD_FIRST_FIT_ATTEMPT' if result['optimizer_invocations'] else 'HOLD_FIRST_FIT_PRECHECK',
                      error=f'{type(error).__name__}: {error}')
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    result.update(objective_evaluations=0 if objective is None else len(objective.trace),
                  elapsed_sec=time.monotonic()-began, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  scientific_scope='Finite cached-objective feasibility only. Weight concentration/support failures and optimizer termination must be assessed before any prediction interpretation; no independent or control-gain claim.')
    save(output/'summary.json', result)
    print(json.dumps({key:result[key] for key in ('status', 'new_fits', 'optimizer_invocations',
          'objective_evaluations', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--particles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.particles, args.output)
    raise SystemExit(0 if result['status'] == 'COMPLETE_TARGET_ONLY_FIRST_FIT' else 2)
