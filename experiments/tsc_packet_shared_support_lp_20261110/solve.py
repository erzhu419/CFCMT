"""The single authorized min-L1 support-initialization parameter estimate."""
import argparse
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import linprog

from .linear import build_min_l1


def native(value):
    """Store the actual nested NumPy/OptimizeResult solver return as JSON."""
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return 'NaN' if value != value else 'Infinity' if value > 0 else '-Infinity'
    return value


def save(path, value):
    path.write_text(json.dumps(native(value), indent=2, allow_nan=False)+'\n')


def run(registration, output):
    output.mkdir(parents=True, exist_ok=False)
    registered = json.loads(registration.read_text())
    if registered['status'] != 'COMPLETE_BRANCH_REGISTRATION' or not registered['raw_affine_witness_matches_all']:
        raise ValueError('The chosen affine witness cells must be registered before spending the LP call')
    cases = registered['cases']
    actors = {key: actor for case in cases for key, actor in case['actors'].items()}
    gaps = [gap for case in cases for gap in case['gaps']]
    problem = build_min_l1(actors, gaps)
    np.savez_compressed(output/'matrix_server_only.npz', A_ub=problem['A_ub'], b_ub=problem['b_ub'], c=problem['c'])
    save(output/'matrix_definition.json', dict(variable_names=problem['variable_names'], bounds=problem['bounds'],
        constraint_labels=problem['constraint_labels'], actor_endpoints=problem['actor_endpoints']))
    summary = dict(status='READY_SINGLE_LP_CALL', registered_cases=len(cases), registered_actors=len(actors),
        registered_gap_constraints=len(gaps), inequality_rows=len(problem['b_ub']), variables=len(problem['c']),
        shared_beta_dimension=38, absolute_auxiliaries=38, scipy_version=scipy.__version__,
        solver_method='highs', objective='min sum absolute beta in the original scaled38 coordinates',
        options=dict(time_limit=60.), physical_gap_epsilon_m=0., tie_policy='Fixed input order; use the returned optimum without a second solve.')
    summary['nonfinite_solver_return_encoding'] = 'Unbounded-variable residuals are preserved as explicit Infinity/-Infinity strings in the complete solver return.'
    save(output/'problem_summary.json', summary)
    save(output/'solver_call_started.json', dict(status='SOLVER_CALL_STARTED', started_at_utc=datetime.now(timezone.utc).isoformat(),
        solver_calls=1, new_fits=1, fit_kind='Support-initialization parameter-estimation attempt', new_native_starts=0))
    start = time.monotonic()
    # Exactly one real optimizer invocation. There is no retry or alternative cell.
    result = linprog(problem['c'], A_ub=problem['A_ub'], b_ub=problem['b_ub'],
        bounds=problem['bounds'], method='highs', options=dict(time_limit=60.))
    save(output/'returned_result_server_only.json', result)
    candidate = result.x is not None and bool(np.isfinite(result.x).all())
    if candidate:
        beta = result.x[:38]
        save(output/'candidate_beta_server_only.json', dict(beta=beta, fit_kind='Fixed-cell support initialization only',
            coefficient_count=38, new_fits=1, solver_calls=1, admitted_model=False))
        violation = np.maximum(problem['A_ub']@result.x-problem['b_ub'], 0.)
        summary.update(beta_l1=float(np.abs(beta).sum()), beta_linf=float(np.abs(beta).max()),
            max_primal_inequality_violation=float(violation.max()),
            nonzero_beta_coefficients=int(np.count_nonzero(beta)),
            positive_violation_rows=[dict(label=label, violation=float(value)) for label, value in
                                     zip(problem['constraint_labels'], violation) if value > 0.])
    summary.update(status='COMPLETE_SINGLE_LP_CALL', solver_status=int(result.status), solver_success=bool(result.success),
        solver_message=str(result.message), objective_value=None if result.fun is None else float(result.fun),
        iterations=None if result.nit is None else int(result.nit), candidate_available=candidate,
        elapsed_sec=time.monotonic()-start, new_fits=1, solver_calls=1, new_native_starts=0,
        full_nll_fits=0, model_admitted=False, next='Validate the one candidate through the existing exact shared API if available; never call the LP again.')
    save(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.registration, args.output)
