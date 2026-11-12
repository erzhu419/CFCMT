"""One LC refit on the frozen stratified caches, with no new physical calls."""

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.optimize import minimize

from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_lc_discrimination_20261111.readout import DIRECTIONS, direction_readout, training_summary
from experiments.tsc_packet_lc_negative_cache_20261111.readout import matched_summary
from experiments.tsc_packet_lc_refit_20261111.fit import compact, load_regression
from experiments.tsc_packet_lc_refit_20261111.objective import LCObjective
from experiments.tsc_packet_lc_refit_20261111.readout import regression_readouts, training_readouts
from experiments.tsc_packet_mixture_confirmation_20261111.evaluate import frozen_candidate, read_json
from experiments.tsc_packet_real_particle_pilot_20261110.register import save
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES

from .load import load_training


def direction_readouts(laws,refs,theta,output):
    records=[]
    with gzip.open(output/'returned_directions_server_only.jsonl.gz','wt') as stream:
        for ref in refs:
            row=dict(ref,points={'returned':{str(d):direction_readout(laws[ref['case_index']],theta,d)
                for d in (DIRECTIONS if ref['y']==1 else (ref['y'],))}})
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            records.append(row)
    return dict(unpaired=training_summary(records,'returned'),matched=matched_summary(records,'returned'))


def run(base,output):
    output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    result=dict(status='HOLD_STRATIFIED_LC_REFIT_PRECHECK',new_full_forward_calls=0,
        new_canonical_feature_evaluations=0,new_native_starts=0,new_nuisance_estimations=0,
        new_fits=0,optimizer_invocations=0,new_particle_pools=0,calibration_seeds_used=False,
        model_admitted=False,original_fit_hold_preserved=True,optimized_coefficients=76,
        started_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'started.json',result)
    objective=None
    try:
        with physical_forbidden():
            anchor=frozen_candidate(base)
            laws,refs,receipt=load_training(base)
            expected=receipt['aggregate']['native']['points']['original']['primary']
            result['baseline_cache_file']=str(base/'target_packet_lc_negative_cache/evaluation/result.json')
            result['baseline_training']=receipt['aggregate']['native']
            result['baseline_direction']={name:{d:dict(
                unpaired_auc=receipt['aggregate']['unpaired'][name][d]['all']['weighted_auc'],
                matched=receipt['aggregate']['matched'][name][d]['all'])
                for d in ('left','right')} for name in ('zero','original','frequency_bias','refitted')}
            result['baseline_regression']=read_json(base/'target_packet_lc_refit/fit/result.json')['regression']
            def record(row):
                with (output/'objective_trace.jsonl').open('a') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps(dict(stage='stratified_lc_objective',**row)),flush=True)
            objective=LCObjective(laws,refs,anchor,record=record)
            initial=objective.evaluate(objective.initial_vector)
            if (abs(initial['unpenalized']-expected['weighted_nll_contribution']['value'])>1e-10
                    or abs(initial['gradient_l2']-expected['lc_gradient']['penalized_l2'])>1e-10):
                raise ArithmeticError('The frozen original point did not reproduce the new natural objective')
            result['initial']=compact(initial)
            save(output/'frozen_baselines.json',dict(initial=result['initial'],
                baseline_training=result['baseline_training'],baseline_direction=result['baseline_direction'],
                baseline_regression=result['baseline_regression']))
            save(output/'optimizer_attempt.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                new_fits=1,optimizer_invocations=1,initialization='original_first_fit_candidate76',
                optimized_rows=[3,4],method='L-BFGS-B',jac=True,options=dict(maxiter=200,gtol=1e-7)))
            result.update(new_fits=1,optimizer_invocations=1)
            optimized=minimize(objective.fun,objective.initial_vector,jac=True,method='L-BFGS-B',
                               options=dict(maxiter=200,gtol=1e-7))
            result['optimizer']=dict(success=bool(optimized.success),status=int(optimized.status),
                message=str(optimized.message),nit=int(optimized.nit),nfev=int(optimized.nfev),njev=int(optimized.njev),
                objective=float(optimized.fun) if np.isfinite(optimized.fun) else None,
                gradient_l2=float(np.linalg.norm(optimized.jac)) if np.isfinite(optimized.jac).all() else None)
            theta=objective.coefficients(optimized.x)
            if not np.array_equal(theta[:3],anchor[:3]):
                raise ArithmeticError('A frozen non-LC row changed')
            save(output/'coefficients_server_only.json',dict(coefficients=theta.tolist(),
                parameter_names=list(PARAMETERS),feature_names=list(FEATURE_NAMES),optimized_rows=[3,4],
                initialization='original_first_fit_candidate76',training_references=1000))
            result['final']=compact(objective.evaluate(optimized.x))
            result['returned_training']=training_readouts(laws,refs,dict(returned=theta),expected_references=1000)
            returned=result['returned_training']['points']['returned']
            if not all(returned[view]['all_references_supported']
                    and returned[view]['weighted_nll_contribution']['state']=='finite'
                    and returned[view]['weighted_gradient']['defined']
                    and returned[view]['lc_gradient']['defined'] for view in ('primary','128')):
                raise ArithmeticError('The returned point failed complete native support or gradients at a fixed prefix size')
            result['returned_direction']=direction_readouts(laws,refs,theta,output)
            regression_laws,cases=load_regression(base)
            result['returned_regression']=regression_readouts(regression_laws,cases,dict(returned=theta))
            result['status']='COMPLETE_STRATIFIED_LC_REFIT' if optimized.success else 'HOLD_STRATIFIED_LC_OPTIMIZER_TERMINATION'
    except Exception as error:
        result.update(status='HOLD_STRATIFIED_LC_REFIT_ATTEMPT' if result['optimizer_invocations'] else 'HOLD_STRATIFIED_LC_REFIT_PRECHECK',
                      error=f'{type(error).__name__}: {error}')
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    result.update(objective_function_attempts=len(objective.trace) if objective is not None else 0,
        elapsed_sec=time.monotonic()-began,completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'result.json',result)
    execution_file=output.parent/'execution.json'
    if execution_file.exists():
        e=read_json(execution_file)
        e.update({k:result[k] for k in ('status','new_fits','optimizer_invocations','objective_function_attempts','elapsed_sec','completed_at_utc')})
        e.update(cumulative_fit_operations=9+result['new_fits'],cumulative_optimizer_invocations=4+result['optimizer_invocations'],counters_finalized=True)
        save(execution_file,e)
    print(json.dumps({k:result[k] for k in ('status','new_fits','optimizer_invocations','elapsed_sec')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if run(args.base,args.output)['status']=='COMPLETE_STRATIFIED_LC_REFIT' else 2)
