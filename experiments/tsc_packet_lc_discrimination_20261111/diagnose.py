"""One cached direction/coverage diagnosis; no optimizer or physical calls."""

import argparse
from datetime import datetime, timezone
import gzip
import json
from math import fsum
from pathlib import Path
import time
import traceback

import numpy as np

from experiments.tsc_packet_conditional_cache_20261110.cache import _vectors
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_lc_refit_20261111.fit import load_training, load_regression
from experiments.tsc_packet_mixture_confirmation_20261111.evaluate import frozen_candidate, read_json
from experiments.tsc_packet_real_particle_pilot_20261110.register import save
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES

from .coverage import coverage_readouts
from .readout import DIRECTIONS, direction_readout, regression_summary, training_summary


def cohort_readouts(laws, cases, points, cohort, output):
    records=[]
    with gzip.open(output/f'{cohort}_readouts_server_only.jsonl.gz','wt') as stream:
        for case in cases:
            law=laws[case['case_index']]
            directions=list(DIRECTIONS) if cohort=='training' else [case['direction']]
            row=dict(case,cohort=cohort,points={},native_nll={})
            for name,theta in points.items():
                row['points'][name]={str(d):direction_readout(law,theta,d) for d in directions}
                native=law.marginal(theta,'lane_change',case['y'])
                if not native['supported'] or not np.isfinite(native['nll']):
                    raise ArithmeticError(f'A previously supported native label failed: {cohort}/{case["case_index"]}')
                row['native_nll'][name]=native['nll']
            records.append(row)
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(records)%100==0:
                print(json.dumps(dict(stage='cached_readouts',cohort=cohort,completed=len(records))),flush=True)
    return records


def reconcile(training, regression, summaries, previous):
    errors=[]
    for name in ('original','refitted'):
        loss=fsum(r['objective_weight']*r['native_nll'][name] for r in training)
        expected=previous['training']['points'][name]['primary']['weighted_nll_contribution']['value']
        errors.append(abs(loss-expected))
        for d,direction in DIRECTIONS.items():
            current=summaries[name][direction]
            old=previous['regression']['points'][name][direction]
            if current['ranking']['direction_log_probability']['all'] != old['ranking']:
                raise ArithmeticError('The frozen904 direction ranking did not reproduce')
            rows=[r for r in regression if r['direction']==d and r['role']=='positive']
            nll=fsum(r['native_nll'][name] for r in rows)/len(rows)
            errors.append(abs(nll-old['mean_positive_actual_nll']['value']))
    if max(errors)>1e-10:
        raise ArithmeticError('A frozen training/regression likelihood did not reproduce')
    return dict(old_metrics_reproduced=True,maximum_absolute_nll_error=max(errors))


def run(base,output):
    output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    result=dict(status='HOLD_CACHED_LC_DISCRIMINATION',new_full_forward_calls=0,
        new_canonical_feature_evaluations=0,new_native_starts=0,new_fits=0,
        optimizer_invocations=0,new_nuisance_estimations=0,new_particle_pools=0,
        calibration_seeds_used=False,model_admitted=False,original_fit_hold_preserved=True,
        started_at_utc=datetime.now(timezone.utc).isoformat(),feature_names=list(FEATURE_NAMES))
    save(output/'started.json',result)
    try:
        with physical_forbidden():
            previous=read_json(base/'target_packet_lc_refit/fit/result.json')
            if previous['status']!='COMPLETE_LC_ONLY_REFIT' or not previous['optimizer']['success']:
                raise ValueError('The completed single returned LC fit is required')
            original=frozen_candidate(base)
            returned=_vectors(read_json(base/'target_packet_lc_refit/fit/coefficients_server_only.json')['coefficients'])
            if not np.array_equal(original[:3],returned[:3]):
                raise ValueError('The frozen non-LC rows changed')
            points=dict(original=original,refitted=returned)
            laws,training_cases,_=load_training(base,original)
            reg_laws,reg_cases=load_regression(base)
            result['coverage'],coverage=coverage_readouts(base,training_cases,reg_cases)
            strata=result['coverage'].pop('training_strata')
            with gzip.open(output/'training_strata_server_only.json.gz','wt') as stream:
                json.dump(strata,stream,allow_nan=False)
            result['coverage'].update(training_strata_count=len(strata),
                                     training_strata_file='training_strata_server_only.json.gz')
            with gzip.open(output/'coverage_server_only.json.gz','wt') as stream:
                json.dump(coverage,stream,allow_nan=False)
            coverage_by_key={(r['cohort'],r['case_index'],r['direction']):r for r in coverage}
            print(json.dumps(dict(stage='coverage_complete',summary=result['coverage'])),flush=True)
            training=cohort_readouts(laws,training_cases,points,'training',output)
            regression=cohort_readouts(reg_laws,reg_cases,points,'regression',output)
            result['training']={name:training_summary(training,name) for name in points}
            result['regression']={name:regression_summary(regression,name,coverage_by_key) for name in points}
            result['reconciliation']=reconcile(training,regression,result['regression'],previous)
            result.update(status='COMPLETE_CACHED_LC_DISCRIMINATION',training_law_references=len(training),
                regression_law_references=len(regression),total_distinct_law_references=len(training)+len(regression),
                training_weight_sum=fsum(r['objective_weight'] for r in training),
                current_state_metadata_only=True,scope='Fixed internal training/regression diagnosis; no independent prediction or control evaluation')
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    result.update(elapsed_sec=time.monotonic()-began,completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'result.json',result)
    print(json.dumps({k:result[k] for k in ('status','elapsed_sec','new_full_forward_calls','new_fits')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if run(args.base,args.output)['status']=='COMPLETE_CACHED_LC_DISCRIMINATION' else 2)
