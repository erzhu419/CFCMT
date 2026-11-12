"""Arithmetic-only decomposition of saved old/new matched ranking scores."""

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time
import traceback

from .decompose import audit_edges


def read(path):
    return json.loads(path.read_text())


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def project(row,source_point,name):
    directions=('0','2') if row['y']==1 else (str(row['y']),)
    return dict(case_index=row['case_index'],reference=row['reference'],seed=row['seed'],
        y=row['y'],raw_weight=row['raw_weight'],objective_weight=row['objective_weight'],
        proposed_method=row['proposed_method'],
        directions={d:dict(matched_positive_keys=row['directions'][d]['matched_positive_keys'])
                    for d in ('left','right')},
        points={name:{d:{metric:row['points'][source_point][d][metric]
                         for metric in ('direction_log_probability','conditional_log_probability')}
                      for d in directions}})


def run(base,output):
    output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    result=dict(status='HOLD_SAVED_LC_GAIN_AUDIT',new_full_forward_calls=0,
        new_canonical_feature_evaluations=0,new_native_starts=0,new_fits=0,
        optimizer_invocations=0,new_numerical_likelihood_readouts=0,new_particle_pools=0,
        old_records_read=0,new_records_read=0,started_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        old_root=base/'target_packet_lc_negative_cache/evaluation'
        new_root=base/'target_packet_lc_stratified_refit/fit'
        old_summary,new_summary=read(old_root/'result.json'),read(new_root/'result.json')
        if (old_summary['status']!='COMPLETE_STRATIFIED_FOUR_POINT_CACHE'
                or not old_summary['all_four_points_supported']
                or new_summary['status']!='COMPLETE_STRATIFIED_LC_REFIT'
                or new_summary['new_fits']!=1 or new_summary['optimizer_invocations']!=1
                or Path(new_summary['baseline_cache_file'])!=old_root/'result.json'):
            raise ValueError('The completed fixed old and single returned new point are required')
        tasks=read(old_root/'prepared_server_only.json')['cases']
        old_records=[]
        for task in tasks:
            row=read(old_root/task['stem']/'readouts_server_only.json')
            if row['case_index']!=task['case_index'] or row['reference']!=task['reference']:
                raise ValueError('A saved old readout changed its registered identity')
            old_records.append(project(row,'refitted','old'))
        new_records=[]
        with gzip.open(new_root/'returned_directions_server_only.jsonl.gz','rt') as stream:
            for line in stream:
                new_records.append(project(json.loads(line),'returned','new'))
        result.update(old_records_read=len(old_records),new_records_read=len(new_records))
        if len(old_records)!=1000 or len(new_records)!=1000:
            raise ValueError('All1000 old/new selected records must remain present')
        result['decomposition']=audit_edges(old_records,new_records,
            old_summary['aggregate']['matched']['refitted'],new_summary['returned_direction']['matched'])
        result.update(status='COMPLETE_SAVED_LC_GAIN_AUDIT',
            cumulative_native_starts=13,cumulative_fit_operations=10,cumulative_optimizer_invocations=5,
            parameters_and_caches_read=False,scientific_scope='Fixed internal matched metric only; no independent prediction, causal transfer or control claim')
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    result.update(elapsed_sec=time.monotonic()-began,completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('status','error','old_records_read','new_records_read','elapsed_sec')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if run(args.base,args.output)['status']=='COMPLETE_SAVED_LC_GAIN_AUDIT' else 2)
