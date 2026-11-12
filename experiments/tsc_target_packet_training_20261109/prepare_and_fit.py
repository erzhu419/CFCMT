"""Attempt four training-history estimates and exactly one frozen joint fit."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import traceback


def save(path,value):
    path.write_text(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n')


def run(base):
    from .training_data import build_training_data
    from .fit_model import fit_joint

    protocol=json.loads((base/'protocol.json').read_text())
    training=json.loads((base/'training/result.json').read_text())
    expected=protocol['cohorts']['training_seeds']
    if training['status']!='COMPLETE' or [row['seed'] for row in training['seeds']]!=expected:
        raise ValueError('All four fixed training trajectories must be complete before estimation')
    if (base/'fit_result.json').exists() or (base/'model.json').exists():
        raise ValueError('The declared fitting cohort has already started; no retry or replacement')
    receipt=dict(status='PREPARING_TRAINING_ONLY',training_seeds=expected,
                 nuisance_estimation_attempts=4,joint_optimization_attempts=0,total_fitting_operations=4,
                 new_native_starts=0,source_data_used=False,calibration_data_used=False)
    save(base/'fit_result.json',receipt)
    begin=perf_counter()
    try:
        seed_dirs={seed:base/'training'/f'seed{seed}' for seed in expected}
        print(json.dumps(dict(stage='prepare_training',training_seeds=expected)),flush=True)
        static=json.loads((base/'static_graph_server_only.json').read_text())
        executor_seeds=json.loads((seed_dirs[expected[0]]/'executor_seed_server_only.json').read_text())
        data=build_training_data(seed_dirs,protocol,static,executor_seeds,artifact_dir=base)
        save(base/'training_sampling_result.json',data['sampling'])
        save(base/'training_references_server_only.json',data['record_references'])
        import numpy as np
        np.savez_compressed(base/'training_arrays_server_only.npz',
            **{head+'__'+key:value for head,arrays in data['heads'].items() for key,value in arrays.items()})
        receipt.update(status='FITTING',sampling=data['sampling'],joint_optimization_attempts=1,total_fitting_operations=5)
        save(base/'fit_result.json',receipt)
        print(json.dumps(dict(stage='joint_optimization',operations=5)),flush=True)
        coefficients,optimization=fit_joint(data['heads'])
        receipt.update(optimization=optimization,wall_seconds=perf_counter()-begin)
        if coefficients is None or not optimization['success']:
            receipt['status']='HOLD_FIT'
        else:
            model=dict(coefficients=coefficients,target_laws=data['laws'],training_seeds=expected,
                       source_data_used=False,calibration_data_used=False,
                       family=protocol['model']['family'],real_data_fitting_operations=5)
            save(base/'model.json',model)
            receipt['status']='PASS_FIT'
        save(base/'fit_result.json',receipt)
    except Exception as exc:
        receipt.update(status='HOLD_TRAINING_PREPARATION',error=f'{type(exc).__name__}: {exc}',wall_seconds=perf_counter()-begin)
        (base/'fit_error.txt').write_text(traceback.format_exc())
        # Model/arrays remain server-side. The receipt retains the actual
        # attempted-operation count even when data construction fails.
        receipt.pop('sampling',None)
        save(base/'fit_result.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args()
    receipt=run(args.base)
    print(json.dumps(dict(stage='fit_end',status=receipt['status'],
                          total_fitting_operations=receipt['total_fitting_operations'],error=receipt.get('error'))),flush=True)
    raise SystemExit(0 if receipt['status']=='PASS_FIT' else 2)
