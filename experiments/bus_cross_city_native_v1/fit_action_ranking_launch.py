"""Run read-only ranking workers; retrieve their small JSON reports."""

import argparse
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
PREFIX = "fit_action_ranking"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())
R = json.loads((HERE / "runtime.json").read_text())


def collect(admission=False):
    from launch import run_remote
    directories = ['admission'] if admission else [f'seed{s}' for s in P['training_seeds']]
    code = 'import json; from pathlib import Path; root=Path('+repr(R['remote_root'])+')/"fit_action_ranking"; '
    code += 'rows={name:json.loads(p.read_text()) for name in '+repr(directories)+' if (p:=root/name/"summary.json").exists()}; print(json.dumps(rows))'
    rows = json.loads(run_remote(shlex.join([R['python'],'-c',code])))
    for directory,row in rows.items():
        seed = P['training_seeds'][0] if admission else int(directory[4:])
        if not (row['complete'] and row['seed']==seed and row['admission']==admission
                and row['protocol']==P['name'] and row['rows']==(64 if admission else P['target_fit_rows'])
                and row['updates']==0 and row['evaluation_rows_read']==0):
            raise ValueError('Wrong fixed-fit diagnostic result')
    report = dict(complete=len(rows)==len(directories),ready=len(rows),expected=len(directories),
        protocol=P['name'],admission=admission,results=rows)
    (HERE / (PREFIX + ('_admission_summary.json' if admission else '_summary.json'))).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['complete','ready','expected']}),flush=True)


def submit(admission=False):
    from training_launch import job,submit as submit_jobs
    if not admission and not json.loads((HERE / (PREFIX+'_admission_summary.json')).read_text())['complete']:
        raise ValueError('Frozen Q/objective interface not admitted')
    rows=[]
    for seed in P['training_seeds'][:1] if admission else P['training_seeds']:
        command=[R['python'],'-u',R['remote_root']+'/fit_action_ranking.py','--seed',str(seed)]
        if admission:
            command.append('--admission')
        row=job(command,PREFIX+f'/{seed}')
        row.update(signature=P['name']+('/admission' if admission else '')+f'/{seed}',
            resource_family=P['name'],ram_mb=4000,
            description='Read-only fit action ranking and latent objective '+str(seed))
        rows.append(row)
    submit_jobs(PREFIX+('_admission' if admission else '_diagnose'),rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('stage','admission','collect-admission','diagnose','collect'))
    mode=parser.parse_args().mode
    if mode=='stage':
        from launch import stage_code
        stage_code()
    elif mode in ('admission','diagnose'):
        submit(mode=='admission')
    else:
        collect(mode=='collect-admission')
