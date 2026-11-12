#!/usr/bin/env python3
"""Submit the exact-reference smoke/panel only to prepared approved nodes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SCHEDULER=Path('/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py')
REMOTE='/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031'
ALLOWED={'jtl110gpu','jtl110gpu2','jtl311linux'}


def submit(action, nodes):
    assert set(nodes)<=ALLOWED
    receipt=HERE/f'{action}_submission.json'
    if receipt.exists():
        raise FileExistsError(f'Use existing task IDs in {receipt}; no duplicate submission')
    smoke=action=='smoke'
    workers=1 if smoke else 4
    count=1 if smoke else 3
    specs=[]
    for shard in range(count):
        args=[f'{REMOTE}/source/scripts/cluster/run_cfcmt_sumo122.sh',
              f'{REMOTE}/source/scripts/cluster/run_tsc_v93_exact_reference_shard.py',
              '--parent-launch',f'{REMOTE}/relaunch.json',
              '--confirmation-spec',f'{REMOTE}/confirmation_spec.json',
              '--results-root',f'{REMOTE}/{action}',
              '--shard-index',str(shard),'--shard-count',str(count),
              '--workers',str(workers)]
        if smoke: args+=['--scenario','jinan_3x4_real','--seed','5622039']
        specs.append(dict(description=f'Frozen V93 exact-reference {action} shard {shard}/{count}',
            cmd=shlex.join(args),cwd=f'{REMOTE}/source',
            signature=f'CFCMT/v93/exact-reference-20261031/{action}/shard-{shard}',
            project='CFCMT',vram=0,ram_mb=4096,cpu=workers,priority='normal',
            allowed_nodes=nodes,preferred_node=nodes[shard%len(nodes)],
            env_spec='none',skip_launch_staging=True,no_auto_requeue=True,
            extra_env={'CFCMT_SOURCE_ROOT':f'{REMOTE}/source',
                'CFCMT_RUNTIME_ROOT':f'{REMOTE}/runtime',
                'CFCMT_PYTHON':f'{REMOTE}/python/bin/python3.10',
                'CFCMT_EXTERNAL_CONVERSION_ROOT':f'{REMOTE}/conversion',
                'PYTHONUNBUFFERED':'1'}))
    path=HERE/f'{action}_specs.json';path.write_text(json.dumps(specs,indent=2)+'\n')
    command=[sys.executable,str(SCHEDULER),'submit-jsonl','--file',str(path),
             '--trusted','--json','--lock-timeout','180','--intent-label',f'CFCMT exact-reference {action}']
    p=subprocess.run(command,capture_output=True,text=True)
    record=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(p.stdout[-3000:],flush=True)
    if p.returncode:raise RuntimeError(p.stderr[-2000:])
    parsed=json.loads(p.stdout)
    ids=parsed.get('task_ids') or parsed.get('ids') or [row['id'] for row in parsed.get('submitted',[])]
    if not ids:
        print('Inspect submission receipt for task IDs before dispatch.',flush=True)
        return
    dispatch=[sys.executable,str(SCHEDULER),'dispatch','--lock-timeout','180',
              '--intent-label',f'CFCMT exact-reference {action}']
    for ident in ids:dispatch+=['--task-id',str(ident)]
    p=subprocess.run(dispatch,capture_output=True,text=True)
    (HERE/f'{action}_dispatch.json').write_text(json.dumps(dict(command=dispatch,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
    print(p.stdout[-4000:],flush=True)
    if p.returncode:raise RuntimeError(p.stderr[-2000:])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['smoke','evaluation'])
    p.add_argument('--nodes',nargs='+',required=True,choices=sorted(ALLOWED))
    a=p.parse_args();submit(a.action,a.nodes)
