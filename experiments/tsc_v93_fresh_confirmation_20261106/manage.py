#!/usr/bin/env python3
"""Stage small new scripts, submit frozen runs, and collect scalar summaries."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCHEDULER = Path('/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py')
sys.path.insert(0, str(SCHEDULER.parent))
import scheduler

OLD = '/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031'
REMOTE = '/home/erzhu419/scheduleurm_work/CFCMT_V93_fresh_confirmation_20261106'
NODES = ('jtl110gpu2', 'jtl311linux')
WORKER = 'run_tsc_v93_fresh_confirmation_shard.py'


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')


def stage():
    files = [(ROOT / 'scripts/cluster' / WORKER, REMOTE + '/runner/' + WORKER),
             (HERE / 'protocol.json', REMOTE + '/protocol.json')]
    receipt = []
    for node in NODES:
        rc, out, err = scheduler.run_on(node, shlex.join(['mkdir', '-p', REMOTE + '/runner']),
                                         timeout=30, check=False)
        if rc:
            raise RuntimeError(err[-1500:])
        for local, remote in files:
            subprocess.run(['rsync', '-a', '-e', scheduler._ssh_rsync_shell_for_node(node),
                            str(local), scheduler._ssh_target_for_node(node) + ':' + remote],
                           check=True, capture_output=True, text=True, timeout=45)
        receipt.append(dict(node=node, uploaded=[p for _, p in files]))
    save('staging.json', dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                             frozen_runtime=OLD, new_output_root=REMOTE, nodes=receipt,
                             operation='new worker and protocol only; frozen source/models unchanged'))
    print(json.dumps(receipt))


def submit(action):
    receipt = HERE / f'{action}_submission.json'
    if receipt.exists():
        raise FileExistsError(f'Inspect existing task IDs in {receipt}; do not submit twice')
    smoke = action == 'smoke'
    count, workers = (1, 4) if smoke else (5, 4)
    specs = []
    for shard in range(count):
        args = [OLD + '/source/scripts/cluster/run_cfcmt_sumo122.sh',
                REMOTE + '/runner/' + WORKER,
                '--parent-launch', OLD + '/relaunch.json',
                '--candidate-spec', OLD + '/confirmation_spec.json',
                '--protocol', REMOTE + '/protocol.json',
                '--results-root', REMOTE + '/' + action,
                '--shard-index', str(shard), '--shard-count', str(count), '--workers', str(workers)]
        if smoke:
            args.append('--smoke')
        specs.append(dict(description=f'Frozen V93 fresh-seed {action} {shard + 1}/{count}',
                          cmd=shlex.join(args), cwd=OLD + '/source',
                          signature=f'CFCMT/v93/fresh-confirmation-20261106/{action}/shard-{shard}',
                          project='CFCMT', vram=0, ram_mb=4096, cpu=workers, priority='normal',
                          allowed_nodes=list(NODES), preferred_node=NODES[shard % len(NODES)],
                          env_spec='none', skip_launch_staging=True, no_auto_requeue=True,
                          protect_from_auto_eviction=True,
                          extra_env={'CFCMT_SOURCE_ROOT': OLD + '/source',
                                     'CFCMT_RUNTIME_ROOT': OLD + '/runtime',
                                     'CFCMT_PYTHON': OLD + '/python/bin/python3.10',
                                     'CFCMT_EXTERNAL_CONVERSION_ROOT': OLD + '/conversion',
                                     'PYTHONUNBUFFERED': '1'}))
    save(f'{action}_specs.json', specs)
    command = [sys.executable, str(SCHEDULER), 'submit-jsonl', '--file',
               str(HERE / f'{action}_specs.json'), '--trusted', '--json',
               '--lock-timeout', '60', '--intent-label', f'CFCMT V93 fresh {action}']
    p = subprocess.run(command, capture_output=True, text=True)
    save(receipt.name, dict(command=command, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr))
    print(p.stdout[-4000:], flush=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    ids = [row['id'] for row in json.loads(p.stdout)['submitted']]
    dispatch = [sys.executable, str(SCHEDULER), 'dispatch', '--lock-timeout', '60',
                '--intent-label', f'CFCMT V93 fresh {action}']
    for ident in ids:
        dispatch += ['--task-id', ident]
    p = subprocess.run(dispatch, capture_output=True, text=True)
    save(f'{action}_dispatch.json', dict(command=dispatch, returncode=p.returncode,
                                       stdout=p.stdout, stderr=p.stderr))
    print(p.stdout[-4500:], flush=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])


def collect(action):
    summaries, receipt = {}, []
    for node in NODES:
        code = f'''import json
from pathlib import Path
r=Path({REMOTE + '/' + action!r})
summaries={{p.name:json.loads(p.read_text()) for p in r.glob('_shards/shard_*.json')}}
payload=dict(summaries=summaries,completed=len(list(r.glob('**/result.json'))))
print(json.dumps(payload,separators=(',',':')))
'''
        rc, out, err = scheduler.run_on(node, 'python3 -c ' + shlex.quote(code), timeout=30, check=False)
        if rc:
            raise RuntimeError(err[-1000:])
        value = json.loads(out)
        receipt.append(dict(node=node, completed=value['completed'], summaries=list(value['summaries'])))
        for name, payload in value['summaries'].items():
            if name in summaries and summaries[name] != payload:
                raise ValueError(f'Different summaries for {name}')
            summaries[name] = payload
    dest = HERE / action / '_shards'
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, payload in sorted(summaries.items()):
        (dest / name).write_text(json.dumps(payload, indent=2) + '\n')
        for row in payload['rows']:
            compact = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
            compact.update(row['metrics'])
            rows.append(compact)
    if rows:
        path = HERE / action / 'rollouts.csv'
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    save(f'{action}_collection.json', dict(collected_at_utc=datetime.now(timezone.utc).isoformat(),
                                          nodes=receipt, collected_rows=len(rows),
                                          storage='scalar shard summaries only; raw results and tripinfo remote'))
    print(json.dumps(dict(nodes=receipt, collected_rows=len(rows))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'submit-smoke', 'submit-evaluation',
                                            'collect-smoke', 'collect-evaluation'])
    action = parser.parse_args().action
    if action == 'stage':
        stage()
    elif action.startswith('submit-'):
        submit(action.removeprefix('submit-'))
    else:
        collect(action.removeprefix('collect-'))
