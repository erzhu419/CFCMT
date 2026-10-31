"""Stage and schedule the registered eighty frozen-model SUMO evaluations."""

import argparse
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


def read_protocol():
    return json.loads((HERE / 'protocol.json').read_text())


def stage():
    p = read_protocol()
    remote = p['remote_root']
    rc, out, err = scheduler.run_on('node007', 'mkdir -p ' + shlex.quote(remote), timeout=45, check=False)
    if rc:
        raise RuntimeError(err or out)
    files = ['protocol.json', 'evaluation_protocol.json', 'run.py', 'batch.py', 'report.py', 'README.md']
    subprocess.run(['rsync', '-a', '-e', scheduler._ssh_rsync_shell_for_node('node007'),
        *[str(HERE / name) for name in files],
        scheduler._ssh_target_for_node('node007') + ':' + remote + '/'], check=True,
        capture_output=True, text=True, timeout=60)
    print(json.dumps(dict(staged=files, remote_root=remote)))


def submit(dry_run=False):
    p = read_protocol()
    receipt = HERE / ('submission_plan.json' if dry_run else 'submission.json')
    if receipt.exists():
        raise FileExistsError(f'Reuse the existing submission record: {receipt}')
    snapshot, remote = p['original_snapshot_root'], p['remote_root']
    command = shlex.join([snapshot + '/scripts/cluster/run_cfcmt_sumo122.sh',
        remote + '/batch.py', '--start', '{start}', '--end', '{end}', '--workers', '{workers}',
        '--shard-index', '{shard_index}'])
    args = [sys.executable, str(SCHEDULER), 'submit-cpu-batch', '--items', '80',
        '--nodes', 'node001,node002,node003,node004,node005,node006,node007',
        '--cmd-template', command, '--cwd', snapshot, '--project', 'CFCMT',
        '--signature', p['name'] + '/shard-{shard_index}',
        '--description', 'Frozen V43 CFCMT versus dense evaluation [{start},{end})',
        '--ram-mb', '24576', '--priority', 'normal', '--env-spec', 'none',
        '--skip-launch-staging', '--no-auto-requeue',
        '--env', 'CFCMT_SOURCE_ROOT=' + snapshot,
        'CFCMT_EXTERNAL_CONVERSION_ROOT=' + p['frozen_runtime_arguments']['--conversion-root'],
        '--result-dir-template', remote + '/shard_summaries/shard_{shard_index}',
        '--local-result-dir-template', str(HERE / 'shards/shard_{shard_index}'), '--json']
    if dry_run:
        args.append('--dry-run')
    result = subprocess.run(args, capture_output=True, text=True)
    record = dict(analysis_protocol=p['name'], returncode=result.returncode,
        command=args, stdout=result.stdout, stderr=result.stderr)
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stderr)


def collect():
    p = read_protocol()
    remote_python = '/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python'
    script = ('import json\nfrom pathlib import Path\nr=Path(' + repr(p['remote_root']) + ')\n'
        'paths=sorted((r/"summaries").glob("run_*/summary.json"))\n'
        'errors=sorted((r/"summaries").glob("run_*/error.json"))\n'
        'print(json.dumps(dict(records=[json.loads(x.read_text()) for x in paths],'
        'failures=[json.loads(x.read_text()) for x in errors])))\n')
    rc, out, err = scheduler.run_on('node007', shlex.join([remote_python, '-c', script]), timeout=45, check=False)
    if rc:
        raise RuntimeError(err or out)
    data = json.loads(out)
    (HERE / 'results.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    from report import summarize
    report = summarize(data['records'], p)
    report['failures'] = data['failures']
    (HERE / 'reporting.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ['complete', 'ready', 'expected']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'plan', 'submit', 'collect'])
    action = parser.parse_args().action
    if action == 'stage':
        stage()
    elif action == 'collect':
        collect()
    else:
        submit(dry_run=action == 'plan')
