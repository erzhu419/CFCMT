"""Stage the MP-only supplement, schedule six shards, and collect compact JSON."""
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


def protocol():
    return json.loads((HERE / 'protocol.json').read_text())


def stage():
    p = protocol()
    remote = p['remote_root'] + '/runner'
    readiness = json.loads((HERE / 'runtime_readiness.json').read_text())
    nodes = [r['node'] for r in readiness if r.get('available') and r.get('returncode') == 0]
    if not nodes:
        raise RuntimeError('No node has the archived V93 runtime')
    command = shlex.join(['mkdir', '-p', remote + '/scripts/cluster'])
    rc, out, err = scheduler.run_on(nodes[0], command, timeout=40, check=False)
    if rc:
        raise RuntimeError(err or out)
    files = [
        (ROOT / 'scripts/cluster/run_tsc_v93_max_pressure_shard.py', 'scripts/cluster/run_tsc_v93_max_pressure_shard.py'),
        (ROOT / p['parent_v93_launch'], 'parent_v93_launch.json'),
        (ROOT / p['parent_confirmation_spec'], 'confirmation_spec.json'),
        (HERE / 'protocol.json', 'protocol.json'),
    ]
    for source, target in files:
        subprocess.run(['rsync', '-a', '-e', scheduler._ssh_rsync_shell_for_node(nodes[0]),
            str(source), scheduler._ssh_target_for_node(nodes[0]) + ':' + remote + '/' + target],
            check=True, capture_output=True, text=True, timeout=45)
    (HERE / 'staging.json').write_text(json.dumps(dict(remote=remote, nodes=nodes,
        staged=[target for _, target in files]), indent=2) + '\n')
    print(json.dumps(dict(staged=len(files), runtime_nodes=nodes)))


def submit():
    p = protocol()
    receipt = HERE / 'submission.json'
    if receipt.exists():
        raise FileExistsError('Use the existing submission record; do not duplicate tasks')
    staging = json.loads((HERE / 'staging.json').read_text())
    remote, nodes = staging['remote'], staging['nodes']
    specs = []
    for shard in range(p['shard_count']):
        command = shlex.join([
            p['original_snapshot_root'] + '/scripts/cluster/run_cfcmt_sumo122.sh',
            remote + '/scripts/cluster/run_tsc_v93_max_pressure_shard.py',
            '--parent-launch', remote + '/parent_v93_launch.json',
            '--confirmation-spec', remote + '/confirmation_spec.json',
            '--results-root', p['remote_root'] + '/execution',
            '--shard-index', str(shard), '--shard-count', str(p['shard_count']),
            '--workers', str(p['workers_per_shard']),
        ])
        specs.append(dict(description=f'Frozen V93 MaxPressure supplement, shard {shard}/6',
            cmd=command, cwd=p['original_snapshot_root'],
            signature=p['name'] + f'/shard-{shard}', project='CFCMT',
            vram=0, ram_mb=12288, cpu=p['workers_per_shard'], priority='normal',
            allowed_nodes=nodes, preferred_node=nodes[shard % len(nodes)],
            env_spec='none', skip_launch_staging=True, no_auto_requeue=True,
            extra_env={'CFCMT_SOURCE_ROOT': p['original_snapshot_root'],
                'CFCMT_EXTERNAL_CONVERSION_ROOT': '/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_v42r38_external_la_jinan_20260809/conversion/full_networks_v8',
                'PYTHONUNBUFFERED': '1'}))
    spec_file = HERE / 'submission_specs.json'
    spec_file.write_text(json.dumps(specs, indent=2) + '\n')
    args = [sys.executable, str(SCHEDULER), 'submit-jsonl', '--file', str(spec_file),
        '--trusted', '--json', '--lock-timeout', '60', '--intent-label', p['name']]
    result = subprocess.run(args, capture_output=True, text=True)
    record = dict(returncode=result.returncode, command=args, stdout=result.stdout, stderr=result.stderr)
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(result.stdout[-4000:])
    if result.returncode:
        raise RuntimeError(result.stderr)


def collect():
    p = protocol()
    remote = p['remote_root'] + '/execution/_shards'
    script = ('from pathlib import Path\nimport json\nr=Path(' + repr(remote) + ')\n'
        'print(json.dumps({x.name:json.loads(x.read_text()) for x in sorted(r.glob("shard_*.json"))}))\n')
    command = shlex.join(['/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python', '-c', script])
    rc, out, err = scheduler.run_on('node007', command, timeout=45, check=False)
    if rc:
        raise RuntimeError(err or out)
    summaries = json.loads(out)
    local = HERE / 'shards/_shards'
    local.mkdir(parents=True, exist_ok=True)
    for name, value in summaries.items():
        (local / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(shards=len(summaries), completed_rollouts=sum(v['task_count'] for v in summaries.values()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'submit', 'collect'])
    action = parser.parse_args().action
    {'stage': stage, 'submit': submit, 'collect': collect}[action]()
