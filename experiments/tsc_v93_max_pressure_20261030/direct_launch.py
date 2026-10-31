"""Run the supplemental shards with V93's original direct-node mechanism."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/erzhu419/mine_code/scheduleurm/skill')
import scheduler

p = json.loads((HERE / 'protocol.json').read_text())
receipt = HERE / 'direct_submission.json'
if receipt.exists():
    raise FileExistsError('Use the existing direct submission; no duplicate native runs')
# Capacity receipt shows room for sixteen workers on each of these three nodes.
nodes = ['node006', 'node007', 'node004', 'node006', 'node007', 'node004']
runner = p['remote_root'] + '/runner'
commands = []
for shard, node in enumerate(nodes):
    command = shlex.join([
        p['original_snapshot_root'] + '/scripts/cluster/run_cfcmt_sumo122.sh',
        runner + '/scripts/cluster/run_tsc_v93_max_pressure_shard.py',
        '--parent-launch', runner + '/parent_v93_launch.json',
        '--confirmation-spec', runner + '/confirmation_spec.json',
        '--results-root', p['remote_root'] + '/execution',
        '--shard-index', str(shard), '--shard-count', '6', '--workers', '8',
    ])
    commands.append((node, f'CFCMT_SOURCE_ROOT={shlex.quote(p["original_snapshot_root"])} '
        'CFCMT_EXTERNAL_CONVERSION_ROOT=/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_v42r38_external_la_jinan_20260809/conversion/full_networks_v8 '
        'PYTHONUNBUFFERED=1 ' + command))
record = dict(name=p['name'], execution_mode='original-v93-direct-node',
    reason='two scheduler submissions timed out before enqueue; existing tasks left untouched',
    started_at_utc=datetime.now(timezone.utc).isoformat(), status='running',
    workers_per_shard=8, roster=[dict(shard_index=i, node=n, command=c)
        for i, (n, c) in enumerate(commands)], node_results=[])
receipt.write_text(json.dumps(record, indent=2) + '\n')


def run(shard):
    node, command = commands[shard]
    started = time.monotonic()
    rc, out, err = scheduler.run_on(node, command, timeout=1800, check=False)
    return dict(shard_index=shard, node=node, returncode=rc,
        elapsed_sec=time.monotonic()-started, stdout_tail=str(out)[-3500:],
        stderr_tail=str(err)[-2500:])


with ThreadPoolExecutor(max_workers=6) as pool:
    pending = [pool.submit(run, shard) for shard in range(6)]
    for future in as_completed(pending):
        value = future.result()
        record['node_results'].append(value)
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(value), flush=True)
record['status'] = 'complete' if all(r['returncode'] == 0 for r in record['node_results']) else 'failed'
record['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
receipt.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(status=record['status'], completed_shards=len(record['node_results']))), flush=True)
raise SystemExit(0 if record['status'] == 'complete' else 1)
