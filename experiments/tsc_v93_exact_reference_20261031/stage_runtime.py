#!/usr/bin/env python3
"""Stage the archived V93 runtime/assets on the three user-approved servers.

Uses an SSH-to-SSH memory pipe. No archive, model or checkpoint is written to
local disk, no simulation is run, and the original snapshot remains unchanged.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, '/home/erzhu419/mine_code/scheduleurm/skill')
import scheduler

APPROVED = ('jtl110gpu2', 'jtl311linux', 'jtl110gpu')
DEST = '/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031'
OLD = '/home/zhengliang01/scheduleurm_work'
SNAPSHOT = 'CFCMT_SNAPSHOTS/5decfbd35194996e3f2f'
ENV = 'conda_envs/freqduet-cpu-py310'
RUNTIME = 'runtimes/cfcmt-sumo122-py310-v1'
FREEZE = 'CFCMT_RESULTS/tsc_v43r39_external_full_budget_20260809/full_freeze_v3'
CONVERSION = 'CFCMT_RESULTS/tsc_v42r38_external_la_jinan_20260809/conversion/full_networks_v8'
EVIDENCE = 'CFCMT_EVIDENCE/tsc_v43r39_external_full_budget_20260809'
MODELS_MAIN = 'CFCMT_RESULTS/tsc_v92_topology_context_repair_20260831/refit_v2_source18'
MODELS_COMPONENTS = 'CFCMT_RESULTS/tsc_v93_source_city_components_20260831/components_v2'
OLD_PYTHON = f'{OLD}/{ENV}/bin/python3.10'


def ssh_command(node, command):
    return shlex.split(scheduler._ssh_rsync_shell_for_node(node)) + [scheduler._ssh_target_for_node(node), command]


def source_dependencies():
    code = r'''
import json,os,re,subprocess
from pathlib import Path
env=Path(ENV_LITERAL)
paths=[env/'bin/python3.10']+list((env/'lib/python3.10/lib-dynload').glob('*.so'))
deps=set()
for p in paths:
 result=subprocess.run(['ldd',str(p)],capture_output=True,text=True)
 for value in re.findall(r'=>\s+(/\S+)',result.stdout):
  f=Path(os.path.normpath(value))
  if str(f).startswith(str(env)):
   deps.add(str(f.relative_to(env)))
   deps.add(str(f.resolve().relative_to(env)))
print(json.dumps(sorted(deps)))
'''.replace('ENV_LITERAL', repr(f'{OLD}/{ENV}'))
    rc, out, err = scheduler.run_on('node001', shlex.join([OLD_PYTHON, '-c', code]), timeout=45, check=False)
    if rc:
        raise RuntimeError(err[-1000:])
    return json.loads(out)


def archive_command(deps):
    transforms = [(SNAPSHOT, 'source'), (ENV, 'python'), (RUNTIME, 'runtime'),
                  (FREEZE, 'freeze'), (CONVERSION, 'conversion'),
                  (EVIDENCE, 'evidence'), (MODELS_MAIN, 'models/main'),
                  (MODELS_COMPONENTS, 'models/components')]
    members = [f'{SNAPSHOT}/cf_h2o', f'{SNAPSHOT}/scripts',
               f'{ENV}/bin/python3.10', f'{ENV}/lib/python3.10',
               *[f'{ENV}/{dep}' for dep in deps], RUNTIME, FREEZE, CONVERSION,
               f'{EVIDENCE}/joint_freeze_audit_v1.json', f'{EVIDENCE}/heldout_seed8171_result.json',
               *[f'{prefix}/{city}/{filename}'
                 for prefix, filename in [(MODELS_MAIN, 'model.pkl'), (MODELS_COMPONENTS, 'bundle.pkl')]
                 for city in ['jinan', 'los_angeles']]]
    args = ['tar', '-czf', '-', '-C', OLD, '--exclude=*/__pycache__', '--exclude=*.pyc',
            '--exclude=conda_envs/freqduet-cpu-py310/lib/python3.10/site-packages']
    args += [f'--transform=s|^{old}|{new}|' for old, new in transforms]
    return shlex.join(args + members)


def install_worker_overlay(node):
    files = [
        (ROOT/'scripts/cluster/run_tsc_v93_exact_reference_shard.py', DEST+'/source/scripts/cluster/run_tsc_v93_exact_reference_shard.py'),
        (ROOT/'scripts/cluster/run_tsc_v93_max_pressure_shard.py', DEST+'/source/scripts/cluster/run_tsc_v93_max_pressure_shard.py'),
        (HERE/'reference_audit.patch', DEST+'/reference_audit.patch'),
        (HERE/'relaunch.json', DEST+'/relaunch.json'),
        (ROOT/'cf_h2o/config/traffic_signal_tsc_v93_source_city_disjoint_confirmation_v1.json', DEST+'/confirmation_spec.json'),
    ]
    for local, remote in files:
        subprocess.run(['rsync', '-a', '-e', scheduler._ssh_rsync_shell_for_node(node), str(local),
                        scheduler._ssh_target_for_node(node)+':'+remote], check=True, capture_output=True, text=True, timeout=45)
    command = 'cd '+shlex.quote(DEST+'/source')+' && '+shlex.join(['patch','--batch','--forward','--fuzz=0','-p1','-i',DEST+'/reference_audit.patch'])
    rc, out, err = scheduler.run_on(node, command, timeout=30, check=False)
    if rc:
        raise RuntimeError((out+err)[-1500:])
    return {'overlay_files': [remote for _, remote in files], 'original_snapshot_audit_patch': out.strip()}


def runtime_probe(node):
    code = r'''
import json,platform,socket,sys
from importlib import metadata
import numpy,scipy,sklearn,libsumo
from pathlib import Path
root=Path(ROOT_LITERAL)
print(json.dumps(dict(hostname=socket.gethostname(),python_version=platform.python_version(),
 numpy_version=numpy.__version__,scipy_version=scipy.__version__,scikit_learn_version=sklearn.__version__,
 libsumo_version=metadata.version('libsumo'),python_executable=sys.executable,
 source_ready=(root/'source/cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py').is_file(),
 freeze_ready=(root/'freeze').is_dir(),conversion_ready=(root/'conversion/conversion_manifest.json').is_file())))
'''.replace('ROOT_LITERAL', repr(DEST))
    command = shlex.join(['env', f'PYTHONPATH={DEST}/runtime:{DEST}/source',
                          f'SUMO_HOME={DEST}/runtime/sumo', 'PYTHONNOUSERSITE=1',
                          'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                          f'{DEST}/python/bin/python3.10', '-c', code])
    rc, out, err = scheduler.run_on(node, command, timeout=45, check=False)
    result = dict(node=node, returncode=rc, ready=False, stderr=err[-1500:])
    if not rc:
        result.update(json.loads(out))
        result['ready'] = (result['python_version'] == '3.10.20' and result['numpy_version'] == '2.2.6'
                           and result['scikit_learn_version'] == '1.7.2' and result['libsumo_version'] == '1.22.0'
                           and result['source_ready'] and result['freeze_ready'] and result['conversion_ready'])
    return result


def stage(node, deps):
    started = time.monotonic()
    command = shlex.join(['mkdir', '-p', DEST])
    rc, out, err = scheduler.run_on(node, command, timeout=30, check=False)
    if rc:
        return dict(node=node, ready=False, stage_error=(err or out)[-1500:])
    # Logs are small temporary files; the archive itself passes directly between
    # process pipes, so no model/archive is materialized on the local machine.
    with tempfile.TemporaryFile() as source_errors, tempfile.TemporaryFile() as destination_errors:
        source = subprocess.Popen(ssh_command('node001', archive_command(deps)),
                                  stdout=subprocess.PIPE, stderr=source_errors)
        extract = shlex.join(['tar', '-xzf', '-', '-C', DEST])
        destination = subprocess.Popen(ssh_command(node, extract), stdin=source.stdout,
                                       stdout=subprocess.DEVNULL, stderr=destination_errors)
        source.stdout.close()
        dest_rc = destination.wait()
        source_rc = source.wait()
        source_errors.seek(0); destination_errors.seek(0)
        stderr = (source_errors.read() + destination_errors.read()).decode(errors='replace')[-2200:]
    result = dict(node=node, source_returncode=source_rc, destination_returncode=dest_rc,
                  elapsed_sec=round(time.monotonic()-started,2), stage_stderr=stderr)
    if source_rc or dest_rc:
        result['ready'] = False
        return result
    result.update(install_worker_overlay(node))
    result.update(runtime_probe(node))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', nargs='+', choices=APPROVED, default=list(APPROVED[:2]))
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    deps = [] if args.probe_only else source_dependencies()
    receipt = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), destination=DEST,
                   source_snapshot=f'{OLD}/{SNAPSHOT}', operation='asset staging only; no simulations or job submission',
                   transport='SSH-to-SSH memory pipe; no local archive/model materialization',
                   python_prefix='Exact old Python executable, standard library and its prefix shared dependencies; old site-packages omitted in favor of the unchanged archived runtime overlay',
                   python_shared_dependencies=deps, nodes=[])
    target = HERE/'runtime_staging.json'
    with ThreadPoolExecutor(max_workers=min(2, len(args.nodes))) as pool:
        jobs = {pool.submit(runtime_probe if args.probe_only else stage, node, *([] if args.probe_only else [deps])): node for node in args.nodes}
        for future in as_completed(jobs):
            node = jobs[future]
            try: result = future.result()
            except Exception as exc: result = dict(node=node, ready=False, error=str(exc))
            receipt['nodes'].append(result)
            target.write_text(json.dumps(receipt, indent=2)+'\n')
            print(json.dumps(result), flush=True)
    print(json.dumps(dict(receipt=str(target), ready=[r['node'] for r in receipt['nodes'] if r.get('ready')])), flush=True)
    return 0 if any(r.get('ready') for r in receipt['nodes']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
