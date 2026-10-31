"""Copy the evaluation code and compact reference records on shared storage."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from rule_common import HERE, P, SNAPSHOT, REFERENCE, write


SOURCE = Path('/home/zhengliang01/scheduleurm_work/Thesis/experiments/sumo_eval')
FIELDS = (
    'method', 'execution_layer', 'horizon_seconds', 'sumo_version', 'deployment_hold_scale', 'split',
    'demand_scale', 'demand_seed', 'sumo_seed', 'arm', 'scenario',
    'training_seed', 'case', 'execution_actions',
    'generalized_seconds_per_departed_passenger', 'passengers',
)


def compact_record(path):
    row = json.loads(path.read_text())
    return {key: row[key] for key in FIELDS if key in row}


def main():
    marker = HERE / 'snapshot_ready.json'
    if marker.exists():
        raise RuntimeError('This experiment already has its frozen runtime')
    for relative in ('attribution_v1/run.py', 'evaluate_control.py', 'data.py', 'protocol_v1.json'):
        destination = SNAPSHOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / relative, destination)

    package = SOURCE / 'runtime/H2Oplus'
    copied = 0
    for folder, directories, files in os.walk(package):
        directories[:] = [name for name in directories if name not in
            {'__pycache__', '.git', 'runs', 'results', 'checkpoints', 'datasets', 'network_data'}]
        for name in files:
            if not name.endswith(('.py', '.yaml', '.yml')):
                continue
            source = Path(folder) / name
            destination = SNAPSHOT / 'runtime/H2Oplus' / source.relative_to(package)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
    for relative in ('assets', 'contract.h5', 'runtime/H2Oplus/bus_h2o/network_data'):
        source = SOURCE / relative
        if not source.exists():
            raise FileNotFoundError(source)
        destination = SNAPSHOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source, target_is_directory=source.is_dir())
    shutil.copy2(HERE / 'fixed_rules.py', SNAPSHOT / 'fixed_rules.py')

    origin = Path(P['source_reference_path'])
    previous = json.loads((origin / 'completed_analysis.json').read_text())
    if not previous['complete']:
        raise RuntimeError('Reference campaign is incomplete')
    write(REFERENCE / 'completed_analysis.json', previous)
    records = 0
    for scene in P['scenarios']:
        for seed in P['training_seeds']:
            relative = Path('runs') / scene / f"{P['primary_arm']}_seed{seed}" / 'results'
            for case in range(10, 20):
                filename = f'eval_{case:02d}.json'
                write(REFERENCE / relative / filename, compact_record(origin / relative / filename))
                records += 1
    control = origin / 'rules/daganzo_duet/case10/results/case10.json'
    write(REFERENCE / 'runtime_control.json', compact_record(control))
    write(marker, dict(created_utc=datetime.now(timezone.utc).isoformat(),
        code_source=str(SOURCE), reference_source=str(origin),
        package_code_files=copied, primary_records=records,
        data_links=['assets', 'contract.h5', 'runtime/H2Oplus/bus_h2o/network_data'],
        scope='Independent copies of execution code and reference summaries; existing static data stays on the server'))
    print(json.dumps(dict(package_code_files=copied, primary_records=records, snapshot_ready=True)))


if __name__ == '__main__':
    main()
