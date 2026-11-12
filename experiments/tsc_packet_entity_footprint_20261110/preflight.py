"""Read existing input widths and observed lateral states; start no simulator."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
from pathlib import Path
import xml.etree.ElementTree as ET


def input_vehicle_widths(cfg):
    cfg = Path(cfg)
    tree = ET.parse(cfg)
    files = []
    for tag in ('route-files', 'additional-files'):
        node = tree.find('input/'+tag)
        if node is not None:
            files.extend(cfg.parent/name.strip() for name in node.get('value').split(','))
    definitions, widths = [], {}
    for path in files:
        for item in ET.parse(path).iter('vType'):
            definitions.append(dict(source=str(path), **item.attrib))
            if item.get('width') is not None:
                widths[item.get('id')] = float(item.get('width'))
    return widths, definitions


def scan_seed(task):
    seed, directory = task
    types, frames, packets = Counter(), 0, 0
    lateral, speed = 0., 0.
    with gzip.open(directory/'frames_server_only.jsonl.gz', 'rt') as stream:
        for line in stream:
            frame = json.loads(line)
            frames += 1
            for packet in frame['packets'].values():
                packets += 1
                types[packet['type_id']] += 1
                lateral = max(lateral, abs(packet['lateral_position_m']))
                speed = max(speed, abs(packet['lateral_speed_mps']))
    print(json.dumps(dict(stage='entity_input_seed_complete', seed=seed, frames=frames)), flush=True)
    return dict(seed=seed, frames=frames, packet_frames=packets, type_counts=dict(types),
        maximum_abs_lateral_position_m=lateral, maximum_abs_lateral_speed_mps=speed)


def run(cache_base, cfg, output):
    seeds = [76116901, 76116902, 76116903, 76116904]
    widths, definitions = input_vehicle_widths(cfg)
    tasks = [(seed, cache_base/'training'/f'seed{seed}') for seed in seeds]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(scan_seed, tasks))
    types = Counter()
    for row in results:
        types.update(row['type_counts'])
    missing = sorted(set(types)-set(widths))
    observed_widths = sorted({widths[key] for key in types if key in widths})
    lateral = max(row['maximum_abs_lateral_position_m'] for row in results)
    speed = max(row['maximum_abs_lateral_speed_mps'] for row in results)
    centerline = lateral == 0 and speed == 0
    passed = not missing and len(observed_widths) == 1 and centerline and all(r['frames'] == 1801 for r in results)
    result = dict(status='PASS_ENTITY_INPUT_PREFLIGHT' if passed else 'HOLD_ENTITY_INPUT_COVERAGE',
        training_seeds=seeds, vehicle_width_by_type={key: widths[key] for key in types if key in widths},
        observed_type_counts=dict(types), single_vehicle_width_m=observed_widths[0] if len(observed_widths) == 1 else None,
        missing_explicit_type_widths=missing, maximum_abs_lateral_position_m=lateral,
        maximum_abs_lateral_speed_mps=speed, centerline_observed=centerline,
        vtype_definitions=definitions, sumocfg=str(cfg), by_seed=results,
        new_fits=0, new_native_starts=0, scope='Full existing native frame coverage; no future label used to choose width.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-base', type=Path, required=True)
    parser.add_argument('--sumocfg', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Keep the existing input audit; do not overwrite it')
    result = run(args.cache_base, args.sumocfg, args.output)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'PASS_ENTITY_INPUT_PREFLIGHT' else 2)
