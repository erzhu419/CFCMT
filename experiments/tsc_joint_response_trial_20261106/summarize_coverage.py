"""Describe frozen preaction supply support; never read response labels or costs."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median


PROTOCOL = 'tsc-joint-response-development-v2'
FIELDS = ('vehicles', 'halted', 'near_stopline_proxy', 'near_fraction')


def supply(lanes, lane_ids):
    selected = [lanes[key] for key in sorted(set(lane_ids))]
    vehicles = sum(row['vehicle_count'] for row in selected)
    halted = sum(row['lane_measurements']['halted'] for row in selected)
    near = sum(sum(distance <= 10 * row['speed_limit_mps']
                   for distance in row['distances_to_lane_end_m']) for row in selected)
    return [vehicles, halted, near, near / vehicles if vehicles else None]


def describe(values):
    values = [value for value in values if value is not None]
    return dict(count=len(values), min=min(values), median=median(values), max=max(values)) if values else dict(count=0, min=None, median=None, max=None)


def compare(source, target):
    result = {}
    for index, name in enumerate(FIELDS):
        a, b = describe(row[index] for row in source), describe(row[index] for row in target)
        result[name] = dict(source=a, target=b,
            target_above_source_max=sum(row[index] is not None and row[index] > a['max'] for row in target)
                if a['count'] else None)
    return result


def run(spec, collection_root):
    cases = spec['cases']
    if (spec['protocol'] != PROTOCOL or len(cases) != 28
            or Counter(c['city'] for c in cases) != Counter(hangzhou=8, jinan=20)
            or len({c['case_id'] for c in cases}) != 28
            or len(spec['source_seeds']) != 4 or len(spec['target_seeds']) != 10):
        raise ValueError('Expected the frozen 28-group Hangzhou/Jinan protocol')
    for city, seeds in [('hangzhou', spec['source_seeds']), ('jinan', spec['target_seeds'])]:
        if Counter(c['seed'] for c in cases if c['city'] == city) != Counter({s: 2 for s in seeds}):
            raise ValueError('Each frozen seed must have two groups')
    rows, incomplete = [], []
    for case in cases:
        directory = Path(collection_root) / f"seed_{case['seed']}" / case['case_id']
        try:
            receipt = json.loads((directory / 'result.json').read_text())
            branches = receipt.get('branches', [])
            if (receipt.get('status') != 'COMPLETE' or len(branches) != 8
                    or {b['candidate_index'] for b in branches} != set(range(8))
                    or any(b['status'] != 'COMPLETE' or not b['checks'] or not all(b['checks'].values()) for b in branches)):
                incomplete.append(dict(case_id=case['case_id'], status=receipt.get('status')))
                continue
            graph = json.loads((directory / 'joint_interface_server_only.json').read_text())
        except FileNotFoundError as error:
            incomplete.append(dict(case_id=case['case_id'], missing_file=error.filename))
            continue
        if any(receipt[key] != case[key] for key in ('case_id', 'city', 'seed')) or graph['case_id'] != case['case_id']:
            raise ValueError('Receipt or graph identity differs from the frozen case')
        lanes = {row['lane_id']: row for row in graph['lane_nodes']}
        incoming = sorted({lane for movement in graph['movement_nodes'] for lane in movement['incoming_lanes']})
        halted = [lanes[key]['lane_measurements']['halted'] for key in incoming]
        rows.append(dict(case_id=case['case_id'], city=case['city'], seed=case['seed'],
            time_sec=graph['information_time_sec'], incoming_lane_count=len(incoming),
            supply=supply(lanes, incoming), lane_halted=halted,
            max_lane_halted=max(halted), lanes_with_halted=sum(value > 0 for value in halted),
            movement_supply=[supply(lanes, movement['incoming_lanes']) for movement in graph['movement_nodes']]))
    result = dict(protocol=PROTOCOL, status='HOLD_COLLECTION' if incomplete else 'COVERAGE_REVIEW_REQUIRED',
        new_fits=0, new_simulations=0, expected_groups=28, complete_groups=len(rows),
        case_ids=[case['case_id'] for case in cases], incomplete=incomplete, groups=rows,
        fields=list(FIELDS), lane_order='Sorted unique incoming lane IDs within each graph.',
        movement_order='Original graph.movement_nodes order; each row uses its unique incoming lanes.',
        semantics='Near proxy counts observed vehicles within speed_limit*10m of lane end; not observed discharge. Empty-supply fractions are null. Groups deduplicate shared lanes; movements are overlapping units, not summed demand.',
        scope='Only receipt completeness and graph t0 inputs are used. No labels, costs or future signal states; coverage requires scientific judgment, not a score or automatic PASS.')
    if not incomplete:
        source = [row for row in rows if row['city'] == 'hangzhou']
        target = [row for row in rows if row['city'] == 'jinan']
        result['group_support'] = compare([r['supply'] for r in source], [r['supply'] for r in target])
        result['movement_support'] = compare([m for r in source for m in r['movement_supply']],
                                             [m for r in target for m in r['movement_supply']])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'collection-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.spec.read_text()), args.collection_root)
    args.output.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'complete_groups', 'new_fits', 'new_simulations')}), flush=True)
