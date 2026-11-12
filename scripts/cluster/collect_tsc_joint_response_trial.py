#!/usr/bin/env python3
"""Collect one frozen two-state seed shard for the joint response trial."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_tsc_joint_response_measurement import collect_case, write_json

PROTOCOL = 'tsc-joint-response-development-v1'


def validate_spec(spec, seed):
    if (spec['protocol'] != PROTOCOL or spec['max_new_branches'] != 224
            or spec['max_new_fits'] != 30 or spec['response_horizon_sec'] != 10
            or spec['cost_horizon_sec'] != 450 or spec['checkpoint_search_sec'] != 60
            or len(spec['source_seeds']) != 4 or len(spec['target_seeds']) != 10):
        raise ValueError('Use the frozen 224-branch joint response protocol')
    seeds = spec['source_seeds'] + spec['target_seeds']
    cases = spec['cases']
    if (len(set(seeds)) != 14 or seed not in seeds
            or set(seeds) & set(spec['excluded_measurement_seeds'])
            or len(cases) != 28 or len({c['case_id'] for c in cases}) != 28
            or Counter(c['seed'] for c in cases) != Counter({s: 2 for s in seeds})):
        raise ValueError('Keep the disjoint new seeds and two states per seed')
    for city, roster in [('hangzhou', spec['source_seeds']), ('jinan', spec['target_seeds'])]:
        for s in roster:
            own = [c for c in cases if c['seed'] == s]
            if {c['city'] for c in own} != {city} or sorted(c['checkpoint_sec'] for c in own) != [600, 1800]:
                raise ValueError('Each seed retains its city and both frozen anchors')
    return [c for c in cases if c['seed'] == seed]


def run(spec, seed, output):
    cases = validate_spec(spec, seed)
    output.mkdir(parents=True, exist_ok=False)
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = spec['conversion_root']
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
    from static_context import read_static_context
    if '1.22.0' not in str(libsumo_version()):
        raise ValueError('Use the frozen SUMO 1.22.0 runtime')
    start, attempted, groups = time.monotonic(), 0, []

    def progress(case=None, branch=None):
        nonlocal attempted
        attempted += int(branch is not None)
        print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=attempted, total=16, unit='branch',
            elapsed_seconds=time.monotonic() - start, case_id=case['case_id'] if case else None,
            status=branch['status'] if branch else 'START')), flush=True)

    progress()
    api = load_libsumo()
    static = read_static_context(Path(cases[0]['sumocfg']))
    for original in cases:
        case = {**original, 'checkpoint_search_sec': spec['checkpoint_search_sec']}
        result = collect_case(api, v3, case, output / case['case_id'], static, progress)
        groups.append(result)
        if result['status'] != 'COMPLETE':
            break
    result = dict(protocol=PROTOCOL, seed=seed, status='COMPLETE' if len(groups) == 2
        and all(g['status'] == 'COMPLETE' for g in groups) else 'HOLD_COLLECTION',
        groups=groups, new_fits=0, new_branches_attempted=attempted,
        elapsed_sec=time.monotonic() - start)
    write_json(output / 'result.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'seed', 'new_fits', 'new_branches_attempted']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.spec.read_text()), args.seed, args.output)
