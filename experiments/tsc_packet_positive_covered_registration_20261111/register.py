"""Register LC training weights and geometric costs from existing archives."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from math import prod
from pathlib import Path
import time
import traceback

from experiments.tsc_packet_conditional_census_20261110.census import (
    FrameSelector, current_eligibility,
)
from experiments.tsc_packet_conditional_domain_20261110.budget import symbolic_budget
from experiments.tsc_packet_conditional_domain_20261110.current import GeometryView
from experiments.tsc_packet_conditional_domain_20261110.register import (
    RESERVOIR_SEED, selected_current_records,
)
from .cohort import select_cohort
from .inventory import cache_inventory


PARTICLES = 256


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def key(row):
    return int(row['seed']), int(row['time_sec']), row['packet_id']


def seed_registration(seed, references, base, output, static, laws, inventory):
    began = time.monotonic()
    folder = output / f'seed{seed}'
    folder.mkdir(exist_ok=False)
    grouped = defaultdict(list)
    for ref in references:
        grouped[ref['time_sec']].append(ref)
    expected = {key(ref) for ref in references}
    old = {}
    with gzip.open(base/'target_packet_conditional_census/output'/f'seed{seed}_cases_server_only.jsonl.gz', 'rt') as stream:
        for row in map(json.loads, stream):
            if key(row) in expected:
                old[key(row)] = row
    if old.keys() != expected:
        raise ValueError(f'The existing complete census misses selected cases for seed {seed}')
    frames, cases = 0, []
    archive = base/'target_packet_training/training'/f'seed{seed}'
    for frame, action, _ in selected_current_records(archive, grouped):
        t = int(frame['time_s'])
        frames += 1
        view = GeometryView(static, frame, laws, action['applied_states'], RESERVOIR_SEED+seed*100000+t)
        selector = FrameSelector(view)
        current_name = f't{t}_current_server_only.json.gz'
        with gzip.open(folder/current_name, 'wt', compresslevel=1) as stream:
            json.dump(dict(frame=frame, action=action, internal_bindings=view.internal_bindings),
                      stream, allow_nan=False)
        for ref in grouped[t]:
            identity = key(ref)
            focal = ref['packet_id']
            previous = old[identity]
            selection = selector.select(focal, max_stochastic=10)
            eligible = current_eligibility(view, view.state['packets'][focal])
            if (selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE'
                    or selection['required_participants'] != previous['required_participants']
                    or len(view.state['packets']) != previous['frame_actor_count']
                    or eligible != previous['eligibility'] or not eligible['lane_change']):
                raise ValueError(f'The complete current domain changed at {identity}')
            budget = symbolic_budget(view, selection)
            if budget['status'] != 'REGISTERED_SYMBOLIC_UPPER_BOUND':
                raise ValueError(f'An original selected case has an undefined full budget at {identity}')
            f = budget['physical_resolver_calls_upper_bound']
            leaves = budget['lateral_leaf_branches_upper_bound']
            if previous['physical_calls_upper'] is not None and (
                    previous['physical_calls_upper'] != f or previous['leaf_visits_upper'] != leaves):
                raise ValueError(f'The saved within-cap budget changed at {identity}')
            actors = len(selection['participants'])
            goals = prod(row['current_positive_goals'] for row in budget['per_actor'].values())
            available = inventory[identity]
            if any(pool['stochastic_participants'] != selection['required_participants']
                   or pool['frame_actor_count'] != len(view.state['packets'])
                   for pools in available.values() for pool in pools):
                raise ValueError(f'An inventoried cache disagrees with the current domain at {identity}')
            method = 'exact' if f <= PARTICLES else 'mixture256'
            compatible = available['exact'] if method == 'exact' else available['mixture']
            reused = compatible[0] if compatible else None
            new_f = 0 if reused else f if method == 'exact' else PARTICLES
            new_features = 0 if reused else actors*(goals if method == 'exact' else min(goals, PARTICLES))
            cases.append(dict(ref, required_actor_count=actors,
                required_participants=selection['required_participants'], frame_actor_count=len(view.state['packets']),
                current_cache=f'seed{seed}/{current_name}', eligibility=eligible,
                current_goal_combinations=goals, exact_full_forward_ceiling=f,
                exact_feature_ceiling=actors*goals, exact_leaf_ceiling=leaves,
                all_mixture_full_forward_ceiling=PARTICLES,
                all_mixture_feature_ceiling=actors*min(goals, PARTICLES),
                available_caches=available, proposed_method=method, proposed_reused_cache=reused,
                proposed_new_full_forward_ceiling=new_f, proposed_new_feature_ceiling=new_features,
                proposed_new_exact_leaf_ceiling=0 if reused or method != 'exact' else leaves,
                per_actor_budget=budget['per_actor']))
    if len(cases) != len(references) or {key(row) for row in cases} != expected:
        raise ValueError(f'Every frozen weighted reference must be registered for seed {seed}')
    cases.sort(key=key)
    save(folder/'registered_server_only.json', dict(seed=seed, references=cases))
    summary = dict(seed=seed, reference_count=len(cases), current_frames_decoded=frames,
        actor_count_histogram=dict(Counter(row['required_actor_count'] for row in cases)),
        all_exact_full_forward_ceiling=sum(r['exact_full_forward_ceiling'] for r in cases),
        all_exact_feature_ceiling=sum(r['exact_feature_ceiling'] for r in cases),
        all_exact_leaf_ceiling=sum(r['exact_leaf_ceiling'] for r in cases),
        all_mixture_full_forward_ceiling=PARTICLES*len(cases),
        all_mixture_feature_ceiling=sum(r['all_mixture_feature_ceiling'] for r in cases),
        proposed_method_counts=dict(Counter(r['proposed_method'] for r in cases)),
        proposed_reused_cases=sum(r['proposed_reused_cache'] is not None for r in cases),
        proposed_new_full_forward_ceiling=sum(r['proposed_new_full_forward_ceiling'] for r in cases),
        proposed_new_feature_ceiling=sum(r['proposed_new_feature_ceiling'] for r in cases),
        proposed_new_exact_leaf_ceiling=sum(r['proposed_new_exact_leaf_ceiling'] for r in cases),
        elapsed_sec=time.monotonic()-began)
    save(folder/'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


def run(base, output):
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    error, summaries = None, []
    cohort_receipt, inventory_receipt = {}, {}
    try:
        cache = base/'target_packet_training'
        cohort, cohort_receipt = select_cohort(read(cache/'training_reservoir_server_only.json'),
            read(base/'target_packet_particle_first_fit/preflight/cohort.json'))
        save(output/'cohort_server_only.json', dict(cohort_receipt, references=cohort))
        available, inventory_receipt = cache_inventory(base, {key(r) for r in cohort})
        static, laws = read(cache/'static_graph_server_only.json'), read(cache/'target_laws_server_only.json')
        grouped = defaultdict(list)
        for row in cohort:
            grouped[row['seed']].append(row)
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(seed_registration, seed, refs, base, output, static, laws, available)
                       for seed, refs in sorted(grouped.items())]
            for future in futures:
                summaries.append(future.result())
        if sum(r['reference_count'] for r in summaries) != 424:
            raise ValueError('The complete424-reference design was not registered')
    except Exception as failure:
        error = f'{type(failure).__name__}: {failure}'
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    totals = {field:sum(row[field] for row in summaries) for field in (
        'reference_count', 'current_frames_decoded', 'all_exact_full_forward_ceiling',
        'all_exact_feature_ceiling', 'all_exact_leaf_ceiling', 'all_mixture_full_forward_ceiling',
        'all_mixture_feature_ceiling', 'proposed_reused_cases', 'proposed_new_full_forward_ceiling',
        'proposed_new_feature_ceiling', 'proposed_new_exact_leaf_ceiling')}
    method_counts, actor_counts = Counter(), Counter()
    for row in summaries:
        method_counts.update(row['proposed_method_counts'])
        actor_counts.update(row['actor_count_histogram'])
    result = dict(status='COMPLETE_POSITIVE_COVERED_LC_REGISTRATION' if error is None else 'HOLD_POSITIVE_COVERED_LC_REGISTRATION',
        error=error, cohort=cohort_receipt, cache_inventory=inventory_receipt, seeds=summaries,
        totals=totals, proposed_method_counts=dict(method_counts), actor_count_histogram=dict(actor_counts),
        proposed_method='Exact enumeration for registered physical ceiling<=256, otherwise corrected fixed-anchor mixture256. Reuse only receipts of the chosen law and proposal kind.',
        new_full_forward_calls=0, new_canonical_feature_evaluations=0, new_native_starts=0,
        new_nuisance_estimations=0, new_fits=0, optimizer_calls=0, native_next_pairs_formed=0,
        calibration_seeds_used=False, coefficient_regression_seed=76116904, model_admitted=False,
        elapsed_sec=time.monotonic()-began, completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'result.json', result)
    print(json.dumps({k:result[k] for k in ('status','error','totals','elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base, args.output)
    raise SystemExit(0 if result['status'] == 'COMPLETE_POSITIVE_COVERED_LC_REGISTRATION' else 2)
