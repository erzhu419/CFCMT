"""Prepare the frozen 1000-row weighted cohort without physical responses."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from math import fsum, isclose, prod
from pathlib import Path

from experiments.tsc_packet_conditional_domain_20261110.budget import symbolic_budget
from experiments.tsc_packet_conditional_domain_20261110.register import RESERVOIR_SEED
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_conditional_panel_20261110.initialize import prepare_zero_kernel
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_lc_negative_registration_20261111.sample import (
    POPULATION_CLASS_COUNTS, SEEDS,
)
from experiments.tsc_packet_mixture_confirmation_20261111.evaluate import read_json
from experiments.tsc_packet_real_particle_pilot_20261110.evaluate import pool_eligibility
from experiments.tsc_packet_real_particle_pilot_20261110.register import reference_key, save
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


COSTS = {
    'all_exact_full_forward_ceiling': 67368651082,
    'all_exact_feature_ceiling': 2907009,
    'all_exact_leaf_ceiling': 1321236454342254,
    'all_mixture_full_forward_ceiling': 256000,
    'all_mixture_feature_ceiling': 66945,
    'proposed_new_full_forward_ceiling': 32384,
    'proposed_new_feature_ceiling': 60888,
    'proposed_new_exact_leaf_ceiling': 110478,
}
METHODS = {
    'exact': dict(cases=921, reused=325, new=596, physical_limit=14208,
                  feature_limit=5080, leaf_limit=110478),
    'mixture256': dict(cases=79, reused=8, new=71, physical_limit=18176,
                       feature_limit=55808, leaf_limit=0),
}


def _case_bounds(budget, actors):
    if budget['status'] != 'REGISTERED_SYMBOLIC_UPPER_BOUND':
        raise ValueError('Every selected row must retain its registered symbolic domain')
    goals = prod(item['current_positive_goals'] for item in budget['per_actor'].values())
    return dict(current_goal_combinations=goals,
        all_exact_full_forward_ceiling=budget['physical_resolver_calls_upper_bound'],
        all_exact_feature_ceiling=actors*goals,
        all_exact_leaf_ceiling=budget['lateral_leaf_branches_upper_bound'],
        all_mixture_full_forward_ceiling=256,
        all_mixture_feature_ceiling=actors*min(goals, 256))


def task_from_row(row, original, index, base):
    """Validate one frozen registration and retain either kind of paid cache."""
    base = Path(base)
    seed, time_sec, focal = reference_key(row)
    if (seed not in SEEDS or row['case_index'] != index or original['case_index'] != index
            or row['head'] != 'lane_change' or row['split'] != 'train'
            or any(row[name] != value for name, value in original.items())
            or not isclose(row['objective_weight'], row['raw_weight']/30000.,
                           rel_tol=0., abs_tol=1e-15)):
        raise ValueError(f'The frozen selected row, label or weight changed at {(seed, time_sec, focal)}')
    actors = len(row['required_participants'])
    if (actors != row['required_actor_count'] or not row['eligibility']['lane_change']
            or row['symbolic_status'] != 'REGISTERED_SYMBOLIC_UPPER_BOUND'):
        raise ValueError('The complete registered lane-change actor domain is required')
    bounds = _case_bounds(row['symbolic_budget'], actors)
    if any(row[name] != value for name, value in bounds.items()):
        raise ValueError('A registered exact or mixture symbolic bound changed')
    method = 'exact' if bounds['all_exact_full_forward_ceiling'] <= 256 else 'mixture256'
    reuse = row['proposed_reused_cache']
    anchor = base/'target_packet_particle_first_fit/fit/coefficients_server_only.json'
    if reuse is not None:
        kind = 'exact' if method == 'exact' else 'mixture'
        if (reuse['kind'] != kind or reference_key(reuse['reference']) != (seed, time_sec, focal)
                or reuse['stochastic_participants'] != row['required_participants']
                or reuse['frame_actor_count'] != row['frame_actor_count']
                or not Path(reuse['particle_path']).is_file()):
            raise ValueError('A reused exact or mixed cache changed its identity or actor domain')
        if method == 'exact':
            if reuse['particle_count'] is not None:
                raise ValueError('A reused exact cache must retain its exact-law identity')
        elif (reuse['particle_count'] != 256 or Path(reuse['anchor_file']) != anchor
              or not anchor.is_file()):
            raise ValueError('A reused mixture must retain the original frozen proposal anchor')
    physical = 0 if reuse else bounds['all_exact_full_forward_ceiling'] if method == 'exact' else 256
    features = 0 if reuse else bounds['all_exact_feature_ceiling'] if method == 'exact' else bounds['all_mixture_feature_ceiling']
    leaves = 0 if reuse or method != 'exact' else bounds['all_exact_leaf_ceiling']
    if (row['proposed_method'] != method
            or row['proposed_new_full_forward_ceiling'] != physical
            or row['proposed_new_feature_ceiling'] != features
            or row['proposed_new_exact_leaf_ceiling'] != leaves):
        raise ValueError('A registered cache method or new-call limit changed')
    seed_zero, seed_fitted = 20261130+100000*index, 20261201+100000*index
    if reuse is not None and method == 'mixture256':
        seed_zero, seed_fitted = reuse['seed_zero'], reuse['seed_fitted']
    return dict(row, case_index=index,
        reference=dict(seed=seed, time_sec=time_sec, packet_id=focal),
        stem=f'case{index:03d}_seed{seed}_t{time_sec}', method=method,
        reused_cache=reuse,
        reused_exact_path=reuse['particle_path'] if reuse is not None and method == 'exact' else None,
        proposal_anchor_file=str(anchor),
        exact_full_forward_ceiling=bounds['all_exact_full_forward_ceiling'],
        exact_feature_ceiling=bounds['all_exact_feature_ceiling'],
        exact_leaf_ceiling=bounds['all_exact_leaf_ceiling'],
        physical_limit=physical, feature_limit=features, leaf_limit=leaves,
        seed_zero=seed_zero, seed_fitted=seed_fitted)


def frozen_tasks(base):
    """Read all frozen references and reconcile every registered budget total."""
    base = Path(base)
    registration = base/'target_packet_lc_negative_registration/registration'
    source = read_json(registration/'result.json')
    cohort = read_json(registration/'cohort_server_only.json')
    if (source['status'] != 'COMPLETE_STRATIFIED_NEGATIVE_REGISTRATION'
            or source['total_selected_rows'] != 1000
            or source['total_current_frames_decoded'] != 896
            or source['proposed_method_counts'] != {'exact': 921, 'mixture256': 79}
            or source['proposed_reused_cases'] != 333
            or any(source['complete_costs'][name] != value for name, value in COSTS.items())):
        raise ValueError('The completed frozen 1000-row registration and all budgets are required')
    rows = [row for seed in SEEDS for row in read_json(
        registration/f'seed{seed}/registered_server_only.json')['references']]
    rows.sort(key=reference_key)
    originals = {reference_key(row): row for row in cohort['references']}
    if (len(rows) != 1000 or len(originals) != 1000
            or len({reference_key(row) for row in rows}) != 1000
            or {reference_key(row) for row in rows} != originals.keys()
            or Counter(row['y'] for row in rows) != {0: 165, 1: 669, 2: 166}):
        raise ValueError('The fixed 331-positive and 669-stay selected cohort changed')
    anchor = base/'target_packet_particle_first_fit/fit/coefficients_server_only.json'
    if not anchor.is_file():
        raise FileNotFoundError(anchor)
    tasks = [task_from_row(row, originals[reference_key(row)], index, base)
             for index, row in enumerate(rows)]
    for seed in SEEDS:
        current = [task for task in tasks if task['seed'] == seed]
        for y, expected in POPULATION_CLASS_COUNTS[seed].items():
            total = fsum(task['raw_weight'] for task in current if task['y'] == y)
            if not isclose(total, expected, rel_tol=0., abs_tol=1e-9):
                raise ValueError('An original per-seed natural class weight changed')
        if sum(task['y'] == 1 for task in current) != 223:
            raise ValueError('The fixed 223-stay seed budget changed')
    if (not isclose(fsum(task['raw_weight'] for task in tasks), 30000., rel_tol=0., abs_tol=1e-9)
            or not isclose(fsum(task['objective_weight'] for task in tasks), 1., rel_tol=0., abs_tol=1e-12)
            or any(sum(task[name] for task in tasks) != value for name, value in COSTS.items())):
        raise ValueError('A frozen natural weight or aggregate symbolic bound changed')
    for method, expected in METHODS.items():
        current = [task for task in tasks if task['method'] == method]
        reused = sum(task['reused_cache'] is not None for task in current)
        actual = dict(cases=len(current), reused=reused, new=len(current)-reused,
                      **{name: sum(task[name] for task in current)
                         for name in ('physical_limit', 'feature_limit', 'leaf_limit')})
        if actual != expected:
            raise ValueError(f'The registered {method} reuse or new-call totals changed')
    return registration, tasks


def prepare(base, output):
    """Return tasks, zero kernels keyed by (seed,time), and a compact receipt."""
    base, output = Path(base), Path(output)
    registration, tasks = frozen_tasks(base)
    training = base/'target_packet_training'
    static, _ = load_widths(read_json(training/'static_graph_server_only.json'))
    laws = read_json(training/'target_laws_server_only.json')
    width, _ = verify_entity_preflight(base/'target_packet_entity_footprint/entity_input_preflight.json')
    geometry = build_entity_geometry(RoadTopology(static), width)
    executors = {seed: read_json(training/'training'/f'seed{seed}/executor_seed_server_only.json')
                 for seed in SEEDS}
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task['seed'], task['time_sec']].append(task)
    if len(grouped) != 896:
        raise ValueError('The frozen cohort requires exactly 896 current frames')
    kernels = {}
    with physical_forbidden():
        for (seed, time_sec), current_tasks in sorted(grouped.items()):
            if all(task['reused_cache'] is not None for task in current_tasks):
                continue
            current_path = current_tasks[0]['current_cache']
            if any(task['current_cache'] != current_path for task in current_tasks):
                raise ValueError('A shared current frame changed its registered path')
            with gzip.open(registration/current_path, 'rt') as stream:
                current = json.load(stream)
            if current['frame']['time_s'] != time_sec or current['action']['time_sec'] != time_sec:
                raise ValueError('An archived current-state clock changed')
            kernel = prepare_zero_kernel(static, current['frame'], executors[seed], laws,
                current['action'], RESERVOIR_SEED+seed*100000+time_sec, geometry)
            bindings = current['internal_bindings']
            internal = {key for key, packet in kernel.state['packets'].items()
                        if kernel.topology.lanes[packet['lane_id']]['internal']}
            if (internal != bindings.keys()
                    or any(kernel.state['packets'][key]['_movement_key'] != item['movement_key']
                           for key, item in bindings.items())):
                raise ValueError('An archived committed internal path changed')
            kernels[seed, time_sec] = kernel
            for task in current_tasks:
                focal = task['packet_id']
                selection = select_stochastic(kernel, focal, max_stochastic=10)
                if (selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE'
                        or selection['required_participants'] != task['required_participants']
                        or len(selection['participants']) != task['required_actor_count']
                        or len(kernel.state['packets']) != task['frame_actor_count']
                        or pool_eligibility(kernel, focal) != task['eligibility']):
                    raise ValueError(f'The complete current domain changed at {task["reference"]}')
                bounds = _case_bounds(symbolic_budget(kernel, selection), task['required_actor_count'])
                if any(task[name] != value for name, value in bounds.items()):
                    raise ValueError(f'A current symbolic branch bound changed at {task["reference"]}')
            if len(kernels) % 100 == 0:
                print(json.dumps(dict(stage='lc_negative_current_prepared',
                    current_frame_initializations=len(kernels),
                    registered_current_frames=len(grouped))), flush=True)
    receipt = dict(status='PASS_PREPARED_LC_NEGATIVE_COHORT', selected_references=len(tasks),
        selected_class_counts=dict(Counter(task['y'] for task in tasks)),
        registered_current_frames=len(grouped),
        current_frame_initializations=len(kernels),
        reused_only_frames_skipped=len(grouped)-len(kernels),
        method_counts=dict(Counter(task['method'] for task in tasks)),
        reused_exact_cases=sum(task['reused_exact_path'] is not None for task in tasks),
        reused_mixture_cases=sum(task['method'] == 'mixture256' and task['reused_cache'] is not None for task in tasks),
        new_cache_cases=sum(task['reused_cache'] is None for task in tasks),
        new_full_forward_ceiling=sum(task['physical_limit'] for task in tasks),
        new_canonical_feature_ceiling=sum(task['feature_limit'] for task in tasks),
        new_exact_leaf_ceiling=sum(task['leaf_limit'] for task in tasks),
        raw_weight_sum=fsum(task['raw_weight'] for task in tasks),
        objective_weight_sum=fsum(task['objective_weight'] for task in tasks),
        proposal_anchor_file=str(base/'target_packet_particle_first_fit/fit/coefficients_server_only.json'),
        new_full_forward_calls=0, new_canonical_feature_evaluations=0,
        new_native_starts=0, new_fits=0, optimizer_invocations=0,
        prepared_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'prepared_server_only.json', dict(receipt, cases=tasks))
    save(output/'prepared.json', receipt)
    return tasks, kernels, receipt
