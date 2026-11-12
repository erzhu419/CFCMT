"""Prepare the frozen weighted LC cohort before any physical response."""

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
from experiments.tsc_packet_mixture_confirmation_20261111.evaluate import read_json
from experiments.tsc_packet_real_particle_pilot_20261110.evaluate import pool_eligibility
from experiments.tsc_packet_real_particle_pilot_20261110.register import reference_key, save
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


SEEDS = (76116901, 76116902, 76116903)


def frozen_tasks(base):
    registration = base / 'target_packet_positive_covered_registration' / 'registration'
    source = read_json(registration / 'result.json')
    cohort = read_json(registration / 'cohort_server_only.json')
    if (source['status'] != 'COMPLETE_POSITIVE_COVERED_LC_REGISTRATION'
            or source['totals']['reference_count'] != 424
            or source['totals']['current_frames_decoded'] != 405
            or source['totals']['proposed_new_full_forward_ceiling'] != 17510
            or source['totals']['proposed_new_feature_ceiling'] != 24738
            or source['totals']['proposed_new_exact_leaf_ceiling'] != 65436):
        raise ValueError('The complete positive-covered registration is required')
    rows = [row for seed in SEEDS for row in read_json(
        registration / f'seed{seed}/registered_server_only.json')['references']]
    rows.sort(key=reference_key)
    references = {reference_key(row): row for row in cohort['references']}
    if (len(rows) != len(references) or len(rows) != 424
            or len({reference_key(row) for row in rows}) != 424
            or Counter(row['y'] for row in rows) != {0: 165, 1: 93, 2: 166}
            or Counter(row['proposed_method'] for row in rows) != {'exact': 387, 'mixture256': 37}
            or sum(row['proposed_reused_cache'] is not None for row in rows) != 3):
        raise ValueError('The frozen selected labels or numerical methods changed')
    if (not isclose(fsum(row['raw_weight'] for row in rows), 30000., rel_tol=0., abs_tol=1e-9)
            or not isclose(fsum(row['objective_weight'] for row in rows), 1., rel_tol=0., abs_tol=1e-12)):
        raise ValueError('The natural-frequency objective weights changed')
    for seed in SEEDS:
        expected_counts = cohort['per_seed'][str(seed)]['original_class_counts']
        for label in (0, 1, 2):
            total = fsum(row['raw_weight'] for row in rows if row['seed'] == seed and row['y'] == label)
            if not isclose(total, expected_counts[str(label)], rel_tol=0., abs_tol=1e-9):
                raise ValueError(f'The original class weight changed in seed {seed}, class {label}')

    tasks = []
    for index, row in enumerate(rows):
        seed, t, focal = reference_key(row)
        original = references[(seed, t, focal)]
        if (seed not in SEEDS or row['head'] != 'lane_change' or row['split'] != 'train'
                or any(row[field] != original[field] for field in
                       ('y', 'manifest_ordinal', 'raw_weight', 'objective_weight'))
                or not isclose(row['objective_weight'], row['raw_weight']/30000., rel_tol=0., abs_tol=1e-15)):
            raise ValueError(f'Frozen label or weight changed at {(seed, t, focal)}')
        reuse = row['proposed_reused_cache']
        if reuse is not None and (row['proposed_method'] != 'mixture256'
                or reuse['kind'] != 'mixture' or reuse['particle_count'] != 256
                or reference_key(reuse['reference']) != (seed, t, focal)
                or reuse['stochastic_participants'] != row['required_participants']
                or reuse['frame_actor_count'] != row['frame_actor_count']
                or Path(reuse['anchor_file']) != base/'target_packet_particle_first_fit/fit/coefficients_server_only.json'
                or not Path(reuse['anchor_file']).is_file() or not Path(reuse['particle_path']).is_file()):
            raise ValueError(f'Reused mixed pool metadata changed at {(seed, t, focal)}')
        method = 'exact' if row['exact_full_forward_ceiling'] <= 256 else 'mixture256'
        physical = 0 if reuse else row['exact_full_forward_ceiling'] if method == 'exact' else 256
        features = 0 if reuse else row['exact_feature_ceiling'] if method == 'exact' else row['all_mixture_feature_ceiling']
        leaves = 0 if reuse or method != 'exact' else row['exact_leaf_ceiling']
        if (row['proposed_method'] != method
                or row['proposed_new_full_forward_ceiling'] != physical
                or row['proposed_new_feature_ceiling'] != features
                or row['proposed_new_exact_leaf_ceiling'] != leaves):
            raise ValueError(f'The registered numerical method or case budget changed at {(seed, t, focal)}')
        task = dict(row, case_index=index,
            reference=dict(seed=seed, time_sec=t, packet_id=focal),
            stem=f'case{index:03d}_seed{seed}_t{t}',
            method=row['proposed_method'], reused_cache=reuse,
            physical_limit=row['proposed_new_full_forward_ceiling'],
            feature_limit=row['proposed_new_feature_ceiling'],
            leaf_limit=row['proposed_new_exact_leaf_ceiling'],
            seed_zero=20261125+100000*index, seed_fitted=20261126+100000*index)
        tasks.append(task)
    if (sum(t['physical_limit'] for t in tasks) != 17510
            or sum(t['feature_limit'] for t in tasks) != 24738
            or sum(t['leaf_limit'] for t in tasks) != 65436):
        raise ValueError('The registered per-case computational limits changed')
    return registration, tasks


def prepare(base, output):
    """Return ordered tasks, kernels keyed by (seed,time), and a compact receipt."""
    registration, tasks = frozen_tasks(base)
    training = base / 'target_packet_training'
    static, _ = load_widths(read_json(training / 'static_graph_server_only.json'))
    laws = read_json(training / 'target_laws_server_only.json')
    width, _ = verify_entity_preflight(base / 'target_packet_entity_footprint/entity_input_preflight.json')
    geometry = build_entity_geometry(RoadTopology(static), width)
    executors = {seed: read_json(training / 'training' / f'seed{seed}/executor_seed_server_only.json')
                 for seed in SEEDS}
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task['seed'], task['time_sec']].append(task)
    if len(grouped) != 405:
        raise ValueError('The frozen cohort requires exactly405 current-frame initializations')
    kernels = {}
    with physical_forbidden():
        for (seed, t), current_tasks in sorted(grouped.items()):
            current_cache = current_tasks[0]['current_cache']
            if any(task['current_cache'] != current_cache for task in current_tasks):
                raise ValueError('A shared current state has inconsistent archived paths')
            with gzip.open(registration / current_cache, 'rt') as stream:
                current = json.load(stream)
            if current['frame']['time_s'] != t or current['action']['time_sec'] != t:
                raise ValueError(f'Archived current-state clock changed at {(seed, t)}')
            kernel = prepare_zero_kernel(static, current['frame'], executors[seed], laws,
                current['action'], RESERVOIR_SEED+seed*100000+t, geometry)
            bindings = current['internal_bindings']
            internal = {key for key, packet in kernel.state['packets'].items()
                        if kernel.topology.lanes[packet['lane_id']]['internal']}
            if (internal != bindings.keys()
                    or any(kernel.state['packets'][key]['_movement_key'] != item['movement_key']
                           for key, item in bindings.items())):
                raise ValueError(f'An archived committed internal path changed at {(seed, t)}')
            kernels[seed, t] = kernel
            for task in current_tasks:
                focal = task['packet_id']
                selection = select_stochastic(kernel, focal, max_stochastic=10)
                if (selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE'
                        or selection['required_participants'] != task['required_participants']
                        or len(selection['participants']) != task['required_actor_count']
                        or len(kernel.state['packets']) != task['frame_actor_count']
                        or pool_eligibility(kernel, focal) != task['eligibility']
                        or not task['eligibility']['lane_change']):
                    raise ValueError(f'The registered full current domain changed at {task["reference"]}')
                budget = symbolic_budget(kernel, selection)
                goals = prod(item['current_positive_goals'] for item in budget['per_actor'].values())
                if (budget['status'] != 'REGISTERED_SYMBOLIC_UPPER_BOUND'
                        or budget['physical_resolver_calls_upper_bound'] != task['exact_full_forward_ceiling']
                        or budget['lateral_leaf_branches_upper_bound'] != task['exact_leaf_ceiling']
                        or goals != task['current_goal_combinations']):
                    raise ValueError(f'The registered exact current budget changed at {task["reference"]}')

    receipt = dict(status='PASS_PREPARED_POSITIVE_COVERED_COHORT',
        selected_references=len(tasks), selected_class_counts=dict(Counter(row['y'] for row in tasks)),
        current_frame_initializations=len(kernels), method_counts=dict(Counter(t['method'] for t in tasks)),
        reused_mixture_cases=sum(t['reused_cache'] is not None for t in tasks),
        new_full_forward_ceiling=sum(t['physical_limit'] for t in tasks),
        new_canonical_feature_ceiling=sum(t['feature_limit'] for t in tasks),
        new_exact_leaf_ceiling=sum(t['leaf_limit'] for t in tasks),
        raw_weight_sum=fsum(t['raw_weight'] for t in tasks),
        objective_weight_sum=fsum(t['objective_weight'] for t in tasks),
        new_full_forward_calls=0, new_canonical_feature_evaluations=0,
        new_native_starts=0, new_fits=0, optimizer_calls=0,
        prepared_at_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'prepared_server_only.json', dict(receipt, cases=tasks))
    save(output / 'prepared.json', receipt)
    return tasks, kernels, receipt
