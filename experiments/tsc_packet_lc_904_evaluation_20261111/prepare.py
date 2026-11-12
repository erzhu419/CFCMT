"""Prepare the fixed 340-row 904 regression without physical responses."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from math import fsum, isclose
from pathlib import Path

from experiments.tsc_packet_conditional_domain_20261110.budget import symbolic_budget
from experiments.tsc_packet_conditional_domain_20261110.register import RESERVOIR_SEED
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_conditional_panel_20261110.initialize import prepare_zero_kernel
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_lc_904_registration_20261111.inventory import cost_plan
from experiments.tsc_packet_lc_negative_cache_20261111.prepare import _case_bounds
from experiments.tsc_packet_mixture_confirmation_20261111.evaluate import read_json
from experiments.tsc_packet_real_particle_pilot_20261110.evaluate import pool_eligibility
from experiments.tsc_packet_real_particle_pilot_20261110.register import reference_key, save
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


SEED = 76116904
COSTS = {
    'all_exact_full_forward_ceiling': 167794655738,
    'all_exact_feature_ceiling': 4788746,
    'all_exact_leaf_ceiling': 3302275004634834,
    'all_mixture_full_forward_ceiling': 87040,
    'all_mixture_feature_ceiling': 24842,
    'proposed_new_full_forward_ceiling': 11820,
    'proposed_new_feature_ceiling': 24206,
    'proposed_new_exact_leaf_ceiling': 35280,
}
METHODS = {
    'exact': dict(cases=312, reused=70, new=242, new_full_forward_ceiling=4908,
                  new_feature_ceiling=1870, new_exact_leaf_ceiling=35280),
    'mixture256': dict(cases=28, reused=1, new=27, new_full_forward_ceiling=6912,
                       new_feature_ceiling=22336, new_exact_leaf_ceiling=0),
}


def _model_manifest(base, registration, source):
    manifest = read_json(registration/'frozen_models_server_only.json')
    models = manifest['models']
    anchor = base/'target_packet_particle_first_fit/fit/coefficients_server_only.json'
    if (set(models) != {'old_lc', 'new_lc', 'frequency_bias'}
            or manifest['coefficient_arrays_read'] or models != source['frozen_models']
            or Path(models['old_lc']['coefficient_file']) != base/'target_packet_lc_refit/fit/coefficients_server_only.json'
            or Path(models['new_lc']['coefficient_file']) != base/'target_packet_lc_stratified_refit/fit/coefficients_server_only.json'
            or Path(models['frequency_bias']['non_lc_rows_source']) != anchor
            or models['frequency_bias']['training_class_counts'] != dict(left=165, right=166, stay=29669)):
        raise ValueError('The three previously frozen models and training-frequency reference are required')
    for path in (anchor, Path(models['old_lc']['coefficient_file']), Path(models['new_lc']['coefficient_file'])):
        if not path.is_file():
            raise FileNotFoundError(path)
    return manifest


def task_from_row(row, original, index, base):
    """Keep a frozen row and apply the registered paid-law-first method rule."""
    base = Path(base)
    seed, time_sec, focal = reference_key(row)
    if (seed != SEED or row['case_index'] != index or original['case_index'] != index
            or row['head'] != 'lane_change' or row['split'] != 'regression'
            or any(row[name] != value for name, value in original.items())
            or not isclose(row['objective_weight'], row['raw_weight']/10000., rel_tol=0., abs_tol=1e-15)
            or not isclose(row['raw_weight']*row['inclusion_probability'], 1., rel_tol=0., abs_tol=1e-12)):
        raise ValueError('The frozen 904 selected row, inclusion probability or natural weight changed')
    actors = len(row['required_participants'])
    if (actors != row['required_actor_count'] or not row['eligibility']['lane_change']
            or row['symbolic_status'] != 'REGISTERED_SYMBOLIC_UPPER_BOUND'):
        raise ValueError('The complete registered 904 actor domain is required')
    bounds = _case_bounds(row['symbolic_budget'], actors)
    plan = cost_plan(row['symbolic_budget'], row['required_participants'],
                     row['frame_actor_count'], row['available_caches'])
    if (any(row[name] != value for name, value in bounds.items())
            or any(row[name] != value for name, value in plan.items())):
        raise ValueError('A registered paid-law preference or symbolic bound changed')
    method, reuse = plan['proposed_method'], plan['proposed_reused_cache']
    anchor = base/'target_packet_particle_first_fit/fit/coefficients_server_only.json'
    if reuse is not None:
        kind = 'exact' if method == 'exact' else 'mixture'
        if (reuse['kind'] != kind or reference_key(reuse['reference']) != (seed, time_sec, focal)
                or reuse['stochastic_participants'] != row['required_participants']
                or reuse['frame_actor_count'] != row['frame_actor_count']
                or not Path(reuse['particle_path']).is_file()):
            raise ValueError('A paid law changed its saved identity or full geometric actor domain')
        if method == 'exact':
            if reuse['particle_count'] is not None:
                raise ValueError('A paid exact law changed its exact cache kind')
        elif (reuse['particle_count'] != 256 or Path(reuse['anchor_file']) != anchor
              or not anchor.is_file()):
            raise ValueError('A paid mixture changed its original proposal anchor')
    seed_zero, seed_fitted = 20261210+100000*index, 20261211+100000*index
    if reuse is not None and method == 'mixture256':
        seed_zero, seed_fitted = reuse['seed_zero'], reuse['seed_fitted']
    return dict(row, case_index=index,
        reference=dict(seed=seed, time_sec=time_sec, packet_id=focal),
        stem=f'case{index:03d}_seed{seed}_t{time_sec}', method=method, reused_cache=reuse,
        reused_exact_path=reuse['particle_path'] if reuse is not None and method == 'exact' else None,
        proposal_anchor_file=str(anchor),
        exact_full_forward_ceiling=bounds['all_exact_full_forward_ceiling'],
        exact_feature_ceiling=bounds['all_exact_feature_ceiling'],
        exact_leaf_ceiling=bounds['all_exact_leaf_ceiling'],
        physical_limit=plan['proposed_new_full_forward_ceiling'],
        feature_limit=plan['proposed_new_feature_ceiling'],
        leaf_limit=plan['proposed_new_exact_leaf_ceiling'],
        seed_zero=seed_zero, seed_fitted=seed_fitted)


def frozen_tasks(base):
    """Read the 340-row registration, preserving all metadata and denominators."""
    base = Path(base)
    registration = base/'target_packet_lc_904_registration/registration'
    source = read_json(registration/'result.json')
    summary = source['registration']
    if (source['status'] != 'COMPLETE_904_NATURAL_REGISTRATION'
            or summary['reference_count'] != 340 or summary['registered_cases'] != 340
            or summary['unregistered_cases'] != 0 or summary['current_frames_decoded'] != 313
            or summary['methods'] != METHODS or summary['proposed_reused_cases'] != 71
            or any(summary['complete_costs'][name] != value for name, value in COSTS.items())):
        raise ValueError('The complete frozen 340-row 904 registration and all budgets are required')
    _model_manifest(base, registration, source)
    cohort = read_json(registration/'cohort_server_only.json')
    rows = read_json(registration/f'seed{SEED}/registered_server_only.json')['references']
    rows.sort(key=reference_key)
    originals = {reference_key(row): row for row in cohort['references']}
    if (len(rows) != 340 or len(originals) != 340
            or len({reference_key(row) for row in rows}) != 340
            or {reference_key(row) for row in rows} != originals.keys()
            or Counter(row['y'] for row in rows) != {0: 56, 1: 223, 2: 61}):
        raise ValueError('The fixed 117-positive and 223-stay 904 cohort changed')
    tasks = [task_from_row(row, originals[reference_key(row)], index, base)
             for index, row in enumerate(rows)]
    for y, expected in ((0, 56), (1, 9883), (2, 61)):
        total = fsum(task['raw_weight'] for task in tasks if task['y'] == y)
        if not isclose(total, expected, rel_tol=0., abs_tol=1e-9):
            raise ValueError('An original 904 natural class weight changed')
    if (not isclose(fsum(task['raw_weight'] for task in tasks), 10000., rel_tol=0., abs_tol=1e-9)
            or not isclose(fsum(task['objective_weight'] for task in tasks), 1., rel_tol=0., abs_tol=1e-12)
            or any(sum(task[name] for task in tasks) != value for name, value in COSTS.items())):
        raise ValueError('A frozen 904 natural weight or aggregate symbolic bound changed')
    for method, expected in METHODS.items():
        selected = [task for task in tasks if task['method'] == method]
        reused = sum(task['reused_cache'] is not None for task in selected)
        actual = dict(cases=len(selected), reused=reused, new=len(selected)-reused,
            new_full_forward_ceiling=sum(task['physical_limit'] for task in selected),
            new_feature_ceiling=sum(task['feature_limit'] for task in selected),
            new_exact_leaf_ceiling=sum(task['leaf_limit'] for task in selected))
        if actual != expected:
            raise ValueError('A frozen 904 per-method reuse or construction total changed')
    return registration, tasks


def prepare(base, output):
    """Return tasks, only needed zero kernels keyed by (seed,time), and receipt."""
    base, output = Path(base), Path(output)
    registration, tasks = frozen_tasks(base)
    training = base/'target_packet_training'
    static, _ = load_widths(read_json(training/'static_graph_server_only.json'))
    laws = read_json(training/'target_laws_server_only.json')
    width, _ = verify_entity_preflight(base/'target_packet_entity_footprint/entity_input_preflight.json')
    geometry = build_entity_geometry(RoadTopology(static), width)
    executors = read_json(training/'training'/f'seed{SEED}/executor_seed_server_only.json')
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task['seed'], task['time_sec']].append(task)
    if len(grouped) != 313:
        raise ValueError('The fixed 904 cohort requires exactly 313 registered current frames')
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
                raise ValueError('An archived 904 current-state clock changed')
            kernel = prepare_zero_kernel(static, current['frame'], executors, laws,
                current['action'], RESERVOIR_SEED+seed*100000+time_sec, geometry)
            bindings = current['internal_bindings']
            internal = {key for key, packet in kernel.state['packets'].items()
                        if kernel.topology.lanes[packet['lane_id']]['internal']}
            if (internal != bindings.keys()
                    or any(kernel.state['packets'][key]['_movement_key'] != item['movement_key']
                           for key, item in bindings.items())):
                raise ValueError('An archived 904 committed internal path changed')
            kernels[seed, time_sec] = kernel
            for task in current_tasks:
                selection = select_stochastic(kernel, task['packet_id'], max_stochastic=10)
                if (selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE'
                        or selection['required_participants'] != task['required_participants']
                        or len(selection['participants']) != task['required_actor_count']
                        or len(kernel.state['packets']) != task['frame_actor_count']
                        or pool_eligibility(kernel, task['packet_id']) != task['eligibility']):
                    raise ValueError(f'The complete 904 current domain changed at {task["reference"]}')
                bounds = _case_bounds(symbolic_budget(kernel, selection), task['required_actor_count'])
                if any(task[name] != value for name, value in bounds.items()):
                    raise ValueError('A registered 904 current symbolic bound changed')
            if len(kernels) % 100 == 0:
                print(json.dumps(dict(stage='904_current_prepared', current_frame_initializations=len(kernels),
                                      registered_current_frames=len(grouped))), flush=True)
    receipt = dict(status='PASS_PREPARED_904_NATURAL_COHORT', selected_references=len(tasks),
        selected_class_counts=dict(Counter(task['y'] for task in tasks)),
        registered_current_frames=len(grouped), current_frame_initializations=len(kernels),
        reused_only_frames_skipped=len(grouped)-len(kernels),
        method_counts=dict(Counter(task['method'] for task in tasks)), reused_exact_cases=70,
        reused_mixture_cases=1, new_cache_cases=269,
        new_full_forward_ceiling=sum(task['physical_limit'] for task in tasks),
        new_canonical_feature_ceiling=sum(task['feature_limit'] for task in tasks),
        new_exact_leaf_ceiling=sum(task['leaf_limit'] for task in tasks),
        raw_weight_sum=fsum(task['raw_weight'] for task in tasks),
        objective_weight_sum=fsum(task['objective_weight'] for task in tasks), objective_denominator=10000,
        proposal_anchor_file=str(base/'target_packet_particle_first_fit/fit/coefficients_server_only.json'),
        frozen_models_file=str(registration/'frozen_models_server_only.json'),
        new_full_forward_calls=0, new_canonical_feature_evaluations=0, new_native_starts=0,
        new_fits=0, optimizer_invocations=0, coefficient_arrays_read=False,
        prepared_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'prepared_server_only.json', dict(receipt, cases=tasks))
    save(output/'prepared.json', receipt)
    return tasks, kernels, receipt
