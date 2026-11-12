"""Register the archived 904 matched pairs before enumerating any response."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from math import prod

from experiments.tsc_packet_conditional_domain_20261110.budget import symbolic_budget
from experiments.tsc_packet_conditional_domain_20261110.register import RESERVOIR_SEED, selected_current_records
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_conditional_panel_20261110.initialize import prepare_zero_kernel
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_lane_change_coverage_20261111.census import (
    DISTANCE_TOLERANCE, SPEED_TOLERANCE, current_stratum, vehicle,
)
from experiments.tsc_packet_mixture_confirmation_20261111.evaluate import read_json
from experiments.tsc_packet_real_particle_pilot_20261110.evaluate import pool_eligibility
from experiments.tsc_packet_real_particle_pilot_20261110.register import reference_key, save
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


SEED = 76116904
PAIR_COUNTS = {0: 32, 2: 30}
FULL_FORWARD_CEILING = 2444
LEAF_CEILING = 31008
MAX_STOCHASTIC = 10


def frozen_tasks(base, cache):
    coverage = base / 'target_packet_lane_change_coverage' / 'coverage'
    result = read_json(coverage / 'result.json')
    if (result['status'] != 'COMPLETE_ARCHIVED_LC_COVERAGE'
            or result['current_coefficient_holdout_seed'] != SEED):
        raise ValueError('The completed 904 lane-change coverage archive is required')
    pairs = [pair for pair in read_json(coverage / result['matched_pairs_file'])
             if pair['positive']['seed'] == SEED]
    counts = Counter(pair['positive']['y'] for pair in pairs)
    if counts != PAIR_COUNTS:
        raise ValueError('The frozen 904 roster must contain 32 left and 30 right pairs')
    manifest = read_json(cache / 'training_reservoir_server_only.json')[str(SEED)]['lane_change']
    observations = {reference_key(row): (i, row['y']) for i, row in enumerate(manifest)}
    fit = read_json(base / 'target_packet_particle_first_fit/preflight/cohort.json')['references']
    pilot = read_json(base / 'target_packet_real_particle_pilot/evaluation/summary.json')['cases']
    excluded = {vehicle(r) for r in fit} | {vehicle(r['reference']) for r in pilot}
    positive_vehicles = {vehicle(r) for r in manifest if r['y'] != 1}
    used, tasks = set(), []
    for pair_index, pair in enumerate(pairs):
        direction = pair['positive']['y']
        for role, label in (('positive', direction), ('stay', 1)):
            row = pair[role]
            ref = reference_key(row)
            index = len(tasks)
            if (ref[0] != SEED or ref[1] != row['time_sec'] or label != row['y']
                    or vehicle(row) in used or vehicle(row) in excluded
                    or (role == 'stay' and vehicle(row) in positive_vehicles)
                    or ref not in observations or observations[ref][1] != label):
                raise ValueError(f'Invalid archived 904 {role} reference in pair {pair_index}')
            used.add(vehicle(row))
            tasks.append(dict(case_index=index, pair_index=pair_index, role=role,
                reference=dict(zip(('seed', 'time_sec', 'packet_id'), ref)), time_sec=ref[1],
                stem=f'case{index:03d}_seed{SEED}_t{ref[1]}', direction=direction, y=label,
                native_lc_label=label, observations=[dict(head='lane_change', y=label,
                manifest_ordinal=observations[ref][0])]))
    return pairs, tasks, result['prior_seed_seconds']


def census_rows(base, tasks):
    path = base / 'target_packet_conditional_census/output' / f'seed{SEED}_cases_server_only.jsonl.gz'
    wanted = {reference_key(task['reference']) for task in tasks}
    rows = {}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = reference_key(row)
            if key in wanted:
                if key in rows:
                    raise ValueError(f'Duplicate original census case {key}')
                rows[key] = row
    if rows.keys() != wanted:
        raise ValueError('The original census must cover all 124 frozen references')
    return rows


def register_pairs(base, output):
    """Return the complete case roster, fork-shareable kernels by time, and a compact receipt."""
    cache = base / 'target_packet_training'
    pairs, tasks, prior_seconds = frozen_tasks(base, cache)
    original = census_rows(base, tasks)
    archive = cache / 'training' / f'seed{SEED}'
    static, _ = load_widths(read_json(cache / 'static_graph_server_only.json'))
    laws = read_json(cache / 'target_laws_server_only.json')
    executors = read_json(archive / 'executor_seed_server_only.json')
    width, _ = verify_entity_preflight(base / 'target_packet_entity_footprint/entity_input_preflight.json')
    topology = RoadTopology(static)
    geometry = build_entity_geometry(topology, width)
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task['time_sec']].append(task)
    kernels, covariates = {}, {}
    with physical_forbidden():
        for frame, action, _ in selected_current_records(archive, grouped):
            t = int(frame['time_s'])
            kernel = prepare_zero_kernel(static, frame, executors, laws, action,
                                         RESERVOIR_SEED + SEED*100000 + t, geometry)
            kernels[t] = kernel
            lanes = defaultdict(list)
            for packet in frame['packets'].values():
                lanes[packet['lane_id']].append(packet)
            for task in grouped[t]:
                ref = reference_key(task['reference'])
                focal = ref[2]
                old = original[ref]
                selection = select_stochastic(kernel, focal, max_stochastic=MAX_STOCHASTIC)
                required = selection['required_participants']
                if (selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE'
                        or required != old['required_participants']
                        or len(required) != old['required_actor_count']
                        or len(kernel.state['packets']) != old['frame_actor_count']
                        or pool_eligibility(kernel, focal) != old['eligibility']
                        or not old['eligibility']['lane_change']):
                    raise ValueError(f'The original whole-J domain or eligibility changed at {ref}')
                budget = symbolic_budget(kernel, selection)
                physical = budget['physical_resolver_calls_upper_bound']
                leaves = budget['lateral_leaf_branches_upper_bound']
                if (budget['status'] != 'REGISTERED_SYMBOLIC_UPPER_BOUND'
                        or physical != old['physical_calls_upper'] or leaves != old['leaf_visits_upper']):
                    raise ValueError(f'The original exact enumeration budget changed at {ref}')
                goals = prod(row['current_positive_goals'] for row in budget['per_actor'].values())
                matched = current_stratum(frame['packets'][focal], task['direction'], len(required),
                                          topology, lanes, action['applied_states'])
                pair = pairs[task['pair_index']]
                if matched is None or json.loads(json.dumps(matched[0])) != pair['current_stratum']:
                    raise ValueError(f'The original matching conditions changed at {ref}')
                covariates[task['case_index']] = matched[1:]
                task.update(required_participants=required, required_actor_count=len(required),
                    frame_actor_count=old['frame_actor_count'], physical_limit=physical,
                    feature_limit=len(required)*goals, leaf_limit=leaves, current_goal_count=goals)
    for positive, stay in zip(tasks[::2], tasks[1::2]):
        pair = pairs[positive['pair_index']]
        speed, distance = (abs(a-b) for a, b in zip(covariates[positive['case_index']],
                                                   covariates[stay['case_index']]))
        if (speed != pair['speed_difference_mps'] or speed > SPEED_TOLERANCE
                or distance != pair['lane_end_distance_difference_m'] or distance > DISTANCE_TOLERANCE):
            raise ValueError(f'The original matching distances changed in pair {positive["pair_index"]}')
    physical = sum(task['physical_limit'] for task in tasks)
    features = sum(task['feature_limit'] for task in tasks)
    leaves = sum(task['leaf_limit'] for task in tasks)
    if physical > FULL_FORWARD_CEILING or leaves > LEAF_CEILING:
        raise ValueError('The 904 roster exceeds the frozen exact enumeration budget')
    receipt = dict(status='PASS_FROZEN_904_PAIR_REGISTRATION', seed=SEED,
        pair_count=len(pairs), direction_pair_counts=PAIR_COUNTS, case_count=len(tasks),
        current_frame_initializations=len(kernels), full_forward_ceiling=physical,
        feature_ceiling=features, leaf_ceiling=leaves,
        actor_count_histogram=dict(Counter(task['required_actor_count'] for task in tasks)),
        prior_seed_seconds=prior_seconds, new_full_forward_calls=0,
        new_canonical_feature_evaluations=0, new_native_starts=0, new_fits=0,
        registered_at_utc=datetime.now(timezone.utc).isoformat())
    save(output / 'registered_server_only.json', dict(receipt, cases=tasks, pairs=pairs))
    save(output / 'registration.json', receipt)
    return tasks, kernels, receipt
