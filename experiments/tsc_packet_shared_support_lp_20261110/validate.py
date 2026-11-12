"""Validate the single candidate beta, never solve or change coefficients."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from math import exp, log, pi, sqrt
import multiprocessing
from pathlib import Path
import traceback

import numpy as np

from experiments.tsc_packet_shared_motion_20261110.api import prepare_plan, evaluate_shared, shared_decisions
from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver
from experiments.tsc_packet_discrete_braking_20261110.conditional_features import prepare_feature_kernel, prepare_receiving_context, _lane_mask
from experiments.tsc_packet_lateral_conditioning_20261110.order import forced_lateral_order
from experiments.tsc_target_packet_training_20261109.training_data import iter_transitions, RESERVOIR_SEED
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from .solve import save


def replay_registered(kernel, plan, beta, registration, affine_endpoints):
    context = kernel._mean_alignment_context
    resolver = MotionResolver(context, kernel, shared_decisions(kernel, plan['phi'], beta), context['permit_all'], focal_id=plan['focal_id'])
    resolver.resolve(plan['focal_id'])
    for key in plan['participants']:
        response = resolver.resolve(key)
        _lane_mask(kernel, kernel.state['packets'][key], kernel._intent(kernel.state['packets'][key]), response, resolver)
    for key in plan['local']:
        resolver.resolve(key)
    projected = {key: resolver.results[key]['projected_packet'] for key in plan['local']}
    ordered = forced_lateral_order(kernel, projected, registration['actions'], context['packet_order'])
    checks = []
    for actor_id, actor in registration['actors'].items():
        key = actor['packet_id']
        if not np.allclose(plan['phi'][key], actor['phi'], rtol=0., atol=1e-12):
            raise ValueError('The registered canonical feature row changed')
        end = affine_endpoints[actor_id]
        lp_x = end['constant_m']+float(np.asarray(end['beta_coefficients'])@beta)
        response = resolver.results[key]
        checks.append(dict(packet_id=key, registered_cell=actor['cell'], score=float(plan['phi'][key]@beta),
            lp_affine_endpoint_m=lp_x, actual_endpoint_m=projected[key]['position_m'],
            accepted_minus_lp_endpoint_m=projected[key]['position_m']-lp_x,
            raw_proposal_m=response['physical_proposal_mps'], accepted_motion_m=response['travel_m'],
            body_slack_m=response['same_step_path_bound_m']-response['physical_proposal_mps'],
            spatial_slack_m=response['spatial_limit_m']-response['physical_proposal_mps']))
    return dict(all_registered_actions_accepted=ordered['all_requested_actions_accepted'], steps=ordered['steps'], actor_checks=checks)


def inspect_seed(task):
    seed, cases, cache, shared, static, laws, width, beta, registered, endpoints, output = task
    directory = cache/'training'/f'seed{seed}'
    executors = json.loads((directory/'executor_seed_server_only.json').read_text())
    table = build_entity_geometry(RoadTopology(static), width)
    selected = {int(c['reference']['time_sec']): c for c in cases}
    rows = []
    for frame, unused_next, action in iter_transitions(directory):
        t = int(frame['time_s'])
        if t not in selected:
            continue
        old, ref = selected[t], selected[t]['reference']
        pid, other = ref['packet_id'], old['neighbour_id']
        kernel = prepare_feature_kernel(static, frame, executors, laws, action['requests'], RESERVOIR_SEED+seed*100000+t, conflict_geometry=table)
        if kernel.signals != action['applied_states']:
            raise ValueError('The frozen current interval signal must reproduce')
        baseline_file = shared/'output_v2'/f'seed{seed}_t{t}_points_server_only.json'
        baseline = json.loads(baseline_file.read_text())
        point_baseline = json.loads((shared/'output_v2/result.json').read_text())['cases']
        label_source = next(c for c in point_baseline if c['reference']==ref)
        mass, geometry = np.zeros(3), np.zeros(3)
        pair_points, normal_before, normal_after, joint_after = [], 0., 0., 0.
        native_speed = label_source['native_motion_label_mps']
        branch_replay = None
        for zf, pf in old['focal_prior'].items():
            for zn, pn in old['neighbour_prior'].items():
                if pf <= 0 or pn <= 0:
                    continue
                for key, goal in ((pid, zf), (other, zn)):
                    road = kernel.topology.lanes[frame['packets'][key]['lane_id']]['edge_id']
                    kernel.state['intents'][key] = dict(road_id=road, choice=goal)
                    kernel._connections[key] = kernel._connection(kernel.state['packets'][key])
                prepare_receiving_context(kernel)
                plan = prepare_plan(kernel, pid, old['participant_ids'])
                point = evaluate_shared(kernel, plan, beta)
                weight = pf*pn
                previous = baseline[len(pair_points)]
                if previous['weight'] != weight:
                    raise ValueError('The cached motion comparator prior order changed')
                mu_before = previous['points'][0]['motion'][pid]['mean_mps']
                mu_after = point['motion'][pid]['mean_mps']
                n_before = exp(-.5*(native_speed-mu_before)**2)/sqrt(2*pi)
                n_after = exp(-.5*(native_speed-mu_after)**2)/sqrt(2*pi)
                normal_before += weight*n_before
                normal_after += weight*n_after
                joint_after += weight*n_after*point['class_mass'][int(ref['y'])]
                mass += weight*point['class_mass']
                geometry += weight*point['endpoint_class_mass']
                pair_points.append(dict(focal_intent=zf, neighbour_intent=zn, prior_product=weight,
                    class_mass=point['class_mass'], endpoint_class_mass=point['endpoint_class_mass'],
                    motion=point['motion'], decision_count=point['decision_count'],
                    negative_background_count=point['negative_residual_background_count'],
                    resolved_background_count=point['resolved_background']))
                registration = registered[pid]
                if zf == registration['focal_intent'] and zn == registration['neighbour_intent']:
                    branch_replay = replay_registered(kernel, plan, beta, registration, endpoints)
        label = int(ref['y'])
        row = dict(reference=ref, prior_goal_pairs=len(pair_points),
            zero_beta_class_mass=old['intent_marginal_zero_residual_observed_class_mass'],
            candidate_class_mass=float(mass[label]), candidate_endpoint_class_mass=float(geometry[label]),
            conditional_class_total=float(mass.sum()), endpoint_compatible_total=float(geometry.sum()),
            all_goal_class_totals_normalized=all(abs(float(p['class_mass'].sum())-1.)<=1e-10 for p in pair_points),
            all_goal_endpoint_totals_normalized=all(abs(float(p['endpoint_class_mass'].sum())-1.)<=1e-10 for p in pair_points),
            motion_nll_before=-log(normal_before), motion_nll_after=-log(normal_after),
            candidate_joint_nll=-log(joint_after) if joint_after>0 else None,
            native_motion_label_mps=native_speed, registered_branch_replay=branch_replay,
            decision_count_range=[min(p['decision_count'] for p in pair_points), max(p['decision_count'] for p in pair_points)])
        save(output/f'seed{seed}_t{t}_all_pairs_server_only.json', pair_points)
        save(output/f'seed{seed}_t{t}_summary.json', row)
        rows.append(row)
        print(json.dumps(dict(stage='single_candidate_case_validated', reference=ref,
            class_mass=row['candidate_class_mass'], endpoint_mass=row['candidate_endpoint_class_mass'],
            endpoint_total=row['endpoint_compatible_total'])), flush=True)
    if len(rows) != len(cases):
        raise ValueError('All9 original states must be validated')
    return rows


def run(cache, lateral, shared, registration, lp, preflight, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        solver = json.loads((lp/'summary.json').read_text())
        if not solver['candidate_available']:
            raise ValueError('The sole solver call returned no candidate; no validation or second solve is permitted')
        beta = np.asarray(json.loads((lp/'candidate_beta_server_only.json').read_text())['beta'])
        registered = {c['reference']['packet_id']:c for c in json.loads(registration.read_text())['cases']}
        endpoints = json.loads((lp/'matrix_definition.json').read_text())['actor_endpoints']
        cases = json.loads((lateral/'output_intent_marginal/summary.json').read_text())['cases']
        width, unused = verify_entity_preflight(preflight)
        static, unused = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        grouped = defaultdict(list)
        for c in cases:
            grouped[int(c['reference']['seed'])].append(c)
        tasks = [(seed, rows, cache, shared, static, laws, width, beta, registered, endpoints, output) for seed, rows in sorted(grouped.items())]
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
            rows = [row for group in pool.map(inspect_seed, tasks) for row in group]
        support = all(c['candidate_endpoint_class_mass']>0 for c in rows)
        normalized = all(c['all_goal_class_totals_normalized'] and c['all_goal_endpoint_totals_normalized'] for c in rows)
        passed = solver['solver_success'] and support and normalized
        result = dict(status='PASS_LOCAL_SHARED_SUPPORT_CANDIDATE' if passed else 'HOLD_SINGLE_SUPPORT_CANDIDATE_VALIDATION',
            cases=rows, validated_goal_pairs=sum(c['prior_goal_pairs'] for c in rows),
            positive_endpoint_class_cases=sum(c['candidate_endpoint_class_mass']>0 for c in rows),
            all_conditionals_normalized=normalized, solver_success=solver['solver_success'], solver_status=solver['solver_status'],
            motion_nll_sum_before=sum(c['motion_nll_before'] for c in rows), motion_nll_sum_after=sum(c['motion_nll_after'] for c in rows),
            new_fits=1, solver_calls=1, new_native_starts=0, calibration_data_used=False,
            model_admitted=False, full_reservoir_or_control_gain_claimed=False,
            scope='Exactly the one fixed-cell parameter estimate, validated through the original all-frame shared API on the9 selected training states. Not a full-data fit, smoothness guarantee, independent evaluation or control result.')
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_SINGLE_SUPPORT_CANDIDATE_VALIDATION', error=f'{type(error).__name__}: {error}', new_fits=1, solver_calls=1, new_native_starts=0)
    save(output/'summary.json', result)
    print(json.dumps(dict(status=result['status'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache','lateral','shared','registration','lp','preflight','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.cache,args.lateral,args.shared,args.registration,args.lp,args.preflight,args.output)
