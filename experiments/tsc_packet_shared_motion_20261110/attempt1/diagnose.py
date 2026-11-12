"""Supplied shared-beta directional audit on the same nine existing states."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from math import exp, log, pi, sqrt
import multiprocessing
from pathlib import Path
import traceback

import numpy as np

from experiments.tsc_packet_discrete_braking_20261110.conditional_features import prepare_feature_kernel, prepare_receiving_context
from experiments.tsc_target_packet_training_20261109.training_data import iter_transitions, RESERVOIR_SEED
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths, save
from .api import prepare_plan, evaluate_shared
from .algebra import shared_gap


def probes():
    result = [('zero', np.zeros(38), None, None)]
    for feature in ('bias', 'speed_mps', 'acceleration_mps2'):
        for sign in (-1, 1):
            direction = np.zeros(38)
            direction[FEATURE_NAMES.index(feature)] = sign
            for h in (1e-4, 5e-5):
                result.append((f'{feature}:{sign}:{h}', h*direction, direction, h))
    return result


def actor(kernel, plan, pid):
    packet = kernel.state['packets'][pid]
    p = packet['parameters']
    return dict(phi=plan['phi'][pid].tolist(), b=p['decel_mps2'], a=p['accel_mps2'],
        P=plan['P'][pid], x=packet['position_m'], length=p['length_m'], min_gap=p['min_gap_m'])


def accepted_margin(f, n, role, result, pid, other):
    mf, mn = result['motion'][pid]['mean_mps'], result['motion'][other]['mean_mps']
    return (f['x']+mf-f['length']-n['x']-mn-n['min_gap'] if role == 'rear' else
            n['x']+mn-n['length']-f['x']-mf-f['min_gap'])


def strict_raw_response(result, pid):
    row = result['motion'][pid]
    return row['mean_matches_proposal'] and min(row['body_slack_m'], row['spatial_slack_m']) > 1e-8


def aggregate(components, ref, native_speed):
    label, pid = int(ref['y']), ref['packet_id']
    output = []
    for index, (name, beta, direction, h) in enumerate(probes()):
        mass, endpoint_mass = np.zeros(3), np.zeros(3)
        density, grad_l, grad_r = 0., np.zeros(38), np.zeros(38)
        mask_same, affine, numerator = True, True, 0.
        decisions, negative, resolved = [], [], []
        for c in components:
            result, base, weight = c['points'][index], c['points'][0], c['weight']
            mu = result['motion'][pid]['mean_mps']
            normal = exp(-.5*(native_speed-mu)**2)/sqrt(2*pi)
            mass += weight*result['class_mass']
            endpoint_mass += weight*result['endpoint_class_mass']
            density += weight*normal*result['class_mass'][label]
            grad_l += weight*normal*result['grad_mass_left'][label]
            grad_r += weight*normal*result['grad_mass_right'][label]
            mask_same &= result['lateral_signature'] == base['lateral_signature']
            affine &= strict_raw_response(result, pid) and strict_raw_response(base, pid)
            if direction is not None:
                dmu = min(float(direction@np.asarray(c['focal']['phi'])), 0.)
                mu0 = base['motion'][pid]['mean_mps']
                normal0 = exp(-.5*(native_speed-mu0)**2)/sqrt(2*pi)
                numerator += weight*normal0*base['class_mass'][label]*(mu0-native_speed)*dmu
            decisions.append(result['decision_count'])
            negative.append(result['negative_residual_background_count'])
            resolved.append(result['resolved_background'])
        row = dict(name=name, h=h, class_mass=mass.tolist(), endpoint_class_mass=endpoint_mass.tolist(),
            observed_class_mass=float(mass[label]), joint_density=density,
            joint_nll=-log(density) if density > 0 else None,
            joint_status='FINITE' if density > 0 else 'ZERO_DENSITY_UNDEFINED_LOG_GRADIENT',
            masks_unchanged_from_zero=bool(mask_same), focal_strict_raw_response=bool(affine),
            decision_count_range=[min(decisions), max(decisions)],
            negative_background_count_range=[min(negative), max(negative)],
            resolved_background_count_range=[min(resolved), max(resolved)])
        if index == 0:
            row['gamma_nll_gradient_left'] = (-grad_l/density).tolist() if density > 0 else None
            row['gamma_nll_gradient_right'] = (-grad_r/density).tolist() if density > 0 else None
            row['ordinary_beta_gradient_available'] = False
        else:
            zero = output[0]
            row['joint_nll_one_sided_difference'] = ((row['joint_nll']-zero['joint_nll'])/h
                if row['joint_nll'] is not None and zero['joint_nll'] is not None else None)
            row['joint_nll_analytic_direction_if_fixed_branch'] = (numerator/zero['joint_density']
                if zero['joint_density'] > 0 and mask_same and affine else None)
        output.append(row)
    return output


def inspect_seed(task):
    seed, cases, cache, static, laws, width, conditions, output = task
    directory = cache/'training'/f'seed{seed}'
    executors = json.loads((directory/'executor_seed_server_only.json').read_text())
    table = build_entity_geometry(RoadTopology(static), width)
    selected = {int(c['reference']['time_sec']): c for c in cases}
    rows = []
    for frame, following, action in iter_transitions(directory):
        t = int(frame['time_s'])
        if t not in selected:
            continue
        old, ref = selected[t], selected[t]['reference']
        pid, other = ref['packet_id'], old['neighbour_id']
        kernel = prepare_feature_kernel(static, frame, executors, laws, action['requests'], RESERVOIR_SEED+seed*100000+t, conflict_geometry=table)
        if kernel.signals != action['applied_states']:
            raise ValueError('Current request/executor signal must reproduce')
        participants = old['participant_ids']
        components, pair_checks, arrays, representative = [], [], {}, None
        expected_pairs = {(p['focal_intent'], p['neighbour_intent']): p for p in old['intent_pairs']}
        for zf, pf in old['focal_prior'].items():
            for zn, pn in old['neighbour_prior'].items():
                if pf <= 0 or pn <= 0:
                    continue
                for key, goal in ((pid, zf), (other, zn)):
                    road = kernel.topology.lanes[frame['packets'][key]['lane_id']]['edge_id']
                    kernel.state['intents'][key] = dict(road_id=road, choice=goal)
                    kernel._connections[key] = kernel._connection(kernel.state['packets'][key])
                prepare_receiving_context(kernel)
                plan = prepare_plan(kernel, pid, participants)
                points = [evaluate_shared(kernel, plan, beta) for name, beta, direction, h in probes()]
                if abs(points[0]['class_mass'][int(ref['y'])]-expected_pairs[(zf, zn)]['zero_residual_observed_class_mass']) > 1e-8:
                    raise ValueError('The new all-actor beta0 API must reproduce each saved intent-pair class mass')
                f, n = actor(kernel, plan, pid), actor(kernel, plan, other)
                components.append(dict(weight=pf*pn, focal=f, points=points))
                pair_checks.append(dict(focal_intent=zf, neighbour_intent=zn, packet_count=len(plan['phi']), baseline_reproduced=True))
                prefix='goal'+str(len(pair_checks))
                arrays[prefix+'_packet_ids'] = np.asarray(list(plan['phi']))
                arrays[prefix+'_phi'] = np.asarray(list(plan['phi'].values()))
                condition = conditions.get(pid)
                if condition and zf == condition['focal_intent'] and zn == condition['held_intent']:
                    role = condition['role']
                    baseline_margin = accepted_margin(f, n, role, points[0], pid, other)
                    if abs(baseline_margin+condition['deficit_m']) > 1e-8:
                        raise ValueError('The registered representative gap condition must reproduce')
                    checks = []
                    for index, (name, beta, direction, h) in enumerate(probes()):
                        if direction is None:
                            continue
                        algebra = shared_gap(np.zeros(38), f, n, role, direction=direction)
                        predicted = shared_gap(beta, f, n, role)
                        margin = accepted_margin(f, n, role, points[index], pid, other)
                        clean = all(strict_raw_response(points[k], key) for k in (0, index) for key in (pid, other))
                        checks.append(dict(name=name, h=h, gap_margin_m=margin,
                            numerical_gap_direction=(margin-baseline_margin)/h,
                            proposal_algebra_direction=algebra['gap_directional'],
                            both_actor_strict_raw_response=bool(clean),
                            numerical_hinge_direction=(max(-margin, 0.)**2-max(-baseline_margin, 0.)**2)/h,
                            proposal_hinge_direction=algebra['squared_hinge_directional'],
                            accepted_minus_proposal_margin_m=margin-predicted['gap_margin_m']))
                    representative = dict(role=role, focal=f, neighbour=n, baseline_margin_m=baseline_margin, checks=checks)
        # Predictions/derivatives above use current packets only. Read the next
        # native speed solely now as the Gaussian working-density label.
        native_speed = following['packets'][pid]['speed_mps']
        points = aggregate(components, ref, native_speed)
        if abs(points[0]['observed_class_mass']-old['intent_marginal_zero_residual_observed_class_mass']) > 1e-8:
            raise ValueError('The marginal beta0 control must reproduce')
        np.savez_compressed(output/f'seed{seed}_t{t}_all_packet_phi_server_only.npz', **arrays)
        save(output/f'seed{seed}_t{t}_points_server_only.json', [dict(weight=c['weight'],
            points=[{key: value.tolist() if isinstance(value, np.ndarray) else value
                     for key, value in v.items() if key != 'lateral_signature'} for v in c['points']]) for c in components])
        row = dict(reference=ref, native_motion_label_mps=native_speed, point_readouts=points,
            intent_pair_checks=pair_checks, representative_gap=representative,
            normalized_all_point_masses=all(abs(sum(p['class_mass'])-1.) < 1e-10 for p in points))
        rows.append(row)
        print(json.dumps(dict(stage='shared_beta_case_complete', reference=ref,
            joint_zero_status=points[0]['joint_status'], p0=points[0]['observed_class_mass'])), flush=True)
    if len(rows) != len(cases):
        raise ValueError('Every original fixed case must be consumed')
    return rows


def run(cache, baseline, condition_file, preflight, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        cases = json.loads(baseline.read_text())['cases']
        if len(cases) != 9:
            raise ValueError('Exactly the same9 cases are required')
        conditions = {c['reference']['packet_id']: c for c in json.loads(condition_file.read_text())['cases']}
        width, unused = verify_entity_preflight(preflight)
        static, unused = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        grouped = defaultdict(list)
        for case in cases:
            grouped[int(case['reference']['seed'])].append(case)
        tasks = [(seed, rows, cache, static, laws, width, conditions, output) for seed, rows in sorted(grouped.items())]
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
            rows = [row for group in pool.map(inspect_seed, tasks) for row in group]
        result = dict(status='COMPLETE_SHARED_BETA_SUPPORT_DIFFERENTIAL_AUDIT', cases=rows,
            baseline_pair_controls=sum(len(c['intent_pair_checks']) for c in rows),
            finite_zero_joint_cases=sum(c['point_readouts'][0]['joint_density'] > 0 for c in rows),
            coefficient_vector_estimated=False, general_190_objective_implemented=False,
            new_native_starts=0, new_fits=0, new_nuisance_estimations=0, optimizer_calls=0,
            calibration_data_used=False, production_operator_changed=False,
            scope='Local normalized Normal(native speed; shared-resolver mean,1) times ordered focal LC probability, intent-marginalized. Background longitudinal responses use the same supplied beta, background lateral actions do(stay). This is not a whole-network190-parameter likelihood or a fit. At beta0 clipping kinks require one-sided derivatives; zero joint density has no finite NLL gradient.')
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_SHARED_BETA_SUPPORT_DIFFERENTIAL_AUDIT', error=f'{type(error).__name__}: {error}', new_native_starts=0, new_fits=0)
    save(output/'result.json', result)
    print(json.dumps(dict(status=result['status'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache', 'baseline', 'conditions', 'preflight', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cache, args.baseline, args.conditions, args.preflight, args.output)
    raise SystemExit(0 if result['status'] == 'COMPLETE_SHARED_BETA_SUPPORT_DIFFERENTIAL_AUDIT' else 2)
