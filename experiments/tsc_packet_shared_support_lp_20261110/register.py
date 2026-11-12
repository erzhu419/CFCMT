"""Read-only registration of nine fixed branches before the single LP call."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import traceback

from experiments.tsc_packet_shared_motion_20261110.api import prepare_plan
from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver
from experiments.tsc_packet_discrete_braking_20261110.conditional_features import prepare_feature_kernel, prepare_receiving_context, _lane_mask
from experiments.tsc_packet_lateral_conditioning_20261110.order import forced_lateral_order
from experiments.tsc_target_packet_training_20261109.training_data import iter_transitions, RESERVOIR_SEED
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths, save


def adjacent_constraints(packets, prefix, stage):
    lanes = defaultdict(list)
    for pid, packet in packets.items():
        lanes[packet['lane_id']].append((packet['position_m'], pid))
    return [dict(label=f'{prefix}:{stage}:{lane}:{rear[1]}->{front[1]}',
                 front=prefix+':'+front[1], rear=prefix+':'+rear[1])
            for lane, rows in sorted(lanes.items()) for rear, front in zip(sorted(rows), sorted(rows)[1:])]


def inspect_seed(task):
    seed, cases, cache, lateral, static, laws, width, conditions, output = task
    directory = cache/'training'/f'seed{seed}'
    executors = json.loads((directory/'executor_seed_server_only.json').read_text())
    table = build_entity_geometry(RoadTopology(static), width)
    selected = {int(case['reference']['time_sec']): case for case in cases}
    rows = []
    for frame, unused_next, action in iter_transitions(directory):
        t = int(frame['time_s'])
        if t not in selected:
            continue
        case, ref = selected[t], selected[t]['reference']
        pid, other = ref['packet_id'], case['neighbour_id']
        condition = conditions.get(pid)
        if condition:
            zf, zn, category = condition['focal_intent'], condition['held_intent'], 'external_residual'
        else:
            first = next(pair for pair in case['intent_pairs'] if pair['zero_residual_observed_class_mass'] > 0)
            zf, zn, category = first['focal_intent'], first['neighbour_intent'], 'zero'
        full = json.loads((lateral/'output_intent_marginal'/f'seed{seed}_t{t}_intent_components_server_only.json').read_text())
        saved = next(c for c in full if c['intent'] == zf and c['held_participant_intents'][other] == zn)
        label = ('left', 'stay', 'right')[int(ref['y'])]
        witness = saved['examples'][label+category]
        kernel = prepare_feature_kernel(static, frame, executors, laws, action['requests'], RESERVOIR_SEED+seed*100000+t, conflict_geometry=table)
        for key, goal in ((pid, zf), (other, zn)):
            road = kernel.topology.lanes[frame['packets'][key]['lane_id']]['edge_id']
            kernel.state['intents'][key] = dict(road_id=road, choice=goal)
            kernel._connections[key] = kernel._connection(kernel.state['packets'][key])
        prepare_receiving_context(kernel)
        plan = prepare_plan(kernel, pid, case['participant_ids'])
        context = kernel._mean_alignment_context
        decisions = dict(context['zero_decisions'])
        decisions.update({key: dict(motion_residual_mps=r) for key, r in witness['branch']['residuals_mps'].items()})
        resolver = MotionResolver(context, kernel, decisions, context['permit_all'], focal_id=pid)
        resolver.resolve(pid)
        for key in plan['participants']:
            result = resolver.resolve(key)
            _lane_mask(kernel, frame['packets'][key], kernel._intent(frame['packets'][key]), result, resolver)
        for key in plan['local']:
            resolver.resolve(key)
        projected = {key: resolver.results[key]['projected_packet'] for key in plan['local']}
        ordered = forced_lateral_order(kernel, projected, witness['actions'], context['packet_order'])
        if not ordered['all_requested_actions_accepted']:
            raise ValueError('The registered existing witness must reproduce its complete lateral sequence')
        old_end = {p['packet_id']: p for p in witness['branch']['projected']}
        if set(old_end) != set(projected) or any(
                ordered['packets'][key]['lane_id'] != old_end[key]['lane_id'] or
                abs(ordered['packets'][key]['position_m']-old_end[key]['position_m']) > 1e-8 for key in projected):
            raise ValueError('Every stored joint endpoint must reproduce')
        prefix = f'{seed}:{t}:{pid}'
        actors, gaps, cap_differences = {}, adjacent_constraints(projected, prefix, 'before_LC'), []
        for key in plan['local']:
            packet, result = frame['packets'][key], resolver.results[key]
            p, P = packet['parameters'], plan['P'][key]
            r = decisions[key]['motion_residual_mps']
            cell = 'plateau' if r == 0 else 'linear' if P+r > 0 else 'zero_floor'
            raw_expected = max(P+min(r, 0.), 0.)
            actors[prefix+':'+key] = dict(phi=plan['phi'][key].tolist(), P=P, x=packet['position_m'],
                length=p['length_m'], min_gap=p['min_gap_m'], b=p['decel_mps2'], cell=cell,
                packet_id=key, witness_residual_mps=r, witness_accepted_motion_m=result['travel_m'],
                witness_proposal_motion_m=raw_expected, current_lane=packet['lane_id'])
            if abs(result['travel_m']-raw_expected) > 1e-8:
                cap_differences.append(dict(packet_id=key, raw_proposal_m=raw_expected,
                    accepted_m=result['travel_m'], body_bound_m=result['same_step_path_bound_m'],
                    spatial_bound_m=result['spatial_limit_m'], binding_body_ids=result['receiving_binding_summary']['packet_ids']))
        working = deepcopy(projected)
        for key in sorted(witness['actions'], key=context['packet_order'].__getitem__):
            choice = witness['actions'][key]
            if choice != 'stay':
                working[key]['lane_id'] = kernel.topology.adjacent(working[key]['lane_id'], 1 if choice == 'left' else -1)
            gaps.extend(adjacent_constraints(working, prefix, 'after_'+key+'_'+choice))
        gaps.extend(adjacent_constraints(working, prefix, 'final'))
        row = dict(reference=ref, focal_intent=zf, neighbour_intent=zn, selection_category=category,
            prior_product=saved['focal_prior']*saved['neighbour_prior'], actions=witness['actions'],
            participant_order={key: context['packet_order'][key] for key in plan['participants']},
            actors=actors, gaps=gaps, witness_cap_differences=cap_differences,
            witness_reproduced=True, raw_affine_witness_matches=not cap_differences)
        rows.append(row)
        print(json.dumps(dict(stage='branch_registered', reference=ref, local_actors=len(actors),
            gap_constraints=len(gaps), additional_caps=len(cap_differences))), flush=True)
    if len(rows) != len(cases):
        raise ValueError('Every original9 state must have a registered branch')
    save(output/f'seed{seed}_registered_server_only.json', rows)
    return rows


def run(cache, lateral, preflight, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        cases = json.loads((lateral/'output_intent_marginal/summary.json').read_text())['cases']
        conditions = {c['reference']['packet_id']: c for c in json.loads((lateral/'support_conditions.json').read_text())['cases']}
        width, unused = verify_entity_preflight(preflight)
        static, unused = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        grouped = defaultdict(list)
        for case in cases:
            grouped[int(case['reference']['seed'])].append(case)
        tasks = [(seed, rows, cache, lateral, static, laws, width, conditions, output) for seed, rows in sorted(grouped.items())]
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
            rows = [row for group in pool.map(inspect_seed, tasks) for row in group]
        result = dict(status='COMPLETE_BRANCH_REGISTRATION', cases=rows,
            raw_affine_witness_matches_all=all(c['raw_affine_witness_matches'] for c in rows),
            solver_calls=0, new_fits=0, new_native_starts=0)
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_BRANCH_REGISTRATION', error=f'{type(error).__name__}: {error}', solver_calls=0, new_fits=0, new_native_starts=0)
    save(output/'registration_server_only.json', result)
    print(json.dumps(dict(status=result['status'], raw_affine=result.get('raw_affine_witness_matches_all'))), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache', 'lateral', 'preflight', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cache, args.lateral, args.preflight, args.output)
    raise SystemExit(0 if result['status'] == 'COMPLETE_BRANCH_REGISTRATION' else 2)
