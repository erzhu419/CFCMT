"""Nine fixed LC cases: finite joint interventions, never a fitted likelihood."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
from itertools import product
import json
import multiprocessing
from pathlib import Path
import traceback

from experiments.tsc_packet_discrete_braking_20261110.conditional_features import (
    prepare_feature_kernel, candidate_features, _lane_mask,
)
from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver
from experiments.tsc_target_packet_training_20261109.training_data import iter_transitions, _candidates, RESERVOIR_SEED
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths, save
from .feasibility import screen_endpoint_boxes
from .order import forced_lateral_order


CHOICES = ('left', 'stay', 'right')


def deniers(packet, candidate):
    """Current forecast blockers of both lawful sides; no observed choice input."""
    detail, result = candidate['lane_detail'], set()
    x = detail['projected_position_m']
    for side in detail['sides'].values():
        if not side.get('potential_legal'):
            continue
        neighbours = side.get('neighbour_states', [])
        for other in neighbours:
            if other['position_m'] >= x and other['position_m']-other['length_m']-x < packet['parameters']['min_gap_m']:
                result.add(other['packet_id'])
        rear = [other for other in neighbours if other['position_m'] < x]
        if rear:
            other = max(rear, key=lambda row: row['position_m'])
            if x-packet['parameters']['length_m']-other['position_m'] < other['min_gap_m']:
                result.add(other['packet_id'])
    return result


def metadata(packet, P):
    return dict(packet_id=packet['packet_id'], current_lane=packet['lane_id'], current_x_m=packet['position_m'],
        length_m=packet['parameters']['length_m'], min_gap_m=packet['parameters']['min_gap_m'],
        physical_P_mps=P, b_mps2=packet['parameters']['decel_mps2'], a_mps2=packet['parameters']['accel_mps2'])


def endpoint(packet):
    return dict(packet_id=packet['packet_id'], lane_id=packet['lane_id'], position_m=packet['position_m'],
                length_m=packet['parameters']['length_m'], min_gap_m=packet['parameters']['min_gap_m'])


def inspect_intent(kernel, pid, intent, participants, reference, out):
    context, packets = kernel._mean_alignment_context, kernel.state['packets']
    connection, compatible = kernel._connections[pid]
    base = MotionResolver(context, kernel, dict(context['zero_decisions']), context['permit_all'], focal_id=pid)
    road = kernel.topology.lanes[packets[pid]['lane_id']]['edge_id']
    on_road = [key for key, row in packets.items() if kernel.topology.lanes[row['lane_id']]['edge_id'] == road]
    metas = {key: metadata(packets[key], base.raw(key)) for key in on_road}
    # Include every road actor whose outer endpoint could violate a participant's
    # landing gap, whichever same-road adjacent lane it selects. Other bodies
    # still participate in the unmodified longitudinal dependency resolver.
    local = []
    for key, other in metas.items():
        for actor in participants:
            own = metas[actor]
            margin = max(own['length_m']+other['min_gap_m'], other['length_m']+own['min_gap_m'])
            if (other['current_x_m'] <= own['current_x_m']+own['physical_P_mps']+margin and
                    own['current_x_m'] <= other['current_x_m']+other['physical_P_mps']+margin):
                local.append(key)
                break
    local = sorted(local, key=context['packet_order'].__getitem__)
    for key in local:
        packet = packets[key]
        if not (packet['position_m'] > packet['parameters']['length_m'] and
                packet['position_m']+metas[key]['physical_P_mps'] < kernel.topology.lanes[packet['lane_id']]['length_m']):
            raise ValueError('This fixed interior-road audit must not silently approximate crossing/tail cases')
    stats = {choice: Counter() for choice in CHOICES}
    examples, boxes, conditional_readouts = {}, {}, []
    for bits in product((0, 1), repeat=len(participants)):
        residuals = {key: -packets[key]['parameters']['decel_mps2']*bit for key, bit in zip(participants, bits)}
        decisions = dict(context['zero_decisions'])
        decisions.update({key: dict(motion_residual_mps=value) for key, value in residuals.items()})
        resolver = MotionResolver(context, kernel, decisions, context['permit_all'], focal_id=pid)
        resolver.resolve(pid)
        masks, options = {}, {}
        for key in participants:
            result = resolver.resolve(key)
            current_intent = kernel._intent(packets[key])
            masks[key], unused = _lane_mask(kernel, packets[key], current_intent, result, resolver)
            projected = result['projected_packet']
            options[key] = [('stay', projected['lane_id'])]
            for choice, delta in (('left', 1), ('right', -1)):
                target = kernel.topology.adjacent(projected['lane_id'], delta)
                if kernel._lateral_legal(projected, target) and projected['position_m'] <= kernel.topology.lanes[target]['length_m']:
                    options[key].append((choice, target))
        for key in local:
            result = resolver.resolve(key)
            if result['ended'] or kernel.topology.lanes[result['projected_packet']['lane_id']]['edge_id'] != road:
                raise ValueError('A selected local actor unexpectedly left the audited road')
        projected = {key: resolver.results[key]['projected_packet'] for key in local}
        class_mass = dict.fromkeys(CHOICES, 0.)
        endpoint_mass = dict.fromkeys(CHOICES, 0.)
        for actions_and_lanes in product(*(options[key] for key in participants)):
            actions = {key: pair[0] for key, pair in zip(participants, actions_and_lanes)}
            own_choice, target = actions[pid], dict(zip(participants, actions_and_lanes))[pid][1]
            after = deepcopy(projected)
            for key, pair in zip(participants, actions_and_lanes):
                after[key]['lane_id'] = pair[1]
            branch_id = intent+':'+''.join(map(str, bits))+':'+','.join(actions.values())
            branch = dict(branch_id=branch_id, residuals_mps=residuals, projected=[endpoint(row) for row in after.values()],
                topology_goal_legal=True, pre_lateral_mask_legal=all(masks[key][actions[key]] for key in participants))
            screened = screen_endpoint_boxes(metas[pid], [metas[key] for key in local if key != pid], target_lane=target, branches=[branch])
            checked = screened['supplied_joint_branch_checks'][0]
            ordered = forced_lateral_order(kernel, projected, actions, context['packet_order'])
            sequential = checked['topology_legal_endpoint_witness'] and ordered['all_requested_actions_accepted']
            mass = ordered['zero_logit_tuple_mass']
            class_mass[own_choice] += mass
            endpoint_mass[own_choice] += mass*checked['topology_legal_endpoint_witness']
            zero = not any(bits)
            stat = stats[own_choice]
            stat['lateral_tuples'] += 1
            stat['endpoint_witnesses'] += int(checked['topology_legal_endpoint_witness'])
            stat['pre_lateral_mask_and_endpoint_witnesses'] += int(checked['endpoints_and_declared_masks_pass'])
            stat['ordered_execution_witnesses'] += int(sequential)
            stat['zero_residual_endpoint_witnesses'] += int(zero and checked['topology_legal_endpoint_witness'])
            stat['zero_residual_ordered_witnesses'] += int(zero and sequential)
            stat['ordered_with_other_lateral_actions'] += int(sequential and any(actions[key] != 'stay' for key in participants if key != pid))
            stat['ordered_focal_brake_only'] += int(sequential and residuals[pid] < 0 and all(residuals[key] == 0 for key in participants if key != pid))
            stat['ordered_neighbour_brake_only'] += int(sequential and residuals[pid] == 0 and any(residuals[key] < 0 for key in participants if key != pid))
            record = dict(reference=reference, branch=branch, actions=actions, endpoint_check=checked,
                sequential_check={key: ordered[key] for key in ('all_requested_actions_accepted', 'steps', 'zero_logit_tuple_mass')}, ordered_witness=bool(sequential))
            out.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
            category = own_choice+('zero' if zero else 'external_residual')
            if sequential and category not in examples:
                examples[category] = record
            if checked['topology_legal_endpoint_witness'] and own_choice+'_endpoint' not in examples:
                examples[own_choice+'_endpoint'] = record
            if own_choice not in boxes:
                boxes[own_choice] = {key: value for key, value in screened.items() if key != 'supplied_joint_branch_checks'}
        total_mass = sum(class_mass.values())
        if abs(total_mass-1.) > 1e-10:
            raise ValueError('The complete static action tuples must normalize the sequential conditional LC law')
        conditional_readouts.append(dict(residuals_mps=residuals, zero_motion_residual=not any(bits),
            total_mass=total_mass, lateral_class_mass=class_mass,
            endpoint_feasible_class_mass=endpoint_mass, endpoint_feasible_total_mass=sum(endpoint_mass.values())))
    return dict(intent=intent, compatible=compatible, connection_key=None if connection is None else connection['key'],
        local_actor_ids=local, participant_ids=participants, participant_current=[metas[key] for key in participants],
        held_participant_intents={key: kernel._intent(packets[key]) for key in participants},
        actor_order={key: context['packet_order'][key] for key in participants},
        statistics={key: dict(value) for key, value in stats.items()}, outer_boxes=boxes, examples=examples,
        conditional_lateral_readouts=conditional_readouts)


def inspect_seed(task):
    seed, cases, cache, static, laws, width, output = task
    table = build_entity_geometry(RoadTopology(static), width)
    directory = cache/'training'/f'seed{seed}'
    executors = json.loads((directory/'executor_seed_server_only.json').read_text())
    selected = {int(case['reference']['time_sec']): case for case in cases}
    rows = []
    with gzip.open(output/f'seed{seed}_branches_server_only.jsonl.gz', 'xt', compresslevel=1) as out:
        for frame, unused_next, action in iter_transitions(directory):
            t = int(frame['time_s'])
            if t not in selected:
                continue
            case, pid = selected[t], selected[t]['reference']['packet_id']
            kernel = prepare_feature_kernel(static, frame, executors, laws, action['requests'], RESERVOIR_SEED+seed*100000+t, conflict_geometry=table)
            if kernel.signals != action['applied_states']:
                raise ValueError('Current interval signal must reproduce')
            priors = _candidates(kernel, pid)
            candidates = [candidate_features(kernel, pid, intent) for intent in priors]
            neighbours = set()
            for candidate, saved, intent in zip(candidates, case['components'], priors):
                if (saved['intent'] != intent or saved['prior'] != priors[intent] or saved['lane_mask'] != candidate['lane_mask'] or
                        abs(saved['mean0_mps']-candidate['mean0_mps']) > 1e-8 or abs(saved['physical_ceiling_mps']-candidate['physical_ceiling_mps']) > 1e-8):
                    raise ValueError('The exact discrete current baseline must reproduce')
                neighbours.update(deniers(frame['packets'][pid], candidate))
            participants = sorted({pid, *neighbours}, key=kernel._mean_alignment_context['packet_order'].__getitem__)
            components = []
            for intent in priors:
                old_intent, old_connection = deepcopy(kernel.state['intents'].get(pid)), kernel._connections[pid]
                try:
                    kernel.state['intents'][pid] = dict(road_id=kernel.topology.lanes[frame['packets'][pid]['lane_id']]['edge_id'], choice=intent)
                    kernel._connections[pid] = kernel._connection(kernel.state['packets'][pid])
                    components.append(dict(inspect_intent(kernel, pid, intent, participants, case['reference'], out), prior=priors[intent]))
                finally:
                    kernel.state['intents'][pid] = old_intent
                    kernel._connections[pid] = old_connection
            # The frozen observed class enters only this posthoc summary.
            label = CHOICES[int(case['reference']['y'])]
            relevant = [c['statistics'][label] for c in components if c['prior'] > 0]
            row = dict(reference=case['reference'], baseline_reproduced=True, components=components,
                observed_label=label, endpoint_witness=any(c.get('endpoint_witnesses', 0) for c in relevant),
                ordered_witness=any(c.get('ordered_execution_witnesses', 0) for c in relevant),
                zero_residual_ordered_witness=any(c.get('zero_residual_ordered_witnesses', 0) for c in relevant))
            row['zero_residual_local_observed_class_mass'] = sum(c['prior']*next(
                value['lateral_class_mass'][label] for value in c['conditional_lateral_readouts']
                if value['zero_motion_residual']) for c in components)
            rows.append(row)
            print(json.dumps(dict(stage='lateral_case_complete', reference=case['reference'], participants=len(participants),
                endpoint=row['endpoint_witness'], ordered=row['ordered_witness'], zero_ordered=row['zero_residual_ordered_witness'])), flush=True)
    if len(rows) != len(cases):
        raise ValueError('Every original fixed lane reference must be diagnosed')
    save(output/f'seed{seed}_cases_server_only.json', rows)
    return rows


def run(cache, baseline, preflight, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        cases = [c for c in json.loads(baseline.read_text())['cases'] if c['head'] == 'lane_change']
        if len(cases) != 9:
            raise ValueError('Exactly the original9 lane exclusions are required')
        width, unused = verify_entity_preflight(preflight)
        static, unused = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        grouped = defaultdict(list)
        for case in cases:
            grouped[int(case['reference']['seed'])].append(case)
        tasks = [(seed, rows, cache, static, laws, width, output) for seed, rows in sorted(grouped.items())]
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
            rows = [case for group in pool.map(inspect_seed, tasks) for case in group]
        result = dict(status='COMPLETE_FIXED9_LATERAL_CONDITIONING_WITNESSES', cases=rows,
            endpoint_witness_cases=sum(row['endpoint_witness'] for row in rows), ordered_witness_cases=sum(row['ordered_witness'] for row in rows),
            zero_residual_ordered_witness_cases=sum(row['zero_residual_ordered_witness'] for row in rows),
            new_native_starts=0, new_fits=0, optimizer_calls=0, free_rollouts=0, calibration_data_used=False,
            production_operator_changed=False, probabilities_assigned_to_residual_interventions=False,
            conditional_probability_definition='For each fixed intent and external residual tuple separately: lane logits0, original sequential dynamic masks, other lateral actors do(stay), potential start/yield gates; no mixture over residual tuples and no endpoint-mass renormalization.',
            scope='Current-only finite interventions. Endpoint and frozen-order witnesses are not native subsecond trajectories or proof of a common fitted beta; no sampled residual range or calibrated probability is inferred.')
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_FIXED9_LATERAL_CONDITIONING_WITNESSES', error=f'{type(error).__name__}: {error}', new_native_starts=0, new_fits=0)
    save(output/'result.json', result)
    compact = {key: value for key, value in result.items() if key != 'cases'}
    if 'cases' in result:
        compact['cases'] = []
        for row in result['cases']:
            item = {key: value for key, value in row.items() if key != 'components'}
            item['components'] = []
            for c in row['components']:
                detail = {key: value for key, value in c.items() if key not in ('examples', 'outer_boxes')}
                detail['outer_box_possible_by_choice'] = {choice: box['relaxed_independent_box_possible'] for choice, box in c['outer_boxes'].items()}
                keys = (row['observed_label']+'zero', row['observed_label']+'external_residual', row['observed_label']+'_endpoint')
                detail['posthoc_observed_class_example'] = next((c['examples'][key] for key in keys if key in c['examples']), None)
                item['components'].append(detail)
            compact['cases'].append(item)
    save(output/'summary.json', compact)
    print(json.dumps(dict(status=result['status'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache', 'baseline', 'preflight', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cache, args.baseline, args.preflight, args.output)
    raise SystemExit(0 if result['status'] == 'COMPLETE_FIXED9_LATERAL_CONDITIONING_WITNESSES' else 2)
