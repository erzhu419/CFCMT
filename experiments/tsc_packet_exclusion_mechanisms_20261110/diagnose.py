"""Explain the fifteen preselected references without modifying any predictor."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import traceback

from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths, save
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_training_20261109.training_data import iter_transitions, _candidates, _front_path, RESERVOIR_SEED
from experiments.tsc_target_packet_training_20261109.audit_exclusions import match_observed_entry
from experiments.tsc_packet_mean_alignment_20261110.conditional_features import prepare_feature_kernel, candidate_features
from experiments.tsc_packet_mean_alignment_20261110.mean import make_conditional_resolver
from experiments.tsc_packet_residual_truth_20261110.receiving_trace import _observed_motion, _snapshot
from .proposal_terms import proposal_terms
from .lane_evidence import explain_lane


TOL = 1e-8


def inspect_component(kernel, pid, intent):
    packet = kernel.state['packets'][pid]
    previous, existed = deepcopy(kernel.state['intents'].get(pid)), pid in kernel.state['intents']
    old_connection = kernel._connections[pid]
    try:
        if not kernel.topology.lanes[packet['lane_id']]['internal']:
            kernel.state['intents'][pid] = dict(road_id=kernel.topology.lanes[packet['lane_id']]['edge_id'], choice=intent)
        connection, compatible = kernel._connection(packet)
        kernel._connections[pid] = connection, compatible
        predicted = candidate_features(kernel, pid, intent)
        resolver = make_conditional_resolver(kernel._mean_alignment_context, kernel, pid, connection, compatible)
        response = resolver.resolve(pid)
        terms = proposal_terms(kernel, packet, connection, compatible, resolver.view(pid))
        if (abs(terms['physical_proposal_mps']-predicted['physical_ceiling_mps']) > TOL or
                abs(response['travel_m']-predicted['mean0_mps']) > TOL):
            raise ValueError('Term decomposition and fresh response must reproduce the frozen component')
        leader_id = terms['leader']['packet_id']
        leader_model = None
        if leader_id is not None:
            already = leader_id in resolver.results
            leader_response = resolver.resolve(leader_id)
            if leader_response is not None:
                leader_model = dict(already_required_by_focal_query=already,
                    query='Current-state local leader projection; extra resolution never changes the saved focal prediction.',
                    mean0_mps=leader_response['travel_m'], ended=leader_response['ended'],
                    projected_packet=_snapshot(leader_response['projected_packet']))
        return dict(intent=intent, connection=connection, compatible=bool(compatible),
            signal=kernel._signal(connection, compatible), prediction=predicted,
            proposal=terms, spatial=response['spatial_summary'], leader_model_projection=leader_model)
    finally:
        if existed:
            kernel.state['intents'][pid] = previous
        else:
            kernel.state['intents'].pop(pid, None)
        kernel._connections[pid] = old_connection


def verify_baseline(case, priors, components):
    old = case['baseline_record']
    if old['candidate_intents'] != list(priors) or old['prior'] != list(priors.values()):
        raise ValueError('The exact prior masses and candidate order must reproduce')
    head = case['head']
    risk = [row['prediction']['yield_risk'] for row in components] if head == 'yield_entry' else (
        [row['prediction']['lane_mask']['left'] or row['prediction']['lane_mask']['right']
         for row in components] if head == 'lane_change' else [True]*len(components))
    if old['risk'] != risk:
        raise ValueError('The frozen reference risk mask changed')
    if head == 'lane_change':
        mask = [[row['prediction']['lane_mask'][key] for key in ('left', 'stay', 'right')] for row in components]
        if old['mask'] != mask:
            raise ValueError('The original lane masks changed')
        for before, row in zip(case['baseline_lane_components'], components):
            if before['intent'] != row['intent'] or abs(before['mean0_mps']-row['prediction']['mean0_mps']) > TOL:
                raise ValueError('The original lane-component responses changed')
    if head == 'motion':
        for index, row in enumerate(components):
            for key in ('mean0_mps', 'physical_ceiling_mps'):
                if abs(old[key][index]-row['prediction'][key]) > TOL:
                    raise ValueError('The registered motion comparison changed')


def posthoc_component(kernel, before, after, pid, component, paths, head, label):
    row = deepcopy(component)
    packet, actual = before['packets'][pid], after['packets'].get(pid)
    distance = kernel.topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
    predicted, proposal = row['prediction'], row['proposal']
    row['entry_evidence'] = dict(current_lane_remaining_m=distance,
        eligible_current_g_entry=bool(row['connection'] is not None and row['compatible'] and row['signal'] == 'g'),
        physical_proposal_reaches_entry=proposal['physical_proposal_mps'] >= distance,
        mean0_reaches_entry=predicted['mean0_mps'] >= distance,
        same_step_body_bound_reaches_entry=predicted['admission_detail']['path_bound_m'] >= distance,
        spatial_bound_reaches_entry=predicted['admission_detail']['spatial_limit_m'] >= distance)
    leader_id = proposal['leader']['packet_id']
    if leader_id is not None:
        current_leader, next_leader = before['packets'][leader_id], after['packets'].get(leader_id)
        row['leader_observation'] = dict(current=_snapshot(current_leader), next=_snapshot(next_leader),
            observed_motion=_observed_motion(kernel.topology, current_leader, next_leader, paths.get(leader_id)),
            changed_lane=None if next_leader is None else current_leader['lane_id'] != next_leader['lane_id'],
            arrived=leader_id in after['arrived_ids'], future_used_as_predictor=False)
    if head == 'lane_change':
        row['lane_evidence'] = explain_lane(kernel.topology, before['packets'], after['packets'],
            packet, actual, predicted['lane_detail'], label,
            physical_proposal_mps=proposal['physical_proposal_mps'], mean0_mps=predicted['mean0_mps'],
            observed_focal_path=paths.get(pid, [packet['lane_id']]), observed_neighbour_paths=paths,
            candidate_connection=row['connection'])
        for other in row['lane_evidence']['denying_neighbours']:
            actual_gap = other['actual_terminal_gap']
            if actual_gap['status'] == 'same_actual_lane' and other['actual_role_matches_model']:
                focal_error = actual['position_m']-predicted['lane_detail']['projected_position_m']
                neighbour_error = other['next']['position_m']-other['model_accepted_tplus1']['position_m']
                other['terminal_gap_difference_decomposition'] = dict(
                    focal_position_actual_minus_model_m=focal_error,
                    neighbour_position_actual_minus_model_m=neighbour_error,
                    actual_minus_model_gap_m=actual_gap['gap_m']-other['model_gap_m'],
                    coordinate_identity_m=neighbour_error-focal_error if other['model_role'] == 'front' else focal_error-neighbour_error)
    return row


def summarize_case(case):
    output = {key: case[key] for key in ('case_id', 'role', 'head', 'reference', 'baseline_reproduced', 'focal_observation')}
    actual_entry = case['actual_entry']
    matches = actual_entry.get('matches', [])
    output['actual_entry'] = dict(event_kind=actual_entry['event_kind'],
        actual_signal=actual_entry['actual_signal'],
        matched_connections=[{key: value[key] for key in ('key', 'from_lane', 'to_lane', 'to_edge', 'direction')}
                             for value in matches])
    components = []
    for row in case['components']:
        item = dict(intent=row['intent'], prior=row['prior'], compatible=row['compatible'], signal=row['signal'],
            P_mps=row['proposal']['physical_proposal_mps'], mean0_mps=row['prediction']['mean0_mps'],
            binding_terms=row['proposal']['binding_terms'], leader_id=row['proposal']['leader']['packet_id'],
            entry_evidence=row['entry_evidence'])
        connection = row['connection']
        item['matches_observed_physical_entry'] = (None if not matches else
            connection is not None and any(connection['key'] == value['key'] for value in matches))
        item['matches_observed_target_road'] = (None if not matches else
            connection is not None and any(connection['to_edge'] == value['to_edge'] for value in matches))
        if 'lane_evidence' in row:
            evidence = row['lane_evidence']
            item.update(lane_reasons=evidence['current_denial_reasons'],
                model_target_lane=evidence['model_target_lane'], arrival=evidence['arrival_evidence'],
                denying_neighbours=[{key: other[key] for key in (
                    'packet_id', 'model_role', 'model_gap_m', 'model_required_gap_m',
                    'observed_same_road_lane_change', 'next_front_left_model_target_lane',
                    'next_body_still_on_model_target_lane', 'actual_terminal_gap',
                    'model_accepted_tplus1', 'next', 'actual_step_displacement_m')}
                    | ({'gap_difference': other['terminal_gap_difference_decomposition']}
                       if 'terminal_gap_difference_decomposition' in other else {})
                    for other in evidence['denying_neighbours']])
        if 'leader_observation' in row:
            item['leader_observation'] = row['leader_observation']
            item['leader_model_projection'] = row['leader_model_projection']
        components.append(item)
    output['components'] = components
    return output


def inspect_seed(task):
    seed, cases, cache, static, laws, width, output = task
    topology = RoadTopology(static)
    table = build_entity_geometry(topology, width)
    directory = cache/'training'/f'seed{seed}'
    executors = json.loads((directory/'executor_seed_server_only.json').read_text())
    by_time = defaultdict(list)
    for case in cases:
        by_time[int(case['reference']['time_sec'])].append(case)
    results = []
    for frame, following, action in iter_transitions(directory):
        t = int(frame['time_s'])
        if t not in by_time:
            continue
        kernel = prepare_feature_kernel(static, frame, executors, laws, action['requests'],
            RESERVOIR_SEED+seed*100000+t, conflict_geometry=table)
        if kernel.signals != action['applied_states']:
            raise ValueError('The original current signal/action must reproduce')
        for case in by_time[t]:
            pid = case['reference']['packet_id']
            priors = _candidates(kernel, pid)
            components = [inspect_component(kernel, pid, intent) for intent in priors]
            verify_baseline(case, priors, components)
            # All saved predictions are finalized before any next-state use.
            paths = {event['vehicle_id']: event['observed_lane_path'] for event in following['events']
                     if event['kind'] == 'lane_transition'}
            packet, actual = frame['packets'][pid], following['packets'].get(pid)
            if actual is None:
                raise ValueError('Every selected reference has an observed next packet')
            motion = _observed_motion(topology, packet, actual, paths.get(pid, [packet['lane_id']]))
            if case['head'] == 'motion' and abs(actual['speed_mps']-case['reference']['y']) > TOL:
                raise ValueError('The registered native motion label changed')
            details = [dict(posthoc_component(kernel, frame, following, pid, row, paths,
                case['head'], case['reference']['y']), prior=priors[row['intent']]) for row in components]
            record = dict(case_id=case['case_id'], role=case['role'], head=case['head'],
                reference=case['reference'], baseline_reproduced=True,
                focal_observation=dict(current=_snapshot(packet), next=_snapshot(actual), observed_motion=motion),
                actual_entry=match_observed_entry(topology, _front_path(frame, following, pid), action['applied_states']),
                components=details)
            results.append(record)
            print(json.dumps(dict(stage='case_complete', case_id=case['case_id'])), flush=True)
        if t >= max(by_time):
            break
    if len(results) != len(cases):
        raise ValueError('Every fixed reference must occur in its original trajectory')
    save(output/f'seed{seed}_cases_server_only.json', results)
    return [summarize_case(case) for case in results]


def run(cache, selection_path, preflight, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        selection = json.loads(selection_path.read_text())
        if len(selection['cases']) != 15 or Counter(case['role'] for case in selection['cases']) != Counter(yield_exclusion=3, lane_exclusion=9, motion_comparison=3):
            raise ValueError('Exactly the preselected3+9+3 cases are required')
        width, unused = verify_entity_preflight(preflight)
        static, width_source = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        grouped = defaultdict(list)
        for case in selection['cases']:
            grouped[int(case['reference']['seed'])].append(case)
        tasks = [(seed, cases, cache, static, laws, width, output) for seed, cases in sorted(grouped.items())]
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
            results = list(pool.map(inspect_seed, tasks))
        cases = [case for group in results for case in group]
        result = dict(status='COMPLETE_FIXED_EXCLUSION_MECHANISMS', cases=cases,
            baseline_reproduced_cases=sum(case['baseline_reproduced'] for case in cases),
            role_counts=dict(Counter(case['role'] for case in cases)), lane_width_source=width_source,
            new_native_starts=0, new_fits=0, new_nuisance_estimations=0, optimizer_calls=0,
            free_rollouts=0, calibration_data_used=False,
            scope='Fixed theta0 configuration and local working masks. Actual next states are posthoc evidence only; terminal gaps do not establish subsecond lateral-event feasibility.')
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_FIXED_EXCLUSION_MECHANISMS', error=f'{type(error).__name__}: {error}',
            new_native_starts=0, new_fits=0)
    save(output/'result.json', result)
    print(json.dumps(dict(status=result['status'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for flag in ('cache-base', 'selection', 'preflight', 'output'):
        parser.add_argument('--'+flag, required=True, type=Path)
    args = parser.parse_args()
    result = run(args.cache_base, args.selection, args.preflight, args.output)
    raise SystemExit(0 if result['status'] == 'COMPLETE_FIXED_EXCLUSION_MECHANISMS' else 2)
