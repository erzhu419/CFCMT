"""Integrate both actors' declared intent priors; never mix residual trials."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
from pathlib import Path
import traceback

from experiments.tsc_packet_discrete_braking_20261110.conditional_features import prepare_feature_kernel, prepare_receiving_context
from experiments.tsc_target_packet_training_20261109.training_data import iter_transitions, _candidates, RESERVOIR_SEED
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths, save
from .diagnose import CHOICES, inspect_intent


def inspect_seed(task):
    seed, cases, cache, static, laws, width, output = task
    table = build_entity_geometry(RoadTopology(static), width)
    directory = cache/'training'/f'seed{seed}'
    executors = json.loads((directory/'executor_seed_server_only.json').read_text())
    selected = {int(c['reference']['time_sec']): c for c in cases}
    rows = []
    with gzip.open(output/f'seed{seed}_all_branches_server_only.jsonl.gz', 'xt', compresslevel=1) as out:
        for frame, unused_next, action in iter_transitions(directory):
            t = int(frame['time_s'])
            if t not in selected:
                continue
            original = selected[t]
            ref, pid = original['reference'], original['reference']['packet_id']
            kernel = prepare_feature_kernel(static, frame, executors, laws, action['requests'], RESERVOIR_SEED+seed*100000+t, conflict_geometry=table)
            participants = original['components'][0]['participant_ids']
            if len(participants) != 2 or any(c['participant_ids'] != participants for c in original['components']):
                raise ValueError('The nine registered local sets each contain the focal and one fixed denier')
            other = next(key for key in participants if key != pid)
            if kernel._intent(frame['packets'][other]) != original['components'][0]['held_participant_intents'][other]:
                raise ValueError('The original held-neighbour draw must reproduce before marginalization')
            focal_prior, neighbour_prior = _candidates(kernel, pid), _candidates(kernel, other)
            if list(focal_prior) != [c['intent'] for c in original['components']]:
                raise ValueError('The original focal prior order must remain fixed')
            mixed, pair_summaries, full, example = {}, [], [], None
            observed_label = CHOICES[int(ref['y'])]
            for intent, focal_mass in focal_prior.items():
                for other_intent, neighbour_mass in neighbour_prior.items():
                    if focal_mass <= 0 or neighbour_mass <= 0:
                        continue
                    for key, goal in ((pid, intent), (other, other_intent)):
                        road = kernel.topology.lanes[frame['packets'][key]['lane_id']]['edge_id']
                        kernel.state['intents'][key] = dict(road_id=road, choice=goal)
                        kernel._connections[key] = kernel._connection(kernel.state['packets'][key])
                    prepare_receiving_context(kernel)
                    component = inspect_intent(kernel, pid, intent, participants,
                        dict(ref, conditioned_neighbour_intent=other_intent), out)
                    weight = focal_mass*neighbour_mass
                    full.append(dict(component, focal_prior=focal_mass, neighbour_prior=neighbour_mass))
                    for conditional in component['conditional_lateral_readouts']:
                        key = tuple(conditional['residuals_mps'][actor] for actor in participants)
                        target = mixed.setdefault(key, dict(residuals_mps=conditional['residuals_mps'],
                            zero_motion_residual=conditional['zero_motion_residual'], total_mass=0.,
                            lateral_class_mass=dict.fromkeys(CHOICES, 0.), endpoint_feasible_class_mass=dict.fromkeys(CHOICES, 0.)))
                        target['total_mass'] += weight*conditional['total_mass']
                        for choice in CHOICES:
                            target['lateral_class_mass'][choice] += weight*conditional['lateral_class_mass'][choice]
                            target['endpoint_feasible_class_mass'][choice] += weight*conditional['endpoint_feasible_class_mass'][choice]
                    stat = component['statistics'][observed_label]
                    zero = next(c for c in component['conditional_lateral_readouts'] if c['zero_motion_residual'])
                    pair_summaries.append(dict(focal_intent=intent, neighbour_intent=other_intent, prior_product=weight,
                        endpoint_witness=bool(stat.get('endpoint_witnesses', 0)), ordered_witness=bool(stat.get('ordered_execution_witnesses', 0)),
                        zero_residual_observed_class_mass=zero['lateral_class_mass'][observed_label]))
                    if example is None and stat.get('ordered_execution_witnesses', 0):
                        for suffix in ('zero', 'external_residual'):
                            if observed_label+suffix in component['examples']:
                                example = dict(focal_intent=intent, neighbour_intent=other_intent,
                                    prior_product=weight, witness=component['examples'][observed_label+suffix])
                                break
            slices = list(mixed.values())
            if any(abs(c['total_mass']-1.) > 1e-10 for c in slices):
                raise ValueError('Every fixed residual slice must remain normalized after exact intent mixing')
            for conditional in slices:
                conditional['endpoint_feasible_total_mass'] = sum(conditional['endpoint_feasible_class_mass'].values())
            zero = next(c for c in slices if c['zero_motion_residual'])
            row = dict(reference=ref, participant_ids=participants, neighbour_id=other,
                focal_prior=focal_prior, neighbour_prior=neighbour_prior,
                held_neighbour_control_mass=original['zero_residual_local_observed_class_mass'],
                intent_marginal_zero_residual_observed_class_mass=zero['lateral_class_mass'][observed_label],
                intent_marginal_zero_residual_endpoint_observed_class_mass=zero['endpoint_feasible_class_mass'][observed_label],
                endpoint_witness=any(c['endpoint_witness'] for c in pair_summaries),
                ordered_witness=any(c['ordered_witness'] for c in pair_summaries),
                per_residual_conditional_intent_marginals=slices, intent_pairs=pair_summaries, posthoc_example=example)
            rows.append(row)
            save(output/f'seed{seed}_t{t}_intent_components_server_only.json', full)
            print(json.dumps(dict(stage='intent_marginal_case_complete', reference=ref,
                ordered=row['ordered_witness'], zero_class_mass=row['intent_marginal_zero_residual_observed_class_mass'])), flush=True)
    if len(rows) != len(cases):
        raise ValueError('Every original9 case must be accounted for')
    return rows


def run(cache, first_output, preflight, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        cases = json.loads(first_output.read_text())['cases']
        if len(cases) != 9:
            raise ValueError('Exactly the original9 cases are required')
        width, unused = verify_entity_preflight(preflight)
        static, unused = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        grouped = defaultdict(list)
        for c in cases:
            grouped[int(c['reference']['seed'])].append(c)
        tasks = [(seed, rows, cache, static, laws, width, output) for seed, rows in sorted(grouped.items())]
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
            rows = [row for group in pool.map(inspect_seed, tasks) for row in group]
        result = dict(status='COMPLETE_FIXED9_NEIGHBOUR_INTENT_MARGINAL', cases=rows,
            endpoint_witness_cases=sum(c['endpoint_witness'] for c in rows), ordered_witness_cases=sum(c['ordered_witness'] for c in rows),
            zero_residual_positive_observed_class_cases=sum(c['intent_marginal_zero_residual_observed_class_mass'] > 0 for c in rows),
            new_native_starts=0, new_fits=0, optimizer_calls=0, calibration_data_used=False,
            scope='Exact positive current-history intent marginal for the two fixed local actors. Residual interventions remain separate; other actors retain fixed intents and do(stay). No future goal selection, fitted probabilities, global-forward or full190-likelihood claim.')
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_FIXED9_NEIGHBOUR_INTENT_MARGINAL', error=f'{type(error).__name__}: {error}', new_native_starts=0, new_fits=0)
    save(output/'summary.json', result)
    print(json.dumps(dict(status=result['status'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache', 'first-output', 'preflight', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cache, args.first_output, args.preflight, args.output)
    raise SystemExit(0 if result['status'] == 'COMPLETE_FIXED9_NEIGHBOUR_INTENT_MARGINAL' else 2)
