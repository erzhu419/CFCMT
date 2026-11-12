"""Current-only complete reservoir domain and symbolic-cost census.

Only geometry, current signals and frozen priors are used. Native labels are
projected out of the saved manifest; no transition/model module is imported.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import time
import traceback

from experiments.tsc_packet_conditional_domain_20261110.current import GeometryView
from experiments.tsc_packet_conditional_domain_20261110.budget import symbolic_budget
from experiments.tsc_packet_conditional_law_20261110.selection import _envelope


SEEDS = (76116901, 76116902, 76116903, 76116904)
HEADS = ('motion', 'start', 'yield_entry', 'lane_change')
REGISTERED = 'REGISTERED_SYMBOLIC_UPPER_BOUND'
RESERVOIR_SEED = 76116999


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def selected_current_records(directory, requested_times):
    """Stream current frames/aligned actions once; never form next pairs."""
    requested = set(requested_times)
    last = max(requested)
    found = set()
    with gzip.open(directory/'frames_server_only.jsonl.gz', 'rt') as frames, gzip.open(
            directory/'execution_server_only.jsonl.gz', 'rt') as actions:
        for index, (frame_line, action_line) in enumerate(zip(frames, actions)):
            if index in requested:
                frame, action = json.loads(frame_line), json.loads(action_line)
                if frame['time_s'] != index or action['time_sec'] != index:
                    raise ValueError('Selected current frame and action are not pre-step aligned')
                found.add(index)
                yield frame, action, index+1
            if index == last:
                break
    if found != requested:
        raise ValueError('Every frozen requested current state must be recovered')


class FrameSelector:
    """Reuse immutable frame envelopes; the old selection predicate is exact."""
    def __init__(self, view):
        self.view = view
        self.error = None
        self.bounds = {}
        try:
            self.bounds = {key: _envelope(view, packet)
                           for key, packet in view.state['packets'].items()}
        except ValueError as error:
            self.error = str(error)

    def select(self, focal_id, *, max_stochastic=4):
        packets = self.view.state['packets']
        ids = list(packets)
        result = dict(status=None, focal_id=focal_id, participants=[], required_participants=[],
                      max_stochastic=max_stochastic, bounds={}, reasons={})
        if self.error is not None:
            result.update(status='OBSTRUCTION_GEOMETRIC_INTENT_DOMAIN', reason=self.error)
            return result
        result['bounds'] = self.bounds
        focal = packets[focal_id]
        required = {focal_id}
        result['reasons'][focal_id] = ['focal']
        focal_ranges = self.bounds[focal_id]['potential_landing_fronts']
        for key, packet in packets.items():
            if key == focal_id:
                continue
            matching = []
            for lane, (low, high) in self.bounds[key]['potential_landing_fronts'].items():
                if lane not in focal_ranges:
                    continue
                focal_low, focal_high = focal_ranges[lane]
                if (high >= focal_low-focal['parameters']['length_m']-packet['parameters']['min_gap_m']
                        and low <= focal_high+packet['parameters']['length_m']+focal['parameters']['min_gap_m']):
                    matching.append(lane)
            if matching:
                required.add(key)
                result['reasons'][key] = matching
        result['required_participants'] = [key for key in ids if key in required]
        result['required_count'] = len(required)
        if len(required) > max_stochastic:
            result.update(status='OBSTRUCTION_STOCHASTIC_SET_EXCEEDS_CAP',
                          reason='The full geometry-selected set exceeds the declared enumeration limit')
            return result
        result.update(status='SELECTED_GEOMETRIC_ENVELOPE',
                      participants=list(result['required_participants']), reason=None)
        return result


def current_eligibility(view, packet):
    """The old law's selector, copied without importing any response modules."""
    topology = view.topology
    geometry = topology.lanes[packet['lane_id']]
    reach = packet['speed_mps']+packet['parameters']['accel_mps2']
    connections = topology.connections_from(packet['lane_id'])
    entry = (not geometry['internal'] and reach >= geometry['length_m']-packet['position_m'] and
        any(c['tls_id'] is not None and view.signals[c['tls_id']][c['link_index']] == 'g' for c in connections))
    lateral = not geometry['internal'] and len(topology.road_lanes(geometry['edge_id'])) > 1
    if not lateral:
        possible = topology.internal_connections[packet['lane_id']] if geometry['internal'] else connections
        lateral = any(reach >= topology.path_distance(c, packet['lane_id'], packet['position_m'], c['to_lane'], 0.)
            and len(topology.road_lanes(c['to_edge'])) > 1 for c in possible)
    return dict(motion=True, start=packet['speed_mps'] < .1, yield_entry=bool(entry), lane_change=bool(lateral))


def validate_manifest(reservoirs):
    if sorted(map(int, reservoirs)) != list(SEEDS):
        raise ValueError('The exact four original training seeds are required')
    for seed in SEEDS:
        rows = reservoirs[str(seed)]
        if set(rows) != set(HEADS) or any(len(rows[head]) != 10000 for head in HEADS):
            raise ValueError('All four original10000-reference heads per seed are required')


def group_references(reservoir, seed):
    """Project original rows to current identifiers and ordinal head linkage.

    The stored native label field y is never read, compared, or copied.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for head in HEADS:
        for ordinal, row in enumerate(reservoir[head]):
            if int(row['seed']) != seed or row['time_sec'] != int(row['time_sec']):
                raise ValueError('A reference must match its outer seed and whole current second')
            grouped[int(row['time_sec'])][row['packet_id']].append((head, ordinal))
    return {t: dict(rows) for t, rows in grouped.items()}


def empty_statistics():
    return dict(reference_records=0, eligible_records=0, geometry_in_domain=0,
        defined_records=0, case_status_counts=Counter(), disposition_counts=Counter(),
        required_actor_count_histogram=Counter(),
        eligible_case_status_counts=Counter(), ineligible_case_status_counts=Counter(),
        defined_subset_full_forward_upper_bound=0, defined_subset_leaf_upper_bound=0)


def add_reference(stat, case, head_eligible):
    stat['reference_records'] += 1
    stat['eligible_records'] += int(head_eligible)
    defined = case['status'] == REGISTERED
    stat['geometry_in_domain'] += int(defined)
    stat['case_status_counts'][case['status']] += 1
    stat['required_actor_count_histogram'][str(case['required_actor_count']) if case['required_actor_count'] is not None else 'unknown'] += 1
    stat['eligible_case_status_counts' if head_eligible else 'ineligible_case_status_counts'][case['status']] += 1
    ready = defined and head_eligible
    stat['defined_records'] += int(ready)
    disposition = ('DEFINED_TARGET_REFERENCE' if ready else
                   'OUTSIDE_HEAD_SELECTOR' if not head_eligible else case['status'])
    stat['disposition_counts'][disposition] += 1
    if ready:
        stat['defined_subset_full_forward_upper_bound'] += case['physical_calls_upper']
        stat['defined_subset_leaf_upper_bound'] += case['leaf_visits_upper']


def finalize_statistics(stat):
    result = {key: dict(value) if isinstance(value, Counter) else value for key, value in stat.items()}
    complete = result['defined_records'] == result['reference_records']
    result.update(full_target_defined=complete,
        full_full_forward_upper_bound=result['defined_subset_full_forward_upper_bound'] if complete else None,
        full_leaf_upper_bound=result['defined_subset_leaf_upper_bound'] if complete else None)
    return result


def merge_statistics(rows):
    result = empty_statistics()
    for row in rows:
        for key in result:
            if isinstance(result[key], Counter):
                result[key].update(row[key])
            else:
                result[key] += row[key]
    return finalize_statistics(result)


def _seed(task):
    seed, cache, static, laws, reservoir, output = task
    began, last_progress = time.monotonic(), time.monotonic()
    expected_references = sum(len(rows) for rows in reservoir.values())
    grouped = group_references(reservoir, seed)
    del reservoir
    reference_stats, case_stats = empty_statistics(), empty_statistics()
    head_stats = {head: empty_statistics() for head in HEADS}
    frames, source_lines = 0, 0
    actor_hist, required_max = Counter(), 0
    bindings = Counter()
    examples = {}
    shared_forward = shared_leaves = unique_cases = 0
    head_path = output/f'seed{seed}_references_server_only.jsonl.gz'
    case_path = output/f'seed{seed}_cases_server_only.jsonl.gz'
    with gzip.open(head_path, 'xt', compresslevel=1) as references, gzip.open(case_path, 'xt', compresslevel=1) as cases:
        for frame, action, source_lines in selected_current_records(cache/'training'/f'seed{seed}', grouped):
            t = int(frame['time_s'])
            view = GeometryView(static, frame, laws, action['applied_states'], RESERVOIR_SEED+seed*100000+t)
            selector = FrameSelector(view)
            frames += 1
            actor_hist[str(len(view.state['packets']))] += 1
            bindings['internal_actor_current_frames'] += len(view.internal_bindings)
            bindings['ambiguous_internal_actor_current_frames'] += sum(x['ambiguous'] for x in view.internal_bindings.values())
            bindings['body_trail_fallback_actor_current_frames'] += sum(x['body_trail_fallback'] for x in view.internal_bindings.values())
            for focal in sorted(grouped[t]):
                if focal not in view.state['packets']:
                    raise ValueError('An original focal is absent from its exact current state')
                selection = selector.select(focal, max_stochastic=4)
                budget = symbolic_budget(view, selection)
                status = (REGISTERED if budget['status'] == REGISTERED else
                          selection['status'] if selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE' else budget['status'])
                eligible = current_eligibility(view, view.state['packets'][focal])
                case_id = unique_cases
                case = dict(seed=seed, time_sec=t, packet_id=focal, case_id=case_id,
                    status=status, selector_status=selection['status'], budget_status=budget['status'],
                    reason=selection.get('reason') or budget.get('reason'),
                    required_actor_count=selection.get('required_count'),
                    required_participants=selection['required_participants'],
                    frame_actor_count=len(view.state['packets']), eligibility=eligible,
                    physical_calls_upper=budget['physical_resolver_calls_upper_bound'],
                    leaf_visits_upper=budget['lateral_leaf_branches_upper_bound'])
                cases.write(json.dumps(case, separators=(',', ':'))+'\n')
                unique_cases += 1
                if case['required_actor_count'] is not None:
                    required_max = max(required_max, case['required_actor_count'])
                add_reference(case_stats, case, True)  # Law domain before head selection.
                linked = grouped[t][focal]
                if status == REGISTERED and any(eligible[head] for head, ordinal in linked):
                    shared_forward += case['physical_calls_upper']
                    shared_leaves += case['leaf_visits_upper']
                for head, ordinal in linked:
                    add_reference(reference_stats, case, eligible[head])
                    add_reference(head_stats[head], case, eligible[head])
                    disposition = ('DEFINED_TARGET_REFERENCE' if status == REGISTERED and eligible[head] else
                                   'OUTSIDE_HEAD_SELECTOR' if not eligible[head] else status)
                    references.write(json.dumps(dict(seed=seed, head=head, manifest_ordinal=ordinal,
                        time_sec=t, packet_id=focal, case_id=case_id, head_eligible=eligible[head],
                        disposition=disposition), separators=(',', ':'))+'\n')
                    example_key = head+'__'+status+'__'+('eligible' if eligible[head] else 'outside_head_selector')
                    if example_key not in examples:
                        filename = f'seed{seed}_{example_key}_example_server_only.json'
                        save(output/filename, dict(case=case, selection=selection,
                            symbolic_budget=budget, selected_current_fields={key: {
                                name: view.state['packets'][key][name] for name in
                                ('lane_id', 'position_m', 'speed_mps', 'parameters')}
                                for key in selection['required_participants']}))
                        examples[example_key] = dict(reference={key: case[key] for key in
                            ('seed', 'time_sec', 'packet_id')}, required_actor_count=case['required_actor_count'],
                            detail_file=filename)
            if frames % 100 == 0 or time.monotonic()-last_progress >= 30:
                save(output/f'seed{seed}_progress.json', dict(seed=seed,current_frames=frames,
                    references=reference_stats['reference_records'], unique_cases=unique_cases,
                    current_time_sec=t, elapsed_sec=time.monotonic()-began))
                print(json.dumps(dict(stage='current_only_census_progress', seed=seed, current_frames=frames,
                    references=reference_stats['reference_records'], unique_cases=unique_cases,
                    elapsed_sec=time.monotonic()-began)), flush=True)
                last_progress = time.monotonic()
    expected = sum(len(rows) for rows in grouped.values())
    if unique_cases != expected or reference_stats['reference_records'] != expected_references:
        raise ValueError('Every original unique case and seed reference must be accounted for')
    ref = finalize_statistics(reference_stats)
    result = dict(seed=seed, reference_statistics=ref, unique_case_geometry_statistics=finalize_statistics(case_stats),
        heads={head: finalize_statistics(stat) for head, stat in head_stats.items()},
        unique_current_cases=unique_cases, current_frames_decoded=frames, interval_actions_decoded=frames,
        source_current_action_line_pairs_streamed=source_lines, native_next_pairs_formed=0,
        frame_actor_count_histogram=dict(actor_hist), maximum_required_actor_count=required_max,
        binding_counts=dict(bindings), examples=examples,
        shared_defined_subset_full_forward_upper_bound=shared_forward,
        shared_defined_subset_leaf_upper_bound=shared_leaves,
        full_shared_target_full_forward_upper_bound=shared_forward if ref['full_target_defined'] else None,
        full_shared_target_leaf_upper_bound=shared_leaves if ref['full_target_defined'] else None,
        elapsed_sec=time.monotonic()-began)
    save(output/f'seed{seed}_summary.json',result)
    print(json.dumps(dict(stage='current_only_census_seed_complete',seed=seed,
        references=ref['reference_records'], unique_cases=unique_cases,
        full_target_defined=ref['full_target_defined'],elapsed_sec=result['elapsed_sec'])),flush=True)
    return result


def run(cache, output, workers=4):
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    try:
        reservoirs = json.loads((cache/'training_reservoir_server_only.json').read_text())
        validate_manifest(reservoirs)
        # Project before any worker receives data; no native y leaves this object.
        projected = {str(seed): {head: [dict(seed=row['seed'], time_sec=row['time_sec'],packet_id=row['packet_id'])
            for row in reservoirs[str(seed)][head]] for head in HEADS} for seed in SEEDS}
        del reservoirs
        static = json.loads((cache/'static_graph_server_only.json').read_text())
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        tasks = [(seed,cache,static,laws,projected[str(seed)],output) for seed in SEEDS]
        if workers == 1:
            results = [_seed(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(workers,4)) as executor:
                results = list(executor.map(_seed,tasks))
        refs = merge_statistics(row['reference_statistics'] for row in results)
        unique = merge_statistics(row['unique_case_geometry_statistics'] for row in results)
        actor_hist, binding_counts = Counter(), Counter()
        for row in results:
            actor_hist.update(row['frame_actor_count_histogram'])
            binding_counts.update(row['binding_counts'])
        total_frames = sum(row['current_frames_decoded'] for row in results)
        if refs['reference_records'] != 160000 or total_frames > 7204:
            raise ValueError('Complete160000-reference and current-frame budget accounting failed')
        shared_forward = sum(row['shared_defined_subset_full_forward_upper_bound'] for row in results)
        shared_leaves = sum(row['shared_defined_subset_leaf_upper_bound'] for row in results)
        result = dict(status=('PASS_COMPLETE_CURRENT_DOMAIN_CENSUS' if refs['full_target_defined']
            else 'HOLD_COMPLETE_TARGET_DOMAIN_OR_HEAD_SELECTOR'),
            reference_statistics=refs, unique_case_geometry_statistics=unique,
            heads={head: merge_statistics(row['heads'][head] for row in results) for head in HEADS},
            seeds=results, original_manifest_records=160000, unique_current_cases=unique['reference_records'],
            current_frames_decoded=total_frames, interval_actions_decoded=total_frames,
            native_next_pairs_formed=0,max_stochastic_actors=4,
            frame_actor_count_histogram=dict(actor_hist),binding_counts=dict(binding_counts),
            shared_defined_subset_full_forward_upper_bound=shared_forward,
            shared_defined_subset_leaf_upper_bound=shared_leaves,
            full_shared_target_full_forward_upper_bound=shared_forward if refs['full_target_defined'] else None,
            full_shared_target_leaf_upper_bound=shared_leaves if refs['full_target_defined'] else None,
            partial_bound_scope='Defined, eligible reference subset only; unique-case sums share a case whenever at least one linked head is eligible. No full-target bound unless every original head reference is defined.',
            selector_implementation='Frozen old envelope and exact matching predicate; immutable envelopes computed once per current frame.',
            native_label_fields_read=False, response_modules_imported=False,model_admitted=False)
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_CURRENT_ONLY_CENSUS_TECHNICAL',error=f'{type(error).__name__}: {error}')
    result.update(real_record_response_queries=0,real_full_frame_forward_calls=0,
        real_observation_likelihood_evaluations=0,real_latent_cache_constructions=0,
        new_native_starts=0,new_fits=0,new_nuisance_estimations=0,optimizer_calls=0,
        free_rollouts=0,calibration_seeds_used=False,elapsed_sec=time.monotonic()-began)
    save(output/'result.json',result)
    print(json.dumps({key:result[key] for key in ('status','elapsed_sec')}),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    result=run(args.cache,args.output,args.workers)
    raise SystemExit(0 if result['status']!='HOLD_CURRENT_ONLY_CENSUS_TECHNICAL' else 2)
