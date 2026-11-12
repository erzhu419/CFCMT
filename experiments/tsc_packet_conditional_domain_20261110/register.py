"""Register nine current-state domains and symbolic costs, without responses."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import time
import traceback

from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from .budget import symbolic_budget
from .current import GeometryView


RESERVOIR_SEED = 76116999  # Frozen training_data.RESERVOIR_SEED; no training module import.


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def selected_current_records(directory, requested_times):
    """Decode only requested current frames and same-interval applied actions.

    Existing gzip files must be streamed to retrieve selected lines; unselected
    frames are not decoded and no current/next observation pair is formed.
    """
    requested = set(requested_times)
    last = max(requested)
    found = set()
    with gzip.open(directory/'frames_server_only.jsonl.gz', 'rt') as frames, gzip.open(
            directory/'execution_server_only.jsonl.gz', 'rt') as actions:
        for index, (frame_line, action_line) in enumerate(zip(frames, actions)):
            if index in requested:
                frame, action = json.loads(frame_line), json.loads(action_line)
                if frame['time_s'] != index or action['time_sec'] != index:
                    raise ValueError('Selected current frame and interval action are not pre-step aligned')
                found.add(index)
                yield frame, action, index+1
            if index == last:
                break
    if found != requested:
        raise ValueError('Every frozen requested current state must be recovered')


def register_case(static, laws, reference, frame, action, output):
    seed, t, focal = reference['seed'], int(reference['time_sec']), reference['packet_id']
    if focal not in frame['packets']:
        raise ValueError('The fixed focal identity is missing from its exact current state')
    view = GeometryView(static, frame, laws, action['applied_states'], RESERVOIR_SEED+seed*100000+t)
    selection = select_stochastic(view, focal, max_stochastic=3)
    budget = symbolic_budget(view, selection)
    required = selection['required_participants']
    bindings = view.internal_bindings
    save(output/f'seed{seed}_t{t}_geometry_server_only.json', dict(reference=reference,
        selection=selection, internal_bindings=bindings, symbolic_budget=budget))
    status = ('DOMAIN_REGISTERED_NO_RESPONSES' if budget['status'] == 'REGISTERED_SYMBOLIC_UPPER_BOUND'
              else 'OUTSIDE_REGISTERED_STOCHASTIC_DOMAIN' if selection['status'] == 'OBSTRUCTION_STOCHASTIC_SET_EXCEEDS_CAP'
              else 'HOLD_GEOMETRIC_OR_INTENT_DOMAIN')
    return dict(reference=reference, status=status, frame_actor_count=len(view.state['packets']),
        pending_actor_count=len(view.state['pending']), selector_status=selection['status'],
        selector_reason=selection.get('reason'), required_actor_count=len(required),
        required_participants=required, stochastic_participants=selection['participants'],
        participant_reasons={key: selection['reasons'][key] for key in required},
        participant_bounds={key: selection['bounds'][key] for key in required},
        participant_current_fields={key: {name: view.state['packets'][key][name]
            for name in ('lane_id', 'position_m', 'speed_mps', 'parameters')} for key in required},
        internal_binding_count=len(bindings), ambiguous_internal_binding_count=sum(b['ambiguous'] for b in bindings.values()),
        body_trail_fallback_count=sum(b['body_trail_fallback'] for b in bindings.values()),
        selected_internal_bindings={key: bindings[key] for key in required if key in bindings},
        symbolic_budget=budget, real_record_response_queries=0)


def run(cache, cohort, output):
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    try:
        references = json.loads(cohort.read_text())['references']
        if len(references) != 9 or len({(r['seed'], r['time_sec'], r['packet_id']) for r in references}) != 9:
            raise ValueError('Exactly the original9 unique current-state references are required')
        if any(set(r) != {'seed', 'time_sec', 'packet_id'} for r in references):
            raise ValueError('The registration cohort must contain only seed/time/focal IDs, no labels or witness actions')
        static = json.loads((cache/'static_graph_server_only.json').read_text())
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        grouped = defaultdict(dict)
        for r in references:
            if r['time_sec'] != int(r['time_sec']) or int(r['time_sec']) in grouped[r['seed']]:
                raise ValueError('The fixed9 panel requires one focal at each registered whole-second state')
            grouped[r['seed']][int(r['time_sec'])] = r
        rows, streamed = [], {}
        for seed, cases in sorted(grouped.items()):
            for frame, action, source_lines in selected_current_records(cache/'training'/f'seed{seed}', cases):
                ref = cases[int(frame['time_s'])]
                row = register_case(static, laws, ref, frame, action, output)
                rows.append(row)
                streamed[str(seed)] = source_lines
                print(json.dumps(dict(stage='current_geometry_registered', reference=ref,
                    frame_actors=row['frame_actor_count'], required_actors=row['required_actor_count'],
                    status=row['status'], physical_calls_upper=row['symbolic_budget']['physical_resolver_calls_upper_bound'])), flush=True)
        if len(rows) != 9:
            raise ValueError('All9 cases must remain in the registration panel')
        defined = [r for r in rows if r['status'] == 'DOMAIN_REGISTERED_NO_RESPONSES']
        result = dict(status='COMPLETE_CURRENT_ONLY_DOMAIN_REGISTRATION', cases=rows,
            requested_states=9, current_frames_decoded=len(rows), interval_actions_decoded=len(rows),
            source_current_action_line_pairs_streamed_for_retrieval=streamed, native_next_pairs_formed=0,
            case_status_counts=dict(Counter(r['status'] for r in rows)), in_domain_states=len(defined),
            full9_domain_registered=len(defined) == 9,
            in_domain_physical_resolver_calls_upper_bound=sum(r['symbolic_budget']['physical_resolver_calls_upper_bound'] for r in defined),
            in_domain_lateral_leaf_upper_bound=sum(r['symbolic_budget']['lateral_leaf_branches_upper_bound'] for r in defined),
            full9_physical_resolver_calls_upper_bound=sum(r['symbolic_budget']['physical_resolver_calls_upper_bound'] for r in defined) if len(defined) == 9 else None,
            max_stochastic_actors=3, selector_changed=False, nuisance_seed_definition='76116999+seed*100000+time_sec',
            model_admitted=False, calibration_data_used=False,
            scope='Current geometry and symbolic latent-count registration only. No physical mean/endpoint, LC probability, observation likelihood or true future was queried.')
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_CURRENT_ONLY_DOMAIN_REGISTRATION_TECHNICAL', error=f'{type(error).__name__}: {error}')
    result.update(new_native_starts=0, new_fits=0, new_nuisance_estimations=0, optimizer_calls=0,
        real_record_response_queries=0, free_rollouts=0, elapsed_sec=time.monotonic()-began)
    save(output/'result.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache', 'cohort', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    result = run(args.cache, args.cohort, args.output)
    raise SystemExit(0 if result['status'] == 'COMPLETE_CURRENT_ONLY_DOMAIN_REGISTRATION' else 2)
