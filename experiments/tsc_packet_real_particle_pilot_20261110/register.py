"""Freeze fifteen current states and their unchanged geometry before responses."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time
import traceback

from experiments.tsc_packet_conditional_domain_20261110.current import GeometryView
from experiments.tsc_packet_conditional_domain_20261110.budget import symbolic_budget
from experiments.tsc_packet_conditional_domain_20261110.register import (
    RESERVOIR_SEED, selected_current_records,
)
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def reference_key(reference):
    return int(reference['seed']), int(reference['time_sec']), reference['packet_id']


def select_cases(original_refs, rows):
    """Use only current census identifiers/counts; stored labels are ignored."""
    if len(original_refs) != 9 or len({reference_key(r) for r in original_refs}) != 9:
        raise ValueError('Exactly the original nine ordered references are required')
    originals = {reference_key(r) for r in original_refs}
    found, additions = {}, {}
    for row in rows:
        key = reference_key(row)
        count = row['required_actor_count']
        if key not in originals and count not in range(5, 11):
            continue
        current = dict(reference=dict(seed=key[0], time_sec=key[1], packet_id=key[2]),
            required_participants=list(row['required_participants']),
            required_actor_count=count, frame_actor_count=row['frame_actor_count'],
            census_status=row['status'])
        if key in originals:
            found[key] = current
        if count in range(5, 11):
            if count not in additions or key < reference_key(additions[count]['reference']):
                additions[count] = current
    if set(found) != originals or set(additions) != set(range(5, 11)):
        raise ValueError('All original nine and each required size5-10 must exist in the census')
    selected = [found[reference_key(r)] for r in original_refs]+[additions[n] for n in range(5, 11)]
    if len({reference_key(r['reference']) for r in selected}) != 15:
        raise ValueError('The frozen pilot must have fifteen distinct current cases')
    return [dict(row, case_index=index, origin='original9' if index < 9 else 'first_current_J'+str(index-4))
            for index, row in enumerate(selected)]


def census_rows(directory):
    for path in sorted(Path(directory).glob('seed*_cases_server_only.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                yield json.loads(line)


def run(cache, census, original_cohort, output):
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    rows, frames = [], 0
    try:
        originals = json.loads(original_cohort.read_text())['references']
        chosen = select_cases(originals, census_rows(census))
        if sum(r['required_actor_count'] for r in chosen[:9]) != 22 or sum(r['required_actor_count'] for r in chosen) != 67:
            raise ValueError('The declared original22 plus added45 actor budget must be preserved')
        # This receipt precedes current-frame recovery and all native label access.
        save(output/'cohort.json', dict(status='FROZEN_LABELS_FREE_CURRENT_COHORT',
            frozen_at_utc=datetime.now(timezone.utc).isoformat(), cases=chosen,
            references=[r['reference'] for r in chosen], labels_used=False,
            selection='Original9 order, followed by lexicographic first current case for each J5-10.'))
        static = json.loads((cache/'static_graph_server_only.json').read_text())
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        grouped = defaultdict(lambda: defaultdict(list))
        for row in chosen:
            seed, t, unused = reference_key(row['reference'])
            grouped[seed][t].append(row)
        for seed, times in sorted(grouped.items()):
            for frame, action, unused in selected_current_records(cache/'training'/f'seed{seed}', times):
                t = int(frame['time_s'])
                frames += 1
                stem = f'seed{seed}_t{t}'
                current_file = stem+'_current_server_only.json'
                save(output/current_file, dict(frame=frame, action=action))
                view = GeometryView(static, frame, laws, action['applied_states'], RESERVOIR_SEED+seed*100000+t)
                for chosen_row in times[t]:
                    ref, index = chosen_row['reference'], chosen_row['case_index']
                    selection = select_stochastic(view, ref['packet_id'], max_stochastic=10)
                    unchanged = (selection['required_participants'] == chosen_row['required_participants']
                        and len(view.state['packets']) == chosen_row['frame_actor_count'])
                    budget = symbolic_budget(view, selection)
                    if (not unchanged or selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE'
                            or budget['status'] != 'REGISTERED_SYMBOLIC_UPPER_BOUND'):
                        raise ValueError('A frozen census case changed required set/initialization or prior domain')
                    selection_file = f'case{index:02d}_selection_server_only.json'
                    save(output/selection_file, dict(selection=selection, internal_bindings=view.internal_bindings))
                    row = dict(chosen_row, current_cache=current_file, selection_cache=selection_file,
                        census_required_set_preserved=unchanged,
                        positive_goal_counts={key:value['current_positive_goals'] for key,value in budget['per_actor'].items()},
                        particle_full_forward_ceiling=1024,
                        canonical_feature_ceiling=1024*chosen_row['required_actor_count'])
                    rows.append(row)
                    print(json.dumps(dict(stage='real_particle_current_registered', case_index=index,
                        reference=ref, required_actors=row['required_actor_count'], unchanged=unchanged)), flush=True)
        rows.sort(key=lambda row: row['case_index'])
        if len(rows) != 15 or frames > 15:
            raise ValueError('The complete15-state current-frame budget did not register')
        result = dict(status='PASS_FROZEN_REAL_PARTICLE_COHORT', cases=rows, requested_states=15,
            current_frames_decoded=frames, interval_actions_decoded=frames, native_next_pairs_formed=0,
            particle_full_forward_ceiling=sum(row['particle_full_forward_ceiling'] for row in rows),
            canonical_feature_ceiling=sum(row['canonical_feature_ceiling'] for row in rows),
            all_census_required_sets_preserved=True, labels_used=False,
            frozen_before_real_responses=True, frozen_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_REAL_PARTICLE_CURRENT_REGISTRATION', error=f'{type(error).__name__}: {error}',
            cases=rows, current_frames_decoded=frames, labels_used=False)
    result.update(real_record_response_queries=0, new_native_starts=0, new_fits=0,
        optimizer_calls=0, elapsed_sec=time.monotonic()-began)
    save(output/'summary.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache', 'census', 'original-cohort', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cache, args.census, args.original_cohort, args.output)
    raise SystemExit(0 if result['status'] == 'PASS_FROZEN_REAL_PARTICLE_COHORT' else 2)
