"""Freeze a stratified coefficient cohort and its current geometry, without y."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from random import Random
import time
import traceback

from experiments.tsc_packet_conditional_census_20261110.census import current_eligibility, validate_manifest
from experiments.tsc_packet_conditional_domain_20261110.budget import symbolic_budget
from experiments.tsc_packet_conditional_domain_20261110.current import GeometryView
from experiments.tsc_packet_conditional_domain_20261110.register import RESERVOIR_SEED, selected_current_records
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_real_particle_pilot_20261110.register import census_rows, reference_key, save


SEEDS = (76116901, 76116902, 76116903, 76116904)
HEADS = ('motion', 'start', 'yield_entry', 'lane_change')
SAMPLE_SEED = 20261112
PARTICLES = 256


def select_references(manifest):
    """Sample ordinals before reading any stored label; preserve multiplicity."""
    validate_manifest(manifest)
    rng, selected = Random(SAMPLE_SEED), []
    for seed in SEEDS:
        for head in HEADS:
            for ordinal in sorted(rng.sample(range(10000), 32)):
                old = manifest[str(seed)][head][ordinal]
                selected.append(dict(seed=seed, time_sec=int(old['time_sec']), packet_id=old['packet_id'],
                    head=head, manifest_ordinal=ordinal, split='development' if seed == SEEDS[-1] else 'train'))
                if reference_key(old) != reference_key(selected[-1]):
                    raise ValueError('The manifest reference seed must match its frozen stratum')
    return selected


def group_cases(references):
    grouped = defaultdict(list)
    for reference in references:
        grouped[reference_key(reference)].append(dict(reference))
    return [dict(case_index=index, reference=dict(seed=key[0], time_sec=key[1], packet_id=key[2]),
        references=grouped[key]) for index, key in enumerate(sorted(grouped))]


def pilot_cache_map(pilot):
    """Read only completed pilot metadata, not its labels or particle arrays."""
    old = json.loads((pilot/'summary.json').read_text())
    if old['status'] != 'PASS_BOUNDED_REAL_PARTICLE_PILOT':
        raise ValueError('Only the completed frozen pilot may supply a reusable particle pool')
    result = {}
    for case in old['cases']:
        ref, index = case['reference'], case['case_index']
        directory = pilot/f"case{index:02d}_seed{int(ref['seed'])}_t{int(ref['time_sec'])}"
        path = directory/'particles_server_only.json.gz'
        if case['status'] != 'PASS_REAL_PARTICLE_CASE' or not path.is_file():
            raise ValueError('A declared complete pilot case lacks its frozen pool')
        result[reference_key(ref)] = dict(reuse_particle_path=str(path.resolve()),
            reused_pilot_case_index=index, particle_seed=case['particle_seed'])
    return result


def run(cache, census, pilot, output):
    output.mkdir(parents=True, exist_ok=False)
    began, rows, frames = time.monotonic(), [], 0
    references, chosen = [], []
    try:
        manifest = json.loads((cache/'training_reservoir_server_only.json').read_text())
        references = select_references(manifest)
        del manifest
        chosen = group_cases(references)
        save(output/'cohort.json', dict(status='FROZEN_LABELS_FREE_COEFFICIENT_COHORT',
            frozen_at_utc=datetime.now(timezone.utc).isoformat(), sample_seed=SAMPLE_SEED,
            references=references, cases=chosen, selected_references=len(references), labels_used=False,
            history_exposure='Frozen laws already used all four trajectories,7200 seconds;904 is internal coefficient development.'))
        # The complete cohort is now fixed. Current-domain failure never redraws it.
        wanted = {reference_key(case['reference']) for case in chosen}
        matched = {reference_key(row): row for row in census_rows(census) if reference_key(row) in wanted}
        if set(matched) != wanted:
            raise ValueError('Every selected current case must exist in the completed census')
        reusable = pilot_cache_map(pilot)
        static = json.loads((cache/'static_graph_server_only.json').read_text())
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        grouped = defaultdict(lambda: defaultdict(list))
        for case in chosen:
            seed, t, unused = reference_key(case['reference'])
            grouped[seed][t].append(case)
        for seed, times in sorted(grouped.items()):
            for frame, action, unused in selected_current_records(cache/'training'/f'seed{seed}', times):
                t = int(frame['time_s'])
                frames += 1
                current_file = f'seed{seed}_t{t}_current_server_only.json'
                save(output/current_file, dict(frame=frame, action=action))
                view = GeometryView(static, frame, laws, action['applied_states'], RESERVOIR_SEED+seed*100000+t)
                for case in times[t]:
                    ref, index = case['reference'], case['case_index']
                    key, original = reference_key(ref), matched[reference_key(ref)]
                    selected = select_stochastic(view, ref['packet_id'], max_stochastic=10)
                    budget = symbolic_budget(view, selected)
                    eligible = current_eligibility(view, view.state['packets'][ref['packet_id']])
                    if (selected['status'] != 'SELECTED_GEOMETRIC_ENVELOPE'
                            or budget['status'] != 'REGISTERED_SYMBOLIC_UPPER_BOUND'
                            or selected['required_participants'] != original['required_participants']
                            or len(selected['required_participants']) != original['required_actor_count']
                            or len(view.state['packets']) != original['frame_actor_count']
                            or eligible != original['eligibility']
                            or any(not eligible[r['head']] for r in case['references'])):
                        raise ValueError('Selected complete current domain or original head eligibility changed')
                    selection_file = f'case{index:03d}_selection_server_only.json'
                    save(output/selection_file, dict(selection=selected, internal_bindings=view.internal_bindings))
                    reused = key in reusable
                    count = len(selected['required_participants'])
                    row = dict(case, current_cache=current_file, selection_cache=selection_file,
                        required_participants=selected['required_participants'], required_actor_count=count,
                        frame_actor_count=len(view.state['packets']), eligibility=eligible,
                        census_required_set_preserved=True, reuse_pilot_pool=reused,
                        particle_seed=20261112+100000*index,
                        particle_full_forward_ceiling=0 if reused else PARTICLES,
                        canonical_feature_ceiling=0 if reused else PARTICLES*count)
                    if reused:
                        row.update(reusable[key])
                    rows.append(row)
                print(json.dumps(dict(stage='first_fit_current_registered', seed=seed, time_sec=t,
                    registered_cases=len(rows), requested_cases=len(chosen))), flush=True)
        rows.sort(key=lambda row: row['case_index'])
        new = [row for row in rows if not row['reuse_pilot_pool']]
        if len(rows) != len(chosen) or len(references) != 512 or frames > 512:
            raise ValueError('The complete512-reference registration was not retained')
        physical = sum(row['particle_full_forward_ceiling'] for row in rows)
        features = sum(row['canonical_feature_ceiling'] for row in rows)
        if physical > 131072 or features > 1310720:
            raise ValueError('Registered new computation exceeds the fixed first-fit ceiling')
        result = dict(status='PASS_FROZEN_FIRST_FIT_COHORT', cases=rows,
            selected_references=512, unique_current_cases=len(rows), current_frames_decoded=frames,
            interval_actions_decoded=frames, native_next_pairs_formed=0, labels_used=False,
            split_counts=dict(Counter(ref['split'] for ref in references)),
            per_head_counts=dict(Counter(ref['head'] for ref in references)),
            new_particle_cases=len(new), reused_particle_cases=len(rows)-len(new),
            reused_pilot_references=[r for row in rows if row['reuse_pilot_pool'] for r in row['references']],
            particle_full_forward_ceiling=physical, canonical_feature_ceiling=features,
            total_particle_capacity=256*len(rows),
            total_canonical_feature_capacity=256*sum(row['required_actor_count'] for row in rows),
            frozen_before_labels_and_responses=True, frozen_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_FIRST_FIT_CURRENT_REGISTRATION', error=f'{type(error).__name__}: {error}',
            cases=rows, selected_references=len(references), requested_cases=len(chosen),
            current_frames_decoded=frames, labels_used=False)
    result.update(real_record_response_queries=0, new_native_starts=0, new_fits=0,
        optimizer_calls=0, elapsed_sec=time.monotonic()-began)
    save(output/'summary.json', result)
    print(json.dumps({key: result[key] for key in ('status','elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache','census','pilot','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cache,args.census,args.pilot,args.output)
    raise SystemExit(0 if result['status']=='PASS_FROZEN_FIRST_FIT_COHORT' else 2)
