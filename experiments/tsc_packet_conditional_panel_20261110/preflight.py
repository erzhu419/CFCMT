"""Freeze the cap4 full-panel budget before any physical response is made."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from experiments.tsc_packet_conditional_domain_20261110.budget import symbolic_budget
from experiments.tsc_packet_conditional_domain_20261110.current import GeometryView
from experiments.tsc_packet_conditional_domain_20261110.register import RESERVOIR_SEED, selected_current_records
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic


PHYSICAL_CEILING = 10000
LEAF_CEILING = 1000000


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def reference_key(reference):
    return int(reference['seed']), int(reference['time_sec']), reference['packet_id']


def run(cache, old_registration, cohort, synthetic_result, output):
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    try:
        proof = json.loads(synthetic_result.read_text())
        if proof['status'] != 'PASS_SYNTHETIC_CAP4_STREAMING':
            raise ValueError('The exact4-actor streaming proof must precede real-state preflight')
        refs = json.loads(cohort.read_text())['references']
        previous = json.loads(old_registration.read_text())
        old_rows = {reference_key(row['reference']): row for row in previous['cases']}
        if len(refs) != 9 or {reference_key(r) for r in refs} != set(old_rows):
            raise ValueError('The complete original9-case cohort must remain unchanged')
        if any(set(r) != {'seed', 'time_sec', 'packet_id'} for r in refs):
            raise ValueError('No label or witness selection may enter preflight')
        static = json.loads((cache/'static_graph_server_only.json').read_text())
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        grouped = defaultdict(dict)
        for ref in refs:
            grouped[ref['seed']][int(ref['time_sec'])] = ref
        rows = []
        for seed, cases in sorted(grouped.items()):
            for frame, action, unused in selected_current_records(cache/'training'/f'seed{seed}', cases):
                ref = cases[int(frame['time_s'])]
                view = GeometryView(static, frame, laws, action['applied_states'], RESERVOIR_SEED+seed*100000+int(frame['time_s']))
                selected = select_stochastic(view, ref['packet_id'], max_stochastic=4)
                old = old_rows[reference_key(ref)]
                required = selected['required_participants']
                bounds = json.loads(json.dumps({key: selected['bounds'][key] for key in required}))
                same = (required == old['required_participants'] and bounds == old['participant_bounds']
                        and len(view.state['packets']) == old['frame_actor_count'])
                budget = symbolic_budget(view, selected)
                stem = f'seed{seed}_t{int(frame["time_s"])}'
                save(output/(stem+'_current_server_only.json'), dict(reference=ref, frame=frame, action=action))
                save(output/(stem+'_selection_server_only.json'), dict(selection=selected, internal_bindings=view.internal_bindings))
                row = dict(reference=ref, frame_actor_count=len(view.state['packets']),
                    required_participants=required, required_actor_count=len(required),
                    selection_status=selected['status'], original_required_set_and_bounds_preserved=same,
                    symbolic_budget=budget, current_cache=stem+'_current_server_only.json')
                rows.append(row)
                print(json.dumps(dict(stage='cap4_budget_registered', reference=ref, actors=len(required),
                    unchanged=same, physical_upper=budget['physical_resolver_calls_upper_bound'])), flush=True)
        all_defined = len(rows) == 9 and all(row['original_required_set_and_bounds_preserved'] and
            row['symbolic_budget']['status'] == 'REGISTERED_SYMBOLIC_UPPER_BOUND' for row in rows)
        physical = sum(r['symbolic_budget']['physical_resolver_calls_upper_bound'] for r in rows) if all_defined else None
        leaves = sum(r['symbolic_budget']['lateral_leaf_branches_upper_bound'] for r in rows) if all_defined else None
        passed = all_defined and physical <= PHYSICAL_CEILING and leaves <= LEAF_CEILING
        result = dict(status='PASS_FROZEN_CAP4_FULL9_BUDGET' if passed else 'HOLD_CAP4_FULL9_PREFLIGHT',
            cases=rows, complete9_domain_registered=all_defined,
            full_frame_forward_calls_upper_bound=physical, lateral_leaf_upper_bound=leaves,
            full_frame_forward_ceiling=PHYSICAL_CEILING, lateral_leaf_ceiling=LEAF_CEILING,
            max_stochastic_actors=4, selector_changed=False, coefficients='New model allzero; one candidate only.',
            frozen_before_real_responses=True, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
            real_record_response_queries=0, new_native_starts=0, new_fits=0, optimizer_calls=0,
            synthetic_proof_status=proof['status'], elapsed_sec=time.monotonic()-began)
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_CAP4_FULL9_PREFLIGHT', error=f'{type(error).__name__}: {error}',
            real_record_response_queries=0, new_native_starts=0, new_fits=0, optimizer_calls=0,
            elapsed_sec=time.monotonic()-began)
    save(output/'summary.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache', 'old-registration', 'cohort', 'synthetic-result', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cache, args.old_registration, args.cohort, args.synthetic_result, args.output)
    raise SystemExit(0 if result['status'] == 'PASS_FROZEN_CAP4_FULL9_BUDGET' else 2)
