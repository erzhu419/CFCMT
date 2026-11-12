"""Compact measurement receipts and observed post-action execution differences."""
import argparse
import json
from pathlib import Path


def _read(path):
    return json.loads(path.read_text())


def _focal_t10(future, tls):
    if future is None or '10' not in future['endpoint_states']:
        return None
    row = future['endpoint_states']['10']['executors'][tls]
    snapshot = {key: value for key, value in row['snapshot'].items() if key != 'audit'}
    return dict(operational_snapshot=snapshot, feasible_state_count=len(row['feasible_states']),
                minimum_green_sec=row['minimum_green_sec'], mode=snapshot['mode'],
                green_age_sec=snapshot['green_elapsed_sec'])


def _request_difference(future, pp):
    if future is None or pp is None:
        return None
    reference = {row['offset_sec']: row for row in pp['execution_trace'] if row['offset_sec'] >= 10}
    rows = [row for row in future['execution_trace'] if row['offset_sec'] >= 10]
    differences = [row for row in rows if row['offset_sec'] in reference and
                   row['requests'] != reference[row['offset_sec']]['requests']]
    return dict(compared_epochs=sum(row['offset_sec'] in reference for row in rows),
        different_epochs=len(differences),
        first_offset_sec=differences[0]['offset_sec'] if differences else None,
        first_time_sec=differences[0]['time_sec'] if differences else None,
        role='observed_future_execution_label')


def run(base):
    base = Path(base)
    receipt = _read(base/'result.json')
    cities = {}
    for group in receipt['groups']:
        case, reference = group['case'], group['reference_index']
        directory = base/'measurement'/case['case_id']
        futures = {}
        for branch in group['branches']:
            index = branch['candidate_index']
            path = directory/('first_pp' if index == reference else f'action_{index}')/'future_execution_server_only.json'
            futures[index] = _read(path) if path.exists() else None
        pp = futures.get(reference)
        pp_branch = next(branch for branch in group['branches'] if branch['candidate_index'] == reference)
        pp_t10 = _focal_t10(pp, case['tls_id'])
        actions = []
        for branch in sorted(group['branches'], key=lambda row: row['candidate_index']):
            index, future = branch['candidate_index'], futures[branch['candidate_index']]
            t10 = _focal_t10(future, case['tls_id'])
            if t10 is not None:
                t10['differs_from_pp'] = (t10['operational_snapshot'] != pp_t10['operational_snapshot']) if pp_t10 else None
            execution = branch.get('execution_audit', {}).get('execution')
            cost, pp_cost = branch.get('cost'), pp_branch.get('cost')
            checks = branch.get('checks', {})
            safety = branch.get('safety')
            actions.append(dict(candidate_index=index, candidate_state=group['candidate_states'][index],
                is_pp=index == reference, status=branch['status'],
                cost=cost, cost_minus_pp=cost-pp_cost if cost is not None and pp_cost is not None else None,
                integrals=branch.get('integrals'), segments=branch.get('segments'),
                clock=dict(start_time_sec=future['start_time_sec'], end_time_sec=future['end_time_sec'],
                           completed_steps=future['completed_steps']) if future is not None else None,
                all_checks_passed=bool(checks) and all(checks.values()),
                failed_checks=[key for key, value in checks.items() if not value],
                collision_teleport_counts={key: safety[key] for key in
                    ('raw_collision_events', 'collision_event_steps', 'unique_collision_incidents',
                     'starting_teleports', 'ending_teleports')} if safety is not None else None,
                t10_focal=t10, later_requests_vs_pp=_request_difference(future, pp),
                clearance_execution_10=dict(matches=execution['matches'],
                    extension_step_indices_0based=execution['clearance_extension_steps'],
                    extension_step_count=len(execution['clearance_extension_steps']),
                    nominal_mismatch_step_indices_0based=execution['nominal_mismatch_steps']) if execution is not None else None))
        cities[case['city']] = dict(case_id=case['case_id'], seed=case['seed'], focal_tls=case['tls_id'],
            status=group['status'], resolved_checkpoint_sec=group['resolved_checkpoint_sec'],
            reference_index=reference, controlled_lane_count=group['controlled_lane_count'],
            attempted_branches=group['new_branches_attempted'],
            all_eight_complete=len(actions) == 8 and all(row['status'] == 'COMPLETE' for row in actions),
            all_checks_passed=bool(actions) and all(row['all_checks_passed'] for row in actions), actions=actions)
    compact = dict(status=receipt['status'], new_fits=receipt['new_fits'],
        new_branches_attempted=receipt['new_branches_attempted'], measurement_only=True,
        cost_semantics='Mean halted vehicles per controlled lane over 450s; not trip waiting time.',
        execution_semantics='Future observed paths only; all eight actions retained without chooser evaluation.', cities=cities)
    (base/'measurement_summary.json').write_text(json.dumps(compact, indent=2, allow_nan=False)+'\n')
    return compact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base)
    print(result['status'])
