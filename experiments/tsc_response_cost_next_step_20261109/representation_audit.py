"""Zero-fit within-group alias and hold/nominal-clock audit of frozen results."""
import argparse
from itertools import combinations
import json
from math import comb
from pathlib import Path
from statistics import mean
import sys

import numpy as np


def _classes(values):
    classes = []
    for index, value in enumerate(values):
        existing = next((group for group in classes if np.array_equal(value, values[group[0]])), None)
        if existing is None:
            classes.append([index])
        else:
            existing.append(index)
    return classes


def _alias_summary(values, costs):
    aliases = [group for group in _classes(values) if len(group) > 1]
    ranges = [float(max(costs[index] for index in group)-min(costs[index] for index in group)) for group in aliases]
    return dict(unique_classes=len(_classes(values)), alias_classes=len(aliases),
                aliased_action_pairs=sum(comb(len(group), 2) for group in aliases),
                classes_with_different_cost=sum(value > 0 for value in ranges),
                max_class_cost_range=max(ranges, default=0.))


def _choose(scores, pp, candidates=None):
    candidates = list(range(8)) if candidates is None else list(candidates)
    rounded = np.round(scores, 12)
    lowest = min(rounded[index] for index in candidates)
    ties = [index for index in candidates if rounded[index] == lowest]
    return pp if pp in ties else min(ties)


def _actual_state(endpoint, inputs):
    result = {}
    for node in inputs['lane_nodes']:
        row = endpoint['lanes'][node['lane_id']]
        n, storage = row['vehicle_count'], node['storage']
        moment = n*row['mean_speed_mps']/node['speed_limit_mps'] if n else 0.
        result[node['lane_id']] = np.asarray([n/storage, row['halted_count']/storage,
                                             moment/storage, row['occupancy_raw']], dtype=float)
    return result


def _correlation(rows, x, y):
    left, right = np.asarray([row[x] for row in rows]), np.asarray([row[y] for row in rows])
    return float(np.corrcoef(left, right)[0, 1]) if left.std() and right.std() else None


def _alias_evidence(group, branches, futures, costs, saved, a, b):
    """Keep only three measured alias pairs' phase and continuation metadata."""
    tls = group['case']['tls_id']
    endpoint_a, endpoint_b = (futures[index]['endpoint_states']['10'] for index in (a, b))
    executor_a, executor_b = (endpoint['executors'][tls]['snapshot'] for endpoint in (endpoint_a, endpoint_b))
    phase_fields = ('mode', 'current_state', 'current_green_state', 'target_green_state',
                    'green_elapsed_sec', 'remaining_sec')
    request_difference, accepted_difference = None, None
    for left, right in zip(futures[a]['execution_trace'], futures[b]['execution_trace']):
        if left['offset_sec'] != right['offset_sec']:
            raise ValueError('Paired continuation request clocks differ')
        if left['offset_sec'] < 10:
            continue
        differing = [key for key in left['requests'] if left['requests'][key] != right['requests'][key]]
        if request_difference is None and differing:
            request_difference = dict(offset_sec=left['offset_sec'], tls_ids=differing,
                focal_requests=[left['requests'][tls], right['requests'][tls]],
                focal_returns=[left['request_returns'][tls], right['request_returns'][tls]])
        if (accepted_difference is None and (left['request_returns'][tls] or right['request_returns'][tls]) and
                (left['requests'][tls] != right['requests'][tls] or
                 left['request_returns'][tls] != right['request_returns'][tls])):
            accepted_difference = dict(offset_sec=left['offset_sec'],
                focal_requests=[left['requests'][tls], right['requests'][tls]],
                focal_returns=[left['request_returns'][tls], right['request_returns'][tls]])
    def lanes_equal(lanes):
        return all(endpoint_a['lanes'][lane] == endpoint_b['lanes'][lane] for lane in lanes)
    inputs = branches[a]['inputs']
    ordered_costs = sorted(set(map(float, costs)))
    return dict(PP_index=group['pp_index'], includes_PP=group['pp_index'] in (a, b),
        S_selected_action=saved['S']['selected_action'], S_selected_is_in_pair=saved['S']['selected_action'] in (a, b),
        observed_cost_ranks=[ordered_costs.index(float(costs[index]))+1 for index in (a, b)],
        contains_observed_minimum=any(costs[index] == min(costs) for index in (a, b)),
        contains_observed_second_cost=any(ordered_costs.index(float(costs[index])) == 1 for index in (a, b)),
        nominal_executor10_green_states=[branches[index]['inputs']['nominal_executor10']['snapshot']['current_green_state'] for index in (a, b)],
        actual_executor10=[{key: snapshot[key] for key in phase_fields} for snapshot in (executor_a, executor_b)],
        actual10_receiving_states_equal=lanes_equal(inputs['receiving_nodes']),
        actual10_internal_states_equal=lanes_equal(inputs['internal_nodes']),
        actual10_all_observed_lane_states_equal=lanes_equal(endpoint_a['lanes']),
        first_post10_request_difference=request_difference,
        first_post10_accepted_focal_request_difference=accepted_difference)


def run(study_root, output):
    study_root = Path(study_root)
    sys.path.insert(0, str(study_root))
    import numeric_projection as projection

    data = json.loads((study_root/'training_dataset_server_only.json').read_text())
    frozen = json.loads((study_root/'nested_fit/result.json').read_text())
    prior_diagnostic = json.loads((study_root/'nested_fit/zero_fit_diagnostic.json').read_text())
    old_blocks = {row['group_id']: row for row in prior_diagnostic['records']}
    targets = sorted((group for group in data['groups'] if group['city'] == 'jinan'),
                     key=lambda group: (group['seed'], group['anchor']))
    if len(targets) != 20 or len({group['seed'] for group in targets}) != 10:
        raise ValueError('Representation audit requires the unchanged twenty target states')
    readers = {}
    for entry in frozen['fit_ledger']:
        if entry['role'] == 'target_cost_reader':
            with np.load(study_root/'nested_fit'/entry['model_file']) as arrays:
                readers[entry['outer_fold'], entry['arm']] = arrays['coef']/arrays['scale']
    clock_columns = np.asarray([*range(8), *(15+15*region+column for region in range(12) for column in range(8))])
    exposure_columns = np.asarray([*range(8, 15), *(15+15*region+column for region in range(12) for column in range(8, 15))])
    representations = ('nominal_clock', 'u', 'actual_N_H', 'actual_full_state', 'actual_Phi',
                       'u_and_actual_state', 'u_and_actual_Phi', 'full_nominal_joint')
    records, contrasts, alias_examples, compression_examples = [], [], [], []
    max_label_reconstruction_error = 0.
    for group in targets:
        identity = f"jinan/seed{group['seed']}/anchor{group['anchor']}"
        saved = {row['arm']: row for row in frozen['records'] if row['group_id'] == identity}
        branches = sorted(group['branches'], key=lambda branch: branch['candidate_index'])
        if [branch['candidate_index'] for branch in branches] != list(range(8)):
            raise ValueError('Canonical action order changed')
        costs = np.asarray([branch['cost'] for branch in branches])
        pp = group['pp_index']
        holds = [index for index, branch in enumerate(branches) if not branch['inputs']['request_differs']]
        if len(holds) != 1:
            raise ValueError('An eight-action group must contain exactly one unchanged current request')
        hold = holds[0]
        actual, matrices, phi, u, nominal, futures = [], [], [], [], [], []
        for index, branch in enumerate(branches):
            inputs = branch['inputs']
            raw_path = study_root/'measurement'/group['case']['case_id']/branch['raw_directory']/'future_execution_server_only.json'
            future = json.loads(raw_path.read_text())
            futures.append(future)
            endpoint = future['endpoint_states']['10']
            state = _actual_state(endpoint, inputs)
            matrix = np.asarray([state[node['lane_id']] for node in inputs['lane_nodes']])
            baseline = np.asarray([projection._state0(node) for node in inputs['lane_nodes']])
            error = float(np.max(np.abs(matrix-baseline-np.asarray(branch['targets']['y']))))
            max_label_reconstruction_error = max(max_label_reconstruction_error, error)
            if error > 1e-12:
                raise ValueError('Actual endpoint and archived response targets no longer align')
            actual.append(state); matrices.append(matrix)
            phi.append(projection.project_reader(inputs, state)[195:])
            u.append(projection._reader_u(inputs))
            nominal.append(np.asarray(inputs['full_nominal_states']))
        values = dict(nominal_clock=[value[:8] for value in u], u=u,
            actual_N_H=[value[:, :2] for value in matrices], actual_full_state=matrices,
            actual_Phi=phi, u_and_actual_state=[np.r_[a, b.ravel()] for a, b in zip(u, matrices)],
            u_and_actual_Phi=[np.r_[a, b] for a, b in zip(u, phi)], full_nominal_joint=nominal)
        summaries = {name: _alias_summary(value, costs) for name, value in values.items()}
        # Retain measured examples, distinguishing compression from identical raw states.
        for a, b in combinations(range(8), 2):
            if np.array_equal(phi[a], phi[b]) and not np.array_equal(matrices[a], matrices[b]):
                compression_examples.append(dict(group_id=identity, actions=[a, b],
                    full_state_max_abs_difference=float(np.max(np.abs(matrices[a]-matrices[b]))),
                    u_equal=bool(np.array_equal(u[a], u[b])), cost_difference=float(costs[a]-costs[b])))
            if np.array_equal(values['u_and_actual_Phi'][a], values['u_and_actual_Phi'][b]) and costs[a] != costs[b]:
                alias_examples.append(dict(group_id=identity, actions=[a, b],
                    actual_full_state_equal=bool(np.array_equal(matrices[a], matrices[b])),
                    nominal_joint_equal=bool(np.array_equal(nominal[a], nominal[b])),
                    cost_minus_PP=[float(costs[a]-costs[pp]), float(costs[b]-costs[pp])],
                    absolute_cost_difference=float(abs(costs[a]-costs[b])),
                    nominal_last_states=[str(nominal[a][-1]), str(nominal[b][-1])],
                    **_alias_evidence(group, branches, futures, costs, saved, a, b)))
        arm_rows = {}
        g0 = projection._reader_g0(branches[pp]['inputs'])
        du = np.asarray(u)-u[pp]
        nominal_features = np.c_[du, np.asarray([np.outer(g0, value).ravel() for value in du])]
        for arm in ('T', 'S', 'P'):
            entry = saved[arm]
            scores = np.asarray(entry['predicted_delta_costs'])
            if _choose(scores, pp) != entry['selected_action']:
                raise ValueError('Frozen saved score selection changed')
            coefficients = readers[entry['outer_fold'], arm]
            clock = nominal_features[:, clock_columns]@coefficients[clock_columns]
            exposure = nominal_features[:, exposure_columns]@coefficients[exposure_columns]
            selected = entry['selected_action']
            previous = old_blocks[identity]['arms'][arm]['blocks']
            if abs(clock[selected]+exposure[selected]-previous['nominal']-previous['global_nominal']) > 1e-12:
                raise ValueError('Nominal coefficient partition differs from the prior frozen-score audit')
            competitor = _choose(scores, pp, [index for index in range(8) if index != hold])
            margin = float(scores[hold]-scores[competitor])
            clock_margin = float(clock[hold]-clock[competitor])
            exposure_margin = float(exposure[hold]-exposure[competitor])
            arm_rows[arm] = dict(selected_action=selected, selected_is_hold=selected == hold,
                selected_minus_PP=float(costs[selected]-costs[pp]),
                clock_block_only_choice_is_hold=_choose(clock, pp) == hold,
                nominal_blocks_only_choice_is_hold=_choose(clock+exposure, pp) == hold,
                best_nonhold_action=competitor, hold_minus_best_nonhold_predicted_margin=margin,
                clock_margin=clock_margin, halted_exposure_margin=exposure_margin,
                predicted_Phi_margin=margin-clock_margin-exposure_margin,
                clock_favors_hold=clock_margin < 0)
        minimum = float(costs.min())
        best = [index for index, cost in enumerate(costs) if cost == minimum]
        row = dict(group_id=identity, seed=group['seed'], anchor=group['anchor'], pp_index=pp,
            hold_index=hold, PP_is_hold=pp == hold, observed_min_actions=best,
            observed_min_contains_hold=hold in best, hold_minus_PP=float(costs[hold]-costs[pp]),
            hold_minus_observed_min=float(costs[hold]-minimum), representations=summaries, arms=arm_rows,
            switch_clock_unique_classes=len(_classes([u[index][:8] for index in range(8) if index != hold])))
        records.append(row)
        for action in range(8):
            if action == pp:
                continue
            contrasts.append(dict(group_id=identity, seed=group['seed'], action=action,
                delta_mean_actual_rhoH10=float(phi[action][4]-phi[pp][4]),
                delta_cost450=float(costs[action]-costs[pp])))
    summary = {name: dict(groups_with_alias=sum(row['representations'][name]['alias_classes'] > 0 for row in records),
        alias_action_pairs=sum(row['representations'][name]['aliased_action_pairs'] for row in records),
        groups_with_same_representation_different_cost=sum(row['representations'][name]['classes_with_different_cost'] > 0 for row in records),
        max_class_cost_range=max(row['representations'][name]['max_class_cost_range'] for row in records))
        for name in representations}
    misleading = [row for row in contrasts if row['delta_mean_actual_rhoH10'] < 0 and row['delta_cost450'] > 0]
    result = dict(status='ZERO_FIT_REPRESENTATION_AUDIT', new_fits=0, new_native_starts=0,
        target_states=len(records), target_seeds=10, actual_endpoint_branches=160,
        endpoint_target_reconstruction_max_error=max_label_reconstruction_error,
        PP_hold_states=sum(row['PP_is_hold'] for row in records),
        PP_switch_states=sum(not row['PP_is_hold'] for row in records),
        observed_min_contains_hold_states=sum(row['observed_min_contains_hold'] for row in records),
        states_with_one_common_switch_clock=sum(row['switch_clock_unique_classes'] == 1 for row in records),
        arm_summary={arm: dict(selected_hold_states=sum(row['arms'][arm]['selected_is_hold'] for row in records),
            clock_block_only_hold_states=sum(row['arms'][arm]['clock_block_only_choice_is_hold'] for row in records),
            nominal_blocks_only_hold_states=sum(row['arms'][arm]['nominal_blocks_only_choice_is_hold'] for row in records),
            clock_favors_hold_over_best_switch_states=sum(row['arms'][arm]['clock_favors_hold'] for row in records),
            mean_clock_margin=mean(row['arms'][arm]['clock_margin'] for row in records),
            mean_halted_exposure_margin=mean(row['arms'][arm]['halted_exposure_margin'] for row in records),
            mean_predicted_Phi_margin=mean(row['arms'][arm]['predicted_Phi_margin'] for row in records)) for arm in ('T', 'S', 'P')},
        exact_alias_summary=summary,
        empirical_Phi_compression=dict(equal_Phi_but_different_lane_state_pairs=len(compression_examples),
            examples=compression_examples[:8]),
        equal_true_reader_representation_different_cost_examples=sorted(alias_examples,
            key=lambda row: -row['absolute_cost_difference'])[:8],
        actual_endpoint_rhoH10_vs_cost=dict(nonPP_action_pairs=len(contrasts),
            descriptive_pearson=_correlation(contrasts, 'delta_mean_actual_rhoH10', 'delta_cost450'),
            lower_local_halted_density_but_higher_cost=len(misleading), examples=misleading[:8]),
        records=records,
        interpretation='Exact aliases concern observed same-group representations. Component-only choices are algebraic diagnostics, not newly fitted/deployed policies. Repeated actions are correlated; no iid intervals or new performance gate. Phi is permutation-invariant by construction, but theoretical loss and observed compression aliases are counted separately. One realized future per action does not establish irreducible expected-cost variance.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.study_root, args.output)
    print(json.dumps({key: result[key] for key in ('status', 'target_states', 'PP_hold_states',
        'arm_summary', 'exact_alias_summary', 'empirical_Phi_compression', 'actual_endpoint_rhoH10_vs_cost')}, allow_nan=False))
