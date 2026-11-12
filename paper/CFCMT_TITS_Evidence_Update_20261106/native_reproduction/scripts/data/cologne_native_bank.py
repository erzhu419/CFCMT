"""Build B100 labels from native PP prefixes, without SUMO snapshot restoration."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


PROTOCOL = 'tsc-v154p-cologne-native-prefix-b100-v1'
B195_PROTOCOL = 'tsc-v155k-cologne-native-prefix-b195-v1'
TARGET = 'prefix_mean_cost_450s'
AGGREGATES = {
    'total_queue': 'total_q', 'green_queue': 'green_q', 'red_queue': 'red_q',
    'downstream_occupancy': 'green_down_occ', 'mean_speed': 'mean_speed',
}
TARGETS = (*('one_step_' + name for name in AGGREGATES),
           *('next_' + name for name in AGGREGATES),
           'interval_cost', 'terminal_system_load', TARGET)
ROW_METADATA = ('action_group_ids', 'row_tls', 'row_times', 'candidate_states')
# These describe the unchanged network, features, priors, and estimand. Old
# collection traces, snapshot/replay claims, and safety counts do not transfer.
CONTRACT_METADATA = (
    'scenario', 'sumocfg', 'occupancy_equation_protocol', 'control_interval_sec',
    'warmup_sec', 'counterfactual_horizon_intervals', 'counterfactual_horizon_sec',
    'local_mechanism_horizon_sec', 'rollout_value_horizon_sec',
    'rollout_prefix_horizons_sec', 'counterfactual_estimand',
    'counterfactual_cost_mode', 'counterfactual_cost_scope',
    'counterfactual_cost_population', 'counterfactual_cost_normalization',
    'mechanism_output_names', 'mechanism_estimands', 'sumo_execution_protocol',
    'strict_safety_monitoring', 'behavior_policy', 'signal_routing_graph',
)


def _safe(audit):
    return (audit.get('raw_collision_events') == 0
            and audit.get('starting_teleports') == 0
            and audit.get('ending_teleports') == 0
            and audit.get('no_teleport_passed') is True)


def build_native_group(original: MechanismDataset, group_result: Mapping,
                       cost_traces: Mapping, *, native_protocol=PROTOCOL) -> MechanismDataset:
    """Replace all 13 targets in one original group using its eight native runs.

    A V154O outer status may be INVALID solely because its restored control
    failed. Only the native branches establish this group's label validity.
    P retains its safety gate; K retains collisions and requires complete valid rows.
    """
    if native_protocol not in (PROTOCOL, B195_PROTOCOL):
        raise ValueError('Use the original P or the fixed K native collection protocol')
    diagnostic_collisions = native_protocol == B195_PROTOCOL
    group = group_result['group_id']
    rows = np.flatnonzero(np.asarray(original.metadata['action_group_ids']) == group)
    phases = np.asarray(original.metadata['candidate_states'])[rows].tolist()
    if len(rows) != 8 or len(set(phases)) != 8:
        raise ValueError(f'{group}: original group must contain eight distinct actions')
    if group_result['candidate_states'] != phases:
        raise ValueError(f'{group}: native candidate order differs from original rows')
    times = np.asarray(original.metadata['row_times'])[rows]
    if not np.all(times == group_result['time_sec']):
        raise ValueError(f'{group}: native capture time differs from original rows')
    if set(original.targets) != set(TARGETS) or set(original.priors) != set(TARGETS):
        raise ValueError('Expected the original thirteen-target/prior schema')
    branches = group_result['native_branches']
    if len(branches) != 8:
        raise ValueError(f'{group}: incomplete native action group')
    targets = {name: [] for name in TARGETS}
    for index, (phase, branch) in enumerate(zip(phases, branches)):
        if (branch.get('candidate_index') != index or branch.get('phase') != phase
                or branch.get('status') != ('COMPLETE' if diagnostic_collisions else 'PASS') or not branch.get('checks')
                or not all(branch['checks'].values())
                or any(not (audit.get('no_teleport_passed') is True
                            and audit.get('starting_teleports') == audit.get('ending_teleports') == 0)
                       if diagnostic_collisions else not _safe(audit)
                       for audit in (branch.get('prefix_safety', {}), branch.get('safety', {})))):
            raise ValueError(f'{group}: native branch {index} is unsafe, invalid, or misaligned')
        costs = np.asarray(cost_traces[f'{group}:native:{index}'], dtype=float)
        if (costs.shape != (450,) or not np.all(np.isfinite(costs))
                or np.any(costs < 0) or branch.get('sample_count') != 450
                or branch.get('end_time_sec') != group_result['time_sec'] + 450
                or not np.array_equal(costs[:10], branch['first_interval_costs'])
                or abs(float(np.mean(costs)) - branch['cost']) > 1e-12):
            raise ValueError(f'{group}: native branch {index} trace does not match its outcome')
        for prefix, field in (('one_step_', 'one_step'), ('next_', 'terminal')):
            for suffix, aggregate in AGGREGATES.items():
                targets[prefix + suffix].append(branch[field][aggregate])
        targets['interval_cost'].append(float(np.mean(costs)))
        targets[TARGET].append(float(np.mean(costs)))
        targets['terminal_system_load'].append(float(costs[-1]))
    target_arrays = {name: np.asarray(values, dtype=float) for name, values in targets.items()}
    if any(not np.all(np.isfinite(values)) for values in target_arrays.values()):
        raise ValueError(f'{group}: nonfinite native mechanism outcome')
    metadata = {key: original.metadata[key] for key in CONTRACT_METADATA
                if key in original.metadata}
    metadata.update({key: np.asarray(original.metadata[key])[rows].tolist()
                     for key in ROW_METADATA})
    metadata.update({
        'protocol': native_protocol, 'native_bank_protocol': native_protocol,
        'state_restores': 0,
        'native_prefix_protocol': {
            'start': 'fresh_sumo_start_with_original_seed',
            'prefix_policy': 'phase_pressure', 'snapshot_restore_used': False,
            'first_interval_sec': 10, 'continuation_policy': 'phase_pressure',
            'horizon_sec': 450, 'cost_samples': 450, 'controlled_lane_count': 8,
            'cost_aggregation': 'mean_native_halted_vehicles_per_controlled_lane',
            'features_context_priors': ('unchanged_original_candidate_rows' if diagnostic_collisions
                                        else 'unchanged_original_B100_rows'),
        },
        'native_group_sources': [{
            'group_id': group, 'source_group_status': group_result.get('status'),
            'bank_path': group_result.get('bank_path'),
            'source_result_path': group_result.get('source_result_path'),
            'reused_v154o': bool(group_result.get('reused', False)),
        }],
        'native_collection_audit': {'valid_groups': [group], 'failed_groups': [],
                                    'retained_groups': 1, 'retained_rows': 8},
        'fit_ready': False,
    })
    if diagnostic_collisions:
        metadata['native_prefix_protocol']['collision_accounting'] = 'diagnostic_not_admission_gate'
        metadata['strict_safety_monitoring'] = {**metadata.get('strict_safety_monitoring', {}),
            'counterfactual_collision_handling': 'retain_complete_matched_action_group_and_audit_collisions'}
        source = metadata['native_group_sources'][0]
        source['native_bank_protocol'] = B195_PROTOCOL
        source['collision_admission_gate'] = False
        source['branch_safety'] = [{
            'candidate_index': branch['candidate_index'],
            **{f'{prefix}_{key}': branch[field][key]
               for prefix, field in (('prefix', 'prefix_safety'), ('window', 'safety'))
               for key in ('raw_collision_events', 'unique_collision_incidents', 'no_teleport_passed')},
        } for branch in branches]
    return MechanismDataset(
        feature_names=original.feature_names, features=original.features[rows].copy(),
        context_names=original.context_names, context=original.context[rows].copy(),
        priors={name: np.asarray(values)[rows].copy() for name, values in original.priors.items()},
        targets=target_arrays, domains=original.domains[rows].copy(), metadata=metadata)


def _validate_native_rows(data: MechanismDataset, native_protocol=PROTOCOL):
    if data.metadata.get('native_bank_protocol') != native_protocol:
        raise ValueError('Only the native-prefix bank protocol is accepted')
    if (data.metadata.get('state_restores') != 0
            or data.metadata.get('native_prefix_protocol', {}).get('snapshot_restore_used') is not False
            or 'state_snapshot_protocol' in data.metadata
            or 'counterfactual_replay_audit' in data.metadata
            or 'counterfactual_safety_audit' in data.metadata):
        raise ValueError('Native labels cannot carry an old snapshot collection contract')
    if set(data.targets) != set(TARGETS) or set(data.priors) != set(TARGETS):
        raise ValueError('Native bank target/prior schema changed')
    for key in ROW_METADATA:
        if np.asarray(data.metadata[key]).shape != (data.size,):
            raise ValueError(f'Native bank {key} is not row aligned')
    groups = np.asarray(data.metadata['action_group_ids'])
    phases = np.asarray(data.metadata['candidate_states'])
    for group in set(groups):
        selected = groups == group
        if int(selected.sum()) != 8 or len(set(phases[selected])) != 8:
            raise ValueError(f'{group}: incomplete or duplicate native candidate rows')
    arrays = [data.features, data.context, *data.targets.values(), *data.priors.values()]
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError('Native bank contains nonfinite training data')
    if not np.array_equal(data.targets[TARGET], data.targets['interval_cost']):
        raise ValueError('Native 450-second target differs from interval_cost')


def assemble_native_bank(parts: Sequence[MechanismDataset], expected_groups: Sequence[str],
                         failed_groups: Sequence[str] = (), *, native_protocol=PROTOCOL) -> MechanismDataset:
    """Materialize complete safe groups; a partial bank remains explicitly unfit."""
    if not parts:
        raise ValueError('No complete native groups to assemble')
    expected, explicit_failed = list(expected_groups), list(failed_groups)
    if len(expected) != len(set(expected)) or len(explicit_failed) != len(set(explicit_failed)):
        raise ValueError('The frozen or failed group roster contains duplicates')
    failed = list(dict.fromkeys([*explicit_failed, *(group for part in parts
        for group in part.metadata.get('native_collection_audit', {}).get('failed_groups', []))]))
    base, row_groups = parts[0], []
    for part in parts:
        _validate_native_rows(part, native_protocol)
        if (part.feature_names != base.feature_names or part.context_names != base.context_names
                or any(part.metadata.get(key) != base.metadata.get(key)
                       for key in (*CONTRACT_METADATA, 'native_prefix_protocol'))):
            raise ValueError('Native group feature/context or collection contract changed')
        row_groups.extend(part.metadata['action_group_ids'])
    counts = Counter(row_groups)
    if any(count != 8 for count in counts.values()):
        raise ValueError('Native group parts overlap or contain partial groups')
    if not set(counts) <= set(expected) or not set(failed) <= set(expected):
        raise ValueError('Native group is outside the frozen roster')
    if set(counts) & set(failed):
        raise ValueError('A failed group cannot contribute rows to the native bank')
    # Preserve the preselected roster's order and each original group's phase order.
    row_groups = np.asarray(row_groups)
    order = np.concatenate([np.flatnonzero(row_groups == group)
                            for group in expected if group in counts])
    metadata = dict(base.metadata)
    for key in ROW_METADATA:
        values = [value for part in parts for value in part.metadata[key]]
        metadata[key] = np.asarray(values)[order].tolist()
    valid = [group for group in expected if group in counts]
    missing = [group for group in expected if group not in counts]
    metadata.update({
        'selected_action_group_ids': expected,
        'fit_ready': len(expected) == (195 if native_protocol == B195_PROTOCOL else 100) and not missing,
        'native_group_sources': [source for part in parts for source in part.metadata['native_group_sources']],
        'native_collection_audit': {'valid_groups': valid, 'failed_groups': failed,
            'missing_groups': missing, 'retained_groups': len(valid), 'retained_rows': len(order),
            'expected_groups': len(expected), 'complete_frozen_roster': not missing},
    })
    return MechanismDataset(
        feature_names=base.feature_names,
        features=np.concatenate([part.features for part in parts])[order],
        context_names=base.context_names,
        context=np.concatenate([part.context for part in parts])[order],
        priors={name: np.concatenate([part.priors[name] for part in parts])[order] for name in base.priors},
        targets={name: np.concatenate([part.targets[name] for part in parts])[order] for name in base.targets},
        domains=np.concatenate([part.domains for part in parts])[order], metadata=metadata)


def assert_complete_b100(data: MechanismDataset, expected_groups: Sequence[str]) -> None:
    """The original model fit may run only after all original 100 groups exist."""
    _validate_native_rows(data)
    expected = list(expected_groups)
    actual = list(dict.fromkeys(data.metadata['action_group_ids']))
    audit = data.metadata['native_collection_audit']
    if (len(expected) != 100 or len(set(expected)) != 100 or data.size != 800
            or actual != expected or data.metadata.get('selected_action_group_ids') != expected
            or audit.get('failed_groups') or audit.get('missing_groups')
            or audit.get('complete_frozen_roster') is not True
            or data.metadata.get('fit_ready') is not True):
        raise ValueError('Fitting requires the complete unchanged original B100: 100 groups and 800 rows')


def assemble_expanded_native_bank(original_b100, additional_parts, expected_groups):
    """Explicitly combine the unchanged P100 with the complete newly native K95."""
    original_groups = original_b100.metadata['selected_action_group_ids']
    assert_complete_b100(original_b100, original_groups)
    expected = list(expected_groups)
    additional_groups = [group for group in expected if group not in set(original_groups)]
    if (len(expected) != 195 or len(set(expected)) != 195
            or not set(original_groups) <= set(expected) or len(additional_groups) != 95):
        raise ValueError('K must retain the original100 groups and exactly95 distinct additions')
    additional = assemble_native_bank(additional_parts, additional_groups, native_protocol=B195_PROTOCOL)
    if not additional.metadata['native_collection_audit']['complete_frozen_roster']:
        raise ValueError('All95 new native groups must be complete before the B195 fit')
    # This is an explicit reuse of P labels; preserve their original provenance.
    reused = replace(original_b100, metadata={**original_b100.metadata,
        'protocol': B195_PROTOCOL, 'native_bank_protocol': B195_PROTOCOL,
        'native_prefix_protocol': dict(additional.metadata['native_prefix_protocol']),
        'strict_safety_monitoring': dict(additional.metadata['strict_safety_monitoring']),
        'native_group_sources': [{**source, 'native_bank_protocol': PROTOCOL, 'collision_admission_gate': True}
                                 for source in original_b100.metadata['native_group_sources']]})
    combined = assemble_native_bank([reused, additional], expected, native_protocol=B195_PROTOCOL)
    combined.metadata['reused_original_b100_groups'] = list(original_groups)
    combined.metadata['new_native_additional_groups'] = additional_groups
    assert_complete_b195(combined, expected)
    return combined


def assert_complete_b195(data, expected_groups):
    _validate_native_rows(data, B195_PROTOCOL)
    expected = list(expected_groups)
    audit = data.metadata['native_collection_audit']
    if (len(expected) != 195 or len(set(expected)) != 195 or data.size != 1560
            or list(dict.fromkeys(data.metadata['action_group_ids'])) != expected
            or data.metadata.get('selected_action_group_ids') != expected
            or audit.get('failed_groups') or audit.get('missing_groups')
            or audit.get('complete_frozen_roster') is not True or data.metadata.get('fit_ready') is not True
            or len(data.metadata.get('reused_original_b100_groups', [])) != 100
            or len(data.metadata.get('new_native_additional_groups', [])) != 95):
        raise ValueError('Fitting requires the fixed complete B195: P100 plus K95 and1560 rows')
