"""Cologne1-only B100 refit from a fixed, label-independent temporal roster."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import pickle
import time

import numpy as np

_spec = importlib.util.spec_from_file_location('cologne_local_refit_shared', Path(__file__).with_name('run_cologne_fraction_refit.py'))
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)
PREVIOUS_PROTOCOL = shared.PROTOCOL
PROTOCOL = 'tsc-v154j-cologne-local-fraction-refit-v2'
shared.PROTOCOL = PROTOCOL
SCENARIO, TLS = 'cologne1', 'cluster_357187_359543'
TRAIN_SEEDS = (2027, 3037, 4047)
QUOTAS = dict(zip(TRAIN_SEEDS, (34, 33, 33)))
SETTINGS = dict(duration_sec=3600, control_interval_sec=10, warmup_sec=60, max_focal_tls=4,
    counterfactual_horizon_intervals=45, rollout_prefix_horizons_sec=[450],
    behavior_policy='phase_pressure', counterfactual_cost_mode='halted_queue')


def new_group_requests(seed, previous_requested):
    if seed not in TRAIN_SEEDS:
        raise ValueError('Use the three frozen training seeds')
    previous = set(previous_requested)
    return [g for i in range(6, 360, 5)
            if (g := f'{SCENARIO}:seed{seed}:{i}:{TLS}') not in previous]


def load_protocol(path):
    protocol = json.loads(path.read_text())
    if (protocol['protocol'] != PROTOCOL or protocol['collection'] != SETTINGS
            or protocol['frozen_candidate'] != 'constant_alpha_0'
            or len(protocol['selected_groups']) != len(set(protocol['selected_groups']))):
        raise ValueError('Use the frozen Cologne1 local-refit protocol')
    expected = []
    for seed in TRAIN_SEEDS:
        previous = previous_cell(protocol, seed)
        expected.extend(new_group_requests(seed, previous['requested_groups']))
    if set(protocol['selected_groups']) != set(expected):
        raise ValueError('New requests must be the fixed temporal roster minus all prior requests')
    return protocol


def previous_cell(protocol, seed):
    cell = Path(protocol['previous_results_root'])/'collection'/SCENARIO/f'seed_{seed}'
    result = json.loads((cell/'result.json').read_text())
    if ((result['protocol'], result['scenario'], result['seed']) != (PREVIOUS_PROTOCOL, SCENARIO, seed)
            or result['settings'] != SETTINGS or result['sumocfg'] != protocol['sumocfgs'][SCENARIO]):
        raise ValueError('Reuse only the original Cologne1 collection under the same settings and net')
    return result


def audit_dataset(data, seed, requested, sumocfg):
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (merge_counterfactual_datasets_v3,
        STATE_SNAPSHOT_PROTOCOL_V3, SUMO_EXECUTION_PROTOCOL, STRICT_SAFETY_MONITORING_V3)
    from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import _assert_waiting_aligned_bank
    from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL
    meta = data.metadata
    if (meta.get('scenario') != SCENARIO or meta.get('seed') != seed or meta.get('sumocfg') != str(sumocfg)
            or any(meta.get(k) != v for k, v in SETTINGS.items())
            or meta.get('occupancy_equation_protocol') != OCCUPANCY_EQUATION_PROTOCOL
            or meta.get('counterfactual_horizon_sec') != 450
            or meta.get('local_mechanism_horizon_sec') != 10
            or meta.get('rollout_value_horizon_sec') != 450):
        raise ValueError('Collection metadata differs from the frozen fraction/450-second contract')
    for key, expected in (('state_snapshot_protocol', STATE_SNAPSHOT_PROTOCOL_V3),
            ('sumo_execution_protocol', SUMO_EXECUTION_PROTOCOL),
            ('strict_safety_monitoring', STRICT_SAFETY_MONITORING_V3)):
        if meta.get(key) != expected:
            raise ValueError(f'Collection {key} differs from the frozen contract')
    _assert_waiting_aligned_bank({SCENARIO: data}, target_name='prefix_mean_cost_450s', tolerance=1e-12)
    merge_counterfactual_datasets_v3([data])  # Original replay and symmetric-censor validation.
    for audit in (meta['behavior_safety_audit'], meta['counterfactual_safety_audit']):
        if not audit.get('no_teleport_passed') or any(audit.get(k) != 0 for k in ('starting_teleports', 'ending_teleports')):
            raise ValueError('Collection teleport safety contract failed')
    retained = set(meta['action_group_ids'])
    safety = meta['counterfactual_safety_audit']
    censored = set(safety['censored_group_ids'])
    requested = set(requested)
    if (set(meta.get('selected_action_group_ids', ())) != requested
            or retained & censored or not (retained | censored) <= requested
            or len(retained) != safety['retained_groups']
            or len(censored) != safety['censored_groups']):
        raise ValueError('Requested, retained and censored action groups are inconsistent')
    return {'retained_groups': sorted(retained), 'censored_groups': sorted(censored),
            'unavailable_groups': sorted(requested - retained - censored)}


def select_local_b100(group_ids):
    groups = set(group_ids)
    selected, audit = [], {}
    for seed, quota in QUOTAS.items():
        ordered = sorted((g for g in groups if g.startswith(f'{SCENARIO}:seed{seed}:')),
                         key=lambda g: int(g.split(':')[2]))
        if any(g.rsplit(':', 1)[1] != TLS for g in ordered):
            raise ValueError('The candidate pool contains another junction')
        if len(ordered) < quota:
            raise ValueError(f'seed {seed}: {len(ordered)} available groups below fixed quota {quota}')
        positions = np.linspace(0, len(ordered)-1, quota).round().astype(int)
        chosen = [ordered[i] for i in positions]
        selected.extend(chosen)
        audit[str(seed)] = {'pool_groups': len(ordered), 'quota': quota,
                           'selected_groups': chosen, 'selected_positions': positions.tolist()}
    if len(selected) != 100 or len(set(selected)) != 100:
        raise ValueError('The local B100 selection is incomplete')
    return selected, audit


def collect(protocol, seed, output):
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import collect_counterfactual_transitions_v3
    from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
    from cf_h2o.traffic_signal.dataset_cache import save_mechanism_dataset
    started = time.monotonic()
    previous = previous_cell(protocol, seed)
    requested = new_group_requests(seed, previous['requested_groups'])
    frozen = [g for g in protocol['selected_groups'] if g.startswith(f'{SCENARIO}:seed{seed}:')]
    if set(frozen) != set(requested):
        raise ValueError('The frozen additional group roster changed')
    if (output/'bank.npz').exists() or (output/'result.json').exists():
        raise FileExistsError(output)
    if '1.22.0' not in str(libsumo_version()):
        raise ValueError('Use SUMO 1.22.0')
    data = collect_counterfactual_transitions_v3(sumo_api=load_libsumo(),
        sumocfg=Path(protocol['sumocfgs'][SCENARIO]), scenario=SCENARIO, seed=seed,
        selected_action_group_ids=requested, collection_shard_index=0, collection_shard_count=1, **SETTINGS)
    audit = audit_dataset(data, seed, requested, protocol['sumocfgs'][SCENARIO])
    data = replace(data, metadata={**data.metadata, 'fraction_refit_protocol': PROTOCOL})
    output.mkdir(parents=True, exist_ok=True)
    save_mechanism_dataset(output/'bank.npz', data)
    result = {'protocol': PROTOCOL, 'status': 'PASS', 'collection_complete': True,
        'scenario': SCENARIO, 'seed': seed, 'settings': SETTINGS, 'sumocfg': protocol['sumocfgs'][SCENARIO],
        'bank': str(output/'bank.npz'), 'requested_groups': requested, **audit, 'rows': data.size,
        'occupancy_equation_protocol': data.metadata['occupancy_equation_protocol'],
        'behavior_safety_audit': data.metadata['behavior_safety_audit'],
        'counterfactual_safety_audit': data.metadata['counterfactual_safety_audit'],
        'elapsed_sec': time.monotonic()-started}
    shared.write_result(output/'result.json', result)
    return result


def fit(protocol, results_root, output):
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import merge_counterfactual_datasets_v3
    from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import _target_model
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    from cf_h2o.traffic_signal.fraction_target_rigid import MODEL_PROTOCOL
    from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL
    from cf_h2o.traffic_signal.right_of_way_context import (augment_dataset_with_right_of_way_context,
        net_file_from_sumocfg, read_network_right_of_way_context)
    started = time.monotonic(); parts, cells = [], []
    for seed in TRAIN_SEEDS:
        previous = previous_cell(protocol, seed)
        for root, expected, requested in (
                (Path(protocol['previous_results_root']), PREVIOUS_PROTOCOL, previous['requested_groups']),
                (results_root, PROTOCOL, new_group_requests(seed, previous['requested_groups']))):
            cell = root/'collection'/SCENARIO/f'seed_{seed}'
            result = json.loads((cell/'result.json').read_text())
            if (result['protocol'] != expected or (result['scenario'], result['seed']) != (SCENARIO, seed)
                    or (expected == PROTOCOL and result['status'] != 'PASS')):
                raise ValueError('The pool collection cell is incomplete or belongs to another protocol')
            data = load_mechanism_dataset(cell/'bank.npz')
            audit = audit_dataset(data, seed, requested, protocol['sumocfgs'][SCENARIO])
            parts.append(data); cells.append({'result': str(cell/'result.json'), 'original_status': result['status'],
                'reused': expected == PREVIOUS_PROTOCOL, 'elapsed_sec': result['elapsed_sec'], 'rows': data.size, **audit})
    pool = merge_counterfactual_datasets_v3(parts)  # Also rejects overlapping old/new group IDs.
    try:
        selected, selection = select_local_b100(pool.metadata['action_group_ids'])
    except ValueError as error:
        result = {'protocol': PROTOCOL, 'status': 'FAIL', 'reason': str(error),
                  'pool_groups': len(set(pool.metadata['action_group_ids'])), 'cells': cells,
                  'elapsed_sec': time.monotonic()-started}
        shared.write_result(output/'result.json', result)
        return result
    context = read_network_right_of_way_context(net_file_from_sumocfg(Path(protocol['sumocfgs'][SCENARIO])))
    training = augment_dataset_with_right_of_way_context(pool, {SCENARIO: context})
    model = _target_model(training, group_ids=selected, candidate='constant_alpha_0', target_domain='cologne')
    payload = {'protocol': MODEL_PROTOCOL, 'occupancy_equation_protocol': OCCUPANCY_EQUATION_PROTOCOL,
        'city': 'cologne', 'target_budget': 100, 'selected_groups': selected,
        'fit_protocol': PROTOCOL, 'training_scenarios': [SCENARIO],
        'frozen_candidate': 'constant_alpha_0', 'rigid_model': model}
    output.mkdir(parents=True, exist_ok=True)
    with (output/'model.pkl').open('xb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    result = {'protocol': PROTOCOL, 'status': 'PASS', 'model_path': str(output/'model.pkl'),
        'model_protocol': MODEL_PROTOCOL, 'occupancy_equation_protocol': OCCUPANCY_EQUATION_PROTOCOL,
        'training_group_count': 100, 'selected_groups': selected, 'selection': selection,
        'training_row_count': sum(g in set(selected) for g in pool.metadata['action_group_ids']),
        'pool_groups': len(set(pool.metadata['action_group_ids'])), 'pool_rows': pool.size, 'cells': cells,
        'collection_elapsed_seconds_sum': sum(c['elapsed_sec'] for c in cells),
        'new_collection_elapsed_seconds_sum': sum(c['elapsed_sec'] for c in cells if not c['reused']),
        'anchor_feature_names': list(model.anchor_model.feature_names),
        'anchor_type': type(model.anchor_model).__name__, 'correction_type': type(model.correction_model).__name__,
        'frozen_candidate': 'constant_alpha_0', 'elapsed_sec': time.monotonic()-started}
    shared.write_result(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('collect', 'fit', 'evaluate'))
    for name in ('protocol', 'results-root', 'output', 'model'):
        parser.add_argument('--'+name, type=Path, required=name in ('protocol', 'output'))
    parser.add_argument('--scenario', choices=(SCENARIO,), default=SCENARIO)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--arm', choices=('rigid_target_only',), default='rigid_target_only')
    args = parser.parse_args(); protocol = load_protocol(args.protocol)
    if args.stage == 'collect': result = collect(protocol, args.seed, args.output)
    elif args.stage == 'fit': result = fit(protocol, args.results_root, args.output)
    else: result = shared.evaluate(protocol, args.model, SCENARIO, args.seed, args.arm, args.output)
    if result['status'] != 'PASS': raise SystemExit(2)


if __name__ == '__main__': main()
