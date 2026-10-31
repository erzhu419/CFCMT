"""Recollect the original B100 groups, refit rigid, and evaluate fixed Cologne."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

PROTOCOL = 'tsc-v154j-cologne-fraction-refit-v1'
SCENARIOS = ('cologne1', 'cologne3', 'cologne8')
TRAIN_SEEDS = (2027, 3037, 4047)
EVAL_SEEDS = (52282, 41242, 63792)


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(result, f, separators=(',', ':')); f.write('\n')
    print(json.dumps({k: result[k] for k in ('protocol', 'status', 'elapsed_sec')}))


def load_protocol(path):
    protocol = json.loads(path.read_text())
    if protocol['protocol'] != PROTOCOL or len(set(protocol['selected_groups'])) != 100:
        raise ValueError('Use the frozen original B100 collection roster')
    return protocol


def collect(protocol, scenario, seed, output):
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import collect_counterfactual_transitions_v3
    from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
    from cf_h2o.traffic_signal.dataset_cache import save_mechanism_dataset
    from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL
    started = time.monotonic()
    if scenario not in SCENARIOS or seed not in TRAIN_SEEDS:
        raise ValueError('Training must use the original three scenarios and three seeds')
    requested = [g for g in protocol['selected_groups'] if g.startswith(f'{scenario}:seed{seed}:')]
    if not requested or (output/'bank.npz').exists() or (output/'result.json').exists():
        raise ValueError('Use a fresh nonempty collection cell')
    settings = protocol['collection']
    if '1.22.0' not in str(libsumo_version()):
        raise ValueError('Use the frozen SUMO 1.22.0 runtime')
    dataset = collect_counterfactual_transitions_v3(sumo_api=load_libsumo(),
        sumocfg=Path(protocol['sumocfgs'][scenario]), scenario=scenario, seed=seed,
        selected_action_group_ids=requested, collection_shard_index=0, collection_shard_count=1,
        **settings)
    retained = set(dataset.metadata['action_group_ids'])
    if dataset.metadata.get('occupancy_equation_protocol') != OCCUPANCY_EQUATION_PROTOCOL:
        raise ValueError('Fresh labels have the wrong occupancy contract')
    dataset = replace(dataset, metadata={**dataset.metadata, 'fraction_refit_protocol': PROTOCOL})
    output.mkdir(parents=True, exist_ok=True)
    save_mechanism_dataset(output/'bank.npz', dataset)
    occupied = dataset.features[:, dataset.feature_names.index('green_down_occ')]
    result = {'protocol': PROTOCOL, 'status': 'PASS' if retained == set(requested) else 'FAIL',
        'scenario': scenario, 'seed': seed, 'settings': settings,
        'sumocfg': protocol['sumocfgs'][scenario], 'bank': str(output/'bank.npz'),
        'requested_groups': requested, 'retained_groups': sorted(retained),
        'missing_groups': sorted(set(requested)-retained), 'unexpected_groups': sorted(retained-set(requested)),
        'rows': dataset.size, 'occupancy_equation_protocol': OCCUPANCY_EQUATION_PROTOCOL,
        'maximum_downstream_occupancy_fraction': float(np.max(occupied)) if len(occupied) else None,
        'behavior_safety_audit': dataset.metadata.get('behavior_safety_audit'),
        'counterfactual_safety_audit': dataset.metadata.get('counterfactual_safety_audit'),
        'elapsed_sec': time.monotonic()-started}
    write_result(output/'result.json', result)
    return result


def fit(protocol, results_root, output):
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import merge_counterfactual_datasets_v3
    from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _merge_city_datasets
    from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import _target_model
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    from cf_h2o.traffic_signal.fraction_target_rigid import MODEL_PROTOCOL
    from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL
    from cf_h2o.traffic_signal.right_of_way_context import (augment_dataset_with_right_of_way_context,
        net_file_from_sumocfg, read_network_right_of_way_context)
    started = time.monotonic(); bank = {}; cells = []
    for scenario in SCENARIOS:
        parts = []
        for seed in TRAIN_SEEDS:
            cell = results_root/'collection'/scenario/f'seed_{seed}'
            result = json.loads((cell/'result.json').read_text())
            if result['protocol'] != PROTOCOL or result['status'] != 'PASS':
                raise ValueError(f'Original B100 groups incomplete: {cell}')
            data = load_mechanism_dataset(cell/'bank.npz')
            if data.metadata.get('occupancy_equation_protocol') != OCCUPANCY_EQUATION_PROTOCOL:
                raise ValueError('Refitting requires newly collected fraction labels')
            parts.append(data); cells.append(result)
        data = merge_counterfactual_datasets_v3(parts)
        context = read_network_right_of_way_context(net_file_from_sumocfg(Path(protocol['sumocfgs'][scenario])))
        bank[scenario] = augment_dataset_with_right_of_way_context(data, {scenario: context})
    training = _merge_city_datasets(bank, SCENARIOS, city='cologne')
    if set(training.metadata['action_group_ids']) != set(protocol['selected_groups']):
        raise ValueError('The original B100 group roster changed')
    model = _target_model(training, group_ids=protocol['selected_groups'],
                          candidate=protocol['frozen_candidate'], target_domain='cologne')
    payload = {'protocol': MODEL_PROTOCOL, 'occupancy_equation_protocol': OCCUPANCY_EQUATION_PROTOCOL,
        'city': 'cologne', 'target_budget': 100, 'selected_groups': protocol['selected_groups'],
        'frozen_candidate': protocol['frozen_candidate'], 'rigid_model': model}
    output.mkdir(parents=True, exist_ok=True)
    with (output/'model.pkl').open('xb') as f: pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    result = {'protocol': PROTOCOL, 'status': 'PASS', 'model_path': str(output/'model.pkl'),
        'model_protocol': MODEL_PROTOCOL, 'occupancy_equation_protocol': OCCUPANCY_EQUATION_PROTOCOL,
        'training_group_count': 100, 'training_row_count': training.size,
        'frozen_candidate': protocol['frozen_candidate'],
        'anchor_feature_names': list(model.anchor_model.feature_names),
        'anchor_type': type(model.anchor_model).__name__, 'correction_type': type(model.correction_model).__name__,
        'training_cell_count': len(cells), 'collection_elapsed_seconds_sum': sum(c['elapsed_sec'] for c in cells),
        'elapsed_sec': time.monotonic()-started}
    write_result(output/'result.json', result)
    return result


def evaluate(protocol, model_path, scenario, seed, arm, output):
    from cf_h2o.eval import traffic_signal_cologne_collision_replay as runtime
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import _runtime_models, RUNTIME_POLICY
    from cf_h2o.traffic_signal.fraction_target_rigid import FractionTargetRigidOriginator
    from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL
    spec = importlib.util.spec_from_file_location('v154f_recording', Path(__file__).with_name('run_cologne_repaired_controller_comparison.py'))
    comparison = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparison)
    started = time.monotonic()
    if scenario != 'cologne1' or seed not in EVAL_SEEDS or arm not in comparison.ARMS:
        raise ValueError('Use the frozen Cologne1 evaluation roster')
    reference = json.loads(Path(protocol['geometry_reference']).read_text())
    sumocfg = Path(protocol['sumocfgs'][scenario])
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(sumocfg))
    if geometry != reference['bilateral_static_geometry']:
        raise ValueError('The fixed bilateral geometry changed')
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = protocol['conversion_root']
    models = None; originator = None; policy = 'phase_pressure'
    if arm == 'rigid_target_only':
        originator = FractionTargetRigidOriginator.load(model_path, expected_city='cologne',
            network_context=runtime.read_network_right_of_way_context(runtime.net_file_from_sumocfg(sumocfg)))
        models = _runtime_models(originator, prediction_horizon_sec=450); policy = RUNTIME_POLICY
    api = runtime.load_libsumo()
    if '1.22.0' not in str(runtime.libsumo_version()): raise ValueError('Use SUMO 1.22.0')
    routes = {v: reference['focal_demand']['vehicles'][v]['route'] for v in comparison.full.FOCAL_IDS}
    observer = comparison.full.FullObserver(api, lambda **_: None, routes, runtime, v3, geometry)
    observer.tracker = comparison.RouteAwarePassageTracker(routes, geometry)
    scratch = Path(protocol['scratch_root'])/f'{arm}_{seed}.xml'
    if scratch.exists(): raise FileExistsError(scratch)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics = runtime.evaluate_policy_v3(sumo_api=api, sumocfg=sumocfg, scenario=scenario,
            policy=policy, models=models, duration_sec=3600, control_interval_sec=10,
            warmup_sec=60, seed=seed, tripinfo_output=scratch, residual_coordination_mode='direct',
            residual_cooldown_intervals_override=0, step_observer=observer)
    finally: scratch.unlink(missing_ok=True)
    actual = {'metrics': metrics, 'collision_inventory': observer.inventory(), 'step_times': observer.step_times}
    checks = comparison.comparison_validity(actual, 2015); valid = all(checks.values())
    zero = metrics.get('collision_events') == 0 and metrics.get('collision_incidents') == 0
    result = {'protocol': PROTOCOL, 'status': 'INVALID' if not valid else 'PASS' if zero else 'FAIL',
        'run_valid': valid, 'zero_incident_status': 'UNAVAILABLE' if not valid else 'PASS' if zero else 'FAIL',
        'validity_checks': checks, 'seed': seed, 'arm': arm, 'scenario': scenario,
        'repaired_sumocfg': str(sumocfg), 'occupancy_equation_protocol': OCCUPANCY_EQUATION_PROTOCOL,
        'source_root': os.environ['CFCMT_SOURCE_ROOT'], 'model_path': str(model_path) if models else None,
        'originator_diagnostics': originator.diagnostics() if originator else None,
        **actual, 'elapsed_sec': time.monotonic()-started}
    write_result(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('collect', 'fit', 'evaluate'))
    for name in ('protocol', 'results-root', 'output', 'model'):
        parser.add_argument('--'+name, type=Path, required=name in ('protocol','output'))
    parser.add_argument('--scenario', choices=SCENARIOS); parser.add_argument('--seed', type=int)
    parser.add_argument('--arm', choices=('rigid_target_only', 'phase_pressure'))
    a = parser.parse_args(); protocol = load_protocol(a.protocol)
    if a.stage == 'collect': result = collect(protocol, a.scenario, a.seed, a.output)
    elif a.stage == 'fit': result = fit(protocol, a.results_root, a.output)
    else: result = evaluate(protocol, a.model, a.scenario, a.seed, a.arm, a.output)
    if result['status'] != 'PASS': raise SystemExit(2)


if __name__ == '__main__': main()
