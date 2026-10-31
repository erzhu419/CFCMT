"""Run one frozen B195, B100 or PhasePressure cell on the six fresh seeds."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from run_cologne_threshold_closed_loop import (
    SPACING_CONTRACT, load_tool, spacing_validity, validate_expanded_fit,
    validate_raw_target_fit,
)


PROTOCOL = 'tsc-v155n-cologne-expanded-data-fresh-seed-validation-v1'
ORIGINAL_SEEDS = (52282, 41242, 63792)
SEEDS = tuple(seed + offset for offset in (100000, 200000) for seed in ORIGINAL_SEEDS)
EXPANDED = 'expanded_data_spaced_no_stay_rigid'
ORIGINAL_A = 'raw_target_spaced_no_stay_rigid'
PP = 'phase_pressure'
ARMS = (EXPANDED, ORIGINAL_A, PP)
SCORE_UNITS = 'raw_native_halted_per_lane_cost_difference'


def validate_protocol(protocol, seed, arm):
    evaluation = protocol['evaluation']
    expected = {'scenario': 'cologne1', 'duration_sec': 3600, 'warmup_sec': 60,
                'control_interval_sec': 10, 'prediction_horizon_sec': 450,
                'residual_coordination_mode': 'direct', 'residual_cooldown_intervals': 44}
    if (protocol['protocol'] != PROTOCOL or tuple(evaluation['seeds']) != SEEDS
            or seed not in SEEDS or tuple(protocol['arms']) != ARMS or arm not in ARMS
            or any(evaluation[key] != value for key, value in expected.items())
            or protocol['spacing_contract'] != SPACING_CONTRACT):
        raise ValueError('Use the six fixed fresh seeds, three arms and unchanged3600/60/10 spacing contract')
    for frozen in (protocol['frozen_threshold'], protocol['original_a_frozen_threshold']):
        if (frozen['threshold'] != 0 or frozen['mode'] != 'minimum_advantage'
                or frozen['comparison'] != 'predicted_advantage > threshold + 1e-12'
                or frozen['score_units'] != SCORE_UNITS
                or frozen['source_root'] != protocol['source_root']):
            raise ValueError('Both frozen models must retain raw units and threshold zero')


def yellow_priority_check(executor_class, timing_class):
    """Exercise the actual evaluator's clearance write before starting SUMO."""
    writes = []
    api = SimpleNamespace(trafficlight=SimpleNamespace(
        setRedYellowGreenState=lambda tls, state: writes.append([tls, state])))
    executor = executor_class(sumo_api=api, tls_id='priority_check',
        timings=[timing_class('Ggrr', 10., 3., 2.), timing_class('rrGG', 10., 3., 2.)],
        initial_state='Ggrr', initial_green_elapsed_sec=20.)
    accepted = executor.request('rrGG')
    if not (accepted and executor.mode == 'yellow'
            and writes == [['priority_check', 'Yyrr']] and executor.current_state == 'Yyrr'):
        raise ValueError('The active J executor must retain G->Y and g->y yellow priority')
    return {'passed': True, 'initial_state': 'Ggrr', 'target_state': 'rrGG',
            'clearance_state': executor.current_state}


def load_originator(protocol, arm, runtime):
    if arm == PP:
        return None, None
    expanded = arm == EXPANDED
    frozen_path = Path(protocol['frozen_threshold_path'] if expanded
                       else protocol['original_a_frozen_threshold_path'])
    frozen = json.loads(frozen_path.read_text())
    expected = protocol['frozen_threshold'] if expanded else protocol['original_a_frozen_threshold']
    if frozen != expected:
        raise ValueError('The frozen model or threshold binding changed')
    fit = json.loads(Path(frozen['fit_result']).read_text())
    if expanded:
        validate_expanded_fit(protocol, frozen, fit,
                              json.loads(Path(frozen['calibration_result']).read_text()))
    else:
        validate_raw_target_fit(protocol, frozen, fit)
    sys.path.insert(0, str(Path(protocol['threshold_originator_path']).parent))
    if expanded:
        sys.path.insert(0, str(Path(protocol['ablation_originator_path']).parent))
        module = load_tool(protocol['expanded_originator_path'], 'v155n_expanded_originator')
        originator_class = module.ExpandedDataNoStayOriginator
    else:
        module = load_tool(protocol['ablation_originator_path'], 'v155n_raw_originator')
        originator_class = module.NoStayFractionRigidOriginator
    originator = originator_class.load(Path(frozen['model_path']),
        frozen_threshold_path=frozen_path, expected_city='cologne',
        network_context=runtime.read_network_right_of_way_context(
            runtime.net_file_from_sumocfg(Path(protocol['sumocfg']))))
    if (list(originator.payload['rigid_model'].anchor_model.feature_names) != protocol['original_feature_names']
            or (expanded and originator.payload['selected_groups'] != protocol['training_group_ids'])):
        raise ValueError('Preserve original29 feature order and the full B195 roster')
    return originator, frozen


def run(protocol, seed, arm, output):
    from cf_h2o.eval import traffic_signal_cologne_collision_replay as runtime
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import _runtime_models, RUNTIME_POLICY
    from cf_h2o.traffic_signal import safe_phase_controller as safe
    from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL

    started = time.monotonic()
    validate_protocol(protocol, seed, arm)
    source = Path(protocol['source_root'])
    if (os.environ['CFCMT_SOURCE_ROOT'] != str(source)
            or Path(v3.__file__).resolve() != (source / 'cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py').resolve()
            or Path(safe.__file__).resolve() != (source / 'cf_h2o/traffic_signal/safe_phase_controller.py').resolve()
            or runtime.evaluate_policy_v3 is not v3.evaluate_policy_v3
            or v3.SafePhaseExecutor is not safe.SafePhaseExecutor):
        raise ValueError('All three arms must use the unchanged J evaluator and executor')
    yellow = yellow_priority_check(v3.SafePhaseExecutor, safe.PhaseTiming)
    if output.exists():
        raise FileExistsError(output)
    reference = json.loads(Path(protocol['geometry_reference']).read_text())
    sumocfg = Path(protocol['sumocfg'])
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(sumocfg))
    if geometry != reference['bilateral_static_geometry']:
        raise ValueError('The repaired bilateral network changed')
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = protocol['conversion_root']
    originator, frozen = load_originator(protocol, arm, runtime)
    models = _runtime_models(originator, prediction_horizon_sec=450) if originator is not None else None
    policy = RUNTIME_POLICY if originator is not None else 'phase_pressure'
    api = runtime.load_libsumo()
    sumo_version = runtime.libsumo_version()
    if '1.22.0' not in str(sumo_version):
        raise ValueError('Use the frozen SUMO1.22.0 runtime')
    comparison = load_tool(source / 'scripts/data/run_cologne_repaired_controller_comparison.py', 'v155n_recording')
    routes = {v: reference['focal_demand']['vehicles'][v]['route'] for v in comparison.full.FOCAL_IDS}
    observer = comparison.full.FullObserver(api, lambda **_: None, routes, runtime, v3, geometry)
    observer.tracker = comparison.RouteAwarePassageTracker(routes, geometry)
    max_green_elapsed = {}

    def observe(*, stage, time_sec, executors):
        observer(stage=stage, time_sec=time_sec, executors=executors)
        if stage == 'after_executor_advance':
            for tls, executor in executors.items():
                if executor.mode == 'green':
                    max_green_elapsed[tls] = max(max_green_elapsed.get(tls, 0.), float(executor.green_elapsed_sec))

    scratch = Path(protocol['scratch_root']) / f'{arm}_{seed}.xml'
    if scratch.exists():
        raise FileExistsError(scratch)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics = runtime.evaluate_policy_v3(
            sumo_api=api, sumocfg=sumocfg, scenario='cologne1', policy=policy, models=models,
            duration_sec=3600, control_interval_sec=10, warmup_sec=60, seed=seed,
            tripinfo_output=scratch, residual_coordination_mode='direct',
            residual_cooldown_intervals_override=44 if originator is not None else 0,
            step_observer=observe)
    finally:
        scratch.unlink(missing_ok=True)
    actual = {'metrics': metrics, 'collision_inventory': observer.inventory(), 'step_times': observer.step_times}
    checks = comparison.comparison_validity(actual, 2015)
    checks['eight_controlled_lanes'] = metrics.get('controlled_lane_count') == 8
    if originator is not None:
        checks.update(spacing_validity(metrics))
    else:
        checks['phase_pressure_without_model_overrides'] = (
            models is None and metrics['accepted_intervention_trace'] == []
            and metrics['guard_audit'] == {} and metrics['residual_deployment'] == {})
    valid = all(checks.values())
    zero = metrics.get('collision_events') == 0 and metrics.get('collision_incidents') == 0
    result = {
        'protocol': PROTOCOL, 'status': 'COMPLETE' if valid else 'INVALID', 'run_valid': valid,
        'zero_incident_status': 'UNAVAILABLE' if not valid else 'PASS' if zero else 'FAIL',
        'validity_checks': checks, 'seed': seed, 'arm': arm, 'scenario': 'cologne1',
        'source_root': str(source), 'repaired_sumocfg': str(sumocfg),
        'occupancy_equation_protocol': OCCUPANCY_EQUATION_PROTOCOL,
        'sumo_version': sumo_version,
        'runtime_bindings': {'evaluator': str(Path(v3.__file__).resolve()),
                             'executor': str(Path(safe.__file__).resolve())},
        'deployed_policy': policy, 'model_path': frozen['model_path'] if frozen else None,
        'frozen_threshold': frozen, 'originator_diagnostics': originator.diagnostics() if originator else None,
        'yellow_priority_check': yellow,
        'maximum_green_elapsed_sec_by_tls': max_green_elapsed,
        **actual, 'elapsed_sec': time.monotonic() - started,
    }
    if originator is not None:
        result.update(threshold_originator_path=protocol['threshold_originator_path'],
                      ablation_originator_path=protocol['ablation_originator_path'],
                      spacing_contract=SPACING_CONTRACT, score_units=SCORE_UNITS)
    if arm == EXPANDED:
        result.update(expanded_originator_path=protocol['expanded_originator_path'],
                      full_fit_result=protocol['full_fit_result'], calibration_result=protocol['calibration_result'])
    output.mkdir(parents=True)
    with (output / 'result.json').open('x') as file:
        json.dump(result, file, separators=(',', ':'), allow_nan=False)
        file.write('\n')
    print(json.dumps({key: result[key] for key in ('protocol', 'seed', 'arm', 'status', 'elapsed_sec')}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.protocol.read_text()), args.seed, args.arm, args.output)
    if not result['run_valid']:
        raise SystemExit(2)
    print('Eval complete')


if __name__ == '__main__':
    main()
