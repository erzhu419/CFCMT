"""Evaluate the frozen OOF threshold on the original repaired Cologne1 network."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


PROTOCOL = 'tsc-v154m-cologne-threshold-closed-loop-v1'
NATIVE_PROTOCOL = 'tsc-v154q-cologne-native-label-closed-loop-v1'
NO_STAY_PROTOCOL = 'tsc-v154s-cologne-no-stay-closed-loop-v1'
RECALIBRATED_PROTOCOL = 'tsc-v154u-cologne-recalibrated-no-stay-closed-loop-v1'
SPACED_PROTOCOL = 'tsc-v154x-cologne-spaced-no-stay-closed-loop-v1'
RAW_TARGET_PROTOCOL = 'tsc-v155b-cologne-raw-target-closed-loop-v1'
EXPANDED_PROTOCOL = 'tsc-v155m-cologne-expanded-data-closed-loop-v1'
SPACING_CONTRACT = {'control_interval_sec': 10, 'cooldown_intervals': 44,
                    'minimum_intervention_gap_sec': 450,
                    'model_action_sec': 10, 'pp_continuation_sec': 440}
SEEDS = (52282, 41242, 63792)
ARM = 'rigid_thresholded'


def load_tool(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spacing_validity(metrics):
    deployment = metrics['residual_deployment']
    audit = metrics['guard_audit']
    trace = metrics['accepted_intervention_trace']
    times = [row['time_sec'] for row in trace]
    return {
        'spacing_runtime_configuration': deployment['cooldown_intervals'] == 44
            and deployment['cooldown_intervals_override'] == 44,
        'spacing_complete_executed_trace': len(trace) == audit['accepted_overrides']
            == audit['executed_switch_overrides'] and audit['executed_stay_overrides'] == 0
            and audit['rejected_executor'] == 0,
        'spacing_effective_switches': all(row['override_kind'] == 'switch'
            and row['request_accepted'] and row['execution_effective'] for row in trace),
        'spacing_at_least_450_seconds': all(b-a >= 450 for a, b in zip(times, times[1:])),
    }


def validate_raw_target_fit(protocol, frozen, fit):
    full = fit['fit_diagnostics']['full']
    units = 'raw_native_halted_per_lane_cost_difference'
    if (fit['protocol'] != 'tsc-v155a-cologne-raw-target-refit-v1'
            or fit['status'] != 'COMPLETE' or not all(fit['checks'].values())
            or fit['frozen_threshold'] != frozen or frozen['calibration_protocol'] != fit['protocol']
            or frozen['fit_result'] != frozen['calibration_result']
            or frozen['threshold'] != 0 or frozen['mode'] != 'minimum_advantage'
            or frozen['score_units'] != units or fit['score_units'] != units or full['score_units'] != units
            or full['target_normalization_protocol'] != 'raw_action_delta_v1'
            or full['actual_group_count'] != 100 or full['rows'] != 800 or full['training_rows'] != 700
            or len(fit['selected_groups']) != 100 or len(set(fit['selected_groups'])) != 100
            or fit['model_path'] != frozen['model_path']
            or fit['source_root'] != protocol['source_root'] or frozen['source_root'] != protocol['source_root']
            or fit['sumocfg'] != protocol['sumocfg']
            or frozen['ablation_originator_path'] != protocol['ablation_originator_path']):
        raise ValueError('Use the completed raw-target B100 fit, its threshold zero and unchanged S stay veto')


def completion_status(protocol_name, valid, zero_incidents):
    if not valid:
        return 'INVALID'
    if protocol_name in (RAW_TARGET_PROTOCOL, EXPANDED_PROTOCOL):
        return 'COMPLETE'
    return 'PASS' if zero_incidents else 'FAIL'


def validate_expanded_fit(protocol, frozen, fit, calibration):
    checks = ('completed_k_admitted', 'native_b195_roster_exact', 'feature_order_exact',
              'a_weight_and_hgb_settings_unchanged', 'raw_targets_not_clipped', 'single_full_fit',
              'serialized_scores_exact', 'reference_predictions_zero', 'anchor_correction_alias_preserved',
              'threshold_zero_unchanged')
    full, groups = fit['fit_diagnostics']['full'], protocol['training_group_ids']
    units = 'raw_native_halted_per_lane_cost_difference'
    if (fit['protocol'] != 'tsc-v155l-cologne-expanded-data-full-fit-v1' or fit['status'] != 'COMPLETE'
            or any(fit.get('checks', {}).get(key) is not True for key in checks)
            or fit['frozen_threshold'] != frozen or frozen['fit_result'] != protocol['full_fit_result']
            or frozen['calibration_result'] != protocol['calibration_result']
            or fit['refs']['k_result'] != protocol['calibration_result']
            or calibration['protocol'] != 'tsc-v155k-cologne-expanded-data-oof-v1'
            or calibration['status'] != 'COMPLETE' or not calibration['checks'] or not all(calibration['checks'].values())
            or frozen['calibration_protocol'] != calibration['protocol']
            or groups != calibration['original_group_ids'] + calibration['additional_group_ids']
            or fit['selected_groups'] != groups or len(groups) != 195 or len(set(groups)) != 195
            or fit['anchor_feature_names'] != protocol['original_feature_names'] or len(fit['anchor_feature_names']) != 29
            or full['feature_names'] != protocol['original_feature_names']
            or any(fit[key] != expected for key, expected in (('new_model_fits', 1), ('full_model_fits', 1), ('oof_fold_fits', 0)))
            or full['actual_group_count'] != 195 or full['rows'] != 1560 or full['training_rows'] != 1365
            or full['target_normalization_protocol'] != 'raw_action_delta_v1' or full['raw_target_clipped_count'] != 0
            or full['sample_weight_source_protocol'] != 'group_action_range_v1'
            or any(item['score_units'] != units for item in (fit, full, frozen))
            or frozen['threshold'] != 0 or frozen['mode'] != 'minimum_advantage'
            or fit['model_path'] != frozen['model_path']
            or any(item['source_root'] != protocol['source_root'] for item in (fit, frozen, calibration))
            or any(item['sumocfg'] != protocol['sumocfg'] for item in (fit, calibration))):
        raise ValueError('Require the completed L full B195 fit, its admitted K calibration and original29 raw-target contract')


def run(protocol, seed, output):
    from cf_h2o.eval import traffic_signal_cologne_collision_replay as runtime
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import _runtime_models, RUNTIME_POLICY
    from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL

    started = time.monotonic()
    if protocol['protocol'] not in (PROTOCOL, NATIVE_PROTOCOL, NO_STAY_PROTOCOL, RECALIBRATED_PROTOCOL, SPACED_PROTOCOL, RAW_TARGET_PROTOCOL, EXPANDED_PROTOCOL) or seed not in SEEDS:
        raise ValueError('Use a frozen Cologne threshold comparison and its three evaluation seeds')
    if os.environ['CFCMT_SOURCE_ROOT'] != protocol['source_root']:
        raise ValueError('Use the unchanged V154J fraction-equation core source')
    if output.exists():
        raise FileExistsError(output)
    frozen_path = Path(protocol['frozen_threshold_path'])
    frozen = json.loads(frozen_path.read_text())
    if frozen != protocol['frozen_threshold']:
        raise ValueError('The previously calibrated threshold or model binding changed')
    expanded = protocol['protocol'] == EXPANDED_PROTOCOL
    spaced = protocol['protocol'] in (SPACED_PROTOCOL, RAW_TARGET_PROTOCOL, EXPANDED_PROTOCOL)
    if spaced and (protocol['spacing_contract'] != SPACING_CONTRACT
                   or protocol['evaluation']['residual_cooldown_intervals'] != 44):
        raise ValueError('Use the fixed 10-second model action and 440-second PP continuation')
    if expanded:
        if any(protocol['evaluation'][key] != value for key, value in (
                ('duration_sec', 3600), ('warmup_sec', 60), ('control_interval_sec', 10))):
            raise ValueError('Use the frozen3600-second evaluation,60-second warmup and10-second control interval')
        validate_expanded_fit(protocol, frozen, json.loads(Path(frozen['fit_result']).read_text()),
                              json.loads(Path(frozen['calibration_result']).read_text()))
    elif protocol['protocol'] == RAW_TARGET_PROTOCOL:
        validate_raw_target_fit(protocol, frozen, json.loads(Path(frozen['fit_result']).read_text()))
    elif protocol['protocol'] in (NATIVE_PROTOCOL, NO_STAY_PROTOCOL, RECALIBRATED_PROTOCOL, SPACED_PROTOCOL):
        fit = json.loads(Path(frozen['fit_result']).read_text())
        if (fit['status'] != 'PASS' or fit['training_group_count'] != 100
                or fit['training_row_count'] != 800
                or fit['model_path'] != frozen['model_path']
                or fit['source_root'] != protocol['source_root'] or fit['sumocfg'] != protocol['sumocfg']):
            raise ValueError('Use the completed native B100 model on the unchanged network')
        if protocol['protocol'] in (RECALIBRATED_PROTOCOL, SPACED_PROTOCOL):
            calibration = json.loads(Path(frozen['calibration_result']).read_text())
            if (frozen.get('calibration_protocol') != 'tsc-v154t-cologne-no-stay-oof-threshold-v1'
                    or calibration['protocol'] != frozen['calibration_protocol']
                    or calibration['status'] != 'COMPLETE' or calibration['frozen_threshold'] != frozen
                    or calibration['model_path'] != fit['model_path']
                    or frozen['ablation_originator_path'] != protocol['ablation_originator_path']):
                raise ValueError('Use the completed training-only T calibration and unchanged S stay veto')
        elif (frozen.get('calibration_protocol') != 'tsc-v154p-cologne-native-b100-v1'
                or frozen['threshold'] != 0 or frozen['mode'] != 'minimum_advantage'
                or fit['frozen_threshold'] != frozen):
            raise ValueError('Use the native B100 model and its frozen threshold zero')
    comparison = load_tool(Path(protocol['source_root']) / 'scripts/data/run_cologne_repaired_controller_comparison.py',
                           'v154m_recording')
    no_stay = protocol['protocol'] in (NO_STAY_PROTOCOL, RECALIBRATED_PROTOCOL, SPACED_PROTOCOL, RAW_TARGET_PROTOCOL, EXPANDED_PROTOCOL)
    if no_stay:
        sys.path.insert(0, str(Path(protocol['threshold_originator_path']).parent))
        if expanded:
            sys.path.insert(0, str(Path(protocol['ablation_originator_path']).parent))
            ablation_tool = load_tool(Path(protocol['expanded_originator_path']), 'v155m_expanded_originator')
            originator_class = ablation_tool.ExpandedDataNoStayOriginator
        else:
            ablation_tool = load_tool(Path(protocol['ablation_originator_path']), 'v154s_no_stay_originator')
            originator_class = ablation_tool.NoStayFractionRigidOriginator
    else:
        threshold_tool = load_tool(Path(protocol['threshold_originator_path']), 'v154m_threshold_originator')
        originator_class = threshold_tool.ThresholdedFractionRigidOriginator
    reference = json.loads(Path(protocol['geometry_reference']).read_text())
    sumocfg = Path(protocol['sumocfg'])
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(sumocfg))
    if geometry != reference['bilateral_static_geometry']:
        raise ValueError('The repaired bilateral geometry changed')
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = protocol['conversion_root']
    originator = originator_class.load(
        Path(frozen['model_path']), frozen_threshold_path=frozen_path, expected_city='cologne',
        network_context=runtime.read_network_right_of_way_context(runtime.net_file_from_sumocfg(sumocfg)))
    if expanded and (originator.payload['selected_groups'] != protocol['training_group_ids']
            or list(originator.payload['rigid_model'].anchor_model.feature_names) != protocol['original_feature_names']):
        raise ValueError('The loaded B195 payload must preserve the full-fit roster and original29 features')
    models = _runtime_models(originator, prediction_horizon_sec=450)
    api = runtime.load_libsumo()
    if '1.22.0' not in str(runtime.libsumo_version()):
        raise ValueError('Use the frozen SUMO 1.22.0 runtime')
    routes = {v: reference['focal_demand']['vehicles'][v]['route'] for v in comparison.full.FOCAL_IDS}
    observer = comparison.full.FullObserver(api, lambda **_: None, routes, runtime, v3, geometry)
    observer.tracker = comparison.RouteAwarePassageTracker(routes, geometry)
    max_green_elapsed = {}

    def observe_no_stay(*, stage, time_sec, executors):
        observer(stage=stage, time_sec=time_sec, executors=executors)
        if stage == 'after_executor_advance':
            for tls, executor in executors.items():
                if executor.mode == 'green':
                    max_green_elapsed[tls] = max(max_green_elapsed.get(tls, 0.0), float(executor.green_elapsed_sec))

    scratch = Path(protocol['scratch_root']) / f'{ARM}_{seed}.xml'
    if scratch.exists():
        raise FileExistsError(scratch)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics = runtime.evaluate_policy_v3(
            sumo_api=api, sumocfg=sumocfg, scenario='cologne1', policy=RUNTIME_POLICY, models=models,
            duration_sec=3600, control_interval_sec=10, warmup_sec=60, seed=seed,
            tripinfo_output=scratch, residual_coordination_mode='direct',
            residual_cooldown_intervals_override=44 if spaced else 0,
            step_observer=observe_no_stay if no_stay else observer)
    finally:
        scratch.unlink(missing_ok=True)
    actual = {'metrics': metrics, 'collision_inventory': observer.inventory(), 'step_times': observer.step_times}
    checks = comparison.comparison_validity(actual, 2015)
    if spaced:
        checks.update(spacing_validity(metrics))
    valid = all(checks.values())
    zero = metrics.get('collision_events') == 0 and metrics.get('collision_incidents') == 0
    result = {
        'protocol': protocol['protocol'], 'status': completion_status(protocol['protocol'], valid, zero),
        'run_valid': valid, 'zero_incident_status': 'UNAVAILABLE' if not valid else 'PASS' if zero else 'FAIL',
        'validity_checks': checks, 'seed': seed, 'arm': ARM, 'scenario': 'cologne1',
        'repaired_sumocfg': str(sumocfg), 'occupancy_equation_protocol': OCCUPANCY_EQUATION_PROTOCOL,
        'source_root': os.environ['CFCMT_SOURCE_ROOT'], 'model_path': frozen['model_path'],
        'frozen_threshold': frozen, 'threshold_originator_path': protocol['threshold_originator_path'],
        'originator_diagnostics': originator.diagnostics(), **actual,
        'elapsed_sec': time.monotonic() - started,
    }
    if no_stay:
        result['ablation_originator_path'] = protocol['ablation_originator_path']
        result['maximum_green_elapsed_sec_by_tls'] = max_green_elapsed
    if spaced:
        result['spacing_contract'] = SPACING_CONTRACT
    if protocol['protocol'] in (RAW_TARGET_PROTOCOL, EXPANDED_PROTOCOL):
        result['score_units'] = frozen['score_units']
    if expanded:
        result['expanded_originator_path'] = protocol['expanded_originator_path']
        result['full_fit_result'] = protocol['full_fit_result']
        result['calibration_result'] = protocol['calibration_result']
    output.mkdir(parents=True)
    with (output / 'result.json').open('x') as file:
        json.dump(result, file, separators=(',', ':'), allow_nan=False)
        file.write('\n')
    print(json.dumps({key: result[key] for key in ('protocol', 'status', 'elapsed_sec')}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.protocol.read_text()), args.seed, args.output)
    completed = result['run_valid'] if result['protocol'] in (RAW_TARGET_PROTOCOL, EXPANDED_PROTOCOL) else result['status'] == 'PASS'
    if not completed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
