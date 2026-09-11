"""B fixtures verify M deployment contracts and arithmetic, not M performance."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_fraction_target_rigid import setup_case
from test_thresholded_fraction_rigid import contract
from cf_h2o.traffic_signal.expanded_data_no_stay_rigid import (
    ExpandedDataNoStayOriginator, MODEL_PROTOCOL, FIT_PROTOCOL, ORIGINATOR_PROTOCOL, SCORE_UNITS)
from cf_h2o.traffic_signal.no_stay_fraction_rigid import NoStayFractionRigidOriginator
from cf_h2o.traffic_signal.safe_phase_controller import PhaseTiming, SafePhaseExecutor
from scripts.data import run_cologne_threshold_closed_loop as runner
from scripts.data import summarize_cologne_expanded_data_closed_loop as summary

RESULTS = Path(__file__).parents[1] / 'cf_h2o/results'


def payload_case(tmp_path, scores=(0., -2., -1.)):
    original, context, contrast, preparation = setup_case(tmp_path, scores)
    original['rigid_model'].correction_model = original['rigid_model'].anchor_model
    expanded = {**original, 'protocol': MODEL_PROTOCOL, 'fit_protocol': FIT_PROTOCOL, 'score_units': SCORE_UNITS,
                'target_budget': 195, 'selected_groups': [f'group{i}' for i in range(195)]}
    frozen = {**contract(threshold=0.), 'calibration_protocol': 'tsc-v155k-cologne-expanded-data-oof-v1', 'score_units': SCORE_UNITS}
    return original, expanded, frozen, context, contrast, preparation


@pytest.mark.parametrize('scores,current', [((0., -2., -1.), 1), ((0., -2., -1.), 0), ((0., 0., 2.), 1)])
def test_expanded_payload_preserves_top1_threshold_and_s_veto(tmp_path, scores, current):
    original, payload, frozen, context, contrast, preparation = payload_case(tmp_path, scores)
    states = contrast.metadata['candidate_states']
    executor = SafePhaseExecutor(sumo_api=SimpleNamespace(trafficlight=SimpleNamespace(setRedYellowGreenState=lambda *a: None)),
        tls_id='A', timings=[PhaseTiming(state, 10., 3., 2.) for state in states], initial_state=states[current], initial_green_elapsed_sec=20.)
    preparation['executors'] = {'A': executor}
    old = NoStayFractionRigidOriginator(runtime_payload=original, network_context=context, frozen_threshold=frozen)
    new = ExpandedDataNoStayOriginator(runtime_payload=payload, network_context=context, frozen_threshold=frozen)
    old.prepare_interval(**preparation)
    new.prepare_interval(**preparation)
    assert new.select(contrast, reference_index=0).to_dict() == old.select(contrast, reference_index=0).to_dict()
    if scores == (0., -2., -1.) and current == 1:
        assert new.diagnostics()['model_stay_veto_count'] == 1  # No I-style re-ranking to action2.
    assert new.diagnostics()['protocol'] == ORIGINATOR_PROTOCOL
    assert new.diagnostics()['target_budget'] == new.diagnostics()['training_group_count'] == 195


@pytest.mark.parametrize('change', ['old_payload', 'budget', 'duplicate_group', 'threshold'])
def test_expanded_loader_rejects_incompatible_model_or_threshold(tmp_path, change):
    original, payload, frozen, context, _, _ = payload_case(tmp_path)
    if change == 'old_payload':
        payload['protocol'] = original['protocol']
    elif change == 'budget':
        payload['target_budget'] = 100
    elif change == 'duplicate_group':
        payload['selected_groups'][-1] = payload['selected_groups'][0]
    else:
        frozen['threshold'] = .1
    with pytest.raises(ValueError):
        ExpandedDataNoStayOriginator(runtime_payload=payload, network_context=context, frozen_threshold=frozen)


def inputs():
    p = json.loads((RESULTS / 'cluster/tsc_v155b_cologne_raw_target_closed_loop_20260910/protocol_v1.json').read_text())
    k = json.loads((RESULTS / 'cluster/tsc_v155k_cologne_expanded_data_oof_20260910/protocol_v1.json').read_text())
    baseline = json.loads((RESULTS / 'paper_artifacts/tsc_v155b_cologne_raw_target_closed_loop_v1.json').read_text())
    p.update(protocol=summary.PROTOCOL, training_group_ids=k['original_group_ids'] + k['additional_group_ids'],
             original_feature_names=k['original_feature_names'], expanded_originator_path='fixture/expanded.py',
             full_fit_result='fixture/full/result.json', calibration_result='fixture/oof/result.json')
    p['controls'] = {'old_model_path': baseline['model_path'], 'old_frozen_threshold': deepcopy(baseline['frozen_threshold']),
                     'original_a_tasks': baseline['new_task_ids'], 'phase_pressure_tasks': baseline['reused_task_ids'][summary.PP]}
    p['frozen_threshold'].update(calibration_protocol='tsc-v155k-cologne-expanded-data-oof-v1',
                                 model_path='fixture/full/model.pkl', fit_result=p['full_fit_result'], calibration_result=p['calibration_result'])
    p['frozen_threshold'].pop('ablation_originator_path', None)
    rows = []
    for seed in summary.SEEDS:
        path = RESULTS / f'cluster/tsc_v155b_cologne_raw_target_closed_loop_20260910/evaluation/seed_{seed}/rigid_thresholded/result.json'
        row = json.loads(path.read_text())
        row.update(protocol=summary.PROTOCOL, model_path=p['frozen_threshold']['model_path'], frozen_threshold=deepcopy(p['frozen_threshold']),
                   source_result=str(path), task_id=f'fixture-{seed}', expanded_originator_path=p['expanded_originator_path'],
                   full_fit_result=p['full_fit_result'], calibration_result=p['calibration_result'])
        row['originator_diagnostics'].update(protocol=ORIGINATOR_PROTOCOL, target_budget=195, training_group_count=195,
                                             model_protocol=MODEL_PROTOCOL, threshold_model_path=p['frozen_threshold']['model_path'])
        rows.append(row)
    return p, rows, baseline


def full_fit_case():
    p, _, _ = inputs()
    fit = json.loads((RESULTS / 'cluster/tsc_v155a_cologne_raw_target_refit_20260910/fit_v1/result.json').read_text())
    fit.update(protocol=FIT_PROTOCOL, selected_groups=p['training_group_ids'], model_path=p['frozen_threshold']['model_path'],
               frozen_threshold=p['frozen_threshold'], new_model_fits=1, full_model_fits=1, oof_fold_fits=0)
    fit['checks'] = dict.fromkeys(('completed_k_admitted', 'native_b195_roster_exact', 'feature_order_exact',
        'a_weight_and_hgb_settings_unchanged', 'raw_targets_not_clipped', 'single_full_fit', 'serialized_scores_exact',
        'reference_predictions_zero', 'anchor_correction_alias_preserved', 'threshold_zero_unchanged'), True)
    fit['refs']['k_result'] = p['calibration_result']
    fit['fit_diagnostics']['full'].update(actual_group_count=195, rows=1560, training_rows=1365)
    k = json.loads((RESULTS / 'cluster/tsc_v155k_cologne_expanded_data_oof_20260910/protocol_v1.json').read_text())
    calibration = {key: k[key] for key in ('protocol', 'source_root', 'sumocfg', 'original_group_ids', 'additional_group_ids')}
    calibration.update(status='COMPLETE', checks={'complete_native195_roster': True})
    return p, p['frozen_threshold'], fit, calibration


def test_completed_l_fit_and_k_fixed_threshold_are_bound():
    runner.validate_expanded_fit(*full_fit_case())
    assert runner.completion_status(runner.EXPANDED_PROTOCOL, True, False) == 'COMPLETE'
    assert runner.completion_status(runner.EXPANDED_PROTOCOL, False, False) == 'INVALID'


@pytest.mark.parametrize('change', ['rows', 'model', 'calibration', 'roster'])
def test_incompatible_full_fit_or_calibration_cannot_deploy(change):
    p, frozen, fit, calibration = full_fit_case()
    if change == 'rows':
        fit['fit_diagnostics']['full']['training_rows'] = 700
    elif change == 'model':
        fit['model_path'] = 'wrong.pkl'
    elif change == 'calibration':
        calibration['source_root'] = 'different-source'
    else:
        fit['selected_groups'] = fit['selected_groups'][:-1]
    with pytest.raises(ValueError):
        runner.validate_expanded_fit(p, frozen, fit, calibration)


def test_nine_cells_compare_queue_with_original_a_and_pp():
    p, rows, baseline = inputs()
    report = summary.aggregate(p, rows, baseline)
    assert len(report['cells']) == 9 and report['matrix_status'] == 'COMPLETE'
    assert report['all_seed_service']['expanded_vs_original_a']['mean_queue']['difference_mean'] == 0.
    assert report['primary_efficiency']['metric'] == 'mean_queue'
    assert report['reused_task_ids'][summary.ORIGINAL_A] == ['t92044', 't92045', 't92046']
    assert report['execution_diagnosis']['totals']['executed_stay_overrides'] == 0
    assert not report['decision']['expanded_improves_original_a']
    assert report['next_step']['runs'] == 18 and len(report['next_step']['seeds']) == 6


def test_collisions_remain_diagnostic_and_invalid_run_disables_means():
    p, rows, baseline = inputs()
    rows[0].update(zero_incident_status='FAIL', status='COMPLETE')
    rows[0]['metrics'].update(collision_events=2, collision_incidents=1)
    report = summary.aggregate(p, rows, baseline)
    assert report['all_seed_service'] is not None and report['safety_by_arm'][summary.NEW]['zero_incident_status'] == 'FAIL'
    rows[0].update(run_valid=False, status='INVALID', zero_incident_status='UNAVAILABLE')
    report = summary.aggregate(p, rows, baseline)
    assert report['matrix_status'] == 'INVALID' and report['all_seed_service'] is None


@pytest.mark.parametrize('change', ['loader', 'queue_denominator', 'original_a_task'])
def test_changed_loader_or_efficiency_control_binding_is_rejected(change):
    p, rows, baseline = inputs()
    if change == 'loader':
        rows[0]['expanded_originator_path'] = p['ablation_originator_path']
    elif change == 'queue_denominator':
        rows[0]['metrics']['fixed_horizon_sample_seconds'] -= 1
    else:
        next(row for row in baseline['cells'] if row['arm'] == summary.ORIGINAL_A)['task_id'] = 'different'
    with pytest.raises(ValueError):
        summary.aggregate(p, rows, baseline)
