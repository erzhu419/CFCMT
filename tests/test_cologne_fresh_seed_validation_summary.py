"""Existing small results supply schema fixtures, not fresh-seed outcomes."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.data import summarize_cologne_fresh_seed_validation as summary


ROOT = Path(__file__).parents[1]
RESULTS = ROOT / 'cf_h2o/results'


@pytest.fixture
def inputs():
    protocol = json.loads((RESULTS / 'cluster/tsc_v155n_cologne_expanded_data_fresh_seed_validation_20260910/protocol_v1.json').read_text())
    baseline = json.loads((RESULTS / 'paper_artifacts/tsc_v155m_cologne_expanded_data_closed_loop_v1.json').read_text())
    templates = {}
    for arm in summary.ARMS:
        original = next(row for row in baseline['cells'] if row['arm'] == arm)
        templates[arm] = json.loads((ROOT / original['source_result']).read_text())
    cells = []
    for index, seed in enumerate(summary.SEEDS):
        for arm in summary.ARMS:
            row = deepcopy(templates[arm])
            row.update(protocol=summary.PROTOCOL, scenario='cologne1', seed=seed, arm=arm,
                status='COMPLETE', run_valid=True, validity_checks={'runtime_complete': True}, zero_incident_status='PASS',
                source_root=protocol['source_root'], repaired_sumocfg=protocol['sumocfg'],
                task_id=f'fixture-{seed}-{arm}', task_signature=summary.task_signature(arm, seed), source_result='fixture-only.json',
                scheduler_task_signature=summary.task_signature(arm, seed), task_attempt=1,
                occupancy_equation_protocol=summary.OCCUPANCY_PROTOCOL, sumo_version=[22, 'SUMO 1.22.0'],
                runtime_bindings={'evaluator': protocol['source_root'] + '/cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py',
                                  'executor': protocol['source_root'] + '/cf_h2o/traffic_signal/safe_phase_controller.py'},
                yellow_priority_check={'passed': True, 'initial_state': 'Ggrr', 'target_state': 'rrGG', 'clearance_state': 'Yyrr'},
                deployed_policy='phase_pressure' if arm == summary.PP else 'causal_rigid_advantage_contrast_source_utility')
            if arm == summary.PP:
                row.update(model_path=None, frozen_threshold=None, originator_diagnostics=None)
                row['metrics'].update(accepted_intervention_trace=[], guard_audit={}, residual_deployment={})
            cost = index + 6. if arm == summary.NEW else 10. if arm == summary.ORIGINAL_A else 8.
            row['metrics'].update(mean_queue=cost, mean_queue_per_lane=cost / summary.LANES,
                halted_vehicle_seconds=cost * summary.SAMPLES, fixed_horizon_sample_seconds=summary.SAMPLES,
                controlled_lane_count=summary.LANES)
            cells.append(row)
    return protocol, cells


def test_eighteen_new_cells_equal_seed_contrasts_and_primary_decisions(inputs):
    report = summary.aggregate(*inputs)
    service = report['all_seed_service']
    assert len(report['cells']) == report['new_simulation_count'] == 18
    assert report['reused_cell_count'] == report['new_model_fit_count'] == 0
    assert service['seeds'] == list(summary.SEEDS)
    assert service['means_by_arm'][summary.NEW]['mean_queue'] == 8.5
    assert service['expanded_vs_original_a']['mean_queue']['difference_mean'] == -1.5
    pp = service['expanded_vs_phase_pressure']['mean_queue']
    assert pp['difference_mean'] == .5
    assert pp['relative_difference_of_means_percent'] == 6.25
    assert [row['expanded_minus_reference'] for row in pp['paired_differences']] == [-2, -1, 0, 1, 2, 3]
    assert pp['paired_sign_counts'] == {'lower': 2, 'higher': 3, 'equal': 1}
    assert report['decision']['b195_data_effect_retained']
    assert not report['decision']['b195_beats_phase_pressure']
    assert report['primary_efficiency']['primary_contrast'] == 'expanded_vs_phase_pressure'
    assert report['execution_by_model_arm'][summary.NEW]['totals']['actual_accepted_overrides'] == 48
    assert report['safety_by_arm'][summary.NEW]['zero_incident_seed_count'] == 6
    assert report['safety_by_arm'][summary.NEW]['zero_incident_status'] == 'PASS'


def test_waiting_does_not_replace_queue_and_equal_cost_is_not_improvement(inputs):
    protocol, cells = inputs
    for row in cells:
        if row['arm'] == summary.NEW:
            row['metrics'].update(mean_queue=8., mean_queue_per_lane=1., halted_vehicle_seconds=8 * summary.SAMPLES,
                                  mean_vehicle_waiting_time=0.)
    report = summary.aggregate(protocol, cells)
    assert report['decision']['b195_data_effect_retained']
    assert not report['decision']['b195_beats_phase_pressure']
    assert report['all_seed_service']['expanded_vs_phase_pressure']['mean_queue']['paired_sign_counts']['equal'] == 6


def test_colliding_valid_cell_retained_and_invalid_cell_disables_whole_matrix(inputs):
    protocol, cells = inputs
    cells[0].update(zero_incident_status='FAIL')
    cells[0]['metrics'].update(collision_events=2, collision_incidents=1)
    report = summary.aggregate(protocol, cells)
    assert report['matrix_status'] == 'COMPLETE' and report['all_seed_service'] is not None
    assert report['safety_by_arm'][summary.NEW]['zero_incident_status'] == 'FAIL'
    assert len(report['all_seed_service']['expanded_vs_phase_pressure']['mean_queue']['paired_differences']) == 6
    cells[0].update(run_valid=False, status='INVALID', validity_checks={'runtime_complete': False})
    report = summary.aggregate(protocol, cells)
    assert report['matrix_status'] == 'INVALID'
    assert report['all_seed_service'] is None and report['decision'] is None
    assert len(report['cells']) == 18


@pytest.mark.parametrize('change', ['missing', 'duplicate_task', 'signature', 'executor', 'yellow', 'sumo',
                                    'network', 'model', 'threshold', 'spacing', 'denominator', 'pp_model'])
def test_incomplete_matrix_or_changed_frozen_binding_is_rejected(inputs, change):
    protocol, cells = inputs
    if change == 'missing':
        cells.pop()
    elif change == 'duplicate_task':
        cells[1]['task_id'] = cells[0]['task_id']
    elif change == 'signature':
        cells[0]['task_signature'] = summary.task_signature(summary.NEW, 52282)
    elif change == 'executor':
        cells[0]['runtime_bindings']['executor'] = 'other-executor.py'
    elif change == 'yellow':
        cells[0]['yellow_priority_check']['clearance_state'] = 'yyrr'
    elif change == 'sumo':
        cells[0]['sumo_version'] = [23, 'SUMO 1.23.0']
    elif change == 'network':
        cells[0]['repaired_sumocfg'] = 'other.sumocfg'
    elif change == 'model':
        cells[0]['model_path'] = protocol['original_a_frozen_threshold']['model_path']
    elif change == 'threshold':
        protocol['original_a_frozen_threshold']['threshold'] = .1
    elif change == 'spacing':
        cells[0]['metrics']['residual_deployment']['cooldown_intervals_override'] = 0
    elif change == 'denominator':
        cells[0]['metrics']['fixed_horizon_sample_seconds'] -= 1
    else:
        cells[2]['model_path'] = protocol['frozen_threshold']['model_path']
    with pytest.raises(ValueError):
        summary.aggregate(protocol, cells)


def test_effective_launch_record_preserves_reporting_retry_provenance(inputs):
    _, cells = inputs
    records = []
    for index, row in enumerate(cells):
        logical = row['task_signature']
        retry = row['arm'] == summary.PP
        records.append({'logical_signature': logical,
                        'scheduler_signature': logical + '/reporting-fix1' if retry else logical,
                        'task_id': f'task-{index}', 'attempt': 2 if retry else 1})
    mapped = summary.effective_task_records({'effective_tasks': records})
    assert len(mapped) == 18
    assert mapped[summary.task_signature(summary.PP, summary.SEEDS[0])]['attempt'] == 2
