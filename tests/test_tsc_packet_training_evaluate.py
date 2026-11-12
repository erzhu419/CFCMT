"""Synthetic independent-calibration checks; never fit or start SUMO."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.tsc_target_packet_kernel_20261109.kernel import PacketKernel
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    coefficients, corridor, executor_seed, lane, laws, packet, road_graph,
)
from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_target_packet_training_20261109 import evaluate as e


class NeverCopyFuture:
    def __deepcopy__(self, memo):
        raise AssertionError('A future/route field entered a model state')


class GuardedLabels(list):
    ready = staticmethod(lambda: True)
    def __iter__(self):
        assert self.ready(), 'Future outcome labels were inspected before free rollouts completed'
        return super().__iter__()


class GuardedExecution(dict):
    ready = staticmethod(lambda: True)
    def __getitem__(self, key):
        if key in ('requests', 'applied_states', 'due_ids'):
            assert self.ready(), 'Actual future execution entered the free phase'
        return super().__getitem__(key)


def frame(static, packets, time_s, executor_seeds=None, arrived=()):
    packets = {p['packet_id']: deepcopy(p) for p in packets}
    sensors = PacketObserver(static)._reconstruct(packets)
    for p in packets.values():
        p['route'] = NeverCopyFuture()
        p['future_response'] = NeverCopyFuture()
    return dict(time_s=time_s, packets=packets, native_lanes=sensors,
        native_active_ids=list(packets), native_pending_ids=[], departed_ids=[],
        arrived_ids=GuardedLabels(arrived), unresolved=[], checks=dict(admitted=True),
        executors={tls: dict(snapshot=deepcopy(seed['snapshot'])) for tls, seed in (executor_seeds or {}).items()})


def panel_fixture():
    protocol = json.loads((Path(e.__file__).parents[1]/'tsc_target_packet_protocol_20261109/protocol.json').read_text())
    static = road_graph([lane(f'l{i}', 100) for i in range(144)], [], cost_lanes=[f'l{i}' for i in range(144)])
    profile = packet('p', 'l0', 10, 0)
    inputs = {}
    for seed in e.CALIBRATION_SEEDS:
        inputs[seed] = dict(frames=[frame(static, [profile], t) for t in range(1801)],
            execution=[GuardedExecution(time_sec=t, requests={}, applied_states={}, due_ids=NeverCopyFuture())
                       for t in range(1800)], executor_seeds={})
    model = dict(coefficients=coefficients(start=0), target_laws=laws(static),
                 training_seeds=list(protocol['cohorts']['training_seeds']))
    return static, inputs, model, protocol


def test_fixed_panel_free_phase_cannot_read_future_and_has_32_draws_then_360_diagnostics():
    class CountKernel(PacketKernel):
        steps_called = 0
        def step(self, initial_requests=None):
            result = super().step(initial_requests)
            type(self).steps_called += 1
            return result
    GuardedLabels.ready = GuardedExecution.ready = staticmethod(lambda: CountKernel.steps_called >= 32*450)
    try:
        static, inputs, model, protocol = panel_fixture()
        events = []
        result = e.evaluate_panel(static, inputs, model, protocol, kernel_cls=CountKernel,
                                  on_draw=lambda row: events.append((row['status'], row['checked_steps'])))
        assert result['status'] == 'PASS_TARGET_PACKET_TECHNICAL_CALIBRATION'
        assert all(result['checks'].values())
        assert result['completed_draws'] == result['planned_draws'] == 32
        assert result['failed_draws'] == result['unstarted_draws'] == 0
        assert events.count(('STARTED', 0)) == events.count(('COMPLETE', 450)) == 32
        assert CountKernel.steps_called == 32*450+360
        assert len(result['numerical_calibration']['windows']) == 4
        assert all(row['absolute_cost_error'] == 0 for row in result['numerical_calibration']['windows'])
        assert all(row['relative_active_error'] == 0 for row in result['numerical_calibration']['windows'])
        for seed in result['one_step']['per_seed']:
            assert seed['samples'] == 180 and seed['speed_pairs'] == seed['observed_survivors'] == 180
            assert seed['missing_predicted_survivors'] == 0 and seed['speed_MAE'] == 0
        assert result['new_native_starts'] == result['new_real_data_fits'] == 0
    finally:
        GuardedLabels.ready = GuardedExecution.ready = staticmethod(lambda: True)


def test_independent_audit_rejects_true_kernel_flag_and_keeps_all_failed_unstarted_draws():
    class DroppingKernel(PacketKernel):
        def step(self, initial_requests=None):
            row = super().step(initial_requests)
            self.state['packets'].clear()  # Its returned self-check still says True.
            assert row['ledger_passed']
            return row
    static, inputs, model, protocol = panel_fixture()
    records = []
    result = e.evaluate_panel(static, inputs, model, protocol, kernel_cls=DroppingKernel, on_draw=records.append)
    assert result['status'] == 'HOLD_TARGET_PACKET_CALIBRATION'
    assert result['failed_draws'] == 1 and result['unstarted_draws'] == 31
    assert result['completed_draws'] == 0 and len(result['draws']) == 32
    assert 'identity ledger' in result['draws'][0]['error']
    assert result['draws'][0]['checked_steps'] == 0
    assert result['numerical_calibration'] is result['one_step'] is None
    assert sum(row['status'] == 'STARTED' for row in records) == 1
    assert sum(row['status'] == 'NOT_RUN' for row in records) == 31
    failed = next(row for row in records if row['status'] == 'FAILED')
    evidence = failed['failure_evidence']
    assert evidence['partial_halted_total'] == evidence['partial_active_count'] == [1]
    assert evidence['partial_pending_count'] == [0]
    assert set(evidence['last_before_state']['packets']) == {'p'}
    assert evidence['last_before_state']['time_s'] == 600
    assert evidence['last_after_state']['packets'] == {}
    assert evidence['last_after_state']['time_s'] == 601
    assert 'failure_evidence' not in result['draws'][0]
    json.dumps(failed, allow_nan=False)
    leaked = dict(model, training_seeds=[*model['training_seeds'][:-1], e.CALIBRATION_SEEDS[0]])
    with pytest.raises(ValueError, match='training/calibration'):
        e.evaluate_panel(static, inputs, leaked, protocol)


def test_conditional_speed_diagnostic_reports_predicted_END_of_true_survivor_as_missing():
    static, target_laws = corridor(incoming_length=20, internal_lengths=(5,), outgoing_length=100)
    seeds = {'J': executor_seed(('G',), clearance=('I0',))}
    p = packet('survivor', 'out', 99, 5)
    before, actual = frame(static, [p], 0, seeds), frame(static, [p], 1, seeds)
    model = dict(coefficients=coefficients(), target_laws=target_laws)
    value = e._one_step(static, before, actual, dict(requests={'J': 'G'}, applied_states={'J': 'G'}), seeds, model)
    assert value['observed_survivors'] == value['missing_predicted_survivors'] == 1
    assert value['speed_pairs'] == 0
    assert value['predicted_terminals'] == 1 and value['actual_terminals'] == 0
    assert value['lane_error_sums']['N'] == 1


def test_truth_uses_poststeps_and_previous_observed_lane_for_nonsink_terminal():
    static, target_laws = corridor(incoming_length=20, internal_lengths=(5,))
    assert static['successors']['in']
    frames = {600: frame(static, [packet('exit', 'in', 19, 0)], 600)}
    frames.update({t: frame(static, [], t, arrived=['exit'] if t == 601 else []) for t in range(601, 1051)})
    actual = e._truth_window(static, frames, e.CALIBRATION_SEEDS[0], 600)
    assert actual['halted_total'][0] == 1 and sum(actual['halted_total'][1:]) == 0
    assert actual['active_count'][0] == 1 and actual['active_count'][1:] == [0]*450
    assert actual['terminal_count'] == actual['nonsink_terminal_count'] == 1


def test_parallel_first_failure_retains_four_open_draws_and_never_schedules_the_rest():
    static, target_laws = corridor()
    seeds = {'J': executor_seed(('G',), clearance=('I0', 'I1'))}
    initial = e.initialize_observation(frame(static, [packet('p', 'in', 9, 0)], 0, seeds), 0)
    invalid = coefficients()
    invalid['start'] = invalid['start'][:-1]
    model = dict(coefficients=invalid, target_laws=target_laws)
    tasks = [(static, initial, seeds, model, seed, anchor, scenario)
             for seed in e.CALIBRATION_SEEDS for anchor in e.ANCHORS for scenario in e.SCENARIOS]
    events = []
    rows = e._free_tasks(tasks, 4, events.append, None)
    assert sum(row['status'] == 'FAILED' for row in rows) == 4
    assert sum(row['status'] == 'NOT_RUN' for row in rows) == 28
    assert sum(row['status'] == 'STARTED' for row in events) == 4
    assert all(row['checked_steps'] == 0 for row in rows)


def test_free_accuracy_pass_cannot_override_incomplete_conditional_engineering_records(monkeypatch):
    static, inputs, model, protocol = panel_fixture()
    def perfect_free(tasks, workers, on_draw, kernel_cls):
        return [dict(seed=task[4], anchor_sec=task[5], scenario_seed=task[6], status='COMPLETE',
            structural_passed=True, checked_steps=450, halted_total=[1]*451,
            active_count=[1]*451, pending_count=[0]*451, terminal_count=0, nonsink_terminal_count=0)
            for task in tasks]
    monkeypatch.setattr(e, '_free_tasks', perfect_free)
    monkeypatch.setattr(e, '_one_step_panel', lambda *_: dict(status='HOLD_CONDITIONAL_DIAGNOSTIC', per_seed=[
        dict(seed=e.CALIBRATION_SEEDS[0], samples=180, failures=[]),
        dict(seed=e.CALIBRATION_SEEDS[1], samples=0, failures=[dict(time_s=0, error='Unresolved initial packet')])]))
    result = e.evaluate_panel(static, inputs, model, protocol)
    assert result['numerical_calibration']['status'] == 'PASS_NUMERICAL_CALIBRATION'
    assert result['checks']['all_scenarios_structurally_complete']
    assert result['checks']['numerical_calibration_passed']
    assert result['checks']['conditional_records_complete'] is False
    assert result['status'] == 'HOLD_TARGET_PACKET_CALIBRATION'
