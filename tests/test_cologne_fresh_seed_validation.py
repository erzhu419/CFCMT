import importlib
import json
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from cf_h2o.traffic_signal import safe_phase_controller as safe


ROOT = Path(__file__).parents[1]
CLUSTER = ROOT / 'cf_h2o/results/cluster'


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/data'))
    return importlib.import_module('run_cologne_fresh_seed_validation')


@pytest.fixture
def protocol(runner):
    p = json.loads((CLUSTER / 'tsc_v155m_cologne_expanded_data_closed_loop_20260910/protocol_v1.json').read_text())
    a = json.loads((CLUSTER / 'tsc_v155b_cologne_raw_target_closed_loop_20260910/protocol_v1.json').read_text())
    p.update(protocol=runner.PROTOCOL, arms=list(runner.ARMS),
             original_a_frozen_threshold=a['frozen_threshold'],
             original_a_frozen_threshold_path=a['frozen_threshold_path'])
    p['evaluation']['seeds'] = list(runner.SEEDS)
    return p


def test_fresh_roster_excludes_original_seeds_and_rejects_changed_threshold(runner, protocol):
    assert runner.SEEDS == (152282, 141242, 163792, 252282, 241242, 263792)
    for seed in runner.SEEDS:
        runner.validate_protocol(protocol, seed, runner.PP)
    with pytest.raises(ValueError, match='six fixed fresh seeds'):
        runner.validate_protocol(protocol, 52282, runner.EXPANDED)
    protocol['original_a_frozen_threshold']['threshold'] = .1
    with pytest.raises(ValueError, match='threshold zero'):
        runner.validate_protocol(protocol, runner.SEEDS[0], runner.ORIGINAL_A)


def test_actual_executor_preserves_yellow_priority_and_rejects_old_mapping(runner, monkeypatch):
    assert runner.yellow_priority_check(safe.SafePhaseExecutor, safe.PhaseTiming)['clearance_state'] == 'Yyrr'
    monkeypatch.setattr(safe, '_clearance_yellow_state', lambda state: ''.join('y' if c in 'Gg' else 'r' for c in state))
    with pytest.raises(ValueError, match='G->Y and g->y'):
        runner.yellow_priority_check(safe.SafePhaseExecutor, safe.PhaseTiming)


@pytest.mark.parametrize('arm_name,budget', [('EXPANDED', 195), ('ORIGINAL_A', 100)])
def test_each_model_loads_its_frozen_fit_and_original_veto_loader(runner, protocol, tmp_path, monkeypatch, arm_name, budget):
    arm = getattr(runner, arm_name)
    expanded = arm == runner.EXPANDED
    frozen_key = 'frozen_threshold' if expanded else 'original_a_frozen_threshold'
    path_key = 'frozen_threshold_path' if expanded else 'original_a_frozen_threshold_path'
    frozen = protocol[frozen_key]
    fit_name = 'tsc_v155l_cologne_expanded_data_full_fit_20260910' if expanded else 'tsc_v155a_cologne_raw_target_refit_20260910'
    fit = json.loads((CLUSTER / fit_name / 'fit_v1/result.json').read_text())
    fit_path, frozen_path = tmp_path / 'fit.json', tmp_path / 'frozen.json'
    frozen['fit_result'] = str(fit_path)
    protocol[path_key] = str(frozen_path)
    fit['frozen_threshold'] = deepcopy(frozen)
    if expanded:
        calibration_path = tmp_path / 'calibration.json'
        calibration = json.loads((CLUSTER / 'tsc_v155k_cologne_expanded_data_oof_20260910/fit_v1/result.json').read_text())
        calibration_path.write_text(json.dumps(calibration))
        protocol.update(full_fit_result=str(fit_path), calibration_result=str(calibration_path))
        frozen['calibration_result'] = str(calibration_path)
        fit['refs']['k_result'] = str(calibration_path)
    else:
        frozen['calibration_result'] = str(fit_path)
    fit['frozen_threshold'] = deepcopy(frozen)
    fit_path.write_text(json.dumps(fit))
    frozen_path.write_text(json.dumps(frozen))
    calls = []
    originator = SimpleNamespace(payload={
        'target_budget': budget, 'selected_groups': protocol['training_group_ids'] if expanded else fit['selected_groups'],
        'rigid_model': SimpleNamespace(anchor_model=SimpleNamespace(feature_names=protocol['original_feature_names']))})

    class Loader:
        @classmethod
        def load(cls, path, **kwargs):
            calls.append((str(path), kwargs))
            return originator

    def load_tool(path, name):
        expected_path = protocol['expanded_originator_path'] if expanded else protocol['ablation_originator_path']
        assert path == expected_path
        return SimpleNamespace(ExpandedDataNoStayOriginator=Loader, NoStayFractionRigidOriginator=Loader)

    monkeypatch.setattr(runner, 'load_tool', load_tool)
    runtime = SimpleNamespace(net_file_from_sumocfg=lambda path: 'network',
                              read_network_right_of_way_context=lambda path: 'context')
    loaded, actual_frozen = runner.load_originator(protocol, arm, runtime)
    assert loaded is originator and actual_frozen == frozen
    assert calls[0][0] == frozen['model_path'] and calls[0][1]['frozen_threshold_path'] == frozen_path
    frozen_path.write_text(json.dumps({**frozen, 'model_path': 'other-model.pkl'}))
    with pytest.raises(ValueError, match='model or threshold binding changed'):
        runner.load_originator(protocol, arm, runtime)


@pytest.mark.parametrize('arm_name', ['EXPANDED', 'ORIGINAL_A', 'PP'])
def test_three_arms_share_runtime_and_keep_colliding_complete_cells(runner, protocol, tmp_path, monkeypatch, arm_name):
    from cf_h2o.eval import traffic_signal_cologne_collision_replay as runtime
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.eval import traffic_signal_state_conditioned_source_closed_loop as source_runtime
    comparison = importlib.import_module('run_cologne_repaired_controller_comparison')
    arm = getattr(runner, arm_name)
    protocol['source_root'] = str(ROOT)
    for key in ('frozen_threshold', 'original_a_frozen_threshold'):
        protocol[key]['source_root'] = str(ROOT)
    protocol['scratch_root'] = str(tmp_path / 'scratch')
    reference_path = tmp_path / 'geometry.json'
    reference_path.write_text(json.dumps({'bilateral_static_geometry': {}, 'focal_demand': {'vehicles': {
        vehicle: {'route': ['in', 'out']} for vehicle in comparison.full.FOCAL_IDS}}}))
    protocol['geometry_reference'] = str(reference_path)
    monkeypatch.setenv('CFCMT_SOURCE_ROOT', str(ROOT))
    recorded = json.loads((CLUSTER / 'tsc_v155m_cologne_expanded_data_closed_loop_20260910/evaluation/seed_52282/rigid_thresholded/result.json').read_text())
    metrics, inventory = deepcopy(recorded['metrics']), deepcopy(recorded['collision_inventory'])
    metrics.update(collision_events=1, collision_incidents=1, collision_event_steps=1)
    inventory.update(native_event_count=1, native_incident_count=1, collision_event_steps=1)
    if arm == runner.PP:
        metrics['accepted_intervention_trace'] = []
        metrics['guard_audit'] = {}
        metrics['residual_deployment'] = {}
    originator = None if arm == runner.PP else SimpleNamespace(diagnostics=lambda: {'target_budget': 195 if arm == runner.EXPANDED else 100})
    frozen = None if arm == runner.PP else protocol['frozen_threshold' if arm == runner.EXPANDED else 'original_a_frozen_threshold']
    monkeypatch.setattr(runner, 'load_originator', lambda *_: (originator, frozen))
    model_object = object()
    model_loads = []

    def runtime_models(actual_originator, **kwargs):
        model_loads.append((actual_originator, kwargs))
        return model_object

    monkeypatch.setattr(source_runtime, '_runtime_models', runtime_models)
    monkeypatch.setattr(runtime, 'static_geometry', lambda _: {})
    monkeypatch.setattr(runtime, 'net_file_from_sumocfg', lambda _: 'network')
    monkeypatch.setattr(runtime, 'load_libsumo', lambda: object())
    monkeypatch.setattr(runtime, 'libsumo_version', lambda: (22, 'SUMO 1.22.0'))
    observer = SimpleNamespace(step_times=list(range(25201, 28801)), inventory=lambda: inventory)
    tooling = SimpleNamespace(full=SimpleNamespace(FOCAL_IDS=comparison.full.FOCAL_IDS, FullObserver=lambda *_: observer),
                              RouteAwarePassageTracker=lambda *_: None, comparison_validity=comparison.comparison_validity)
    monkeypatch.setattr(runner, 'load_tool', lambda *_: tooling)
    evaluations = []

    def evaluate(**kwargs):
        evaluations.append(kwargs)
        return metrics

    monkeypatch.setattr(runtime, 'evaluate_policy_v3', evaluate)
    monkeypatch.setattr(v3, 'evaluate_policy_v3', evaluate)
    result = runner.run(protocol, runner.SEEDS[0], arm, tmp_path / 'out')
    call = evaluations[0]
    assert call['duration_sec'] == 3600 and call['warmup_sec'] == 60 and call['control_interval_sec'] == 10
    assert call['seed'] == 152282
    assert call['models'] is (None if arm == runner.PP else model_object)
    assert call['policy'] == ('phase_pressure' if arm == runner.PP else source_runtime.RUNTIME_POLICY)
    assert call['residual_cooldown_intervals_override'] == (0 if arm == runner.PP else 44)
    assert len(model_loads) == (0 if arm == runner.PP else 1)
    assert result['run_valid'] and result['status'] == 'COMPLETE' and result['zero_incident_status'] == 'FAIL'
    assert result['frozen_threshold'] == frozen
    assert json.loads((tmp_path / 'out/result.json').read_text())['arm'] == arm


def test_cli_prints_complete_only_for_valid_output(runner, tmp_path, monkeypatch, capsys):
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['runner', '--protocol', str(protocol_path), '--seed', '152282',
                                    '--arm', runner.PP, '--output', str(tmp_path / 'out')])
    result = {'run_valid': True, 'status': 'COMPLETE', 'zero_incident_status': 'FAIL'}
    monkeypatch.setattr(runner, 'run', lambda *_: result)
    runner.main()
    assert capsys.readouterr().out == 'Eval complete\n'
    result['run_valid'] = False
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2 and capsys.readouterr().out == ''
