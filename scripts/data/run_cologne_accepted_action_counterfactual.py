"""Replay the preselected V154M decision with the original 450-second label."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

from cologne_action_branch_checks import physical_snapshot, compare_physical


PROTOCOL = 'tsc-v154n-cologne-accepted-action-counterfactual-v1'
TLS = 'cluster_357187_359543'
SEED = 52282
TARGET_TIME = 25800.0


def load_tool(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_vectors(left, right, tolerance=1e-8):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    same_shape = a.shape == b.shape
    difference = float(np.max(np.abs(a - b))) if same_shape and a.size else None
    return {'passed': bool(same_shape and difference is not None and difference <= tolerance),
            'left_count': int(a.size), 'right_count': int(b.size),
            'max_absolute_difference': difference}


def run(protocol, output):
    from cf_h2o.eval import traffic_signal_cologne_collision_replay as runtime
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import _runtime_models, RUNTIME_POLICY

    started = time.monotonic()
    if protocol['protocol'] != PROTOCOL:
        raise ValueError('Use the frozen V154N protocol')
    if os.environ['CFCMT_SOURCE_ROOT'] != protocol['source_root']:
        raise ValueError('Use the unchanged V154J source')
    if output.exists():
        raise FileExistsError(output)
    frozen_path = Path(protocol['frozen_threshold_path'])
    frozen = json.loads(frozen_path.read_text())
    if frozen != protocol['frozen_threshold']:
        raise ValueError('The frozen model/threshold binding changed')
    reference = json.loads(Path(protocol['closed_loop_reference']).read_text())
    selected_trace = protocol['selected_trace']
    traces = reference['metrics']['accepted_intervention_trace']
    if (reference['seed'] != SEED or selected_trace not in traces
            or selected_trace['time_sec'] != TARGET_TIME
            or selected_trace != max(traces, key=lambda row: row['priority'])):
        raise ValueError('The preselected highest-score decision changed')
    sumocfg = Path(protocol['sumocfg'])
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(sumocfg))
    geometry_reference = json.loads(Path(protocol['geometry_reference']).read_text())
    if geometry != geometry_reference['bilateral_static_geometry']:
        raise ValueError('The fixed bilateral network changed')
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = protocol['conversion_root']
    threshold_tool = load_tool(protocol['threshold_originator_path'], 'v154n_frozen_threshold')
    api = runtime.load_libsumo()
    if '1.22.0' not in str(runtime.libsumo_version()):
        raise ValueError('Use SUMO 1.22.0')
    scratch = Path(protocol['scratch_root'])
    scratch.mkdir(parents=True, exist_ok=False)
    snapshot_path = scratch / 'before_action.xml.gz'
    captured, prefix_costs = {}, []

    class CapturingOriginator(threshold_tool.ThresholdedFractionRigidOriginator):
        def prepare_interval(self, **kwargs):
            super().prepare_interval(**kwargs)
            if float(api.simulation.getTime()) != TARGET_TIME:
                return
            states, executors = kwargs['states'], kwargs['executors']
            infos = {tls: state.info for tls, state in states.items()}
            captured.update(states=states, infos=infos, executors=executors,
                            snapshots={tls: ex.snapshot() for tls, ex in executors.items()},
                            physical=physical_snapshot(api, executors), context=states[TLS].context,
                            controlled_lanes=tuple(sorted(v3._controlled_lane_set(infos))))
            api.simulation.saveState(str(snapshot_path))
            (scratch / 'executor_state.json').write_text(json.dumps(captured['snapshots'], indent=2) + '\n')

        def select(self, contrast, *, reference_index):
            decision = super().select(contrast, reference_index=reference_index)
            if float(api.simulation.getTime()) == TARGET_TIME:
                phases = list(contrast.metadata['candidate_states'])
                captured['decision'] = {
                    'selected_phase': str(phases[decision.selected_index]),
                    'reference_phase': str(phases[decision.reference_index]),
                    'selected_index': int(decision.selected_index),
                    'reference_index': int(decision.reference_index),
                    'predicted_advantage': float(decision.reference_score - decision.predicted_score),
                    'candidate_states': [str(value) for value in phases],
                }
            return decision

    originator = CapturingOriginator.load(
        Path(frozen['model_path']), frozen_threshold_path=frozen_path, expected_city='cologne',
        network_context=runtime.read_network_right_of_way_context(runtime.net_file_from_sumocfg(sumocfg)))

    def observer(*, stage, time_sec, executors):
        if stage == 'after_executor_advance' and TARGET_TIME < time_sec <= TARGET_TIME + 10:
            lanes = captured['controlled_lanes']
            prefix_costs.append(float(sum(api.lane.getLastStepHaltingNumber(lane) for lane in lanes)) / len(lanes))

    metrics = runtime.evaluate_policy_v3(
        sumo_api=api, sumocfg=sumocfg, scenario='cologne1', policy=RUNTIME_POLICY,
        models=_runtime_models(originator, prediction_horizon_sec=450), duration_sec=610,
        control_interval_sec=10, warmup_sec=60, seed=SEED, tripinfo_output=scratch / 'prefix_tripinfo.xml',
        residual_coordination_mode='direct', residual_cooldown_intervals_override=0, step_observer=observer)
    expected_trace = [row for row in traces if row['time_sec'] < TARGET_TIME + 10]
    checks = {
        'prefix_ok': metrics.get('ok') is True,
        'prefix_trace_exact': metrics.get('accepted_intervention_trace') == expected_trace,
        'prefix_zero_collisions': metrics.get('collision_events') == 0 and metrics.get('collision_incidents') == 0,
        'prefix_zero_teleports': metrics.get('starting_teleports') == 0 and metrics.get('ending_teleports') == 0,
        'snapshot_captured': 'decision' in captured and snapshot_path.exists(),
        'prefix_first_interval_complete': len(prefix_costs) == 10,
    }
    result = {'protocol': PROTOCOL, 'status': 'INVALID', 'seed': SEED, 'target_time_sec': TARGET_TIME,
              'source_root': protocol['source_root'], 'sumocfg': str(sumocfg), 'frozen_threshold': frozen,
              'selected_trace': selected_trace, 'checks': checks, 'branches': {},
              'prefix': {'metrics': metrics, 'expected_accepted_count': len(expected_trace),
                         'native_first_interval_costs': prefix_costs},
              'snapshot_path_server_only': str(snapshot_path),
              'label_contract': protocol['label_contract']}
    if checks['snapshot_captured']:
        decision = captured['decision']
        result['decision'] = decision
        checks.update(
            selected_phase_exact=decision['selected_phase'] == selected_trace['selected_phase_state'],
            reference_phase_exact=decision['reference_phase'] == selected_trace['prior_phase_state'],
            predicted_advantage_exact=decision['predicted_advantage'] == selected_trace['priority'],
            controlled_lane_count_exact=len(captured['controlled_lanes']) == 8,
            single_focal_tls=list(captured['states']) == [TLS])
        result['controlled_lanes'] = list(captured['controlled_lanes'])
    if all(checks.values()):
        branches, starts, vectors = result['branches'], {}, {}
        v3._start_sumo(api, sumocfg, SEED)
        try:
            for arm in ('rigid', 'phase_pressure', 'rigid_replay'):
                if arm == 'rigid_replay' and branches['rigid']['status'] != 'PASS':
                    branches[arm] = {'status': 'NOT_RUN', 'reason': 'Primary rigid branch did not complete safely'}
                    continue
                v3._reload_sumo_state(api, sumocfg, SEED, snapshot_path, begin_time=TARGET_TIME)
                v3._restore_executors(captured['executors'], captured['snapshots'])
                physical = physical_snapshot(api, captured['executors'])
                starts[arm] = physical
                restore_check = compare_physical(captured['physical'], physical)
                checks[f'{arm}_restore_matches_prefix'] = restore_check['passed']
                phase = decision['reference_phase'] if arm == 'phase_pressure' else decision['selected_phase']
                branch = branches[arm] = {'status': 'INVALID', 'first_phase_request': phase,
                    'restoration': restore_check, 'restored_pending_raw': physical['pending_raw']}
                reference_phase = v3._rule_candidate(captured['states'][TLS], captured['executors'][TLS], 'phase_pressure').state
                checks[f'{arm}_pp_reference_matches'] = reference_phase == decision['reference_phase']
                if not restore_check['passed'] or not checks[f'{arm}_pp_reference_matches']:
                    continue
                ledger = v3._new_safety_ledger()
                try:
                    outcome = v3._counterfactual_branch_outcome_v3(
                        sumo_api=api, infos=captured['infos'], states=captured['states'],
                        executors=captured['executors'], focal_id=TLS,
                        candidate=v3._candidate_for_state(captured['infos'][TLS], phase),
                        context=captured['context'], control_interval_sec=10,
                        counterfactual_horizon_intervals=45, controlled_lane_count=8,
                        controlled_lanes=captured['controlled_lanes'], counterfactual_cost_mode='halted_queue',
                        safety_ledger=ledger)
                except v3.UnsafeRolloutError as error:
                    branch.update(status='FAIL_UNSAFE', safety_audit=error.safety_audit,
                                  error=str(error), stop_time_sec=float(api.simulation.getTime()))
                else:
                    costs, one_step, terminal = outcome
                    branch.update(status='PASS', safety_audit=v3._finalize_safety_ledger(ledger),
                                  stop_time_sec=float(api.simulation.getTime()), costs=costs,
                                  mean_halted_per_lane=float(np.mean(costs)), one_step=one_step, terminal=terminal)
                    vectors[arm] = v3._counterfactual_outcome_vector(*outcome)
                    checks[f'{arm}_full_horizon'] = len(costs) == 450 and branch['stop_time_sec'] == TARGET_TIME + 450
                print(json.dumps({'arm': arm, 'status': branch['status'], 'stop_time_sec': branch.get('stop_time_sec')}), flush=True)
        finally:
            api.close()
        result['captured_pending_raw'] = captured['physical']['pending_raw']
        result['start_pair_comparisons'] = {
            arm: compare_physical(starts['rigid'], start, tolerance=0)
            for arm, start in starts.items() if arm != 'rigid'}
        checks['all_branch_starts_exact'] = all(row['passed'] for row in result['start_pair_comparisons'].values())
        if 'rigid' in vectors:
            result['first_interval_replay'] = compare_vectors(prefix_costs, branches['rigid']['costs'][:10])
            checks['first_interval_replay_exact'] = result['first_interval_replay']['passed']
        if 'rigid' in vectors and 'rigid_replay' in vectors:
            result['same_action_replay'] = compare_vectors(vectors['rigid'], vectors['rigid_replay'])
            checks['same_action_replay_exact'] = result['same_action_replay']['passed']
        complete = all(branches[arm]['status'] == 'PASS' for arm in ('rigid', 'phase_pressure', 'rigid_replay'))
        if all(checks.values()) and complete:
            rigid, pp = (branches[arm]['mean_halted_per_lane'] for arm in ('rigid', 'phase_pressure'))
            result['status'] = 'PASS'
            result['comparison'] = {'rigid_cost': rigid, 'phase_pressure_cost': pp,
                'rigid_minus_pp': rigid - pp, 'actual_advantage_pp_minus_rigid': pp - rigid,
                'relative_cost_change_pct': (rigid / pp - 1) * 100 if pp else None,
                'ranking': 'harmful' if rigid > pp + 1e-12 else 'beneficial' if rigid < pp - 1e-12 else 'tie'}
        elif any(branch['status'] == 'FAIL_UNSAFE' for branch in branches.values()):
            result['status'] = 'FAIL_UNSAFE'
    result['elapsed_sec'] = time.monotonic() - started
    output.mkdir(parents=True)
    (output / 'result.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('protocol', 'status', 'elapsed_sec')}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.protocol.read_text()), args.output)
    if result['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
