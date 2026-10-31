"""Compare the selected actions after fresh, identical prefixes without loading a SUMO state."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from cologne_action_branch_checks import physical_snapshot, compare_physical
from run_cologne_accepted_action_counterfactual import load_tool, compare_vectors, TLS, SEED, TARGET_TIME


PROTOCOL = 'tsc-v154n-cologne-accepted-action-counterfactual-v2'
HANDOFF_MESSAGE = 'V154N live branch finished; stop the outer policy loop'


class BranchComplete(RuntimeError):
    """Hand control back through the evaluator's existing close/finally block."""


def run(protocol, output):
    from cf_h2o.eval import traffic_signal_cologne_collision_replay as runtime
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import _runtime_models, RUNTIME_POLICY

    started = time.monotonic()
    if protocol['protocol'] != PROTOCOL or os.environ['CFCMT_SOURCE_ROOT'] != protocol['source_root']:
        raise ValueError('Use the V154N live-prefix protocol and unchanged V154J core')
    if output.exists():
        raise FileExistsError(output)
    frozen_path = Path(protocol['frozen_threshold_path'])
    frozen = json.loads(frozen_path.read_text())
    if frozen != protocol['frozen_threshold']:
        raise ValueError('Frozen threshold/model binding changed')
    reference = json.loads(Path(protocol['closed_loop_reference']).read_text())
    trace = reference['metrics']['accepted_intervention_trace']
    target = protocol['selected_trace']
    if target != max(trace, key=lambda row: row['priority']) or target['time_sec'] != TARGET_TIME:
        raise ValueError('The preselected decision changed')
    sumocfg = Path(protocol['sumocfg'])
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(sumocfg))
    if geometry != json.loads(Path(protocol['geometry_reference']).read_text())['bilateral_static_geometry']:
        raise ValueError('The fixed network changed')
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = protocol['conversion_root']
    threshold = load_tool(protocol['threshold_originator_path'], 'v154n_live_threshold')
    api = runtime.load_libsumo()
    if '1.22.0' not in str(runtime.libsumo_version()):
        raise ValueError('Use SUMO 1.22.0')
    scratch = Path(protocol['scratch_root'])
    scratch.mkdir(parents=True, exist_ok=False)
    runs, captures, vectors = {}, {}, {}

    for arm in ('reference', 'rigid', 'phase_pressure', 'rigid_replay'):
        if arm == 'rigid_replay' and runs['rigid']['status'] != 'PASS':
            runs[arm] = {'status': 'NOT_RUN', 'reason': 'Primary rigid branch did not complete safely'}
            continue
        capture, decisions, first_costs = {}, [], []
        prefix_ledger = v3._new_safety_ledger()
        branch = {'status': 'INVALID'}

        class LiveOriginator(threshold.ThresholdedFractionRigidOriginator):
            def prepare_interval(self, **kwargs):
                super().prepare_interval(**kwargs)
                if float(api.simulation.getTime()) == TARGET_TIME:
                    states, executors = kwargs['states'], kwargs['executors']
                    infos = {tls: state.info for tls, state in states.items()}
                    capture.update(states=states, infos=infos, executors=executors,
                        physical=physical_snapshot(api, executors), context=states[TLS].context,
                        lanes=tuple(sorted(v3._controlled_lane_set(infos))))

            def select(self, contrast, *, reference_index):
                decision = super().select(contrast, reference_index=reference_index)
                phases = list(contrast.metadata['candidate_states'])
                record = {'time_sec': float(api.simulation.getTime()),
                    'selected_phase': str(phases[decision.selected_index]),
                    'reference_phase': str(phases[decision.reference_index]),
                    'selected_index': int(decision.selected_index), 'reference_index': int(decision.reference_index),
                    'priority': float(decision.priority),
                    'predicted_advantage': float(decision.reference_score - decision.predicted_score)}
                decisions.append(record)
                if record['time_sec'] != TARGET_TIME:
                    return decision
                capture['decision'] = record
                if arm == 'reference':
                    return decision
                phase = record['reference_phase'] if arm == 'phase_pressure' else record['selected_phase']
                rule_phase = v3._rule_candidate(capture['states'][TLS], capture['executors'][TLS], 'phase_pressure').state
                branch.update(first_phase_request=phase,
                    pp_reference_matches=rule_phase == record['reference_phase'])
                if (not branch['pp_reference_matches'] or len(capture['lanes']) != 8
                        or list(capture['states']) != [TLS]):
                    raise BranchComplete(HANDOFF_MESSAGE)
                ledger = v3._new_safety_ledger()
                try:
                    outcome = v3._counterfactual_branch_outcome_v3(
                        sumo_api=api, infos=capture['infos'], states=capture['states'],
                        executors=capture['executors'], focal_id=TLS,
                        candidate=v3._candidate_for_state(capture['infos'][TLS], phase),
                        context=capture['context'], control_interval_sec=10,
                        counterfactual_horizon_intervals=45, controlled_lane_count=8,
                        controlled_lanes=capture['lanes'], counterfactual_cost_mode='halted_queue', safety_ledger=ledger)
                except v3.UnsafeRolloutError as error:
                    branch.update(status='FAIL_UNSAFE', error=str(error), safety_audit=error.safety_audit)
                else:
                    costs, one_step, terminal = outcome
                    branch.update(status='PASS', costs=costs, mean_halted_per_lane=float(np.mean(costs)),
                        one_step=one_step, terminal=terminal, safety_audit=v3._finalize_safety_ledger(ledger))
                    vectors[arm] = v3._counterfactual_outcome_vector(*outcome)
                branch['stop_time_sec'] = float(api.simulation.getTime())
                raise BranchComplete(HANDOFF_MESSAGE)

        def observer(*, stage, time_sec, executors):
            if stage != 'after_executor_advance':
                return
            if time_sec <= TARGET_TIME:
                v3._record_safety_step(ledger=prefix_ledger, sumo_api=api, executors=executors,
                    starting_teleports=int(api.simulation.getStartingTeleportNumber()),
                    ending_teleports=int(api.simulation.getEndingTeleportNumber()),
                    collisions=tuple(api.simulation.getCollisions()))
            elif arm == 'reference' and time_sec <= TARGET_TIME + 10:
                first_costs.append(float(sum(api.lane.getLastStepHaltingNumber(lane) for lane in capture['lanes'])) / 8)

        originator = LiveOriginator.load(Path(frozen['model_path']), frozen_threshold_path=frozen_path,
            expected_city='cologne', network_context=runtime.read_network_right_of_way_context(runtime.net_file_from_sumocfg(sumocfg)))
        metrics = runtime.evaluate_policy_v3(sumo_api=api, sumocfg=sumocfg, scenario='cologne1', policy=RUNTIME_POLICY,
            models=_runtime_models(originator, prediction_horizon_sec=450), duration_sec=610,
            control_interval_sec=10, warmup_sec=60, seed=SEED, tripinfo_output=scratch / f'{arm}.xml',
            residual_coordination_mode='direct', residual_cooldown_intervals_override=0, step_observer=observer)
        captures[arm] = capture
        branch.update(prefix_decisions=decisions, prefix_safety=v3._finalize_safety_ledger(prefix_ledger))
        if arm == 'reference':
            branch.update(status='PASS' if metrics.get('ok') else 'INVALID', metrics=metrics,
                          native_first_interval_costs=first_costs)
        else:
            branch['controlled_handoff_completed'] = metrics.get('error') == f'BranchComplete: {HANDOFF_MESSAGE}'
            if not branch['controlled_handoff_completed']:
                branch['runtime_error'] = metrics.get('error')
                branch['status'] = 'INVALID'
        runs[arm] = branch
        print(json.dumps({'arm': arm, 'status': branch['status'], 'stop_time_sec': branch.get('stop_time_sec')}), flush=True)

    normal = runs['reference']
    normal_capture = captures['reference']
    expected_trace = [row for row in trace if row['time_sec'] < TARGET_TIME + 10]
    checks = {'reference_completed': normal['status'] == 'PASS',
        'reference_trace_exact': normal['metrics'].get('accepted_intervention_trace') == expected_trace,
        'reference_first_interval_complete': len(normal['native_first_interval_costs']) == 10,
        'reference_zero_collisions': normal['metrics'].get('collision_events') == 0,
        'reference_zero_teleports': normal['metrics'].get('starting_teleports') == 0 and normal['metrics'].get('ending_teleports') == 0}
    comparisons = {}
    for arm, branch in runs.items():
        if arm not in captures or 'decision' not in captures[arm]:
            checks[f'{arm}_target_reached'] = False
            continue
        capture = captures[arm]
        decision = capture['decision']
        safety = branch['prefix_safety']
        checks[f'{arm}_prefix_safe'] = safety['raw_collision_events'] == 0 and safety['no_teleport_passed']
        checks[f'{arm}_selected_decision_exact'] = (
            decision['selected_phase'] == target['selected_phase_state']
            and decision['reference_phase'] == target['prior_phase_state']
            and decision['predicted_advantage'] == target['priority'])
        if arm == 'reference':
            continue
        comparisons[arm] = compare_physical(normal_capture['physical'], capture['physical'], tolerance=0)
        checks[f'{arm}_physical_start_exact'] = comparisons[arm]['passed']
        checks[f'{arm}_pending_start_exact'] = capture['physical']['pending_raw'] == normal_capture['physical']['pending_raw']
        checks[f'{arm}_decision_prefix_exact'] = branch['prefix_decisions'] == normal['prefix_decisions']
        checks[f'{arm}_pp_reference_exact'] = branch.get('pp_reference_matches') is True
        checks[f'{arm}_full_horizon'] = len(branch.get('costs', ())) == 450 and branch.get('stop_time_sec') == TARGET_TIME + 450
        checks[f'{arm}_controlled_handoff'] = branch.get('controlled_handoff_completed') is True
    result = {'protocol': PROTOCOL, 'status': 'INVALID', 'source_root': protocol['source_root'],
        'sumocfg': str(sumocfg), 'frozen_threshold': frozen, 'seed': SEED, 'target_time_sec': TARGET_TIME,
        'selected_trace': target, 'label_contract': protocol['label_contract'], 'checks': checks,
        'runs': runs, 'physical_start_comparisons': comparisons,
        'previous_invalid_result': protocol['previous_invalid_result']}
    if 'rigid' in vectors:
        result['first_interval_replay'] = compare_vectors(normal['native_first_interval_costs'], runs['rigid']['costs'][:10], tolerance=0)
        checks['first_interval_exact'] = result['first_interval_replay']['passed']
    if 'rigid' in vectors and 'rigid_replay' in vectors:
        result['same_action_replay'] = compare_vectors(vectors['rigid'], vectors['rigid_replay'], tolerance=0)
        checks['same_action_replay_exact'] = result['same_action_replay']['passed']
    if all(checks.values()) and all(row['status'] == 'PASS' for row in runs.values()):
        rigid, pp = (runs[arm]['mean_halted_per_lane'] for arm in ('rigid', 'phase_pressure'))
        result.update(status='PASS', comparison={'rigid_cost': rigid, 'phase_pressure_cost': pp,
            'rigid_minus_pp': rigid - pp, 'actual_advantage_pp_minus_rigid': pp - rigid,
            'relative_cost_change_pct': (rigid / pp - 1) * 100 if pp else None,
            'ranking': 'harmful' if rigid > pp + 1e-12 else 'beneficial' if rigid < pp - 1e-12 else 'tie'})
    elif any(row['status'] == 'FAIL_UNSAFE' for row in runs.values()):
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
    if run(json.loads(args.protocol.read_text()), args.output)['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
