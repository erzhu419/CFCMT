"""Frozen numerical gates for the proposed independent target calibration.

The eventual full engine must separately establish identity/executor/geometry
closure on every scenario. This module evaluates prediction accuracy only.
"""


def _window_metrics(window, *, scenarios, horizon, cost_lanes):
    true = window['actual']
    draws = window['predicted_scenarios']
    if len(draws) != scenarios:
        raise ValueError('Use every frozen model scenario')
    for row in [true, *draws]:
        if any(len(row[name]) != horizon+1 for name in ('halted_total', 'active_count')):
            raise ValueError('Retain frame0 and every poststep')
        if any(value < 0 for name in ('halted_total', 'active_count') for value in row[name]):
            raise ValueError('Population and halting counts are nonnegative')
    cost = sum(true['halted_total'][1:])/(horizon*cost_lanes)
    predicted_cost = sum(sum(row['halted_total'][1:]) for row in draws)/(scenarios*horizon*cost_lanes)
    expected_active = [sum(row['active_count'][step] for row in draws)/scenarios
                       for step in range(1, horizon+1)]
    observed_active = true['active_count'][1:]
    normalizer = sum(observed_active)
    active_error = sum(abs(a-b) for a, b in zip(expected_active, observed_active))
    relative_active_error = active_error/normalizer if normalizer else (0. if active_error == 0 else float('inf'))
    exit_mean = sum(row['terminal_count'] for row in draws)/scenarios
    nonsink_mean = sum(row['nonsink_terminal_count'] for row in draws)/scenarios
    return dict(seed=window['seed'], anchor_sec=window['anchor_sec'],
                actual_C450=cost, predicted_C450=predicted_cost,
                absolute_cost_error=abs(predicted_cost-cost),
                relative_active_error=relative_active_error,
                final_active_error=expected_active[-1]-observed_active[-1],
                terminal_error=abs(exit_mean-true['terminal_count']),
                terminal_tolerance=max(5., .10*true['terminal_count']),
                nonsink_terminal_error=abs(nonsink_mean-true['nonsink_terminal_count']))


def evaluate_calibration(windows, protocol):
    evaluation = protocol['calibration']
    expected = {(seed, anchor) for seed in protocol['cohorts']['calibration_seeds']
                for anchor in evaluation['anchors_sec']}
    actual = [(row['seed'], row['anchor_sec']) for row in windows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('All frozen seed/anchor windows must be retained exactly once')
    metrics = [_window_metrics(row, scenarios=evaluation['model_scenarios'],
                               horizon=evaluation['horizon_sec'], cost_lanes=protocol['cost']['lane_count'])
               for row in windows]
    limits = evaluation['accuracy_gates']
    per_seed = {str(seed): sum(row['absolute_cost_error'] for row in metrics if row['seed'] == seed)
                /len(evaluation['anchors_sec']) for seed in protocol['cohorts']['calibration_seeds']}
    checks = dict(
        every_seed_cost_MAE=all(value <= limits['per_seed_cost_MAE_max'] for value in per_seed.values()),
        every_window_cost_error=all(row['absolute_cost_error'] <= limits['window_cost_absolute_error_max'] for row in metrics),
        every_window_population_error=all(row['relative_active_error'] <= limits['window_relative_active_error_max'] for row in metrics),
        every_window_terminal_error=all(row['terminal_error'] <= row['terminal_tolerance'] for row in metrics),
        every_window_nonsink_terminal_error=all(row['nonsink_terminal_error'] <= limits['window_nonsink_terminal_error_max'] for row in metrics),
    )
    return dict(status='PASS_NUMERICAL_CALIBRATION' if all(checks.values()) else 'HOLD_CALIBRATION',
                checks=checks, per_seed_cost_MAE=per_seed, windows=metrics,
                scope='Numerical gates only; full admission also requires every scenario identity, geometry and executor audit. No action-ranking/control-gain test.')
