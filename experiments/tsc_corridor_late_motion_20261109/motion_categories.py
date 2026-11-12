"""Describe late straight arrivals at the original t+10 observation snapshot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tsc_corridor_outcome_decomposition_20261109'))
from outcome_categories import classify_outcome
from residual_model import release_id

MOTION_CLASSES = ('stopped_at10', 'moving_straight_at10', 'moving_other_lane_at10')
OUTCOME_CLASSES = ('straight_by10', *MOTION_CLASSES, 'other_by10', 'other_after10',
                   'terminal_by10', 'terminal_after10', 'censored')


def classify_late_motion(flight, outcome, time_s, future_row):
    """Classify a known straight-after10 release using its complete next row.

    The caller supplies the same TLS's original row at t+10, including rows
    flagged spillback or unmatched. A release with later straight detection is
    still active at t+10; absence from this row's moving-flight list therefore
    means instantaneous speed <0.1 m/s under ArrivalSensor.sample. This says
    nothing about how long it stopped during the interval. Progress is a net
    endpoint displacement only when both observations use the monitored straight
    lane; positions and progress are unavailable for a missing stopped flight.
    """
    _, outcome_class = classify_outcome(flight, outcome, time_s)
    if outcome_class != 'straight_after10':
        raise ValueError('Late motion requires an observed straight arrival after t+10')
    if future_row['t'] != time_s+10:
        raise ValueError('Late motion requires the complete original row at t+10')
    future = next((item for item in future_row['flights']
                   if release_id(item) == release_id(flight)), None)
    motion_class = ('stopped_at10' if future is None else 'moving_straight_at10'
                    if future['on_straight_lane'] else 'moving_other_lane_at10')
    progress = None
    if future is not None and flight['on_straight_lane'] and future['on_straight_lane']:
        progress = float(flight['remaining_m']-future['remaining_m'])
    return dict(motion_class=motion_class, progress_m=progress,
                realized_10s_mean_speed=progress/10 if progress is not None else None,
                initial_remaining_m=float(flight['remaining_m']),
                future_remaining_m=float(future['remaining_m']) if future is not None else None)
