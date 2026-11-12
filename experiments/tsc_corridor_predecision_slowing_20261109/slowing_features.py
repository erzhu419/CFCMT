"""Causal changes in two consecutive ten-second straight-lane progress rates."""


def _identity(flight):
    return flight['edge'], flight['vid'], flight['entry_s']


def causal_slowing(flight, time_s, past20_row, past10_row):
    """Keep unobserved history unknown; compare net rates with a fixed zero cut.

    The caller supplies complete rows from this same TLS. Historical flight
    lists contain moving vehicles only. Missing entries provide no position or
    zero-speed observation. Even comparable endpoint progress rates describe
    intervals, rather than instantaneous acceleration.
    """
    history = ((20, past20_row), (10, past10_row))
    if any(row is not None and row['t'] > time_s for _, row in history):
        raise ValueError('Predecision slowing cannot use a future row')
    result = dict(group='unknown', reason=None, old_progress_rate=None,
                  recent_progress_rate=None, delta=None,
                  current_remaining=float(flight['remaining_m']),
                  history20_available=False, history10_available=False)
    comparable, reasons = {}, []
    if not flight['on_straight_lane']:
        reasons.append('nonstraight_current')
    for lag, row in history:
        if row is None:
            reasons.append(f'missing_past{lag}_row')
        elif row['t'] != time_s-lag:
            reasons.append(f'past{lag}_wrong_time')
        elif flight['entry_s'] > row['t']:
            reasons.append(f'not_yet_entered_at{lag}')
        else:
            previous = next((item for item in row['flights'] if _identity(item) == _identity(flight)), None)
            if previous is None:
                reasons.append(f'not_listed_moving_at{lag}')
            elif not previous['on_straight_lane']:
                reasons.append(f'nonstraight_past{lag}')
            else:
                comparable[lag] = previous
                result[f'history{lag}_available'] = True
    if 20 in comparable and 10 in comparable:
        result['old_progress_rate'] = (comparable[20]['remaining_m']-comparable[10]['remaining_m'])/10.
    if 10 in comparable and flight['on_straight_lane']:
        result['recent_progress_rate'] = (comparable[10]['remaining_m']-flight['remaining_m'])/10.
    if result['old_progress_rate'] is not None and result['recent_progress_rate'] is not None:
        result['delta'] = result['recent_progress_rate']-result['old_progress_rate']
        result['group'] = 'slowing' if result['delta'] < 0 else 'not_slowing'
        result['reason'] = 'available'
    else:
        result['reason'] = reasons[0]
    return result
