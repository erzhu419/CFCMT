"""Normalize an already observed slowing rate by its actual posted speed limit."""
from math import isfinite


def normalized_magnitude(flight, slowing):
    """Return dimensionless (old-recent)/limit only for the fixed slowing group.

    The supplied rates and delta retain their original m/s units. Historical
    snapshots, outcome labels and magnitude thresholds are not inputs here.
    """
    result = dict(value=None, group=slowing['group'], reason=slowing['reason'],
                  old_progress_rate=slowing['old_progress_rate'],
                  recent_progress_rate=slowing['recent_progress_rate'],
                  delta=slowing['delta'], speed_limit_mps=flight['speed_limit_mps'])
    if result['group'] != 'slowing':
        return result
    limit = float(flight['speed_limit_mps'])
    if not isfinite(limit) or limit <= 0:
        raise ValueError('Normalized slowing requires the actual finite positive speed limit')
    result['value'] = (result['old_progress_rate']-result['recent_progress_rate'])/limit
    return result
