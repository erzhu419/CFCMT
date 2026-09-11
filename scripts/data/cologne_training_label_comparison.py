"""Compare the eight stored and native action costs of a Cologne training group."""

import math
import operator
from numbers import Real


def _costs(values, name):
    try:
        values = list(values)
    except TypeError as error:
        raise ValueError(f"{name} must contain eight finite scalar costs") from error
    if len(values) != 8 or not all(isinstance(value, Real) and math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain eight finite scalar costs")
    return [float(value) for value in values]


def compare_labels(stored_costs, native_costs, reference_index, tolerance=1e-12):
    """Return cost and action-ranking changes, treating values within tolerance as ties.

    Advantages are PP-reference cost minus action cost, so positive means better
    than PP. Cost deltas are native minus stored. A strict sign flip excludes
    changes to or from a tie; the separate sign-change list includes those.
    """
    stored = _costs(stored_costs, "stored_costs")
    native = _costs(native_costs, "native_costs")
    try:
        reference_index = operator.index(reference_index)
    except TypeError as error:
        raise ValueError("reference_index must be an integer from 0 to 7") from error
    if not 0 <= reference_index < 8:
        raise ValueError("reference_index must be an integer from 0 to 7")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")

    deltas = [new - old for old, new in zip(stored, native)]
    errors = [abs(delta) for delta in deltas]
    stored_advantages = [stored[reference_index] - cost for cost in stored]
    native_advantages = [native[reference_index] - cost for cost in native]

    def sign(value):
        return int(value > tolerance) - int(value < -tolerance)

    sign_pairs = list(zip(map(sign, stored_advantages), map(sign, native_advantages)))
    stored_min, native_min = min(stored), min(native)
    stored_optimal = [i for i, cost in enumerate(stored) if cost - stored_min <= tolerance]
    native_optimal = [i for i, cost in enumerate(native) if cost - native_min <= tolerance]
    return {
        "mean_absolute_cost_error": math.fsum(errors) / 8,
        "max_absolute_cost_error": max(errors),
        "cost_deltas": deltas,
        "stored_advantages": stored_advantages,
        "native_advantages": native_advantages,
        "strict_sign_flip_indices": [i for i, (old, new) in enumerate(sign_pairs) if old * new == -1],
        "any_advantage_sign_change_indices": [i for i, (old, new) in enumerate(sign_pairs) if old != new],
        "stored_optimal_indices": stored_optimal,
        "native_optimal_indices": native_optimal,
        "optimal_set_changed": stored_optimal != native_optimal,
        "disjoint_optimal_sets": set(stored_optimal).isdisjoint(native_optimal),
        "any_cost_changed": max(errors) > tolerance,
    }
