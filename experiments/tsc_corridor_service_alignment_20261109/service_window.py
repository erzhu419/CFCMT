"""Extend a stored nominal request hold from ten to thirteen service seconds.

Only an unlocked row is identifiable here. Its green state equals this TLS's
preceding PP request; all intervening activity was executor clearance/holding.
The extension retains the collector's no-future-occupancy-extension assumption.
"""

OBSERVED_SECONDS = 10
FULL_MASK = (1 << OBSERVED_SECONDS) - 1


def _suffix_start(mask):
    """A switched protected inlet has clearance zeros followed by stable ones."""
    first = (mask & -mask).bit_length() - 1
    if mask != FULL_MASK ^ ((1 << first) - 1):
        raise ValueError('Stored protection is not a clearance prefix and stable green suffix')
    return first


def extend_window(row, previous_request, horizon=13):
    """Return an extended row and exact reconstruction metadata; never drop rows.

    Requests remain in their original feasible ordering. Queue service rates are
    recovered from the original base and inferred nominal green exposure, without
    inventing missing per-lane queues or reading future traffic observations.
    """
    states, base, masks = row['states'], row['base'], row['protected']
    if len(states) <= 1:
        raise ValueError('Only unlocked decisions require a reconstructed choice window')
    if previous_request not in states:
        raise ValueError('The preceding PP request does not identify the current green')
    if horizon < OBSERVED_SECONDS:
        raise ValueError('The service window must extend the observed ten seconds')
    if len(base) != len(states):
        raise ValueError('Stored queue bases do not match the feasible requests')
    current = states.index(previous_request)
    starts = set()
    for edge, values in masks.items():
        if len(values) != len(states):
            raise ValueError(f'Protection mask count differs from requests for {edge}')
        for index, value in enumerate(values):
            mask = int(value)
            if not 0 <= mask <= FULL_MASK:
                raise ValueError('Protection mask extends beyond the stored ten seconds')
            if index == current:
                if mask not in (0, FULL_MASK):
                    raise ValueError('The recovered current request does not retain its stable green')
            elif mask:
                starts.add(_suffix_start(mask))
    if len(starts) != 1:
        raise ValueError('A common nominal clearance below ten seconds cannot be identified')
    clearance, = starts
    green10 = [OBSERVED_SECONDS if index == current else OBSERVED_SECONDS-clearance
               for index in range(len(states))]
    green_extended = [horizon if index == current else horizon-clearance
                      for index in range(len(states))]
    queues, restored = [], []
    for value, green in zip(base, green10):
        queue = round(float(value) / green)
        if queue < 0 or abs(float(value)-queue*green) > 1e-9:
            raise ValueError('Stored base cannot recover a nonnegative integer queue service rate')
        queues.append(queue)
        restored.append(queue*green)
    extra = ((1 << (horizon-OBSERVED_SECONDS))-1) << OBSERVED_SECONDS
    extended_masks = {edge: [int(value) | (extra if int(value) & (1 << 9) else 0)
                             for value in values] for edge, values in masks.items()}
    if any((new & FULL_MASK) != int(old)
           for edge, values in masks.items()
           for old, new in zip(values, extended_masks[edge])):
        raise ValueError('Extending the service mask changed its observed prefix')
    metadata = dict(observed_seconds=OBSERVED_SECONDS, service_seconds=horizon,
        current_request_index=current, current_request=previous_request, clearance_sec=clearance,
        green_seconds_10=green10, green_seconds_extended=green_extended,
        queue_demand=queues, base10_reconstructed=restored, mask_prefix_verified=True,
        request_count=len(states), edge_count=len(masks),
        assumption='Hold the selected request; no future occupancy clearance extensions.')
    return dict(row, base=[queue*green for queue, green in zip(queues, green_extended)],
                protected=extended_masks, service_window=metadata)
