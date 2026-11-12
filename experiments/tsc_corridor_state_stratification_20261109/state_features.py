"""Recover current protected service and phase aggregate queue from stored rows."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tsc_corridor_service_alignment_20261109'))
from service_window import extend_window


def recover_state_features(row, previous_request):
    """Return one row's identifiable 2x2 state, without changing input or history.

    The queue is the halted count across all distinct G/g inlet lanes covered
    by the current phase. The records do not contain a current per-flight lane
    queue. Locked rows may be clearing toward states[0], so remain unknown.
    The caller advances its previous-PP request for every row, including rows
    without observable flights.
    """
    unlocked = len(row['states']) > 1
    unknown = dict(monitored_straight_protected=None, queue_positive=None,
                   phase_relation='unknown', queue_bin='unknown',
                   stratum=['unknown', 'unknown'])
    result = dict(unlocked=unlocked, identified=False, reason='locked',
                  current_request_index=None, current_request=None,
                  current_phase_aggregate_queue=None,
                  by_edge={edge: dict(unknown) for edge in row['protected']})
    if not unlocked:
        return result
    try:
        aligned = extend_window(row, previous_request)
    except ValueError as exc:
        result['reason'] = str(exc)
        return result
    metadata = aligned['service_window']
    current = metadata['current_request_index']
    queue = metadata['queue_demand'][current]
    result.update(identified=True, reason='identified', current_request_index=current,
                  current_request=metadata['current_request'],
                  current_phase_aggregate_queue=queue)
    for edge, masks in row['protected'].items():
        protected = bool(int(masks[current]) & 1)
        phase_relation = 'protected' if protected else 'not_protected'
        queue_bin = 'positive' if queue else 'zero'
        result['by_edge'][edge] = dict(monitored_straight_protected=protected,
            queue_positive=bool(queue), phase_relation=phase_relation, queue_bin=queue_bin,
            stratum=[phase_relation, queue_bin])
    return result
