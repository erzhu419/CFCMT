"""Fixed arrival baselines from current position and the preceding 10 s snapshot.

The collector records only moving flights, so a missing preceding observation
does not mean zero speed.  Call for every stored row, in time order, with a fresh
history dict per hour; select evaluation rows only after updating this history.
"""

from math import isfinite


def release_id(flight):
    return flight["edge"], flight["vid"], flight["entry_s"]


def historical_speed_mps(flight, time_s, history):
    """Return the past 10 s progress speed, or None when it is not observable.

    Endpoint lane classification is the stored lane information.  Each monitored
    edge has one dedicated straight lane; requiring both endpoints on that lane
    avoids comparing positions across an observed change of lane class.  Lane
    changes between the two observations cannot be reconstructed.
    """
    previous = history.get(release_id(flight))
    if previous is None or previous[0] != time_s - 10:
        return None
    if not (previous[2] and flight["on_straight_lane"]):
        return None
    speed = (previous[1] - flight["remaining_m"]) / 10.0
    return speed if isfinite(speed) and speed > 0 else None


def causal_predictions(row, history):
    """Return edge cumulative counts for seconds 1..10, then store this row.

    ``analytic`` uses current remaining distance / posted speed. ``history``
    uses past 10 s progress speed when available and otherwise the same analytic
    ETA. Both count only vehicles currently on the dedicated straight lane.
    They predict the detector crossing, without adding detector-to-stop-line
    travel time. No outcome records or future snapshots enter either baseline.
    """
    predictions = {"analytic": {}, "history": {}}
    for flight in row["flights"]:
        edge = flight["edge"]
        for by_edge in predictions.values():
            by_edge.setdefault(edge, [0] * 10)
        if not flight["on_straight_lane"]:
            continue
        analytic_eta = flight["analytic_eta_s"]
        speed = historical_speed_mps(flight, row["t"], history)
        history_eta = flight["remaining_m"] / speed if speed is not None else analytic_eta
        for name, eta in (("analytic", analytic_eta), ("history", history_eta)):
            for second in range(1, 11):
                predictions[name][edge][second - 1] += int(eta <= second)

    # Defer all writes until every prediction in the current row has been read.
    for flight in row["flights"]:
        history[release_id(flight)] = (
            row["t"], flight["remaining_m"], flight["on_straight_lane"]
        )
    return predictions
