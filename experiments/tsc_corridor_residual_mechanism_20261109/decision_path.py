"""Explain frozen S-versus-T forecasts through the existing legal phase scores."""
from collections import Counter

import numpy as np


T, S = 0, 1


def phase_scores(row, forecasts):
    """Return queue base plus the original protected-window arrival credits."""
    count = next(iter(forecasts.values())).shape[0] if forecasts else 5
    scores = np.broadcast_to(row["base"], (count, len(row["base"]))).astype(float).copy()
    for edge, masses in forecasts.items():
        for request, mask in enumerate(row["protected"][edge]):
            for bit in range(10):
                if int(mask) & (1 << bit):
                    scores[:, request] += masses[:, bit]
    return np.round(scores, 12)


def _changed(values):
    return bool(np.any(np.abs(values) > 1e-12))


def _flight_flags(details):
    counts = Counter()
    for detail in details:
        raw = np.asarray(detail["raw_cdfs"])
        decoded = raw[:3] >= .5
        delta = decoded[S].astype(int) - decoded[T].astype(int)
        raw_change, service, endpoint = _changed(raw[S] - raw[T]), _changed(delta[:10]), bool(delta[10])
        decoded_change = _changed(delta)
        detector_change = endpoint
        if "detector_cdfs" in detail:
            detector_raw = np.asarray(detail["detector_cdfs"])[:3]
            detector = detector_raw >= .5
            changed = np.flatnonzero(detector[S] != detector[T])
            raw_change |= _changed(detector_raw[S] - detector_raw[T])
            detector_change |= bool(len(changed))
            decoded_change |= bool(len(changed))
            if len(changed):
                counts[f"first_detector_change_second_{int(changed[0])+1}_flights"] += 1
                counts["detector_changes_only_seconds8_to10_flights"] += int(changed[0]) >= 7
        counts["detailed_flights"] += 1
        counts["raw_cdf_changed_flights"] += raw_change
        counts["raw_changed_but_decode_unchanged_flights"] += raw_change and not decoded_change
        counts["decoded_changed_flights"] += decoded_change
        counts["decoded_detector_curve_changed_flights"] += detector_change
        counts["detector_change_unseen_at_endpoint_or_service_flights"] += detector_change and not endpoint and not service
        counts["endpoint_changed_flights"] += endpoint
        counts["service_changed_flights"] += service
        counts["endpoint_only_flights"] += endpoint and not service
        counts["service_only_flights"] += service and not endpoint
        if "label_category" in detail:
            actual = detail["label_category"] != 0
            loss_delta = abs(int(decoded[S, 10]) - actual) - abs(int(decoded[T, 10]) - actual)
            counts["individual_endpoint_improved_flights"] += loss_delta < 0
            counts["individual_endpoint_worsened_flights"] += loss_delta > 0
            counts["endpoint_improved_but_no_service_change_flights"] += loss_delta < 0 and not service
            counts["endpoint_worsened_but_no_service_change_flights"] += loss_delta > 0 and not service
    return counts


def explain_row(row, forecasts, flight_details=()):
    """No simulator or fitted model access: all input predictions are already fixed."""
    flights = _flight_flags(flight_details)
    endpoint = any(_changed(m[S, 10] - m[T, 10]) for m in forecasts.values())
    service = any(_changed(m[S, :10] - m[T, :10]) for m in forecasts.values())
    protected = False
    for edge, masses in forecasts.items():
        changed_bits = np.flatnonzero(np.abs(masses[S, :10] - masses[T, :10]) > 1e-12)
        protected |= any(any(int(mask) & (1 << int(bit)) for mask in row["protected"][edge]) for bit in changed_bits)
    scores = phase_scores(row, forecasts)
    target, source = scores[T], scores[S]
    winner, source_winner = int(np.argmax(target)), int(np.argmax(source))
    delta = source - target
    score_change = _changed(delta)
    relative_change = _changed(delta - delta[winner])
    unlocked = len(target) > 1
    flags = dict(
        raw_cdf_changed=flights["raw_cdf_changed_flights"] > 0,
        decoded_flight_changed=flights["decoded_changed_flights"] > 0,
        decoded_detector_curve_changed=flights["decoded_detector_curve_changed_flights"] > 0 or endpoint,
        endpoint_edge_count_changed=endpoint, service_edge_curve_changed=service,
        changed_service_has_any_protected_slot=protected,
        legal_scores_changed=score_change, relative_scores_changed=relative_change,
        request_changed=winner != source_winner,
    )
    rivals = [i for i in range(len(target)) if i != winner]
    base = np.asarray(row["base"], dtype=float)
    comparisons = [dict(
        request=i, base_margin=float(base[winner] - base[i]),
        target_arrival_margin=float((target[winner]-base[winner])-(target[i]-base[i])),
        target_total_margin=float(target[winner]-target[i]),
        source_differential_gain=float(delta[i]-delta[winner]),
        source_margin_at_target_winner=float(source[winner]-source[i]),
    ) for i in rivals]
    metrics = {}
    if rivals:
        metrics = dict(target_total_margin=min(c["target_total_margin"] for c in comparisons),
                       source_margin_at_target_winner=min(c["source_margin_at_target_winner"] for c in comparisons),
                       score_delta_range=float(np.ptp(delta)),
                       max_challenger_relative_gain=max(c["source_differential_gain"] for c in comparisons),
                       base_range=float(np.ptp(base)))
        flags["target_margin_exceeds_all_source_perturbations"] = metrics["target_total_margin"] > metrics["score_delta_range"]
        flags["target_matches_queue_only_request"] = winner == int(np.argmax(base))
        flags["source_matches_queue_only_request"] = source_winner == int(np.argmax(base))
        flags["relative_change_preserves_request"] = relative_change and winner == source_winner
        flags["relative_change_blocked_by_margin_bound"] = (relative_change and winner == source_winner
            and metrics["target_total_margin"] > metrics["score_delta_range"])
        # A positive queue lead covers either head's adverse arrival differential.
        # Restrict this diagnostic to actual relative changes, rather than count
        # unchanged forecasts as evidence of a decision bottleneck.
        flags["queue_lead_covers_both_arrival_disadvantages"] = (relative_change and winner == source_winner
            and all(c["base_margin"] > max(0.0, -c["target_arrival_margin"],
                    c["source_differential_gain"] - c["target_arrival_margin"]) for c in comparisons))
        if relative_change:
            for key in ("target_total_margin", "source_margin_at_target_winner", "score_delta_range",
                        "max_challenger_relative_gain"):
                metrics["relative_change_rows_" + key] = metrics[key]
    target_tied = int(np.count_nonzero(target == target.max())) > 1
    source_tied = int(np.count_nonzero(source == source.max())) > 1
    if not unlocked:
        reason = "locked"
    elif winner != source_winner:
        reason = "source_tie_first_order_changes_request" if source_tied else "source_breaks_target_tie" if target_tied else "strict_score_crossing"
    elif not service:
        reason = ("vehicle_service_changes_cancel_on_edge" if flights["service_changed_flights"]
                  else "decoded_detector_change_outside_service_window" if endpoint or flights["decoded_detector_curve_changed_flights"]
                  else "threshold_hides_raw_changes" if flights["raw_cdf_changed_flights"] and not flights["decoded_changed_flights"]
                  else "no_decoded_service_change")
    elif not score_change:
        reason = "credited_changes_cancel_in_scores" if protected else "changed_slots_have_no_protected_service"
    elif not relative_change:
        reason = "same_shift_all_requests"
    elif target_tied or source_tied:
        reason = "tie_order_preserves_request"
    else:
        reason = "relative_change_below_margin"
    case = dict(t=row["t"], tls=row["tls"], reason=reason, target_request=winner, source_request=source_winner,
                base=list(map(float, base)), target_scores=list(map(float, target)), source_scores=list(map(float, source)),
                target_arrival_credits=list(map(float, target-base)), source_arrival_credits=list(map(float, source-base)),
                rivals=comparisons, flags=flags)
    case.update({key: row[key] for key in ("city", "seed", "block") if key in row})
    return dict(flags=flags, flight_counts=dict(flights), reason=reason, metrics=metrics, case=case)


def _cell():
    return dict(counts=Counter(), reasons=Counter(), metrics={}, samples={})


def _metric_add(table, name, value):
    if name not in table:
        table[name] = dict(n=0, sum=0.0, min=value, max=value)
    item = table[name]
    item["n"] += 1
    item["sum"] += value
    item["min"], item["max"] = min(item["min"], value), max(item["max"], value)


class DecisionPathObserver:
    """Accumulate all rows, then explain only each scope's enabled risk-set rows.

    ``flight_details`` elements have ``raw_cdfs`` (T,S,P,... x 11), optional
    ``label_category`` (0=no next10s straight arrival), and optional
    ``detector_cdfs`` (same arms x detector seconds1..10). These must be computed
    before updating model history. Labels affect descriptive error counts only.
    """
    def __init__(self, sample_limit=2):
        self.scopes = {}
        self.sample_limit = sample_limit

    def observe(self, row, forecasts, flight_details=(), *, enabled):
        mode = "unlocked" if len(row["states"]) > 1 else "locked"
        explained = explain_row(row, forecasts, flight_details) if any(enabled.values()) else None
        for scope, active in enabled.items():
            cells = self.scopes.setdefault(scope, {name: _cell() for name in ("all", "locked", "unlocked")})
            for name in ("all", mode):
                cell = cells[name]
                cell["counts"]["decisions"] += 1
                cell["counts"]["evaluated_decisions"] += active
                cell["counts"]["fallback_decisions"] += not active
                if not active:
                    continue
                cell["counts"].update({key: int(value) for key, value in explained["flags"].items()})
                cell["counts"].update(explained["flight_counts"])
                cell["reasons"][explained["reason"]] += 1
                for key, value in explained["metrics"].items():
                    _metric_add(cell["metrics"], key, value)
                # Only retain examples that distinguish the real bottlenecks.
                if explained["reason"] not in ("locked", "no_decoded_service_change", "threshold_hides_raw_changes"):
                    examples = cell["samples"].setdefault(explained["reason"], [])
                    if len(examples) < self.sample_limit:
                        examples.append(explained["case"])
        return explained

    def merge(self, other):
        for scope, cells in other.scopes.items():
            target = self.scopes.setdefault(scope, {name: _cell() for name in ("all", "locked", "unlocked")})
            for name, cell in cells.items():
                dest = target[name]
                dest["counts"].update(cell["counts"])
                dest["reasons"].update(cell["reasons"])
                for key, stat in cell["metrics"].items():
                    if key not in dest["metrics"]:
                        dest["metrics"][key] = stat.copy()
                    else:
                        value = dest["metrics"][key]
                        value["n"] += stat["n"]
                        value["sum"] += stat["sum"]
                        value["min"], value["max"] = min(value["min"], stat["min"]), max(value["max"], stat["max"])
                for reason, cases in cell["samples"].items():
                    samples = dest["samples"].setdefault(reason, [])
                    samples.extend(cases[:max(0, self.sample_limit-len(samples))])

    def result(self):
        out = {}
        for scope, cells in self.scopes.items():
            out[scope] = {}
            for name, cell in cells.items():
                metrics = {key: dict(value, mean=value["sum"]/value["n"]) for key, value in cell["metrics"].items()}
                out[scope][name] = dict(counts=dict(cell["counts"]), reasons=dict(cell["reasons"]),
                                       metrics=metrics, samples=cell["samples"])
        return out
