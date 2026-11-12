"""Real decision-path bottlenecks without simulation or another candidate policy."""
import numpy as np

from experiments.tsc_corridor_residual_mechanism_20261109.decision_path import (
    DecisionPathObserver, explain_row,
)


def row(base=(0, 0), masks=(0, 1023)):
    return dict(t=100, tls="signal", states=[str(i) for i in range(len(base))],
                base=list(base), protected={"edge": list(masks)})


def forecast(target=None, source=None):
    value = np.zeros((5, 11))
    if target is not None:
        value[0] = target
    if source is not None:
        value[1] = source
    return {"edge": value}


def test_endpoint_correction_after_tail_does_not_enter_the_service_score():
    f = forecast(source=[0]*10+[1])
    detail = dict(raw_cdfs=f["edge"].copy(), label_category=10,
                  detector_cdfs=np.array([[0]*10, [0]*7+[1]*3, [0]*10, [0]*10, [0]*10]))
    result = explain_row(row(), f, [detail])
    assert result["reason"] == "decoded_detector_change_outside_service_window"
    assert result["flags"]["endpoint_edge_count_changed"]
    assert not result["flags"]["legal_scores_changed"]
    assert result["flight_counts"]["endpoint_improved_but_no_service_change_flights"] == 1
    assert result["flight_counts"]["detector_changes_only_seconds8_to10_flights"] == 1


def test_raw_probability_difference_can_be_hidden_by_the_fixed_half_threshold():
    raw = np.zeros((5, 11))
    raw[0] = .2
    raw[1] = .4
    result = explain_row(row(), forecast(), [dict(raw_cdfs=raw)])
    assert result["reason"] == "threshold_hides_raw_changes"
    assert result["flight_counts"]["raw_changed_but_decode_unchanged_flights"] == 1


def test_detector_second8_change_is_not_misclassified_as_threshold_hidden():
    f = forecast(target=[0]*10+[1], source=[0]*10+[1])
    detector = np.zeros((5, 10))
    detector[0, 7:] = [.6, .8, 1]
    detector[1, 7:] = [.4, .8, 1]
    annotated = dict(row(), city="hangzhou", seed=123, block=2)
    result = explain_row(annotated, f, [dict(raw_cdfs=f["edge"], detector_cdfs=detector)])
    assert result["reason"] == "decoded_detector_change_outside_service_window"
    assert result["flags"]["raw_cdf_changed"]
    assert result["flags"]["decoded_flight_changed"]
    assert not result["flags"]["endpoint_edge_count_changed"]
    assert not result["flags"]["service_edge_curve_changed"]
    assert result["flight_counts"]["detector_change_unseen_at_endpoint_or_service_flights"] == 1
    assert result["flight_counts"]["raw_changed_but_decode_unchanged_flights"] == 0
    assert result["case"]["city"] == "hangzhou"
    assert result["case"]["seed"] == 123


def test_service_change_without_a_legal_protected_slot_has_no_score_credit():
    source = [0]*11
    source[9] = 1
    result = explain_row(row(masks=(0, 0)), forecast(source=source))
    assert result["reason"] == "changed_slots_have_no_protected_service"
    assert result["flags"]["service_edge_curve_changed"]
    assert not result["flags"]["legal_scores_changed"]


def test_rival_margin_decomposes_queue_arrivals_and_source_differential():
    source = [0]*11
    source[9] = 2
    result = explain_row(row(base=(10, 0)), forecast(source=source))
    assert result["reason"] == "relative_change_below_margin"
    rival = result["case"]["rivals"][0]
    assert rival == dict(request=1, base_margin=10, target_arrival_margin=0,
                         target_total_margin=10, source_differential_gain=2,
                         source_margin_at_target_winner=8)
    assert result["flags"]["target_margin_exceeds_all_source_perturbations"]
    assert result["flags"]["queue_lead_covers_both_arrival_disadvantages"]
    assert result["metrics"]["relative_change_rows_target_total_margin"] == 10
    # Arrival credits can select a different winner from queue-only pressure;
    # that is not a queue-dominance explanation even if source keeps the winner.
    target = [0]*11
    target[9] = 2
    source[9] = 3
    arrival_winner = explain_row(row(base=(1, 0)), forecast(target=target, source=source))
    assert not arrival_winner["flags"]["queue_lead_covers_both_arrival_disadvantages"]


def test_identical_score_shift_across_requests_cannot_change_the_winner():
    source = [0]*11
    source[9] = 1
    result = explain_row(row(base=(2, 0), masks=(1023, 1023)), forecast(source=source))
    assert result["reason"] == "same_shift_all_requests"
    assert result["flags"]["legal_scores_changed"]
    assert not result["flags"]["relative_scores_changed"]


def test_first_feasible_tie_and_service_only_changes_are_kept_distinct():
    source = [0]*11
    source[9] = 1
    result = explain_row(row(base=(0, 1), masks=(1023, 0)), forecast(source=source))
    assert result["reason"] == "source_tie_first_order_changes_request"
    assert result["case"]["target_request"] == 1
    assert result["case"]["source_request"] == 0
    assert not result["flags"]["endpoint_edge_count_changed"]
    assert result["flags"]["request_changed"]


def test_per_vehicle_service_changes_can_cancel_before_the_edge_score():
    raw_a = np.zeros((5, 11)); raw_b = raw_a.copy()
    raw_a[1, 9] = 1
    raw_b[0, 9] = 1
    result = explain_row(row(), forecast(), [dict(raw_cdfs=raw_a), dict(raw_cdfs=raw_b)])
    assert result["reason"] == "vehicle_service_changes_cancel_on_edge"
    assert result["flight_counts"]["service_changed_flights"] == 2


def test_scope_and_locked_denominators_include_excluded_rows_and_merge():
    left, right = DecisionPathObserver(), DecisionPathObserver()
    enabled = dict(original_common_scored=False, all_observable_risk=True)
    left.observe(row(base=(0,), masks=(0,)), forecast(), enabled=enabled)
    right.observe(row(), forecast(), enabled=dict(original_common_scored=True, all_observable_risk=True))
    left.merge(right)
    result = left.result()
    assert result["original_common_scored"]["all"]["counts"]["decisions"] == 2
    assert result["original_common_scored"]["all"]["counts"]["evaluated_decisions"] == 1
    assert result["all_observable_risk"]["locked"]["reasons"] == {"locked": 1}
    assert result["all_observable_risk"]["unlocked"]["counts"]["evaluated_decisions"] == 1
    assert result["all_observable_risk"]["unlocked"]["metrics"]["target_total_margin"]["n"] == 1
