from __future__ import annotations

from scripts.cluster.audit_tsc_source_city_disjoint_confirmation import (
    _holm_adjust,
    _paired_summary,
)


def test_paired_summary_reports_exact_tie_without_wilcoxon_failure() -> None:
    summary = _paired_summary([0.0, 0.0, 0.0], bootstrap_seed=7)

    assert summary["exact_tie"] is True
    assert summary["wilcoxon_p_two_sided"] == 1.0
    assert summary["bootstrap_mean_ci95"] == [0.0, 0.0]


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = _holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})

    assert adjusted["a"] == 0.03
    assert adjusted["b"] == 0.06
    assert adjusted["c"] == 0.2
