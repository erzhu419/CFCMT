import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("fresh_v93_analysis", Path(__file__).with_name("analyze.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def panel():
    protocol = dict(confirmation_seeds=[101, 102], methods=["frozen_selector", "max_pressure"],
                    city_scenarios={"los_angeles": ["la"], "jinan": ["j1", "j2", "j3"]})
    rows = []
    for method in protocol["methods"]:
        for city, scenarios in protocol["city_scenarios"].items():
            for scenario in scenarios:
                for seed in protocol["confirmation_seeds"]:
                    wait = (50 if method == "frozen_selector" else 100) if city == "los_angeles" else 10
                    rows.append(dict(method=method, city=city, scenario=scenario, seed=seed,
                                     **{key: wait for key in analysis.METRICS}))
    return protocol, rows


def test_macro_is_mean_city_relative_effect_not_ratio_of_means():
    protocol, rows = panel()
    indexed = analysis.index_rows(rows, protocol)
    assert analysis.paired_effects(indexed, protocol, "max_pressure", "macro").tolist() == [-0.25, -0.25]


@pytest.mark.parametrize("change", ["missing", "duplicate", "replacement"])
def test_incomplete_or_changed_panel_cannot_be_reported(change):
    protocol, rows = panel()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[-1]["seed"] = 103
    with pytest.raises(ValueError, match="Complete unique"):
        analysis.index_rows(rows, protocol)


def test_holm_stepdown_uses_the_whole_nine_test_family():
    rows = [dict(comparator=str(i), scope="macro", wilcoxon_p_two_sided=p,
                 mean_relative_delta_pct=-1, bootstrap_ci95_high_pct=-0.1)
            for i, p in enumerate([0.001, 0.002, 0.01, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])]
    analysis.holm_adjust(rows)
    assert [r["wilcoxon_p_holm9"] for r in rows[:3]] == pytest.approx([0.009, 0.016, 0.07])
    assert [r["waiting_superiority_confirmed"] for r in rows[:3]] == [True, True, False]


def test_complete_exact_tie_panel_reports_no_confirmed_advantage(tmp_path):
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    rows = [dict(method=method, city=city, scenario=scenario, seed=seed,
                 **{key: 10.0 if key in analysis.METRICS[:5] else 0 for key in analysis.METRICS})
            for method in protocol["methods"]
            for city, scenarios in protocol["city_scenarios"].items()
            for scenario in scenarios for seed in protocol["confirmation_seeds"]]
    records = tmp_path / "records.csv"
    analysis.write_csv(records, rows)
    report = analysis.build(records, protocol, tmp_path / "analysis")
    assert report["rollout_count"] == 896
    assert report["comparison_count"] == 9
    assert all(not r["waiting_superiority_confirmed"] and r["tied_seed_count"] == 56
               for r in report["macro_comparisons"])
    assert report["exact_gp_outcome_ties"]["jinan"]["all_ten_metrics_exact_ties"] == 168
