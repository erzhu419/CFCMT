"""Keep incomplete/incorrect reference arms out and preserve prior evidence."""

import json

import pytest

from experiments.tsc_v93_exact_reference_20261031 import analyze


def reference_payload(tmp_path):
    original = analyze.read_csv(analyze.DATA / "confirmation_rollouts.csv")
    rows = []
    for row in original:
        if row["method"] != "frozen_selector":
            continue
        # A deliberately identical fixture arm tests the zero-effect estimand.
        rows.append(dict(city=row["city"], scenario=row["scenario"], seed=int(row["seed"]),
                         policy=analyze.POLICY, source_policy="selected_source_prior",
                         runtime_policy="selected_source_prior", reference_key=analyze.REFERENCE_KEY,
                         evaluation_horizon_sec=3600, control_interval_sec=10, warmup_sec=60,
                         reference_decision_requests=360,
                         reference_post_warmup_decision_requests=354,
                         result_path="fixture/result.json", tripinfo_path="fixture/tripinfo.xml",
                         metrics={**{key: float(row[key]) for key in analyze.METRICS},
                                  "collision_events": 0}))
    return dict(protocol=analyze.SHARD_PROTOCOL, passed=True, source_policy="selected_source_prior",
                task_count=len(rows), shard_index=0, shard_count=1, rows=rows)


def save_shard(tmp_path, payload):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "shard_0.json").write_text(json.dumps(payload))


def seeds():
    return json.loads(analyze.SPEC.read_text())["confirmation_seeds"]


def test_missing_reference_run_cannot_enter_report(tmp_path):
    payload = reference_payload(tmp_path)
    payload["rows"].pop()
    payload["task_count"] -= 1
    save_shard(tmp_path, payload)
    with pytest.raises(ValueError, match="complete method/scenario/seed matrix"):
        analyze.load_reference_rows(tmp_path, seeds())


def test_similarly_named_pressure_baseline_cannot_substitute(tmp_path):
    payload = reference_payload(tmp_path)
    payload["rows"][0]["runtime_policy"] = "phase_pressure"
    save_shard(tmp_path, payload)
    with pytest.raises(ValueError, match="Reference identity"):
        analyze.load_reference_rows(tmp_path, seeds())


def test_identical_fixture_retains_old_results_and_expands_multiplicity(tmp_path):
    shards, out = tmp_path / "shards", tmp_path / "analysis"
    save_shard(shards, reference_payload(tmp_path))
    report = analyze.build(shards, analyze.DATA, out)
    assert report["total_rollout_count"] == 1568
    assert all(row["registered_mean_relative_delta_pct"] == 0
               for row in report["reference_comparisons"].values())
    assert all(gate["conclusion"] == "positive_incremental_benefit_not_established"
               for gate in report["incremental_benefit_assessment"].values())
    decomposition = report["outcome_decomposition"]
    assert decomposition["by_city"]["jinan"]["waiting_exact_tie_run_count"] == 168
    assert decomposition["by_city"]["los_angeles"]["city_mean_waiting_exact_tie_seed_count"] == 56
    assert decomposition["unique_jinan_override_case"]["accepted_nonreference_switch_requests"] == 1
    assert decomposition["unique_jinan_override_case"]["waiting_exact_tie"]
    for suffix in ("rollouts", "summary", "comparisons", "seed"):
        old = analyze.read_csv(analyze.DATA / f"confirmation_{suffix}.csv")
        new = analyze.read_csv(out / f"seven_controller_{suffix}.csv")
        for before, after in zip(old, new):
            assert all(after[key] == cell for key, cell in before.items())
    comparisons = analyze.read_csv(out / "seven_controller_comparisons.csv")
    assert len(comparisons) == 18
    mp = next(row for row in comparisons if row["scope"] == "macro"
              and row["comparator"] == "max_pressure")
    assert float(mp["wilcoxon_p_holm_expanded18"]) > float(mp["wilcoxon_p_holm_expanded15"])

