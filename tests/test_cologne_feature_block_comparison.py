"""Fixed-threshold deployment and equal-seed policy costs for the G feature block."""

import pytest

from scripts.data.cologne_feature_block_comparison import compare_actions


def fixture():
    rows = []
    for seed, count, cost in ((2027, 34, 10.), (3037, 33, 20.), (4047, 33, 30.)):
        for index in range(count):
            rows.append({"group_id": f"cologne1:seed{seed}:{index}", "seed": seed, "reference_index": 0,
                         "label_costs": [cost, cost - 3, cost - 3, cost + 1, cost, cost, cost, cost],
                         "old_effective_index": 0, "new_effective_index": 0,
                         "old_deployed_index": 0, "new_deployed_index": 0,
                         "old_raw_predicted_advantage": 0., "new_raw_predicted_advantage": 0.})
    return rows


def test_policy_cost_weights_seeds_equally_and_keeps_all100_denominators():
    rows = fixture()
    for row in rows[:34]:
        row.update(new_effective_index=1, new_deployed_index=1, new_raw_predicted_advantage=.1)
    report = compare_actions(rows)
    assert report["groups"] == 100 and report["weighting"] == "equal_seed"
    assert report["model"]["mean_cost"] == 19.
    assert report["previous_model"]["mean_cost"] == report["phase_pressure"]["mean_cost"] == 20.
    assert report["model"]["accepted_overrides"] == report["model"]["beneficial"] == 34
    assert [report["model"]["per_seed"][str(seed)]["groups"] for seed in (2027, 3037, 4047)] == [34, 33, 33]
    assert report["model"]["mean_cost"] != pytest.approx((34 * 7 + 33 * 20 + 33 * 30) / 100)
    assert report["comparisons"]["overall"]["model_minus_previous_model_cost"] == -1.
    assert report["comparisons"]["overall"]["model_vs_phase_pressure_percent"] == pytest.approx(-5.)
    assert report["action_changes"]["overall"]["model_lower_cost"] == 34


def test_accepted_ties_masked_reference_and_changed_action_same_cost_are_distinct():
    rows = fixture()
    for row in rows[:6]:
        row["label_costs"] = [10., 9., 9., 11., 10., 10., 10., 10.]
    rows[0].update(old_effective_index=1, old_deployed_index=1, old_raw_predicted_advantage=.2,
                   new_effective_index=2, new_deployed_index=2, new_raw_predicted_advantage=.1)
    rows[1].update(new_effective_index=4, new_deployed_index=4, new_raw_predicted_advantage=.1)
    rows[2].update(new_effective_index=0, new_deployed_index=0, new_raw_predicted_advantage=.2)
    rows[3].update(new_effective_index=1, new_deployed_index=0, new_raw_predicted_advantage=1e-12)
    rows[4].update(new_effective_index=1, new_deployed_index=1, new_raw_predicted_advantage=1.01e-12)
    rows[5].update(new_effective_index=3, new_deployed_index=3, new_raw_predicted_advantage=.1)
    report = compare_actions(rows)
    assert {key: report["model"][key] for key in ("accepted_overrides", "beneficial", "harmful", "equal_cost")} == {
        "accepted_overrides": 4, "beneficial": 2, "harmful": 1, "equal_cost": 1}
    assert report["model"]["per_seed"]["2027"]["mean_cost"] == pytest.approx(10. - 1. / 34)
    assert report["model"]["per_seed"]["3037"]["accepted_overrides"] == 0
    assert report["action_changes"]["overall"] == {
        "groups": 100, "deployed_action_changed": 4, "model_lower_cost": 1,
        "model_higher_cost": 1, "same_cost": 98, "changed_action_same_cost": 2}
    assert report["comparisons"]["overall"]["model_minus_previous_model_cost"] == 0.


@pytest.mark.parametrize("score,deployed", [(1e-12, 1), (1.01e-12, 0), (-.1, 1)])
def test_deployed_action_cannot_override_the_fixed_threshold(score, deployed):
    rows = fixture()
    rows[0].update(new_effective_index=1, new_deployed_index=deployed, new_raw_predicted_advantage=score)
    with pytest.raises(ValueError, match="fixed zero threshold"):
        compare_actions(rows)


@pytest.mark.parametrize("change", ["missing_group", "duplicate_group", "changed_seed_count", "missing_cost"])
def test_partial_groups_or_costs_cannot_produce_policy_average(change):
    rows = fixture()
    if change == "missing_group":
        rows.pop()
    elif change == "duplicate_group":
        rows[0]["group_id"] = rows[1]["group_id"]
    elif change == "changed_seed_count":
        rows[0]["seed"] = 3037
    else:
        rows[0]["label_costs"].pop()
    with pytest.raises(ValueError):
        compare_actions(rows)
