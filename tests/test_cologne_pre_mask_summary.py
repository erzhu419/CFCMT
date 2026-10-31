"""All-group policy accounting and the fixed32 veto-group recovery decomposition."""

from statistics import mean

import pytest

from scripts.data.cologne_pre_mask_summary import MISSED, OPTIMAL, summarize_pre_mask


def fixture():
    rows = []
    for seed, count, missed, optimal in ((2027, 34, 9, 4), (3037, 33, 6, 2), (4047, 33, 8, 3)):
        for index in range(count):
            vetoed = index < missed + optimal
            has_gain = index < missed
            costs = [10., 9. if has_gain else 10., 12., 10., 10., 10., 10., 10.]
            selected = 1 if has_gain else 2 if vetoed else 0
            oracle = min(costs)
            rows.append({"group_id": f"cologne1:seed{seed}:{index}", "seed": seed, "label_costs": costs,
                         "reference_index": 0, "reference_cost": 10., "oracle_cost": oracle,
                         "original_model_stay_vetoed": vetoed,
                         "baseline_selected_index": 0, "baseline_deployed_index": 0,
                         "baseline_predicted_advantage": 0., "baseline_policy_cost": 10.,
                         "baseline_regret": 10. - oracle, "baseline_category": MISSED if has_gain else OPTIMAL,
                         "premask_selected_index": selected, "premask_deployed_index": selected,
                         "premask_predicted_advantage": .2 if selected else 0.,
                         "premask_policy_cost": costs[selected], "premask_regret": costs[selected] - oracle})
    return rows


def test_all100_policy_costs_include_beneficial_recovery_and_new_harm():
    report = summarize_pre_mask(fixture())
    assert report["groups"] == 100 and report["weighting"] == "equal_seed"
    assert report["previous_model"]["mean_cost"] == report["phase_pressure"]["mean_cost"] == 10.
    assert {key: report["model"][key] for key in ("accepted_overrides", "beneficial", "harmful", "equal_cost")} == {
        "accepted_overrides": 32, "beneficial": 23, "harmful": 9, "equal_cost": 0}
    recovery = report["veto_recovery"]["overall"]
    assert recovery["groups"] == 32 and recovery["changed_actions"] == 32
    assert recovery["missed_veto"]["groups"] == 23 and recovery["pp_optimal_veto"]["groups"] == 9
    assert recovery["raw_sums"] == {"original_regret": 23., "remaining_regret": 18.,
        "net_regret_recovery": 5., "gross_benefit": 23., "introduced_harm": 18.}
    assert recovery["missed_veto"]["recovered_fraction_of_original_regret"] == 1.
    assert recovery["pp_optimal_veto"]["raw_sums"]["introduced_harm"] == 18.
    assert report["action_changes"]["overall"]["same_cost"] == 68
    assert all(report["checks"].values())


def test_contributions_use_full_seed_denominators_before_equal_seed_average():
    report = summarize_pre_mask(fixture())
    recovery = report["veto_recovery"]
    expected = mean([1. / 34, 2. / 33, 2. / 33])
    assert recovery["overall"]["contributions"]["net_regret_recovery"] == pytest.approx(expected)
    assert recovery["overall"]["contributions"]["net_regret_recovery"] != pytest.approx(5. / 100)
    assert recovery["per_seed"]["2027"]["missed_veto"]["contributions"]["original_regret"] == pytest.approx(9. / 34)
    assert report["model"]["mean_cost"] == pytest.approx(10. - expected)
    assert report["comparisons"]["overall"]["model_minus_previous_model_cost"] == pytest.approx(-expected)


def test_tied_new_override_and_threshold_rejection_do_not_drop_veto_groups():
    rows = fixture()
    # A PP-optimal group accepts an equal-cost switch, while one missed group remains PP.
    rows[9].update(premask_selected_index=3, premask_deployed_index=3,
                   premask_policy_cost=10., premask_regret=0.)
    rows[0].update(premask_selected_index=1, premask_deployed_index=0, premask_predicted_advantage=1e-12,
                   premask_policy_cost=10., premask_regret=1.)
    recovery = summarize_pre_mask(rows)["veto_recovery"]["overall"]
    assert recovery["groups"] == 32 and recovery["changed_actions"] == 31
    assert recovery["new_override_outcomes"] == {"accepted_overrides": 31, "beneficial": 22, "harmful": 8, "equal_cost": 1}
    assert recovery["missed_veto"]["raw_sums"]["remaining_regret"] == 1.


def test_new_harm_in_original_missed_groups_can_make_recovery_negative():
    rows = fixture()
    for row in rows:
        if row["baseline_category"] == MISSED:
            row.update(premask_selected_index=2, premask_deployed_index=2,
                       premask_policy_cost=12., premask_regret=3.)
    recovery = summarize_pre_mask(rows)["veto_recovery"]["overall"]["missed_veto"]
    assert recovery["raw_sums"]["introduced_harm"] == 46.
    assert recovery["raw_sums"]["net_regret_recovery"] == -46.
    assert recovery["recovered_fraction_of_original_regret"] == -2.


@pytest.mark.parametrize("change", ["non_veto_action", "lost_group", "veto_roster", "native_cost"])
def test_changed_control_or_inconsistent_recovery_cannot_produce_summary(change):
    rows = fixture()
    if change == "non_veto_action":
        rows[13].update(premask_selected_index=1, premask_deployed_index=1, premask_predicted_advantage=.2)
    elif change == "lost_group":
        rows.pop()
    elif change == "veto_roster":
        rows[0]["original_model_stay_vetoed"] = False
    else:
        rows[0]["premask_policy_cost"] = 8.
    with pytest.raises(ValueError):
        summarize_pre_mask(rows)
