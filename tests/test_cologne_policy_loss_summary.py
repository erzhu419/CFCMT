"""Equal-seed, exhaustive policy loss accounting without conditioning on losses."""

import pytest

from scripts.data.cologne_policy_loss_summary import CATEGORIES, summarize_loss


def set_costs(row, reference, policy, oracle, deployed, category):
    row.update(reference_cost=reference, policy_cost=policy, oracle_cost=oracle,
               pp_to_oracle_gain=reference - oracle, realized_gain=reference - policy,
               regret=policy - oracle, reference_index=0, deployed_index=deployed, category=category)


def fixture():
    rows = []
    for seed, count in ((2027, 34), (3037, 33), (4047, 33)):
        for index in range(count):
            row = {"group_id": f"cologne1:seed{seed}:{index}", "seed": seed}
            set_costs(row, 1., 1., 1., 0, CATEGORIES[0])
            rows.append(row)
    return rows


def test_unequal_seed_sizes_use_all_groups_and_equal_seed_contributions():
    rows = fixture()
    for row in rows[:34]:
        set_costs(row, 4., 3., 1., 1, CATEGORIES[2])
    report = summarize_loss(rows)
    assert report["groups"] == 100 and report["weighting"] == "equal_seed"
    assert report["means"]["policy_cost"] == pytest.approx(5. / 3)
    assert report["means"]["pp_to_oracle_gain"] == 1.
    assert report["means"]["realized_gain"] == pytest.approx(1. / 3)
    assert report["means"]["regret"] == pytest.approx(2. / 3)
    assert report["raw_sums"]["regret"] == 68.
    assert report["per_seed"]["2027"]["raw_sums"]["pp_to_oracle_gain"] == 102.
    category = report["categories"][CATEGORIES[2]]
    assert category["groups"] == 34 and category["contributions"]["regret"] == pytest.approx(2. / 3)
    assert category["contributions"]["regret"] != pytest.approx(category["raw_sums"]["regret"] / 100)
    assert category["regret_share"] == 1.
    assert report["realized_fraction_of_available_gain"] == pytest.approx(1. / 3)


def test_four_categories_include_optimal_ties_and_account_for_every_loss():
    rows = fixture()
    set_costs(rows[0], 3., 1., 1., 1, CATEGORIES[0])
    set_costs(rows[1], 1., 2., 1., 1, CATEGORIES[1])
    set_costs(rows[2], 4., 2., 1., 1, CATEGORIES[2])
    set_costs(rows[3], 4., 4., 1., 0, CATEGORIES[3])
    set_costs(rows[4], 1., 1., 1., 2, CATEGORIES[0])  # Accepted, equal-cost alternative optimum.
    report = summarize_loss(rows)
    assert [report["categories"][category]["groups"] for category in CATEGORIES] == [97, 1, 1, 1]
    assert report["policy_outcomes"] == {"accepted_overrides": 4, "beneficial": 2, "harmful": 1, "equal_cost": 1}
    assert report["means"]["regret"] == pytest.approx(5. / 102)
    assert [report["categories"][category]["regret_share"] for category in CATEGORIES] == pytest.approx([0., .2, .2, .6])
    assert sum(report["categories"][category]["contributions"]["regret"] for category in CATEGORIES[1:]) == pytest.approx(report["means"]["regret"])
    assert report["means"]["pp_to_oracle_gain"] == pytest.approx(report["means"]["realized_gain"] + report["means"]["regret"])
    assert report["realized_fraction_of_available_gain"] == pytest.approx(3. / 8)
    assert all(report["checks"].values())


def test_available_gain_fraction_is_none_when_absent_and_can_be_negative():
    rows = fixture()
    report = summarize_loss(rows)
    assert report["realized_fraction_of_available_gain"] is None
    assert all(category["regret_share"] is None for category in report["categories"].values())
    set_costs(rows[0], 1., 3., 1., 1, CATEGORIES[1])
    set_costs(rows[1], 2., 2., 1., 0, CATEGORIES[3])
    assert summarize_loss(rows)["realized_fraction_of_available_gain"] == pytest.approx(-2.)


@pytest.mark.parametrize("change", ["missing_group", "duplicate_group", "wrong_seed_count", "wrong_gain", "wrong_category"])
def test_incomplete_or_inconsistent_decomposition_is_rejected(change):
    rows = fixture()
    if change == "missing_group":
        rows.pop()
    elif change == "duplicate_group":
        rows[0]["group_id"] = rows[1]["group_id"]
    elif change == "wrong_seed_count":
        rows[0]["seed"] = 3037
    elif change == "wrong_gain":
        rows[0]["realized_gain"] = .1
    else:
        rows[0]["category"] = CATEGORIES[3]
    with pytest.raises(ValueError):
        summarize_loss(rows)
