import numpy as np

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.target_action_support import TargetActionSupport


def _dataset(features, targets):
    values = np.asarray(features, dtype=float)
    rows = values.shape[0]
    return MechanismDataset(
        feature_names=("total_q", "mean_speed", "delta_green_q"),
        features=values,
        context_names=("network",),
        context=np.zeros((rows, 1)),
        priors={"interval_cost": np.zeros(rows)},
        targets={"interval_cost": np.asarray(targets, dtype=float)},
        domains=np.asarray(["target"] * rows),
        metadata={"action_group_ids": [f"g{row}" for row in range(rows)]},
    )


def test_target_action_support_is_local_and_label_free():
    train = _dataset([[0, 10, 0], [10, 8, 2], [20, 6, 4]], [1, 2, 3])
    support = TargetActionSupport.fit(train)
    identical = support.support(_dataset([[10, 8, 2]], [999]))
    far = support.support(_dataset([[100, 1, 30]], [-999]))
    assert np.isclose(identical[0], 1.0)
    assert far[0] < identical[0]

    relabeled = _dataset(train.features, [-10, 0, 20])
    relabeled_support = TargetActionSupport.fit(relabeled)
    assert np.allclose(support.center, relabeled_support.center)
    assert np.allclose(support.scale, relabeled_support.scale)
    assert np.allclose(support.prototypes, relabeled_support.prototypes)


def test_oof_validated_support_enforces_operational_floor():
    names = (
        "total_q",
        "total_veh",
        "mean_speed",
        "mean_occ",
        "green_q",
        "red_q",
        "green_down_q",
        "green_down_occ",
        "delta_green_q",
        "delta_red_q",
        "delta_green_down_q",
        "delta_green_down_occ",
        "delta_service_pressure",
        "delta_switch_indicator",
        "delta_clearance_fraction",
        "delta_current_phase_overlap",
    )
    row = {name: 0.0 for name in names}
    row.update({"total_q": 1.0, "total_veh": 1.0, "mean_speed": 10.0})
    support = TargetActionSupport.fit_validated_records(
        [{"candidate_features": row}]
    )
    empty = MechanismDataset(
        feature_names=names,
        features=np.asarray([[0.0, 0.0, 10.0, *([0.0] * 13)]]),
        context_names=("network",),
        context=np.zeros((1, 1)),
        priors={"interval_cost": np.zeros(1)},
        targets={"interval_cost": np.zeros(1)},
        domains=np.asarray(["target"]),
        metadata={"action_group_ids": ["g0"]},
    )
    matching = MechanismDataset(
        feature_names=names,
        features=np.asarray([[row[name] for name in names]]),
        context_names=("network",),
        context=np.zeros((1, 1)),
        priors={"interval_cost": np.zeros(1)},
        targets={"interval_cost": np.zeros(1)},
        domains=np.asarray(["target"]),
        metadata={"action_group_ids": ["g1"]},
    )

    assert support.support(empty)[0] == 0.0
    assert np.isclose(support.support(matching)[0], 1.0)
    assert support.diagnostics()["label_free"] is False


def test_exact_tree_support_matches_frozen_bruteforce_distance():
    rng = np.random.default_rng(20260815)
    train = _dataset(rng.normal(size=(500, 3)), np.zeros(500))
    query = _dataset(rng.normal(size=(73, 3)), np.zeros(73))
    support = TargetActionSupport.fit(train)

    expected = support.support_bruteforce(query)
    observed = support.support(query)

    assert np.allclose(observed, expected, rtol=0.0, atol=2e-15)
    del support._prototype_tree
    assert np.allclose(support.support(query), expected, rtol=0.0, atol=2e-15)
