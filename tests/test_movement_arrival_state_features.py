from cf_h2o.traffic_signal.movement_arrival_timeline import (
    MOVEMENT_ARRIVAL_FEATURE_NAMES,
)
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    FEATURE_SETS,
    STATE_ACTION_FEATURES,
)


def test_arrival_state_feature_set_is_a_matched_extension() -> None:
    arrival = FEATURE_SETS["state_action_arrival"]

    assert arrival[: len(STATE_ACTION_FEATURES)] == STATE_ACTION_FEATURES
    for name in MOVEMENT_ARRIVAL_FEATURE_NAMES:
        assert name in arrival
        assert f"delta_{name}" in arrival
