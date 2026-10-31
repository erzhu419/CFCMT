import pytest

from scripts.data.summarize_sumo_collision_trace import (
    PROTOCOL,
    summarize_collision_trace,
)


def _vehicle(vehicle_id: str, **values):
    return {"vehicle_id": vehicle_id, "present": True, **values}


def test_collision_trace_summary_reports_dynamics_without_inferring_cause() -> None:
    payload = {
        "admission_result": {
            "observed": {
                "collision_samples": [{"time_sec": 12.0, "type": "collision"}],
                "diagnostic_trace": {
                    "protocol": "trace-v1",
                    "read_only": True,
                    "samples": [
                        {
                            "time_sec": 10.0,
                            "vehicle_ids_by_lane": {
                                "target_1": ["collider", "victim"],
                                "foe_0": [],
                            },
                            "vehicles": [
                                _vehicle(
                                    "collider",
                                    Speed=10.0,
                                    Acceleration=0.0,
                                    Decel=4.5,
                                    EmergencyDecel=9.0,
                                    LaneID="upstream_1",
                                    RoadID="upstream",
                                    MinGap=1.5,
                                    LeaderWithin100m=["victim", 4.0],
                                ),
                                _vehicle(
                                    "victim",
                                    Speed=9.0,
                                    Acceleration=-1.0,
                                    Decel=4.5,
                                    EmergencyDecel=9.0,
                                    LaneID="target_1",
                                    RoadID="target",
                                ),
                            ],
                        },
                        {
                            "time_sec": 11.0,
                            "vehicle_ids_by_lane": {
                                "target_1": ["collider", "victim"],
                                "foe_0": ["foe"],
                            },
                            "vehicles": [
                                _vehicle(
                                    "collider",
                                    Speed=8.0,
                                    Acceleration=-5.0,
                                    Decel=4.5,
                                    EmergencyDecel=9.0,
                                    LaneID="target_1",
                                    RoadID="target",
                                    MinGap=1.5,
                                    LeaderWithin100m=["victim", -0.6],
                                ),
                                _vehicle(
                                    "victim",
                                    Speed=4.0,
                                    Acceleration=-9.0,
                                    Decel=4.5,
                                    EmergencyDecel=9.0,
                                    LaneID="target_1",
                                    RoadID="target",
                                ),
                            ],
                        },
                    ],
                },
            }
        }
    }

    result = summarize_collision_trace(
        payload,
        collider_id="collider",
        victim_id="victim",
        watched_lane_ids=("target_1", "foe_0"),
    )

    assert result["protocol"] == PROTOCOL
    assert result["participants"]["collider"]["lane_transitions"] == [
        {"time_sec": 11.0, "from": "upstream_1", "to": "target_1"}
    ]
    assert result["participants"]["collider"]["braking_exceeded_declared_decel"]
    assert result["participants"]["victim"]["braking_reached_emergency_decel"]
    assert result["longitudinal_pair"]["minimum_bumper_gap_m"] == 0.9
    assert result["longitudinal_pair"]["below_min_gap_times_sec"] == [11.0]
    assert result["watched_lanes"]["foe_0"]["nonparticipant_vehicle_ids"] == [
        "foe"
    ]
    assert "do not by themselves identify" in result["interpretation_boundary"]


def test_boston_v18_getleader_clearance_matches_saved_vehicle_geometry() -> None:
    # Retained t90329 samples: getLeader excludes the 1.5 m minGap. In
    # particular, a positive 0.788 m return does not violate that minimum.
    observations = [
        (17264.0, 34.41051730281374, 41.69885578417067, 0.7883384813569307),
        (17265.0, 46.06277863851114, 52.005570522780445, -0.5572081157306954),
    ]
    samples = [
        {
            "time_sec": time_sec,
            "vehicles": [
                _vehicle(
                    "271533",
                    LaneID="591691873#0_1",
                    LanePosition=collider_position,
                    Length=5.0,
                    MinGap=1.5,
                    LeaderWithin100m=["289458", raw_clearance],
                ),
                _vehicle(
                    "289458",
                    LaneID="591691873#0_1",
                    LanePosition=victim_position,
                    Length=5.0,
                    MinGap=1.5,
                ),
            ],
        }
        for time_sec, collider_position, victim_position, raw_clearance in observations
    ]
    result = summarize_collision_trace(
        {"observed": {"diagnostic_trace": {"samples": samples}}},
        collider_id="271533",
        victim_id="289458",
    )

    pair = result["longitudinal_pair"]
    assert pair["below_min_gap_times_sec"] == [17265.0]
    assert pair["first_bumper_gap_m"] == pytest.approx(2.2883384813569307)
    assert pair["final_bumper_gap_m"] == pytest.approx(0.9427918842693046)
    assert pair["minimum_bumper_gap_m"] > 0.0  # Neither sample has body overlap.
    assert pair["first_raw_leader_clearance_m"] == observations[0][3]
    assert pair["final_raw_leader_clearance_m"] == observations[1][3]
    for sample, pair_sample in zip(samples, pair["samples"]):
        collider, victim = sample["vehicles"]
        geometric_gap = victim["LanePosition"] - victim["Length"] - collider["LanePosition"]
        assert pair_sample["bumper_gap_m"] == pytest.approx(geometric_gap)
        assert pair_sample["below_min_gap"] == (geometric_gap < collider["MinGap"])


def test_collision_trace_summary_requires_samples() -> None:

    with pytest.raises(ValueError, match="contains no samples"):
        summarize_collision_trace(
            {"observed": {"diagnostic_trace": {"samples": []}}},
            collider_id="collider",
            victim_id="victim",
        )
