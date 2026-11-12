from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.traffic_signal.action_contrast import make_target_action_contrasts
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster import collect_tsc_native_action_timeline as timeline


def make_rows(deltas):
    rows = []
    for second in range(1, 451):
        delta, local = next((delta, local) for end, delta, local in deltas if second <= end)
        rows.append({
            "offset_sec": second, "cost": (20 + delta) / 4,
            "incoming_q": {"in": 3 + local}, "outgoing_q": {"out": 2},
            "outgoing_occupancy_fraction": {"out": 0.125},
            "applied": {"mode": "green", "sumo_state": "G", "target_green_state": "G"},
        })
    return rows


def test_fixed_windows_preserve_harm_sign_cumulative_and_local_decomposition():
    branches = {
        "pp": {"rows": make_rows([(450, 0, 0)])},
        "selected": {"rows": make_rows([(10, 4, 1), (60, -2, -1), (120, 1, 0), (450, -1, 0)])},
    }
    result = timeline.summarize_pair({}, branches, 4, ["G"])
    windows = result["windows"]
    assert [(w["start_sec"], w["end_sec"]) for w in windows] == list(timeline.WINDOWS)
    assert [w["delta_vehicle_sec"] for w in windows] == [40, -100, 60, -330]
    assert [w["cumulative_delta_vehicle_sec"] for w in windows] == [40, -60, 0, -330]
    assert [w["focal_delta_vehicle_sec"] for w in windows] == [10, -50, 0, 0]
    assert [w["other_tls_delta_vehicle_sec"] for w in windows] == [30, -50, 60, -330]
    assert result["delta_vehicle_sec"] == -330
    assert windows[0]["arms"]["selected"]["applied_mode_seconds"] == {"green": 10}
    assert windows[-1]["arms"]["pp"]["outgoing_occupancy_mean_fraction"] == 0.125


@pytest.mark.parametrize("change", ["drop", "duplicate"])
def test_missing_or_duplicate_second_cannot_be_aggregated(change):
    rows = make_rows([(450, 0, 0)])
    broken = deepcopy(rows)
    if change == "drop":
        del broken[9]
    else:
        broken[9] = deepcopy(broken[8])
    with pytest.raises(ValueError, match="exactly seconds 1 through 450"):
        timeline.summarize_pair({}, {"pp": {"rows": rows}, "selected": {"rows": broken}}, 4, ["G"])


def observer_fixture(monkeypatch):
    executor = SimpleNamespace(mode="yellow", current="y")
    executor.snapshot = lambda: {
        "mode": executor.mode, "current_state": executor.current,
        "current_green_state": "G", "target_green_state": "G", "audit": {},
    }
    api = SimpleNamespace(
        trafficlight=SimpleNamespace(getRedYellowGreenState=lambda tls: executor.current),
        lane=SimpleNamespace(getLastStepHaltingNumber=lambda lane: 2,
                             getLastStepOccupancy=lambda lane: 0.25),
    )
    originator = SimpleNamespace(
        checkpoint_sec=100, focal_tls="tls", controlled_lanes=("in",),
        info=SimpleNamespace(incoming_lanes=("in",), controlled_links=((('in', 'out', 'via'),),)),
    )
    monkeypatch.setattr(timeline, "physical_snapshot", lambda api, executors: {"pre_request": True})
    monkeypatch.setattr(timeline, "_sample", lambda api, lanes, now: {"time_sec": now, "cost": 2.})
    monkeypatch.setattr(timeline, "_record_step_diagnostics", lambda api, record: None)
    return timeline.TimelineObserver(api, originator), executor, {"tls": executor}


def test_observer_aligns_applied_signal_with_completed_second_and_retains_end_signal(monkeypatch):
    observer, executor, executors = observer_fixture(monkeypatch)
    observer(stage="after_executor_advance", time_sec=100, executors=executors)
    assert observer.start == {"pre_request": True}
    assert observer.rows == []  # The checkpoint itself does not enter the 450-second cost.
    observer(stage="before_simulation_step", time_sec=100, executors=executors)
    executor.mode, executor.current = "green", "G"
    observer(stage="after_executor_advance", time_sec=101, executors=executors)
    row = observer.rows[0]
    assert row["offset_sec"] == 1 and row["time_sec"] == 101
    assert row["applied"]["mode"] == "yellow"
    assert row["applied"]["sumo_state"] == "y"
    assert row["end"]["mode"] == "green"
    assert row["end"]["sumo_state"] == "G"
    assert row["incoming_q"] == {"in": 2}
    assert row["outgoing_occupancy_fraction"] == {"out": 0.25}
    with pytest.raises(ValueError, match="missing applied phase"):
        observer(stage="after_executor_advance", time_sec=102, executors=executors)


def frozen_bank():
    states = [f"state{i}" for i in range(8)]
    names = timeline.v3.FEATURE_NAMES_V3
    features = np.arange(8 * len(names), dtype=float).reshape(8, len(names)) / 100.
    features[5, names.index("green_q")] = 100.  # Unique PP reference, not row zero.
    absolute = MechanismDataset(
        feature_names=names, features=features, context_names=("context",),
        context=np.ones((8, 1)), priors={}, targets={}, domains=np.array(["target"] * 8),
        metadata={"action_group_ids": ["case"] * 8, "candidate_states": states,
                  "row_times": [100.] * 8, "row_tls": ["tls"] * 8},
    )
    contrast, reference = make_target_action_contrasts(
        absolute, reference_policy="phase_pressure", contrast_features=timeline.v3.CONTRAST_FEATURES_V3,
    )
    assert reference == 5
    group = dict(group_id="case", candidate_states=states, reference_index=reference,
                 native_costs=[float(i) for i in range(8)], checkpoint_sec=100, tls_id="tls")
    case = dict(group_id="case", selected_state="state2", reference_state="state5",
                reference_cost_450=5., selected_cost_450=2.)
    # Match the real bank: complete runtime contrasts, then nine legacy predictions.
    bank_features = np.column_stack((contrast.features, np.full((8, 9), -999.)))
    bank = SimpleNamespace(
        metadata={"action_group_ids": ["other", *(["case"] * 8), "other"],
                  "candidate_states": ["unused", *states, "unused"]},
        feature_names=(*contrast.feature_names, *timeline.v158.RAW_PREDICTION_FEATURES),
        features=np.vstack((bank_features[:1], bank_features, bank_features[-1:])),
        context_names=contrast.context_names, context=np.vstack(([[0.]], contrast.context, [[0.]])),
    )
    return case, {"groups": [group]}, bank, contrast


@pytest.mark.parametrize("candidate", [2, 5])
def test_reconstructed_runtime_contrasts_pass_parent_checkpoint_selection(candidate):
    case, old, bank, contrast = frozen_bank()
    group, selected = timeline.reconstruct_group(case, old, bank)
    assert selected == 2 and group["reference_index"] == 5
    expected = group["dataset"]
    np.testing.assert_array_equal(expected.features, contrast.features)
    np.testing.assert_array_equal(expected.context, contrast.context)
    assert expected.feature_names == contrast.feature_names
    assert all(f"reference_{name}" in expected.feature_names for name in timeline.v3.CONTRAST_FEATURES_V3)
    assert all(f"delta_{name}" in expected.feature_names for name in timeline.v3.CONTRAST_FEATURES_V3)
    assert not set(expected.feature_names) & set(timeline.v158.RAW_PREDICTION_FEATURES)
    assert "dataset" not in old["groups"][0]
    originator = timeline.v158.FixedActionBranchOriginator(group, candidate)
    decision = originator.select(contrast, reference_index=5)
    assert all(originator.dataset_match.values())
    assert decision.selected_index == candidate
    assert decision.eligible == (candidate != 5)


def test_reconstruction_rejects_changed_candidate_order_or_frozen_endpoint():
    case, old, bank, _ = frozen_bank()
    bank.metadata["candidate_states"][1:3] = reversed(bank.metadata["candidate_states"][1:3])
    with pytest.raises(ValueError, match="candidate order"):
        timeline.reconstruct_group(case, old, bank)
    case, old, bank, _ = frozen_bank()
    case["selected_cost_450"] += 0.01
    with pytest.raises(ValueError, match="endpoint differs"):
        timeline.reconstruct_group(case, old, bank)
