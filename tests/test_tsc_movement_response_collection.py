from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
from scripts.cluster import collect_tsc_movement_response as runner


def spec():
    case = {"candidate_states": list("ABCDEFGH"), "reference_index": 2,
            "candidate_index": 0, "expected_costs": {"pp": 1.0, "candidate": 1.1}}
    return {"protocol": runner.PROTOCOL, "horizon_sec": 450, "windows_sec": [10, 60, 450],
            "max_new_branches": 4, "new_fits": 0,
            "cases": [{**case, "city": "cologne"}, {**case, "city": "jinan"}]}


def test_fixed_budget_and_windows_cannot_silently_expand():
    runner.validate_spec(spec())
    for key, value in (("max_new_branches", 5), ("new_fits", 1), ("windows_sec", [10, 120, 450])):
        altered = deepcopy(spec())
        altered[key] = value
        with pytest.raises(ValueError, match="four-branch"):
            runner.validate_spec(altered)


def test_frozen_candidate_must_be_a_distinct_non_stay_bank_row():
    case = {"group_id": "g", "checkpoint_sec": 120, **spec()["cases"][0]}
    bank = SimpleNamespace(metadata={"action_group_ids": ["g"] * 8,
        "candidate_states": list("ABCDEFGH"), "row_times": [120] * 8},
        feature_names=("switch_indicator",), features=np.ones((8, 1)))
    np.testing.assert_array_equal(runner.select_bank_group(bank, case), np.arange(8))
    bank.features[0] = 0
    with pytest.raises(ValueError, match="non-stay"):
        runner.select_bank_group(bank, case)


def fake_runtime(teleport=False):
    state = SimpleNamespace(now=0, signal="r", trace=[])
    def request(phase):
        state.signal = "y"
        state.trace.append(("request", state.now, phase))
    def step():
        state.trace.append(("step", state.now, state.signal))
        state.now += 1
    def advance(elapsed):
        state.trace.append(("advance", state.now, elapsed))
        state.signal = "G"
    api = SimpleNamespace(
        simulation=SimpleNamespace(getTime=lambda: state.now,
            getStartingTeleportNumber=lambda: int(teleport and state.now == 1),
            getEndingTeleportNumber=lambda: 0, getCollisions=lambda: ()),
        simulationStep=step,
        lane=SimpleNamespace(getLastStepHaltingNumber=lambda lane: state.now + (lane == "b")),
        vehicle=SimpleNamespace(getIDList=lambda: []),
        trafficlight=SimpleNamespace(getRedYellowGreenState=lambda tls: state.signal,
                                    getIDList=lambda: ["tls"]),
    )
    executors = {"tls": SimpleNamespace(request=request, advance=advance)}
    return api, executors, state


def test_observed_advance_preserves_original_control_order_cost_and_ledger():
    api, executors, state = fake_runtime()
    original_ledger = v3._new_safety_ledger()
    expected = v3._advance_with_actions(sumo_api=api, executors=executors, actions={"tls": "G"},
        control_interval_sec=10, cost_lanes=["a", "b"], collision_mode="audit",
        safety_ledger=original_ledger)
    original_trace = state.trace[:]
    api, executors, state = fake_runtime()
    ledger, samples = v3._new_safety_ledger(), []
    observer = SimpleNamespace(sample=lambda now, applied_states: samples.append((now, applied_states)))
    actual = runner.advance_observed(api, v3, executors, {"tls": "G"}, ["a", "b"], ledger, observer)
    assert actual == [cost / 2 for cost in expected]
    assert state.trace == original_trace
    assert ledger == original_ledger
    assert samples[0] == (1, {"tls": "y"})
    assert samples[1] == (2, {"tls": "G"})
    assert len(samples) == 10


def test_teleport_invalidates_measurement_before_label_is_recorded():
    api, executors, state = fake_runtime(teleport=True)
    samples = []
    observer = SimpleNamespace(sample=lambda now, applied_states: samples.append(now))
    with pytest.raises(ValueError, match="teleported"):
        runner.advance_observed(api, v3, executors, {"tls": "G"}, ["a"], v3._new_safety_ledger(), observer)
    assert samples == []
