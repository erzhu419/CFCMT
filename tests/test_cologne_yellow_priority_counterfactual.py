from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.data import run_cologne_yellow_priority_counterfactual as counterfactual


def test_priority_yellow_preserves_major_and_minor_priority_only():
    assert counterfactual.priority_yellow("rGgGgr", "ryyyyr") == "rYyYyr"
    with pytest.raises(ValueError):
        counterfactual.priority_yellow("Ggr", "Yyr")


def test_observer_changes_exactly_one_trigger_and_retains_original_before_sample(monkeypatch):
    state = {"mode": "yellow", "remaining_sec": 3., "current_green_state": "Ggr",
             "current_state": "yyr", "pending_state": "rGG"}
    writes = []

    def set_state(value):
        writes.append(value)
        state["current_state"] = value

    executor = SimpleNamespace(snapshot=lambda: deepcopy(state), _set_state=set_state)
    api = SimpleNamespace(vehicle=SimpleNamespace(getIDList=lambda: []))
    observer = counterfactual.CounterfactualObserver(api, None)

    def record(self, *, stage, time_sec, executors):
        self.samples.append({"time_sec": time_sec, "stage": stage,
                             "tls_state": state["current_state"], "executor": deepcopy(state)})

    monkeypatch.setattr(counterfactual.replay.WindowObserver, "__call__", record)
    executors = {counterfactual.replay.TLS: executor}
    observer(stage="before_simulation_step", time_sec=27479, executors=executors)
    observer(stage="after_executor_advance", time_sec=27480, executors=executors)
    assert not writes
    observer(stage="before_simulation_step", time_sec=27480, executors=executors)
    assert writes == ["Yyr"]
    assert observer.before_intervention_sample["tls_state"] == "yyr"
    assert observer.before_intervention_sample["executor"]["current_state"] == "yyr"
    assert observer.samples[-1]["tls_state"] == "Yyr"
    intervention = observer.interventions[0]
    assert intervention["changed_indices"] == [0]
    assert intervention["after"] == {**intervention["before"], "current_state": "Yyr"}
    assert state["remaining_sec"] == 3.
    observer(stage="before_simulation_step", time_sec=27481, executors=executors)
    assert writes == ["Yyr"]
    with pytest.raises(ValueError, match="Only one"):
        observer(stage="before_simulation_step", time_sec=27480, executors=executors)
    assert writes == ["Yyr"]


def comparison_fixture():
    def sample(time, stage, remaining):
        return {"time_sec": time, "stage": stage, "tls_state": "yyr",
                "executor": {"mode": "yellow", "remaining_sec": remaining,
                             "current_state": "yyr", "current_green_state": "Ggr"},
                "vehicles": {"focal": {"position": 1.}}}

    samples = [sample(27479, "before_simulation_step", 3.),
               sample(27480, "after_executor_advance", 3.),
               sample(27480, "before_simulation_step", 3.),
               sample(27481, "after_simulation_step_before_executor", 3.),
               sample(27481, "after_executor_advance", 2.)]
    actual = deepcopy(samples)
    for row in actual[2:]:
        row["tls_state"] = "Yyr"
        row["executor"]["current_state"] = "Yyr"
        row["vehicles"]["focal"]["position"] = 2.
    observer = SimpleNamespace(samples=actual, before_intervention_sample=deepcopy(samples[2]),
        interventions=[{"time_sec": 27480}], passages={vehicle: {} for vehicle in counterfactual.replay.IDS})
    metrics = {"ok": True, "collision_events": 0, "collision_incidents": 0,
               "collision_event_steps": 0, "starting_teleports": 0, "ending_teleports": 0}
    inventory = {"native_event_count": 0, "native_incident_count": 0, "collision_event_steps": 0}
    steps = list(range(counterfactual.replay.BEGIN + 1, counterfactual.replay.END + 1))
    return {"samples": samples}, metrics, inventory, observer, steps


def test_native_leader_tuple_matches_retained_json_list_without_relaxing_values():
    reference, metrics, inventory, observer, steps = comparison_fixture()
    # libsumo getLeader returns a tuple; the retained JSON necessarily stores a list.
    for index in (0, 2):
        reference["samples"][index]["vehicles"]["focal"]["leader"] = ["lead_vehicle", 12.5]
        observer.samples[index]["vehicles"]["focal"]["leader"] = ("lead_vehicle", 12.5)
    observer.before_intervention_sample["vehicles"]["focal"]["leader"] = ("lead_vehicle", 12.5)
    checks, _ = counterfactual.counterfactual_checks(reference, metrics, inventory, observer, steps)
    assert all(checks.values())

    observer.samples[0]["vehicles"]["focal"]["leader"] = ("lead_vehicle", 12.5001)
    checks, _ = counterfactual.counterfactual_checks(reference, metrics, inventory, observer, steps)
    assert not checks["exact_original_preintervention_samples"]

    observer.samples[0]["vehicles"]["focal"]["leader"] = ("lead_vehicle", 12.5)
    observer.before_intervention_sample["vehicles"]["focal"]["leader"] = ("lead_vehicle", 12.5001)
    checks, _ = counterfactual.counterfactual_checks(reference, metrics, inventory, observer, steps)
    assert not checks["exact_original_preintervention_samples"]


@pytest.mark.parametrize("change,failed_check", [
    ("prefix", "exact_original_preintervention_samples"),
    ("timing", "same_signal_and_executor_timing"),
    ("collision", "zero_native_collisions"),
    ("passage", "both_focal_vehicles_passed"),
])
def test_counterfactual_fails_changed_prefix_timing_collision_or_missing_passage(change, failed_check):
    reference, metrics, inventory, observer, steps = comparison_fixture()
    checks, comparisons = counterfactual.counterfactual_checks(reference, metrics, inventory, observer, steps)
    assert all(checks.values())
    assert comparisons["timing_differences"] == []
    if change == "prefix":
        observer.samples[0]["vehicles"]["focal"]["position"] += 1.
    elif change == "timing":
        observer.samples[-1]["executor"]["remaining_sec"] += 1.
    elif change == "collision":
        metrics.update(collision_events=2, collision_incidents=1, collision_event_steps=2)
        inventory.update(native_event_count=2, native_incident_count=1, collision_event_steps=2)
    else:
        observer.passages.pop(counterfactual.replay.IDS[0])
    checks, _ = counterfactual.counterfactual_checks(reference, metrics, inventory, observer, steps)
    assert not checks[failed_check]
