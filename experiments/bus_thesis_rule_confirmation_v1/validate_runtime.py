"""Check the frozen SUMO code against one existing full-day duet result."""

import json
import math

from rule_common import HERE, P, REFERENCE, case_spec, write
from run import evaluate_rule


def compare_runtime(row, control):
    metadata = ("method", "execution_layer", "demand_scale", "demand_seed", "sumo_seed", "horizon_seconds")
    cost_delta = row["generalized_seconds_per_departed_passenger"] - control["generalized_seconds_per_departed_passenger"]
    completion_delta = row["passengers"]["completion_rate"] - control["passengers"]["completion_rate"]
    checks = dict(metadata_matched=all(row[key] == control[key] for key in metadata),
        sumo_version_matched=row["sumo_version"][1] == control["sumo_version"][1] == P["sumo_version"],
        departed_passengers_matched=row["passengers"]["departed_passengers"] == control["passengers"]["departed_passengers"],
        cost_matched=math.isfinite(cost_delta) and abs(cost_delta) <= P["runtime_validation"]["cost_absolute_tolerance_seconds"],
        completion_matched=math.isfinite(completion_delta) and abs(completion_delta) <= P["runtime_validation"]["completion_absolute_tolerance"])
    return dict(experiment=P["name"], case=0, complete=True, valid=all(checks.values()),
        decision="VALID_RUNTIME" if all(checks.values()) else "INVALID_RUNTIME", checks=checks,
        cost_delta_seconds=cost_delta, completion_delta=completion_delta,
        departed_passengers=row["passengers"]["departed_passengers"],
        control_departed_passengers=control["passengers"]["departed_passengers"])


def main():
    output = HERE / "runtime_validation/results/validation.json"
    if output.exists():
        return
    control = json.loads((REFERENCE / "runtime_control.json").read_text())
    case = dict(case_spec(0), method="daganzo_duet")
    assert control["method"] == case["method"]
    assert control["horizon_seconds"] == P["evaluation_horizon_seconds"]
    assert all(control[key] == case[key] for key in ("execution_layer", "demand_scale", "demand_seed", "sumo_seed"))
    row = evaluate_rule(case, HERE / "runtime_validation", P["evaluation_horizon_seconds"])
    result = compare_runtime(row, control)
    write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
