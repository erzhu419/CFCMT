"""Run the existing forward proportional rule with the common early60 executor."""

import argparse
import json
import os

from rule_common import P, SNAPSHOT, case_spec, module, result_path, write


def rule_action(runtime, rules):
    def action(method, raw, config):
        event = runtime.current_event_v7(raw)
        fields = ("forward_headway_seconds", "backward_headway_seconds",
            "target_forward_headway_seconds", "target_backward_headway_seconds",
            "forward_bus_present", "backward_bus_present")
        values = [float(event[runtime.RAW_EVENT_STATE_INDEX[key]]) for key in fields]
        hold = rules.hold_seconds(method, *values)
        return runtime.np.asarray([runtime.hold_seconds_to_policy_action_v7(hold, config)], runtime.np.float32)
    return action


def evaluate_rule(case, out, horizon):
    os.environ["LIBSUMO_AS_TRACI"] = "quiet"
    runtime = module("rule_confirmation_runtime", SNAPSHOT / "attribution_v1/run.py")
    rules = module("rule_confirmation_fixed_rules", SNAPSHOT / "fixed_rules.py")
    runtime.rule_action = rule_action(runtime, rules)
    protocol = dict(P, policy_exports={}, training_seed=None,
        evaluation_horizon_seconds=horizon)
    (out / "results").mkdir(parents=True, exist_ok=True)
    cfg, objects = runtime.common.build_scenario(out, case["demand_scale"],
        case["demand_seed"], P["departure_jitter_seconds"])
    return runtime.evaluate((case, protocol, str(out), cfg, objects))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=range(10), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = result_path(args.case, args.smoke)
    if result.exists():
        return
    row = evaluate_rule(case_spec(args.case), result.parents[1],
        1200 if args.smoke else P["evaluation_horizon_seconds"])
    row.update(case=args.case, split="test", comparison_protocol=P["name"],
        deployment_hold_scale=P["deployment_hold_scale"], rule_gain=P["rule_gain"],
        rule_max_hold_seconds=P["rule_max_hold_seconds"], completion_guard=P["completion_guard"],
        smoke=args.smoke)
    write(result, row)
    print(json.dumps(dict(case=args.case, horizon_seconds=row["horizon_seconds"],
        cost_seconds=row["generalized_seconds_per_departed_passenger"])), flush=True)


if __name__ == "__main__":
    main()
