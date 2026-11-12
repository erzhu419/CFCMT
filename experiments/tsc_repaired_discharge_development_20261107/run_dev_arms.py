"""Measure the frozen native Jinan and paired native/aligned Hangzhou development high arms."""
import argparse
import json
from pathlib import Path
import traceback
from xml.etree import ElementTree


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def bound_cases(plan, roster, census):
    if (roster["status"] != "PASS_DEV_ROSTER" or census["status"] != "PASS_DEV_ROSTER"
            or any(plan[k] != roster[k] for k in ("source_cfg", "source_registry", "target_cfg"))):
        raise ValueError("Development arms require both frozen rosters on the two repaired packages")
    points = sorted(tuple(p) for p in plan["points_right_straight"])
    schedule = []
    for role, seeds, variants in (("source", plan["source_seeds"], ("native", "aligned")),
                                  ("target", plan["target_development_seeds"], ("native",))):
        cases = roster[role + "_cases"]
        expected = [(s, p) for s in seeds for p in points]
        actual = [(c["seed"], tuple(c["point"])) for c in cases]
        if (actual != expected or len(set(actual)) != {"source": 12, "target": 24}[role]
                or len({(c["seed"], c["time_sec"]) for c in cases}) != len(cases)):
            raise ValueError(f"Incomplete or duplicate frozen {role} cases")
        schedule.extend((role, variant, c) for c in cases for variant in variants)
    return schedule


def hangzhou_measurement_cfg(cfg, registry, folder):
    # The frozen Hangzhou measurer loads its registry and admission beside the config.
    # Give it a private view of the exact repaired config, net, route and rebound registry.
    admission_path = registry.parent / "static_admission.json"
    gate = read(admission_path)
    if (gate["status"] != "PASS_STATIC_PRIORITY" or Path(gate["candidate_sumocfg"]) != cfg
            or (admission_path.parent / gate["registry"]).resolve() != registry):
        raise ValueError("Hangzhou measurement inputs differ from the admitted repair")
    config = ElementTree.parse(cfg)
    net = Path(config.find("input/net-file").get("value"))
    route = Path(config.find("input/route-files").get("value"))
    if net != Path(net.name) or not route.is_absolute() or not route.is_file():
        raise ValueError("Expected the repaired local network and bound absolute route file")
    folder.mkdir()
    (folder / cfg.name).symlink_to(cfg)
    (folder / net).symlink_to((cfg.parent / net).resolve(strict=True))
    (folder / "movement_pairs.json").symlink_to(registry)
    save(folder / "static_admission.json", dict(status="PASS_STATIC_ADMISSION",
         network_admission=str(admission_path), network_status=gate["status"],
         sumocfg=str(cfg), registry=str(registry)))
    return folder / cfg.name


def verify_leaders(api, start, case, native):
    lane, targets = case["straight_lane"], case["straight_target_ids"]
    length = api.lane.getLength(lane)
    limit = 10 * api.lane.getMaxSpeed(lane)
    vehicles = start["vehicles"]
    ranked = sorted((length - v["lane_position"], vid) for vid, v in vehicles.items()
                    if v["lane_id"] == lane and length - v["lane_position"] <= limit)
    if (len(ranked) != case["straight_near"] or len(targets) != 2
            or [vid for _, vid in ranked[:2]] != targets
            or any(vehicles[vid]["speed"] != 0 or api.vehicle.getAccel(vid) != native for vid in targets)):
        raise ValueError("Frozen straight-lane ranks, stopped condition or native acceleration differ")
    return length


def run(plan_path, roster_path, output):
    from cf_h2o.sumo_runtime import load_libsumo
    from measure_arm import measure_arm as measure_hangzhou
    from measure_target import measure_arm as measure_jinan
    from measure_swap import set_pair_accel
    from readout import arm_data

    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="RUNNING", planned_arms=48, prefix_replays_started=0, matching_starts=0,
                   branches_started=0, branches_completed=0, completed_arms=0, fits=0, policy_comparisons=0)
    rows = []
    save(output / "result.json", receipt)
    save(output / "observations.json", dict(rows=rows))
    try:
        plan, roster = read(plan_path), read(roster_path)
        schedule = bound_cases(plan, roster, read(roster_path.parent / "result.json"))
        hz_cfg = hangzhou_measurement_cfg(Path(roster["source_cfg"]), Path(roster["source_registry"]),
                                          output / "hangzhou_measurement_inputs")
        native_nominal = None
        for index, (role, variant, case) in enumerate(schedule, 1):
            cell = output / f"arm_{index:02d}"
            cell.mkdir()
            native = 2.6 if role == "source" else 2.0
            group = role + "_" + variant
            identity = {k: case[k] for k in ("seed", "time_sec", "tls_id", "point", "right_id", "straight_id",
                                             "straight_target_ids", "high_candidate")}
            receipt["last_attempt"] = dict(index=index, group=group, **identity)
            save(output / "result.json", receipt)
            reference = read(roster_path.parent / case["start_file"])
            before = receipt["prefix_replays_started"]
            details = {}
            lane, targets = case["straight_lane"], case["straight_target_ids"]

            def checkpoint(start):
                if start != reference:
                    keys = [k for k in start.keys() | reference.keys() if start.get(k) != reference.get(k)]
                    raise ValueError(f"Census and replay starts differ in {sorted(keys)}")
                contrast = start["supply"]["contrast"]
                if (any(contrast[k] != case[k] for k in ("right_sequence", "high_straight_sequence",
                                                         "low_straight_sequence", "high_candidate", "low_candidate"))
                        or case["point"] != [case["right_near"], case["straight_near"]]):
                    raise ValueError("The replayed action contrast differs from the sealed case")
                api = load_libsumo()
                length = verify_leaders(api, start, case, native)
                details.update(lane_length_m=length, native_accel_mps2=[native, native],
                               applied_accel_mps2=[native, native])
                receipt["matching_starts"] += 1
                if variant == "aligned":
                    swap = set_pair_accel(api, start["vehicles"], lane, length, case["straight_near"], native, 2.0)
                    details.update(applied_accel_mps2=swap["swapped_accel_mps2"],
                                   unaffected_vehicles=swap["unaffected_vehicles"])
                    save(cell / "intervention_server_only.json", swap)

            def branch_started():
                receipt["prefix_replays_started"] = before + 1
                receipt["branches_started"] += 1
                save(output / "result.json", receipt)

            try:
                if role == "source":
                    arm = measure_hangzhou(case, "high", hz_cfg, cell, checkpoint, branch_started)
                else:
                    arm = measure_jinan(case, "high", Path(roster["target_cfg"]), cell, checkpoint,
                                        branch_started, registry_path=roster["target_registry"])
            finally:
                receipt["prefix_replays_started"] = before + int((cell / "prefix_started.json").is_file())
            receipt["branches_completed"] += 1
            if (arm["right_nominal"] != case["right_sequence"] or
                    arm["straight_nominal"] != case["high_straight_sequence"]):
                raise ValueError("Measured nominal high sequence differs from the frozen case")
            metrics, raw = arm_data(cell, case, targets, lane, details["lane_length_m"])
            if role == "source":
                if variant == "native":
                    native_nominal = raw["nominal"]
                elif raw["nominal"] != native_nominal:
                    raise ValueError("Native and aligned Hangzhou high requests have different nominal plans")
            rows.append(dict(**identity, group=group, city="hangzhou" if role == "source" else "jinan",
                             arm=variant, straight_lane=lane, exposure_s=len(metrics["priority_seconds"]),
                             second_crossed=metrics["second"]["entry_s"] is not None, **details, **metrics))
            receipt["completed_arms"] = len(rows)
            save(output / "observations.json", dict(rows=rows))
            save(output / "result.json", receipt)
            print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=index, total=48, unit="arms")), flush=True)
        if any(receipt[k] != 48 for k in ("prefix_replays_started", "matching_starts", "branches_started",
                                          "branches_completed", "completed_arms")):
            raise ValueError("Development acquisition counts disagree with the fixed 48-arm budget")
        receipt["status"] = "PASS_DEV_ARMS"
    except Exception as exc:
        receipt.update(status="HOLD_DEV_ARMS", error=f"{type(exc).__name__}: {exc}")
        if hasattr(exc, "safety_audit"):
            receipt["safety"] = exc.safety_audit
        traceback.print_exc()
    save(output / "result.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "roster", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan.resolve(), args.roster.resolve(), args.output.resolve())
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "PASS_DEV_ARMS" else 2)
