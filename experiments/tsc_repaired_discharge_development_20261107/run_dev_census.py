"""Bind the old Jinan measurement movements and seal the repaired-network development panel."""
import argparse
import json
from pathlib import Path

from audit_hangzhou import registry_row
from audit_static import compile_view, config, right_straight
from census_stopped import census, save


def load(path):
    return json.loads(Path(path).read_text())


def rebind(cfg, old_registry, folder):
    old_gate = load(old_registry.parent / "static_admission.json")
    new_gate = load(cfg.parent.parent / "static_admission.json")
    if (old_gate["status"] != "PASS_STATIC_ADMISSION" or new_gate["status"] != "PASS_STATIC_PRIORITY"
            or Path(new_gate["candidate_sumocfg"]) != cfg):
        raise ValueError("Missing the original registry or the repaired network's static admission")
    prior = compile_view(Path(old_gate["net"]))
    current = compile_view(config(cfg)[0])
    old = load(old_registry)
    if (old["protocol"] != "right-straight-same-lane-pairs-v1" or len(old["tls"]) != 12
            or any(len(rows) != 2 for rows in old["tls"].values())):
        raise ValueError("Expected the original 24 Jinan measurement movements")
    _, directed = right_straight(current)
    rebound = dict(protocol=old["protocol"], tls={})
    bindings = []
    for tls, rows in sorted(old["tls"].items()):
        rebound["tls"][tls] = []
        for row in rows:
            ids = row["right_id"], row["straight_id"]
            if registry_row(prior, tls, *ids) != row:
                raise ValueError(f"Original physical registry mismatch at {ids}")
            new_row = registry_row(current, tls, *ids)
            keys_by_side = []
            for side in ("right", "straight"):
                old_keys = sorted(k for k in prior["dirs"]
                                  if k[0] == tls and prior["signal_index"][k] in row[side + "_links"])
                new_keys = sorted(k for k in current["dirs"]
                                  if k[0] == tls and current["signal_index"][k] in new_row[side + "_links"])
                if old_keys != new_keys or len(old_keys) != 3:
                    raise ValueError(f"Lane connections changed for {row[side + '_id']}")
                keys_by_side.append(new_keys)
                for k in new_keys:
                    bindings.append(dict(movement=row[side + "_id"], lanes=k[1:],
                                         old_signal=prior["signal_index"][k], signal=current["signal_index"][k],
                                         old_request=prior["requests"][k][0], request=current["requests"][k][0]))
            if not all(directed.get((r, s), False) for r in keys_by_side[0] for s in keys_by_side[1]):
                raise ValueError(f"Registered 3-by-3 lane pairing lacks the intended priority at {ids}")
            rebound["tls"][tls].append(new_row)
    registry_file = folder / "target_registry.json"
    save(registry_file, rebound)
    save(folder / "target_link_bindings.json", bindings)
    save(folder / "static_admission.json", dict(
        status="PASS_STATIC_ADMISSION", sumocfg=str(cfg), original_registry=str(old_registry),
        network_admission=str(cfg.parent.parent / "static_admission.json"), measurement_registry=str(registry_file),
        tls=len(rebound["tls"]), directed_pairs=sum(map(len, rebound["tls"].values())),
        lane_bindings=len(bindings), remapped_signal_indices=sum(r["signal"] != r["old_signal"] for r in bindings),
        remapped_request_indices=sum(r["request"] != r["old_request"] for r in bindings)))
    return rebound, registry_file


def complete_panel(cases, seeds, points):
    expected = [(s, p) for s in seeds for p in points]
    return ([(r["seed"], tuple(r["point"])) for r in cases] == expected
            and len({(r["seed"], r["time_sec"]) for r in cases}) == len(cases)
            and all(Path(r["start_file"]).is_file() for r in cases))


def run(plan_file, old_registry, source_census, output):
    plan = load(plan_file)
    points = sorted(tuple(p) for p in plan["points_right_straight"])
    target_seeds = plan["target_development_seeds"]
    target_cfg = Path(plan["target_cfg"])
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="INCOMPLETE", pp_hours_attempted=0, pp_hours_started=0, pp_hours_complete=0,
                  target_seeds=target_seeds, per_seed=[], source_cells=0, target_cells=0, high_branches=0,
                  fits=0, policy_comparisons=0, roster_frozen=False)
    try:
        source = load(source_census)
        if (len(points) != 6 or len(set(points)) != 6 or len(target_seeds) != 4 or len(set(target_seeds)) != 4
                or plan["source_roster"] != str(source_census) + "#observed_cases"
                or source["status"] != "PASS_SOURCE_PP" or source["alternative_branches"] != 0
                or source["sumocfg"] != plan["source_cfg"] or source["rebound_registry"] != plan["source_registry"]
                or not complete_panel(source["observed_cases"], plan["source_seeds"], points)):
            raise ValueError("The archived source panel or target development design differs from the plan")
        result["source_cells"] = len(source["observed_cases"])
        registry, registry_file = rebind(target_cfg, old_registry, output)
        target_cases = []
        for seed in target_seeds:
            directory = output / f"pp_{seed}"
            result["pp_hours_attempted"] += 1
            try:
                receipt, selected = census(target_cfg, registry, directory, seed, points)
            except Exception:
                receipt_file = directory / "result.json"
                if receipt_file.exists():
                    receipt = load(receipt_file)
                    result["pp_hours_started"] += 1
                    result["per_seed"].append(receipt)
                    if any(receipt["safety"][k] for k in ("raw_collision_events", "starting_teleports",
                                                          "ending_teleports")):
                        result["status"] = "HOLD_CENSUS_SAFETY"
                raise
            result["per_seed"].append(receipt)
            result["pp_hours_started"] += 1
            result["pp_hours_complete"] += 1
            target_cases.extend(selected)
            result["target_cells"] = len(target_cases)
            if receipt["missing_points"]:
                result["status"] = "HOLD_DEVELOPMENT_COVERAGE"
                break
        else:
            if not complete_panel(target_cases, target_seeds, points):
                raise ValueError("The target census does not yield a complete ordered checkpoint panel")
            roster = dict(status="PASS_DEV_ROSTER", source_cfg=plan["source_cfg"], target_cfg=str(target_cfg),
                          source_registry=plan["source_registry"], target_registry=str(registry_file),
                          source_cases=source["observed_cases"], target_cases=target_cases)
            save(output / "roster.json", roster)
            result.update(status="PASS_DEV_ROSTER", roster_frozen=True, roster=str(output / "roster.json"))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    save(output / "result.json", result)
    print(json.dumps({k:result[k] for k in ("status", "pp_hours_started", "target_cells", "source_cells")}),
          flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "old-registry", "source-census", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan.resolve(), args.old_registry.resolve(), args.source_census.resolve(), args.output.resolve())
    raise SystemExit(0 if result["status"] in ("PASS_DEV_ROSTER", "HOLD_DEVELOPMENT_COVERAGE") else 2)
