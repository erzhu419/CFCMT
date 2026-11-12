"""Record preaction supply and feasible phase pairs on one frozen Jinan PP hour."""
import argparse
import json
from pathlib import Path

from source_pilot import observe


SEEDS = (33061101, 33061102, 33061103, 33061104)
CONTRAST_FIELDS = ("gap_s", "high_candidate", "low_candidate", "high_exposure_s",
                   "low_exposure_s", "right_sequence", "high_straight_sequence",
                   "low_straight_sequence")


def pair_rows(api, executor, pairs, seed, epoch):
    candidates = executor.feasible_states_now()
    eligible = len(candidates) == 8
    # Count supply even during clearance or minimum green, so a continuous
    # both-supplied visit is not split by the absence of a choice at one epoch.
    sample = observe(api, executor, pairs if eligible else (), candidates if eligible else ())
    near = sample["near_by_inlet"]
    contrasts = {p["registry_index"]: p["contrast"] for p in sample["pairs"]}
    for index, pair in enumerate(pairs):
        contrast = contrasts[index] if eligible else None
        row = dict(seed=seed, time_sec=epoch, tls_id=executor.tls_id,
                   right_id=pair["right_id"], straight_id=pair["straight_id"],
                   right_near=sum(near[lane] for lane in pair["right_lanes"]),
                   straight_near=sum(near[lane] for lane in pair["straight_lanes"]),
                   eight_eligible=eligible)
        row.update({field: contrast[field] if contrast else None for field in CONTRAST_FIELDS})
        yield row


def run(cfg, registry_path, output, seed):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import libsumo_version, load_libsumo

    admission = json.loads((registry_path.parent / "static_admission.json").read_text())
    registry = json.loads(registry_path.read_text())
    pairs = registry["tls"]
    tls_ids = sorted(pairs)
    # With one registered opponent per right, observe's (-gap, high, low)
    # tie break is identical to the historical target's (-gap, high, low, opponent).
    if (admission["status"] != "PASS_STATIC_ADMISSION" or Path(admission["sumocfg"]) != cfg
            or registry["protocol"] != "right-straight-same-lane-pairs-v1"
            or len(tls_ids) != 12 or any(len(ps) != 2 or len({p["right_id"] for p in ps}) != 2
                                       for ps in pairs.values())):
        raise ValueError("Expected twelve admitted Jinan signals with two distinct right-turn pairs each")
    api = load_libsumo()
    version = libsumo_version(api)
    if version != "1.22.0":
        raise ValueError(f"Expected SUMO 1.22.0, got {version}")
    output.mkdir(parents=True, exist_ok=False)
    ledger = v3._new_safety_ledger()
    receipt = dict(protocol="jinan-fresh-preaction-opportunity-v1", status="INVALID",
                   seed=seed, controller="pp", sumocfg=str(cfg), registry=str(registry_path),
                   sumo_version=version, trajectories_started=0, horizon_sec=3600,
                   scheduled_tls_states=4320, observed_tls_states=0, movement_rows=0,
                   eight_eligible_tls_states=0, new_alternative_branches=0, new_fits=0)
    try:
        v3._start_sumo(api, cfg, seed)
        receipt["trajectories_started"] = 1
        infos = v3._tls_phase_infos(api)
        graph = v3.build_signal_routing_graph(api, infos)
        context = v3._scenario_context_v3(sumocfg=cfg, infos=infos, control_interval_sec=10)
        executors = v3.build_safe_phase_executors(api, infos)
        if set(executors) != set(pairs) or any(len(e.feasible_states) != 8 for e in executors.values()):
            raise ValueError("Runtime signal IDs or stable phase counts differ from the Jinan registry")
        with (output / "rows.jsonl").open("x") as stream:
            for epoch in range(0, 3600, 10):
                if api.simulation.getTime() != epoch:
                    raise ValueError(f"Native time differs from frozen decision epoch {epoch}")
                for tls in tls_ids:
                    rows = list(pair_rows(api, executors[tls], pairs[tls], seed, epoch))
                    for row in rows:
                        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                    receipt["movement_rows"] += len(rows)
                    receipt["observed_tls_states"] += 1
                    receipt["eight_eligible_tls_states"] += int(rows[0]["eight_eligible"])
                states = v3._read_graph_states_v3(api, infos, context, graph)
                actions = {t: v3._rule_candidate(states[t], executors[t], "phase_pressure").state
                           for t in tls_ids}
                v3._advance_with_actions(sumo_api=api, executors=executors, actions=actions,
                                         control_interval_sec=10, collision_mode="fail", safety_ledger=ledger)
                if (epoch + 10) % 600 == 0:
                    print("SCHEDULEURM_PROGRESS " + json.dumps(
                        dict(seed=seed, current=epoch + 10, total=3600, unit="s")), flush=True)
        if (api.simulation.getTime() != 3600 or receipt["movement_rows"] != 8640
                or receipt["observed_tls_states"] != receipt["scheduled_tls_states"]):
            raise ValueError("Incomplete one-hour Jinan census")
        receipt["status"] = "COMPLETE"
    except v3.UnsafeRolloutError as exc:
        receipt.update(status="HOLD_SAFETY", error=str(exc))
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        receipt["final_time_sec"] = api.simulation.getTime() if receipt["trajectories_started"] else None
        if receipt["trajectories_started"]:
            api.close()
        receipt["safety"] = v3._finalize_safety_ledger(ledger)
        (output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumocfg", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.sumocfg.resolve(), args.registry.resolve(), args.output, args.seed)
    print(json.dumps(result), flush=True)
    raise SystemExit({"COMPLETE": 0, "HOLD_SAFETY": 2, "INVALID": 1}[result["status"]])
