"""Observe the admitted Hangzhou package throughout one frozen PP trajectory."""
import argparse
import json
from pathlib import Path

from source_pilot import TLS_IDS, observe


SEEDS = (32061101, 32061102)


def pair_rows(api, executor, pairs, seed, epoch):
    candidates = executor.feasible_states_now()
    eligible = len(candidates) == 8
    # Every epoch uses the pilot's physical supply observation. At busy or
    # minimum-green epochs there is no eight-action comparison to project.
    sample = observe(api, executor, pairs if eligible else (), candidates if eligible else ())
    contrasts = {p["registry_index"]: p["contrast"] for p in sample["pairs"]}
    near = sample["near_by_inlet"]
    for index, pair in enumerate(pairs):
        contrast = contrasts[index] if eligible else None
        yield dict(seed=seed, time_sec=epoch, tls_id=executor.tls_id,
                   right_id=pair["right_id"], straight_id=pair["straight_id"],
                   right_near=sum(near[lane] for lane in set(pair["right_lanes"])),
                   straight_near=sum(near[lane] for lane in set(pair["straight_lanes"])),
                   eight_eligible=eligible, gap_s=contrast["gap_s"] if contrast else None,
                   high_exposure_s=contrast["high_exposure_s"] if contrast else None,
                   low_exposure_s=contrast["low_exposure_s"] if contrast else None,
                   high_candidate=contrast["high_candidate"] if contrast else None,
                   low_candidate=contrast["low_candidate"] if contrast else None)


def run(cfg, output, seed):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import libsumo_version, load_libsumo

    admission = json.loads((cfg.parent / "static_admission.json").read_text())
    registry = json.loads((cfg.parent / "movement_pairs.json").read_text())
    pairs = registry["tls"]
    if (admission["status"] != "PASS_STATIC_ADMISSION"
            or registry["protocol"] != "right-straight-same-lane-pairs-v1"
            or set(pairs) != set(TLS_IDS) or any(len(p) != 2 for p in pairs.values())):
        raise ValueError("Frozen pilot requires the 32 admitted pairs on the 16 Hangzhou signals")
    api = load_libsumo()
    version = libsumo_version(api)
    if version != "1.22.0":
        raise ValueError(f"Expected SUMO 1.22.0, got {version}")
    output.mkdir(parents=True, exist_ok=False)
    ledger = v3._new_safety_ledger()
    receipt = dict(protocol="hangzhou-full-hour-supply-v1", seed=seed,
                   status="INVALID", controller="pp", sumocfg=str(cfg), sumo_version=version,
                   trajectories_started=0, horizon_sec=3600, scheduled_tls_states=5760,
                   observed_tls_states=0, eight_eligible_tls_states=0, movement_rows=0,
                   new_alternative_branches=0, new_fits=0)
    try:
        v3._start_sumo(api, cfg, seed)
        receipt["trajectories_started"] = 1
        infos = v3._tls_phase_infos(api)
        graph = v3.build_signal_routing_graph(api, infos)
        context = v3._scenario_context_v3(sumocfg=cfg, infos=infos, control_interval_sec=10)
        executors = v3.build_safe_phase_executors(api, infos)
        if set(executors) != set(TLS_IDS) or any(len(e.feasible_states) != 8 for e in executors.values()):
            raise ValueError("Runtime signals or stable programs differ from the admitted package")
        with (output / "rows.jsonl").open("x") as rows:
            for epoch in range(0, 3600, 10):
                if api.simulation.getTime() != epoch:
                    raise ValueError(f"Native clock differs from planned sample time {epoch}")
                for tls in TLS_IDS:
                    measured = list(pair_rows(api, executors[tls], pairs[tls], seed, epoch))
                    for row in measured:
                        rows.write(json.dumps(row, separators=(",", ":")) + "\n")
                    receipt["movement_rows"] += len(measured)
                    receipt["observed_tls_states"] += 1
                    receipt["eight_eligible_tls_states"] += int(measured[0]["eight_eligible"])
                observed = v3._read_graph_states_v3(api, infos, context, graph)
                actions = {t: v3._rule_candidate(observed[t], executors[t], "phase_pressure").state
                           for t in TLS_IDS}
                v3._advance_with_actions(sumo_api=api, executors=executors, actions=actions,
                                         control_interval_sec=10, collision_mode="fail", safety_ledger=ledger)
                if (epoch + 10) % 600 == 0:
                    print("SCHEDULEURM_PROGRESS " + json.dumps(
                        dict(seed=seed, current=epoch + 10, total=3600, unit="s")), flush=True)
        if (api.simulation.getTime() != 3600 or receipt["movement_rows"] != 11520
                or receipt["observed_tls_states"] != receipt["scheduled_tls_states"]):
            raise ValueError("Incomplete frozen one-hour sweep")
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    args = parser.parse_args()
    result = run(args.sumocfg.resolve(), args.output, args.seed)
    print(json.dumps(result), flush=True)
    raise SystemExit({"COMPLETE": 0, "HOLD_SAFETY": 2, "INVALID": 1}[result["status"]])
