"""Run one registered fresh-seed comparison with the unchanged V43 evaluator."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ARGUMENTS = {
    "--offline-authorization": "offline_authorization_path",
    "--expected-offline-authorization-sha256": "expected_offline_authorization_sha256",
    "--joint-freeze-audit": "joint_freeze_audit_path",
    "--expected-joint-freeze-sha256": "expected_joint_freeze_sha256",
    "--freeze-root": "freeze_root", "--external-manifest": "external_manifest_path",
    "--conversion-root": "conversion_root",
    "--expected-external-manifest-sha256": "expected_external_manifest_sha256",
    "--conversion-manifest": "conversion_manifest_path",
    "--expected-conversion-manifest-sha256": "expected_conversion_manifest_sha256",
    "--expected-conversion-tree-sha256": "expected_conversion_tree_sha256",
    "--expected-sumo-version": "expected_sumo_version",
}
PATH_ARGUMENTS = {"offline_authorization_path", "joint_freeze_audit_path",
    "freeze_root", "external_manifest_path", "conversion_root", "conversion_manifest_path"}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identities(protocol):
    rows = [(city, scenario, seed, policy) for seed in protocol["seeds"]
        for city, scenarios in protocol["city_scenarios"].items() for scenario in scenarios
        for policy in protocol["policies"]]
    if len(rows) != protocol["rollout_count"] or len(set(rows)) != len(rows):
        raise ValueError("The registered rollout roster contains missing or repeated identities")
    return rows


def runtime_arguments(protocol):
    frozen = protocol["frozen_runtime_arguments"]
    kwargs = {name: Path(frozen[flag]) if name in PATH_ARGUMENTS else frozen[flag]
        for flag, name in ARGUMENTS.items()}
    kwargs["protocol_spec_path"] = HERE / "evaluation_protocol.json"
    return kwargs


def validate(result, identity):
    city, scenario, seed, policy = identity
    if (tuple(result.get(key) for key in ("city", "scenario", "seed", "policy")) != identity
            or result.get("sumo_version") != "1.22.0"
            or result.get("evaluation_horizon_sec") != 3600
            or not result.get("metrics", {}).get("ok", False)):
        raise RuntimeError(f"Invalid original V43 result for {city}/{scenario}/{seed}/{policy}")


def compact(result, protocol, index, result_path):
    return dict(complete=True, analysis_protocol=protocol["name"], run_index=index,
        **{key: result[key] for key in ("city", "scenario", "seed", "policy",
            "runtime_policy", "metrics", "sumo_version", "evaluation_horizon_sec")},
        source_policy=result["policy"],
        original_result_protocol=result["protocol"], selected_candidate=result.get("selected_candidate"),
        model_artifacts={name: {"sha256": result["model_artifacts"][name]["sha256"]}
            for name in ("method_model", "baseline_model")},
        prediction_horizon_sec=result["prediction_horizon_sec"],
        control_interval_sec=result["control_interval_sec"], warmup_sec=result["warmup_sec"],
        raw_result_path=str(result_path), new_model_refits=0, new_target_labels=0)


def run(index):
    protocol = json.loads((HERE / "protocol.json").read_text())
    roster = identities(protocol)
    if not 0 <= index < len(roster):
        raise ValueError("Rollout index is outside the frozen eighty-run matrix")
    identity = roster[index]
    city, scenario, seed, policy = identity
    raw = HERE / "raw" / scenario / f"seed_{seed}" / policy
    result_path, tripinfo_path = raw / "result.json", raw / "tripinfo.xml"
    summary_root = HERE / "summaries" / f"run_{index:03d}"
    error_path = summary_root / "error.json"
    started = time.monotonic()
    print(f"CFCMT_PROGRESS 0/1 unit=rollout index={index}", flush=True)
    try:
        if result_path.is_file() and tripinfo_path.is_file() and tripinfo_path.stat().st_size > 0:
            result = json.loads(result_path.read_text())
            validate(result, identity)
            reused = True
        else:
            if error_path.exists() or (raw.exists() and any(raw.iterdir())):
                raise RuntimeError("Partial or failed raw rollout preserved; explicit review required before rerunning")
            # The deployment wrapper's PYTHONPATH points only to the original V43 snapshot.
            from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import run_external_closed_loop_rollout
            raw.mkdir(parents=True, exist_ok=True)
            result = run_external_closed_loop_rollout(**runtime_arguments(protocol),
                scenario=scenario, seed=seed, policy=policy, tripinfo_out=tripinfo_path)
            validate(result, identity)
            if not tripinfo_path.is_file() or tripinfo_path.stat().st_size == 0:
                raise RuntimeError("Original evaluator returned without its tripinfo evidence")
            write(result_path, result)
            reused = False
        summary = dict(compact(result, protocol, index, result_path),
            reused_complete_raw_result=reused, elapsed_seconds=time.monotonic() - started)
        write(summary_root / "summary.json", summary)
    except Exception as error:
        if not error_path.exists():
            write(error_path, dict(complete=False, analysis_protocol=protocol["name"], run_index=index,
                city=city, scenario=scenario, seed=seed, policy=policy,
                error_type=type(error).__name__, error=str(error),
                created_at_utc=datetime.now(timezone.utc).isoformat()))
        raise
    print(f"CFCMT_PROGRESS 1/1 unit=rollout index={index} reused={int(reused)}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    run(parser.parse_args().index)
