"""Stage selected city tables and small code; keep native transitions on the server."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P = json.loads((HERE / "protocol.json").read_text())
R = json.loads((HERE / "runtime.json").read_text())
sys.path.insert(0, str(Path(R["scheduler"]).parent))
import scheduler


def run_remote(command, timeout=45):
    rc, stdout, stderr = scheduler.run_on("node007", command, timeout=timeout, check=False)
    if rc:
        raise RuntimeError(f"Remote command failed ({rc}): {stderr[-1500:]}")
    return stdout


def stage_code():
    run_remote("mkdir -p " + shlex.quote(R["remote_root"]))
    subprocess.run(["rsync", "-a", "--include=*.py", "--include=protocol.json",
                    "--include=runtime.json", "--include=training_protocol.json",
                    "--include=value_diagnostic_protocol.json",
                    "--include=dense_source_protocol.json",
                    "--include=source_dense_adaptation_protocol.json",
                    "--include=source_dense_continuation_protocol.json",
                    "--include=source_dense_real_only_protocol.json",
                    "--include=source_dense_adapted_eval_protocol.json",
                    "--include=source_dense_iql_continuation_protocol.json",
                    "--include=target_adapted_eval_protocol.json",
                    "--include=zero_hold_eval_protocol.json",
                    "--include=executed_action_eval_protocol.json",
                    "--include=fresh_confirmation_protocol.json",
                    "--include=projected_continuation_protocol.json",
                    "--include=fit_action_ranking_protocol.json",
                    "--include=iql_actor_continuation_protocol.json",
                    "--include=iql_backup_diagnostic_protocol.json",
                    "--include=entropy_free_continuation_protocol.json",
                    "--include=iql_execution_signal_protocol.json",
                    "--include=free_execution_actor_protocol.json",
                    "--include=native_action_replay_protocol.json",
                    "--include=native_actor_preference_protocol.json",
                    "--include=native_fixed_action_replay_protocol.json",
                    "--include=source_iql_config.json", "--exclude=*", "-e",
                    scheduler._ssh_rsync_shell_for_node("node007"), str(HERE) + "/",
                    scheduler._ssh_target_for_node("node007") + ":" + R["remote_root"] + "/"],
                   check=True, capture_output=True, text=True, timeout=60)
    print("Staged native experiment code and protocol only", flush=True)


def stage_inputs():
    city = ROOT / "H2Oplus/downloads/open_transit/halifax_transit/h2o_city_envs/HALIFAXTRANSIT_weekday_all_routes"
    selection = json.loads((HERE / "selection/selection_summary.json").read_text())
    destination = R["city_input_root"]
    run_remote("mkdir -p " + shlex.quote(destination + "/data"))
    paths = ["gtfs_city_manifest.json"]
    total = (city / "gtfs_city_manifest.json").stat().st_size
    for line in selection["selected_services"]:
        for name in P["asset_selection"]["required_tables"]:
            source = city / "data" / line["line_key"] / name
            paths.append(str(source.relative_to(city)))
            total += source.stat().st_size
    path_list = HERE / "selected_input_files.txt"
    path_list.write_text("\n".join(paths) + "\n")
    subprocess.run(["rsync", "-a", "--files-from=" + str(path_list), "-e",
                    scheduler._ssh_rsync_shell_for_node("node007"), str(city) + "/",
                    scheduler._ssh_target_for_node("node007") + ":" + destination + "/"],
                   check=True, capture_output=True, text=True, timeout=60)
    report = dict(selected_services=len(selection["selected_services"]), uploaded_bytes=total,
                  source_manifest=str(city / "gtfs_city_manifest.json"), destination=destination)
    (HERE / "input_staging.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def stage_timetables():
    source = HERE / "selection/weekday_timetables.json"
    subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node("node007"),
                    str(source), scheduler._ssh_target_for_node("node007") + ":" +
                    R["city_input_root"] + "/"], check=True, capture_output=True, text=True, timeout=60)
    print(json.dumps(dict(weekday_timetable_bytes=source.stat().st_size)), flush=True)


def prepare():
    python = shlex.quote(R["python"])
    base = R["remote_root"]
    print(run_remote(python + " " + shlex.quote(base + "/freeze_supplement.py")), flush=True)
    options = P["asset_selection"]
    command = [R["python"], base + "/asset_adapter.py", "--city-manifest",
               R["city_input_root"] + "/gtfs_city_manifest.json", "--out", R["assets_root"],
               "--start-hour", str(options["start_hour"]), "--service-hours", str(options["service_hours"]),
               "--episode-seconds", str(P["smoke"]["asset_episode_seconds"]),
               "--max-services", str(options["max_directional_services"]),
               "--demand-seed", str(P["smoke"]["demand_seed"]), "--netconvert", R["netconvert"]]
    command += ["--weekday-timetables", R["city_input_root"] + "/weekday_timetables.json"]
    print(run_remote(shlex.join(command), timeout=60), flush=True)


def prepare_data_assets():
    selection = P["asset_selection"]
    for case in P["collection_cases"]:
        folder = R["assets_root"] + f"/case{case['case']:02d}"
        command = [R["python"], R["remote_root"] + "/asset_adapter.py", "--city-manifest",
                   R["city_input_root"] + "/gtfs_city_manifest.json", "--out", folder,
                   "--start-hour", str(selection["start_hour"]), "--service-hours", str(selection["service_hours"]),
                   "--episode-seconds", str(case["episode_seconds"]), "--max-services",
                   str(selection["max_directional_services"]), "--demand-seed", str(case["demand_seed"]),
                   "--netconvert", R["netconvert"], "--weekday-timetables",
                   R["city_input_root"] + "/weekday_timetables.json"]
        print(run_remote(shlex.join(command), timeout=60), flush=True)


def jobs(data=False):
    family = P["name"] + ("/collection" if data else "/smoke")
    rows = []
    cases = P["collection_cases"] if data else [dict(method=method) for method in P["smoke"]["methods"]]
    for case in cases:
        method = case["method"]
        name = f"case{case['case']:02d}" if data else method
        assets = R["assets_root"] + ("/" + name if data else "")
        command = [R["python"], "-u", R["remote_root"] + "/collect_native.py", "--assets",
                   assets + "/asset_manifest.json", "--output", R["remote_root"] + "/runs/" + name]
        if data:
            command += ["--case", str(case["case"])]
        else:
            command += ["--method", method, "--episode-seconds", str(P["smoke"]["episode_seconds"])]
        env = dict(R["env"], THESIS_H2O_ROOT=R["frozen_snapshot"] + "/runtime/H2Oplus",
                   THESIS_SUMO_ROOT=R["assets_root"])
        rows.append(dict(project="CFCMT", signature=family + "/" + name,
                         description="Halifax native passenger-state " + name,
                         cwd=str(HERE), cmd=shlex.join(command), allowed_nodes=R["nodes"],
                         cpu=1, ram_mb=2500, vram=0, env=env, resource_family=family,
                         ram_resource_family=family, vram_resource_family=family,
                         skip_launch_staging=True, allow_no_ckpt=True, allow_no_resume=True))
    return rows


def submit(data=False):
    batch = "collection" if data else "smoke"
    result_path = HERE / (batch + "_submission.json")
    if result_path.exists():
        raise ValueError("Native batch already submitted: " + str(result_path))
    path = HERE / (batch + "_jobs.json")
    path.write_text(json.dumps(jobs(data), indent=2) + "\n")
    result = json.loads(subprocess.check_output(["python3", R["scheduler"], "submit-jsonl",
                        "--file", str(path), "--trusted", "--json"], text=True))
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def collect():
    script = "import json; from pathlib import Path; "
    script += "root=Path(" + repr(R["remote_root"]) + "); methods=" + repr(P["smoke"]["methods"]) + "; "
    script += "rows={m:json.loads((root/'runs'/m/'summary.json').read_text()) for m in methods if (root/'runs'/m/'summary.json').exists()}; "
    script += "print(json.dumps(dict(complete=len(rows)==len(methods),ready=len(rows),expected=len(methods),summaries=rows)))"
    report = json.loads(run_remote(shlex.quote(R["python"]) + " -c " + shlex.quote(script)))
    report["experiment"] = P["name"]
    report["scope"] = P["stage"]
    (HERE / "smoke_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def collect_pool():
    script = "import json; from pathlib import Path; root=Path(" + repr(R["remote_root"]) + "); "
    script += "cases=" + repr([c["case"] for c in P["collection_cases"]]) + "; "
    script += "rows={i:json.loads((root/'runs'/f'case{i:02d}'/'summary.json').read_text()) for i in cases if (root/'runs'/f'case{i:02d}'/'summary.json').exists()}; "
    script += "print(json.dumps(dict(ready=len(rows),expected=len(cases),cases=[dict(case=i,complete=r['complete'],total_rows=r['total_rows']) for i,r in rows.items()])))"
    inventory = json.loads(run_remote(shlex.quote(R["python"]) + " -c " + shlex.quote(script)))
    if inventory["ready"] != inventory["expected"]:
        report = dict(inventory, complete=False, decision="PENDING")
        (HERE / "pool_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    print(run_remote(shlex.join([R["python"], R["remote_root"] + "/target_pool.py"]), timeout=60), flush=True)
    path = R["remote_root"] + "/pool_summary.json"
    command = shlex.quote(R["python"]) + " -c " + shlex.quote("from pathlib import Path; print(Path(" + repr(path) + ").read_text())")
    report = json.loads(run_remote(command))
    (HERE / "pool_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "stage", "stage-inputs", "stage-timetables", "prepare", "prepare-data", "smoke", "submit-data", "collect", "collect-pool"))
    mode = parser.parse_args().mode
    if mode == "plan":
        print(json.dumps(dict(smoke_jobs=len(jobs()), collection_jobs=len(jobs(True)), training_jobs=0, protocol=P["name"],
                             assets=R["assets_root"], target_unique_budget=P["planned_training"]["target_unique_transition_budget"])))
    elif mode == "stage":
        stage_code()
    elif mode == "stage-inputs":
        stage_inputs()
    elif mode == "stage-timetables":
        stage_timetables()
    elif mode == "prepare":
        prepare()
    elif mode == "prepare-data":
        prepare_data_assets()
    elif mode == "smoke":
        submit()
    elif mode == "submit-data":
        submit(True)
    elif mode == "collect-pool":
        collect_pool()
    else:
        collect()
