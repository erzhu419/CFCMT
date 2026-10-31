"""Prepare and evaluate three new demand realizations with six fixed actors."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PREFIX = "fresh_confirmation"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())
R = json.loads((HERE / "runtime.json").read_text())


def prepare(p=None, prefix=PREFIX):
    from launch import run_remote
    p = P if p is None else p
    code = "from pathlib import Path; import json; root=Path(" + repr(R["assets_root"]) + "); "
    code += "print(json.dumps(sorted({json.loads(p.read_text())['demand_seed'] for p in root.rglob('asset_manifest.json')})))"
    existing = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    if set(existing) & {case["demand_seed"] for case in p["evaluation"]["cases"]}:
        raise ValueError("Fresh demand seeds already occur in server assets; do not regenerate or reuse them")
    settings = p["asset_selection"]
    for case in p["evaluation"]["cases"]:
        folder = R["assets_root"] + f"/{prefix}/case{case['case']:02d}"
        command = [R["python"], R["remote_root"] + "/asset_adapter.py", "--city-manifest",
            R["city_input_root"] + "/gtfs_city_manifest.json", "--out", folder,
            "--start-hour", str(settings["start_hour"]), "--service-hours", str(settings["service_hours"]),
            "--episode-seconds", str(p["evaluation"]["episode_seconds"]), "--max-services",
            str(settings["max_directional_services"]), "--demand-seed", str(case["demand_seed"]),
            "--netconvert", R["netconvert"], "--weekday-timetables", R["city_input_root"] + "/weekday_timetables.json"]
        print(run_remote(shlex.join(command), timeout=60), flush=True)
    code = "import sys,json; sys.path.insert(0," + repr(R["remote_root"]) + "); "
    code += "from pathlib import Path; from fresh_confirmation_launch import check_assets; "
    code += "p=json.loads(Path(" + repr(R["remote_root"] + "/" + prefix + "_protocol.json") + ").read_text()); "
    code += "print(json.dumps(check_assets(p," + repr(prefix) + ")))"
    report = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    report["existing_server_demand_seeds"] = existing
    (HERE / (prefix + "_assets_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def check_assets(p=None, prefix=PREFIX):
    p = P if p is None else p
    reference = Path(R["assets_root"]) / "test00"
    old = json.loads((reference / "asset_manifest.json").read_text())
    static_files = ["network_data/" + name for name in (
        "nodes.nod.xml", "edges.edg.xml", "connections.con.xml", "stops.add.xml", "a_sorted_busline_edge.xml")]
    static_files += ["routes/buses.rou.xml"]
    static_files += [f"online_control/initialize_obj/save_obj_{kind}.add.xml"
        for kind in ("lane", "stop", "signal", "line", "bus")]
    lines = lambda row: [{k:v for k,v in line.items() if k != "person_count"} for line in row["lines"]]
    results = []
    for case in p["evaluation"]["cases"]:
        folder = Path(R["assets_root"]) / prefix / f"case{case['case']:02d}"
        row = json.loads((folder / "asset_manifest.json").read_text())
        if row["demand_seed"] != case["demand_seed"]:
            raise ValueError("Fresh demand seed differs from the frozen case")
        for field in ("city", "episode_seconds", "start_hour", "service_hours", "passenger_arrival_end_seconds",
                "line_ids", "scheduled_service_date", "station_max_by_line", "line_id_to_canonical_index", "departure_count"):
            if row[field] != old[field]:
                raise ValueError("Fixed service assets changed: " + field)
        if lines(row) != lines(old) or any((folder/name).read_bytes() != (reference/name).read_bytes() for name in static_files):
            raise ValueError("Fresh assets changed pinned topology, speeds, service or demand intensity")
        results.append(dict(**case, departure_count=row["departure_count"], person_count=row["person_count"]))
    return dict(complete=True, static_assets_equal=True, line_and_trip_metadata_equal=True, cases=results,
        unchanged_reference=str(reference), interpretation="New random passengers; same pinned route/service and OD intensities")


def run(branch, seed):
    spec = P["branches"][branch] if branch != "strong_rule" else P["rule"]
    for case in P["evaluation"]["cases"]:
        output = HERE / PREFIX / branch
        command = [sys.executable, "-u", str(HERE / "evaluate_native.py"), "--method", spec["method"],
            "--evaluation-protocol", str(HERE / (PREFIX + "_protocol.json")), "--case", str(case["case"]),
            "--assets", R["assets_root"] + f"/{PREFIX}/case{case['case']:02d}/asset_manifest.json"]
        if seed is not None:
            output = output / f"seed{seed}"
            command += ["--seed", str(seed), "--actor-action-space", "executed", "--checkpoint",
                str(HERE / spec["artifact_folder"] / f"seed{seed}" / "policy.pt")]
        command += ["--output", str(output / f"case{case['case']:02d}")]
        subprocess.run(command, check=True)


def evaluate():
    from training_launch import job, submit
    assets = json.loads((HERE / (PREFIX + "_assets_summary.json")).read_text())
    if not assets["complete"] or not json.loads((HERE / "executed_action_eval_summary.json").read_text())["complete"]:
        raise ValueError("Fresh assets or fixed-interface prerequisite is incomplete")
    rows = []
    choices = [(branch, seed) for branch in P["branches"] for seed in P["training_seeds"]] + [("strong_rule", None)]
    for branch, seed in choices:
        command = [R["python"], "-u", R["remote_root"] + "/fresh_confirmation_launch.py", "run", "--branch", branch]
        if seed is not None:
            command += ["--seed", str(seed)]
        row = job(command, PREFIX + f"/{branch}/{seed}")
        row.update(signature=P["name"] + f"/{branch}/{seed}", resource_family=P["name"],
            description="Frozen actor/rule, three fresh demand cases " + branch + " " + str(seed))
        rows.append(row)
    submit(PREFIX, rows)


def collect():
    from launch import run_remote
    from fresh_confirmation_summary import summarize
    code = "import sys,json; sys.path.insert(0," + repr(R["remote_root"]) + "); "
    code += "from fresh_confirmation_summary import compact_remote; print(json.dumps(compact_remote()))"
    report = summarize(json.loads(run_remote(shlex.join([R["python"], "-c", code]))))
    (HERE / (PREFIX + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key:report[key] for key in ("complete", "ready", "expected")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "prepare", "evaluate", "run", "collect"))
    parser.add_argument("--branch", choices=(*P["branches"], "strong_rule"))
    parser.add_argument("--seed", type=int, choices=P["training_seeds"])
    args = parser.parse_args()
    if args.mode == "stage":
        from launch import stage_code
        stage_code()
    elif args.mode == "run":
        if args.branch is None or ((args.branch == "strong_rule") != (args.seed is None)):
            parser.error("A learned branch needs its fixed seed; the rule has no training seed")
        run(args.branch, args.seed)
    else:
        {"prepare":prepare, "evaluate":evaluate, "collect":collect}[args.mode]()
