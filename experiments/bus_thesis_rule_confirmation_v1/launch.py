"""Stage small code, launch ten CPU rule days, and retrieve one JSON summary."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from rule_common import HERE, P, SNAPSHOT, write

RUNTIME = json.loads((HERE / "runtime.json").read_text())
sys.path.insert(0, str(Path(RUNTIME["scheduler"]).parent))
import scheduler


def remote(path=HERE):
    return str(Path(RUNTIME["remote_root"]) / Path(path).relative_to(HERE))


def environment():
    snapshot = remote(SNAPSHOT)
    return dict(RUNTIME["env"], THESIS_H2O_ROOT=snapshot+"/runtime/H2Oplus",
        THESIS_SUMO_ROOT=snapshot+"/assets/SUMO_ruiguang", THESIS_DATASET=snapshot+"/contract.h5",
        SUMO_HOME=snapshot+"/assets/SUMO_ruiguang")


def jobs(smoke=False):
    family = P["name"] + ("/smoke" if smoke else "/eval")
    rows = [dict(project="CFCMT", description=f"Fixed forward rule case{case} {'smoke' if smoke else 'evaluation'}",
        signature=family+f"/case{case}", cwd=str(HERE),
        cmd=f"{shlex.quote(RUNTIME['python'])} -u {shlex.quote(remote()+'/run.py')} --case {case}" + (" --smoke" if smoke else ""),
        allowed_nodes=P["training_nodes"], cpu=1, ram_mb=2500, vram=0,
        env=environment(), resource_family=family,
        ram_resource_family=family, vram_resource_family=family,
        skip_launch_staging=True, stage_excludes=["rules", "rule_smoke", "reference_snapshot", "thesis_snapshot", "previous_thesis_location"],
        allow_no_ckpt=True, allow_no_resume=True) for case in ([0, 5] if smoke else range(10))]
    if not smoke:
        rows.append(dict(rows[0], description="Frozen SUMO full-day duet runtime validation",
            signature=P["name"]+"/runtime_validation",
            cmd=f"{shlex.quote(RUNTIME['python'])} -u {shlex.quote(remote()+'/validate_runtime.py')}"))
    return rows


def stage():
    scheduler.run_on("node007", "mkdir -p " + shlex.quote(remote()), timeout=40, check=True)
    subprocess.run(["rsync", "-a", "--include=*.py", "--include=protocol.json", "--include=runtime.json", "--exclude=*",
        "-e", scheduler._ssh_rsync_shell_for_node("node007"), str(HERE)+"/",
        scheduler._ssh_target_for_node("node007")+":"+remote()+"/"], check=True)
    print("Staged CFCMT control code/protocol/runtime only; server snapshots are created separately once", flush=True)


def collect(smoke=False):
    prefix = "smoke_" if smoke else ""
    for node in ("node007", "node004", "node005"):
        try:
            command = shlex.quote(RUNTIME["python"])+" -u "+shlex.quote(remote()+"/collect_results.py")
            scheduler.run_on(node, command + (" --smoke" if smoke else ""), timeout=60, check=True)
            source = scheduler._ssh_target_for_node(node)+":"+remote()+"/"+prefix+"analysis_summary.json"
            subprocess.run(["rsync", "-az", "-e", scheduler._ssh_rsync_shell_for_node(node),
                source, str(HERE)+"/"], check=True, timeout=60, capture_output=True, text=True)
            report = json.loads((HERE / (prefix+"analysis_summary.json")).read_text())
            assert report["experiment"] == P["name"] and report["analysis_location"] == "server"
            return report
        except (RuntimeError, subprocess.SubprocessError) as error:
            if node == "node005":
                raise
            print(f"JSON collection via {node} unavailable: {str(error)[-180:]}", flush=True)


def submit(smoke=False):
    name = "smoke" if smoke else "campaign"
    output = HERE / (name+"_submission.json")
    if output.exists():
        raise ValueError("Submission already recorded: "+str(output))
    path = HERE / (name+"_jobs.json")
    write(path, jobs(smoke))
    result = json.loads(subprocess.check_output(["python3", RUNTIME["scheduler"], "submit-jsonl",
        "--file", str(path), "--trusted", "--json"], text=True))
    write(output, result)
    print(json.dumps(dict(batch=name, submitted=len(result["submitted"]))), flush=True)


def watch(smoke=False):
    previous = None
    deadline = time.monotonic() + 6*3600
    while time.monotonic() < deadline:
        report = collect(smoke)
        state = (report["rule_ready"], report["runtime_validation"]["decision"], report["complete"], report["decision"])
        if state != previous:
            print(json.dumps(dict(runtime_validation=report["runtime_validation"]["decision"],
                **{key: report[key] for key in ("complete", "rule_ready", "rule_expected", "decision")})), flush=True)
            previous = state
        if report["complete"]:
            return
        time.sleep(45)
    print(json.dumps(dict(decision="PENDING", reason="Six-hour collection deadline exceeded")), flush=True)
    raise TimeoutError("Fixed-rule confirmation remains PENDING after six hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "stage", "smoke", "submit", "collect", "watch", "watch-smoke"))
    mode = parser.parse_args().mode
    if mode == "plan":
        print(json.dumps(dict(new_training_jobs=0, full_evaluation_jobs=len(jobs()),
            strong_rule_days=10, runtime_validation_days=1,
            smoke_evaluation_jobs=len(jobs(True)), primary_resident_records=600,
            resources=dict(cpu=1, ram_mb=2500, vram=0)), indent=2))
    elif mode == "stage":
        stage()
    elif mode in ("smoke", "submit"):
        submit(mode == "smoke")
    elif mode.startswith("watch"):
        watch(mode == "watch-smoke")
    else:
        report = collect()
        print(json.dumps({key: report[key] for key in ("complete", "rule_ready", "rule_expected", "decision")}))
