"""Collect frozen independent PP trajectories; retain all started hours and compact receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


def collect(job, scripts, root):
    part, city, cfg, seed = job
    dst = root / part / city / str(seed)
    dst.parent.mkdir(exist_ok=True, parents=True)
    with (dst.parent / f"{seed}.log").open("x") as stream:
        proc = subprocess.run([sys.executable, str(scripts / "collect_hour.py"), "--city", city,
                               "--sumocfg", cfg, "--seed", str(seed), "--output", str(dst)],
                              stdout=stream, stderr=subprocess.STDOUT)
    result = dst / "result.json"
    row = json.loads(result.read_text()) if result.exists() else dict(status="INCOMPLETE")
    fields = ("status", "elapsed_sec", "started", "counts", "sensor", "safety", "error", "sensor_error")
    print(city, seed, row["status"], flush=True)
    return dict(part=part, city=city, seed=seed, returncode=proc.returncode,
                **{k: row[k] for k in fields if k in row})


def run(root, phase, workers):
    scripts = Path(__file__).resolve().parent
    cities = json.loads((scripts / "protocol.json").read_text())["cities"]
    if phase == "collection" and json.loads((root / "pilot/result.json").read_text())["status"] != "PASS_PILOT":
        raise ValueError("Native pilot not passed")
    parts = ["pilot"] if phase == "pilot" else ["donor", "adapt", "dev"]
    jobs = []
    for part in parts:
        by_city = {c: (sum(d[part], []) if part == "adapt" else d[part]) for c, d in cities.items()}
        for seeds in zip(*by_city.values()):
            jobs.extend((part, c, cities[c]["cfg"], s) for c, s in zip(by_city, seeds))
    rows = []
    for start in range(0, len(jobs), workers):
        batch = jobs[start:start+workers]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            handles = [pool.submit(collect, j, scripts, root) for j in batch]
            for f in as_completed(handles):
                rows.append(f.result())
        if any(r["status"] != "PASS_HOUR" for r in rows):
            break
    ok = len(rows) == len(jobs) and all(r["status"] == "PASS_HOUR" for r in rows)
    if phase == "pilot" and ok:
        for city in cities:
            city_rows = [r for r in rows if r["city"] == city]
            ok = ok and sum(r["sensor"]["counts_by_status"]["straight"] for r in city_rows) > 0
            ok = ok and sum(r["counts"]["opportunity_epochs"] for r in city_rows) > 0
    status = ("PASS_" if ok else "HOLD_") + phase.upper()
    dst = root / phase / "result.json"
    dst.parent.mkdir(exist_ok=True, parents=True)
    dst.write_text(json.dumps(dict(status=status, hours=sorted(rows, key=lambda r:(r["city"],r["seed"])),
                                   scheduled=len(jobs)), indent=2) + "\n")
    print(status, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pilot", "collection"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    run(args.root, args.phase, args.workers)
