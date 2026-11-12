"""Run the frozen stopped-queue acceleration swaps serially against archived high arms."""
import argparse
from collections import Counter
import json
from pathlib import Path
import traceback

from measure_swap import measure
from readout import read_pair


def save(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def bind_cases(plan, inputs):
    roster = plan["roster"]
    if (Counter(row["city"] for row in roster) != {"hangzhou": 9, "jinan": 15}
            or len({(r["city"], r["seed"], r["time_sec"]) for r in roster}) != 24
            or plan["budget"] != dict(maximum_prefix_replays=24, maximum_new_10s_branches=24,
                                       unchanged_arms_from_archive=24, fits=0, closed_loop_evaluations=0)):
        raise ValueError("Expected the frozen 9 Hangzhou and 15 Jinan high-arm swaps")
    archived = {}
    for city, n_pairs in (("hangzhou", 14), ("jinan", 28)):
        directory = Path(inputs[city]["archive_dir"])
        metadata = json.loads((directory / "result.json").read_text())
        pairs = json.loads((directory / "pairs.json").read_text())
        if metadata["status"] != "COMPLETE" or metadata["completed_pairs"] != n_pairs or len(pairs) != n_pairs:
            raise ValueError(f"Original {city} paired experiment is incomplete")
        for index, case in enumerate(pairs, 1):
            key = city, case["seed"], case["time_sec"]
            if key in archived:
                raise ValueError(f"Ambiguous archived checkpoint {key}")
            archived[key] = index, case, directory / f"case_{index:02d}" / "high"

    bound = []
    for row in roster:
        key = row["city"], row["seed"], row["time_sec"]
        index, case, original = archived[key]
        if (case["tls_id"] != row["tls_id"] or [case["right_near"], case["straight_near"]] != row["point"]
                or json.loads((original / "arm.json").read_text()) != case["high"]):
            raise ValueError(f"Archived high arm differs from the frozen case {key}")
        bound.append((row["city"], index, case, original))
    return bound


def run(plan, inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    receipt = dict(status="RUNNING", planned_pairs=24, completed_pairs=0, prefix_replays_started=0,
                   branches_started=0, branches_completed=0, reused_archived_arms=0,
                   fits=0, closed_loop_policy_evaluations=0)
    save(output / "pairs.json", rows)
    save(output / "result.json", receipt)

    def branch_started():
        receipt["branches_started"] += 1
        save(output / "result.json", receipt)

    try:
        for index, (city, old_index, case, original) in enumerate(bind_cases(plan, inputs), 1):
            cell = output / f"case_{index:02d}"
            cell.mkdir()
            receipt["last_attempt"] = dict(index=index, city=city, seed=case["seed"], time_sec=case["time_sec"])
            save(output / "result.json", receipt)
            try:
                measure(city, case, Path(inputs[city]["sumocfg"]), Path(inputs[city]["registry"]),
                        original, cell, on_branch_start=branch_started)
            finally:
                receipt["prefix_replays_started"] += int((cell / "prefix_started.json").is_file())
            receipt["branches_completed"] += 1
            row = read_pair(original, cell, dict(case, city=city))
            row.update(index=index, archive_case_index=old_index, point=[case["right_near"], case["straight_near"]])
            rows.append(row)
            receipt["completed_pairs"] = len(rows)
            receipt["reused_archived_arms"] = len(rows)
            save(output / "pairs.json", rows)
            save(output / "result.json", receipt)
            print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=index, total=24, unit="pairs")), flush=True)
        if any(receipt[k] != 24 for k in ("prefix_replays_started", "branches_started", "branches_completed",
                                         "completed_pairs", "reused_archived_arms")):
            raise ValueError("Incomplete frozen swap accounting")
        receipt["status"] = "COMPLETE"
    except Exception as exc:
        receipt.update(status="HOLD", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    save(output / "result.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("plan", "inputs", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    receipt = run(json.loads(args.plan.read_text()), json.loads(args.inputs.read_text()), args.output)
    print(json.dumps(receipt), flush=True)
    raise SystemExit(0 if receipt["status"] == "COMPLETE" else 2)
