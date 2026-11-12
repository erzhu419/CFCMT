#!/usr/bin/env python3
"""Admit the 833 original completions plus the 63 manually adjudicated recovery cells."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import analyze


HERE = Path(__file__).resolve().parent
REQUIRED_GUARD = (
    "decisions", "prior_agreements", "proposed_overrides", "rejected_confidence",
    "rejected_service_safety", "rejected_cooldown", "rejected_coordination", "rejected_executor",
    "accepted_overrides", "accepted_switch_overrides", "accepted_stay_overrides",
    "executed_switch_overrides", "executed_stay_overrides", "effective_overrides",
)


def identity(row):
    return row["method"], row["scenario"], int(row["seed"])


def read_json(path):
    return json.loads(Path(path).read_text())


def finalize(recovery_path):
    protocol = read_json(HERE / "protocol.json")
    manifest = read_json(HERE / "recovery_manifest.json")
    inputs, admitted = [], []
    for index in range(4):
        path = HERE / f"evaluation/_shards/shard_{index}.json"
        payload = read_json(path)
        expected_count = 180 if index == 0 else 179
        if not (payload["complete"] is True and payload["shard_index"] == index
                and payload["shard_count"] == 5 and payload["task_count"] == len(payload["rows"]) == expected_count):
            raise ValueError(f"Original completed shard {index} is incomplete or mismatched")
        inputs.append(dict(path=str(path), admitted_rows=len(payload["rows"]), execution_source=f"original_shard_{index}"))
        admitted.extend((row, f"original_shard_{index}") for row in payload["rows"])

    first_path = HERE / "evaluation/interrupted_shard_4_first_attempt.json"
    first = read_json(first_path)
    if first["task_id"] != "t213641" or len(first["completed"]) != 116:
        raise ValueError("Original interrupted shard must retain its 116 completed records")
    if {identity(r) for r in first["completed"]} != {identity(r) for r in manifest["reuse_completed"]}:
        raise ValueError("Original completed identities differ from the recovery manifest")
    inputs.append(dict(path=str(first_path), admitted_rows=116, execution_source="original_shard_4_first_attempt"))
    admitted.extend((row, "original_shard_4_first_attempt") for row in first["completed"])

    recovery = read_json(recovery_path)
    if not (recovery["complete"] is True and recovery["original_task_id"] == "t213641"
            and recovery["task_count"] == len(recovery["rows"]) == 63):
        raise ValueError("The manually adjudicated recovery must supply all 63 records")
    planned = {identity(row): row["execution_kind"] for row in manifest["execute"]}
    if {identity(row) for row in recovery["rows"]} != set(planned):
        raise ValueError("Recovery identities differ from the 63-cell manifest")
    for row in recovery["rows"]:
        if row["recovery_execution_kind"] != planned[identity(row)]:
            raise ValueError("Recovery retry/first-execution provenance differs from the manifest")
        admitted.append((row, "manual_recovery_attempt2_" + row["recovery_execution_kind"]))
    recovery_kinds = Counter(row["recovery_execution_kind"] for row in recovery["rows"])
    if recovery_kinds != {"technical_retry": 4, "first_execution": 59}:
        raise ValueError("Recovery must contain exactly four technical retries and 59 first executions")
    inputs.append(dict(path=str(recovery_path), admitted_rows=63, execution_source="manual_recovery_attempt2"))

    compact = []
    guard_counts, guard_runs = defaultdict(Counter), Counter()
    for row, source in admitted:
        item = {key: row[key] for key in ("method", "city", "scenario", "seed", "result_path")}
        item["execution_source"] = source
        item.update({key: row["metrics"][key] for key in analyze.METRICS})
        compact.append(item)
        if row["method"] == "frozen_selector":
            guard = row["guard_audit"]
            if any(not isinstance(guard.get(k), int) or isinstance(guard[k], bool) for k in REQUIRED_GUARD):
                raise ValueError(f"Incomplete integer guard counters in {identity(row)}")
            guard_counts[row["city"]].update({k: v for k, v in guard.items()
                                               if isinstance(v, int) and not isinstance(v, bool)})
            guard_runs[row["city"]] += 1
    # The preregistered matrix check rejects omissions, duplicates and substituted identities.
    analyze.index_rows(compact, protocol)
    compact.sort(key=lambda row: (row["scenario"], row["seed"], row["method"]))
    execution = {}
    for city, counts in guard_counts.items():
        disagreements = counts["decisions"] - counts["prior_agreements"]
        counts["model_reference_disagreements"] = disagreements
        rates = {f"{key}_pct_of_decisions": 100 * counts[key] / counts["decisions"]
                 for key in ("prior_agreements", "model_reference_disagreements", "rejected_confidence",
                             "rejected_service_safety", "accepted_overrides", "accepted_switch_overrides",
                             "accepted_stay_overrides", "executed_switch_overrides", "executed_stay_overrides",
                             "effective_overrides")}
        rates["confidence_rejection_pct_of_disagreements"] = (
            100 * counts["rejected_confidence"] / disagreements if disagreements else None)
        rates["service_rejection_pct_of_disagreements"] = (
            100 * counts["rejected_service_safety"] / disagreements if disagreements else None)
        execution[city] = dict(rollout_count=guard_runs[city], integer_counts=dict(counts), derived_rates=rates)

    csv_path = HERE / "evaluation/rollouts.csv"
    analysis_dir = HERE / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    analyze.write_csv(csv_path, compact)
    guard_receipt = dict(method="frozen_selector", by_city=execution,
                         count_semantics="Summed integer per-run counters only; derived rates use summed denominators. Accepted switches are requests, not completed phase transitions.",
                         sources=[entry["path"] for entry in inputs])
    (analysis_dir / "guard_execution.json").write_text(json.dumps(guard_receipt, indent=2) + "\n")
    receipt = dict(
        status="COMPLETE", collected_at_utc=datetime.now(timezone.utc).isoformat(),
        original_completed_rows=833, original_complete_shard_rows=717,
        interrupted_shard_original_completed_rows=116, admitted_recovery_rows=63,
        technical_retries=4, first_executions=59, final_rows=len(compact),
        excluded_automatic_relaunch_records=json.loads(
            (HERE / "unplanned_retry_exclusion.json").read_text())["completed"],
        excluded_automatic_relaunch=dict(node="jtl311linux", admitted_rows=0,
                                       stop_receipt=str(HERE / "unplanned_retry_stop.json"),
                                       reason="Entire unplanned automatic rerun excluded; no input from that execution is read"),
        identity_completeness="PASS: exactly the 896 unique method/scenario/seed identities in protocol.json, with all ten finite metrics",
        inputs=inputs, output_csv=str(csv_path), guard_summary=str(analysis_dir / "guard_execution.json"),
        provenance="Original input shards and first-attempt inventory retained; no combined JSON copy or historical-panel pooling")
    (HERE / "final_collection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, default=HERE / "recovery/_shards/recovery_4.json")
    args = parser.parse_args()
    result = finalize(args.recovery)
    print(json.dumps({key: result[key] for key in ("status", "original_completed_rows", "admitted_recovery_rows", "final_rows")}))


if __name__ == "__main__":
    main()
