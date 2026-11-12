#!/usr/bin/env python3
"""Extract existing V93 JSON counters, or recompute manuscript summaries from CSV.

On the archive server:
  python3 recompute_execution_summaries.py --cohort confirmation_rollouts.csv --extract --out OUTPUT
From the supplied compact records (no archive or simulator required):
  python3 recompute_execution_summaries.py --out .
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

INCIDENT_FIELDS = (
    "collision_events", "collision_event_steps", "collision_incidents",
    "starting_teleports", "ending_teleports",
)
NON_GUARD_REJECTIONS = {"rejected_cooldown", "rejected_coordination", "rejected_executor"}
DISPLAY_NAMES = {
    "frozen_selector": "CFCMT", "target_only": "Target anchor",
    "equal_all_sources": "Equal source weights", "h2oplus": "Dense linear residual",
    "phase_pressure": "PhasePressure", "max_pressure": "MaxPressure",
    "exact_reference": "Exact generalized-pressure reference",
}
BASE_FIELDS = (
    "method", "city", "scenario", "seed", "result_path", "guard_audit_present",
    *INCIDENT_FIELDS, "accepted_trace_records", "accepted_trace_effective_records",
    "accepted_trace_at_limit",
)
DEFINITIONS = {
    "decisions": "Post-warm-up intersection-by-decision-epoch choices, excluding the 60-s warm-up.",
    "collision_events": "Sum of len(simulation.getCollisions()) over simulation steps: raw collision records, not a count of colliding vehicles. A continuing contact can contribute repeated records.",
    "collision_event_steps": "Number of simulation steps with at least one collision record.",
    "collision_incidents": "Within-run distinct (sorted collider/victim IDs, collision type, lane) keys.",
    "effective_overrides": "Executor-accepted non-reference switch requests plus effective holds of current green. An accepted request is not a completed green-phase change.",
    "guard_rejections": "All rejected_* counters except rejected_cooldown, rejected_coordination, and rejected_executor.",
    "accepted_trace": "Coordinator-approved proposals; may include ineffective requests and is capped at 256 per run. Rates use complete guard counters, not trace lengths.",
    "pressure_policies": "PP and MP have no learned guard audit; empty guard fields mean not applicable, not zero.",
    "accounting": "decisions = prior_agreements + guard_rejections + proposed_overrides; proposed_overrides = effective_overrides + rejected_cooldown + rejected_coordination + rejected_executor.",
}


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_records(cohort):
    """Read only scalar counters and the single effective Jinan-selector trace."""
    rows, special = [], []
    for identity in cohort:
        path = Path(identity["result_path"])
        result = json.loads(path.read_text())
        metrics = result["metrics"]
        guard = metrics.get("guard_audit") or {}
        trace = metrics.get("accepted_intervention_trace") or []
        row = {k: identity[k] for k in ("method", "city", "scenario", "seed", "result_path")}
        row.update({field: int(metrics[field]) for field in INCIDENT_FIELDS})
        row["guard_audit_present"] = int(bool(guard))
        row.update({"guard_" + k: v for k, v in guard.items()
                    if isinstance(v, int) and not isinstance(v, bool)})
        row.update(accepted_trace_records=len(trace),
                   accepted_trace_effective_records=sum(t.get("execution_effective") is True for t in trace),
                   accepted_trace_at_limit=int(len(trace) == 256))
        rows.append(row)
        if identity["method"] == "frozen_selector" and identity["city"] == "jinan" and guard.get("effective_overrides", 0):
            special.append({"scenario": identity["scenario"], "seed": int(identity["seed"]),
                            "result_path": str(path), "guard_audit": guard,
                            "accepted_intervention_trace": trace})
    return {"hostname": socket.gethostname(), "rows": rows, "jinan_effective_override_runs": special}


def save_export(export, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(export["rows"], key=lambda r: (r["method"], r["scenario"], int(r["seed"])))
    guard_fields = sorted({k for row in rows for k in row if k.startswith("guard_")} - {"guard_audit_present"})
    write_csv(out / "execution_rollouts.csv", rows, list(BASE_FIELDS) + guard_fields)
    special = {
        "source": "Existing frozen-selector result.json records; no replay or new simulation.",
        "counter_semantics": DEFINITIONS["effective_overrides"],
        "trace_semantics": DEFINITIONS["accepted_trace"],
        "runs": export["jinan_effective_override_runs"],
    }
    (out / "jinan_single_override_trace.json").write_text(json.dumps(special, indent=2) + "\n")
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "Read-only scalar extraction from existing archived result.json files; no simulation/replay and no model or raw log transfer.",
        "archive_hostname": export["hostname"], "rollout_count": len(rows),
        "identity": ["method", "scenario", "seed"],
        "cohort_source": "confirmation_rollouts.csv", "record_source": "result_path in each CSV row",
        "definitions": DEFINITIONS,
        "recompute": "python3 recompute_execution_summaries.py --out .",
    }
    (out / "execution_rollouts.metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def summarize(records, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(records)
    identities = [(r["method"], r["scenario"], int(r["seed"])) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate method/scenario/seed records")
    by_method, groups = defaultdict(list), defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
        groups[(row["method"], row["city"])].append(row)
    incidents, execution, audit_groups = [], [], []
    for method, runs in by_method.items():
        incidents.append({
            "method": method, "display_name": DISPLAY_NAMES[method], "rollout_count": len(runs),
            "collision_incidents_total": sum(int(r["collision_incidents"]) for r in runs),
            "collision_events_total": sum(int(r["collision_events"]) for r in runs),
            "rollouts_with_collision_incidents": sum(int(r["collision_incidents"]) > 0 for r in runs),
            "starting_teleports_total": sum(int(r["starting_teleports"]) for r in runs),
            "ending_teleports_total": sum(int(r["ending_teleports"]) for r in runs),
        })
    for (method, city), runs in groups.items():
        raw = Counter()
        for row in runs:
            guard = {k.removeprefix("guard_"): int(v) for k, v in row.items()
                     if k.startswith("guard_") and k != "guard_audit_present" and v != ""}
            if guard:
                rejected = sum(v for k, v in guard.items() if k.startswith("rejected_") and k not in NON_GUARD_REJECTIONS)
                assert guard["decisions"] == guard["prior_agreements"] + rejected + guard["proposed_overrides"]
                assert guard["effective_overrides"] == guard["executed_switch_overrides"] + guard["executed_stay_overrides"]
                assert guard["proposed_overrides"] == guard["effective_overrides"] + sum(guard[k] for k in NON_GUARD_REJECTIONS)
                raw.update(guard)
        audit_groups.append({
            "method": method, "city": city, "read_runs": len(runs),
            "incident_totals": {k: sum(int(r[k]) for r in runs) for k in INCIDENT_FIELDS},
            "guard_audit_runs": sum(int(r["guard_audit_present"]) for r in runs), "guard_totals": dict(raw),
        })
        if not raw:
            continue
        decisions = raw["decisions"]
        guard_rejections = sum(v for k, v in raw.items() if k.startswith("rejected_") and k not in NON_GUARD_REJECTIONS)
        item = {
            "method": method, "city": city, "rollout_count": len(runs),
            "post_warmup_intersection_decisions": decisions,
            "model_reference_agreements": raw["prior_agreements"],
            "model_reference_disagreements": decisions - raw["prior_agreements"],
            "guard_eligible_overrides": raw["proposed_overrides"],
            "guard_rejections_total": guard_rejections,
            "cooldown_rejections": raw["rejected_cooldown"],
            "spatial_rejections": raw["rejected_coordination"],
            "executor_rejections": raw["rejected_executor"],
            "effective_switch_overrides": raw["executed_switch_overrides"],
            "effective_stay_overrides": raw["executed_stay_overrides"],
            "effective_overrides": raw["effective_overrides"],
        }
        for name, count in [("effective_overrides", raw["effective_overrides"]),
                            ("guard_rejections", guard_rejections),
                            ("cooldown_rejections", raw["rejected_cooldown"]),
                            ("spatial_rejections", raw["rejected_coordination"]),
                            ("executor_rejections", raw["rejected_executor"])]:
            item[name + "_pct_of_decisions"] = count / decisions * 100
        execution.append(item)
    write_csv(out / "execution_summary.csv", execution)
    order = list(DISPLAY_NAMES)
    incidents.sort(key=lambda r: order.index(r["method"]))
    write_csv(out / "simulator_incident_summary.csv", incidents)
    audit = {
        "source": "execution_rollouts.csv", "operation": "Recomputed by recompute_execution_summaries.py; no simulator needed.",
        "expected_runs": len(rows), "missing_run_count": 0, "definitions": DEFINITIONS, "groups": audit_groups,
    }
    (out / "execution_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return {"rollouts": len(rows), "learned_rollouts": sum(r["rollout_count"] for r in execution),
            "execution_rows": len(execution), "incident_rows": len(incidents)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("confirmation_rollouts.csv"))
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--records", type=Path, default=Path(__file__).with_name("execution_rollouts.csv"), help="Compact per-run CSV used when not extracting the native archive")
    parser.add_argument("--extract", action="store_true", help="Read archived JSON paths; otherwise use supplied compact CSV")
    args = parser.parse_args()
    if args.extract:
        save_export(export_records(read_csv(args.cohort)), args.out)
    records = args.out / "execution_rollouts.csv" if args.extract else args.records
    print(json.dumps(summarize(records, args.out)))


if __name__ == "__main__":
    main()
