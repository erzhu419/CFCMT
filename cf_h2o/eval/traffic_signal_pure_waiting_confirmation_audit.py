"""Audit and analyze the complete frozen V117 confirmation matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_pure_waiting_confirmation_freeze import (
    AUTHORIZATION_DECISION,
    FREEZE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_pure_waiting_confirmation_rollout import (
    RESULT_PROTOCOL as ROLLOUT_PROTOCOL,
    STATIC_INPUT_AUDIT_PROTOCOL,
)
from scripts.cluster.run_tsc_pure_waiting_confirmation_shard import (
    SHARD_PROTOCOL,
    shard_identities,
)


RESULT_PROTOCOL = "tsc-v117-pure-waiting-confirmation-audit-v2"
NODES = 6


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _bootstrap_interval(
    values: np.ndarray, *, replicates: int, seed: int
) -> tuple[float, float]:
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("V117 bootstrap requires at least two finite paired seeds")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(replicates), dtype=float)
    batch = 1000
    for start in range(0, int(replicates), batch):
        stop = min(start + batch, int(replicates))
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = np.mean(values[indices], axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def _one_sided_sign_p(improved: int, total: int) -> float:
    return float(
        sum(math.comb(total, value) for value in range(improved, total + 1))
        / (2**total)
    )


def _comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("V117 comparison lost its paired seed matrix")
    relative = (candidate - reference) / np.maximum(reference, 1e-12)
    lower, upper = _bootstrap_interval(relative, replicates=replicates, seed=seed)
    improved = int(np.sum(relative < 0.0))
    return {
        "seed_count": int(relative.size),
        "mean_relative_delta": float(np.mean(relative)),
        "median_relative_delta": float(np.median(relative)),
        "worst_seed_relative_delta": float(np.max(relative)),
        "improved_seed_count": improved,
        "improved_seed_fraction": float(improved / relative.size),
        "paired_bootstrap_95pct_interval": [lower, upper],
        "one_sided_exact_sign_test_p": _one_sided_sign_p(
            improved, int(relative.size)
        ),
        "seed_relative_deltas": relative.tolist(),
    }


def audit_confirmation(
    *, freeze_path: Path, expected_freeze_sha256: str, results_root: Path
) -> dict[str, Any]:
    freeze = _read_json(freeze_path)
    if (
        _sha256(freeze_path) != str(expected_freeze_sha256)
        or freeze.get("protocol") != FREEZE_PROTOCOL
        or freeze.get("decision") != AUTHORIZATION_DECISION
    ):
        raise ValueError("V117 audit freeze identity changed")
    active_arms = tuple(str(value) for value in freeze["active_arms"])
    seeds = tuple(int(value) for value in freeze["fresh_seeds"])
    expected = {(seed, arm) for seed in seeds for arm in active_arms}
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    errors = []
    shard_records = []
    for shard_index in range(NODES):
        summary_path = results_root / "_shards" / f"shard_{shard_index}.json"
        if not summary_path.is_file():
            errors.append(f"missing shard summary {shard_index}")
            continue
        summary = _read_json(summary_path)
        expected_shard = set(
            shard_identities(
                freeze, shard_index=shard_index, shard_count=NODES
            )
        )
        observed_shard = {
            (int(row["seed"]), str(row["arm"])) for row in summary.get("rows", ())
        }
        audit_path = Path(str(summary.get("static_input_audit_path", "")))
        audit = _read_json(audit_path) if audit_path.is_file() else {}
        if (
            summary.get("protocol") != SHARD_PROTOCOL
            or summary.get("freeze_sha256") != str(expected_freeze_sha256)
            or int(summary.get("shard_index", -1)) != shard_index
            or int(summary.get("shard_count", -1)) != NODES
            or observed_shard != expected_shard
            or not audit_path.is_file()
            or _sha256(audit_path)
            != str(summary.get("static_input_audit_sha256", ""))
            or audit.get("protocol") != STATIC_INPUT_AUDIT_PROTOCOL
            or audit.get("status") != "PASS"
            or audit.get("freeze_sha256") != str(expected_freeze_sha256)
        ):
            errors.append(f"invalid shard evidence {shard_index}")
            continue
        shard_records.append(
            {
                "shard_index": shard_index,
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": _sha256(summary_path),
                "hostname": summary["hostname"],
                "identity_count": len(expected_shard),
                "static_input_audit_path": str(audit_path),
                "static_input_audit_sha256": _sha256(audit_path),
            }
        )
        for seed, arm in expected_shard:
            root = results_root / f"seed_{seed}" / arm
            result_path = root / "result.json"
            tripinfo_path = root / "tripinfo.xml"
            if not result_path.is_file() or not tripinfo_path.is_file():
                errors.append(f"missing rollout {seed}/{arm}")
                continue
            result = _read_json(result_path)
            metrics = dict(result.get("metrics", {}))
            if (
                result.get("protocol") != ROLLOUT_PROTOCOL
                or result.get("freeze_sha256") != str(expected_freeze_sha256)
                or int(result.get("seed", -1)) != seed
                or result.get("arm") != arm
                or result.get("city") != freeze["deployment"]["city"]
                or result.get("scenario") != freeze["deployment"]["scenario"]
                or result.get("tripinfo_evidence", {}).get("sha256")
                != _sha256(tripinfo_path)
                or not bool(metrics.get("ok", False))
                or int(metrics.get("starting_teleports", -1)) != 0
                or int(metrics.get("ending_teleports", -1)) != 0
                or not math.isfinite(
                    float(metrics.get("mean_tripinfo_waiting_time", math.nan))
                )
            ):
                errors.append(f"invalid rollout {seed}/{arm}")
                continue
            rows[(seed, arm)] = result
    if set(rows) != expected:
        errors.append(
            f"complete matrix mismatch: observed={len(rows)} expected={len(expected)}"
        )
    if errors:
        return {
            "protocol": RESULT_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAIL",
            "decision": "block_v117_confirmation",
            "freeze_sha256": str(expected_freeze_sha256),
            "errors": errors,
            "observed_identity_count": len(rows),
            "expected_identity_count": len(expected),
            "shards": shard_records,
        }

    waiting = {
        arm: np.asarray(
            [
                float(rows[(seed, arm)]["metrics"]["mean_tripinfo_waiting_time"])
                for seed in seeds
            ],
            dtype=float,
        )
        for arm in active_arms
    }
    arm_summary = {}
    for arm in active_arms:
        arm_rows = [rows[(seed, arm)] for seed in seeds]
        guard_overrides = np.asarray(
            [
                int(row["metrics"].get("guard_audit", {}).get("effective_overrides", 0))
                for row in arm_rows
            ],
            dtype=int,
        )
        arm_summary[arm] = {
            "mean_waiting_time": float(np.mean(waiting[arm])),
            "median_waiting_time": float(np.median(waiting[arm])),
            "collision_incidents": int(
                sum(int(row["metrics"]["collision_incidents"]) for row in arm_rows)
            ),
            "starting_teleports": int(
                sum(int(row["metrics"]["starting_teleports"]) for row in arm_rows)
            ),
            "ending_teleports": int(
                sum(int(row["metrics"]["ending_teleports"]) for row in arm_rows)
            ),
            "active_seed_count": int(np.sum(guard_overrides > 0)),
            "active_seed_fraction": float(np.mean(guard_overrides > 0)),
            "effective_override_count": int(np.sum(guard_overrides)),
        }
    pairs = {
        "source_selected_b100_vs_h2oplus_dense_b100": (
            "source_selected_b100",
            "h2oplus_dense_b100",
        ),
        "source_selected_b100_vs_target_only_b100": (
            "source_selected_b100",
            "target_only_b100",
        ),
        "source_selected_b100_vs_phase_pressure": (
            "source_selected_b100",
            "phase_pressure",
        ),
        "source_selected_b0_vs_h2oplus_dense_b0": (
            "source_selected_b0",
            "h2oplus_dense_b0",
        ),
        "source_selected_b100_vs_source_selected_b0": (
            "source_selected_b100",
            "source_selected_b0",
        ),
        "h2oplus_dense_b100_vs_h2oplus_dense_b0": (
            "h2oplus_dense_b100",
            "h2oplus_dense_b0",
        ),
    }
    gate = dict(freeze["confirmation_gate"])
    comparisons = {}
    for index, (name, (candidate, reference)) in enumerate(pairs.items()):
        if candidate not in waiting or reference not in waiting:
            comparisons[name] = {
                "applicable": False,
                "reason": "selector_gate_deployed_phase_pressure_fallback",
            }
            continue
        comparisons[name] = {
            "applicable": True,
            "candidate": candidate,
            "reference": reference,
            **_comparison(
                waiting[candidate],
                waiting[reference],
                replicates=int(gate["bootstrap_replicates"]),
                seed=int(gate["bootstrap_seed"]) + index,
            ),
        }
    source_active = "source_selected_b100" in active_arms
    primary_gate = {
        "source_profile_authorized_by_v116": source_active,
        "complete_matrix": len(rows) == len(expected),
        "source_b100_active_seed_fraction": bool(
            source_active
            and arm_summary["source_selected_b100"]["active_seed_fraction"]
            >= float(gate["minimum_active_source_seed_fraction"])
        ),
        "source_b100_superior_to_h2oplus_b100": bool(
            source_active
            and comparisons[
                "source_selected_b100_vs_h2oplus_dense_b100"
            ]["paired_bootstrap_95pct_interval"][1]
            <= float(
                gate[
                    "maximum_95pct_upper_relative_delta_source_b100_vs_h2oplus_b100"
                ]
            )
        ),
        "source_b100_benefits_over_target_only_b100": bool(
            source_active
            and comparisons[
                "source_selected_b100_vs_target_only_b100"
            ]["paired_bootstrap_95pct_interval"][1]
            <= float(
                gate[
                    "maximum_95pct_upper_relative_delta_source_b100_vs_target_only_b100"
                ]
            )
        ),
        "source_b100_pressure_safety_noninferiority": bool(
            source_active
            and comparisons[
                "source_selected_b100_vs_phase_pressure"
            ]["paired_bootstrap_95pct_interval"][1]
            <= float(
                gate[
                    "maximum_95pct_upper_relative_delta_source_b100_vs_phase_pressure"
                ]
            )
        ),
        "source_b100_collision_incidents_not_above_phase": bool(
            source_active
            and arm_summary["source_selected_b100"]["collision_incidents"]
            <= arm_summary["phase_pressure"]["collision_incidents"]
        ),
        "source_b100_teleports_not_above_phase": bool(
            source_active
            and arm_summary["source_selected_b100"]["starting_teleports"]
            + arm_summary["source_selected_b100"]["ending_teleports"]
            <= arm_summary["phase_pressure"]["starting_teleports"]
            + arm_summary["phase_pressure"]["ending_teleports"]
        ),
    }
    passed = bool(all(primary_gate.values()))
    decision = (
        "authorize_v117_cross_city_source_benefit_claim"
        if passed
        else (
            "retain_phase_pressure_fallback_no_source_claim"
            if not source_active
            else "reject_v117_cross_city_source_benefit_claim"
        )
    )
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "decision": decision,
        "confirmation_gate_passed": passed,
        "freeze": {
            "path": str(Path(freeze_path).resolve()),
            "sha256": str(expected_freeze_sha256),
        },
        "claim_scope": freeze["claim_scope"],
        "seed_count": len(seeds),
        "active_arms": list(active_arms),
        "inactive_arm_fallbacks": freeze["inactive_arm_fallbacks"],
        "expected_identity_count": len(expected),
        "observed_identity_count": len(rows),
        "shards": shard_records,
        "arm_summary": arm_summary,
        "comparisons": comparisons,
        "primary_gate": primary_gate,
        "errors": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V117 confirmation audit")
    result = audit_confirmation(
        freeze_path=args.freeze,
        expected_freeze_sha256=args.freeze_sha256,
        results_root=args.results_root,
    )
    _atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "confirmation_gate_passed": result.get(
                    "confirmation_gate_passed", False
                ),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
