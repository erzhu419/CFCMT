#!/usr/bin/env python3
"""Run v65 target-veto training only when the v66 executable freeze matches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_long_horizon_target_veto_freeze import (  # noqa: E402
    run_target_veto_freeze,
)
from scripts.cluster.audit_tsc_external_long_horizon_target_veto_cache import (  # noqa: E402
    AUDIT_DECISION as CACHE_AUDIT_DECISION,
    AUDIT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_training import (  # noqa: E402
    PROTOCOL,
)


RUN_PROTOCOL = "tsc-v66r62-external-v9-long-horizon-target-veto-training-run-v1"


def validate_training_freeze(
    *, training_protocol_path: Path, cache_audit_path: Path
) -> dict[str, Any]:
    training = json.loads(Path(training_protocol_path).read_text(encoding="utf-8"))
    cache_audit = json.loads(Path(cache_audit_path).read_text(encoding="utf-8"))
    source_gate = {
        str(relative): (PROJECT_ROOT / str(relative)).is_file()
        and _sha256(PROJECT_ROOT / str(relative)) == str(expected)
        for relative, expected in training.get("executable_sources", {}).items()
    }
    gate = {
        "training_protocol_matches": training.get("protocol") == PROTOCOL,
        "all_executable_source_hashes_match": bool(source_gate)
        and all(source_gate.values()),
        "cache_audit_protocol_matches": cache_audit.get("protocol")
        == CACHE_AUDIT_PROTOCOL,
        "cache_audit_passed": cache_audit.get("status") == "PASS"
        and cache_audit.get("decision") == CACHE_AUDIT_DECISION
        and bool(cache_audit.get("integrity_gate", {}).get("passed", False)),
        "cache_audit_path_matches": str(
            Path(training["required_cache_audit"]["future_path"])
        )
        == str(Path(cache_audit_path).resolve().relative_to(PROJECT_ROOT)),
    }
    gate["passed"] = all(gate.values())
    return {
        "training_protocol_sha256": _sha256(training_protocol_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "source_gate": source_gate,
        "gate": gate,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite target-veto training result: {args.out}")
    executable_freeze = validate_training_freeze(
        training_protocol_path=args.training_protocol,
        cache_audit_path=args.cache_audit,
    )
    if not bool(executable_freeze["gate"]["passed"]):
        raise ValueError(f"v66 executable training freeze failed: {executable_freeze}")
    payload = run_target_veto_freeze(
        cache_root=args.cache_root,
        cache_audit_path=args.cache_audit,
        expected_cache_audit_sha256=str(executable_freeze["cache_audit_sha256"]),
        protocol_path=args.protocol,
        parent_protocol_path=args.parent_protocol,
        freeze_audit_path=args.freeze_audit,
        freeze_root=args.freeze_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        output_root=args.output_root,
        workers=args.workers,
    )
    payload["run_protocol"] = RUN_PROTOCOL
    payload["executable_training_freeze"] = executable_freeze
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "active_cities": payload["active_cities"],
                "training_protocol_sha256": executable_freeze[
                    "training_protocol_sha256"
                ],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
