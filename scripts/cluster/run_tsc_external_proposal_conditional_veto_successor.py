#!/usr/bin/env python3
"""Run the v49 proposal-conditional successor under its executable freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_external_proposal_conditional_veto_freeze import (  # noqa: E402
    run_proposal_conditional_veto_freeze,
)
from scripts.cluster.freeze_tsc_external_v9_proposal_conditional_veto_successor import (  # noqa: E402
    PROTOCOL,
)


RUN_PROTOCOL = "tsc-v67r63-proposal-conditional-veto-successor-run-v1"


def validate_executable_freeze(protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    source_gate = {
        str(relative): (PROJECT_ROOT / str(relative)).is_file()
        and _sha256(PROJECT_ROOT / str(relative)) == str(expected)
        for relative, expected in protocol.get("executable_sources", {}).items()
    }
    gate = {
        "protocol_matches": protocol.get("protocol") == PROTOCOL,
        "all_executable_source_hashes_match": bool(source_gate)
        and all(source_gate.values()),
        "future_result_file_count_was_zero": int(
            protocol.get("future_result_file_count_at_freeze", -1)
        )
        == 0,
    }
    gate["passed"] = all(gate.values())
    return {
        "successor_protocol_sha256": _sha256(protocol_path),
        "source_gate": source_gate,
        "gate": gate,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successor-protocol", type=Path, required=True)
    parser.add_argument("--failed-v66-result", type=Path, required=True)
    parser.add_argument("--failed-v66-artifact-root", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite successor result: {args.out}")
    executable = validate_executable_freeze(args.successor_protocol)
    if not bool(executable["gate"]["passed"]):
        raise ValueError(f"successor executable freeze failed: {executable}")
    payload = run_proposal_conditional_veto_freeze(
        failed_v66_result_path=args.failed_v66_result,
        failed_v66_artifact_root=args.failed_v66_artifact_root,
        cache_audit_path=args.cache_audit,
        successor_protocol_path=args.successor_protocol,
        output_root=args.output_root,
    )
    payload["run_protocol"] = RUN_PROTOCOL
    payload["executable_freeze"] = executable
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "active_cities": payload["active_cities"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
