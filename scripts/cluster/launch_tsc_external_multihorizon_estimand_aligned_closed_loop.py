#!/usr/bin/env python3
"""Launch the frozen v81 matrix durably on six 192-core CPU nodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_multihorizon_estimand_aligned_closed_loop_development import (  # noqa: E402
    RESULT_PROTOCOL,
    all_rollout_identities,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_state_latent_artifacts import (  # noqa: E402
    ARTIFACT_KEYS,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_estimand_aligned_closed_loop_development import (  # noqa: E402
    NODES,
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_closed_loop_development import (  # noqa: E402
    _remote_postflight,
    _remote_preflight,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _direct_remote_command,
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


LAUNCH_PROTOCOL = "v83r79-multihorizon-estimand-aligned-six-node-durable-launch-v1"
PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v81_external_v9_"
    "multihorizon_estimand_aligned_closed_loop_development.json"
)
V80_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v80_external_v9_"
    "multihorizon_state_latent_closed_loop_development.json"
)
V80_AUDIT_RELATIVE = Path(
    "cf_h2o/results/cluster/tsc_v83r79_external_v9_runtime_queue_trust_"
    "redevelopment_20260830/multihorizon_state_latent_closed_loop_audit_v1.json"
)
ALIGNMENT_DIAGNOSTIC_RELATIVE = Path(
    "cf_h2o/results/cluster/tsc_v83r79_external_v9_runtime_queue_trust_"
    "redevelopment_20260830/multihorizon_estimand_alignment_diagnostic_v1.json"
)
PARENT_RUNTIME_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v46_external_v9_"
    "hierarchical_closed_loop_development.json"
)
HIERARCHICAL_PARENT_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v44_external_v9_"
    "hierarchical_guard_confirmation.json"
)
ENVIRONMENT_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v41_external_v9_full_budget_refit.json"
)
ENVIRONMENT_PARENT_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v40_external_v9_"
    "network_repair_confirmation.json"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def build_specs(
    *,
    snapshot_root: Path,
    v80_audit_remote: Path,
    alignment_diagnostic_remote: Path,
    artifact_result_local: Path,
    artifact_result_remote: Path,
    artifact_audit_local: Path,
    artifact_audit_remote: Path,
    artifact_root_local: Path,
    artifact_root_remote: Path,
    offline_result_local: Path,
    offline_result_remote: Path,
    support_audit_local: Path,
    support_audit_remote: Path,
    freeze_audit_local: Path,
    freeze_audit_remote: Path,
    freeze_root: Path,
    conversion_root: Path,
    conversion_manifest_local: Path,
    conversion_manifest_remote: Path,
    remote_results_root: Path,
    workers: int,
    cpu_cores: int,
    ram_mb: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol_path = PROJECT_ROOT / PROTOCOL_RELATIVE
    protocol = _read_json(protocol_path)
    hashes: dict[str, Any] = {
        "protocol": _sha256(protocol_path),
        "v80_protocol": _sha256(PROJECT_ROOT / V80_PROTOCOL_RELATIVE),
        "v80_audit": _sha256(PROJECT_ROOT / V80_AUDIT_RELATIVE),
        "alignment_diagnostic": _sha256(
            PROJECT_ROOT / ALIGNMENT_DIAGNOSTIC_RELATIVE
        ),
        "artifact_result": _sha256(artifact_result_local),
        "artifact_audit": _sha256(artifact_audit_local),
        "artifact_models": {
            key: _sha256(artifact_root_local / key / "model.pkl")
            for key in ARTIFACT_KEYS
        },
        "artifact_certificates": {
            key: _sha256(artifact_root_local / key / "freeze.json")
            for key in ARTIFACT_KEYS
        },
        "parent_runtime": _sha256(PROJECT_ROOT / PARENT_RUNTIME_RELATIVE),
        "hierarchical_parent": _sha256(PROJECT_ROOT / HIERARCHICAL_PARENT_RELATIVE),
        "offline_result": _sha256(offline_result_local),
        "support_audit": _sha256(support_audit_local),
        "freeze_audit": _sha256(freeze_audit_local),
        "environment_protocol": _sha256(PROJECT_ROOT / ENVIRONMENT_RELATIVE),
        "environment_parent": _sha256(PROJECT_ROOT / ENVIRONMENT_PARENT_RELATIVE),
        "external_manifest": _sha256(PROJECT_ROOT / MANIFEST_RELATIVE),
        "conversion_manifest": _sha256(conversion_manifest_local),
    }
    expected = {
        "v80_protocol": protocol["development_parent"]["v80_protocol"][
            "sha256"
        ],
        "v80_audit": protocol["development_parent"][
            "v80_independent_audit"
        ]["sha256"],
        "alignment_diagnostic": protocol["development_parent"][
            "estimand_alignment_diagnostic"
        ]["sha256"],
        "artifact_result": protocol["artifact"]["freeze_result"]["sha256"],
        "artifact_audit": protocol["artifact"]["independent_audit"]["sha256"],
        "parent_runtime": protocol["runtime_parent"]["protocol"]["sha256"],
        "hierarchical_parent": protocol["runtime_parent"]["hierarchical_parent"][
            "sha256"
        ],
        "offline_result": protocol["runtime_parent"]["offline_result"]["sha256"],
        "support_audit": protocol["runtime_parent"]["support_audit"]["sha256"],
        "freeze_audit": protocol["runtime_parent"]["hierarchical_freeze_audit"][
            "sha256"
        ],
        "environment_protocol": protocol["environment"]["protocol"]["sha256"],
        "environment_parent": protocol["environment"]["network_parent_protocol"][
            "sha256"
        ],
        "external_manifest": protocol["environment"]["external_manifest"][
            "sha256"
        ],
        "conversion_manifest": protocol["environment"]["conversion_manifest"][
            "sha256"
        ],
    }
    artifact_hashes_ok = all(
        hashes["artifact_models"][key]
        == protocol["artifact"]["artifacts"][key]["model"]["sha256"]
        and hashes["artifact_certificates"][key]
        == protocol["artifact"]["artifacts"][key]["certificate"]["sha256"]
        for key in ARTIFACT_KEYS
    )
    identities = all_rollout_identities(protocol)
    if (
        protocol.get("protocol") != PROTOCOL
        or any(hashes[key] != str(value) for key, value in expected.items())
        or not artifact_hashes_ok
        or len(identities) != 198
        or not 1 <= int(workers) <= int(cpu_cores) <= 192
    ):
        raise ValueError("v81 launch authorization changed")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    runner = (
        snapshot_root
        / "scripts/cluster/run_tsc_external_multihorizon_estimand_aligned_closed_loop_shard.py"
    )
    specs = []
    for shard_index, node in enumerate(NODES):
        command = shlex.join(
            [
                str(wrapper),
                str(runner),
                "--protocol",
                str(snapshot_root / PROTOCOL_RELATIVE),
                "--v80-protocol",
                str(snapshot_root / V80_PROTOCOL_RELATIVE),
                "--v80-audit",
                str(v80_audit_remote),
                "--alignment-diagnostic",
                str(alignment_diagnostic_remote),
                "--artifact-result",
                str(artifact_result_remote),
                "--artifact-audit",
                str(artifact_audit_remote),
                "--artifact-root",
                str(artifact_root_remote),
                "--parent-runtime-protocol",
                str(snapshot_root / PARENT_RUNTIME_RELATIVE),
                "--hierarchical-parent",
                str(snapshot_root / HIERARCHICAL_PARENT_RELATIVE),
                "--offline-result",
                str(offline_result_remote),
                "--support-audit",
                str(support_audit_remote),
                "--hierarchical-freeze-audit",
                str(freeze_audit_remote),
                "--hierarchical-freeze-root",
                str(freeze_root),
                "--environment-protocol",
                str(snapshot_root / ENVIRONMENT_RELATIVE),
                "--environment-parent",
                str(snapshot_root / ENVIRONMENT_PARENT_RELATIVE),
                "--external-manifest",
                str(snapshot_root / MANIFEST_RELATIVE),
                "--conversion-root",
                str(conversion_root),
                "--conversion-manifest",
                str(conversion_manifest_remote),
                "--results-root",
                str(remote_results_root),
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(len(NODES)),
                "--workers",
                str(workers),
            ]
        )
        shard_size = sum(
            index % len(NODES) == shard_index for index in range(len(identities))
        )
        specs.append(
            {
                "description": f"CFCMT v81 estimand-aligned shard {shard_index}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v83r79/multihorizon-estimand-aligned/shard-{shard_index}",
                "resource_family": "CFCMT-v81-estimand-aligned-closed-loop-v1",
                "vram": 0,
                "ram_mb": int(ram_mb),
                "cpu": int(cpu_cores),
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {
                    "CFCMT_SOURCE_ROOT": str(snapshot_root),
                    "CFCMT_EXTERNAL_CONVERSION_ROOT": str(conversion_root),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                "result_dir": str(remote_results_root),
                "matrix_rows": int(shard_size),
            }
        )
    return specs, hashes


def _sync_results(
    *,
    remote_results_root: Path,
    local_results_root: Path,
    scheduler_path: Path,
) -> int:
    if local_results_root.exists():
        raise FileExistsError(f"refusing to overwrite local v81 results: {local_results_root}")
    scheduler = _load_scheduler(scheduler_path)
    local_results_root.mkdir(parents=True, exist_ok=False)
    target = scheduler._ssh_target_for_node("node001")
    ssh_shell = scheduler._ssh_rsync_shell_for_node("node001")
    for attempt in range(1, 6):
        completed = subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                "--append-verify",
                "--timeout=300",
                "-e",
                ssh_shell,
                f"{target}:{remote_results_root}/",
                f"{local_results_root}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            return attempt
        if attempt == 5:
            raise RuntimeError(f"v81 result sync failed: {completed.stderr[-4000:]}")
        time.sleep(2**attempt)
    raise AssertionError("unreachable v81 rsync retry state")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--v80-audit-remote", type=Path, required=True)
    parser.add_argument("--alignment-diagnostic-remote", type=Path, required=True)
    parser.add_argument("--artifact-result-local", type=Path, required=True)
    parser.add_argument("--artifact-result-remote", type=Path, required=True)
    parser.add_argument("--artifact-audit-local", type=Path, required=True)
    parser.add_argument("--artifact-audit-remote", type=Path, required=True)
    parser.add_argument("--artifact-root-local", type=Path, required=True)
    parser.add_argument("--artifact-root-remote", type=Path, required=True)
    parser.add_argument("--offline-result-local", type=Path, required=True)
    parser.add_argument("--offline-result-remote", type=Path, required=True)
    parser.add_argument("--support-audit-local", type=Path, required=True)
    parser.add_argument("--support-audit-remote", type=Path, required=True)
    parser.add_argument("--freeze-audit-local", type=Path, required=True)
    parser.add_argument("--freeze-audit-remote", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest-local", type=Path, required=True)
    parser.add_argument("--conversion-manifest-remote", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=21)
    parser.add_argument("--cpu-cores", type=int, default=32)
    parser.add_argument("--ram-mb", type=int, default=65536)
    parser.add_argument(
        "--scheduleurm-root",
        type=Path,
        default=Path("/home/erzhu419/mine_code/scheduleurm"),
    )
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--timeout-sec", type=int, default=10800)
    parser.add_argument("--transport-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() or args.transport_root.exists():
        raise FileExistsError("refusing to overwrite v81 launch evidence")
    stage = _read_json(args.stage_manifest)
    protocol = _read_json(PROJECT_ROOT / PROTOCOL_RELATIVE)
    specs, hashes = build_specs(
        snapshot_root=Path(stage["snapshot_root"]),
        v80_audit_remote=args.v80_audit_remote,
        alignment_diagnostic_remote=args.alignment_diagnostic_remote,
        artifact_result_local=args.artifact_result_local,
        artifact_result_remote=args.artifact_result_remote,
        artifact_audit_local=args.artifact_audit_local,
        artifact_audit_remote=args.artifact_audit_remote,
        artifact_root_local=args.artifact_root_local,
        artifact_root_remote=args.artifact_root_remote,
        offline_result_local=args.offline_result_local,
        offline_result_remote=args.offline_result_remote,
        support_audit_local=args.support_audit_local,
        support_audit_remote=args.support_audit_remote,
        freeze_audit_local=args.freeze_audit_local,
        freeze_audit_remote=args.freeze_audit_remote,
        freeze_root=args.freeze_root,
        conversion_root=args.conversion_root,
        conversion_manifest_local=args.conversion_manifest_local,
        conversion_manifest_remote=args.conversion_manifest_remote,
        remote_results_root=args.remote_results_root,
        workers=args.workers,
        cpu_cores=args.cpu_cores,
        ram_mb=args.ram_mb,
    )
    scheduler = _load_scheduler(args.scheduler)
    payload: dict[str, Any] = {
        "protocol": LAUNCH_PROTOCOL,
        "result_protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": stage["snapshot_root"],
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "input_hashes": hashes,
        "conversion_tree_sha256": protocol["environment"][
            "conversion_tree_sha256"
        ],
        "remote_results_root": str(args.remote_results_root),
        "local_results_root": str(args.local_results_root.resolve()),
        "workers_per_shard": int(args.workers),
        "node_count": len(specs),
        "matrix_size": sum(int(spec["matrix_rows"]) for spec in specs),
        "specs": specs,
        "submitted": False,
    }
    try:
        payload["remote_preflight"] = _remote_preflight(
            scheduler_path=args.scheduler,
            snapshot_root=Path(stage["snapshot_root"]),
            conversion_root=args.conversion_root,
            expected_conversion_tree_sha256=str(
                protocol["environment"]["conversion_tree_sha256"]
            ),
            remote_results_root=args.remote_results_root,
            allow_existing_results=bool(args.resume),
        )
        sys.path.insert(0, str(args.scheduleurm_root.resolve()))
        from algorithm.experiments.durable_remote_command import (
            run_durable_remote_capture,
        )

        args.transport_root.mkdir(parents=True, exist_ok=False)

        def run(spec: Mapping[str, Any]) -> dict[str, Any]:
            node = str(spec["require_node"])
            returncode, stdout, run_stderr, durable = run_durable_remote_capture(
                node,
                _direct_remote_command(spec),
                args.transport_root / node,
                run_id=f"cfcmt-v81-estimand-aligned-{node}-v1",
                timeout_s=int(args.timeout_sec),
                poll_interval_s=15.0,
            )
            return {
                "node": node,
                "returncode": int(returncode),
                "stdout_tail": str(stdout)[-12000:],
                "stderr_tail": str(run_stderr)[-12000:],
                "durable_transport": durable,
            }

        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            payload["direct_results"] = list(pool.map(run, specs))
        if not all(row["returncode"] == 0 for row in payload["direct_results"]):
            raise RuntimeError("one or more durable v81 shards failed")
        payload["remote_postflight"] = _remote_postflight(
            scheduler_path=args.scheduler,
            snapshot_root=Path(stage["snapshot_root"]),
            conversion_root=args.conversion_root,
            expected_conversion_tree_sha256=str(
                protocol["environment"]["conversion_tree_sha256"]
            ),
            remote_results_root=args.remote_results_root,
            expected_result_file_count=(2 * len(all_rollout_identities(protocol)))
            + len(NODES),
        )
        payload["sync_attempts"] = _sync_results(
            remote_results_root=args.remote_results_root,
            local_results_root=args.local_results_root,
            scheduler_path=args.scheduler,
        )
        payload["submitted"] = True
        payload["execution_mode"] = (
            "durable_six_physical_nodes_twenty_one_workers_each"
        )
    except Exception as exc:
        payload["execution_error"] = repr(exc)[-8000:]
        _atomic_json(args.out, payload)
        raise
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "node_count": len(specs),
                "matrix_size": payload["matrix_size"],
                "sync_attempts": payload["sync_attempts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
