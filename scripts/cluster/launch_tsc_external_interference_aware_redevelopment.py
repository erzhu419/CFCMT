#!/usr/bin/env python3
"""Launch the frozen v86 interference-aware matrix on six CPU nodes."""

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
from cf_h2o.eval.traffic_signal_interference_aware_redevelopment import (  # noqa: E402
    RESULT_PROTOCOL,
    all_rollout_identities,
)
from scripts.cluster.freeze_tsc_external_v9_interference_aware_redevelopment import (  # noqa: E402
    NODES,
    PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_uncertainty_horizon_artifacts import (  # noqa: E402
    ARTIFACT_KEYS,
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


LAUNCH_PROTOCOL = "v86r82-interference-aware-six-node-durable-launch-v2"
PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v86_external_v9_"
    "interference_aware_redevelopment.json"
)
V83_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v83_external_v9_"
    "uncertainty_horizon_closed_loop_development.json"
)
V84_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v84_external_v9_"
    "uncertainty_horizon_confirmation.json"
)
V81_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v81_external_v9_"
    "multihorizon_estimand_aligned_closed_loop_development.json"
)
ARTIFACT_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v82_external_v9_"
    "uncertainty_horizon_artifacts.json"
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


def _record_sha(record: Mapping[str, Any]) -> str:
    return str(record["sha256"])


def build_specs(
    *,
    snapshot_root: Path,
    v83_audit_local: Path,
    v83_audit_remote: Path,
    v84_audit_local: Path,
    v84_audit_remote: Path,
    v81_audit_local: Path,
    v81_audit_remote: Path,
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
    local_results_root: Path,
    workers: int,
    cpu_cores: int,
    ram_mb: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol_path = PROJECT_ROOT / PROTOCOL_RELATIVE
    protocol = _read_json(protocol_path)
    local_paths = {
        "v83_protocol": PROJECT_ROOT / V83_PROTOCOL_RELATIVE,
        "v83_audit": v83_audit_local,
        "v84_protocol": PROJECT_ROOT / V84_PROTOCOL_RELATIVE,
        "v84_audit": v84_audit_local,
        "v81_protocol": PROJECT_ROOT / V81_PROTOCOL_RELATIVE,
        "v81_audit": v81_audit_local,
        "artifact_protocol": PROJECT_ROOT / ARTIFACT_PROTOCOL_RELATIVE,
        "artifact_result": artifact_result_local,
        "artifact_audit": artifact_audit_local,
        "parent_runtime": PROJECT_ROOT / PARENT_RUNTIME_RELATIVE,
        "hierarchical_parent": PROJECT_ROOT / HIERARCHICAL_PARENT_RELATIVE,
        "offline_result": offline_result_local,
        "support_audit": support_audit_local,
        "freeze_audit": freeze_audit_local,
        "environment_protocol": PROJECT_ROOT / ENVIRONMENT_RELATIVE,
        "environment_parent": PROJECT_ROOT / ENVIRONMENT_PARENT_RELATIVE,
        "external_manifest": PROJECT_ROOT / MANIFEST_RELATIVE,
        "conversion_manifest": conversion_manifest_local,
    }
    hashes: dict[str, Any] = {
        key: _sha256(path) for key, path in local_paths.items()
    }
    hashes["protocol"] = _sha256(protocol_path)
    hashes["artifact_models"] = {
        key: _sha256(artifact_root_local / key / "model.pkl")
        for key in ARTIFACT_KEYS
    }
    hashes["artifact_certificates"] = {
        key: _sha256(artifact_root_local / key / "freeze.json")
        for key in ARTIFACT_KEYS
    }
    expected = {
        "v83_protocol": _record_sha(
            protocol["redevelopment_parent"]["v83_protocol"]
        ),
        "v83_audit": _record_sha(
            protocol["redevelopment_parent"]["v83_independent_audit"]
        ),
        "v84_protocol": _record_sha(
            protocol["redevelopment_parent"]["v84_protocol"]
        ),
        "v84_audit": _record_sha(
            protocol["redevelopment_parent"]["v84_independent_audit"]
        ),
        "v81_protocol": _record_sha(
            protocol["development_parent"]["v81_protocol"]
        ),
        "v81_audit": _record_sha(
            protocol["development_parent"]["v81_independent_audit"]
        ),
        "artifact_protocol": _record_sha(protocol["artifact"]["protocol"]),
        "artifact_result": _record_sha(protocol["artifact"]["freeze_result"]),
        "artifact_audit": _record_sha(
            protocol["artifact"]["independent_audit"]
        ),
        "parent_runtime": _record_sha(protocol["runtime_parent"]["protocol"]),
        "hierarchical_parent": _record_sha(
            protocol["runtime_parent"]["hierarchical_parent"]
        ),
        "offline_result": _record_sha(
            protocol["runtime_parent"]["offline_result"]
        ),
        "support_audit": _record_sha(
            protocol["runtime_parent"]["support_audit"]
        ),
        "freeze_audit": _record_sha(
            protocol["runtime_parent"]["hierarchical_freeze_audit"]
        ),
        "environment_protocol": _record_sha(protocol["environment"]["protocol"]),
        "environment_parent": _record_sha(
            protocol["environment"]["network_parent_protocol"]
        ),
        "external_manifest": _record_sha(
            protocol["environment"]["external_manifest"]
        ),
        "conversion_manifest": _record_sha(
            protocol["environment"]["conversion_manifest"]
        ),
    }
    artifact_hashes_ok = all(
        hashes["artifact_models"][key]
        == protocol["artifact"]["artifacts"][key]["model"]["sha256"]
        and hashes["artifact_certificates"][key]
        == protocol["artifact"]["artifacts"][key]["certificate"]["sha256"]
        for key in ARTIFACT_KEYS
    )
    identities = all_rollout_identities(protocol)
    output_roots_match = (
        Path(protocol["output_roots"]["remote_results_root"])
        == Path(remote_results_root)
        and Path(protocol["output_roots"]["local_results_root"]).resolve()
        == Path(local_results_root).resolve()
    )
    if not (
        protocol.get("protocol") == PROTOCOL
        and all(hashes[key] == value for key, value in expected.items())
        and artifact_hashes_ok
        and len(identities) == len(set(identities)) == 270
        and int(protocol["development"]["matrix_size"]) == 270
        and output_roots_match
        and 1 <= int(workers) <= int(cpu_cores) <= 192
    ):
        raise ValueError("v86 launch authorization changed")

    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    runner = (
        snapshot_root
        / "scripts/cluster/run_tsc_external_interference_aware_redevelopment_shard.py"
    )
    specs = []
    for shard_index, node in enumerate(NODES):
        command = shlex.join(
            [
                str(wrapper),
                str(runner),
                "--protocol",
                str(snapshot_root / PROTOCOL_RELATIVE),
                "--v83-protocol",
                str(snapshot_root / V83_PROTOCOL_RELATIVE),
                "--v83-audit",
                str(v83_audit_remote),
                "--v84-protocol",
                str(snapshot_root / V84_PROTOCOL_RELATIVE),
                "--v84-audit",
                str(v84_audit_remote),
                "--v81-protocol",
                str(snapshot_root / V81_PROTOCOL_RELATIVE),
                "--v81-audit",
                str(v81_audit_remote),
                "--artifact-protocol",
                str(snapshot_root / ARTIFACT_PROTOCOL_RELATIVE),
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
                "description": f"CFCMT v86 interference shard {shard_index}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v86r82/interference/shard-{shard_index}",
                "resource_family": "CFCMT-v86-interference-redevelopment-v2",
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


def _sync_shard_summaries(
    *,
    remote_results_root: Path,
    local_results_root: Path,
    scheduler_path: Path,
) -> int:
    if local_results_root.exists():
        raise FileExistsError(
            f"refusing to overwrite local v86 results: {local_results_root}"
        )
    scheduler = _load_scheduler(scheduler_path)
    local_summary_root = local_results_root / "_shards"
    local_summary_root.mkdir(parents=True, exist_ok=False)
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
                f"{target}:{remote_results_root}/_shards/",
                f"{local_summary_root}/",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            return attempt
        if attempt == 5:
            raise RuntimeError(
                f"v86 shard-summary sync failed: {completed.stderr[-4000:]}"
            )
        time.sleep(2**attempt)
    raise AssertionError("unreachable v86 rsync retry state")


def _assert_remote_prospective_sealed(
    *, scheduler: Any, path: Path
) -> dict[str, Any]:
    command = (
        f"if test -e {shlex.quote(str(path))}; then "
        "printf '__V85_STATE__ PRESENT\\n'; exit 23; "
        "else printf '__V85_STATE__ ABSENT\\n'; fi"
    )
    code, stdout, stderr = scheduler.run_on(
        "node001", command, timeout=120, check=False
    )
    if int(code) != 0 or "__V85_STATE__ ABSENT" not in str(stdout):
        raise RuntimeError(
            f"sealed v85 remote root is present or unverifiable: {stderr or stdout}"
        )
    return {"path": str(path), "state": "ABSENT"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--v83-audit-local", type=Path, required=True)
    parser.add_argument("--v83-audit-remote", type=Path, required=True)
    parser.add_argument("--v84-audit-local", type=Path, required=True)
    parser.add_argument("--v84-audit-remote", type=Path, required=True)
    parser.add_argument("--v81-audit-local", type=Path, required=True)
    parser.add_argument("--v81-audit-remote", type=Path, required=True)
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
    parser.add_argument("--sealed-v85-local", type=Path, required=True)
    parser.add_argument("--sealed-v85-remote", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=19)
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
        raise FileExistsError("refusing to overwrite v86 launch evidence")
    if args.sealed_v85_local.exists():
        raise FileExistsError(f"sealed v85 local root exists: {args.sealed_v85_local}")

    stage = _read_json(args.stage_manifest)
    protocol = _read_json(PROJECT_ROOT / PROTOCOL_RELATIVE)
    specs, hashes = build_specs(
        snapshot_root=Path(stage["snapshot_root"]),
        v83_audit_local=args.v83_audit_local,
        v83_audit_remote=args.v83_audit_remote,
        v84_audit_local=args.v84_audit_local,
        v84_audit_remote=args.v84_audit_remote,
        v81_audit_local=args.v81_audit_local,
        v81_audit_remote=args.v81_audit_remote,
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
        local_results_root=args.local_results_root,
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
        "local_materialization": "six_shard_summary_json_files_only",
        "full_rollouts_remain_remote": True,
        "workers_per_shard": int(args.workers),
        "node_count": len(specs),
        "matrix_size": sum(int(spec["matrix_rows"]) for spec in specs),
        "specs": specs,
        "complete": False,
    }
    try:
        payload["sealed_v85_local"] = {
            "path": str(args.sealed_v85_local),
            "state": "ABSENT",
        }
        payload["sealed_v85_remote"] = _assert_remote_prospective_sealed(
            scheduler=scheduler, path=args.sealed_v85_remote
        )
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
                run_id=f"cfcmt-v86-interference-{node}-v2",
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
            raise RuntimeError("one or more durable v86 shards failed")
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
        payload["summary_sync_attempts"] = _sync_shard_summaries(
            remote_results_root=args.remote_results_root,
            local_results_root=args.local_results_root,
            scheduler_path=args.scheduler,
        )
        payload["complete"] = True
        payload["execution_mode"] = (
            "durable_six_physical_nodes_nineteen_spawn_workers_each"
        )
    except Exception as exc:
        payload["execution_error"] = repr(exc)[-8000:]
        _atomic_json(args.out, payload)
        raise
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "complete": payload["complete"],
                "node_count": len(specs),
                "matrix_size": payload["matrix_size"],
                "summary_sync_attempts": payload["summary_sync_attempts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
