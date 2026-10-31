#!/usr/bin/env python3
"""Submit the three frozen V157C Jinan native-prefix seed shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_feature_aligned_b100_runtime_freeze import (  # noqa: E402
    ARM_MANIFEST_PROTOCOL,
    ARM_ORDER,
    RESULT_PROTOCOL as V157B_RESULT_PROTOCOL,
    RUNTIME_BUNDLE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_v157c_jinan_native_action_branches import (  # noqa: E402
    INPUT_VALIDATION_PROTOCOL,
    LEARNED_ARMS,
    PROTOCOL,
    SEEDS,
    validate_protocol,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)
from scripts.cluster.derive_tsc_v157c_jinan_native_action_branches_snapshot import (  # noqa: E402
    DERIVATION_PROTOCOL,
    EVALUATOR_PATCH_RELATIVE,
    EVALUATOR_RELATIVE,
    OCCUPANCY_EQUATIONS_RELATIVE,
    OVERLAYS,
    PARENT_ROOT,
    PARENT_SNAPSHOT_SHA256,
    PARENT_SOURCE_TREE_SHA256,
    REMOTE_BASE,
    REMOTE_PYTHON,
    SAFE_PHASE_CONTROLLER_RELATIVE,
    V2_EVALUATOR_RELATIVE,
    V157C_EVALUATOR_SHA256,
    runtime_dependency_contract,
)


LAUNCH_PROTOCOL = "tsc-v157c-jinan-native-one-action-launch-v1"
SIGNATURE_PREFIX = "CFCMT/v157c/jinan-native-one-action"
RESOURCE_FAMILY = "CFCMT-v157c-jinan-native-one-action-v1"
MODULE = "cf_h2o.eval.traffic_signal_v157c_jinan_native_action_branches"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v157c_jinan_native_action_branches.json"
)
RUNNER_RELATIVE = Path(
    "cf_h2o/eval/traffic_signal_v157c_jinan_native_action_branches.py"
)
LAUNCHER_RELATIVE = Path(
    "scripts/cluster/launch_tsc_v157c_jinan_native_action_branches.py"
)
V157B_MODULE_RELATIVE = Path(
    "cf_h2o/eval/traffic_signal_feature_aligned_b100_runtime_freeze.py"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)
DEFAULT_NODES = ("node003", "node004", "node005")
SNAPSHOT_IDENTITY_ORDER = (
    "stage_marker",
    "runner",
    "config",
    "launcher",
    "evaluator",
    "v2_evaluator",
    "occupancy_equations",
    "safe_phase_controller",
    "evaluator_patch",
    "v157b_module",
    "network_manifest",
)

REMOTE_RESULT_SUMMARY = """\
import json,sys
p=json.load(open(sys.argv[1], encoding='utf-8'))
print(json.dumps({
 'protocol':p.get('protocol'),
 'scientific_status':p.get('scientific_status'),
 'city':p.get('city'),
 'arm_order':p.get('arm_order'),
 'target_action_groups':p.get('information_budget',{}).get('target_action_groups'),
 'v157a_exact_refit_identity_passed':p.get('v157a_exact_refit_identity_audit',{}).get('passed'),
 'domain_aligned_source_gate_passed':p.get('domain_aligned_source_gate',{}).get('passed'),
 'placebo_capacity_passed':p.get('placebo',{}).get('capacity_match_audit',{}).get('passed'),
 'runtime_models':p.get('artifacts',{}).get('runtime_models'),
 'arm_manifest':p.get('artifacts',{}).get('arm_manifest'),
},sort_keys=True,separators=(',',':')))
"""


def _result_summary_command(result_path: Path) -> str:
    return shlex.join(
        [str(REMOTE_PYTHON), "-c", REMOTE_RESULT_SUMMARY, str(result_path)]
    )


def _input_validation_command(
    *,
    snapshot_root: Path,
    conversion_root: Path,
    authorization_path: Path,
    runtime_bundle_path: Path,
    runtime_bundle_sha256: str,
    arm_manifest_path: Path,
    arm_manifest_sha256: str,
) -> str:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    return shlex.join(
        [
            "env",
            f"CFCMT_SOURCE_ROOT={snapshot_root}",
            f"CFCMT_EXTERNAL_CONVERSION_ROOT={conversion_root}",
            f"CFCMT_PYTHON={REMOTE_PYTHON}",
            str(wrapper),
            "-m",
            MODULE,
            "validate-inputs",
            "--protocol",
            str(snapshot_root / CONFIG_RELATIVE),
            "--authorization",
            str(authorization_path),
            "--runtime-bundle",
            str(runtime_bundle_path),
            "--runtime-bundle-sha256",
            str(runtime_bundle_sha256),
            "--arm-manifest",
            str(arm_manifest_path),
            "--arm-manifest-sha256",
            str(arm_manifest_sha256),
        ]
    )


def validate_v157b_summary(
    summary: Mapping[str, Any],
    *,
    runtime_models: Path,
    runtime_models_sha256: str,
    arm_manifest: Path,
    arm_manifest_sha256: str,
) -> None:
    runtime = dict(summary.get("runtime_models") or {})
    manifest = dict(summary.get("arm_manifest") or {})
    if (
        summary.get("protocol") != V157B_RESULT_PROTOCOL
        or summary.get("scientific_status")
        != "runtime_arms_frozen_branch_not_run"
        or summary.get("city") != "jinan"
        or tuple(summary.get("arm_order") or ()) != tuple(ARM_ORDER)
        or int(summary.get("target_action_groups", -1)) != 100
        or summary.get("v157a_exact_refit_identity_passed") is not True
        or summary.get("domain_aligned_source_gate_passed") is not True
        or summary.get("placebo_capacity_passed") is not True
        or Path(runtime.get("path", "")).resolve() != runtime_models.resolve()
        or runtime.get("sha256") != str(runtime_models_sha256)
        or runtime.get("protocol") != RUNTIME_BUNDLE_PROTOCOL
        or Path(manifest.get("path", "")).resolve() != arm_manifest.resolve()
        or manifest.get("sha256") != str(arm_manifest_sha256)
        or manifest.get("protocol") != ARM_MANIFEST_PROTOCOL
    ):
        raise ValueError("V157B result does not authorize V157C execution")


def _validate_stage_manifest(stage: Mapping[str, Any]) -> Path:
    """Bind execution to the reviewed V157B-v2-derived V157C snapshot."""

    digest = str(stage.get("snapshot_sha256", ""))
    snapshot_root = Path(str(stage.get("snapshot_root", "")))
    expected_overlays = [
        {
            "path": relative.as_posix(),
            "sha256": _sha256(PROJECT_ROOT / relative),
        }
        for relative in OVERLAYS
    ]
    if (
        stage.get("protocol") != DERIVATION_PROTOCOL
        or Path(str(stage.get("derived_from_snapshot_root", ""))) != PARENT_ROOT
        or stage.get("derived_from_snapshot_sha256") != PARENT_SNAPSHOT_SHA256
        or stage.get("derived_from_source_tree_sha256")
        != PARENT_SOURCE_TREE_SHA256
        or stage.get("overlays") != expected_overlays
        or stage.get("runtime_dependencies") != runtime_dependency_contract()
        or len(digest) != 64
        or any(value not in "0123456789abcdef" for value in digest)
        or snapshot_root != REMOTE_BASE / digest[:20]
    ):
        raise ValueError("V157C derived snapshot manifest contract changed")
    return snapshot_root


def _validate_remote_hashes(
    scheduler: Any,
    *,
    node: str,
    paths: Sequence[Path],
    expected: Sequence[str],
) -> list[str]:
    code, stdout, stderr = scheduler.run_on(
        node,
        shlex.join(["sha256sum", *(str(path) for path in paths)]),
        timeout=300,
        check=False,
    )
    observed = [line.split()[0] for line in stdout.splitlines() if line.split()]
    if int(code) != 0 or observed != list(expected):
        raise RuntimeError(
            f"remote V157C identities changed: {observed} != "
            f"{list(expected)}; {stderr}"
        )
    return observed


def _validate_remote_inputs(
    scheduler: Any,
    *,
    node: str,
    snapshot_root: Path,
    conversion_root: Path,
    authorization_path: Path,
    runtime_bundle_path: Path,
    runtime_bundle_sha256: str,
    arm_manifest_path: Path,
    arm_manifest_sha256: str,
) -> dict[str, Any]:
    code, stdout, stderr = scheduler.run_on(
        node,
        _input_validation_command(
            snapshot_root=snapshot_root,
            conversion_root=conversion_root,
            authorization_path=authorization_path,
            runtime_bundle_path=runtime_bundle_path,
            runtime_bundle_sha256=runtime_bundle_sha256,
            arm_manifest_path=arm_manifest_path,
            arm_manifest_sha256=arm_manifest_sha256,
        ),
        timeout=300,
        check=False,
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V157C input validation failed: {stderr}")
    lines = [line for line in stdout.splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("remote V157C input validation output changed") from exc
    if (
        result.get("protocol") != INPUT_VALIDATION_PROTOCOL
        or result.get("status") != "PASS"
        or tuple(result.get("learned_arms", ())) != LEARNED_ARMS
        or result.get("sumo_started") is not False
        or result.get("artifacts_written") is not False
    ):
        raise RuntimeError("remote V157C input validation contract changed")
    return result


def build_specs(
    *,
    snapshot_root: Path,
    conversion_root: Path,
    v157a_authorization_remote: Path,
    runtime_models_remote: Path,
    runtime_models_sha256: str,
    arm_manifest_remote: Path,
    arm_manifest_sha256: str,
    remote_output_root: Path,
    remote_scratch_root: Path,
    nodes: Sequence[str],
) -> list[dict[str, Any]]:
    selected_nodes = tuple(str(value) for value in nodes)
    if len(selected_nodes) != len(SEEDS) or len(set(selected_nodes)) != len(SEEDS):
        raise ValueError("V157C requires three distinct execution nodes")
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    specs = []
    for seed, node in zip(SEEDS, selected_nodes, strict=True):
        output = remote_output_root / f"seed_{seed}"
        scratch = remote_scratch_root / f"seed_{seed}"
        command = shlex.join(
            [
                "env",
                f"CFCMT_SOURCE_ROOT={snapshot_root}",
                f"CFCMT_EXTERNAL_CONVERSION_ROOT={conversion_root}",
                "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                "NUMEXPR_NUM_THREADS=1",
                "PYTHONUNBUFFERED=1",
                str(wrapper),
                "-m",
                MODULE,
                "run-seed",
                "--protocol",
                str(snapshot_root / CONFIG_RELATIVE),
                "--authorization",
                str(v157a_authorization_remote),
                "--manifest",
                str(snapshot_root / MANIFEST_RELATIVE),
                "--conversion-root",
                str(conversion_root),
                "--runtime-bundle",
                str(runtime_models_remote),
                "--runtime-bundle-sha256",
                str(runtime_models_sha256),
                "--arm-manifest",
                str(arm_manifest_remote),
                "--arm-manifest-sha256",
                str(arm_manifest_sha256),
                "--seed",
                str(seed),
                "--scratch-root",
                str(scratch),
                "--output",
                str(output),
            ]
        ) + " && printf 'TASK_DONE\n'"
        specs.append(
            {
                "description": f"CFCMT V157C Jinan native one-action seed {seed}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"{SIGNATURE_PREFIX}/seed{seed}",
                "resource_family": RESOURCE_FAMILY,
                "ram_resource_family": RESOURCE_FAMILY,
                "vram": 0,
                "ram_mb": 32768,
                "cpu": 2,
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {},
                "result_dir": str(output),
            }
        )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--v157a-authorization-remote", type=Path, required=True)
    parser.add_argument("--v157b-runtime-models-remote", type=Path, required=True)
    parser.add_argument("--v157b-runtime-models-sha256", required=True)
    parser.add_argument("--v157b-arm-manifest-remote", type=Path, required=True)
    parser.add_argument("--v157b-arm-manifest-sha256", required=True)
    parser.add_argument("--v157b-result-remote", type=Path, required=True)
    parser.add_argument("--v157b-result-sha256", required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--remote-scratch-root", type=Path, required=True)
    parser.add_argument("--nodes", nargs=3, default=DEFAULT_NODES)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V157C launch artifact")

    config_path = PROJECT_ROOT / CONFIG_RELATIVE
    config = _read_json(config_path)
    validate_protocol(config)
    stage = _read_json(args.stage_manifest)
    snapshot_root = _validate_stage_manifest(stage)
    snapshot_files = {
        "stage_marker": snapshot_root / ".cfcmt_snapshot.json",
        "runner": snapshot_root / RUNNER_RELATIVE,
        "config": snapshot_root / CONFIG_RELATIVE,
        "launcher": snapshot_root / LAUNCHER_RELATIVE,
        "evaluator": snapshot_root / EVALUATOR_RELATIVE,
        "v2_evaluator": snapshot_root / V2_EVALUATOR_RELATIVE,
        "occupancy_equations": snapshot_root / OCCUPANCY_EQUATIONS_RELATIVE,
        "safe_phase_controller": snapshot_root / SAFE_PHASE_CONTROLLER_RELATIVE,
        "evaluator_patch": snapshot_root / EVALUATOR_PATCH_RELATIVE,
        "v157b_module": snapshot_root / V157B_MODULE_RELATIVE,
        "network_manifest": snapshot_root / MANIFEST_RELATIVE,
        "wrapper": snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh",
    }
    v157b_files = (
        args.v157b_runtime_models_remote,
        args.v157b_arm_manifest_remote,
        args.v157b_result_remote,
    )
    remote_files = (
        *v157b_files,
        args.v157a_authorization_remote,
        *snapshot_files.values(),
    )
    scheduler = _load_scheduler(args.scheduler)
    probe_node = str(args.nodes[0])
    preflight = " && ".join(
        [
            shlex.join(["test", "-d", str(args.conversion_root)]),
            shlex.join(["test", "!", "-e", str(args.remote_output_root)]),
            shlex.join(["test", "!", "-e", str(args.remote_scratch_root)]),
            *(
                shlex.join(["test", "-f", str(path)]) for path in remote_files
            ),
        ]
    )
    code, _, stderr = scheduler.run_on(
        probe_node, preflight, timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V157C preflight failed: {stderr}")

    ordered_hash_paths = (
        *v157b_files,
        args.v157a_authorization_remote,
        *(snapshot_files[name] for name in SNAPSHOT_IDENTITY_ORDER),
    )
    expected_hashes = [
        str(args.v157b_runtime_models_sha256),
        str(args.v157b_arm_manifest_sha256),
        str(args.v157b_result_sha256),
        str(config["authorized_by"]["sha256"]),
        _sha256(args.stage_manifest),
        _sha256(PROJECT_ROOT / RUNNER_RELATIVE),
        _sha256(config_path),
        _sha256(PROJECT_ROOT / LAUNCHER_RELATIVE),
        V157C_EVALUATOR_SHA256,
        _sha256(PROJECT_ROOT / V2_EVALUATOR_RELATIVE),
        _sha256(PROJECT_ROOT / OCCUPANCY_EQUATIONS_RELATIVE),
        _sha256(PROJECT_ROOT / SAFE_PHASE_CONTROLLER_RELATIVE),
        _sha256(PROJECT_ROOT / EVALUATOR_PATCH_RELATIVE),
        _sha256(PROJECT_ROOT / V157B_MODULE_RELATIVE),
        str(config["network_manifest"]["sha256"]),
    ]
    observed_hashes = _validate_remote_hashes(
        scheduler,
        node=probe_node,
        paths=ordered_hash_paths,
        expected=expected_hashes,
    )
    code, stdout, stderr = scheduler.run_on(
        probe_node,
        _result_summary_command(args.v157b_result_remote),
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise RuntimeError(f"cannot read V157B result summary: {stderr}")
    summary = json.loads(stdout)
    validate_v157b_summary(
        summary,
        runtime_models=args.v157b_runtime_models_remote,
        runtime_models_sha256=args.v157b_runtime_models_sha256,
        arm_manifest=args.v157b_arm_manifest_remote,
        arm_manifest_sha256=args.v157b_arm_manifest_sha256,
    )
    input_validation = _validate_remote_inputs(
        scheduler,
        node=probe_node,
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        authorization_path=args.v157a_authorization_remote,
        runtime_bundle_path=args.v157b_runtime_models_remote,
        runtime_bundle_sha256=args.v157b_runtime_models_sha256,
        arm_manifest_path=args.v157b_arm_manifest_remote,
        arm_manifest_sha256=args.v157b_arm_manifest_sha256,
    )

    specs = build_specs(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        v157a_authorization_remote=args.v157a_authorization_remote,
        runtime_models_remote=args.v157b_runtime_models_remote,
        runtime_models_sha256=args.v157b_runtime_models_sha256,
        arm_manifest_remote=args.v157b_arm_manifest_remote,
        arm_manifest_sha256=args.v157b_arm_manifest_sha256,
        remote_output_root=args.remote_output_root,
        remote_scratch_root=args.remote_scratch_root,
        nodes=args.nodes,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v157c-jinan-native-one-action-v1",
        ],
        input=json.dumps(specs),
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": completed.returncode == 0,
        "scheduler_returncode": int(completed.returncode),
        "scheduler_stdout": completed.stdout,
        "scheduler_stderr": completed.stderr,
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage.get("snapshot_sha256"),
        "source_tree_sha256": stage.get("source_tree_sha256"),
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "input_validation": input_validation,
        "v157b_inputs": {
            "runtime_models": {
                "path": str(args.v157b_runtime_models_remote),
                "sha256": str(args.v157b_runtime_models_sha256),
            },
            "arm_manifest": {
                "path": str(args.v157b_arm_manifest_remote),
                "sha256": str(args.v157b_arm_manifest_sha256),
            },
            "result": {
                "path": str(args.v157b_result_remote),
                "sha256": str(args.v157b_result_sha256),
            },
        },
        "remote_identity_sha256": observed_hashes,
        "seeds": list(SEEDS),
        "nodes": list(args.nodes),
        "task_count": len(specs),
        "expected_native_simulation_count": 30,
        "remote_output_root": str(args.remote_output_root),
        "remote_scratch_root": str(args.remote_scratch_root),
        "task_specs": specs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "task_count": payload["task_count"],
                "out": str(args.out),
            }
        )
    )
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V157C seed shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
